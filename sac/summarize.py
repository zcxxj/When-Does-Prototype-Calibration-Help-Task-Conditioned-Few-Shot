"""Recompute balanced tables and paired episode intervals from predictions."""
import math
import random
import statistics


class IncompleteResults(ValueError):
    pass


def paired_aggregation(candidate, baseline, *, shared_folds=True, resamples=2000):
    """Inputs [seed, fold, episode], accuracies in [0,1].

    Pre-average correlated operator/severity axes per episode before calling.
    Shared synthetic test folds use identical resampled episode indices. Scan
    and dev folds have independent pools. Fixed fold sets are not resampled.
    """
    def shape(a):
        return (len(a), tuple(tuple(len(f) for f in s) for s in a))
    if not candidate or shape(candidate) != shape(baseline):
        raise IncompleteResults('incomplete: paired seed/fold/episode shape mismatch')
    nfold = len(candidate[0])
    nepisode = len(candidate[0][0]) if nfold else 0
    if not nepisode or any(len(s) != nfold or any(len(f) != nepisode for f in s) for s in candidate):
        raise IncompleteResults('incomplete: empty or unequal fold/episode counts')
    diff = [[[100 * (float(a) - float(b)) for a, b in zip(cf, bf)]
             for cf, bf in zip(cs, bs)] for cs, bs in zip(candidate, baseline)]
    if any(not math.isfinite(x) for s in diff for f in s for x in f):
        raise ValueError('Nonfinite prediction accuracy')
    means = [statistics.mean(statistics.mean(f) for f in s) for s in diff]
    import numpy as np
    values = np.asarray(diff)
    rng = np.random.RandomState(400007)
    samples = []
    for _ in range(resamples):
        seeds = rng.randint(len(diff), size=len(diff)) if len(diff) > 1 else [0]
        if shared_folds:
            ids = rng.randint(nepisode, size=nepisode)
            samples.append(float(values[seeds][:, :, ids].mean()))
        else:
            ids = rng.randint(nepisode, size=(nfold, nepisode))
            samples.append(float(values[seeds][:, np.arange(nfold)[:, None], ids].mean()))
    low, high = np.percentile(samples, [2.5, 97.5])
    return dict(delta_pp=statistics.mean(means), ci95_pp=[float(low), float(high)],
                per_seed_delta_pp=means, seed_std_pp=statistics.stdev(means) if len(means) > 1 else None,
                bootstrap_seed=400007, resamples=resamples, shared_folds=shared_folds)


def retention_decision(clean_deltas, robust_deltas):
    if len(clean_deltas) != 6 or len(robust_deltas) != 4:
        raise IncompleteResults('incomplete: selection needs six clean and four robust cells')
    return all(d >= -1 for d in clean_deltas) and any(d > 0 for d in clean_deltas + robust_deltas)


def method_seeds(out, requested, method):
    import json
    from pathlib import Path
    from .evaluate import METHODS
    path = Path(out) / 'selection.json'
    if not path.exists() or all(seed == 7 for seed in requested):
        return list(requested)
    selection = json.loads(path.read_text())
    family = METHODS[method][0]
    seeds = [seed for seed in requested if seed == 7 or
             (seed in selection['additional_seeds'] and family in selection['additional_families'])]
    if not seeds:
        raise IncompleteResults(f'incomplete: no selected seeds for {method}')
    return seeds


def load_matrix(out, train_seeds, pool, methods=None):
    """Require every named cell, recompute OA; default stays the old six methods."""
    import json
    from .campaign import training_jobs
    from .evaluate import METHODS, method_spec, evaluation_conditions, prediction_path
    names = tuple(METHODS) if methods is None else tuple(methods)
    if not names or len(names) != len(set(names)):
        raise ValueError('Expected nonempty unique methods')
    specs = [(method, method_spec(method)) for method in names]
    expected = []
    for job in training_jobs(train_seeds):
        for method, (family, _, _) in specs:
            if family != job['family'] or job['seed'] not in method_seeds(out, train_seeds, method):
                continue
            for ckpt in (('best',) if pool == 'dev' else ('best', 'last')):
                for cond in evaluation_conditions(job['dataset'], method, ckpt):
                    expected.append((job, method, ckpt, cond, prediction_path(out, job, pool, method, ckpt, cond)))
    missing = [str(path) for _, _, _, _, path in expected if not path.is_file() or not path.with_suffix('.json').is_file()]
    if missing:
        raise IncompleteResults(f'incomplete: missing {len(missing)} prediction cells; first: {missing[0]}')
    import numpy as np
    result = {}
    reference_targets = {}
    reference_metadata = {}
    for job, method, ckpt, cond, path in expected:
        meta = json.loads(path.with_suffix('.json').read_text())
        if methods is not None:
            identity = dict(**job, method=method, pool=pool, checkpoint=ckpt, condition=list(cond))
            if any(meta.get(key) != value for key, value in identity.items()):
                raise ValueError(f'Prediction metadata identity mismatch: {path}')
            if meta.get('checkpoint_epoch') is None or not meta.get('manifest'):
                raise ValueError(f'Prediction metadata missing checkpoint/manifest: {path}')
            pair = (job['dataset'], job['fold'], job['shot'], job['seed'], job['family'], ckpt, *cond)
            signature = (meta['checkpoint_epoch'], meta['manifest'])
            if pair in reference_metadata and reference_metadata[pair] != signature:
                raise ValueError(f'Paired metadata checkpoint/manifest mismatch: {path}')
            reference_metadata[pair] = signature
        if meta['smoke'] or meta['episodes'] != 700:
            raise IncompleteResults(f'incomplete: smoke/nonstandard count cannot enter formal table: {path}')
        with np.load(path) as values:
            targets = values['local_target']
            pred = values['pred']
            if targets.shape != (700, 75) or pred.shape != targets.shape:
                raise IncompleteResults(f'incomplete: prediction shape: {path}')
            accuracy = (pred == targets).mean(-1)
            if not np.array_equal(values['episode_id'], np.arange(700)) or not np.allclose(accuracy, values['episode_accuracy']):
                raise ValueError(f'Prediction IDs/OA inconsistent: {path}')
            pairing = (job['dataset'], job['fold'], job['shot'])
            if pairing in reference_targets and not np.array_equal(reference_targets[pairing], targets):
                raise ValueError(f'Paired targets mismatch: {path}')
            reference_targets[pairing] = targets.copy()
        key = (job['dataset'], job['shot'], method, ckpt, *cond, job['seed'], job['fold'])
        result[key] = accuracy
    return result


def cell_array(matrix, dataset, shot, method, checkpoint, condition, seeds):
    import numpy as np
    from .campaign import DATASETS
    return np.stack([[matrix[(dataset, shot, method, checkpoint, *condition, seed, fold)]
                      for fold in range(DATASETS[dataset])] for seed in seeds])


def robust_array(matrix, dataset, shot, method, seeds, operator=None):
    import numpy as np
    from .evaluate import OPERATORS
    # Average all severities within each operator, then operators. Preserve E.
    return np.stack([np.stack([cell_array(matrix, dataset, shot, method, 'best',
                           ('query', op, s), seeds) for s in range(1, 6)]).mean(0)
                     for op in ([operator] if operator else OPERATORS)]).mean(0)


def select_candidates(out, train_seeds=(7,)):
    from pathlib import Path
    from .campaign import DATASETS
    from .train import write_json
    matrix = load_matrix(out, [7], 'dev')
    decisions = {}
    for method in ('F2', 'J2'):
        clean, robust = [], []
        for dataset in DATASETS:
            for shot in (1, 5):
                clean.append(float(100 * (cell_array(matrix, dataset, shot, method, 'best', ('clean', None, 0), [7]) -
                                         cell_array(matrix, dataset, shot, 'P0', 'best', ('clean', None, 0), [7])).mean()))
                if not dataset.startswith('Scan'):
                    robust.append(float(100 * (robust_array(matrix, dataset, shot, method, [7]) -
                                               robust_array(matrix, dataset, shot, 'P0', [7])).mean()))
        decisions[method] = dict(retained=retention_decision(clean, robust), clean_delta_pp=clean, robust_delta_pp=robust)
    families = ['P0', 'J', 'H'] if decisions['J2']['retained'] else (['P0', 'H'] if decisions['F2']['retained'] else [])
    result = dict(source_pool='dev', selection_seed=7, candidates=decisions,
                  additional_seeds=[17, 27] if families else [], additional_families=families,
                  note='Wide development retention, not the +3pp success criterion. Novel results not read.')
    path = Path(out) / 'selection.json'
    if path.exists():
        import json
        if json.loads(path.read_text()) != result:
            raise ValueError('Refusing to overwrite a different frozen selection.json')
    else:
        write_json(path, result)
    print(__import__('json').dumps(result, indent=2))
    return result


def write_report(out, train_seeds=(7,)):
    import csv
    import json
    from pathlib import Path
    from .campaign import DATASETS
    from .evaluate import METHODS, evaluation_conditions, OPERATORS
    from .train import write_json
    matrix = load_matrix(out, train_seeds, 'test')
    tables = {'protocol_main': [], 'robust_query': [], 'robust_aux': [], 'paired_effects': []}
    for dataset in DATASETS:
        for shot in (1, 5):
            for method in METHODS:
                seeds = method_seeds(out, train_seeds, method)
                clean = cell_array(matrix, dataset, shot, method, 'best', ('clean', None, 0), seeds)
                last = cell_array(matrix, dataset, shot, method, 'last', ('clean', None, 0), seeds)
                tables['protocol_main'].append(dict(dataset=dataset, shot=shot, method=method,
                    train_seeds=seeds, best_accuracy_pct=float(100 * clean.mean()), last_accuracy_pct=float(100 * last.mean()),
                    paper_style_max_pct=float(100 * max(clean.mean(), last.mean())),
                    max_note='additional descriptive only' if dataset.startswith('Scan') else 'max of fold means, never per-fold max'))
                for cond in evaluation_conditions(dataset, method, 'best')[1:]:
                    array = cell_array(matrix, dataset, shot, method, 'best', cond, seeds)
                    row = dict(dataset=dataset, shot=shot, method=method, side=cond[0], operator=cond[1], severity=cond[2],
                               accuracy_pct=float(100 * array.mean()), drop_from_clean_pp=float(100 * (clean - array).mean()))
                    tables['robust_query' if cond[0] == 'query' else 'robust_aux'].append(row)
                if not dataset.startswith('Scan'):
                    array = robust_array(matrix, dataset, shot, method, seeds)
                    tables['robust_query'].append(dict(dataset=dataset, shot=shot, method=method, side='query', operator='equal_weight_mean', severity='all',
                        accuracy_pct=float(100 * array.mean()), drop_from_clean_pp=float(100 * (clean - array).mean())))
                if method == 'P0':
                    continue
                base = cell_array(matrix, dataset, shot, 'P0', 'best', ('clean', None, 0), seeds)
                comparisons = [('clean', clean, base)]
                if not dataset.startswith('Scan'):
                    comparisons.append(('robust_query', robust_array(matrix, dataset, shot, method, seeds), robust_array(matrix, dataset, shot, 'P0', seeds)))
                    comparisons.extend((op, robust_array(matrix, dataset, shot, method, seeds, op), robust_array(matrix, dataset, shot, 'P0', seeds, op)) for op in OPERATORS)
                for setting, candidate, baseline in comparisons:
                    effect = paired_aggregation(candidate.tolist(), baseline.tolist(), shared_folds=not dataset.startswith('Scan'))
                    tables['paired_effects'].append(dict(dataset=dataset, shot=shot, method=method, reference='P0', setting=setting, train_seeds=seeds, **effect))
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        with (root / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)
    decisions = {}
    for method in ('F2', 'J2'):
        effects = [r for r in tables['paired_effects'] if r['method'] == method]
        clean = [r for r in effects if r['setting'] == 'clean']
        robust = [r for r in effects if r['setting'] == 'robust_query']
        operators = [r for r in effects if r['setting'] in OPERATORS]
        point = all(r['delta_pp'] >= -1 for r in clean) and all(r['delta_pp'] >= 3 for r in robust) and all(r['delta_pp'] >= -2 for r in operators)
        uncertain = any(r['ci95_pp'][0] < -1 for r in clean) or any(r['ci95_pp'][0] <= 0 for r in robust)
        decisions[method] = 'provisionally meets target, evidence insufficient' if point and uncertain else ('meets predefined exchange target' if point else 'does not meet cross-setting target; inspect local effects')
    write_json(root / 'report_status.json', dict(complete=True, train_seeds=list(train_seeds), decisions=decisions))
    (root / 'REPORT.md').write_text('# PCIA + HPR-D adapted experiment\n\n'
        'Complete required prediction matrix; all accuracy values recomputed from episode predictions.\n\n'
        + '\n'.join(f'- {method}: {decision}' for method, decision in decisions.items()) + '\n\n'
        'See protocol_main.csv (best/last/max of means), robust_query.csv, robust_aux.csv, paired_effects.csv.\n'
        'Effect decisions always use best-validation. Auxiliary support corruption is not part of the main gain.\n'
        '95% percentile paired bootstrap: 2000 draws, seed400007; shared synthetic novel episode IDs remain shared across folds/operators/severities. '
        'Scan pools resample separately. Fixed folds are not independent training seeds. Multi-seed intervals resample training seeds and keep paired episodes; '
        'few seeds and repeated objects limit inference. No paper accuracy is substituted for local results.\n')
    return tables


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--train-seeds', type=int, nargs='+', default=[7])
    args = p.parse_args(argv)
    try:
        write_report(args.out, args.train_seeds)
    except (ValueError, FileNotFoundError) as exc:
        p.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
