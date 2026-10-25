"""Read-only, protocol-aware statistics for the frozen calibration study.

No training, model selection, or test-driven rule selection occurs here.  Raw
predictions are scored externally. Partial reports omit incomplete aggregates.
"""
import argparse
import csv
import hashlib
import itertools
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

KEY_FIELDS = ('protocol', 'pool', 'dataset', 'fold', 'shot', 'seed', 'method',
              'checkpoint', 'side', 'operator', 'severity')
OPERATORS = ('Scale', 'Jitter', 'Drop-Global', 'Drop-Local', 'Add-Global', 'Add-Local', 'Rotate')
METHODS = ('P0', 'E2', 'UNIT', 'QONLY', 'SOFTK')
METRICS = ('micro', 'macro')
SEEDS = (7, 17, 27)


def cell_key(cell):
    return tuple(cell[k] for k in KEY_FIELDS)


def digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def score_predictions(target, before_logits, pred, shot, method):
    """Return episode metrics and externally scored mechanisms from raw arrays.

    Macro transition weights are 1 / (C * true-class query count), while micro
    weights are 1 / M. Group reliability is agreement with the prototype class,
    never majority-label purity. Empty groups remain NaN.
    """
    target, logits, pred = np.asarray(target), np.asarray(before_logits), np.asarray(pred)
    if target.ndim != 2 or pred.shape != target.shape or logits.shape != (*target.shape, 5):
        raise ValueError('Prediction/target/logit shapes differ')
    if not np.isfinite(logits).all() or not np.isin(target, range(5)).all() or not np.isin(pred, range(5)).all():
        raise ValueError('Invalid labels or logits')
    if shot not in (1, 5) or method not in METHODS:
        raise ValueError('Unsupported shot/method')
    before = logits.argmax(-1)
    if method == 'P0' and not np.array_equal(before, pred):
        raise ValueError('P0 must equal initial-logit argmax')
    quota = np.stack([(target == c).sum(-1) for c in range(5)], axis=-1)
    if np.any(quota == 0):
        raise ValueError('Macro accuracy requires all five classes')
    weights = 1. / (5 * np.take_along_axis(quota, target.astype(int), axis=1))
    old, new = before == target, pred == target
    corrected, damaged = ~old & new, old & ~new
    values = dict(micro=new.mean(-1), macro=(new * weights).sum(-1),
                  initial_micro=old.mean(-1), initial_macro=(old * weights).sum(-1),
                  corrected_micro=corrected.mean(-1), damaged_micro=damaged.mean(-1),
                  corrected_macro=(corrected * weights).sum(-1),
                  damaged_macro=(damaged * weights).sum(-1), query_quota=quota)
    for metric in METRICS:
        net = values['corrected_' + metric] - values['damaged_' + metric]
        if not np.allclose(net, values[metric] - values['initial_' + metric], atol=1e-12, rtol=0):
            raise ValueError('Transition contribution does not reproduce accuracy increment')
    if np.all(quota == quota[:, :1]) and not np.allclose(values['micro'], values['macro'], atol=1e-12, rtol=0):
        raise ValueError('Balanced macro must equal micro')
    stable = logits.astype(np.float64) - logits.max(-1, keepdims=True)
    prob = np.exp(stable)
    prob /= prob.sum(-1, keepdims=True)
    logp = np.log(np.maximum(prob, np.finfo(float).tiny))
    values.update(soft_entropy=-(prob * logp).sum(-1), soft_max_probability=prob.max(-1),
                  soft_mass=prob.sum(1))
    size = np.stack([(before == c).sum(-1) for c in range(5)], axis=-1)
    agreement = np.full(size.shape, np.nan)
    purity = np.full(size.shape, np.nan)
    grouped = {name: np.zeros(size.shape) for name in
               ('corrected_micro', 'damaged_micro', 'corrected_macro', 'damaged_macro')}
    for c in range(5):
        mask = before == c
        np.divide((mask & (target == c)).sum(-1), size[:, c], out=agreement[:, c], where=size[:, c] > 0)
        counts = np.stack([(mask & (target == label)).sum(-1) for label in range(5)], axis=-1)
        np.divide(counts.max(-1), size[:, c], out=purity[:, c], where=size[:, c] > 0)
        for name, flipped in (('corrected', corrected), ('damaged', damaged)):
            grouped[name + '_micro'][:, c] = (mask & flipped).mean(-1)
            grouped[name + '_macro'][:, c] = ((mask & flipped) * weights).sum(-1)
    if method == 'P0':
        anchor = np.ones(size.shape)
    elif method == 'SOFTK':
        anchor = shot / (shot + values['soft_mass'])
    else:
        mass = {'E2': shot, 'UNIT': 1, 'QONLY': 0}[method]
        anchor = np.divide(mass, mass + size, out=np.ones(size.shape), where=size > 0)
    values.update(group_size=size, group_class_accuracy=agreement, group_purity=purity,
                  anchor_fraction=anchor, **{'group_' + k: v for k, v in grouped.items()})
    return values


@lru_cache(maxsize=4)
def _bootstrap_indices(episodes, resamples):
    indices = np.random.RandomState(400007).randint(episodes, size=(resamples, episodes))
    indices.flags.writeable = False
    return indices


def bootstrap(candidate, baseline, resamples=2000):
    """One training seed; shared episode indices across all fixed synthetic folds."""
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    if candidate.shape != baseline.shape or candidate.ndim != 2 or not candidate.size:
        raise ValueError('Expected paired fold-by-episode arrays')
    delta = 100 * (candidate - baseline)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite paired delta')
    # Linear averaging over fixed folds commutes with shared episode resampling.
    per_episode = delta.mean(0)
    indices = _bootstrap_indices(delta.shape[1], resamples)
    low, high = np.percentile(per_episode[indices].mean(-1), [2.5, 97.5])
    return dict(delta_pp=float(delta.mean()), ci95_pp=[float(low), float(high)],
                bootstrap_seed=400007, resamples=resamples,
                uncertainty='conditional_episode_only', shared_folds=True)


def _path(out, cell):
    condition = 'clean' if cell['side'] == 'clean' else f"{cell['side']}_{cell['operator']}_{cell['severity']}"
    return (Path(out) / 'protocols' / cell['protocol'] / 'predictions' / cell['pool'] /
            cell['dataset'] / f"fold{cell['fold']}" / f"shot{cell['shot']}" /
            f"seed{cell['seed']}" / cell['method'] / cell['checkpoint'] / (condition + '.npz'))


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_predictions(out, cells, partial=False, reuse_index=None):
    """New-format records only. Legacy reuse must be explicitly adopted upstream."""
    cells = list(cells)
    keys = [cell_key(c) for c in cells]
    if not cells or len(set(keys)) != len(keys):
        raise ValueError('Empty or duplicate frozen matrix')
    matrix, diagnostics, sources, missing, identities = {}, [], [], [], {}
    protocol_queries = {}
    reused = {}
    if reuse_index is not None:
        index_path = Path(reuse_index).resolve()
        index = json.loads(index_path.read_text())
        from .empirical_protocol import freeze_document, prediction_cells
        if index.get('version') != 1 or index.get('matrix_sha256') != freeze_document()['matrix_sha256']:
            raise ValueError('Reuse index protocol identity differs')
        reused = {cell_key(e['cell']): e for e in index['entries']}
        if (len(reused) != len(index['entries']) or len(reused) != 600 or
                set(reused) != {cell_key(c) for c in prediction_cells() if c['reused']}):
            raise ValueError('Reuse index does not identify the 600 authorized cells')


    def paired(key, value, path):
        if key in identities and identities[key] != value:
            raise ValueError(f'Unpaired input identity {key}: {path}')
        identities[key] = value

    for c in cells:
        key, path = cell_key(c), _path(out, c)
        entry = reused.get(key)
        if entry:
            path = (index_path.parent / entry['prediction']).resolve()
            metadata = (index_path.parent / entry['metadata']).resolve()
        else:
            metadata = path.with_suffix('.json')
        if not path.is_file() or not metadata.is_file():
            missing.append(list(key))
            continue
        meta = json.loads(metadata.read_text())
        if entry:
            for name, source in (('prediction', path), ('metadata', metadata)):
                if sha256(source) != entry[name + '_sha256']:
                    raise ValueError(f'Legacy {name} bytes changed after validation: {source}')
            meta = dict(meta, **{k:entry[k] for k in ('protocol', 'manifest_sha256', 'source_manifest_sha256',
                                                    'point_indices_sha256', 'checkpoint_sha256')})
        expected = {k: c[k] for k in ('protocol', 'pool', 'dataset', 'fold', 'shot', 'seed', 'method', 'checkpoint')}
        expected.update(family='P0', condition=[c['side'], c['operator'], c['severity']], episodes=700, smoke=False)
        if any(meta.get(k) != v for k, v in expected.items()) or meta.get('smoke') is not False:
            raise ValueError(f'Metadata does not identify complete frozen cell: {path}')
        pair = (c['protocol'], c['dataset'], c['shot'])
        model_pair = (*pair, c['fold'], c['seed'], c['checkpoint'])
        for name in ('manifest_sha256', 'source_manifest_sha256', 'point_indices_sha256', 'checkpoint_sha256'):
            value = meta.get(name, '')
            if len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                raise ValueError(f'Missing/invalid {name}: {path}')
            # Source inputs and weights are invariant across query protocols;
            # only the newly sampled manifest itself is protocol-specific.
            identity = (c['dataset'], c['fold'], c['shot'], c['seed'], c['checkpoint']) if name == 'checkpoint_sha256' else \
                       pair if name == 'manifest_sha256' else (c['dataset'], c['shot'])
            paired((name, *identity), value, path)
        if not entry:
            for name, identity in (('data_sha256', (c['dataset'],)), ('scientific_code_sha256', ())):
                value = meta.get(name, '')
                if len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                    raise ValueError(f'Missing/invalid {name}: {path}')
                paired((name, *identity), value, path)
            if not isinstance(meta.get('scientific_code_files'), dict) or not meta['scientific_code_files']:
                raise ValueError(f'Missing scientific source fingerprints: {path}')
            code_hash = hashlib.sha256(json.dumps(meta['scientific_code_files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if code_hash != meta['scientific_code_sha256']:
                raise ValueError(f'Scientific source fingerprints do not match their identity: {path}')
        condition_pair = (*model_pair, c['side'], c['operator'], c['severity'])
        with np.load(path, allow_pickle=False) as archive:
            # NpzFile decompresses an array on every lookup. Materialize the
            # required arrays once; validation and all statistics are unchanged.
            names = ('local_target', 'pred', 'logits_before', 'logits_after', 'episode_id',
                     'episode_accuracy', 'query_quota', 'corruption_seeds', 'macro_accuracy',
                     'micro_accuracy', 'episode_rows', 'support_rows', 'query_rows', 'class_ids')
            data = {name: archive[name] for name in names if name in archive}
            shapes = dict(local_target=(700, 75), pred=(700, 75), logits_before=(700, 75, 5),
                          logits_after=(700, 75, 5), episode_id=(700,), episode_accuracy=(700,),
                          query_quota=(700, 5), corruption_seeds=(700, 5 * c['shot'] + 75))
            if entry:
                shapes.pop('query_quota')
            if any(name not in data or data[name].shape != shape for name, shape in shapes.items()):
                raise ValueError(f'Incomplete prediction arrays: {path}')
            if not np.array_equal(data['episode_id'], np.arange(700)) or not np.isfinite(data['logits_after']).all():
                raise ValueError(f'Invalid episode IDs/final logits: {path}')
            if not np.array_equal(data['pred'], data['logits_after'].argmax(-1)):
                raise ValueError(f'Prediction and logit argmax differ: {path}')
            scores = score_predictions(data['local_target'], data['logits_before'], data['pred'], c['shot'], c['method'])
            quota = scores['query_quota']
            expected_quota = [15] * 5 if c['protocol'] == 'balanced' else [5, 10, 15, 20, 25]
            if c['protocol'] not in ('balanced', 'imbalanced') or not np.all(np.sort(quota, axis=-1) == expected_quota):
                raise ValueError(f'Query quotas disagree with protocol: {path}')
            if 'query_quota' in data and not np.array_equal(quota, data['query_quota']):
                raise ValueError(f'Stored query quota differs from targets: {path}')
            for stored, metric in (('episode_accuracy', 'micro'), ('micro_accuracy', 'micro'), ('macro_accuracy', 'macro')):
                if entry and stored not in data and stored != 'episode_accuracy':
                    continue
                if stored not in data or not np.allclose(data[stored], scores[metric], atol=1e-8, rtol=1e-6):
                    raise ValueError(f'Stored {stored} differs from raw predictions: {path}')
            for name in ('local_target', 'episode_id', 'episode_rows', 'support_rows', 'query_rows', 'class_ids'):
                if name not in data:
                    raise ValueError(f'Missing pairing arrays {name}: {path}')
                identity = (c['dataset'], c['shot']) if name in ('support_rows', 'class_ids') else pair
                paired((name, *identity), digest(data[name]), path)
            if pair not in protocol_queries:
                queries = data['query_rows'].copy()
                if queries.shape != (700, 75):
                    raise ValueError(f'Invalid query rows: {path}')
                opposite = ('imbalanced' if c['protocol'] == 'balanced' else 'balanced', c['dataset'], c['shot'])
                if opposite in protocol_queries:
                    shared = (queries[:, :, None] == protocol_queries[opposite][:, None, :]).any(-1).sum(-1)
                    if not np.all(shared == 60):
                        raise ValueError(f'Cross-protocol query overlap must be exactly 60 per episode: {path}')
                protocol_queries[pair] = queries
            paired(('corruption_seeds', *pair, c['side'], c['operator'], c['severity']), digest(data['corruption_seeds']), path)
            paired(('logits_before', *condition_pair), digest(data['logits_before']), path)
            matrix[key] = {k: scores[k] for k in METRICS}
            diagnostics.extend(diagnostic_rows(c, scores))
        sources.append(dict(key=list(key), prediction=str(path), metadata=str(metadata),
                            prediction_sha256=sha256(path), metadata_sha256=sha256(metadata)))
    coverage = dict(expected=len(cells), present=len(matrix), complete=not missing,
                    missing_keys=missing, key_fields=list(KEY_FIELDS))
    if missing and not partial:
        raise ValueError(f'Incomplete frozen matrix: {len(missing)} missing prediction cells')
    return matrix, diagnostics, coverage, sources


def diagnostic_rows(cell, scores):
    """Class and fixed reliability-bin summaries; no bins selected from outcomes."""
    common = {k: cell[k] for k in KEY_FIELDS}
    size, agreement = scores['group_size'], scores['group_class_accuracy']
    rows = []
    for c in range(5):
        seen = size[:, c] > 0
        row = dict(**common, scope='prototype_class', class_id=c, episodes=len(size),
                   empty_group_rate=float((~seen).mean()), group_size=float(size[:, c].mean()),
                   group_class_accuracy=float(agreement[seen, c].mean()) if seen.any() else None,
                   group_purity=float(scores['group_purity'][seen, c].mean()) if seen.any() else None,
                   anchor_fraction=float(scores['anchor_fraction'][:, c].mean()),
                   soft_mass=float(scores['soft_mass'][:, c].mean()) if cell['method'] == 'SOFTK' else None)
        for metric in METRICS:
            for name in ('corrected', 'damaged'):
                row[name + '_' + metric + '_pp'] = float(100 * scores['group_' + name + '_' + metric][:, c].mean())
        rows.append(row)
    episode = dict(**common, scope='episode', episodes=len(size),
                   group_size_sd=float(size.astype(float).std(ddof=0)),
                   soft_entropy=float(scores['soft_entropy'].mean()) if cell['method'] == 'SOFTK' else None,
                   soft_max_probability=float(scores['soft_max_probability'].mean()) if cell['method'] == 'SOFTK' else None)
    for metric in METRICS:
        for name in ('initial', 'corrected', 'damaged'):
            episode[name + '_' + metric + '_pp'] = float(100 * scores[name + '_' + metric].mean())
        episode['net_' + metric + '_pp'] = float(100 * (scores[metric] - scores['initial_' + metric]).mean())
    rows.append(episode)
    for low, high in zip((0., .25, .5, .75), (.25, .5, .75, 1.)):
        selected = (agreement >= low) & ((agreement < high) if high < 1 else (agreement <= high))
        count = int(selected.sum())
        row = dict(**common, scope='reliability_bin', lower=low, upper=high,
                   upper_inclusive=high == 1, nonempty_groups=count,
                   group_class_accuracy=float(agreement[selected].mean()) if count else None,
                   group_size=float(size[selected].mean()) if count else None,
                   anchor_fraction=float(scores['anchor_fraction'][selected].mean()) if count else None)
        for metric in METRICS:
            for name in ('corrected', 'damaged'):
                # Contribution per original episode; summing bins reconstructs
                # each episode-wide contribution (empty groups contain no query).
                row[name + '_' + metric + '_pp'] = float(100 * scores['group_' + name + '_' + metric][selected].sum() / len(size))
        rows.append(row)
    return rows


def scopes(protocol, shot):
    """All predeclared primary, auxiliary and detailed comparisons."""
    clean = ('clean', None, 0)
    query = [('query', op, s) for op in OPERATORS for s in (range(1, 6) if protocol == 'balanced' and shot == 1 else [3])]
    result = [('clean', 'clean', None, 0, [clean]),
              ('query_all' if protocol == 'balanced' and shot == 1 else 'query_s3', 'query', 'seven_operator_mean',
               'all' if protocol == 'balanced' and shot == 1 else 3, query)]
    conditions = list(query)
    if protocol == 'balanced':
        for side in ('support', 'both'):
            auxiliary = [(side, op, 3) for op in ('Add-Global', 'Drop-Local')]
            result.append(('auxiliary', side, 'two_operator_mean', 3, auxiliary))
            conditions += auxiliary
    if protocol == 'balanced' and shot == 1:
        for op in OPERATORS:
            result.append(('operator', 'query', op, 'all', [('query', op, s) for s in range(1, 6)]))
        for s in range(1, 6):
            result.append(('query_s3' if s == 3 else 'severity_mean', 'query', 'seven_operator_mean', s,
                           [('query', op, s) for op in OPERATORS]))
    result += [('condition', *condition, [condition]) for condition in conditions]
    return result


def aggregate(matrix, cells, resamples=2000):
    """Only complete fixed-fold aggregates; seed summaries require 7/17/27."""
    cells = list(cells)
    allowed = {cell_key(c) for c in cells}
    if set(matrix) - allowed:
        raise ValueError('Predictions outside frozen matrix')
    tables = {k: [] for k in ('per_seed', 'per_fold', 'paired_effects', 'seed_summary')}
    groups = sorted({(c['protocol'], c['dataset'], c['shot']) for c in cells})
    for protocol, dataset, shot in groups:
        methods = [m for m in METHODS if any(c['method'] == m and (c['protocol'], c['dataset'], c['shot']) == (protocol, dataset, shot) for c in cells)]
        group_cells = [c for c in cells if (c['protocol'], c['dataset'], c['shot']) == (protocol, dataset, shot)]
        folds = sorted({c['fold'] for c in group_cells})
        seeds = sorted({c['seed'] for c in group_cells})
        for setting, side, op, severity, conditions in scopes(protocol, shot):
            for seed in seeds:
                for metric in METRICS:
                    arrays = {}
                    for method in methods:
                        keys = [[(protocol, 'test', dataset, f, shot, seed, method, 'best', *cond) for f in folds] for cond in conditions]
                        if all(k in matrix for batch in keys for k in batch):
                            arrays[method] = np.stack([np.stack([matrix[k][metric] for k in batch]) for batch in keys]).mean(0)
                    common = dict(protocol=protocol, pool='test', dataset=dataset, shot=shot, seed=seed,
                                  setting=setting, side=side, operator=op, severity=severity, metric=metric,
                                  primary_metric='macro' if protocol == 'imbalanced' else 'micro', folds=folds)
                    for method, value in arrays.items():
                        tables['per_seed'].append(dict(**common, method=method, accuracy_pct=float(100 * value.mean())))
                        for fold, acc in zip(folds, value.mean(-1)):
                            tables['per_fold'].append(dict(**common, method=method, fold=fold, accuracy_pct=float(100 * acc)))
                    for baseline, candidate in itertools.combinations(methods, 2):
                        if candidate in arrays and baseline in arrays:
                            row = dict(**common, comparison=candidate + '-' + baseline,
                                       **bootstrap(arrays[candidate], arrays[baseline], resamples=resamples))
                            tables['paired_effects'].append(row)
                            for fold, delta in zip(folds, (arrays[candidate] - arrays[baseline]).mean(-1)):
                                tables['per_fold'].append(dict(**common, comparison=candidate + '-' + baseline,
                                                               fold=fold, delta_pp=float(100 * delta)))
    grouped = {}
    fields = ('protocol', 'pool', 'dataset', 'shot', 'setting', 'side', 'operator', 'severity', 'metric', 'primary_metric')
    for row in tables['per_seed'] + tables['paired_effects']:
        identity = {k: row[k] for k in fields}
        identity.update({k: row[k] for k in ('method', 'comparison') if k in row})
        key = tuple(identity.items())
        group = grouped.setdefault(key, dict(identity=identity, seeds={}))
        if row['seed'] in group['seeds']:
            raise ValueError('Duplicate training seed in summary')
        group['seeds'][row['seed']] = row.get('accuracy_pct', row.get('delta_pp'))
    for group in grouped.values():
        if set(group['seeds']) == set(SEEDS):
            values = [group['seeds'][s] for s in SEEDS]
            tables['seed_summary'].append(dict(**group['identity'], train_seeds=list(SEEDS),
                per_seed_values=values, n_train_seeds=3, mean=float(np.mean(values)),
                sample_sd=float(np.std(values, ddof=1)), unit='percent' if 'method' in group['identity'] else 'pp',
                uncertainty='between_training_seed_sample_sd'))
    return tables


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['protocol']
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def report(out, cells, destination, partial=False, reuse_index=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    matrix, diagnostics, coverage, sources = load_predictions(out, cells, partial=True, reuse_index=reuse_index)
    _json(destination / 'coverage.json', coverage)
    _json(destination / 'sources.json', dict(key_fields=list(KEY_FIELDS), predictions=sources))
    status = dict(complete=coverage['complete'] and not partial, status='partial' if partial else
                  'complete' if coverage['complete'] else 'incomplete', **{k: coverage[k] for k in ('expected', 'present')})
    _json(destination / 'status.json', status)
    if not coverage['complete'] and not partial:
        for name in ('per_seed', 'per_fold', 'paired_effects', 'seed_summary', 'diagnostics'):
            (destination / (name + '.csv')).unlink(missing_ok=True)
        (destination / 'REPORT.md').write_text('Incomplete: frozen matrix has missing cells; see coverage.json. No final statistical report.\n')
        raise ValueError('Incomplete matrix; final reporting refused')
    tables = aggregate(matrix, cells)
    tables['diagnostics'] = diagnostics
    for name, rows in tables.items():
        write_csv(destination / (name + '.csv'), rows)
    lines = ['# Prototype calibration empirical study', '',
             '**COMPLETE frozen matrix.**' if status['complete'] else '**PARTIAL: progress only; not final evidence.**', '',
             f"Prediction cells: {coverage['present']}/{coverage['expected']}.", '',
             'Primary accuracy: micro for balanced batches; macro for imbalanced batches. Both metrics are retained.',
             'Conditional 95% intervals use 2,000 paired episode resamples (seed 400007), shared across fixed synthetic folds; correlated conditions are averaged per episode first.',
             'Training seeds 7/17/27 are summarized separately by mean and sample SD (ddof=1); incomplete seed groups are omitted. Episode intervals are not training-seed confidence intervals.',
             'All primary rule contrasts are within protocol and episode. Balanced versus imbalanced batches share support and 60/75 queries; changed objects and full frontend interactions prevent a pure class-proportion causal interpretation.',
             'Diagnostics are associations. Empty group agreement is missing; macro and micro correction/damage use their own weights. Soft entropy uses natural logarithms of initial distance-softmax responsibilities.', '',
             '| Protocol | Dataset | Shot | Setting | Side | Metric | Method / contrast | Mean | Sample SD |',
             '|---|---|---:|---|---|---|---|---:|---:|']
    for row in tables['seed_summary']:
        if row['setting'] not in ('clean', 'query_all', 'query_s3', 'auxiliary') or row['metric'] != row['primary_metric']:
            continue
        lines.append('| ' + ' | '.join(str(row[k]) for k in ('protocol', 'dataset', 'shot', 'setting', 'side', 'metric')) +
                     f" | {row.get('method', row.get('comparison'))} | {row['mean']:.3f} | {row['sample_sd']:.3f} |")
    (destination / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    _json(destination / 'status.json', status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--destination')
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--reuse-index')
    args = parser.parse_args(argv)
    from .empirical_protocol import prediction_cells
    status = report(args.out, prediction_cells(), args.destination or Path(args.out) / 'reports', partial=args.partial, reuse_index=args.reuse_index)
    print(json.dumps(status))


if __name__ == '__main__':
    main()
