"""Read-only, fixed seed7 D0/D2 versus P0/E2 review1 reporting."""
import json
from itertools import product
from pathlib import Path

import numpy as np

from .evaluate import OPERATORS, prediction_path
from .report_review import KEY_FIELDS, _digest, cell_key
from .summarize import IncompleteResults, cell_array

METHODS = ('D0', 'D2', 'P0', 'E2')


def expected_cells():
    conditions = [('clean', None, 0)] + [('query', op, s) for op in OPERATORS for s in range(1, 6)]
    return [dict(pool='test', dataset=d, fold=f, shot=k, seed=7, method=m,
                 checkpoint='best', side=c[0], operator=c[1], severity=c[2])
            for d, f, k, m, c in product(('ModelNet40', 'ShapeNet70'), range(5),
                                        (1, 5), METHODS, conditions)]


def _cells(cells):
    cells = list(expected_cells() if cells is None else cells)
    keys = [cell_key(c) for c in cells]
    allowed = {cell_key(c) for c in expected_cells()}
    if not keys or len(keys) != len(set(keys)) or not set(keys) <= allowed:
        raise ValueError('Expected unique fixed seed7/best D0/D2/P0/E2 protocol keys')
    return cells


def load_predictions(new_out, reference_out, *, cells=None, partial=False):
    """Read explicit raw prediction paths only; never create episode manifests/locks.

    Top-level coverage counts NEW D predictions. Reference coverage is separate.
    Paths in a copied manifest's metadata may differ; compare arrays, not strings.
    """
    cells = _cells(cells)
    matrix, missing, sources = {}, [], []
    pairing, manifests = {}, {}

    def paired(key, value):
        if key in pairing and pairing[key] != value:
            raise ValueError(f'Unpaired artifact: {key}')
        pairing[key] = value

    for c in cells:
        key = cell_key(c)
        root = Path(new_out if c['method'] in ('D0', 'D2') else reference_out)
        path = prediction_path(root, c, 'test', c['method'], 'best',
                               (c['side'], c['operator'], c['severity']))
        metadata = path.with_suffix('.json')
        index_path = Path(reference_out) / 'reuse_index.json'
        if c['method'] in ('P0', 'E2') and not path.exists() and not metadata.exists() and index_path.is_file():
            index = json.loads(index_path.read_text())
            if index.get('version') != 1:
                raise ValueError('Unsupported reuse index version')
            entries = [entry for entry in index['cells'] if cell_key(entry['cell']) == key]
            if len(entries) > 1:
                raise ValueError(f'Duplicate reuse index key: {key}')
            if entries:
                path, metadata = (Path(entries[0][name]) for name in ('prediction', 'metadata'))
                if not path.is_absolute() or not metadata.is_absolute():
                    raise ValueError('Reuse paths must be absolute')
        sources.append(dict(key=list(key), prediction=str(path), metadata=str(metadata)))
        if not path.is_file() or not metadata.is_file():
            missing.append(list(key))
            continue
        meta = json.loads(metadata.read_text())
        family = 'D' if c['method'] in ('D0', 'D2') else 'P0'
        identity = {k: c[k] for k in ('dataset', 'fold', 'shot', 'seed', 'method', 'checkpoint', 'pool')}
        identity.update(family=family, condition=[c['side'], c['operator'], c['severity']],
                        smoke=False, episodes=700)
        if any(meta.get(k) != v for k, v in identity.items()) or meta.get('smoke') is not False:
            raise ValueError(f'Prediction metadata identity mismatch: {path}')
        if meta.get('checkpoint_epoch') is None or not meta.get('manifest'):
            raise ValueError(f'Missing checkpoint/manifest: {path}')
        checkpoint_key = (c['dataset'], c['fold'], c['shot'], family)
        paired(('checkpoint', *checkpoint_key), meta['checkpoint_epoch'])
        if meta.get('checkpoint_path'):
            paired(('checkpoint_path', *checkpoint_key), meta['checkpoint_path'])
        # Prefer the colocated copied assets over stale absolute metadata paths.
        local = root / 'episodes/test' / c['dataset'] / 'fold0' / f"shot{c['shot']}" / 'episodes.npz'
        manifest = local if local.is_file() else Path(meta['manifest'])
        if manifest not in manifests:
            with np.load(manifest, allow_pickle=False) as data:
                manifests[manifest] = {k: data[k].copy() for k in
                    ('rows', 'support_rows', 'query_rows', 'class_ids', 'query_y')}
            with np.load(manifest.parent / 'point_indices.npz', allow_pickle=False) as data:
                points = data['points']
                if points.shape != (512,):
                    raise ValueError(f'Invalid point indices: {manifest}')
                manifests[manifest]['points'] = points.copy()
        episode = manifests[manifest]
        pair = (c['dataset'], c['shot'])  # synthetic novel folds share episodes
        for name, array in episode.items():
            paired((*pair, name), _digest(array))
        with np.load(path, allow_pickle=False) as saved:
            n, objects = 700, 5 * c['shot'] + 75
            shapes = dict(pred=(n, 75), local_target=(n, 75), episode_id=(n,),
                episode_accuracy=(n,), corruption_seeds=(n, objects), point_counts=(n, objects),
                logits_before=(n, 75, 5), logits_after=(n, 75, 5))
            if any(k not in saved or saved[k].shape != shape for k, shape in shapes.items()):
                raise ValueError(f'Incomplete prediction arrays: {path}')
            target, pred = saved['local_target'], saved['pred']
            accuracy = (pred == target).mean(-1)
            if (not np.isin(target, range(5)).all() or not np.isin(pred, range(5)).all() or
                    not np.array_equal(saved['episode_id'], np.arange(n)) or
                    not np.allclose(accuracy, saved['episode_accuracy'], atol=1e-8, rtol=1e-6) or
                    not np.array_equal(pred, saved['logits_after'].argmax(-1)) or
                    any(not np.isfinite(saved[k]).all() for k in ('logits_before', 'logits_after'))):
                raise ValueError(f'Invalid predictions/IDs/accuracy: {path}')
            for name, source in (('local_target', 'query_y'), ('episode_rows', 'rows'),
                                 ('support_rows', 'support_rows'), ('query_rows', 'query_rows'),
                                 ('class_ids', 'class_ids')):
                # Legacy evaluate NPZ omits rows: its read-only manifest supplies them.
                if name in saved and not np.array_equal(saved[name], episode[source]):
                    raise ValueError(f'Unpaired {name}: {path}')
            for name in ('corruption_seeds', 'point_counts'):
                paired((*pair, c['fold'], c['side'], c['operator'], c['severity'], name),
                       _digest(saved[name]))
            matrix[key] = accuracy

    def count(methods):
        keys = [cell_key(c) for c in cells if c['method'] in methods]
        return dict(expected=len(keys), present=sum(k in matrix for k in keys),
                    missing_keys=[list(k) for k in keys if list(k) in missing])

    coverage = dict(**count(('D0', 'D2')), reference=count(('P0', 'E2')),
                    complete=not missing, key_fields=list(KEY_FIELDS), sources=sources)
    if missing and not partial:
        raise IncompleteResults('incomplete: missing prediction cells')
    return matrix, coverage


def build_tables(matrix, *, cells=None):
    """Query means enter as episode accuracies; episodes then folds are equal-weight.

    Explicit miniature cells are for artificial unit fixtures, not formal coverage.
    """
    cells = _cells(cells)
    if set(matrix) != {cell_key(c) for c in cells}:
        raise IncompleteResults('incomplete: missing or unexpected prediction keys')
    for value in matrix.values():
        a = np.asarray(value)
        if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
            raise ValueError('Invalid episode accuracy')
    groups = {(c['dataset'], c['shot'], c['side'], c['operator'], c['severity']) for c in cells}
    for d, k, side, op, s in groups:
        required = {('test', d, f, k, 7, m, 'best', side, op, s)
                    for f in range(5) for m in METHODS}
        if not required <= set(matrix):
            raise IncompleteResults('incomplete: fixed five folds/four methods required')
        if len({np.asarray(matrix[key]).shape for key in required}) != 1:
            raise ValueError('Unequal paired episode shapes')
    legacy = {(c['dataset'], c['shot'], c['method'], c['checkpoint'], c['side'],
               c['operator'], c['severity'], c['seed'], c['fold']): matrix[cell_key(c)]
              for c in cells}
    tables = dict(cells=[], contrasts=[])

    def add(context, arrays):
        for method, values in arrays.items():
            tables['cells'].append(dict(**context, method=method,
                                       accuracy_pct=float(100 * values.mean(-1).mean(-1).mean())))
        gd, gp = arrays['D2'] - arrays['D0'], arrays['E2'] - arrays['P0']
        for comparison, delta in (('D2-D0', gd), ('E2-P0', gp),
                                  ('(E2-P0)-(D2-D0)', gp - gd)):
            tables['contrasts'].append(dict(**context, comparison=comparison,
                delta_pp=float(100 * delta.mean(-1).mean(-1).mean()),
                uncertainty='descriptive_fixed_seed7_no_ci'))

    groups = list(dict.fromkeys((c['dataset'], c['shot'], c['side'], c['operator'],
                                c['severity']) for c in cells))
    for dataset, shot, side, operator, severity in groups:
        context = dict(pool='test', dataset=dataset, shot=shot, seed=7, checkpoint='best',
                       setting='clean' if side == 'clean' else 'severity', side=side,
                       operator=operator, severity=severity, folds=list(range(5)))
        arrays = {m: cell_array(legacy, dataset, shot, m, 'best',
                               (side, operator, severity), [7]) for m in METHODS}
        add(context, arrays)
    for dataset, shot in dict.fromkeys((c['dataset'], c['shot']) for c in cells):
        conditions = {(c['side'], c['operator'], c['severity']) for c in cells
                      if c['dataset'] == dataset and c['shot'] == shot}
        query = [('query', op, s) for op in OPERATORS for s in range(1, 6)]
        summaries = [('robust_query', 'seven_operator_mean', 'all', query)]
        summaries += [('operator', op, 'all', [('query', op, s) for s in range(1, 6)])
                      for op in OPERATORS]
        summaries += [('query_severity3' if s == 3 else 'severity_mean',
                       'seven_operator_mean', s, [('query', op, s) for op in OPERATORS])
                      for s in range(1, 6)]
        for setting, operator, severity, selected in summaries:
            if not set(selected) <= conditions:
                continue  # only explicit miniature fixtures omit full conditions
            arrays = {m: np.stack([cell_array(legacy, dataset, shot, m, 'best', c, [7])
                                   for c in selected]).mean(0) for m in METHODS}
            add(dict(pool='test', dataset=dataset, shot=shot, seed=7, checkpoint='best',
                     setting=setting, side='query', operator=operator, severity=severity,
                     folds=list(range(5))), arrays)
    return tables


def write_report(new_out, reference_out, out, *, cells=None):
    """Write four reporting products, never the prediction/episode source trees.

    Passing explicit cells labels all products as artificial testing, never complete.
    The public CLI has no way to shrink the frozen matrix.
    """
    import csv

    new, reference, target = map(lambda p: Path(p).resolve(), (new_out, reference_out, out))
    if (target.is_relative_to(reference) or reference.is_relative_to(target) or
            new == reference or target == new or new.is_relative_to(target) or
            target.is_relative_to(new / 'predictions') or target.is_relative_to(new / 'episodes')):
        raise ValueError('Report output must not overlap read-only source artifacts')
    fixture = cells is not None
    selected = _cells(cells)
    failure = None
    try:
        matrix, coverage = load_predictions(new, reference, cells=selected, partial=True)
        coverage['conflicts'] = []
        tables = build_tables(matrix, cells=selected) if coverage['complete'] else dict(cells=[], contrasts=[])
    except (ValueError, OSError, KeyError, EOFError) as exc:
        # Stop on conflicts; unscanned sources are unknown, NOT counted as missing/present.
        failure = exc
        coverage = dict(expected=sum(c['method'] in ('D0', 'D2') for c in selected),
            present=None, missing_keys=[],
            reference=dict(expected=sum(c['method'] in ('P0', 'E2') for c in selected),
                           present=None, missing_keys=[]),
            complete=False, conflicts=[str(exc)], scan_complete=False,
            key_fields=list(KEY_FIELDS), unchecked_keys=[list(cell_key(c)) for c in selected])
        tables = dict(cells=[], contrasts=[])
    subset_complete = coverage['complete']
    coverage.update(complete=subset_complete and not fixture, fixture=fixture,
                    subset_complete=subset_complete, training_seed=7,
                    new_out=str(new), reference_out=str(reference))
    target.mkdir(parents=True, exist_ok=True)
    # Invalidate any old success before writing replacement tables.
    (target / 'REPORT.md').write_text('# review1 — INCOMPLETE\n', encoding='utf-8')
    for name, rows in tables.items():
        fields = list(rows[0]) if rows else ['dataset', 'shot', 'seed',
            'method' if name == 'cells' else 'comparison',
            'accuracy_pct' if name == 'cells' else 'delta_pp']
        with (target / f'{name}.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, list) else v
                              for k, v in row.items()} for row in rows)
    label = ('ARTIFICIAL TEST FIXTURE — not experimental results' if fixture else
             'COMPLETE' if coverage['complete'] else 'INCOMPLETE — no final comparison')
    lines = ['# review1: fixed seed7 supplementary comparison', '', label, '',
        f"New D predictions: {coverage['present']}/{coverage['expected']}; "
        f"reference P0/E2: {coverage['reference']['present']}/{coverage['reference']['expected']}.", '',
        '- D0/D2: independently trained no-PCIA Euclidean DGCNN, same best checkpoint.',
        '- P0/E2: read-only seed7 reference; no mixing with seeds17/27.',
        '- Recompute query accuracy, then episode means, then five fixed folds equally.',
        '- Query summaries weight five severities then seven operators equally; clean is separate.',
        '- Only D2-D0, E2-P0, and (E2-P0)-(D2-D0); negative results retained.',
        '- No CI is computed or combined from existing intervals. Descriptive fixed-seed comparison only.',
        '- No cross-training-seed stability, strict synergy, or single-module causal claim.',
        '- Test data were inspected only after all experiments had been completed. Engineering completeness is not scientific improvement.', '',
        'See cells.csv (clean, raw operator/severity, operator and severity means) and contrasts.csv.', '']
    if tables['contrasts']:
        lines += ['| dataset | shot | setting | comparison | delta_pp |', '|---|---|---|---|---|']
        for row in tables['contrasts']:
            if row['setting'] in ('clean', 'robust_query'):
                lines.append(f"| {row['dataset']} | {row['shot']} | {row['setting']} | "
                             f"{row['comparison']} | {row['delta_pp']:.3f} |")
    (target / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (target / 'coverage.json').write_text(json.dumps(coverage, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if not subset_complete:
        raise IncompleteResults(f'incomplete: see {target / "coverage.json"}') from failure
    return coverage


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('new-out', 'reference-out', 'out'):
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args(argv)
    try:
        result = write_report(args.new_out, args.reference_out, args.out)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps({k: result[k] for k in ('complete', 'expected', 'present')}))


if __name__ == '__main__':
    main()
