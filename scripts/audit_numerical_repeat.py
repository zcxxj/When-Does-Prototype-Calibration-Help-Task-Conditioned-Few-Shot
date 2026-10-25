#!/usr/bin/env python3
"""Compare overlapping fresh and earlier balanced predictions without inference."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sac.empirical_protocol import prediction_cells
from sac.report_calibration_study import _path


def audit(study, legacy, output):
    rows = []
    for cell in prediction_cells():
        if cell['protocol'] != 'balanced' or cell['method'] not in ('P0', 'E2') or cell['reused']:
            continue
        current = _path(study, cell)
        prior = legacy / 'sac-review-v1' / current.relative_to(study / 'protocols/balanced')
        if not prior.is_file():
            continue  # Historical 1-shot support/joint cells were not evaluated.
        with np.load(current) as a, np.load(prior) as b:
            for key in ('local_target', 'episode_rows', 'corruption_seeds'):
                if not np.array_equal(a[key], b[key]):
                    raise ValueError(f'Repeat audit input mismatch: {current.name}: {key}')
            x, y, target = a['pred'], b['pred'], a['local_target']
            changed = int(np.count_nonzero(x != y))
            delta = float(100 * ((x == target).mean() - (y == target).mean()))
        new_meta = json.loads(current.with_suffix('.json').read_text())
        old_meta = json.loads(prior.with_suffix('.json').read_text())
        rows.append({**{k: cell[k] for k in ('dataset', 'shot', 'seed', 'fold', 'method', 'side', 'operator', 'severity')},
                     'query_predictions': int(x.size), 'changed_predictions': changed,
                     'accuracy_difference_pp': delta, 'new_gpu': new_meta.get('gpu'),
                     'prior_gpu': old_meta.get('gpu')})
    if len(rows) != 2480:
        raise ValueError(f'Expected 2480 overlapping fresh cells, found {len(rows)}')
    per_seed = []
    means = []
    for dataset, shot, method, side in itertools.product(('ModelNet40', 'ShapeNet70'), (1, 5), ('P0', 'E2'), ('clean', 'query')):
        values = []
        for seed in (7, 17, 27):
            subset = [r for r in rows if (r['dataset'], r['shot'], r['method'], r['side'], r['seed']) == (dataset, shot, method, side, seed)]
            reused = shot == 5 and seed == 7
            expected = 0 if reused else 5 if side == 'clean' else 175 if shot == 1 else 35
            if len(subset) != expected:
                raise ValueError('Incomplete matched repeat scope')
            delta = 0.0 if reused else float(np.mean([r['accuracy_difference_pp'] for r in subset]))
            values.append(delta)
            per_seed.append(dict(dataset=dataset, shot=shot, method=method, side=side,
                                 seed=seed, reused_identical_record=reused, difference_pp=delta))
        means.append(float(np.mean(values)))
    devices = {}
    for row in rows:
        key = (row['new_gpu'], row['prior_gpu'])
        device = devices.setdefault(key, dict(new_gpu=key[0], prior_gpu=key[1], compared_cells=0,
                                              query_predictions=0, changed_predictions=0))
        device['compared_cells'] += 1
        device['query_predictions'] += row['query_predictions']
        device['changed_predictions'] += row['changed_predictions']
    summary = dict(complete=True, compared_cells=len(rows), input_pairing_equal=True,
        query_predictions=sum(r['query_predictions'] for r in rows),
        changed_query_predictions=sum(r['changed_predictions'] for r in rows),
        changed_cells=sum(bool(r['changed_predictions']) for r in rows),
        max_abs_cell_accuracy_difference_pp=max(abs(r['accuracy_difference_pp']) for r in rows),
        max_abs_matched_scope_seed_difference_pp=max(abs(r['difference_pp']) for r in per_seed),
        max_abs_matched_scope_three_seed_mean_difference_pp=max(map(abs, means)),
        matched_scope='clean and query means; query averages all severities for 1-shot, severity 3 for 5-shot',
        reused_scope='5-shot seed 7 uses the same historical records and therefore has zero repeat difference',
        devices=list(devices.values()), per_seed_scope_differences=per_seed)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'cells.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT / 'attachments/study')
    parser.add_argument('--legacy', type=Path, default=ROOT / 'attachments/legacy')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/numerical-repeat')
    args = parser.parse_args()
    summary = audit(args.study.resolve(), args.legacy.resolve(), args.output.resolve())
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_seed_scope_differences'}, indent=2))


if __name__ == '__main__':
    main()
