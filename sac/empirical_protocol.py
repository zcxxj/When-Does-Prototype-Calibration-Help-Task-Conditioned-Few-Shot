"""Frozen calibration study: two query protocols, no training or parameter search."""
from itertools import product
import hashlib
import json
from pathlib import Path

VERSION = 'empirical-calibration-v1'
PROTOCOLS = ('balanced', 'imbalanced')
DATASETS = ('ModelNet40', 'ShapeNet70')
SEEDS = (7, 17, 27)
OPERATORS = ('Scale', 'Jitter', 'Drop-Global', 'Drop-Local', 'Add-Global', 'Add-Local', 'Rotate')
KEYS = ('protocol', 'pool', 'dataset', 'fold', 'shot', 'seed', 'method', 'checkpoint', 'side', 'operator', 'severity')


def cell_key(cell):
    return tuple(cell[k] for k in KEYS)


def methods(shot):
    return ('P0', 'E2', 'QONLY', 'SOFTK') if shot == 1 else ('P0', 'E2', 'UNIT', 'QONLY', 'SOFTK')


def conditions(protocol, shot):
    values = [('clean', None, 0)]
    severities = range(1, 6) if protocol == 'balanced' and shot == 1 else (3,)
    values += [('query', op, sev) for op, sev in product(OPERATORS, severities)]
    if protocol == 'balanced':
        values += [(side, op, 3) for side, op in product(('support', 'both'), ('Add-Global', 'Drop-Local'))]
    return values


def prediction_cells():
    cells = []
    for protocol, dataset, fold, shot, seed in product(PROTOCOLS, DATASETS, range(5), (1, 5), SEEDS):
        for method, (side, operator, severity) in product(methods(shot), conditions(protocol, shot)):
            cells.append(dict(protocol=protocol, pool='test', dataset=dataset, fold=fold,
                              shot=shot, seed=seed, method=method, checkpoint='best',
                              side=side, operator=operator, severity=severity,
                              reused=protocol == 'balanced' and shot == 5 and seed == 7))
    return cells


def evaluation_jobs(include_reused=False):
    grouped = {}
    for cell in prediction_cells():
        if cell['reused'] and not include_reused:
            continue
        key = tuple(cell[k] for k in KEYS if k != 'method')
        if key not in grouped:
            grouped[key] = {k: v for k, v in cell.items() if k not in ('method', 'reused')}
            grouped[key].update(family='P0', methods=[])
        grouped[key]['methods'].append(cell['method'])
    return list(grouped.values())


def job_id(job):
    fields = ('protocol', 'dataset', 'fold', 'shot', 'seed', 'side', 'operator', 'severity')
    return '_'.join(str(job[k]) for k in fields)


def freeze_document():
    cells = prediction_cells()
    encoded = json.dumps(cells, sort_keys=True, separators=(',', ':')).encode()
    return dict(version=VERSION, protocols=list(PROTOCOLS), episodes=700,
                training_seeds=list(SEEDS), datasets=list(DATASETS), folds=list(range(5)),
                query_total=75, imbalance_quotas=[5, 10, 15, 20, 25], shared_queries=60,
                no_training=True, no_parameter_search=True,
                primary_metric={'balanced': 'micro', 'imbalanced': 'macro'},
                matrix_sha256=hashlib.sha256(encoded).hexdigest(),
                prediction_cells=len(cells), reused_cells=sum(c['reused'] for c in cells),
                frontend_jobs=len(evaluation_jobs()), frontend_episodes=len(evaluation_jobs()) * 700,
                cells=cells)


def write_freeze(out):
    path = Path(out) / 'freeze.json'
    value = freeze_document()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Existing freeze differs; refusing mutation')
    else:
        with path.open('x') as handle:
            json.dump(value, handle, indent=2)
            handle.write('\n')
    return path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = write_freeze(args.out)
    print(json.dumps({k:v for k,v in freeze_document().items() if k != 'cells'}, indent=2))
    print(path)
