"""Finite SAC review matrix; no data reads or implicit experiment launches."""
from itertools import product

OPERATORS = ('Scale', 'Jitter', 'Drop-Global', 'Drop-Local', 'Add-Global', 'Add-Local', 'Rotate')
SYNTHETIC = ('ModelNet40', 'ShapeNet70')
KEYS = ('pool', 'dataset', 'fold', 'shot', 'seed', 'method', 'checkpoint', 'side', 'operator', 'severity')


def cell_key(cell):
    return tuple(cell[k] for k in KEYS)


def training_jobs():
    return [dict(dataset=d, fold=f, shot=k, seed=s, family='P0')
            for d, f, k, s in product(SYNTHETIC, range(5), (1, 5), (17, 27))]


def representative_conditions():
    return [('clean', None, 0)] + [('query', o, 3) for o in OPERATORS] + [
        (side, o, 3) for side, o in product(('support', 'both'), ('Add-Global', 'Drop-Local'))]


def prediction_cells(pool='test', include_reused=False, expanded_dev=False):
    if expanded_dev and pool != 'dev':
        raise ValueError('expanded_dev is dev only')
    if pool not in ('dev', 'test'):
        raise ValueError('pool must be dev or test')
    cells = {}

    def add(d, f, k, s, m, checkpoint, condition, reused=False):
        side, operator, severity = condition
        cell = dict(pool=pool, dataset=d, fold=f, shot=k, seed=s, method=m,
                    checkpoint=checkpoint, side=side, operator=operator, severity=severity,
                    reused=reused)
        cells[cell_key(cell)] = cell

    if pool == 'test':
        main = [('clean', None, 0)] + [('query', o, v) for o, v in product(OPERATORS, range(1, 6))]
        for d, f, k, s, m, c in product(SYNTHETIC, range(5), (1, 5), (7, 17, 27), ('P0', 'E2'), main):
            add(d, f, k, s, m, 'best', c)
        for d, f, k, m, c in product(SYNTHETIC, range(5), (1, 5), ('H', 'F1', 'F2'), main):
            add(d, f, k, 7, m, 'best', c)
        for f, k, m in product(range(3), (1, 5), ('P0', 'E2', 'H')):
            add('ScanObjectNN', f, k, 7, m, 'best', ('clean', None, 0))
        for cell in list(cells.values()):
            if cell['side'] == 'clean':
                copy = dict(cell, checkpoint='last')
                cells[cell_key(copy)] = copy
    seeds = (7, 17, 27) if expanded_dev else (7,)
    for d, f, s, m, c in product(SYNTHETIC, range(5), seeds, ('P0', 'E2', 'QONLY', 'UNIT', 'SOFTK'), representative_conditions()):
        reused = pool == 'dev' and s == 7 and m in ('P0', 'E2')
        if not reused or include_reused:
            add(d, f, 5, s, m, 'best', c, reused)
    return list(cells.values())


def softk_dev_cells():
    """Additional development analysis, never an expansion of novel permission."""
    main = [('clean', None, 0)] + [('query', op, severity)
            for op, severity in product(OPERATORS, range(1, 6))]
    aux = [c for c in representative_conditions() if c[0] in ('support', 'both')]
    cells = []
    for shot in (5, 1):
        methods = ('P0', 'E2', 'QONLY', 'UNIT', 'SOFTK') if shot == 5 else ('P0', 'E2', 'SOFTK')
        conditions = main + aux if shot == 5 else main
        for dataset, fold, seed, method, condition in product(
                SYNTHETIC, range(5), (7, 17, 27), methods, conditions):
            side, operator, severity = condition
            cells.append(dict(pool='dev', dataset=dataset, fold=fold, shot=shot,
                              seed=seed, method=method, checkpoint='best', side=side,
                              operator=operator, severity=severity, reused=False))
    return cells


def evaluation_jobs(pool, expanded_dev=False):
    grouped = {}
    for cell in prediction_cells(pool, expanded_dev=expanded_dev):
        family = 'H' if cell['method'] == 'H' else 'P0'
        key = tuple(cell[k] for k in KEYS if k != 'method') + (family,)
        if key not in grouped:
            grouped[key] = {k: v for k, v in cell.items() if k not in ('method', 'reused')}
            grouped[key].update(family=family, methods=[])
        grouped[key]['methods'].append(cell['method'])
    return list(grouped.values())
