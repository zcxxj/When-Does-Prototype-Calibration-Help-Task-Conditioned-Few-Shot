"""Validate the fixed efficiency benchmark from its raw measurements."""


def load_rows(data):
    """Validate scope and recompute summaries from all 960 raw measurements."""
    import math
    from statistics import median
    settings = {(s['shot'], s['side'], s['operator'], s['severity']) for s in data['settings']}
    expected_settings = {(s, side, op, v) for s in (1, 5)
                         for side, op, v in (('clean', None, 0), ('query', 'Add-Global', 5))}
    if (not data['complete'] or data['methods'] != ['P0', 'E2']
            or len(data['settings']) != 4 or settings != expected_settings
            or any((s['dataset'], s['fold'], s['seed'], s['pool']) !=
                   ('ModelNet40', 0, 7, 'dev') for s in data['settings'])):
        raise ValueError('Unexpected fixed dev benchmark scope')
    rows = data['rows']
    expected = {setting + (method, mode) for setting in settings
                for method in ('P0', 'E2') for mode in ('end_to_end', 'backend_only')}
    actual = [tuple(r[k] for k in ('shot', 'side', 'operator', 'severity', 'method', 'mode'))
              for r in rows]
    if len(actual) != 16 or set(actual) != expected:
        raise ValueError('Incomplete or duplicated benchmark rows')
    for r in rows:
        raw = r['measurements']
        if (tuple(r[k] for k in ('episodes', 'warmup', 'repeats', 'samples')) != (20, 5, 3, 60)
                or len(raw) != 60 or {(x['repeat'], x['episode_id']) for x in raw}
                != {(i, j) for i in range(3) for j in range(20)}):
            raise ValueError('Incomplete raw measurements')
        values = sorted(x['seconds'] for x in raw)
        index = (len(values) - 1) * 0.95
        lower = int(index)
        p95 = values[lower] + (index - lower) * (values[lower + 1] - values[lower])
        for key, value in [('median_seconds', median(values)), ('p95_seconds', p95),
                           ('peak_allocated_bytes', max(x['allocated_bytes'] for x in raw)),
                           ('peak_reserved_bytes', max(x['reserved_bytes'] for x in raw))]:
            if not math.isclose(r[key], value, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(f'Raw measurements disagree with {key}')
    return rows
