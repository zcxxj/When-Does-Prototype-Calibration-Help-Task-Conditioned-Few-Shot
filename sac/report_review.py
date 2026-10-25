"""Read-only SAC review reports; never trains, evaluates or selects a method."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate import OPERATORS, prediction_path
from .summarize import IncompleteResults, paired_aggregation

KEY_FIELDS = ('pool', 'dataset', 'fold', 'shot', 'seed', 'method',
              'checkpoint', 'side', 'operator', 'severity')


def cell_key(cell):
    return tuple(cell[field] for field in KEY_FIELDS)


def expected_cells(pool, expanded_dev=False, softk_dev=False):
    # Lazy import: report helpers can be tested without running the protocol CLI.
    from .review_protocol import prediction_cells
    if softk_dev:
        if pool != 'dev':
            raise ValueError('softk_dev is restricted to the dev pool')
        from .review_protocol import softk_dev_cells
        return softk_dev_cells()
    if expanded_dev and pool != 'dev':
        raise ValueError('expanded_dev is restricted to the dev pool')
    return prediction_cells(pool=pool, include_reused=True, expanded_dev=expanded_dev)


def source_path(out, source_out, cell):
    root = source_out if cell['reused'] else out
    return prediction_path(root, cell, cell['pool'], cell['method'], cell['checkpoint'],
                           (cell['side'], cell['operator'], cell['severity']))


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _index_entries(reuse_index, cells):
    if reuse_index is None:
        return None
    data = json.loads(Path(reuse_index).read_text())
    entries = data['cells']
    indexed = {cell_key(entry['cell']): entry for entry in entries}
    if (data.get('version') != 1 or len(indexed) != len(entries) or
            set(indexed) != {cell_key(c) for c in cells}):
        raise ValueError('Reuse index must contain exactly the requested unique protocol keys')
    for entry in entries:
        if entry['source'] not in ('full-v1', 'sac-review-v1', 'softk-dev-v1'):
            raise ValueError('Unknown reuse index source')
        if any(not Path(entry[name]).is_absolute() for name in ('prediction', 'metadata')):
            raise ValueError('Reuse index paths must be absolute')
    return indexed


def load_review_predictions(out, source_out, pool, *, cells=None, partial=False, reuse_index=None):
    """Load only the explicit protocol keys, with old dev references read-only."""
    cells = list(expected_cells(pool) if cells is None else cells)
    keys = [cell_key(c) for c in cells]
    if pool not in ('dev', 'test') or any(c['pool'] != pool for c in cells):
        raise ValueError('Invalid/mixed report pool')
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate protocol keys')
    indexed = _index_entries(reuse_index, cells)
    matrix, missing, sources = {}, [], []
    identities, checkpoints, targets, damage = {}, {}, {}, {}

    def paired(cache, key, value, label, path):
        if key in cache and cache[key] != value:
            raise ValueError(f'Unpaired {label}: {path}')
        cache[key] = value

    for cell in cells:
        key = cell_key(cell)
        entry = indexed[key] if indexed is not None else None
        path = Path(entry['prediction']) if entry is not None else source_path(out, source_out, cell)
        metadata = Path(entry['metadata']) if entry is not None else path.with_suffix('.json')
        source = dict(key=list(key), reused=entry['source'] != 'softk-dev-v1' if entry else cell['reused'],
                      path=str(path), metadata=str(metadata))
        if entry:
            source['source'] = entry['source']
        sources.append(source)
        if not path.is_file() or not metadata.is_file():
            missing.append(list(key))
            continue
        meta = json.loads(metadata.read_text())
        identity = {k: cell[k] for k in ('dataset', 'fold', 'shot', 'seed', 'method', 'checkpoint', 'pool')}
        identity.update(family='H' if cell['method'] == 'H' else 'P0',
                        condition=[cell['side'], cell['operator'], cell['severity']])
        if any(meta.get(k) != v for k, v in identity.items()):
            raise ValueError(f'Prediction metadata identity mismatch: {path}')
        if meta.get('smoke') is not False or meta.get('episodes') != 700:
            raise IncompleteResults(f'incomplete: smoke/nonstandard count: {path}')
        if meta.get('checkpoint_epoch') is None or not meta.get('manifest'):
            raise ValueError(f'Prediction metadata missing checkpoint/manifest: {path}')
        fold = 0 if pool == 'test' and not cell['dataset'].startswith('Scan') else cell['fold']
        pair = (cell['dataset'], fold, cell['shot'])
        paired(identities, pair, meta['manifest'], 'metadata manifest', path)
        checkpoint_pair = (cell['dataset'], cell['fold'], cell['shot'], cell['seed'],
                           identity['family'], cell['checkpoint'])
        paired(checkpoints, checkpoint_pair, meta['checkpoint_epoch'], 'metadata checkpoint', path)
        with np.load(path, allow_pickle=False) as values:
            target, pred = values['local_target'], values['pred']
            if target.shape != (700, 75) or pred.shape != target.shape:
                raise IncompleteResults(f'incomplete: prediction shape: {path}')
            if not np.isin(target, range(5)).all() or not np.isin(pred, range(5)).all():
                raise ValueError(f'Invalid local labels: {path}')
            accuracy = (pred == target).mean(-1)
            if (not np.array_equal(values['episode_id'], np.arange(700)) or
                    values['episode_accuracy'].shape != (700,) or
                    not np.allclose(accuracy, values['episode_accuracy'], atol=1e-8, rtol=1e-6)):
                raise ValueError(f'Prediction IDs/accuracy inconsistent: {path}')
            paired(targets, pair, _digest(target), 'targets', path)
            for name in ('episode_rows', 'support_rows', 'query_rows', 'class_ids'):
                if name in values:
                    paired(targets, (*pair, name), _digest(values[name]), name, path)
            if 'corruption_seeds' in values:
                paired(damage, (*pair, cell['side'], cell['operator'], cell['severity']),
                       _digest(values['corruption_seeds']), 'corruption seeds', path)
            matrix[key] = accuracy
    coverage = dict(pool=pool, complete=not missing, expected=len(keys), present=len(matrix),
                    reused_present=sum(s['reused'] and tuple(s['key']) in matrix for s in sources),
                    missing_keys=missing, key_fields=list(KEY_FIELDS), sources=sources)
    if missing and not partial:
        raise IncompleteResults(f'incomplete: missing {len(missing)} prediction cells; first: {missing[0]}')
    return matrix, coverage


MAIN_COMPARISONS = (('E2', 'P0'), ('E2', 'H'), ('F1', 'P0'), ('F2', 'E2'))
ABLATION_COMPARISONS = (('E2', 'UNIT'), ('UNIT', 'QONLY'), ('E2', 'QONLY'),
                        ('E2', 'SOFTK'), ('QONLY', 'P0'), ('UNIT', 'P0'), ('SOFTK', 'P0'))
CONTROL_METHODS = ('P0', 'E2', 'QONLY', 'UNIT', 'SOFTK')


def build_tables(matrix, pool, *, cells=None, partial=False, softk_dev=False):
    """All correlated condition axes are averaged before single-seed bootstrap.

    Explicit expected cells define fold coverage, even for partial reports.
    A partial aggregate is omitted rather than averaging its surviving inputs.
    Tiny in-memory fixtures may supply their own explicit cells for tests only.
    """
    if softk_dev and pool != 'dev':
        raise ValueError('softk_dev is restricted to the dev pool')
    cells = list(expected_cells(pool, softk_dev=True) if cells is None and softk_dev else
                 expected_cells(pool) if cells is None else cells)
    expected = {cell_key(c) for c in cells}
    if set(matrix) - expected:
        raise ValueError('Unexpected prediction keys')
    if expected - set(matrix) and not partial:
        raise IncompleteResults('incomplete: missing prediction cells')
    tables = {name: [] for name in ('cells', 'protocol_main', 'protocol_clean_best_last',
              'robust_query', 'ablation', 'paired_effects', 'per_seed', 'per_fold', 'seed_summary')}
    groups = sorted({(c['dataset'], c['shot'], c['seed']) for c in cells})
    summary_values = []

    for dataset, shot, seed in groups:
        group_cells = [c for c in cells if (c['dataset'], c['shot'], c['seed']) == (dataset, shot, seed)]
        methods = sorted({c['method'] for c in group_cells})
        folds = sorted({c['fold'] for c in group_cells})
        context = dict(pool=pool, dataset=dataset, shot=shot, seed=seed, checkpoint='best')
        control_run = shot == 5 and (seed == 7 or pool == 'dev' and seed in (17, 27))
        shared = pool == 'test' and not dataset.startswith('Scan')
        arrays = {}
        for c in group_cells:
            condition = (c['side'], c['operator'], c['severity'])
            name = (c['method'], c['checkpoint'], condition)
            if name in arrays:
                continue
            keys = [(pool, dataset, f, shot, seed, c['method'], c['checkpoint'], *condition) for f in folds]
            if all(k in matrix for k in keys):
                values = [np.asarray(matrix[k]) for k in keys]
                if len({v.shape for v in values}) != 1 or any(v.ndim != 1 or not len(v) for v in values):
                    raise ValueError('Unequal/empty episode arrays')
                array = np.stack(values)
                if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
                    raise ValueError('Invalid accuracy array')
                arrays[name] = array
        clean = {m: arrays[(m, 'best', ('clean', None, 0))] for m in methods
                 if (m, 'best', ('clean', None, 0)) in arrays}

        def values_for(conditions, eligible=None):
            result = {}
            for method in (methods if eligible is None else eligible):
                keys = [(method, 'best', cond) for cond in conditions]
                if all(k in arrays for k in keys):
                    result[method] = np.stack([arrays[k] for k in keys]).mean(0)
            return result

        def effect(setting_context, candidate, baseline, comparison):
            delta = candidate - baseline
            estimate = paired_aggregation(candidate[None].tolist(), baseline[None].tolist(),
                                          shared_folds=shared, resamples=2000)
            row = dict(**setting_context, comparison=comparison, **estimate,
                       uncertainty='conditional_episode_only')
            tables['paired_effects'].append(row)
            tables['per_seed'].append(row)
            summary_values.append((setting_context, dict(comparison=comparison), 'pp', estimate['delta_pp']))
            for fold, value in zip(folds, delta.mean(-1)):
                tables['per_fold'].append(dict(**setting_context, fold=fold, comparison=comparison,
                                               delta_pp=float(100 * value)))

        def add(setting, side, operator, severity, values, ablation=False):
            scope = dict(**context, setting=setting, side=side, operator=operator, severity=severity)
            for method, array in values.items():
                row = dict(**scope, method=method, accuracy_pct=float(100 * array.mean()),
                           drop_from_clean_pp=float(100 * (clean[method] - array).mean())
                           if method in clean else None, folds=folds)
                tables['cells'].append(row)
                tables['per_seed'].append(row)
                if setting == 'clean' and method not in ('QONLY', 'UNIT', 'SOFTK'):
                    tables['protocol_main'].append(row)
                if side == 'query':
                    tables['robust_query'].append(row)
                if ablation and method in CONTROL_METHODS:
                    tables['ablation'].append(row)
                summary_values.append((scope, dict(method=method), 'percent', row['accuracy_pct']))
                for fold, value in zip(folds, array.mean(-1)):
                    tables['per_fold'].append(dict(**scope, method=method, fold=fold,
                        accuracy_pct=float(100 * value)))
            comparisons = ([('SOFTK', 'P0'), ('SOFTK', 'E2'), ('E2', 'P0')] +
                           ([('SOFTK', 'UNIT'), ('SOFTK', 'QONLY')] if shot == 5 else [])
                           if softk_dev else list(MAIN_COMPARISONS))
            if ablation and not softk_dev:
                comparisons += list(ABLATION_COMPARISONS)
            for candidate, baseline in comparisons:
                if candidate in values and baseline in values:
                    if not softk_dev and seed != 7 and not (pool == 'dev' and ablation) and (candidate, baseline) != ('E2', 'P0'):
                        continue
                    effect(scope, values[candidate], values[baseline], f'{candidate}-{baseline}')

        add('clean', 'clean', None, 0, clean, ablation=control_run)
        query_conditions = [('query', op, s) for op in OPERATORS for s in range(1, 6)]
        robust = values_for(query_conditions)
        if robust:
            add('robust_query', 'query', 'seven_operator_mean', 'all', robust)
            drop_pairs = [('E2', 'P0')] + ([('SOFTK', 'P0'), ('SOFTK', 'E2')] if softk_dev else [])
            for candidate, baseline in drop_pairs:
                if all(m in clean and m in robust for m in (candidate, baseline)):
                    effect(dict(**context, setting='drop_differential', side='query',
                                operator='seven_operator_mean', severity='all'),
                           clean[candidate] - robust[candidate], clean[baseline] - robust[baseline],
                           f'{candidate}-{baseline}')
        for operator in OPERATORS:
            values = values_for([('query', operator, s) for s in range(1, 6)])
            if values:
                add('operator', 'query', operator, 'all', values)
        for severity in range(1, 6):
            values = values_for([('query', op, severity) for op in OPERATORS])
            if values:
                add('query_severity3' if severity == 3 else 'severity_mean', 'query',
                    'seven_operator_mean', severity, values, ablation=severity == 3 and control_run)
        conditions = sorted({(c['side'], c['operator'], c['severity']) for c in group_cells
                             if c['checkpoint'] == 'best' and c['side'] != 'clean'})
        for condition in conditions:
            side, operator, severity = condition
            representative = control_run and severity == 3
            add('severity' if side == 'query' else 'representative_condition',
                side, operator, severity, values_for([condition]), ablation=representative)
        if control_run:
            for side in ('support', 'both'):
                values = values_for([(side, op, 3) for op in ('Add-Global', 'Drop-Local')], CONTROL_METHODS)
                if values:
                    add('representative_aux', side, 'two_operator_mean', 3, values, ablation=True)
        for method in methods:
            last = arrays.get((method, 'last', ('clean', None, 0)))
            if method not in clean or last is None:
                continue
            best_mean, last_mean = float(100 * clean[method].mean()), float(100 * last.mean())
            maximum = max(best_mean, last_mean)
            row = dict(pool=pool, dataset=dataset, shot=shot, seed=seed, method=method,
                       best_accuracy_pct=best_mean, last_accuracy_pct=last_mean,
                       paper_style_max_pct=maximum,
                       note='descriptive: max after fold averaging, before seed averaging')
            tables['protocol_clean_best_last'].append(row)
            summary_values.append((dict(**context, setting='clean_best_last_max', side='clean',
                                       operator=None, severity=0), dict(method=method), 'percent', maximum))

    grouped = {}
    for context, identity, unit, value in summary_values:
        context = {k: v for k, v in context.items() if k != 'seed'}
        key = tuple(context.items()) + tuple(identity.items())
        grouped.setdefault(key, dict(**context, **identity, unit=unit, values=[]))['values'].append(value)
    # Rebuild seed labels from the same ordered per-run records, never bootstrap them.
    seed_labels = {}
    for context, identity, _, _ in summary_values:
        key = tuple((k, v) for k, v in context.items() if k != 'seed') + tuple(identity.items())
        seed_labels.setdefault(key, []).append(context['seed'])
    for key, row in grouped.items():
        values = row.pop('values')
        row.update(train_seeds=seed_labels[key], per_seed_values=values, mean=float(np.mean(values)),
                   sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                   n_train_seeds=len(values), uncertainty='between_training_seed_sample_sd',
                   partial=partial)
        tables['seed_summary'].append(row)
    return tables


def representative_cell(cell):
    return ((cell['seed'] == 7 or cell['pool'] == 'dev' and cell['seed'] in (17, 27))
            and cell['shot'] == 5 and cell['checkpoint'] == 'best'
            and cell['method'] in CONTROL_METHODS and not cell['dataset'].startswith('Scan')
            and (cell['side'] == 'clean' or cell['severity'] == 3 and
                 (cell['side'] == 'query' or cell['side'] in ('support', 'both') and
                  cell['operator'] in ('Add-Global', 'Drop-Local'))))


def diagnostic_path(out, cell):
    path = prediction_path(out, cell, cell['pool'], cell['method'], 'best',
                           (cell['side'], cell['operator'], cell['severity']))
    return Path(out) / 'diagnostics' / path.relative_to(Path(out) / 'predictions')


def load_diagnostics(out, cells, *, partial=False, matrix=None, reuse_index=None):
    """Summarize all representative episode/class diagnostics, not selected wins."""
    selected = [c for c in cells if representative_cell(c)]
    indexed = _index_entries(reuse_index, cells)
    rows, missing = [], []
    for cell in selected:
        location = indexed[cell_key(cell)].get('diagnostic') if indexed is not None else diagnostic_path(out, cell)
        path = Path(location) if location else None
        if path is None or not path.is_file():
            missing.append(list(cell_key(cell)))
            continue
        with np.load(path, allow_pickle=False) as data:
            scalars = {k: data[k] for k in ('corrected_count', 'damaged_count', 'initial_accuracy')}
            groups = {k: data[k] for k in ('group_size', 'anchor_fraction', 'displacement')}
            groups['purity'] = data['group_purity' if 'group_purity' in data else 'purity']
            groups['class_accuracy'] = data['group_class_accuracy' if 'group_class_accuracy' in data else 'class_accuracy']
            if cell['method'] == 'SOFTK':
                groups['soft_mass'] = data['soft_mass']
            if (any(v.shape != (700,) or not np.isfinite(v).all() for v in scalars.values()) or
                    any(v.shape != (700, 5) for v in groups.values())):
                raise ValueError(f'Invalid diagnostic shape/scalars: {path}')
            sizes = groups['group_size']
            if (not np.isfinite(sizes).all() or (sizes < 0).any() or
                    not np.equal(sizes, sizes.astype(int)).all() or not np.all(sizes.sum(-1) == 75)):
                raise ValueError(f'Invalid diagnostic group sizes: {path}')
            for name in ('corrected_count', 'damaged_count'):
                value = scalars[name]
                if np.any((value < 0) | (value > 75)) or not np.equal(value, value.astype(int)).all():
                    raise ValueError(f'Invalid diagnostic count: {path}')
            if np.any((scalars['initial_accuracy'] < 0) | (scalars['initial_accuracy'] > 1)):
                raise ValueError(f'Invalid initial accuracy: {path}')
            for name, values in groups.items():
                observed = values[sizes > 0] if name in ('purity', 'class_accuracy') else values
                if not np.isfinite(observed).all():
                    raise ValueError(f'Invalid diagnostic {name}: {path}')
                if (observed < 0).any() or (name in ('purity', 'class_accuracy', 'anchor_fraction') and (observed > 1).any()):
                    raise ValueError(f'Invalid diagnostic {name} range: {path}')
            if matrix is not None and cell_key(cell) in matrix:
                after = scalars['initial_accuracy'] + (scalars['corrected_count'] - scalars['damaged_count']) / 75
                if not np.allclose(after, matrix[cell_key(cell)], atol=1e-7, rtol=1e-6):
                    raise ValueError(f'Diagnostic net gain/accuracy mismatch: {path}')
            corrected = float(scalars['corrected_count'].mean())
            damaged = float(scalars['damaged_count'].mean())
            common = dict(**{k: cell[k] for k in KEY_FIELDS}, episodes=700, query_denominator=75,
                          corrected_count=corrected, damaged_count=damaged,
                          corrected_rate=corrected / 75, damaged_rate=damaged / 75,
                          net_gain_pp=100 * (corrected - damaged) / 75,
                          initial_accuracy=float(scalars['initial_accuracy'].mean()),
                          source=str(path), aggregation='episode_mean_within_fold')
            for class_id in range(5):
                row = dict(**common, class_id=class_id,
                           empty_group_rate=float((sizes[:, class_id] == 0).mean()))
                for name, values in groups.items():
                    observed = values[:, class_id]
                    if name in ('purity', 'class_accuracy'):
                        observed = observed[sizes[:, class_id] > 0]
                    row[name] = float(observed.mean()) if len(observed) else None
                rows.append(row)
    coverage = dict(expected=len(selected), present=len(selected) - len(missing),
                    complete=not missing, missing_keys=missing)
    if missing and not partial:
        raise IncompleteResults(f'incomplete: missing {len(missing)} diagnostic cells; first: {missing[0]}')
    return rows, coverage


def _markdown(tables, status):
    lines = ['# SAC review — ' + ('开发消融' if status['pool'] == 'dev' else '最终 novel'), '',
             '**PARTIAL / incomplete：仅供进度查看，不是最终结果。**' if not status['complete']
             else '**完整矩阵报告；工程完整不代表科学结论成立。**', '',
             f"预测配置 {status['present']}/{status['expected']}；诊断 "
             f"{status['diagnostics']['present']}/{status['diagnostics']['expected']}。", '',
             '## 统计与范围', '',
             '- 主效果统一 best-validation。每 episode 先五强度、再七算子等权平均；fold 等权。',
             '- 2000 次 percentile paired bootstrap，seed400007，每次仅一个训练 seed。',
             '- synthetic novel shared_folds=True；dev / Scan shared_folds=False；固定 fold 不重采样。',
             '- 条件性 episode CI 不含训练变异，不代表对象总体区间。跨 seed 均值与样本 SD(ddof=1) 单列；三 seed 不支持精确总体推断。',
             '- clean-to-corruption drop = clean − corruption；drop differential = E2 drop − P0 drop，负值才是下降更小。',
             '- query severity3 七算子代表诊断不是七算子×五强度主鲁棒；support/both 只平均 Add-Global、Drop-Local 两算子。',
             '- seed7 E2−H 是系统比较，F1−P0 / F2−E2 为固定几何对照，不与三 seed E2 均值混减。',
             '- best/last max 在每 seed 的 fold 均值之后取最大，再跨 seed 平均；仅测试后描述，不作选模或 robust checkpoint 选择。',
             '- 纠错/改错分母为75；纯度与原型类别一致率分列，空组 NA；SOFTK soft_mass 不是硬人数。诊断相关性不证明因果或几何恢复。',
             '- P0=PCIA-E；E2=PCIA-SAC；F1=PCIA-H；F2=PCIA-HR；H=HPR-D。',
             '- HPR-D 固定实现曲率 c=0.01（两 shot）；源码相邻5-shot注释0.005不是执行值，不作曲率普遍结论。',
             '- Scan 使用官方 fold 测试池，不能据 fold 数声称对象或类别独立；旧 Scan dev 不纳入 novel。',
             '- 未自动训练/推理/选择方法，不写 selection.json；所有预设负结果保留。', '']

    def table(title, rows, columns):
        lines.extend([f'## {title}', ''])
        if not rows:
            lines.extend(['尚无该范围的完整配对数据。', ''])
            return
        lines.extend(['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)])
        for row in rows:
            def format_value(value):
                if value is None:
                    return 'NA'
                if isinstance(value, float):
                    return f'{value:.3f}'
                if isinstance(value, list):
                    return ', '.join(format_value(v) for v in value)
                return str(value)
            lines.append('| ' + ' | '.join(format_value(row.get(k)) for k in columns) + ' |')
        lines.append('')

    if status.get('softk_dev'):
        lines[:] = ['# Soft-K 全强度开发评测', '',
                    '**PARTIAL / incomplete：仅供进度查看，禁止最终结论。**' if not status['complete']
                    else '**完整开发矩阵；不自动选择方法，不启动 novel。**', '',
                    f"预测配置 {status['present']}/{status['expected']}；代表诊断 "
                    f"{status['diagnostics']['present']}/{status['diagnostics']['expected']}。", '',
                    '## 统计与范围', '',
                    '- 全强度 query 主均值：每 episode 先五 severity 平均，再七算子平均，再 fold 等权；clean 单列。',
                    '- 各训练 seed 分别统计，再跨 seed 等权均值及样本 SD(ddof=1)。',
                    '- 2000 次 percentile paired bootstrap，seed400007；单个训练 seed 内各 dev fold 独立抽 episode，',
                    '  同 fold 的方法/算子/强度保持配对；固定 fold 不重采样，不把15个fold×seed视作独立总体。',
                    '- 条件性 episode CI 不含训练随机性；三个训练 seed 不支持精确总体推断，区间跨0不等于等价。',
                    '- 比较方向固定 SOFTK-P0 / SOFTK-E2；5-shot 加 SOFTK-UNIT / SOFTK-QONLY；E2-P0 作参照。',
                    '- drop = clean − corruption；drop differential = 候选 drop − 基线 drop，负值才代表退化更小。',
                    '- support/both 仅 Add-Global、Drop-Local severity3，单列辅助，不混入主均值；所有负结果保留。',
                    '- 1800 旧代表诊断只覆盖5-shot：clean、query七算子severity3及support/both两算子severity3。',
                    '  诊断不能外推到1-shot或全强度；纠错/改错分母75，空组纯度NA，soft_mass不是硬人数。',
                    '- 后追加的开发分析，不自动冻结、换方法或选择有利数据集；条件依赖与负迁移照实保留。', '']
        table('Clean / 全强度主均值：逐 seed',
              [x for x in tables['cells'] if x['setting'] in ('clean', 'robust_query')],
              ['dataset', 'shot', 'seed', 'method', 'setting', 'accuracy_pct', 'drop_from_clean_pp'])
        table('跨训练 seed 主统计与辅助统计',
              [x for x in tables['seed_summary'] if x['setting'] in
               ('clean', 'robust_query', 'drop_differential', 'representative_aux')],
              ['dataset', 'shot', 'setting', 'side', 'method', 'comparison', 'train_seeds',
               'per_seed_values', 'mean', 'sample_sd', 'unit', 'partial'])
        table('逐 seed 配对效果与 drop 差',
              [x for x in tables['paired_effects'] if x['setting'] in
               ('clean', 'robust_query', 'drop_differential', 'representative_aux')],
              ['dataset', 'shot', 'seed', 'setting', 'side', 'comparison', 'delta_pp', 'ci95_pp'])
        table('逐算子 / 逐强度与辅助条件（含负结果）',
              [x for x in tables['seed_summary'] if x['setting'] in
               ('operator', 'severity_mean', 'query_severity3', 'severity', 'representative_condition')],
              ['dataset', 'shot', 'setting', 'side', 'operator', 'severity', 'method', 'comparison',
               'per_seed_values', 'mean', 'sample_sd', 'unit'])
        lines.extend(['## 完整输出', '',
                      'cells.csv 保留全部逐条件准确率与 clean-to-corruption drops；paired_effects.csv 保留各范围配对区间。',
                      'per_seed.csv / per_fold.csv / seed_summary.csv 保存逐seed、逐fold和跨seed统计；不筛选正结果。',
                      'diagnostics.csv 仅代表条件；coverage.json / sources.json 保存完整缺失键与只读来源。',
                      '本报告不包含速度结论；效率需独立真实 benchmark，不能使用共享 forward_seconds 推断单方法耗时。', ''])
        return '\n'.join(lines)

    table('Clean / 主鲁棒：每个训练 run',
          [r for r in tables['cells'] if r['setting'] in ('clean', 'robust_query')],
          ['dataset', 'shot', 'seed', 'method', 'setting', 'accuracy_pct', 'drop_from_clean_pp'])
    summary_settings = ('clean', 'robust_query', 'drop_differential', 'clean_best_last_max')
    summary_columns = ['dataset', 'shot', 'method', 'comparison', 'setting']
    if status.get('expanded_dev'):
        summary_settings += ('query_severity3', 'representative_aux')
        summary_columns += ['side', 'operator', 'severity']
    table('跨训练 seed 统计（不含 episode CI）',
          [r for r in tables['seed_summary'] if r['setting'] in summary_settings],
          summary_columns + ['train_seeds', 'per_seed_values', 'mean', 'sample_sd', 'unit'])
    table('主比较与锚点控制：逐 run 条件区间',
          [r for r in tables['paired_effects'] if r['setting'] in
           ('clean', 'robust_query', 'drop_differential', 'query_severity3', 'representative_aux')],
          ['dataset', 'shot', 'seed', 'side', 'setting', 'comparison', 'delta_pp', 'ci95_pp'])
    table('SOFTK vs E2 开发判读概览' if status['pool'] == 'dev' else 'SOFTK vs E2 预设复验',
          [r for r in tables['paired_effects'] if r['comparison'] == 'E2-SOFTK'],
          ['dataset', 'shot', 'seed', 'side', 'operator', 'severity', 'setting', 'delta_pp', 'ci95_pp'])
    lines.extend(['E2−SOFTK 的负值表示 SOFTK 在该范围更好；区间与各 fold 原值一并查看。',
                  '报告不自动选择主方法；无显著差异不等于等效。', ''])
    table('预设失败条件（无论本次是否仍失败）',
          [r for r in tables['ablation'] if r['dataset'] == 'ModelNet40' and r['side'] == 'both'
           and r['operator'] == 'Add-Global' and r['severity'] == 3],
          ['dataset', 'shot', 'seed', 'method', 'side', 'operator', 'severity', 'accuracy_pct', 'drop_from_clean_pp'])
    table('Clean best / last / max', tables['protocol_clean_best_last'],
          ['dataset', 'shot', 'seed', 'method', 'best_accuracy_pct', 'last_accuracy_pct', 'paper_style_max_pct'])
    lines.extend(['## 完整输出', '',
                  'CSV 保留未舍入值；Markdown 仅显示三位小数。cells / robust_query 含逐算子、逐强度和范围标签；',
                  'ablation 含全部代表条件；paired_effects / per_fold / per_seed / seed_summary 分别保存配对区间、逐折、逐 run 与跨 seed 统计；',
                  'diagnostics 按条件/fold/方法/原型类给出700 episode均值（空组纯度NA），原始逐 episode 标量由 source 指向；',
                  'coverage.json 给出精确缺失键，sources.json 显式区分旧 dev 复用与新阶段来源。', ''])
    return '\n'.join(lines)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_review_report(out, source_out, pool, *, partial=False, expanded_dev=False,
                        softk_dev=False, reuse_index=None):
    """No model/data loading. Final tables require every prediction and diagnostic."""
    if expanded_dev and pool != 'dev':
        raise ValueError('expanded_dev is restricted to the dev pool')
    if softk_dev and pool != 'dev':
        raise ValueError('softk_dev is restricted to the dev pool')
    if softk_dev and (reuse_index is None or expanded_dev) or reuse_index is not None and not softk_dev:
        raise ValueError('Soft-K requires --softk-dev --reuse-index, not --expanded-dev')
    cells = expected_cells(pool, expanded_dev=expanded_dev, softk_dev=softk_dev)
    matrix, coverage = load_review_predictions(out, source_out, pool, cells=cells, partial=True,
                                               reuse_index=reuse_index)
    diagnostics, diagnostic_coverage = load_diagnostics(out, cells, partial=True, matrix=matrix,
                                                       reuse_index=reuse_index)
    complete = coverage['complete'] and diagnostic_coverage['complete']
    target = Path(out) / 'reports' / ('softk_dev_full' if softk_dev else 'dev_ablation_multiseed' if expanded_dev else
                                    'dev_ablation' if pool == 'dev' else 'novel')
    target.mkdir(parents=True, exist_ok=True)
    status = dict(status='complete' if complete and not partial else 'partial' if partial else 'incomplete',
                  complete=complete and not partial, pool=pool, expected=coverage['expected'],
                  present=coverage['present'], diagnostics=diagnostic_coverage,
                  method_decision='not_automated', bootstrap_seed=400007, resamples=2000,
                  source_out=str(Path(source_out).resolve()), out=str(Path(out).resolve()))
    if expanded_dev:
        status['expanded_dev'] = True
    if softk_dev:
        status['softk_dev'] = True
        status['reuse_index'] = str(Path(reuse_index).resolve())
    sources = coverage.pop('sources')
    indexed = _index_entries(reuse_index, cells)
    _write_json(target / 'coverage.json', dict(**coverage, diagnostics=diagnostic_coverage))
    _write_json(target / 'sources.json', dict(key_fields=list(KEY_FIELDS), predictions=sources,
                diagnostics=[dict(key=list(cell_key(c)), path=indexed[cell_key(c)].get('diagnostic')
                                  if indexed is not None else str(diagnostic_path(out, c)))
                             for c in cells if representative_cell(c)]))
    if not complete and not partial:
        _write_json(target / 'status.json', status)
        # Remove only this reporter's obsolete rendered products: never leave a
        # previous COMPLETE markdown alongside a newly incomplete status.
        for name in ('REPORT.md', 'cells.csv', 'protocol_main.csv', 'protocol_clean_best_last.csv',
                     'robust_query.csv', 'ablation.csv', 'paired_effects.csv', 'per_seed.csv',
                     'per_fold.csv', 'seed_summary.csv', 'diagnostics.csv'):
            (target / name).unlink(missing_ok=True)
        raise IncompleteResults('incomplete: prediction/diagnostic coverage; see coverage.json')
    tables = build_tables(matrix, pool, cells=cells, partial=partial, softk_dev=softk_dev)
    tables['diagnostics'] = diagnostics
    for name, rows in tables.items():
        fields = list(dict.fromkeys(k for row in rows for k in row)) or ['pool']
        with (target / f'{name}.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()} for row in rows)
    (target / 'REPORT.md').write_text(_markdown(tables, status), encoding='utf-8')
    _write_json(target / 'status.json', status)  # completion marker last
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source-out', required=True)
    parser.add_argument('--pool', choices=('dev', 'test'), required=True)
    parser.add_argument('--partial', action='store_true', help='Labeled progress, never a final report')
    parser.add_argument('--expanded-dev', action='store_true',
                        help='Dev-only seeds 7/17/27; write reports/dev_ablation_multiseed')
    parser.add_argument('--softk-dev', action='store_true', help='Full 9240-cell Soft-K dev report')
    parser.add_argument('--reuse-index', help='Explicit read-only prediction/diagnostic source index')
    args = parser.parse_args(argv)
    try:
        status = write_review_report(args.out, args.source_out, args.pool, partial=args.partial,
                                     expanded_dev=args.expanded_dev, softk_dev=args.softk_dev,
                                     reuse_index=args.reuse_index)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps(status, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
