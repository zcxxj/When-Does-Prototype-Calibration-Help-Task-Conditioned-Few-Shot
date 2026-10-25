"""Frozen seed7, best-checkpoint, dev-only E2 attribution report."""
import numpy as np

from .campaign import DATASETS
from .evaluate import OPERATORS
from .summarize import (IncompleteResults, cell_array, load_matrix,
                        paired_aggregation, robust_array)

METHODS = ('P0', 'E2', 'F1', 'F2')
AUX_METHODS = ('P0', 'E2', 'F2')
MAIN_CONTRASTS = {
    'E2-P0': {'E2': 1, 'P0': -1},
    'F2-F1': {'F2': 1, 'F1': -1},
    'F1-P0': {'F1': 1, 'P0': -1},
    'F2-E2': {'F2': 1, 'E2': -1},
    '(F2-F1)-(E2-P0)': {'F2': 1, 'F1': -1, 'E2': -1, 'P0': 1},
}
AUX_CONTRASTS = {name: MAIN_CONTRASTS[name] for name in ('E2-P0', 'F2-E2')}
AUX_CONTRASTS['F2-P0'] = {'F2': 1, 'P0': -1}


def build_tables(matrix):
    """Average correlated axes BEFORE paired resampling; fixed dev folds differ."""
    tables = {'cells': [], 'paired_effects': []}

    def add(dataset, shot, setting, side, operator, severity, arrays, ci=True):
        context = dict(dataset=dataset, shot=shot, setting=setting, side=side,
                       operator=operator, severity=severity)
        for method, array in arrays.items():
            tables['cells'].append(dict(**context, method=method,
                                        accuracy_pct=float(100 * array.mean())))
        contrasts = AUX_CONTRASTS if side in ('support', 'both') else MAIN_CONTRASTS
        for comparison, weights in contrasts.items():
            # In particular, difference-in-differences is paired per episode,
            # never subtraction of independently computed confidence limits.
            delta = sum(weight * arrays[method] for method, weight in weights.items())
            if ci:
                effect = paired_aggregation(delta.tolist(), np.zeros_like(delta).tolist(),
                                            shared_folds=False, resamples=2000)
            else:
                effect = dict(delta_pp=float(100 * delta.mean()), ci95_pp=None,
                              per_seed_delta_pp=[float(100 * delta.mean())], seed_std_pp=None,
                              bootstrap_seed=400007, resamples=0, shared_folds=False)
            tables['paired_effects'].append(dict(**context, comparison=comparison, **effect))

    try:
        for dataset in DATASETS:
            for shot in (1, 5):
                def cells(condition, methods=METHODS):
                    return {m: cell_array(matrix, dataset, shot, m, 'best', condition, [7])
                            for m in methods}

                add(dataset, shot, 'clean', 'clean', None, 0, cells(('clean', None, 0)))
                if dataset.startswith('Scan'):
                    continue
                add(dataset, shot, 'robust_query', 'query', 'equal_weight_mean', 'all',
                    {m: robust_array(matrix, dataset, shot, m, [7]) for m in METHODS})
                for operator in OPERATORS:
                    add(dataset, shot, 'operator', 'query', operator, 'all',
                        {m: robust_array(matrix, dataset, shot, m, [7], operator) for m in METHODS})
                    for severity in range(1, 6):
                        add(dataset, shot, 'severity', 'query', operator, severity,
                            cells(('query', operator, severity)), ci=False)
                for side in ('support', 'both'):
                    per_operator = [cells((side, op, 3), AUX_METHODS) for op in OPERATORS]
                    add(dataset, shot, 'robust_aux', side, 'equal_weight_mean', 3,
                        {m: np.stack([a[m] for a in per_operator]).mean(0) for m in AUX_METHODS})
                    for operator, arrays in zip(OPERATORS, per_operator):
                        add(dataset, shot, 'operator', side, operator, 3, arrays)
    except KeyError as exc:
        raise IncompleteResults(f'incomplete: missing E2 attribution cell {exc}') from exc
    return tables


def _markdown(tables):
    lines = ['# E2 单训练seed开发集归因报告', '',
             '仅seed7、dev、best checkpoint；OA由逐episode预测重新计算，单位为%。', '',
             'fold、算子与强度等权；相关强度/算子先在同episode内平均，再做配对bootstrap。',
             '95% percentile区间：2000次，seed400007，shared_folds=False；固定fold不重采样。',
             '区间仅反映固定训练seed下episode变化，不代表多seed稳定性；重复对象也限制推断。',
             'F1辅助未评测；辅助场景不加入主鲁棒平均，不计算依赖F1的辅助交互。',
             '差中差是固定表征的描述性交互，不证明因果协同或均值算法的单独贡献；包含初次伪标签的几何差异。',
             '开发集归因结果不用于测试集方法选择。', '']

    def accuracy_table(title, setting, side, methods):
        rows = [r for r in tables['cells'] if r['setting'] == setting and r['side'] == side]
        if not rows:
            return
        lines.extend([f'## {title}', '',
                      '| 数据集 | shot | 算子 | 强度 | ' + ' | '.join(methods) + ' |',
                      '|---|---:|---|---|' + '---:|' * len(methods)])
        grouped = {}
        for row in rows:
            key = (row['dataset'], row['shot'], row['operator'], row['severity'])
            grouped.setdefault(key, {})[row['method']] = row['accuracy_pct']
        for (dataset, shot, operator, severity), values in grouped.items():
            lines.append(f'| {dataset} | {shot} | {operator or "—"} | {severity} | '
                         + ' | '.join(f'{values[m]:.3f}' for m in methods) + ' |')
        lines.append('')

    accuracy_table('Clean：六个设置', 'clean', 'clean', METHODS)
    accuracy_table('主鲁棒：四个设置（query）', 'robust_query', 'query', METHODS)
    accuracy_table('逐损坏类型（query，五档等权）', 'operator', 'query', METHODS)
    accuracy_table('逐强度（query）', 'severity', 'query', METHODS)
    for side in ('support', 'both'):
        accuracy_table(f'辅助 {side}：第3档，七算子等权', 'robust_aux', side, AUX_METHODS)
        accuracy_table(f'辅助 {side}：逐损坏类型，第3档', 'operator', side, AUX_METHODS)
    lines.extend(['## 配对差值与区间（pp）', '',
                  '逐强度均值及全部差值见CSV/JSON；逐强度不计算CI，其余聚合和逐算子均报告CI。', '',
                  '| 数据集 | shot | 场景 | 算子 | 对比 | 差值pp | 95% CI | 判读 |',
                  '|---|---:|---|---|---|---:|---|---|'])
    for row in tables['paired_effects']:
        if row['ci95_pp'] is None:
            continue
        low, high = row['ci95_pp']
        interpretation = '—'
        if row['comparison'] == 'F2-E2':
            interpretation = ('此设置有双曲后端增益证据' if low > 0 else
                              '此设置E2更好' if high < 0 else '未能分辨差异，不能声称等效')
        lines.append(f'| {row["dataset"]} | {row["shot"]} | {row["side"]}/{row["setting"]} | '
                     f'{row["operator"] or "—"} | {row["comparison"]} | {row["delta_pp"]:.3f} | '
                     f'[{low:.3f}, {high:.3f}] | {interpretation} |')
    return '\n'.join(lines) + '\n'


def write_e2_report(out, report_out):
    """Read existing dev predictions only; write five files to a fresh directory."""
    import csv
    import json
    from pathlib import Path

    source, target = Path(out).resolve(), Path(report_out).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('E2 report must use a separate 独立 directory outside source artifacts')
    if target.exists() and any(target.iterdir()):
        raise ValueError('Refusing to overwrite a nonempty report directory')
    matrix = load_matrix(out, [7], 'dev', methods=METHODS)
    tables = build_tables(matrix)
    markdown = _markdown(tables)
    status = dict(status='completed_e2_dev_only', complete=True, pool='dev',
                  train_seeds=[7], checkpoint='best', methods=list(METHODS),
                  auxiliary_methods=list(AUX_METHODS), source=str(source),
                  bootstrap_seed=400007, resamples=2000, shared_folds=False,
                  severity_ci=False)
    summary = dict(**status, prediction_cells=len(matrix),
                   prediction_cells_by_method={m: sum(key[2] == m for key in matrix) for m in METHODS},
                   tables=tables)
    target.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        with (target / f'{name}.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                              else value for key, value in row.items()} for row in rows)
    (target / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (target / 'REPORT.md').write_text(markdown, encoding='utf-8')
    # Completion marker is written last, never for missing/inconsistent inputs.
    (target / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return tables


def main(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--report-out', required=True)
    args = parser.parse_args(argv)
    try:
        write_e2_report(args.out, args.report_out)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    print((Path(args.report_out) / 'REPORT.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
