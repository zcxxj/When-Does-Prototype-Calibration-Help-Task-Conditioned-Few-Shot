"""Finite, serial PCIA/HPR experiment campaign (no implicit training)."""
from itertools import product

DATASETS = {'ModelNet40': 5, 'ShapeNet70': 5, 'ScanObjectNN': 3}
ALIASES = {'ModelNet': 'ModelNet40', 'ModelNet40-FS': 'ModelNet40',
           'ShapeNet': 'ShapeNet70', 'ShapeNet70-FS': 'ShapeNet70',
    'ScanObjectNN-FS': 'ScanObjectNN', 'ModelNet40_FS': 'ModelNet40',
           'ShapeNet70_FS': 'ShapeNet70', 'ScanObjectNN_FS': 'ScanObjectNN'}


def training_jobs(train_seeds=(7,), *, dataset=None, fold=None, shot=None,
                  family=None, smoke=False):
    dataset = ALIASES.get(dataset, dataset)
    if dataset is not None and dataset not in DATASETS:
        raise ValueError(f'Unsupported dataset: {dataset}')
    if shot is not None and shot not in (1, 5):
        raise ValueError('shot must be 1 or 5')
    if family is not None and family not in ('P0', 'J', 'H', 'D'):
        raise ValueError('family must be P0, J, H or D')
    if family == 'D' and dataset == 'ScanObjectNN':
        raise ValueError('D is restricted to the two synthetic datasets')
    families = ('D',) if family == 'D' else ('P0', 'J', 'H')
    jobs = []
    for name, folds in DATASETS.items():
        if family == 'D' and name == 'ScanObjectNN':
            continue
        if dataset and name != dataset:
            continue
        for f, k, seed, fam in product(range(1 if smoke else folds), (1, 5), train_seeds, families):
            if (fold is None or fold == f) and (shot is None or shot == k) and (family is None or family == fam):
                jobs.append(dict(dataset=name, fold=f, shot=k, seed=seed, family=fam))
    if not jobs:
        raise ValueError('No matching tasks (check fold and filters)')
    return jobs


def run_dir(out, job):
    from pathlib import Path
    return Path(out) / 'runs' / job['dataset'] / f"fold{job['fold']}" / f"shot{job['shot']}" / f"seed{job['seed']}" / job['family']


def parser():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'matrix', 'smoke', 'train', 'dev', 'select', 'test', 'report'))
    p.add_argument('--data-root')
    p.add_argument('--out')
    p.add_argument('--dataset')
    p.add_argument('--fold', type=int)
    p.add_argument('--shot', type=int, choices=(1, 5))
    p.add_argument('--family', choices=('P0', 'J', 'H', 'D'))
    p.add_argument('--train-seeds', type=int, nargs='+', default=[7])
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    return p


def task_commands(args):
    import json
    import sys
    from pathlib import Path
    jobs = training_jobs(args.train_seeds, dataset=args.dataset, fold=args.fold,
                         shot=args.shot, family=args.family, smoke=args.command == 'smoke')
    selection_path = Path(args.out) / 'selection.json'
    if args.command != 'smoke' and selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        jobs = [job for job in jobs if job['seed'] == 7 or
                (job['seed'] in selection['additional_seeds'] and
                 job['family'] in selection['additional_families'])]
        if not jobs:
            raise ValueError('No tasks retained by frozen selection.json')
    for job in jobs:
        common = ['--data-root', args.data_root, '--out', args.out,
                  '--dataset', job['dataset'], '--fold', str(job['fold']),
                  '--shot', str(job['shot']), '--family', job['family'],
                  '--seed', str(job['seed']), '--device', args.device, '--threads', str(args.threads)]
        if args.command == 'smoke':
            common += ['--smoke']
        if args.command in ('train', 'smoke'):
            yield [sys.executable, '-m', 'sac.train', *common,
                   *(['--resume'] if args.resume else [])]
        if args.command in ('dev', 'test', 'smoke'):
            yield [sys.executable, '-m', 'sac.evaluate', *common,
                   '--pool', 'test' if args.command == 'test' else 'dev',
                   *(['--methods', 'D0', 'D2'] if job['family'] == 'D' else [])]


def prepare_data(data_root):
    from .data import prepare_data as prepare
    return prepare(data_root)


def main(argv=None):
    import json
    import subprocess
    p = parser()
    args = p.parse_args(argv)
    try:
        jobs = training_jobs(args.train_seeds, dataset=args.dataset, fold=args.fold,
                             shot=args.shot, family=args.family)
        if args.command == 'matrix':
            from collections import Counter
            counts = dict(Counter(j['family'] for j in jobs))
            print(json.dumps(dict(training_jobs=len(jobs), per_family=26 if len(jobs) == 78 else counts,
                                  family_counts=counts, jobs=jobs), indent=2))
            return 0
        if not args.out:
            p.error('--out is required')
        if args.command in ('select', 'report'):
            from .summarize import select_candidates, write_report
            (select_candidates if args.command == 'select' else write_report)(args.out, args.train_seeds)
            return 0
        if not args.data_root:
            p.error('--data-root is required')
        if args.command == 'prepare':
            from .train import write_json
            result = prepare_data(args.data_root)
            write_json(__import__('pathlib').Path(args.out) / 'prepared.json', result)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == 'test':
            if args.family == 'D':
                from types import SimpleNamespace
                from .evaluate import validate_d_scope
                for job in jobs:
                    validate_d_scope(SimpleNamespace(out=args.out, pool='test'), job)
            elif not (__import__('pathlib').Path(args.out) / 'selection.json').is_file():
                raise ValueError('Freeze development selection.json with campaign select before novel test')
        from contextlib import ExitStack
        from .train_lock import training_lock
        with ExitStack() as locks:
            if args.command in ('dev', 'test'):
                for shot in ([args.shot] if args.shot else [1, 5]):
                    locks.enter_context(training_lock(args.out, dict(
                        dataset='campaign', fold=0, shot=shot, seed=0, family=args.command)))
            for command in task_commands(args):
                print('RUN ' + ' '.join(command), flush=True)
                subprocess.run(command, check=True)
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'incomplete: {exc}', file=__import__('sys').stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
