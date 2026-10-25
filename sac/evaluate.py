"""Shared-episode, external-label evaluation; no test-time updates."""
METHODS = {
    'P0': ('P0', 'euler', False),
    'F1': ('P0', 'hyperbolic', False),
    'F2': ('P0', 'hyperbolic', True),
    'J1': ('J', 'hyperbolic', False),
    'J2': ('J', 'hyperbolic', True),
    'H': ('H', 'hyperbolic', True),
}
EXTRA_METHODS = {'E2': ('P0', 'euler', True),
                 'D0': ('D', 'euler', False), 'D2': ('D', 'euler', True)}
OPERATORS = ('Scale', 'Jitter', 'Drop-Global', 'Drop-Local', 'Add-Global', 'Add-Local', 'Rotate')


def method_spec(name):
    if name in METHODS:
        return METHODS[name]
    if name in EXTRA_METHODS:
        return EXTRA_METHODS[name]
    raise ValueError(f'Unsupported method: {name}')


def selected_methods(family, requested=None):
    names = tuple(requested) if requested is not None else tuple(
        name for name, spec in METHODS.items() if spec[0] == family)
    if not names or len(names) != len(set(names)):
        raise ValueError('Expected nonempty unique methods')
    if any(method_spec(name)[0] != family for name in names):
        raise ValueError('Method/checkpoint family mismatch')
    return names


def evaluation_conditions(dataset, method, checkpoint='best'):
    method_spec(method)
    if method in ('D0', 'D2'):
        from .campaign import ALIASES
        if ALIASES.get(dataset, dataset) not in ('ModelNet40', 'ShapeNet70'):
            raise ValueError('D is restricted to the two synthetic datasets')
        if checkpoint != 'best':
            raise ValueError('D evaluation requires best checkpoint')
    if checkpoint not in ('best', 'last'):
        raise ValueError('Unsupported method/checkpoint')
    result = [('clean', None, 0)]
    if dataset.startswith('Scan') or checkpoint == 'last':
        return result
    result += [('query', op, severity) for op in OPERATORS for severity in range(1, 6)]
    if method in ('P0', 'E2', 'F2', 'J2'):
        result += [(side, op, 3) for side in ('support', 'both') for op in OPERATORS]
    return result


def predict_episode(model, episode, clouds, method):
    import time
    import numpy as np
    import torch
    from .train import model_inputs
    device = next(model.parameters()).device
    _, backend, refine = method_spec(method)
    model.eval()
    inputs = model_inputs(episode, clouds, device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    tic = time.perf_counter()
    with torch.no_grad():
        output = model(**inputs, backend=backend, refine=refine)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - tic
    before, after = [output[k].detach().cpu().numpy() for k in ('logits_before', 'logits_after')]
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise FloatingPointError('Nonfinite evaluation logits')
    pred = after.argmax(-1)
    return dict(logits_before=before, logits_after=after, local_target=episode.query_y.copy(),
                pred=pred, episode_accuracy=np.mean(pred == episode.query_y),
                point_counts=np.asarray([len(x) for x in clouds]), forward_seconds=elapsed,
                peak_allocated=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                peak_reserved=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0)


def predict_d_methods(model, episode, clouds, methods):
    """D0/D2 from one forward; timing is shared, not per-method latency."""
    import numpy as np
    names = selected_methods('D', methods)
    if model.family != 'D':
        raise ValueError('Shared D predictions require D checkpoint family')
    result = predict_episode(model, episode, clouds, 'D2' if 'D2' in names else 'D0')
    records = {}
    for name in names:
        record = dict(result)
        if name == 'D0':
            record['logits_after'] = record['logits_before']
            record['pred'] = record['logits_after'].argmax(-1)
            record['episode_accuracy'] = np.mean(record['pred'] == episode.query_y)
        records[name] = record
    return records


def damage_episode(clouds, episode, *, dataset, fold, shot, pool, episode_id,
                   side, operator, severity):
    import numpy as np
    from .data import corruption_seed
    from .corruptions import corrupt
    if dataset.startswith('Scan') and side != 'clean':
        raise ValueError('ScanObjectNN artificial corruption is not permitted')
    damaged = [np.asarray(x).copy() for x in clouds]
    seeds = np.full(len(clouds), -1, dtype=np.int64)
    if side == 'clean':
        return damaged, seeds
    if side not in ('query', 'support', 'both'):
        raise ValueError(f'Unknown damage side: {side}')
    for role, indices in [('support', episode.support_indices), ('query', episode.query_indices)]:
        if side not in (role, 'both'):
            continue
        for index in indices:
            seed = corruption_seed(dataset + '_FS', fold, shot, pool, episode_id, role,
                                   int(episode.rows[index]), operator, severity)
            seeds[index] = seed
            damaged[index] = corrupt(damaged[index], operator, severity, np.random.RandomState(seed))
    return damaged, seeds


def prediction_path(out, job, pool, method, checkpoint, condition):
    from pathlib import Path
    side, operator, severity = condition
    name = 'clean' if side == 'clean' else f'{side}_{operator}_{severity}'
    return Path(out) / 'predictions' / pool / job['dataset'] / f"fold{job['fold']}" / f"shot{job['shot']}" / f"seed{job['seed']}" / method / checkpoint / f'{name}.npz'


def shared_episodes(out, pool_data, job, pool, count):
    from .train_lock import training_lock
    common_fold = 0 if pool == 'test' and not job['dataset'].startswith('Scan') else job['fold']
    manifest_job = dict(job, fold=common_fold, seed=0, family='episodes-' + pool)
    with training_lock(out, manifest_job):
        return _shared_episodes(out, pool_data, job, pool, count)


def _shared_episodes(out, pool_data, job, pool, count):
    import json
    from pathlib import Path
    import numpy as np
    from .data import make_rng, sample_episode, Episode
    from .train import point_indices, write_json
    common_fold = 0 if pool == 'test' and not job['dataset'].startswith('Scan') else job['fold']
    root = Path(out) / 'episodes' / pool / job['dataset'] / f'fold{common_fold}' / f"shot{job['shot']}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'episodes.npz'
    index_path = root / 'point_indices.npz'
    expected = dict(dataset=job['dataset'], fold=common_fold, shot=job['shot'], pool=pool,
                    count=count, source=str(pool_data.path.resolve()),
                    episode_seed=100007 if pool == 'test' else 200007,
                    episode_stream=20 if pool == 'test' else 21)
    if manifest.exists():
        if json.loads((root / 'config.json').read_text()) != expected:
            raise ValueError(f'Shared episode configuration mismatch: {root}')
        values = dict(np.load(manifest))
        episodes = [Episode(**{key: values[key][i] for key in ('rows', 'class_ids', 'support_indices', 'query_indices', 'support_y', 'query_y')}) for i in range(count)]
        indices = np.load(index_path)['points']
    else:
        indices = point_indices(pool_data, job['dataset'], common_fold, job['shot'], 12 if pool == 'test' else 11)
        rng = make_rng(expected['episode_seed'], job['dataset'] + '_FS', common_fold, job['shot'], expected['episode_stream'])
        episodes = [sample_episode(pool_data.labels, rng, shot=job['shot'], rows=pool_data.rows) for _ in range(count)]
        values = {key: np.stack([getattr(ep, key) for ep in episodes]) for key in
                  ('rows', 'class_ids', 'support_rows', 'query_rows', 'support_indices', 'query_indices', 'support_y', 'query_y')}
        write_json(root / 'config.json', expected)
        np.savez_compressed(index_path, points=indices)
        np.savez_compressed(manifest, **values)
    return episodes, indices, str(manifest)


def validate_d_scope(args, job):
    """Reject out-of-scope D test requests before checkpoint/data access."""
    import json
    from pathlib import Path
    if job['dataset'] not in ('ModelNet40', 'ShapeNet70'):
        raise ValueError('D is restricted to the two synthetic datasets')
    if job['fold'] not in range(5) or job['shot'] not in (1, 5):
        raise ValueError('D requires fold 0..4 and shot 1/5')
    if args.pool != 'test':
        return
    if job['seed'] != 7:
        raise ValueError('Formal D evaluation requires seed 7')
    path = Path(args.out) / 'freeze.json'
    if not path.is_file():
        raise ValueError(f'D test requires freeze.json: {path}')
    freeze = json.loads(path.read_text())
    expected = dict(version='review1-evidence-v1', main_method='E2', new_family='D',
                    new_methods=['D0', 'D2'], reused_methods=['P0', 'E2'],
                    datasets=['ModelNet40', 'ShapeNet70'], folds=[0, 1, 2, 3, 4],
                    shots=[1, 5], training_seeds=[7], checkpoint='best', episodes=700,
                    conditions='clean_and_full_query_7x5',
                    excluded_calibration_controls=['UNIT', 'QONLY', 'SOFTK'],
                    test_previously_viewed=False)
    if not isinstance(freeze, dict) or any(freeze.get(k) != v for k, v in expected.items()):
        raise ValueError('D freeze.json must match the fixed review1 scope')


def evaluate_d_predictions(args, job, methods, model, run, config, pool_data,
                           episodes, indices, manifest):
    """D-only condition-major loop: shared forward, independent atomic cells."""
    import json
    import time
    import numpy as np
    import torch
    from .train import load_checkpoint, write_json
    state = load_checkpoint(run / 'best.pt', model, restore_rng=False)
    if state['config'] != config:
        raise ValueError('Checkpoint does not match training config')
    for condition in evaluation_conditions(job['dataset'], methods[0], 'best'):
        pending = {}
        for method in methods:
            path = prediction_path(args.out, job, args.pool, method, 'best', condition)
            metadata = dict(**job, pool=args.pool, method=method, checkpoint='best',
                            checkpoint_epoch=state['epoch'], condition=list(condition),
                            smoke=args.smoke, episodes=len(episodes), manifest=manifest,
                            gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
                            batch_strategy='512 equal: full episode; ragged: point-count buckets, chunk=8',
                            timing_scope='shared D forward; not independent method latency',
                            corruption_seed_formula='SeedSequence([300007,dataset_id,fold,shot,pool_id,episode_id,side_id,row_id,operator_id,severity])')
            if path.exists():
                if json.loads(path.with_suffix('.json').read_text()) != metadata:
                    raise ValueError(f'Prediction configuration mismatch: {path}')
                with np.load(path) as saved:
                    if len(saved['episode_accuracy']) != len(episodes):
                        raise ValueError(f'incomplete predictions: {path}')
                print(f'Already evaluated: {path}', flush=True)
            else:
                pending[method] = (path, metadata, [])
        if not pending:
            continue
        for episode_id, ep in enumerate(episodes):
            original = pool_data.clouds(ep, indices)
            tic = time.perf_counter()
            damaged, seeds = damage_episode(original, ep, dataset=job['dataset'],
                fold=job['fold'], shot=job['shot'], pool=args.pool, episode_id=episode_id,
                side=condition[0], operator=condition[1], severity=condition[2])
            corruption_seconds = time.perf_counter() - tic
            results = predict_d_methods(model, ep, damaged, tuple(pending))
            for method, result in results.items():
                result.update(corruption_seeds=seeds, corruption_seconds=corruption_seconds,
                              episode_id=episode_id, episode_rows=ep.rows.copy(),
                              support_rows=ep.support_rows.copy(), query_rows=ep.query_rows.copy(),
                              class_ids=ep.class_ids.copy())
                pending[method][2].append(result)
        for path, metadata, records in pending.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path.with_suffix('.json'), metadata)
            tmp = path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, **{key: np.stack([row[key] for row in records]) for key in records[0]})
            tmp.replace(path)
            print(json.dumps(dict(path=str(path), accuracy=float(np.mean([r['episode_accuracy'] for r in records])), episodes=len(records))), flush=True)


def evaluate(args):
    import json
    import time
    import numpy as np
    import torch
    from .campaign import ALIASES, run_dir
    from .train import base_pools, load_checkpoint, seed_everything, write_json
    from .model import EpisodeModel
    job = dict(dataset=ALIASES.get(args.dataset, args.dataset), fold=args.fold,
               shot=args.shot, seed=args.seed, family=args.family)
    methods = selected_methods(args.family, getattr(args, 'methods', None))
    if 'E2' in methods and (args.pool != 'dev' or args.seed != 7):
        raise ValueError('E2 is restricted to dev pool and seed 7')
    if args.smoke and args.pool != 'dev':
        raise ValueError('Smoke must never read novel test data')
    if args.family == 'D':
        validate_d_scope(args, job)
    run = run_dir(args.out, job)
    if not (run / 'completed.json').is_file():
        raise FileNotFoundError(f'incomplete training task: {run}')
    config = json.loads((run / 'config.json').read_text())
    if any(config[key] != value for key, value in job.items()) or config['smoke'] != args.smoke:
        raise ValueError('Checkpoint configuration/smoke mismatch')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    seed_everything(7)
    if args.pool == 'dev':
        pool_data = base_pools(args.data_root, job['dataset'], job['fold'])['val']
    else:
        from .data import load_pools
        pool_data = load_pools(args.data_root, job['dataset'] + '_FS', job['fold'])['test']
    episodes, indices, manifest = shared_episodes(args.out, pool_data, job, args.pool, 3 if args.smoke else 700)
    model = EpisodeModel(family=args.family, shot=args.shot).to(args.device)
    if args.family == 'D':
        return evaluate_d_predictions(args, job, methods, model, run, config, pool_data,
                                      episodes, indices, manifest)
    for checkpoint in (('best',) if args.pool == 'dev' else ('best', 'last')):
        state = load_checkpoint(run / f'{checkpoint}.pt', model, restore_rng=False)
        if state['config'] != config:
            raise ValueError('Checkpoint does not match training config')
        for method in methods:
            for condition in evaluation_conditions(job['dataset'], method, checkpoint):
                path = prediction_path(args.out, job, args.pool, method, checkpoint, condition)
                metadata = dict(**job, pool=args.pool, method=method, checkpoint=checkpoint,
                                checkpoint_epoch=state['epoch'], condition=list(condition),
                                smoke=args.smoke, episodes=len(episodes), manifest=manifest,
                                gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
                                batch_strategy='512 equal: full episode; ragged: point-count buckets, chunk=8',
                                corruption_seed_formula='SeedSequence([300007,dataset_id,fold,shot,pool_id,episode_id,side_id,row_id,operator_id,severity])')
                if path.exists():
                    existing = json.loads(path.with_suffix('.json').read_text())
                    if existing != metadata:
                        raise ValueError(f'Prediction configuration mismatch: {path}')
                    with np.load(path) as saved:
                        if len(saved['episode_accuracy']) != len(episodes):
                            raise ValueError(f'incomplete predictions: {path}')
                    print(f'Already evaluated: {path}', flush=True)
                    continue
                records = []
                for episode_id, ep in enumerate(episodes):
                    original = pool_data.clouds(ep, indices)
                    tic = time.perf_counter()
                    damaged, seeds = damage_episode(original, ep, dataset=job['dataset'],
                        fold=job['fold'], shot=job['shot'], pool=args.pool, episode_id=episode_id,
                        side=condition[0], operator=condition[1], severity=condition[2])
                    corruption_seconds = time.perf_counter() - tic
                    result = predict_episode(model, ep, damaged, method)
                    result.update(corruption_seeds=seeds, corruption_seconds=corruption_seconds,
                                  episode_id=episode_id)
                    records.append(result)
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path.with_suffix('.json'), metadata)
                tmp = path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, **{key: np.stack([row[key] for row in records]) for key in records[0]})
                tmp.replace(path)
                print(json.dumps(dict(path=str(path), accuracy=float(np.mean([r['episode_accuracy'] for r in records])), episodes=len(records))), flush=True)


def main(argv=None):
    from .train import runner_parser
    p = runner_parser(__doc__)
    p.add_argument('--pool', choices=('dev', 'test'), required=True)
    p.add_argument('--methods', nargs='+', choices=tuple(METHODS) + tuple(EXTRA_METHODS))
    args = p.parse_args(argv)
    try:
        from .campaign import ALIASES
        from .train_lock import training_lock
        # Serialize writers per family; shared_episodes has its own short lock.
        job = dict(dataset=ALIASES.get(args.dataset, args.dataset), fold=args.fold,
                   shot=args.shot, seed=args.seed, family='evaluation-' + args.pool + '-' + args.family)
        with training_lock(args.out, job):
            evaluate(args)
    except (ValueError, FileNotFoundError, RuntimeError, FloatingPointError) as exc:
        p.exit(1, f'incomplete: {exc}\n')


if __name__ == '__main__':
    main()
