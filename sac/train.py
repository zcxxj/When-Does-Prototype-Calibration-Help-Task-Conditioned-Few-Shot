"""Paired episodic training and epoch-boundary restart."""
import json
import math
from pathlib import Path


def checkpoint_policy(family):
    return {'P0': ('euler', False, 'logits_before'),
            'D': ('euler', False, 'logits_before'),
            'J': ('hyperbolic', True, 'logits_after'),
            'H': ('hyperbolic', True, 'logits_after')}[family]


def advance_best(score, best, stale):
    if not math.isfinite(score):
        raise FloatingPointError('Nonfinite validation accuracy')
    improved = score >= best
    return (score, 0, True) if improved else (best, stale + 1, False)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def seed_everything(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def save_checkpoint(path, model, optimizer, scheduler, *, epoch, best, stale, config):
    import random
    import numpy as np
    import torch
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), epoch=epoch, best=best, stale=stale, config=config,
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                          torch=torch.get_rng_state(),
                          cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, *, restore_rng=True):
    import random
    import numpy as np
    import torch
    # Only load locally produced trusted checkpoints (RNG includes Python/NumPy).
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    if restore_rng:
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if torch.cuda.is_available() and state['rng']['cuda']:
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state


def model_inputs(episode, clouds, device):
    import torch
    return dict(clouds=[torch.as_tensor(x, dtype=torch.float32, device=device) for x in clouds],
                support_indices=torch.as_tensor(episode.support_indices, dtype=torch.long, device=device),
                support_y=torch.as_tensor(episode.support_y, dtype=torch.long, device=device),
                query_indices=torch.as_tensor(episode.query_indices, dtype=torch.long, device=device))


def run_epoch(model, episodes, *, family, optimizer=None):
    import torch
    backend, refine, score_key = checkpoint_policy(family)
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    losses, accuracies, norms = [], [], []
    with torch.set_grad_enabled(training):
        for episode, clouds in episodes:
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(**model_inputs(episode, clouds, device), backend=backend, refine=refine)
            target = torch.as_tensor(episode.query_y, dtype=torch.long, device=device)
            for key in ('logits_before', 'logits_after'):
                if not torch.isfinite(output[key]).all():
                    raise FloatingPointError(f'Nonfinite {key}')
            loss = torch.nn.functional.cross_entropy(output['logits_before'], target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            if training:
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not grads or any(not torch.isfinite(g).all() for g in grads):
                    raise FloatingPointError('Missing/nonfinite gradients')
                norm = sum(float(g.detach().square().sum()) for g in grads) ** .5
                if norm == 0:
                    raise FloatingPointError('All gradients are zero')
                norms.append(norm)
                optimizer.step()
            losses.append(float(loss.detach()))
            accuracies.append(float((output[score_key].argmax(-1) == target).float().mean()))
    if not losses:
        raise ValueError('Empty epoch')
    return dict(loss=sum(losses) / len(losses), accuracy=sum(accuracies) / len(accuracies),
                episodes=len(losses), gradient_norm_mean=sum(norms) / len(norms) if norms else None)


def base_pools(data_root, dataset, fold):
    """Open base H5 ONLY: training and smoke must never open novel files."""
    import h5py
    import numpy as np
    from .data import official_paths, H5Pool, split_classes
    path = official_paths(data_root, dataset + '_FS', fold)['train']
    if not path.is_file():
        raise FileNotFoundError(f'Missing official PCIA base H5: {path}; run campaign prepare for sources')
    with h5py.File(path, 'r') as handle:
        points = np.asarray(handle['data'], dtype=np.float32)
        labels = np.asarray(handle['label']).reshape(-1).astype(np.int64)
    train_classes, val_classes = split_classes(labels, dataset + '_FS', fold)
    return {name: H5Pool(path, points, labels, np.flatnonzero(np.isin(labels, classes)))
            for name, classes in [('train', train_classes), ('val', val_classes)]}


def point_indices(pool, dataset, fold, shot, stream):
    from .data import make_rng, sample_point_indices
    return sample_point_indices(pool.data.shape[1], make_rng(7, dataset + '_FS', fold, shot, stream))


def epoch_episodes(pool, indices, job, epoch, count, *, validation=False):
    import numpy as np
    from .data import make_rng, sample_episode, augment
    common = (job['seed'], job['dataset'] + '_FS', job['fold'], job['shot'])
    sampling = make_rng(*common, 4 if validation else 2, epoch)
    augmentation = make_rng(*common, 5 if validation else 3, epoch)
    for _ in range(count):
        episode = sample_episode(pool.labels, sampling, shot=job['shot'], rows=pool.rows)
        clouds = augment(np.stack(pool.clouds(episode, indices)), augmentation)
        yield episode, list(clouds)


def runner_parser(description):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--fold', required=True, type=int)
    p.add_argument('--shot', required=True, type=int, choices=(1, 5))
    p.add_argument('--family', required=True, choices=('P0', 'J', 'H', 'D'))
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    return p


def training_config(args):
    from .campaign import ALIASES
    return dict(dataset=ALIASES.get(args.dataset, args.dataset), fold=args.fold, shot=args.shot,
                seed=args.seed, family=args.family, data_root=str(Path(args.data_root).resolve()),
                smoke=args.smoke, epochs=1 if args.smoke else 80,
                train_episodes=3 if args.smoke else 400, val_episodes=3 if args.smoke else 600,
                patience=30, lr=.0008, betas=[.9, .999], eps=1e-8, weight_decay=0,
                step_size=5, gamma=.5, points=512, query=15, way=5,
                split_seed=7, seed_algorithm='numpy.SeedSequence([seed,dataset_id,fold,shot,stream,epoch])',
                selection=checkpoint_policy(args.family)[2], fp32=True, tf32=False)


def train(args):
    import time
    import numpy as np
    import torch
    from .campaign import run_dir, training_jobs
    from .data import stream_seed
    from .model import EpisodeModel
    config = training_config(args)
    job = {k: config[k] for k in ('dataset', 'fold', 'shot', 'seed', 'family')}
    training_jobs([job['seed']], **{k: job[k] for k in ('dataset', 'fold', 'shot', 'family')})
    out = run_dir(args.out, job)
    config_path = out / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f'Configuration mismatch: {config_path}')
        if (out / 'completed.json').exists():
            print(f'Already complete (identical config): {out}', flush=True)
            return
        if not args.resume:
            raise RuntimeError(f'Interrupted run: use --resume: {out}')
        if not (out / 'resume.pt').exists():
            raise RuntimeError(f'No epoch-boundary resume.pt: use a new output directory: {out}')
    elif args.resume:
        raise RuntimeError(f'No run to resume: {out}')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    torch.set_num_threads(args.threads)
    pools = base_pools(args.data_root, job['dataset'], job['fold'])
    indices = {name: point_indices(pools[name], job['dataset'], job['fold'], job['shot'], stream)
               for name, stream in [('train', 10), ('val', 11)]}
    init_seed = stream_seed(job['seed'], job['dataset'] + '_FS', job['fold'], job['shot'], 1, 0)
    seed_everything(init_seed)
    model = EpisodeModel(family=job['family'], shot=job['shot']).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0008, betas=(.9, .999), eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=.5)
    epoch, best, stale = 0, -1., 0
    start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        state = load_checkpoint(out / 'resume.pt', model, optimizer, scheduler)
        epoch, best, stale = state['epoch'], state['best'], state['stale']
        saved = dict(np.load(out / 'point_indices.npz'))
        if any(not np.array_equal(saved[name], value) for name, value in indices.items()):
            raise ValueError('Point indices mismatch on resume')
        # A crash between epoch log and resume commit must not duplicate rows.
        logpath = out / 'epochs.jsonl'
        records = [json.loads(line) for line in logpath.read_text().splitlines()] if logpath.exists() else []
        logpath.write_text(''.join(json.dumps(r) + '\n' for r in records if r['epoch'] <= epoch))
    else:
        write_json(config_path, config)
        np.savez_compressed(out / 'point_indices.npz', **indices)
        write_json(out / 'runtime.json', dict(started_unix=start, torch=torch.__version__,
                   device=args.device, gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
                   init_seed=init_seed, init_seed_array=[job['seed'], {'ModelNet40': 0, 'ShapeNet70': 1, 'ScanObjectNN': 2}[job['dataset']], job['fold'], job['shot'], 1, 0]))
    while epoch < config['epochs'] and stale < config['patience']:
        tic = time.perf_counter()
        train_result = run_epoch(model, epoch_episodes(pools['train'], indices['train'], job, epoch, config['train_episodes']), family=job['family'], optimizer=optimizer)
        val_result = run_epoch(model, epoch_episodes(pools['val'], indices['val'], job, epoch, config['val_episodes'], validation=True), family=job['family'])
        best, stale, improved = advance_best(val_result['accuracy'], best, stale)
        scheduler.step()
        epoch += 1
        row = dict(epoch=epoch, train=train_result, val=val_result, best=best, stale=stale,
                   elapsed_seconds=time.perf_counter() - tic, lr=optimizer.param_groups[0]['lr'],
                   seeds={str(s): stream_seed(job['seed'], job['dataset'] + '_FS', job['fold'], job['shot'], s, epoch - 1) for s in (2, 3, 4, 5)})
        for filename in (['best.pt'] if improved else []) + ['last.pt', 'resume.pt']:
            save_checkpoint(out / filename, model, optimizer, scheduler, epoch=epoch, best=best, stale=stale, config=config)
        with (out / 'epochs.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)
    write_json(out / 'completed.json', dict(epoch=epoch, best=best, stale=stale, smoke=args.smoke,
               finished_unix=time.time(), invocation_seconds=time.time() - start))


def main(argv=None):
    p = runner_parser(__doc__)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    try:
        from .train_lock import training_lock
        config = training_config(args)
        job = {k: config[k] for k in ('dataset', 'fold', 'shot', 'seed', 'family')}
        with training_lock(args.out, job):
            train(args)
    except (ValueError, FileNotFoundError, RuntimeError, FloatingPointError) as exc:
        p.exit(1, f'incomplete: {exc}\n')


if __name__ == '__main__':
    main()
