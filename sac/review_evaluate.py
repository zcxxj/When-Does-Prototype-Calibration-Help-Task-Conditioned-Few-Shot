"""One fixed SAC-review condition, one frontend, explicit read-only sources."""
from pathlib import Path
import json
import time

import numpy as np
import torch

from .evaluate import OPERATORS, damage_episode, prediction_path, shared_episodes, predict_episode
from .campaign import ALIASES, run_dir
from .train import base_pools, load_checkpoint, seed_everything, model_inputs, write_json
from .model import EpisodeModel
from .euclidean import review_scores
from .review_diagnostics import (score_diagnostics, DIAGNOSTIC_KEYS, accuracy_metrics,
                                 extended_score_diagnostics, empirical_diagnostic_shapes)

CONTROLS = ('P0', 'E2', 'QONLY', 'UNIT', 'SOFTK')
METHODS = CONTROLS + ('F1', 'F2', 'H')


def representative(job, pool, checkpoint, condition):
    side, operator, severity = condition
    return ((job['seed'] == 7 or (pool == 'dev' and job['seed'] in (17, 27))) and job['shot'] == 5 and
            job['dataset'] in ('ModelNet40', 'ShapeNet70') and checkpoint == 'best' and
            (side == 'clean' or (side == 'query' and severity == 3) or
             (side in ('support', 'both') and operator in ('Add-Global', 'Drop-Local') and severity == 3)))


def validate_job(args):
    job = dict(dataset=ALIASES.get(args.dataset, args.dataset), fold=args.fold,
               shot=args.shot, seed=args.seed, family=args.family)
    methods = tuple(args.methods)
    condition = (args.side, args.operator, args.severity)
    expanded_dev = getattr(args, 'expanded_dev', False)
    if expanded_dev and args.pool != 'dev':
        raise ValueError('expanded_dev is dev only')
    if Path(args.out).resolve() == Path(args.source_out).resolve():
        raise ValueError('Output must be separate from read-only source')
    if args.pool not in ('dev', 'test') or args.checkpoint not in ('best', 'last'):
        raise ValueError('Unsupported pool/checkpoint')
    if job['dataset'] not in ('ModelNet40', 'ShapeNet70', 'ScanObjectNN'):
        raise ValueError('Unsupported dataset')
    if not 0 <= job['fold'] < (3 if job['dataset'].startswith('Scan') else 5):
        raise ValueError('Unsupported fold')
    if job['shot'] not in (1, 5) or job['seed'] not in (7, 17, 27):
        raise ValueError('Unsupported shot/seed')
    allowed = set(METHODS) - {'H'} if args.family == 'P0' else {'H'} if args.family == 'H' else set()
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= allowed:
        raise ValueError('Expected unique methods matching checkpoint family')
    if args.side == 'clean':
        if args.operator is not None or args.severity != 0:
            raise ValueError('Clean requires no operator and severity 0')
    elif (args.side not in ('query', 'support', 'both') or args.operator not in OPERATORS or
          not 1 <= args.severity <= 5):
        raise ValueError('Unsupported corruption condition')
    if args.checkpoint == 'last' and args.side != 'clean':
        raise ValueError('last is clean only')
    if getattr(args, 'empirical', False):
        return validate_empirical_job(args, job, methods, condition)
    if getattr(args, 'softk_dev', False):
        from .review_protocol import softk_dev_cells, cell_key
        allowed_keys = {cell_key(c) for c in softk_dev_cells()}
        for method in methods:
            cell = dict(job, pool=args.pool, checkpoint=args.checkpoint, method=method,
                        side=args.side, operator=args.operator, severity=args.severity)
            if args.family != 'P0' or cell_key(cell) not in allowed_keys:
                raise ValueError('Outside authorized softk dev matrix')
        checkpoint_out = getattr(args, 'checkpoint_out', None)
        if not checkpoint_out or not getattr(args, 'reuse_index', None):
            raise ValueError('softk dev requires checkpoint-out and reuse-index')
        if Path(args.out).resolve() == Path(checkpoint_out).resolve():
            raise ValueError('Output must be separate from read-only checkpoints')
        count = getattr(args, 'episodes', 700)
        if not isinstance(count, int) or not 1 <= count <= 700:
            raise ValueError('episodes must be 1..700')
        return job, methods, condition, count, representative(job, args.pool, args.checkpoint, condition)
    extra_methods = set(CONTROLS) if expanded_dev else {'P0', 'E2'}
    if job['seed'] != 7 and (job['dataset'].startswith('Scan') or not set(methods) <= extra_methods):
        raise ValueError('Additional train seeds only P0/E2 synthetic')
    if job['dataset'].startswith('Scan') and (args.side != 'clean' or not set(methods) <= {'P0','E2','H'}):
        raise ValueError('Scan only clean P0/E2/H')
    rep = representative(job, args.pool, args.checkpoint, condition)
    if args.pool == 'dev' and (not rep or not set(methods) <= set(CONTROLS) or
                               (job['seed'] != 7 and not expanded_dev)):
        raise ValueError('dev limited to explicitly authorized representative controls')
    if set(methods) & {'QONLY', 'UNIT', 'SOFTK'} and not rep:
        raise ValueError('Controls limited to representative conditions')
    if args.side in ('support', 'both') and (not rep or not set(methods) <= set(CONTROLS)):
        raise ValueError('Support/both limited to representative controls')
    count = getattr(args, 'episodes', 700)
    if not isinstance(count, int) or not 1 <= count <= 700:
        raise ValueError('episodes must be 1..700')
    if count != 700 and args.pool != 'dev':
        raise ValueError('Smoke must never read novel test data')
    if args.pool == 'test':
        freeze = Path(args.out) / 'freeze.json'
        if not freeze.is_file():
            raise FileNotFoundError(f'Novel evaluation requires freeze.json: {freeze}')
        if not isinstance(json.loads(freeze.read_text()), dict):
            raise ValueError('Invalid freeze.json')
    return job, methods, condition, count, rep


def validate_empirical_job(args, job, methods, condition):
    """The new frozen scope is separate from every historical scope switch."""
    protocol = getattr(args, 'protocol', 'balanced')
    if getattr(args, 'expanded_dev', False) or getattr(args, 'softk_dev', False):
        raise ValueError('Empirical scope cannot be mixed with historical development scopes')
    if (protocol not in ('balanced', 'imbalanced') or args.pool != 'test' or args.family != 'P0' or
            job['dataset'] not in ('ModelNet40', 'ShapeNet70') or args.checkpoint != 'best'):
        raise ValueError('Empirical study requires synthetic test data and P0 best checkpoints')
    allowed = set(CONTROLS) - ({'UNIT'} if job['shot'] == 1 else set())
    if not set(methods) <= allowed:
        raise ValueError('Empirical methods outside the frozen scope; one-shot UNIT equals SAC')
    side, operator, severity = condition
    valid_condition = side == 'clean' or (side == 'query' and (
        severity == 3 or protocol == 'balanced' and job['shot'] == 1)) or (
        protocol == 'balanced' and side in ('support', 'both') and severity == 3 and
        operator in ('Add-Global', 'Drop-Local'))
    if not valid_condition:
        raise ValueError('Condition outside the frozen empirical matrix')
    checkpoint_out = getattr(args, 'checkpoint_out', None)
    if job['seed'] != 7 and not checkpoint_out:
        raise ValueError('Additional seeds require explicit checkpoint-out')
    readonly_roots = (args.source_out, checkpoint_out, getattr(args, 'manifest_out', None))
    if any(root and Path(args.out).resolve() == Path(root).resolve() for root in readonly_roots):
        raise ValueError('Empirical output must be separate from read-only sources')
    count = getattr(args, 'episodes', 700)
    if not isinstance(count, int) or not 1 <= count <= 700:
        raise ValueError('episodes must be 1..700')
    freeze_path = Path(args.out) / 'freeze.json'
    if not freeze_path.is_file():
        raise FileNotFoundError(f'Empirical evaluation requires freeze.json: {freeze_path}')
    freeze = json.loads(freeze_path.read_text())
    required = dict(version='empirical-calibration-v1', protocols=['balanced', 'imbalanced'],
                    episodes=700, training_seeds=[7, 17, 27])
    if not isinstance(freeze, dict) or any(freeze.get(key) != value for key, value in required.items()):
        raise ValueError('Empirical freeze.json does not match the frozen study')
    if getattr(args, 'verify_reference', False) and protocol != 'balanced':
        raise ValueError('Historical reference verification requires the balanced protocol')
    return job, methods, condition, count, True


def _verify_empirical_reference(args, job, condition, count, predicted):
    """Check saved smoke predictions against original P0/SAC records, read-only."""
    checked = []
    roots = [args.source_out, getattr(args, 'checkpoint_out', None), getattr(args, 'manifest_out', None)]
    for method in ('P0', 'E2'):
        if method not in predicted:
            continue
        candidates = [prediction_path(root, job, 'test', method, 'best', condition)
                      for root in roots if root]
        path = next((path for path in candidates if path.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Missing historical {method} reference for matched smoke')
        with np.load(path, allow_pickle=False) as saved:
            for key in ('logits_before', 'logits_after', 'pred', 'local_target', 'corruption_seeds', 'episode_id'):
                actual = np.stack([record[key] for record in predicted[method]])
                expected = saved[key][:count]
                equal = (np.allclose(actual, expected, rtol=1e-4, atol=1e-5) if key.startswith('logits')
                         else np.array_equal(actual, expected))
                if not equal:
                    raise ValueError(f'Historical {method} compatibility mismatch: {key}')
        checked.append(dict(method=method, reference=str(path), episodes=count))
    if not checked:
        raise ValueError('Reference verification requires P0 or E2 in methods')
    return checked


def evaluate_empirical_job(args, job, methods, condition, count):
    from .data import load_pools
    from .empirical_sampling import (empirical_episodes, protocol_root, cached_file_sha256,
                                     scientific_code_identity)
    protocol, smoke = getattr(args, 'protocol', 'balanced'), count != 700
    output = Path(args.out) / 'smoke' / f'episodes{count}' if smoke else Path(args.out)
    output = protocol_root(output, protocol)
    checkpoint_root = args.source_out if job['seed'] == 7 else args.checkpoint_out
    run = run_dir(checkpoint_root, job)
    if not (run / 'completed.json').is_file():
        raise FileNotFoundError(f'Incomplete training task: {run}')
    config = json.loads((run / 'config.json').read_text())
    if any(config.get(key) != value for key, value in job.items()) or config.get('smoke') is not False:
        raise ValueError('Checkpoint configuration/smoke mismatch')
    checkpoint_path = run / 'best.pt'
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    seed_everything(7)
    pool_data = load_pools(args.data_root, job['dataset'] + '_FS', job['fold'])['test']
    manifest_out = getattr(args, 'manifest_out', None) or args.source_out
    episodes, indices, manifest, sampling_identity = empirical_episodes(
        args.out, manifest_out, pool_data, job, protocol, count)
    asset_cache = Path(args.out) / 'asset_hashes'
    input_identity = dict(sampling_identity, checkpoint_sha256=cached_file_sha256(checkpoint_path, asset_cache),
                          data_sha256=cached_file_sha256(pool_data.path, asset_cache),
                          **scientific_code_identity())
    model = EpisodeModel(family=args.family, shot=args.shot).to(args.device)
    state = load_checkpoint(checkpoint_path, model, restore_rng=False)
    if state['config'] != config:
        raise ValueError('Checkpoint does not match training config')
    metadata, paths, diag_paths, pending, pending_diag = {}, {}, {}, set(), set()
    for method in methods:
        meta = dict(**job, pool='test', method=method, checkpoint='best', checkpoint_epoch=state['epoch'],
                    condition=list(condition), smoke=smoke, episodes=count, manifest=manifest,
                    checkpoint_path=str(checkpoint_path), review_version=1, empirical_version=1,
                    **input_identity,
                    gpu=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None,
                    timing_scope='one complete PCIA frontend and all requested backends; shared episode latency',
                    batch_strategy='512 equal: full episode; ragged: point-count buckets, chunk=8',
                    corruption_seed_formula='SeedSequence([300007,dataset_id,0,shot,1,episode_id,side_id,row_id,operator_id,severity])')
        metadata[method] = meta
        paths[method] = prediction_path(output, job, 'test', method, 'best', condition)
        diag_paths[method] = diagnostic_path(output, job, 'test', method, 'best', condition)
        if not validate_artifact(paths[method], meta):
            pending.add(method)
        else:
            validate_empirical_pairing(paths[method], episodes, job, condition)
        if not validate_artifact(diag_paths[method], meta, diagnostic=True):
            pending_diag.add(method)
        else:
            validate_empirical_pairing(diag_paths[method], episodes, job, condition)
    result = dict(predictions={k: str(v) for k, v in paths.items()},
                  diagnostics={k: str(v) for k, v in diag_paths.items()}, protocol=protocol,
                  smoke=smoke, episodes=count, skipped=not pending and not pending_diag)
    if result['skipped']:
        if getattr(args, 'verify_reference', False):
            old = {}
            for method in methods:
                with np.load(paths[method], allow_pickle=False) as saved:
                    old[method] = [{key: saved[key][i] for key in saved.files} for i in range(count)]
            result['reference_verified'] = _verify_empirical_reference(args, job, condition, count, old)
        return result
    needed = tuple(method for method in methods if method in pending | pending_diag)
    records, diagnostics = {method: [] for method in needed}, {method: [] for method in pending_diag}
    for episode_id, ep in enumerate(episodes):
        original = pool_data.clouds(ep, indices)
        tic = time.perf_counter()
        damaged, seeds = damage_episode(original, ep, dataset=job['dataset'], fold=job['fold'],
            shot=job['shot'], pool='test', episode_id=episode_id,
            side=condition[0], operator=condition[1], severity=condition[2])
        corruption_seconds = time.perf_counter() - tic
        predicted, scored = predict_methods(model, ep, damaged, needed, diagnostics=True, extended_diagnostics=True)
        paired = dict(episode_id=episode_id, episode_rows=ep.rows.copy(), support_rows=ep.support_rows.copy(),
                      query_rows=ep.query_rows.copy(), class_ids=ep.class_ids.copy(), local_target=ep.query_y.copy(),
                      corruption_seeds=seeds, query_quota=np.bincount(ep.query_y, minlength=5))
        for method in needed:
            records[method].append(dict(predicted[method], **{k: v for k, v in paired.items() if k != 'local_target'},
                                        corruption_seconds=corruption_seconds))
        for method in pending_diag:
            diagnostics[method].append(dict(scored[method], **paired))
    if getattr(args, 'verify_reference', False):
        for method in ('P0', 'E2'):
            if method in methods and method not in records:
                with np.load(paths[method], allow_pickle=False) as saved:
                    records[method] = [{key: saved[key][i] for key in saved.files} for i in range(count)]
        result['reference_verified'] = _verify_empirical_reference(args, job, condition, count, records)
    for method in pending:
        save_records(paths[method], metadata[method], records[method])
    for method in pending_diag:
        save_records(diag_paths[method], metadata[method], diagnostics[method])
    return result


def predict_methods(model, episode, clouds, methods, *, diagnostics=False, extended_diagnostics=False):
    """Predict requested backends from ONE frozen PCIA episode; H stays separate."""
    model.eval()
    if model.family == 'H':
        if tuple(methods) != ('H',) or diagnostics:
            raise ValueError('H uses its independent legacy branch')
        return {'H': predict_episode(model, episode, clouds, 'H')}, {}
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    tic = time.perf_counter()
    with torch.no_grad():
        features = model.episode_features(**model_inputs(episode, clouds, device))
        names = tuple(dict.fromkeys(tuple(methods) + (CONTROLS if diagnostics and not extended_diagnostics else ())))
        controls = tuple(name for name in names if name in CONTROLS)
        outputs = review_scores(features['proto'], features['query'], model.shot, controls) if controls else {}
        if set(names) & {'F1', 'F2'}:
            from .hyperbolic import anchor_scores
            before, after = anchor_scores(features['proto'], features['query'], model.shot, refine='F2' in names)
            for name in names:
                if name in ('F1', 'F2'):
                    outputs[name] = dict(logits_before=before, logits_after=after if name == 'F2' else before)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - tic
    records, scores = {}, {}
    for method, output in outputs.items():
        values = {key: value.detach().cpu().numpy() for key, value in output.items()
                  if key != 'prototypes_after'}
        before, after = values['logits_before'], values['logits_after']
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise FloatingPointError('Nonfinite evaluation logits')
        pred = after.argmax(-1)
        records[method] = dict(logits_before=before, logits_after=after, pred=pred,
            local_target=episode.query_y.copy(), episode_accuracy=np.mean(pred == episode.query_y),
            point_counts=np.asarray([len(x) for x in clouds]), forward_seconds=elapsed,
            peak_allocated=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
            peak_reserved=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0)
        metrics = accuracy_metrics(pred, episode.query_y, after.shape[-1])
        records[method].update(macro_accuracy=metrics['macro_accuracy'], micro_accuracy=metrics['micro_accuracy'])
        if diagnostics and method in CONTROLS:
            scorer = extended_score_diagnostics if extended_diagnostics else score_diagnostics
            scores[method] = scorer(values, episode.query_y)
    return records, scores


def readonly_dev_episodes(source, pool_data, job, count):
    """Read the old 700-episode manifest without creating even a lock file."""
    from .data import Episode
    root = Path(source) / 'episodes' / 'dev' / job['dataset'] / f"fold{job['fold']}" / f"shot{job['shot']}"
    manifest = root / 'episodes.npz'
    for path in (manifest, root / 'config.json', root / 'point_indices.npz'):
        if not path.is_file():
            raise FileNotFoundError(f'Missing read-only dev manifest asset: {path}')
    expected = dict(dataset=job['dataset'], fold=job['fold'], shot=job['shot'], pool='dev',
                    count=700, source=str(pool_data.path.resolve()), episode_seed=200007, episode_stream=21)
    if json.loads((root / 'config.json').read_text()) != expected:
        raise ValueError(f'Shared episode configuration mismatch: {root}')
    keys = ('rows', 'class_ids', 'support_indices', 'query_indices', 'support_y', 'query_y')
    with np.load(manifest) as saved:
        if any(key not in saved or len(saved[key]) != 700 for key in keys):
            raise ValueError(f'incomplete dev manifest: {manifest}')
        episodes = [Episode(**{key: saved[key][i].copy() for key in keys}) for i in range(count)]
    with np.load(root / 'point_indices.npz') as saved:
        indices = saved['points'].copy()
    if indices.shape != (512,):
        raise ValueError('Invalid dev point indices')
    return episodes, indices, str(manifest)


def diagnostic_path(out, job, pool, method, checkpoint, condition):
    path = prediction_path(out, job, pool, method, checkpoint, condition)
    return Path(out) / 'diagnostics' / path.relative_to(Path(out) / 'predictions')


def validate_artifact(path, metadata, *, diagnostic=False, legacy=False):
    """Check identity, required keys and shapes, not repeated hashes."""
    path = Path(path)
    if not path.exists():
        if path.with_suffix('.json').exists():
            raise ValueError(f'incomplete artifact: {path}')
        return False
    if not path.with_suffix('.json').is_file():
        raise ValueError(f'incomplete metadata: {path}')
    old = json.loads(path.with_suffix('.json').read_text())
    identity = ('dataset', 'fold', 'shot', 'seed', 'family', 'pool', 'method', 'checkpoint',
                'checkpoint_epoch', 'condition', 'smoke', 'episodes', 'manifest')
    if old.get('protocol', 'balanced') != metadata.get('protocol', 'balanced'):
        raise ValueError(f'Prediction protocol mismatch: {path}')
    if metadata.get('empirical_version'):
        identity = tuple(key for key in identity if key != 'manifest')
    if any(old.get(key) != metadata[key] for key in identity):
        raise ValueError(f'Prediction configuration mismatch: {path}')
    review_identity = ('review_version',) if metadata.get('empirical_version') else ('review_version', 'checkpoint_path')
    if not legacy and any(old.get(key) != metadata[key] for key in review_identity):
        raise ValueError(f'Review configuration mismatch: {path}')
    if metadata.get('empirical_version'):
        empirical_identity = ('empirical_version', 'protocol', 'manifest_sha256', 'source_manifest_sha256',
                              'source_config_sha256', 'point_indices_sha256', 'checkpoint_sha256', 'data_sha256',
                              'scientific_code_sha256')
        if any(old.get(key) != metadata[key] for key in empirical_identity):
            raise ValueError(f'Empirical input identity mismatch: {path}')
    n, objects = metadata['episodes'], 5 * metadata['shot'] + 75
    shapes = dict(episode_id=(n,), local_target=(n, 75), corruption_seeds=(n, objects))
    if not legacy:
        shapes.update(episode_rows=(n, objects), support_rows=(n, 5 * metadata['shot']),
                      query_rows=(n, 75), class_ids=(n, 5))
    if diagnostic:
        shapes.update({key: (n,) if key in ('corrected_count', 'damaged_count', 'initial_accuracy')
                       else (n, 5) for key in DIAGNOSTIC_KEYS})
    else:
        shapes.update(logits_before=(n, 75, 5), logits_after=(n, 75, 5), pred=(n, 75),
                      episode_accuracy=(n,), point_counts=(n, objects))
    if metadata.get('empirical_version'):
        shapes['query_quota'] = (n, 5)
        if diagnostic:
            shapes.update(empirical_diagnostic_shapes(n))
        else:
            shapes.update(macro_accuracy=(n,), micro_accuracy=(n,))
    try:
        with np.load(path) as saved:
            if any(key not in saved or saved[key].shape != shape for key, shape in shapes.items()):
                raise ValueError(f'incomplete predictions: {path}')
            if metadata.get('empirical_version'):
                validate_empirical_contents(saved, metadata, diagnostic=diagnostic)
    except (OSError, EOFError) as exc:
        raise ValueError(f'incomplete predictions: {path}') from exc
    return True


def validate_empirical_contents(saved, metadata, *, diagnostic=False):
    """Reject internally inconsistent cached records, even when shapes match."""
    n = metadata['episodes']
    target = saved['local_target']
    if (not np.issubdtype(target.dtype, np.integer) or np.any(target < 0) or np.any(target >= 5) or
            not np.array_equal(saved['episode_id'], np.arange(n))):
        raise ValueError('Empirical target/episode identity conflict')
    counts = np.stack([np.bincount(row, minlength=5) for row in target])
    quotas = np.full((n, 5), 15) if metadata['protocol'] == 'balanced' else np.tile([5, 10, 15, 20, 25], (n, 1))
    if (not np.array_equal(saved['query_quota'], counts) or
            not np.array_equal(np.sort(counts, axis=1), quotas)):
        raise ValueError('Empirical query quota conflict')
    if not diagnostic:
        before, after, pred = saved['logits_before'], saved['logits_after'], saved['pred']
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError('Nonfinite empirical logits')
        if not np.array_equal(pred, after.argmax(-1)):
            raise ValueError('Empirical predictions disagree with logits')
        metrics = [accuracy_metrics(p, y, 5) for p, y in zip(pred, target)]
        for key in ('micro_accuracy', 'macro_accuracy'):
            if not np.allclose(saved[key], [value[key] for value in metrics], rtol=0, atol=1e-12):
                raise ValueError(f'Empirical {key} inconsistent with predictions')
        if not np.array_equal(saved['episode_accuracy'], saved['micro_accuracy']):
            raise ValueError('Legacy episode_accuracy differs from micro accuracy')
    else:
        for key in ('class_corrected_count', 'class_damaged_count'):
            values = saved[key]
            if np.any(values < 0) or np.any(values > counts) or not np.all(values == np.floor(values)):
                raise ValueError('Invalid class flip counts')
        if not np.array_equal(saved['class_query_count'], counts):
            raise ValueError('Diagnostic class counts conflict with targets')
        corrected, damaged = saved['class_corrected_count'], saved['class_damaged_count']
        if np.any(corrected + damaged > counts):
            raise ValueError('Diagnostic flips exceed class query counts')
        checks = dict(corrected_count=corrected.sum(1), damaged_count=damaged.sum(1),
            corrected_micro=corrected.sum(1) / 75, damaged_micro=damaged.sum(1) / 75,
            corrected_macro=(corrected / counts).mean(1), damaged_macro=(damaged / counts).mean(1),
            initial_macro_accuracy=saved['initial_class_accuracy'].mean(1),
            final_macro_accuracy=saved['final_class_accuracy'].mean(1),
            initial_accuracy=(saved['initial_class_accuracy'] * counts).sum(1) / 75,
            final_micro_accuracy=(saved['final_class_accuracy'] * counts).sum(1) / 75)
        for key, expected in checks.items():
            if not np.allclose(saved[key], expected, rtol=0, atol=1e-12):
                raise ValueError(f'Diagnostic {key} identity conflict')
        if (not np.allclose(saved['final_class_accuracy'] - saved['initial_class_accuracy'],
                            (corrected - damaged) / counts, rtol=0, atol=1e-12) or
                np.any(saved['group_size'] < 0) or np.any(saved['group_size'].sum(1) != 75)):
            raise ValueError('Diagnostic flip/group identity conflict')
        empty = saved['group_size'] == 0
        for key in ('group_class_accuracy', 'group_purity'):
            if not np.isnan(saved[key][empty]).all() or not np.isfinite(saved[key][~empty]).all():
                raise ValueError('Empty diagnostic groups must have missing agreement')
        for key, lower, upper in (('anchor_fraction', 0, 1), ('soft_entropy', 0, np.log(5)),
                                  ('soft_max_probability', 0, 1), ('initial_class_accuracy', 0, 1),
                                  ('final_class_accuracy', 0, 1)):
            values = saved[key]
            if not np.isfinite(values).all() or np.any(values < lower - 1e-12) or np.any(values > upper + 1e-12):
                raise ValueError(f'Invalid diagnostic {key}')


def validate_empirical_pairing(path, episodes, job, condition):
    """A resume must match current manifest objects and original damage seeds."""
    from .data import corruption_seed
    with np.load(path, allow_pickle=False) as saved:
        for key, attribute in (('episode_rows', 'rows'), ('support_rows', 'support_rows'),
                               ('query_rows', 'query_rows'), ('class_ids', 'class_ids'),
                               ('local_target', 'query_y')):
            if not np.array_equal(saved[key], np.stack([getattr(ep, attribute) for ep in episodes])):
                raise ValueError(f'Empirical manifest pairing conflict: {key}')
        expected = np.full(saved['corruption_seeds'].shape, -1, dtype=np.int64)
        side, operator, severity = condition
        if side != 'clean':
            for i, ep in enumerate(episodes):
                for role, positions in (('support', ep.support_indices), ('query', ep.query_indices)):
                    if side not in (role, 'both'):
                        continue
                    for position in positions:
                        expected[i, position] = corruption_seed(job['dataset'] + '_FS', job['fold'],
                            job['shot'], 'test', i, role, int(ep.rows[position]), operator, severity)
        if not np.array_equal(saved['corruption_seeds'], expected):
            raise ValueError('Empirical corruption identity conflict')


def build_softk_index(out, source_out, checkpoint_out, *, cells=None):
    """Validate existing dev assets once, and map missing keys to the new root."""
    from .review_protocol import softk_dev_cells, cell_key
    from .report_review import _digest
    out, source_out, checkpoint_out = map(Path, (out, source_out, checkpoint_out))
    if out.resolve() in (source_out.resolve(), checkpoint_out.resolve()):
        raise ValueError('Output must be separate from read-only sources')
    cells = softk_dev_cells() if cells is None else list(cells)
    entries, manifests, pairings, epochs = [], {}, {}, {}
    for cell in cells:
        job = {k: cell[k] for k in ('dataset', 'fold', 'shot', 'seed')}
        job['family'] = 'P0'
        condition = (cell['side'], cell['operator'], cell['severity'])
        roots = ([source_out, checkpoint_out] if cell['seed'] == 7 and cell['method'] in ('P0', 'E2')
                 else [checkpoint_out]) + [out]
        path, source, exists = None, out, False
        for root in roots:
            candidate = prediction_path(root, job, 'dev', cell['method'], 'best', condition)
            if candidate.exists() or candidate.with_suffix('.json').exists():
                path, source, exists = candidate, root, True
                break
        if path is None:
            path = prediction_path(out, job, 'dev', cell['method'], 'best', condition)
        if exists:
            if not path.is_file() or not path.with_suffix('.json').is_file():
                raise ValueError(f'incomplete artifact: {path}')
            meta = json.loads(path.with_suffix('.json').read_text())
            manifest = source_out / 'episodes/dev' / cell['dataset'] / f"fold{cell['fold']}" / f"shot{cell['shot']}" / 'episodes.npz'
            checkpoint = run_dir(source_out if cell['seed'] == 7 else checkpoint_out, job) / 'best.pt'
            expected = dict(meta, **job, pool='dev', method=cell['method'], checkpoint='best',
                            condition=list(condition), episodes=700, smoke=False, manifest=str(manifest),
                            checkpoint_path=str(checkpoint), review_version=1)
            legacy = source == source_out
            validate_artifact(path, expected, legacy=legacy)
            if meta.get('checkpoint_epoch') is None:
                raise ValueError(f'Missing checkpoint epoch: {path}')
            epoch_key = tuple(job.values())
            if epoch_key in epochs and epochs[epoch_key] != meta['checkpoint_epoch']:
                raise ValueError(f'Checkpoint epoch mismatch: {path}')
            epochs[epoch_key] = meta['checkpoint_epoch']
            if manifest not in manifests:
                with np.load(manifest) as saved:
                    manifests[manifest] = {k: saved[k].copy() for k in
                                          ('query_y', 'rows', 'support_rows', 'query_rows', 'class_ids')}
                with np.load(manifest.parent / 'point_indices.npz') as saved:
                    if saved['points'].shape != (512,):
                        raise ValueError('Invalid point indices')
            manifest_values = manifests[manifest]
            with np.load(path) as saved:
                if (not np.array_equal(saved['episode_id'], np.arange(700)) or
                        not np.array_equal(saved['local_target'], manifest_values['query_y'])):
                    raise ValueError(f'Episode pairing mismatch: {path}')
                for key, manifest_key in (('episode_rows', 'rows'), ('support_rows', 'support_rows'),
                                          ('query_rows', 'query_rows'), ('class_ids', 'class_ids')):
                    if key in saved and not np.array_equal(saved[key], manifest_values[manifest_key]):
                        raise ValueError(f'Episode {key} mismatch: {path}')
                pair = (cell['dataset'], cell['fold'], cell['shot'], *condition)
                damage = _digest(saved['corruption_seeds'])
                if pair in pairings and pairings[pair] != damage:
                    raise ValueError(f'Corruption pairing mismatch: {path}')
                pairings[pair] = damage
        diag = diagnostic_path(checkpoint_out, job, 'dev', cell['method'], 'best', condition)
        entry = dict(cell=cell, prediction=str(path), metadata=str(path.with_suffix('.json')),
                     source=source.name, exists=exists)
        if diag.exists():
            if not diag.with_suffix('.json').is_file():
                raise ValueError(f'incomplete diagnostic: {diag}')
            entry['diagnostic'] = str(diag)
        entries.append(entry)
    budget = dict(expected=len(cells), reused=sum(e['exists'] and e['source'] != out.name for e in entries),
                  present=sum(e['exists'] for e in entries), missing=sum(not e['exists'] for e in entries))
    budget['by_shot'] = {}
    for shot in (5, 1):
        subset = [e for e in entries if e['cell']['shot'] == shot]
        pending = [e for e in subset if not e['exists']]
        budget['by_shot'][str(shot)] = dict(expected=len(subset), present=len(subset)-len(pending),
            missing=len(pending), forwards=len({cell_key(e['cell'])[:5] + cell_key(e['cell'])[6:] for e in pending}))
    result = dict(version=1, cells=entries, budget=budget)
    write_json(out / 'reuse_index.json', result)
    write_json(out / 'matrix.json', dict(cells=cells, budget=budget))
    return result


def save_records(path, metadata, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {key: np.stack([row[key] for row in records]) for key in records[0]}
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **values)
    write_json(path.with_suffix('.json'), metadata)
    tmp.replace(path)
    validate_artifact(path, metadata, diagnostic='corrected_count' in values)


def evaluate_job(args):
    """Standalone fixed job; CLI serializes writers, shared_episodes locks itself."""
    job, methods, condition, count, representative_cell = validate_job(args)
    if getattr(args, 'empirical', False):
        return evaluate_empirical_job(args, job, methods, condition, count)
    softk_dev = getattr(args, 'softk_dev', False)
    diagnostic = representative_cell and job['family'] == 'P0' and not softk_dev
    smoke = count != 700
    output_root = Path(args.out) / 'smoke' / f'episodes{count}' if smoke else Path(args.out)
    checkpoint_root = args.source_out if job['seed'] == 7 else (args.checkpoint_out if softk_dev else args.out)
    run = run_dir(checkpoint_root, job)
    if not (run / 'completed.json').is_file():
        raise FileNotFoundError(f'incomplete training task: {run}')
    config = json.loads((run / 'config.json').read_text())
    if any(config.get(key) != value for key, value in job.items()) or config.get('smoke') is not False:
        raise ValueError('Checkpoint configuration/smoke mismatch')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    seed_everything(7)
    if args.pool == 'dev':
        pool_data = base_pools(args.data_root, job['dataset'], job['fold'])['val']
        episodes, indices, manifest = readonly_dev_episodes(args.source_out, pool_data, job, count)
    else:
        from .data import load_pools
        pool_data = load_pools(args.data_root, job['dataset'] + '_FS', job['fold'])['test']
        episodes, indices, manifest = shared_episodes(args.out, pool_data, job, args.pool, count)
    model = EpisodeModel(family=args.family, shot=args.shot).to(args.device)
    checkpoint_path = run / f'{args.checkpoint}.pt'
    state = load_checkpoint(checkpoint_path, model, restore_rng=False)
    if state['config'] != config:
        raise ValueError('Checkpoint does not match training config')
    names = tuple(dict.fromkeys(methods + (CONTROLS if diagnostic else ())))
    metadata, paths, diag_paths, pending, pending_diag = {}, {}, {}, set(), set()
    reference = {}
    reuse = {}
    if softk_dev and not smoke:
        from .review_protocol import cell_key
        reuse = {cell_key(e['cell']): e for e in json.loads(Path(args.reuse_index).read_text())['cells']}
    for method in names:
        meta = dict(**job, pool=args.pool, method=method, checkpoint=args.checkpoint,
            checkpoint_epoch=state['epoch'], condition=list(condition), smoke=smoke, episodes=count,
            manifest=manifest, checkpoint_path=str(checkpoint_path), review_version=1,
            gpu=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None,
            batch_strategy='512 equal: full episode; ragged: point-count buckets, chunk=8',
            corruption_seed_formula='SeedSequence([300007,dataset_id,fold,shot,pool_id,episode_id,side_id,row_id,operator_id,severity])')
        metadata[method] = meta
        if softk_dev:
            cell = dict(job, pool='dev', method=method, checkpoint=args.checkpoint,
                        side=condition[0], operator=condition[1], severity=condition[2])
            entry = reuse.get(cell_key(cell)) if not smoke else None
            path = Path(entry['prediction']) if entry else prediction_path(
                output_root, job, args.pool, method, args.checkpoint, condition)
            paths[method] = path
            legacy = path.is_relative_to(Path(args.source_out))
            if not validate_artifact(path, meta, legacy=legacy):
                if not path.is_relative_to(output_root):
                    raise ValueError(f'Missing indexed read-only source: {path}')
                pending.add(method)
        elif args.pool == 'dev' and job['seed'] == 7 and method in ('P0', 'E2'):
            path = prediction_path(args.source_out, job, args.pool, method, args.checkpoint, condition)
            source_meta = dict(meta, smoke=False, episodes=700)
            if not validate_artifact(path, source_meta, legacy=True):
                raise FileNotFoundError(f'Missing read-only reference predictions: {path}')
            with np.load(path) as saved:
                reference[method] = {key: saved[key][:count].copy() for key in
                    ('logits_before', 'logits_after', 'local_target', 'pred', 'corruption_seeds', 'episode_id')}
        elif method in methods:
            path = prediction_path(output_root, job, args.pool, method, args.checkpoint, condition)
            paths[method] = path
            if not validate_artifact(path, meta):
                pending.add(method)
        if diagnostic and method in CONTROLS:
            path = diagnostic_path(output_root, job, args.pool, method, args.checkpoint, condition)
            diag_paths[method] = path
            if not validate_artifact(path, meta, diagnostic=True):
                pending_diag.add(method)
    result = dict(predictions={k: str(v) for k,v in paths.items()},
                  diagnostics={k: str(v) for k,v in diag_paths.items()}, smoke=smoke, episodes=count,
                  skipped=not pending and not pending_diag)
    if result['skipped']:
        return result
    records = {name: [] for name in pending}
    diagnostics = {name: [] for name in pending_diag}
    for episode_id, ep in enumerate(episodes):
        original = pool_data.clouds(ep, indices)
        tic = time.perf_counter()
        damaged, seeds = damage_episode(original, ep, dataset=job['dataset'], fold=job['fold'],
            shot=job['shot'], pool=args.pool, episode_id=episode_id,
            side=condition[0], operator=condition[1], severity=condition[2])
        corruption_seconds = time.perf_counter() - tic
        requested = tuple(m for m in methods if m in pending) if softk_dev else methods
        predicted, scored = predict_methods(model, ep, damaged, requested, diagnostics=diagnostic)
        paired = dict(episode_id=episode_id, episode_rows=ep.rows, support_rows=ep.support_rows,
                      query_rows=ep.query_rows, class_ids=ep.class_ids, local_target=ep.query_y.copy(),
                      corruption_seeds=seeds)
        for method, old in reference.items():
            for key in ('logits_before', 'logits_after'):
                if not np.allclose(predicted[method][key], old[key][episode_id], rtol=1e-4, atol=1e-5):
                    raise ValueError(f'Read-only {method} compatibility mismatch: {key}, episode {episode_id}')
            for key in ('local_target', 'pred'):
                if not np.array_equal(predicted[method][key], old[key][episode_id]):
                    raise ValueError(f'Read-only {method} compatibility mismatch: {key}, episode {episode_id}')
            if not np.array_equal(seeds, old['corruption_seeds'][episode_id]) or old['episode_id'][episode_id] != episode_id:
                raise ValueError(f'Read-only {method} pairing mismatch')
        for method in pending:
            records[method].append(dict(predicted[method], **{k:v for k,v in paired.items() if k != 'local_target'},
                                        corruption_seconds=corruption_seconds))
        for method in pending_diag:
            diagnostics[method].append(dict(scored[method], **paired))
    for method in pending:
        save_records(paths[method], metadata[method], records[method])
    for method in pending_diag:
        save_records(diag_paths[method], metadata[method], diagnostics[method])
    return result


# Compatibility with the legacy evaluator's callable name.
evaluate = evaluate_job


def main(argv=None):
    import argparse
    import sys
    from .train_lock import training_lock
    argv = sys.argv[1:] if argv is None else argv
    if '--build-softk-index' in argv:
        parser = argparse.ArgumentParser(description='Read-only softk development reuse index')
        parser.add_argument('--build-softk-index', action='store_true')
        for key in ('out', 'source-out', 'checkpoint-out'):
            parser.add_argument('--' + key, required=True)
        args = parser.parse_args(argv)
        result = build_softk_index(args.out, args.source_out, args.checkpoint_out)
        print(json.dumps(result['budget']), flush=True)
        return
    p = argparse.ArgumentParser(__doc__)
    for key in ('out', 'data-root', 'dataset'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--source-out', '--base', dest='source_out', required=True)
    for key in ('fold', 'shot', 'seed'):
        p.add_argument('--' + key, type=int, required=True)
    p.add_argument('--family', choices=('P0','H'), required=True)
    p.add_argument('--pool', choices=('dev','test'), required=True)
    p.add_argument('--checkpoint', choices=('best','last'), required=True)
    p.add_argument('--side', choices=('clean','query','support','both'), required=True)
    p.add_argument('--operator', choices=OPERATORS)
    p.add_argument('--severity', type=int, default=0)
    p.add_argument('--methods', nargs='+', choices=METHODS, required=True)
    p.add_argument('--episodes', type=int, default=700)
    p.add_argument('--expanded-dev', action='store_true')
    p.add_argument('--softk-dev', action='store_true')
    p.add_argument('--empirical', action='store_true')
    p.add_argument('--protocol', choices=('balanced', 'imbalanced'), default='balanced')
    p.add_argument('--manifest-out')
    p.add_argument('--verify-reference', action='store_true')
    p.add_argument('--checkpoint-out')
    p.add_argument('--reuse-index')
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args(argv)
    try:
        job, _, condition, count, _ = validate_job(args)
        lock_job = dict(job, family='review-' + '-'.join(map(str, (args.pool, args.family,
                          args.checkpoint, getattr(args, 'protocol', 'balanced'), *condition, count))))
        with training_lock(args.out, lock_job):
            result = evaluate_job(args)
        print(json.dumps(result), flush=True)
    except (ValueError, FileNotFoundError, RuntimeError, FloatingPointError) as exc:
        p.exit(1, f'incomplete: {exc}\n')


if __name__ == '__main__':
    main()
