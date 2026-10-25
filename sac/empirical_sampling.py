"""Frozen balanced/imbalanced manifests, independent of training or scheduling.

The imbalanced episode changes 15 of the 75 queries. Every support and retained
query keeps its batch position, object row, and point sampling. Labels are used
only by the sampler and external evaluator, never the calibration interface.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

from .data import DATASET_IDS, Episode

PROTOCOLS = ('balanced', 'imbalanced')
QUOTAS = (5, 10, 15, 20, 25)
SAMPLING_SEED = 500007
SAMPLING_STREAM = 22
EPISODE_KEYS = ('rows', 'class_ids', 'support_indices', 'query_indices', 'support_y', 'query_y')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_root(out, protocol):
    if protocol not in PROTOCOLS:
        raise ValueError('Unsupported query protocol')
    return Path(out) / 'protocols' / protocol


def imbalance_rng(dataset, shot, episode_id):
    """A separate fixed stream: no fold, model seed, global RNG, or damage RNG."""
    dataset = dataset if dataset.endswith('_FS') else dataset + '_FS'
    if dataset not in ('ModelNet40_FS', 'ShapeNet70_FS') or shot not in (1, 5):
        raise ValueError('Imbalance is restricted to synthetic 1/5-shot episodes')
    seed = np.random.SeedSequence([SAMPLING_SEED, DATASET_IDS[dataset], shot,
                                  SAMPLING_STREAM, int(episode_id)]).generate_state(1)[0]
    return np.random.RandomState(int(seed))


def imbalanced_episode(source, labels, allowed_rows, *, dataset, shot, episode_id):
    """Assign random 5/10/15/20/25 quotas; retain exactly 60 source queries.

    Retention is min(15, quota) in each class. Added objects exclude the entire
    source episode, including discarded queries; sampling is without replacement.
    The 15 dropped batch slots receive the 15 additional objects, so support
    indices and the positions of all shared objects remain unchanged.
    """
    labels, allowed_rows = np.asarray(labels).reshape(-1), np.asarray(allowed_rows)
    if (len(source.class_ids) != 5 or len(source.query_y) != 75 or
            len(source.support_y) != 5 * shot or len(np.unique(source.rows)) != len(source.rows) or
            not np.array_equal(np.bincount(source.query_y, minlength=5), np.full(5, 15))):
        raise ValueError('Expected a distinct-object balanced source episode')
    rng = imbalance_rng(dataset, shot, episode_id)
    quotas = rng.permutation(QUOTAS)
    rows = source.rows.copy()
    retained, dropped, added = [], [], []
    candidates = allowed_rows[~np.isin(allowed_rows, source.rows)]
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError('Pool contains duplicate object rows')
    for local_class, (class_id, quota) in enumerate(zip(source.class_ids, quotas)):
        original = source.query_indices[source.query_y == local_class]
        keep = original if quota >= 15 else np.sort(rng.choice(original, int(quota), replace=False))
        retained.extend(keep.tolist())
        dropped.extend(np.setdiff1d(original, keep).tolist())
        if quota > 15:
            eligible = candidates[labels[candidates] == class_id]
            if len(eligible) < quota - 15:
                raise ValueError(f'Insufficient unused objects for class {class_id}')
            added.extend(rng.choice(eligible, int(quota - 15), replace=False).tolist())
    rows[np.asarray(dropped)] = np.asarray(added)
    query_indices = np.concatenate([np.flatnonzero((labels[rows] == c) &
        ~np.isin(np.arange(len(rows)), source.support_indices)) for c in source.class_ids])
    result = Episode(rows, source.class_ids.copy(), source.support_indices.copy(), query_indices,
                     source.support_y.copy(), np.repeat(np.arange(5), quotas))
    validate_imbalanced_episode(source, result, labels, allowed_rows, quotas)
    return result, quotas


def validate_imbalanced_episode(source, episode, labels, allowed_rows, quotas):
    quotas = np.asarray(quotas)
    if sorted(quotas.tolist()) != list(QUOTAS):
        raise ValueError('Invalid query quota permutation')
    if (not np.array_equal(source.class_ids, episode.class_ids) or
            not np.array_equal(source.support_indices, episode.support_indices) or
            not np.array_equal(source.support_rows, episode.support_rows) or
            not np.array_equal(source.support_y, episode.support_y)):
        raise ValueError('Support or episode classes changed')
    if (len(episode.rows) != len(source.rows) or len(np.unique(episode.rows)) != len(episode.rows) or
            not np.isin(episode.rows, allowed_rows).all()):
        raise ValueError('Invalid or duplicate object rows')
    if (len(episode.query_indices) != 75 or len(episode.query_y) != 75 or
            not np.array_equal(np.sort(np.r_[episode.support_indices, episode.query_indices]),
                               np.arange(len(episode.rows))) or
            not np.array_equal(np.bincount(episode.query_y, minlength=5), quotas)):
        raise ValueError('Invalid query count or index partition')
    if not np.array_equal(np.asarray(labels)[episode.query_rows], episode.class_ids[episode.query_y]):
        raise ValueError('Query labels do not match source objects')
    shared = np.intersect1d(source.query_rows, episode.query_rows)
    if len(shared) != 60:
        raise ValueError('Imbalanced episode must share exactly 60 source queries')
    shared_positions = np.flatnonzero(np.isin(source.rows, shared))
    if not np.array_equal(source.rows[shared_positions], episode.rows[shared_positions]):
        raise ValueError('Retained query batch positions changed')


def readonly_test_episodes(source_out, pool_data, job):
    """Load all 700 original episodes; never create or repair source assets."""
    root = Path(source_out) / 'episodes' / 'test' / job['dataset'] / 'fold0' / f"shot{job['shot']}"
    manifest, points, config_path = root / 'episodes.npz', root / 'point_indices.npz', root / 'config.json'
    for path in (manifest, points, config_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing read-only test manifest asset: {path}')
    config = json.loads(config_path.read_text())
    expected = dict(dataset=job['dataset'], fold=0, shot=job['shot'], pool='test', count=700,
                    episode_seed=100007, episode_stream=20)
    # Source absolute paths can differ between an identical readonly replica and
    # its original host; row labels/points and hashed source assets establish pairing.
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Shared test episode configuration mismatch: {root}')
    with np.load(manifest, allow_pickle=False) as saved:
        if any(key not in saved or len(saved[key]) != 700 for key in EPISODE_KEYS):
            raise ValueError(f'Incomplete source manifest: {manifest}')
        episodes = [Episode(**{key: saved[key][i].copy() for key in EPISODE_KEYS}) for i in range(700)]
    with np.load(points, allow_pickle=False) as saved:
        indices = saved['points'].copy()
    if (indices.shape != (512,) or not np.issubdtype(indices.dtype, np.integer) or
            len(np.unique(indices)) != 512 or indices.min() < 0 or indices.max() >= pool_data.data.shape[1]):
        raise ValueError('Invalid source point indices')
    for episode in episodes:
        if (len(episode.rows) != 5 * job['shot'] + 75 or
                len(np.unique(episode.rows)) != len(episode.rows) or
                not np.isin(episode.rows, pool_data.rows).all() or
                not np.array_equal(np.bincount(episode.query_y, minlength=5), np.full(5, 15)) or
                not np.array_equal(pool_data.labels[episode.query_rows], episode.class_ids[episode.query_y]) or
                not np.array_equal(pool_data.labels[episode.support_rows], episode.support_y)):
            raise ValueError('Source manifest does not match the evaluation data')
    identity = dict(source_manifest=str(manifest), source_manifest_sha256=file_sha256(manifest),
                    source_config_sha256=file_sha256(config_path), point_indices=str(points),
                    point_indices_sha256=file_sha256(points))
    return episodes, indices, str(manifest), identity


def empirical_episodes(out, source_out, pool_data, job, protocol, count=700):
    """Create a full frozen manifest once; a smoke run only slices this manifest."""
    if not isinstance(count, int) or not 1 <= count <= 700:
        raise ValueError('episodes must be 1..700')
    root = protocol_root(out, protocol)
    episodes, indices, source_manifest, identity = readonly_test_episodes(source_out, pool_data, job)
    if protocol == 'balanced':
        return episodes[:count], indices, source_manifest, dict(identity,
            protocol=protocol, manifest_sha256=identity['source_manifest_sha256'])
    from .train_lock import training_lock
    manifest_job = dict(job, fold=0, seed=0, family='empirical-imbalanced-episodes')
    with training_lock(out, manifest_job):
        root = root / 'episodes' / 'test' / job['dataset'] / 'fold0' / f"shot{job['shot']}"
        root.mkdir(parents=True, exist_ok=True)
        manifest, config_path = root / 'episodes.npz', root / 'config.json'
        config = dict(protocol=protocol, dataset=job['dataset'], shot=job['shot'], fold=0, pool='test',
                      count=700, quotas=list(QUOTAS), retained_queries=60,
                      sampling_seed=SAMPLING_SEED, sampling_stream=SAMPLING_STREAM,
                      source_manifest_sha256=identity['source_manifest_sha256'],
                      source_config_sha256=identity['source_config_sha256'],
                      point_indices_sha256=identity['point_indices_sha256'])
        if manifest.exists() or config_path.exists():
            if not manifest.is_file() or not config_path.is_file():
                raise ValueError(f'Incomplete imbalanced manifest: {root}')
            if json.loads(config_path.read_text()) != config:
                raise ValueError(f'Imbalanced manifest configuration mismatch: {root}')
            with np.load(manifest, allow_pickle=False) as saved:
                if any(key not in saved or len(saved[key]) != 700 for key in EPISODE_KEYS + ('query_quota',)):
                    raise ValueError('Incomplete imbalanced manifest arrays')
                converted = [Episode(**{key: saved[key][i].copy() for key in EPISODE_KEYS}) for i in range(700)]
                quotas = saved['query_quota'].copy()
            for original, changed, quota in zip(episodes, converted, quotas):
                validate_imbalanced_episode(original, changed, pool_data.labels, pool_data.rows, quota)
        else:
            pairs = [imbalanced_episode(ep, pool_data.labels, pool_data.rows, dataset=job['dataset'],
                                        shot=job['shot'], episode_id=i) for i, ep in enumerate(episodes)]
            converted, quotas = zip(*pairs)
            values = {key: np.stack([getattr(ep, key) for ep in converted]) for key in
                      EPISODE_KEYS + ('support_rows', 'query_rows')}
            values['query_quota'] = np.stack(quotas)
            temporary = manifest.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **values)
            from .train import write_json
            write_json(config_path, config)
            temporary.replace(manifest)
    return list(converted[:count]), indices, str(manifest), dict(identity,
        protocol=protocol, manifest_sha256=file_sha256(manifest), query_quotas=list(QUOTAS),
        retained_source_queries=60, sampling_seed=SAMPLING_SEED, sampling_stream=SAMPLING_STREAM)


def cached_file_sha256(path, cache_root):
    """Cache large immutable asset hashes by path and full filesystem identity."""
    path = Path(path).resolve()
    stat = path.stat()
    identity = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    ctime_ns=stat.st_ctime_ns, inode=stat.st_ino, device=stat.st_dev)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_root) / (key + '.json')
    if cache.is_file():
        value = json.loads(cache.read_text())
        if value.get('identity') != identity or len(value.get('sha256', '')) != 64:
            raise ValueError(f'Invalid asset hash cache: {cache}')
        return value['sha256']
    digest = file_sha256(path)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino, after.st_dev) != (
            stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev):
        raise ValueError(f'Asset changed while hashing: {path}')
    import tempfile
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=cache.parent, prefix=key + '-', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(dict(identity=identity, sha256=digest), handle, sort_keys=True)
    try:
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def scientific_code_identity():
    """Content hash of scientific execution code, independent of installation root."""
    root = Path(__file__).resolve().parent
    names = ['model.py', 'data.py', 'corruptions.py', 'euclidean.py', 'hyperbolic.py',
             'evaluate.py', 'train.py', 'empirical_sampling.py', 'review_evaluate.py',
             'review_diagnostics.py']
    names += [str(path.relative_to(root)) for path in sorted((root / 'vendor').rglob('*.py'))]
    files = {name: file_sha256(root / name) for name in sorted(names)}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(scientific_code_sha256=identity, scientific_code_files=files)
