"""Protocol properties on artificial objects, independent of experimental results."""
from pathlib import Path
import json

import numpy as np
import pytest

from sac.data import H5Pool, sample_episode
from sac.empirical_sampling import (
    EPISODE_KEYS, empirical_episodes, file_sha256, imbalanced_episode, protocol_root)
from sac.evaluate import damage_episode, shared_episodes


@pytest.fixture
def synthetic_pool(tmp_path):
    labels = np.repeat(np.arange(10), 50)
    data = np.random.RandomState(91).normal(size=(len(labels), 512, 3)).astype(np.float32)
    path = tmp_path / 'artificial.h5'
    path.write_bytes(b'artificial-pool-hash-fixture')
    return H5Pool(path, data, labels, np.arange(len(labels)))


@pytest.mark.parametrize('shot', [1, 5])
def test_fixed_quota_support_shared_objects_and_global_rng_independence(synthetic_pool, shot):
    pool = synthetic_pool
    source = sample_episode(pool.labels, np.random.RandomState(7), shot=shot)
    before = np.random.get_state()
    episode, quota = imbalanced_episode(source, pool.labels, pool.rows,
        dataset='ModelNet40', shot=shot, episode_id=11)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert sorted(quota.tolist()) == [5, 10, 15, 20, 25]
    assert np.array_equal(np.bincount(episode.query_y), quota)
    assert len(np.unique(episode.rows)) == 5 * shot + 75
    assert len(np.intersect1d(source.query_rows, episode.query_rows)) == 60
    assert np.array_equal(source.support_indices, episode.support_indices)
    assert np.array_equal(source.support_rows, episode.support_rows)
    assert np.array_equal(source.class_ids, episode.class_ids)
    assert not np.intersect1d(source.support_rows, episode.query_rows).size
    for local_class, count in enumerate(quota):
        original = source.query_rows[source.query_y == local_class]
        changed = episode.query_rows[episode.query_y == local_class]
        assert len(np.intersect1d(original, changed)) == min(int(count), 15)
    repeat, repeat_quota = imbalanced_episode(source, pool.labels, pool.rows,
        dataset='ModelNet40', shot=shot, episode_id=11)
    assert np.array_equal(repeat.rows, episode.rows)
    assert np.array_equal(repeat_quota, quota)


@pytest.mark.parametrize('side', ['query', 'support', 'both'])
@pytest.mark.parametrize('operator', ['Jitter', 'Drop-Local', 'Add-Global'])
def test_shared_objects_preserve_damage_for_every_role(synthetic_pool, side, operator):
    pool = synthetic_pool
    source = sample_episode(pool.labels, np.random.RandomState(7), shot=5)
    changed, _ = imbalanced_episode(source, pool.labels, pool.rows,
        dataset='ShapeNet70', shot=5, episode_id=7)
    points = np.arange(512)
    damage = dict(dataset='ShapeNet70', shot=5, pool='test', episode_id=7,
                  side=side, operator=operator, severity=3)
    a, seeds_a = damage_episode(pool.clouds(source, points), source, fold=0, **damage)
    b, seeds_b = damage_episode(pool.clouds(changed, points), changed, fold=4, **damage)
    shared = np.intersect1d(source.rows, changed.rows)
    assert len(shared) == 85
    for row in shared:
        left = np.flatnonzero(source.rows == row)[0]
        right = np.flatnonzero(changed.rows == row)[0]
        assert left == right and seeds_a[left] == seeds_b[right]
        assert np.array_equal(a[left], b[right])


def test_manifests_shared_across_folds_seeds_smoke_and_source_readonly(synthetic_pool, tmp_path):
    job = dict(dataset='ModelNet40', fold=0, shot=1, seed=7, family='P0')
    source, output = tmp_path / 'source', tmp_path / 'out'
    old, points, source_path = shared_episodes(source, synthetic_pool, job, 'test', 700)
    snapshot = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in source.rglob('*') if path.is_file()}
    a, p, manifest, identity = empirical_episodes(output, source, synthetic_pool, job, 'imbalanced', 2)
    b, q, second_manifest, identity_b = empirical_episodes(output, source, synthetic_pool,
        dict(job, fold=4, seed=27), 'imbalanced', 700)
    assert manifest == second_manifest and identity == identity_b
    assert np.array_equal(a[1].rows, b[1].rows) and len(b) == 700
    assert np.array_equal(p, q) and np.array_equal(p, points)
    _, _, replicated, replicated_identity = empirical_episodes(tmp_path / 'independent-host-output',
        source, synthetic_pool, dict(job, fold=3, seed=17), 'imbalanced', 2)
    assert identity['manifest_sha256'] == replicated_identity['manifest_sha256']
    assert Path(manifest).read_bytes() == Path(replicated).read_bytes()
    assert identity['manifest_sha256'] == file_sha256(manifest)
    assert identity['source_manifest_sha256'] == file_sha256(source_path)
    balanced, _, balanced_path, balanced_id = empirical_episodes(output, source, synthetic_pool,
        job, 'balanced', 2)
    assert balanced_path == source_path and balanced_id['manifest_sha256'] == file_sha256(source_path)
    assert np.array_equal(balanced[1].rows, old[1].rows)
    for path, initial in snapshot.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == initial
    config_path = Path(manifest).with_name('config.json')
    config = json.loads(config_path.read_text())
    config['source_manifest_sha256'] = '0' * 64
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='mismatch'):
        empirical_episodes(output, source, synthetic_pool, job, 'imbalanced', 2)


def test_missing_source_and_insufficient_objects_fail_without_replacement(synthetic_pool, tmp_path):
    job = dict(dataset='ModelNet40', fold=0, shot=5, seed=7, family='P0')
    with pytest.raises(FileNotFoundError, match='read-only'):
        empirical_episodes(tmp_path / 'out', tmp_path / 'missing', synthetic_pool, job, 'imbalanced', 1)
    labels = np.repeat(np.arange(5), 20)
    source = sample_episode(labels, np.random.RandomState(1), shot=5)
    with pytest.raises(ValueError, match='Insufficient'):
        imbalanced_episode(source, labels, np.arange(len(labels)), dataset='ModelNet40', shot=5, episode_id=0)
