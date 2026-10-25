"""Targeted protocol tests; tiny H5 fixtures are not released datasets."""
import importlib
import random

import numpy as np
import pytest


def test_splits_do_not_follow_train_seed():
    data = importlib.import_module('sac.data')
    for dataset, nbase, ntrain, nval in [('ModelNet40_FS', 30, 24, 6),
                                         ('ShapeNet70_FS', 50, 40, 10)]:
        labels = np.arange(nbase) * 3 + 100
        permutation = random.Random(7).sample(range(nbase), nbase)
        folds = []
        for fold in range(5):
            train, val = data.split_classes(labels, dataset, fold)
            assert len(train) == ntrain and len(val) == nval
            assert not set(train) & set(val)
            assert set(train) | set(val) == set(labels)
            assert set(val) == set(labels[permutation[fold*nval:(fold+1)*nval]])
            folds.append((train.copy(), val.copy()))
        for train_seed in (7, 17, 27):
            np.random.seed(train_seed)
            random.seed(train_seed)
            for fold, (train, val) in enumerate(folds):
                actual = data.split_classes(labels, dataset, fold)
                np.testing.assert_array_equal(actual[0], train)
                np.testing.assert_array_equal(actual[1], val)
    train, val = data.split_classes(np.arange(10), 'ScanObjectNN_FS', 2)
    np.testing.assert_array_equal(train, val)


def test_episode_roles():
    from sac import data
    labels = np.repeat([2, 7, 13, 21, 35, 50], 25)
    allowed = np.arange(25, len(labels))
    ep = data.sample_episode(labels, np.random.RandomState(27), shot=5, rows=allowed)
    assert len(ep.rows) == 100
    assert len(np.unique(ep.rows)) == 100
    assert set(ep.rows) <= set(allowed)
    np.testing.assert_array_equal(ep.support_rows, ep.rows[ep.support_indices])
    np.testing.assert_array_equal(ep.query_rows, ep.rows[ep.query_indices])
    assert not set(ep.support_rows) & set(ep.query_rows)
    np.testing.assert_array_equal(ep.support_y, np.repeat(ep.class_ids, 5))
    np.testing.assert_array_equal(ep.query_y, np.repeat(np.arange(5), 15))
    np.testing.assert_array_equal(labels[ep.query_rows], np.repeat(ep.class_ids, 15))
    # S/Q membership is selected AFTER the whole official sampler batch shuffle.
    for c in ep.class_ids:
        places = np.flatnonzero(labels[ep.rows] == c)
        np.testing.assert_array_equal(ep.support_indices[ep.support_y == c], places[:5])
    clouds = [np.full((32, 3), i, dtype=np.float32) for i in ep.rows]
    before = ep.model_inputs(clouds)
    ep.query_y[:] = (ep.query_y + 1) % 5
    after = ep.model_inputs(clouds)
    assert set(before) == {'clouds', 'support_indices', 'support_y', 'query_indices'}
    assert before['clouds'] is clouds and after['clouds'] is clouds
    for key in set(before) - {'clouds'}:
        np.testing.assert_array_equal(before[key], after[key])
    with pytest.raises(ValueError, match='samples'):
        data.sample_episode(labels, np.random.RandomState(7), shot=20)


def test_method_independent_streams():
    from sac import data
    labels = np.repeat(np.arange(10), 25)
    dataset = 'ModelNet40_FS'
    seed = data.stream_seed(100007, dataset, 0, 5, 20, 0)
    expected = int(np.random.SeedSequence([100007, 0, 0, 5, 20, 0]).generate_state(1)[0])
    assert seed == expected
    episode = data.sample_episode(labels, data.make_rng(100007, dataset, 0, 5, 20, 0), shot=5)
    for epoch in range(3):
        np.random.rand(91)
        data.sample_episode(labels, data.make_rng(7, dataset, 2, 5, 2, epoch), shot=5)
    repeated = data.sample_episode(labels, data.make_rng(100007, dataset, 4, 5, 20, 0), shot=5)
    np.testing.assert_array_equal(episode.rows, repeated.rows)
    assert data.stream_seed(7, dataset, 0, 5, 12, 0) == data.stream_seed(7, dataset, 4, 5, 12, 0)
    assert data.stream_seed(200007, dataset, 0, 5, 21, 0) != data.stream_seed(200007, dataset, 4, 5, 21, 0)
    assert data.stream_seed(100007, 'ScanObjectNN_FS', 0, 5, 20, 0) != data.stream_seed(100007, 'ScanObjectNN_FS', 2, 5, 20, 0)
    points = data.sample_point_indices(1024, data.make_rng(7, dataset, 0, 1, 10, 0))
    assert len(points) == len(np.unique(points)) == 512
    with pytest.raises(ValueError, match='points'):
        data.sample_point_indices(100, np.random.RandomState(7))
    a = data.corruption_seed(dataset, 0, 5, 'test', 2, 'query', 19, 'Drop-Local', 3)
    b = data.corruption_seed(dataset, 4, 5, 'test', 2, 'query', 19, 'Drop-Local', 3)
    assert a == b
    expected = int(np.random.SeedSequence([300007, 0, 0, 5, 1, 2, 1, 19, 3, 3]).generate_state(1)[0])
    assert a == expected
    assert a != data.corruption_seed(dataset, 0, 5, 'test', 2, 'support', 19, 'Drop-Local', 3)


@pytest.mark.parametrize('dataset,nbase,nnovel', [('ModelNet40_FS', 30, 10), ('ShapeNet70_FS', 50, 20), ('ScanObjectNN_FS', 10, 5)])
def test_official_h5_pools(tmp_path, dataset, nbase, nnovel):
    import h5py
    from sac import data
    directory = tmp_path / dataset
    directory.mkdir()
    for mode, nclass in [('train', nbase), ('test', nnovel)]:
        filename = (f'{mode}_scanobjectnn_PB_T50_RS_fsl_2.h5' if dataset == 'ScanObjectNN_FS'
                    else f'{"Modelnet40_FS" if dataset == "ModelNet40_FS" else dataset}_{mode}.h5')
        with h5py.File(directory / filename, 'w') as f:
            f['label'] = np.repeat(np.arange(nclass), 20).reshape(-1, 1)
            f['data'] = np.arange(nclass * 20 * 32 * 3, dtype=np.float32).reshape(-1, 32, 3)
    pools = data.load_pools(tmp_path, dataset, fold=1)
    assert set(pools) == {'train', 'val', 'test'}
    train, val, test = [pools[k] for k in ('train', 'val', 'test')]
    assert train.data is val.data
    assert train.labels is val.labels
    assert len(np.unique(test.labels[test.rows])) == nnovel
    if dataset == 'ScanObjectNN_FS':
        np.testing.assert_array_equal(train.rows, val.rows)
    else:
        assert not set(train.rows) & set(val.rows)
        assert len(set(train.rows) | set(val.rows)) == nbase * 20
    # Numeric label overlap is permitted across released-file namespaces.
    assert train.object_id(0) != test.object_id(0)
    assert train.object_id(0).endswith(':0')
    ep = data.sample_episode(train.labels, np.random.RandomState(7), rows=train.rows, shot=1)
    indices = data.sample_point_indices(32, np.random.RandomState(8), n_points=20)
    clouds = train.clouds(ep, indices)
    assert len(clouds) == 80
    for c, row in zip(clouds, ep.rows):
        np.testing.assert_array_equal(c, train.data[row, indices, :3])
    clouds[0][:] = -1
    assert not np.all(train.data[ep.rows[0], indices] == -1)
    with pytest.raises(FileNotFoundError):
        data.load_pools(tmp_path / 'missing', dataset, fold=1)


def test_augmentation_matches_official_and_does_not_touch_rng():
    from pathlib import Path
    from sac import data
    spec = importlib.util.spec_from_file_location('pcia_data_reference', Path(__file__).resolve().parents[1] / 'tests/fixtures/data_utils_reference.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    points = np.random.RandomState(1).randn(8, 32, 3).astype(np.float32)
    saved = points.copy()
    state = np.random.get_state()
    try:
        np.random.seed(7)
        expected = reference.jitter_point_cloud(reference.rotate_point_cloud(points)).astype(np.float32)
    finally:
        np.random.set_state(state)
    actual = data.augment(points, np.random.RandomState(7))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(points, saved)
    now = np.random.get_state()
    assert now[0] == state[0] and now[2:] == state[2:]
    np.testing.assert_array_equal(now[1], state[1])
    assert actual.dtype == np.float32


def test_prepare_lists_all_missing_files_and_checks_full_matrix(tmp_path):
    import h5py
    from sac import data
    with pytest.raises(FileNotFoundError) as error:
        data.prepare_data(tmp_path)
    message = str(error.value)
    assert 'Modelnet40_FS_train.h5' in message
    assert 'ShapeNet70_FS_test.h5' in message
    assert 'test_scanobjectnn_PB_T50_RS_fsl_3.h5' in message
    assert '18WTIClNOWMM9s6mjhwhfLBTNzINZLfRf' in message
    assert '1DiRlJ7dB7YrGuc90_2NcpNL7MK3eQBZh' in message
    for dataset, nbase, ntest, nfold in [('ModelNet40_FS', 30, 10, 5), ('ShapeNet70_FS', 50, 20, 5), ('ScanObjectNN_FS', 10, 5, 3)]:
        for fold in range(nfold):
            for mode, path in data.official_paths(tmp_path, dataset, fold).items():
                path.parent.mkdir(exist_ok=True)
                n = nbase if mode == 'train' else ntest
                with h5py.File(path, 'w') as f:
                    f['data'] = np.zeros((n, 512, 3), dtype=np.float32)
                    f['label'] = np.arange(n).reshape(-1, 1)
    result = data.prepare_data(tmp_path)
    assert result['complete'] is True
    assert result['fold_shot_settings'] == 26
    assert len(result['files']) == 10
    assert {v['dataset'] for v in result['files']} == set(data.DATASET_IDS)
