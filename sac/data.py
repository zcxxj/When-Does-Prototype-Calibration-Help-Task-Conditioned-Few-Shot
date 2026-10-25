"""PCIA released-H5 protocol with independent explicit NumPy RNG streams.

Source: Chuangguan Ye's PCIA dataloader.py / utils/data_utils.py,
commit 8345631cd92987198eabf523eba3ce657597fd37. No data downloading.

Runner API (all datasets use exact *_FS names):
    pools = load_pools(data_root, dataset, fold)  # train/val/test H5Pool
    pool = pools['train']
    idx = sample_point_indices(pool.data.shape[1], make_rng(7, dataset, fold, shot, 10))
    ep = sample_episode(pool.labels, make_rng(seed, dataset, fold, shot, 2, epoch),
                        shot=shot, rows=pool.rows)
    clouds = pool.clouds(ep, idx)  # list[np.ndarray[N,3]], shuffled batch order
    batch = augment(np.stack(clouds), make_rng(seed, dataset, fold, shot, 3, epoch))
    # Convert batch/clouds and ep.model_inputs(clouds) to the model's device.
    # query_y stays OUTSIDE the model; it contains local targets 0..way-1.

Streams: init=1, train episode/augmentation=2/3, val=4/5;
point indices train/val/test=10/11/12 (seed7); fixed test=20 (seed100007),
fixed dev=21 (seed200007). For synthetic test streams and damage, fold is
internally forced to zero; dev and Scan retain actual fold. Save idx and
Episode fields as manifests in the runner. prepare_data checks asset metadata.
load_pools reads test values too: train/smoke should open only official_paths
(... )['train'] and construct H5Pool, as done by train.base_pools.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET_IDS = {'ModelNet40_FS': 0, 'ShapeNet70_FS': 1, 'ScanObjectNN_FS': 2}
OPERATORS = ('Scale', 'Jitter', 'Drop-Global', 'Drop-Local', 'Add-Global', 'Add-Local', 'Rotate')


def stream_seed(seed, dataset, fold, shot, stream, epoch=0):
    """32-bit SeedSequence seed; synthetic test streams 12/20 share fold=0."""
    if dataset != 'ScanObjectNN_FS' and stream in (12, 20):
        fold = 0
    values = [seed, DATASET_IDS[dataset], fold, shot, stream, epoch]
    return int(np.random.SeedSequence(values).generate_state(1)[0])


def make_rng(seed, dataset, fold, shot, stream, epoch=0):
    """Return a fresh local RandomState; no global NumPy/Torch state use."""
    return np.random.RandomState(stream_seed(seed, dataset, fold, shot, stream, epoch))


def corruption_seed(dataset, fold, shot, pool, episode_id, side, row_id, operator, severity):
    """Stable damage identity, independent of method/checkpoint/scenario/seed.

    pool: dev/test; side: support/query; severity: REPORT number 1..5.
    """
    if not 1 <= severity <= 5:
        raise ValueError('severity must be 1..5')
    if dataset != 'ScanObjectNN_FS' and pool == 'test':
        fold = 0
    values = [300007, DATASET_IDS[dataset], fold, shot, {'dev': 0, 'test': 1}[pool],
              episode_id, {'support': 0, 'query': 1}[side], row_id,
              OPERATORS.index(operator), severity]
    return int(np.random.SeedSequence(values).generate_state(1)[0])


def sample_point_indices(n_available, rng, n_points=512):
    """One no-replacement index vector reused for EVERY object in a pool."""
    if not 0 < n_points <= n_available:
        raise ValueError('Not enough points for no-replacement sampling')
    return rng.choice(n_available, n_points, replace=False)


@dataclass
class Episode:
    """H5 row manifest in original shuffled batch order; labels remain external.

    support_y are original numeric IDs (sorted class-major); query_y are local
    targets 0..way-1, aligned with query_indices. No semantic names are guessed.
    """
    rows: np.ndarray
    class_ids: np.ndarray
    support_indices: np.ndarray
    query_indices: np.ndarray
    support_y: np.ndarray
    query_y: np.ndarray

    @property
    def support_rows(self):
        return self.rows[self.support_indices]

    @property
    def query_rows(self):
        return self.rows[self.query_indices]

    def model_inputs(self, clouds):
        """clouds must already be in rows order; never include query truth."""
        if len(clouds) != len(self.rows):
            raise ValueError('clouds must have one entry per episode row')
        return dict(clouds=clouds, support_indices=self.support_indices,
                    support_y=self.support_y, query_indices=self.query_indices)


def sample_episode(labels, rng, *, shot, way=5, query=15, rows=None):
    """Official class/object choices + whole-batch shuffle, using explicit RNG.

    labels index the whole H5; optional rows restrict the allowed H5 row pool.
    S/Q are the first shot / remaining query class occurrences AFTER shuffle.
    """
    labels = np.asarray(labels).reshape(-1)
    rows = np.arange(len(labels)) if rows is None else np.asarray(rows, dtype=np.int64)
    if shot < 1 or query < 1 or way < 1:
        raise ValueError('shot, query and way must be positive')
    classes = np.unique(labels[rows])
    if len(classes) < way:
        raise ValueError('Not enough classes for episode')
    if any(np.count_nonzero(labels[rows] == c) < shot + query for c in classes):
        raise ValueError('Not enough samples per class for episode')
    selected = classes[rng.choice(len(classes), way, replace=False)]
    batch = np.concatenate([rng.choice(rows[labels[rows] == c], shot + query,
                                       replace=False) for c in selected])
    batch = batch[rng.choice(len(batch), len(batch), replace=False)]
    class_ids = np.sort(selected)
    locations = [np.flatnonzero(labels[batch] == c) for c in class_ids]
    support = np.concatenate([p[:shot] for p in locations])
    queries = np.concatenate([p[shot:] for p in locations])
    return Episode(batch, class_ids, support, queries, labels[batch[support]].copy(),
                   np.repeat(np.arange(way), query))


def split_classes(labels, dataset: str, fold: int):
    """Return sorted (train, validation) H5 class IDs; split seed is always 7.

    Scan's training and validation pools intentionally share objects/classes.
    Synthetic test labels have a separate namespace; never pass them here.
    """
    classes = np.unique(np.asarray(labels).reshape(-1))
    if dataset not in DATASET_IDS:
        raise ValueError(f'Unsupported dataset: {dataset}')
    expected = {'ModelNet40_FS': 30, 'ShapeNet70_FS': 50, 'ScanObjectNN_FS': 10}[dataset]
    if len(classes) != expected:
        raise ValueError(f'{dataset} needs {expected} base classes, found {len(classes)}')
    nfold = 3 if dataset == 'ScanObjectNN_FS' else 5
    if not 0 <= fold < nfold:
        raise ValueError(f'fold must be in range({nfold})')
    if dataset == 'ScanObjectNN_FS':
        return classes.copy(), classes.copy()
    order = random.Random(7).sample(range(expected), expected)
    width = expected // 5
    val = np.sort(classes[order[fold * width:(fold + 1) * width]])
    return np.setdiff1d(classes, val), val


@dataclass
class H5Pool:
    """Full H5 arrays plus allowed ORIGINAL row IDs (not renumbered slices)."""
    path: Path
    data: np.ndarray
    labels: np.ndarray
    rows: np.ndarray

    def object_id(self, row):
        return f'{self.path.name}:{int(row)}'

    def clouds(self, episode, point_indices):
        """Fresh float32 xyz arrays in sampler order, with shared point indices."""
        return [self.data[row, point_indices, :3].astype(np.float32, copy=True)
                for row in episode.rows]


def augment(batch_data, rng):
    """Official batch Y rotation then sigma=.01/clip=.05 jitter, float32.

    Pass a [B,N,3] array in sampler order once per train/standard-val episode.
    Do NOT call for clean test or fixed unaugmented corruption-development pools.
    """
    batch_data = np.asarray(batch_data)
    if batch_data.ndim != 3 or batch_data.shape[-1] != 3:
        raise ValueError('augment requires [B,N,3]')
    rotated = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(len(batch_data)):
        angle = rng.uniform() * 2 * np.pi
        cosval, sinval = np.cos(angle), np.sin(angle)
        matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
        rotated[k] = np.dot(batch_data[k], matrix)
    jitter = np.clip(0.01 * rng.randn(*rotated.shape), -0.05, 0.05)
    jitter += rotated
    return jitter.astype(np.float32)


def official_paths(data_root, dataset, fold=0):
    """Exact published filenames under data_root/{dataset}, no discovery/download."""
    from pathlib import Path
    if dataset not in DATASET_IDS:
        raise ValueError(f'Unsupported dataset: {dataset}')
    nfold = 3 if dataset == 'ScanObjectNN_FS' else 5
    if not 0 <= fold < nfold:
        raise ValueError(f'fold must be in range({nfold})')
    root = Path(data_root) / dataset
    if dataset == 'ScanObjectNN_FS':
        return {mode: root / f'{mode}_scanobjectnn_PB_T50_RS_fsl_{fold + 1}.h5'
                for mode in ('train', 'test')}
    # The published ModelNet filename has lowercase n, unlike its dataset ID.
    prefix = 'Modelnet40_FS' if dataset == 'ModelNet40_FS' else dataset
    return {mode: root / f'{prefix}_{mode}.h5' for mode in ('train', 'test')}


def load_pools(data_root, dataset, fold=0):
    """Load official train/val/test pools; retain file-local label namespaces.

    Base file is loaded once, shared by train/val. Synthetic validation uses
    fixed class folds; Scan validation intentionally reuses its training pool.
    Test is the full published novel H5. Shapes/classes are checked, not guessed.
    """
    import h5py
    paths = official_paths(data_root, dataset, fold)
    arrays = {}
    for mode, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing official PCIA H5: {path}')
        with h5py.File(path, 'r') as handle:
            points = np.asarray(handle['data'], dtype=np.float32)
            labels = np.asarray(handle['label']).reshape(-1)
        if points.ndim != 3 or points.shape[-1] < 3 or len(points) != len(labels):
            raise ValueError(f'Invalid data/label shapes in {path}')
        if not np.issubdtype(labels.dtype, np.integer):
            if not np.all(labels == labels.astype(np.int64)):
                raise ValueError(f'Non-integer class labels in {path}')
        labels = labels.astype(np.int64, copy=False)
        arrays[mode] = (points[:, :, :3], labels)
    base_points, base_labels = arrays['train']
    train_classes, val_classes = split_classes(base_labels, dataset, fold)
    test_points, test_labels = arrays['test']
    expected = {'ModelNet40_FS': 10, 'ShapeNet70_FS': 20, 'ScanObjectNN_FS': 5}[dataset]
    if len(np.unique(test_labels)) != expected:
        raise ValueError(f'{dataset} needs {expected} novel classes')
    return {
        'train': H5Pool(paths['train'], base_points, base_labels,
                        np.flatnonzero(np.isin(base_labels, train_classes))),
        'val': H5Pool(paths['train'], base_points, base_labels,
                      np.flatnonzero(np.isin(base_labels, val_classes))),
        'test': H5Pool(paths['test'], test_points, test_labels, np.arange(len(test_labels))),
    }


def prepare_data(data_root):
    """Check 10 published H5 files and return JSON-ready metadata; never download.

    Only labels and dataset shapes are read, not novel point values or episodes.
    Missing assets are reported together with official dataset source URLs.
    """
    import h5py
    sources = {
        'ModelNet40_FS': 'https://drive.google.com/drive/folders/18WTIClNOWMM9s6mjhwhfLBTNzINZLfRf',
        'ShapeNet70_FS': 'https://drive.google.com/drive/folders/1DiRlJ7dB7YrGuc90_2NcpNL7MK3eQBZh',
        'ScanObjectNN_FS': 'https://drive.google.com/drive/folders/1As3Q0-NPDwJn_9xHviftJIQyhrFcHWrh',
    }
    entries = []
    for dataset in DATASET_IDS:
        for fold in range(3 if dataset == 'ScanObjectNN_FS' else 1):
            for mode, path in official_paths(data_root, dataset, fold).items():
                entries.append((dataset, mode, path))
    missing = [(dataset, path) for dataset, _, path in entries if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing official PCIA H5 files:\n' + '\n'.join(
            f'{path}\n  source: {sources[dataset]}' for dataset, path in missing))
    files = []
    expected = {'ModelNet40_FS': (30, 10), 'ShapeNet70_FS': (50, 20), 'ScanObjectNN_FS': (10, 5)}
    for dataset, mode, path in entries:
        with h5py.File(path, 'r') as handle:
            shape = handle['data'].shape
            labels = np.asarray(handle['label']).reshape(-1)
        if len(shape) != 3 or shape[2] < 3 or shape[1] < 512 or shape[0] != len(labels):
            raise ValueError(f'Invalid official data shape: {path}: {shape}')
        nclass = len(np.unique(labels))
        if nclass != expected[dataset][0 if mode == 'train' else 1]:
            raise ValueError(f'Incorrect released class count: {path}: {nclass}')
        files.append(dict(dataset=dataset, split=mode, path=str(path), shape=list(shape),
                          classes=nclass, source=sources[dataset]))
    return dict(complete=True, fold_shot_settings=26, files=files)
