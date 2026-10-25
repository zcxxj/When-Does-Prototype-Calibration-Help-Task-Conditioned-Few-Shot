"""Seven ModelNet-C operators, adapted to natural variable point counts.

Source: jiawei-ren/ModelNet-C, commit 0c82ce0c21165530dcc2c011450767277e92fa96,
build_modelnetc/corrupt_utils.py. Only explicit RandomState, count scaling,
float32 output and input validation differ. Public severity is 1..5; original
function level is severity-1. Counts use integer half-up relative to 1024.
No resampling, padding, clipping of jitter, or extra normalization is applied.
Add-Local deliberately retains the original division by SQUARED radius.
"""
import math

import numpy as np

OPERATORS = ('Scale', 'Jitter', 'Drop-Global', 'Drop-Local', 'Add-Global', 'Add-Local', 'Rotate')


def _shuffle(points, rng):
    idx = rng.rand(points.shape[0], 1).argsort(axis=0)
    return np.take_along_axis(points, idx, axis=0)


def _cluster_sizes(num_clusters, total, rng):
    assignments = rng.randint(num_clusters, size=total)
    return [sum(assignments == i) for i in range(num_clusters)]


def _sphere(npoints, rng):
    radius = np.power(rng.uniform(0.0, 1.0, (npoints, 1)), 1 / 3)
    theta = np.arccos(rng.uniform(-1.0, 1.0, (npoints, 1)))
    phi = rng.uniform(0, 2 * np.pi, (npoints, 1))
    return np.concatenate([radius * np.sin(theta) * np.cos(phi),
                           radius * np.sin(theta) * np.sin(phi),
                           radius * np.cos(theta)], axis=1)


def corrupt(points, operator, severity, rng):
    """Return fresh float32 [N_out,3]; accepts explicit np.random.RandomState.

    operator is one of OPERATORS (exact report names), severity is 1..5.
    Every call starts from the supplied original observation; do not chain them.
    This function never seeds or consumes the NumPy global RNG.
    """
    if operator not in OPERATORS:
        raise ValueError(f'Unsupported operator: {operator}')
    if not isinstance(severity, (int, np.integer)) or not 1 <= severity <= 5:
        raise ValueError('severity must be an integer in 1..5')
    pointcloud = np.array(points, dtype=np.float32, copy=True)
    if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
        raise ValueError('points must have shape [N,3]')
    if len(pointcloud) < 20:
        raise ValueError('N < 20 is unsupported; no padding or lowered k')
    if not np.isfinite(pointcloud).all():
        raise ValueError('points must be finite')
    level = severity - 1
    num_points = len(pointcloud)
    # Exact rational half-up, not banker's round.
    count = lambda reference: (reference * num_points * 2 + 1024) // 2048
    if operator == 'Scale':
        s = [1.6, 1.7, 1.8, 1.9, 2.0][level]
        xyz = rng.uniform(low=1. / s, high=s, size=[3])
        pointcloud = np.multiply(pointcloud, xyz).astype('float32')
        pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        radius = np.max(np.sqrt(np.sum(pointcloud ** 2, axis=1)))
        if radius == 0:
            raise ValueError('Scale cannot normalize a zero-radius cloud')
        pointcloud = pointcloud / radius
    elif operator == 'Jitter':
        pointcloud = pointcloud + 0.01 * severity * rng.randn(*pointcloud.shape)
    elif operator == 'Rotate':
        bound = math.pi / 6 / 5 * severity
        angles = rng.uniform(-bound, bound, size=3)
        Rx = np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])
        pointcloud = np.dot(pointcloud, np.dot(Rz, np.dot(Ry, Rx)))
    elif operator == 'Drop-Global':
        drop_rate = [0.25, 0.375, 0.5, 0.625, 0.75][level]
        pointcloud = _shuffle(pointcloud, rng)[:int(num_points * (1 - drop_rate))]
    elif operator == 'Drop-Local':
        total = count(100 * severity)
        nclusters = rng.randint(1, 8)
        for K in _cluster_sizes(nclusters, total, rng):
            pointcloud = _shuffle(pointcloud, rng)
            dist = np.sum((pointcloud - pointcloud[:1]) ** 2, axis=1, keepdims=True)
            idx = dist.argsort(axis=0)[::-1, :]
            pointcloud = np.take_along_axis(pointcloud, idx, axis=0)
            num_points -= K
            pointcloud = pointcloud[:num_points]
    elif operator == 'Add-Global':
        additional = _sphere(count(10 * severity), rng)
        pointcloud = np.concatenate([pointcloud, additional], axis=0)
    else:  # Add-Local: preserve original random call and float32 assignment order.
        total = count(100 * severity)
        nclusters = rng.randint(1, 8)
        sizes = _cluster_sizes(nclusters, total, rng)
        pointcloud = _shuffle(pointcloud, rng)
        additional = np.zeros_like(pointcloud)
        offset = 0
        for i, K in enumerate(sizes):
            sigma = rng.uniform(0.075, 0.125)
            additional[offset:offset + K] = np.copy(pointcloud[i:i + 1])
            additional[offset:offset + K] = (additional[offset:offset + K]
                + sigma * rng.randn(*additional[offset:offset + K].shape))
            offset += K
        dist = np.sum(additional ** 2, axis=1, keepdims=True).repeat(3, axis=1)
        additional[dist > 1] = additional[dist > 1] / dist[dist > 1]
        pointcloud = np.concatenate([pointcloud, additional], axis=0)[:num_points + total]
    if len(pointcloud) < 20:
        raise ValueError('Result N < 20 is unsupported; no padding or lowered k')
    return pointcloud.astype(np.float32, copy=False)
