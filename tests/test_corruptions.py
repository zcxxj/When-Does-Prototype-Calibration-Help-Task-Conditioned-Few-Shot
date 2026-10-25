import importlib

import numpy as np
import pytest

OPERATORS = ('Scale', 'Jitter', 'Drop-Global', 'Drop-Local', 'Add-Global', 'Add-Local', 'Rotate')
COUNTS = {'Scale': [512]*5, 'Jitter': [512]*5, 'Rotate': [512]*5,
          'Drop-Global': [384, 320, 256, 192, 128],
          'Drop-Local': [462, 412, 362, 312, 262],
          'Add-Global': [517, 522, 527, 532, 537],
          'Add-Local': [562, 612, 662, 712, 762]}


@pytest.mark.parametrize('operator', OPERATORS)
@pytest.mark.parametrize('severity', range(1, 6))
def test_counts_and_immutability(operator, severity):
    module = importlib.import_module('sac.corruptions')
    points = np.random.RandomState(3).uniform(-.5, .5, (512, 3)).astype(np.float32)
    saved = points.copy()
    result = module.corrupt(points, operator, severity, np.random.RandomState(17))
    assert result.shape == (COUNTS[operator][severity-1], 3)
    assert result.dtype == np.float32 and np.isfinite(result).all()
    assert not np.shares_memory(result, points)
    np.testing.assert_array_equal(points, saved)
    np.testing.assert_array_equal(result, module.corrupt(points, operator, severity, np.random.RandomState(17)))
    if operator == 'Add-Global':
        np.testing.assert_array_equal(result[:512], points)
    elif operator.startswith('Drop-'):
        assert all(any(np.array_equal(p, q) for q in points) for p in result)


def test_half_up_and_unsupported_inputs():
    from sac.corruptions import corrupt
    points = np.random.RandomState(2).uniform(-.5, .5, (256, 3)).astype(np.float32)
    assert len(corrupt(points, 'Add-Global', 1, np.random.RandomState(0))) == 259
    with pytest.raises(ValueError, match='operator'):
        corrupt(points, 'LiDAR', 1, np.random.RandomState(0))
    for severity in [0, 6, 1.5]:
        with pytest.raises(ValueError, match='severity'):
            corrupt(points, 'Jitter', severity, np.random.RandomState(0))
    with pytest.raises(ValueError, match='20'):
        corrupt(points[:19], 'Jitter', 1, np.random.RandomState(0))


@pytest.mark.parametrize('operator', OPERATORS)
@pytest.mark.parametrize('severity', range(1, 6))
def test_reference_at_1024(operator, severity):
    from sac.corruptions import corrupt
    reference = importlib.import_module('sac.vendor.corrupt_utils_reference')
    names = {'Scale': 'scale', 'Jitter': 'jitter', 'Drop-Global': 'dropout_global',
             'Drop-Local': 'dropout_local', 'Add-Global': 'add_global',
             'Add-Local': 'add_local', 'Rotate': 'rotate'}
    points = np.random.RandomState(71).uniform(-1, 1, (1024, 3)).astype(np.float32)
    for seed in (0, 7, 27):
        original_state = np.random.get_state()
        try:
            np.random.seed(seed)
            expected = getattr(reference, 'corrupt_' + names[operator])(points.copy(), severity - 1)
            reference_next = np.random.rand(10)
        finally:
            np.random.set_state(original_state)
        rng = np.random.RandomState(seed)
        actual = corrupt(points, operator, severity, rng)
        np.testing.assert_array_equal(actual, expected.astype(np.float32))
        np.testing.assert_array_equal(rng.rand(10), reference_next)
        state = np.random.get_state()
        assert state[0] == original_state[0] and state[2:] == original_state[2:]
        np.testing.assert_array_equal(state[1], original_state[1])
