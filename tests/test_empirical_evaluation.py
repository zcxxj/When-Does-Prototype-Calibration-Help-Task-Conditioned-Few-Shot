"""Acceptance invariants, not synthetic evidence for reported performance."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from sac import review_evaluate as module
from sac.review_diagnostics import extended_score_diagnostics
from test_empirical_sampling import synthetic_pool
from test_review_evaluation import FeatureModel, args_for


def freeze(out):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'freeze.json').write_text(json.dumps(dict(version='empirical-calibration-v1',
        protocols=['balanced', 'imbalanced'], episodes=700, training_seeds=[7, 17, 27])))


def empirical_args(tmp_path, **kwargs):
    args = args_for(tmp_path, pool='test', shot=1, empirical=True, protocol='balanced',
                    checkpoint_out=tmp_path / 'checkpoint', manifest_out=tmp_path / 'manifests',
                    methods=['P0', 'E2', 'QONLY', 'SOFTK'], **kwargs)
    freeze(args.out)
    return args


@pytest.mark.parametrize('shot,protocol,side,operator,severity,seed', [
    (1, 'balanced', 'query', 'Scale', 1, 17),
    (1, 'balanced', 'both', 'Drop-Local', 3, 27),
    (5, 'balanced', 'query', 'Rotate', 3, 27),
    (1, 'imbalanced', 'query', 'Add-Global', 3, 7),
    (5, 'imbalanced', 'clean', None, 0, 17)])
def test_new_scope_accepts_all_frozen_settings(tmp_path, shot, protocol, side, operator, severity, seed):
    args = empirical_args(tmp_path)
    for key, value in dict(shot=shot, protocol=protocol, side=side, operator=operator,
                           severity=severity, seed=seed, episodes=2).items():
        setattr(args, key, value)
    assert module.validate_job(args)[3] == 2


@pytest.mark.parametrize('updates', [dict(methods=['UNIT']), dict(dataset='ScanObjectNN'),
    dict(protocol='imbalanced', side='both', operator='Drop-Local', severity=3),
    dict(protocol='imbalanced', side='query', operator='Scale', severity=1),
    dict(shot=5, side='query', operator='Rotate', severity=5), dict(pool='dev'),
    dict(checkpoint='last'), dict(family='H', methods=['H']), dict(softk_dev=True)])
def test_empirical_scope_rejects_unfrozen_cells(tmp_path, updates):
    args = empirical_args(tmp_path)
    for key, value in updates.items():
        setattr(args, key, value)
    with pytest.raises(ValueError):
        module.validate_job(args)


@pytest.mark.parametrize('counts', [[15] * 5, [5, 10, 15, 20, 25]])
def test_flip_contribution_exact_for_both_denominators(counts):
    target = np.repeat(np.arange(5), counts)
    before = target.copy()
    after = target.copy()
    before[[0, 1, 45]] = (before[[0, 1, 45]] + 1) % 5
    after[[2, 3, 4, 70]] = (after[[2, 3, 4, 70]] + 2) % 5
    logits = lambda pred: np.eye(5)[pred] * 10
    values = extended_score_diagnostics(dict(logits_before=logits(before), logits_after=logits(after),
        group_size=np.bincount(before, minlength=5), anchor_fraction=np.ones(5),
        displacement=np.zeros(5), soft_mass=np.full(5, np.nan)), target)
    assert values['corrected_micro'] - values['damaged_micro'] == pytest.approx(
        np.mean(after == target) - np.mean(before == target))
    assert values['corrected_macro'] - values['damaged_macro'] == pytest.approx(
        values['final_macro_accuracy'] - values['initial_macro_accuracy'])
    if len(set(counts)) == 1:
        assert values['initial_macro_accuracy'] == pytest.approx(values['initial_accuracy'])
        assert values['final_macro_accuracy'] == pytest.approx(values['final_micro_accuracy'])
    assert values['soft_entropy'].shape == values['soft_max_probability'].shape == (75,)
    assert np.all(values['soft_entropy'] >= 0) and np.all(values['soft_max_probability'] <= 1)


def test_extreme_logit_entropy_and_empty_groups_are_valid():
    target = np.repeat(np.arange(5), [5, 10, 15, 20, 25])
    logits = np.tile([10000, -10000, -20000, -30000, -40000], (75, 1))
    scored = extended_score_diagnostics(dict(logits_before=logits, logits_after=logits,
        group_size=np.array([75, 0, 0, 0, 0]), anchor_fraction=np.ones(5),
        displacement=np.zeros(5), soft_mass=np.array([75., 0., 0., 0., 0.])), target)
    assert np.isnan(scored['group_class_accuracy'][1:]).all()
    assert scored['group_class_accuracy'][0] == 5 / 75
    assert np.isfinite(scored['soft_entropy']).all()
    assert np.all(scored['soft_entropy'] == 0) and np.all(scored['soft_max_probability'] == 1)


@pytest.mark.parametrize('protocol', ['balanced', 'imbalanced'])
def test_empirical_job_single_forward_readonly_roots_identity_and_resume(tmp_path, synthetic_pool, monkeypatch, protocol):
    from sac import data
    from sac.campaign import run_dir
    from sac.evaluate import shared_episodes
    args = empirical_args(tmp_path, episodes=2)
    args.protocol = protocol
    job = dict(dataset='ModelNet40', fold=0, shot=1, seed=7, family='P0')
    shared_episodes(args.manifest_out, synthetic_pool, job, 'test', 700)
    run = run_dir(args.source_out, job)
    run.mkdir(parents=True, exist_ok=True)
    config = dict(job, smoke=False)
    (run / 'config.json').write_text(json.dumps(config))
    (run / 'completed.json').write_text('{}')
    (run / 'best.pt').write_bytes(b'artificial-checkpoint')
    models = []
    def factory(**kwargs):
        model = FeatureModel(**kwargs)
        models.append(model)
        return model
    monkeypatch.setattr(module, 'EpisodeModel', factory)
    monkeypatch.setattr(module, 'load_checkpoint', lambda *a, **k: dict(config=config, epoch=12))
    monkeypatch.setattr(data, 'load_pools', lambda *a: {'test': synthetic_pool})
    readonly = {path: (path.read_bytes(), path.stat().st_mtime_ns) for root in
                (args.source_out, args.manifest_out) for path in root.rglob('*') if path.is_file()}
    result = module.evaluate_job(args)
    assert models[-1].calls == 2 and set(result['diagnostics']) == set(args.methods)
    assert 'UNIT' not in result['predictions'] and 'UNIT' not in result['diagnostics']
    for path, identity in readonly.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == identity
    for name, filename in result['predictions'].items():
        path = Path(filename)
        assert f'/protocols/{protocol}/' in str(path)
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['protocol'] == protocol and len(meta['data_sha256']) == 64
        assert len(meta['checkpoint_sha256']) == 64
        with np.load(path) as saved:
            assert saved['macro_accuracy'].shape == (2,)
            assert saved['micro_accuracy'].shape == (2,)
            np.testing.assert_array_equal(saved['episode_accuracy'], saved['micro_accuracy'])
            assert sorted(saved['query_quota'][0]) == ([15] * 5 if protocol == 'balanced' else [5,10,15,20,25])
    assert module.evaluate_job(args)['skipped'] and models[-1].calls == 0
    path = Path(result['predictions']['P0'])
    meta = json.loads(path.with_suffix('.json').read_text())
    original_metadata = dict(meta)
    meta['protocol'] = 'imbalanced' if protocol == 'balanced' else 'balanced'
    path.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='protocol mismatch'):
        module.evaluate_job(args)

    path.with_suffix('.json').write_text(json.dumps(original_metadata))
    with np.load(path) as saved:
        original_values = {key: saved[key].copy() for key in saved.files}
    for field in ('logits_after', 'pred', 'macro_accuracy', 'micro_accuracy', 'episode_id', 'query_quota'):
        values = {key: value.copy() for key, value in original_values.items()}
        values[field].flat[0] = np.nan if field == 'logits_after' else values[field].flat[0] + 1
        np.savez_compressed(path, **values)
        with pytest.raises(ValueError):
            module.validate_artifact(path, original_metadata)
    np.savez_compressed(path, **original_values)
    assert module.validate_artifact(path, original_metadata)
    migrated = dict(original_metadata, manifest='/different-host/source/episodes.npz',
                    checkpoint_path='/different-host/checkpoints/best.pt')
    assert module.validate_artifact(path, migrated)
    incompatible = dict(original_metadata, scientific_code_sha256='0' * 64)
    with pytest.raises(ValueError, match='identity mismatch'):
        module.validate_artifact(path, incompatible)
