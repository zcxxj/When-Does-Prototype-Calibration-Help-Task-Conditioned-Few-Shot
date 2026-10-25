import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_external_diagnostics_purity_is_not_class_accuracy():
    from sac.review_diagnostics import score_diagnostics
    before = np.array([[9,0,0],[9,0,0],[0,9,0],[0,9,0]])
    after = np.array([[0,9,0],[0,9,0],[0,9,0],[9,0,0]])
    target = np.array([1,1,1,1])
    result = score_diagnostics(dict(logits_before=before, logits_after=after,
        group_size=np.array([2,2,0]), anchor_fraction=np.ones(3),
        displacement=np.zeros(3), soft_mass=np.full(3, np.nan)), target)
    assert result['corrected_count'] == 2 and result['damaged_count'] == 1
    assert (result['corrected_count']-result['damaged_count'])/len(target) == .25
    np.testing.assert_array_equal(result['group_purity'][:2], [1,1])
    np.testing.assert_array_equal(result['group_class_accuracy'][:2], [0,1])
    assert np.isnan(result['group_purity'][2]) and np.isnan(result['group_class_accuracy'][2])
    assert result['initial_accuracy'] == .5


class FeatureModel(torch.nn.Module):
    def __init__(self, family='P0', shot=5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.family, self.shot, self.calls = family, shot, 0
    def episode_features(self, clouds, support_indices, support_y, query_indices):
        assert not self.training and not torch.is_grad_enabled()
        self.calls += 1
        p = torch.arange(5, dtype=torch.float32).reshape(5,1)*10
        return dict(proto=p, query=p.repeat_interleave(15,0)+1, class_ids=torch.arange(5))


def episode(shot=5):
    from sac.data import Episode
    n = shot*5
    return Episode(np.arange(n+75), np.arange(5), np.arange(n), np.arange(n,n+75),
                   np.arange(5).repeat(shot), np.arange(5).repeat(15))


def test_one_frontend_all_controls_labels_only_in_scoring():
    from sac.review_evaluate import predict_methods
    model = FeatureModel()
    ep = episode()
    methods = ('P0','E2','QONLY','UNIT','SOFTK','F1','F2')
    clouds = [np.ones((20,3), dtype=np.float32)]*100
    first, diagnostic = predict_methods(model, ep, clouds, methods, diagnostics=True)
    assert model.calls == 1 and set(first) == set(methods)
    ep.query_y = (ep.query_y+1)%5
    second, changed = predict_methods(model, ep, clouds, methods, diagnostics=True)
    assert model.calls == 2
    for method in methods:
        np.testing.assert_array_equal(first[method]['logits_after'], second[method]['logits_after'])
    assert first['P0']['episode_accuracy'] == 1 and second['P0']['episode_accuracy'] == 0
    assert diagnostic['P0']['initial_accuracy'] == 1 and changed['P0']['initial_accuracy'] == 0
    assert set(diagnostic) == {'P0','E2','QONLY','UNIT','SOFTK'}


def args_for(tmp_path, **updates):
    values = dict(dataset='ModelNet40', fold=0, shot=5, seed=7, family='P0', pool='dev',
        checkpoint='best', side='clean', operator=None, severity=0,
        methods=['P0','E2','QONLY','UNIT','SOFTK'], episodes=700,
        out=tmp_path/'review', source_out=tmp_path/'source', data_root=tmp_path/'data',
        device='cpu', threads=2)
    values.update(updates)
    return SimpleNamespace(**values)


def test_test_requires_freeze_before_data_io(tmp_path):
    from sac.review_evaluate import evaluate_job
    with pytest.raises((ValueError, FileNotFoundError), match='freeze'):
        evaluate_job(args_for(tmp_path, pool='test'))


@pytest.mark.parametrize('updates', [dict(methods=['H']), dict(methods=['UNIT','UNIT']),
    dict(checkpoint='last', side='query', operator='Scale', severity=3),
    dict(side='clean', operator='Scale'), dict(side='both', operator='Jitter', severity=3),
    dict(seed=17), dict(pool='test', episodes=2), dict(out='same', source_out='same')])
def test_scope_rejected_before_io(tmp_path, updates):
    from sac.review_evaluate import evaluate_job
    with pytest.raises(ValueError):
        evaluate_job(args_for(tmp_path, **updates))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from sac import review_evaluate as module
    from sac.evaluate import shared_episodes, prediction_path
    from sac.data import H5Pool
    from sac.campaign import run_dir
    from sac.euclidean import review_scores
    args = args_for(tmp_path, episodes=2, methods=['QONLY','UNIT','SOFTK'])
    job = {key: getattr(args,key) for key in ('dataset','fold','shot','seed','family')}
    pool = H5Pool(tmp_path/'base.h5', np.zeros((120,512,3),dtype=np.float32),
                  np.arange(6).repeat(20), np.arange(120))
    episodes, indices, manifest = shared_episodes(args.source_out, pool, job, 'dev', 700)
    run = run_dir(args.source_out, job)
    run.mkdir(parents=True)
    config = dict(job, smoke=False)
    (run/'config.json').write_text(json.dumps(config))
    (run/'completed.json').write_text('{}')
    p = torch.arange(5, dtype=torch.float32).reshape(5,1)*10
    outputs = review_scores(p, p.repeat_interleave(15,0)+1, 5, ('P0','E2'))
    for method, output in outputs.items():
        path = prediction_path(args.source_out, job, 'dev', method, 'best', ('clean',None,0))
        path.parent.mkdir(parents=True)
        logits = output['logits_after'].numpy()
        np.savez_compressed(path, logits_before=np.repeat(output['logits_before'].numpy()[None],700,0),
            logits_after=np.repeat(logits[None],700,0), pred=np.repeat(logits.argmax(-1)[None],700,0),
            local_target=np.stack([ep.query_y for ep in episodes]), episode_accuracy=np.ones(700),
            episode_id=np.arange(700), corruption_seeds=np.full((700,100),-1),
            point_counts=np.full((700,100),512))
        path.with_suffix('.json').write_text(json.dumps(dict(job, pool='dev', method=method,
            checkpoint='best', checkpoint_epoch=12, condition=['clean',None,0], smoke=False,
            episodes=700, manifest=manifest)))
    models, loaded = [], []
    def factory(**kwargs):
        model = FeatureModel(**kwargs)
        models.append(model)
        return model
    def load(path, model, restore_rng):
        loaded.append(path)
        return dict(config=config, epoch=12)
    monkeypatch.setattr(module, 'EpisodeModel', factory)
    monkeypatch.setattr(module, 'base_pools', lambda *a: {'val': pool})
    monkeypatch.setattr(module, 'load_checkpoint', load)
    return module, args, job, models, loaded


def test_dev_fixed_job_readonly_sources_diagnostics_resume_and_conflict(prepared):
    module, args, job, models, loaded = prepared
    original = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in args.source_out.rglob('*') if p.is_file()}
    result = module.evaluate_job(args)
    assert models[-1].calls == 2 and loaded[-1].parent.parent.name == 'seed7'
    assert result['smoke'] and result['episodes'] == 2
    assert len(result['predictions']) == 3 and len(result['diagnostics']) == 5
    for name, value in original.items():
        path = __import__('pathlib').Path(name)
        assert (path.read_bytes(),path.stat().st_mtime_ns) == value
    assert not (args.out/'predictions').exists()
    for method,path in result['predictions'].items():
        path = __import__('pathlib').Path(path)
        assert 'smoke' in path.parts
        with np.load(path) as saved:
            assert saved['logits_after'].shape == (2,75,5)
            assert saved['episode_rows'].shape == (2,100)
            assert saved['corruption_seeds'].shape == (2,100)
            assert not {'proto','query','prototypes_after'} & set(saved.files)
        assert json.loads(path.with_suffix('.json').read_text())['smoke'] is True
    for method,path in result['diagnostics'].items():
        with np.load(path) as saved:
            assert saved['group_size'].shape == (2,5)
            assert saved['corrected_count'].shape == (2,)
            assert np.all(saved['initial_accuracy'] == 1)
    calls = sum(m.calls for m in models)
    assert module.evaluate_job(args)['skipped'] is True
    assert sum(m.calls for m in models) == calls
    path = __import__('pathlib').Path(result['predictions']['UNIT'])
    meta = json.loads(path.with_suffix('.json').read_text())
    meta['seed'] = 27
    path.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='mismatch'):
        module.evaluate_job(args)


def test_dev_manifest_missing_never_generated(prepared):
    module, args, job, _, _ = prepared
    manifest = args.source_out/'episodes/dev/ModelNet40/fold0/shot5/episodes.npz'
    manifest.unlink()
    with pytest.raises(FileNotFoundError, match='manifest'):
        module.evaluate_job(args)
    assert not manifest.exists()


def test_reject_truncated_artifact(prepared):
    module,args,_,_,_ = prepared
    result = module.evaluate_job(args)
    np.savez_compressed(result['predictions']['UNIT'], episode_accuracy=np.ones(2))
    with pytest.raises(ValueError, match='incomplete'):
        module.evaluate_job(args)


def test_cli_exact_condition(tmp_path, monkeypatch):
    from sac import review_evaluate as module
    seen = []
    monkeypatch.setattr(module, 'evaluate_job', lambda args: seen.append(args))
    module.main(['--out',str(tmp_path/'review'),'--source-out',str(tmp_path/'source'),
        '--data-root',str(tmp_path),'--dataset','ModelNet40','--fold','0','--shot','5',
        '--seed','7','--family','P0','--pool','dev','--checkpoint','best','--side','query',
        '--operator','Scale','--severity','3','--methods','QONLY','UNIT','SOFTK','--device','cpu'])
    assert len(seen) == 1 and seen[0].methods == ['QONLY','UNIT','SOFTK']
    assert seen[0].episodes == 700 and seen[0].side == 'query'


@pytest.mark.parametrize('seed', [7,17,27])
def test_frozen_test_checkpoint_and_shared_manifest_roots(prepared, monkeypatch, seed):
    module, args, job, models, loaded = prepared
    from sac import data
    from sac.campaign import run_dir
    args.pool, args.seed, args.episodes, args.methods = 'test', seed, 700, ['P0','E2']
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'freeze.json').write_text(json.dumps({'decision':'retain SAC'}))
    # Synthetic in-memory fixture, never opens actual novel points.
    pool = data.H5Pool(args.data_root/'fixture_test.h5', np.zeros((120,512,3),dtype=np.float32),
                       np.arange(6).repeat(20), np.arange(120))
    monkeypatch.setattr(data, 'load_pools', lambda *a: {'test':pool})
    job['seed'] = seed
    root = args.source_out if seed == 7 else args.out
    run = run_dir(root, job)
    run.mkdir(parents=True, exist_ok=True)
    config = dict(job, smoke=False)
    (run/'config.json').write_text(json.dumps(config))
    (run/'completed.json').write_text('{}')
    def load(path, model, restore_rng):
        loaded.append(path)
        return dict(config=config, epoch=12)
    monkeypatch.setattr(module,'load_checkpoint',load)
    result = module.evaluate_job(args)
    assert loaded[-1] == run/'best.pt'
    assert models[-1].calls == 700 and not result['smoke']
    assert len(result['diagnostics']) == (5 if seed == 7 else 0)
    assert (args.out/'episodes/test/ModelNet40/fold0/shot5/episodes.npz').is_file()
    for path in result['predictions'].values():
        with np.load(path) as saved:
            assert saved['pred'].shape == (700,75)
        meta = json.loads(__import__('pathlib').Path(path).with_suffix('.json').read_text())
        assert str(args.out/'episodes/test') in meta['manifest']
    assert module.evaluate_job(args)['skipped']


def test_h_stays_independent_legacy_branch():
    from sac.review_evaluate import predict_methods
    from sac.evaluate import predict_episode
    from sac.model import EpisodeModel
    torch.set_num_threads(2)
    model = EpisodeModel('H',1).eval()
    ep = episode(1)
    clouds = [np.random.RandomState(i).randn(20,3).astype(np.float32) for i in range(80)]
    before = predict_episode(model,ep,clouds,'H')
    outputs, diagnostics = predict_methods(model,ep,clouds,('H',))
    assert not diagnostics
    np.testing.assert_array_equal(outputs['H']['logits_after'], before['logits_after'])


def test_all_protocol_jobs_accepted_without_matrix_expansion(tmp_path):
    from sac.review_protocol import evaluation_jobs
    from sac.review_evaluate import validate_job
    root = tmp_path/'review'
    root.mkdir()
    (root/'freeze.json').write_text('{}')
    for pool in ('dev','test'):
        for job in evaluation_jobs(pool):
            args = args_for(tmp_path, **job)
            _, methods, condition, count, _ = validate_job(args)
            assert methods == tuple(job['methods']) and count == 700
            assert condition == (job['side'],job['operator'],job['severity'])


