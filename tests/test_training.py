import pytest


def test_checkpoint_policy():
    from sac.train import checkpoint_policy, advance_best
    assert checkpoint_policy('P0') == ('euler', False, 'logits_before')
    assert checkpoint_policy('J') == ('hyperbolic', True, 'logits_after')
    assert checkpoint_policy('H') == ('hyperbolic', True, 'logits_after')
    assert advance_best(.5, .5, 29) == (.5, 0, True)
    assert advance_best(.4, .5, 29) == (.5, 30, False)
    with pytest.raises(FloatingPointError):
        advance_best(float('nan'), .5, 0)


def test_resume_epoch_boundary(tmp_path):
    from sac.train import save_checkpoint, load_checkpoint
    import torch
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.2))
    opt = torch.optim.Adam(model.parameters(), lr=.0008)
    sched = torch.optim.lr_scheduler.StepLR(opt, 5, .5)
    def step():
        opt.zero_grad()
        x = torch.randn(4, 3)
        loss = model(x).square().mean()
        loss.backward()
        opt.step()
        sched.step()
        return loss.detach().clone(), x
    step()
    path = tmp_path / 'resume.pt'
    save_checkpoint(path, model, opt, sched, epoch=1, best=.5, stale=2, config={'family': 'J'})
    expected, inputs = step()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    torch.rand(100)
    state = load_checkpoint(path, model, opt, sched)
    actual, restored_inputs = step()
    assert state['epoch'] == 1 and state['stale'] == 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs, restored_inputs, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_training_loss_before_selection_after():
    from sac.train import run_epoch
    import numpy as np
    import torch
    from sac.data import Episode
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
        def forward(self, clouds, support_indices, support_y, query_indices, *, backend, refine):
            assert torch.is_grad_enabled() == self.training
            before = self.weight * torch.tensor([[1., -1.]])
            return dict(logits_before=before, logits_after=-before.detach(), class_ids=torch.tensor([5, 8]))
    model = Tiny()
    ep = Episode(np.arange(3), np.array([5, 8]), np.array([0, 1]), np.array([2]), np.array([5, 8]), np.array([1]))
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    result = run_epoch(model, [(ep, [torch.ones(20, 3)] * 3)], family='J', optimizer=opt)
    assert model.weight.item() < 1  # loss BEFORE, despite correct AFTER prediction
    assert result['accuracy'] == 1
    val = run_epoch(model, [(ep, [torch.ones(20, 3)] * 3)], family='J')
    assert val['accuracy'] == 1 and not model.training


def test_smoke_runner_base_only_and_completion(tmp_path, monkeypatch):
    import numpy as np
    import h5py
    import torch
    from sac import train, evaluate, model
    class Tiny(torch.nn.Module):
        def __init__(self, family, shot):
            super().__init__()
            self.layer = torch.nn.Linear(3, 5)
        def forward(self, clouds, support_indices, support_y, query_indices, *, backend, refine):
            x = torch.stack([clouds[i].mean(0) for i in query_indices])
            logits = self.layer(x)
            return dict(logits_before=logits, logits_after=logits.detach() if refine else logits,
                        class_ids=torch.unique(support_y, sorted=True))
    monkeypatch.setattr(model, 'EpisodeModel', Tiny)
    folder = tmp_path / 'data' / 'ScanObjectNN_FS'
    folder.mkdir(parents=True)
    # Purpose-built fixture; intentionally NO test/novel H5 on disk.
    with h5py.File(folder / 'train_scanobjectnn_PB_T50_RS_fsl_1.h5', 'w') as f:
        f['data'] = np.random.RandomState(2).randn(210, 512, 3).astype('float32')
        f['label'] = np.repeat(np.arange(10), 21)
    args = ['--data-root', str(tmp_path / 'data'), '--out', str(tmp_path / 'smoke'),
            '--dataset', 'ScanObjectNN', '--fold', '0', '--shot', '1', '--family', 'J',
            '--smoke', '--device', 'cpu', '--threads', '1']
    train.main(args)
    train.main(args)  # completed identical config is an explicit skip
    evaluate.main(args + ['--pool', 'dev'])
    job = dict(dataset='ScanObjectNN', fold=0, shot=1, seed=7, family='J')
    path = evaluate.prediction_path(tmp_path / 'smoke', job, 'dev', 'J2', 'best', ('clean', None, 0))
    with np.load(path) as values:
        assert values['logits_after'].shape == (3, 75, 5)
        assert values['episode_accuracy'].shape == (3,)
    import json
    run = train.Path(tmp_path / 'smoke/runs/ScanObjectNN/fold0/shot1/seed7/J')
    row = json.loads((run / 'epochs.jsonl').read_text())
    assert row['train']['episodes'] == row['val']['episodes'] == 3
    assert row['train']['gradient_norm_mean'] > 0


@pytest.mark.parametrize('dataset,nclasses', [('ModelNet40', 30), ('ShapeNet70', 50), ('ScanObjectNN', 10)])
def test_all_smoke_base_pools_work_without_novel(tmp_path, dataset, nclasses):
    import h5py
    import numpy as np
    from sac.data import official_paths
    from sac.train import base_pools, epoch_episodes, point_indices
    path = official_paths(tmp_path, dataset + '_FS', 0)['train']
    path.parent.mkdir(parents=True)
    # Deliberate fixture, not released data. Only the base file exists.
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('data', shape=(nclasses * 20, 512, 3), dtype='float32')
        handle['label'] = np.repeat(np.arange(nclasses), 20)
    pools = base_pools(tmp_path, dataset, 0)
    for shot in (1, 5):
        for family in ('P0', 'J', 'H'):
            job = dict(dataset=dataset, fold=0, shot=shot, seed=7, family=family)
            for name, stream in [('train', 10), ('val', 11)]:
                indices = point_indices(pools[name], dataset, 0, shot, stream)
                episode, clouds = next(epoch_episodes(pools[name], indices, job, 0, 1,
                                                      validation=name == 'val'))
                assert len(clouds) == 5 * (shot + 15)
                assert len(episode.query_y) == 75
                assert set(episode.rows) <= set(pools[name].rows)
    if dataset != 'ScanObjectNN':
        assert set(pools['train'].rows).isdisjoint(pools['val'].rows)


def test_epoch_inputs_are_method_and_resume_independent():
    import numpy as np
    from pathlib import Path
    from sac.data import H5Pool
    from sac.train import epoch_episodes
    pool = H5Pool(Path('fixture.h5'), np.random.RandomState(1).randn(120, 512, 3).astype('float32'),
                  np.repeat(np.arange(6), 20), np.arange(120))
    job = dict(dataset='ModelNet40', fold=0, shot=1, seed=7, family='P0')
    a = list(epoch_episodes(pool, np.arange(512), job, 2, 2))
    np.random.rand(100)
    job['family'] = 'H'
    b = list(epoch_episodes(pool, np.arange(512), job, 2, 2))
    for (ea, ca), (eb, cb) in zip(a, b):
        np.testing.assert_array_equal(ea.rows, eb.rows)
        np.testing.assert_array_equal(ca, cb)
