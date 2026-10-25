import pytest
import torch


def test_shared_hard_controls_analytic_empty_tie_and_once():
    from sac.euclidean import review_scores, anchor_scores
    from sac.vendor.pcia import euclidean_dist
    p = torch.tensor([[0.], [10.], [100.]], requires_grad=True)
    q = torch.tensor([[0.], [0.], [0.], [4.9], [5.1], [5.]], requires_grad=True)
    outputs = review_scores(p, q, 5, ('P0', 'QONLY', 'UNIT', 'E2'))
    before = -euclidean_dist(q, p)
    groups = before.argmax(-1)
    assert groups.tolist() == [0, 0, 0, 0, 1, 0]
    for method, mass in [('QONLY', 0), ('UNIT', 1), ('E2', 5)]:
        result = outputs[method]
        want = torch.stack([(mass*p[0]+q[groups == 0].sum(0))/(mass+5),
                            (mass*p[1]+q[groups == 1].sum(0))/(mass+1), p[2]])
        torch.testing.assert_close(result['prototypes_after'], want)
        torch.testing.assert_close(result['logits_after'], -euclidean_dist(q, want))
        assert result['logits_before'] is outputs['P0']['logits_before']
        assert result['initial_pred'] is outputs['P0']['initial_pred']
        assert result['group_size'].tolist() == [5, 1, 0]
        assert not result['logits_after'].requires_grad
    torch.testing.assert_close(outputs['E2']['logits_after'], anchor_scores(p, q, 5)[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs['P0']['logits_after'], before, rtol=0, atol=0)


@pytest.mark.parametrize('shot', [1, 5])
def test_soft_k_analytic_stability_and_one_shot_equivalence(shot):
    from sac.euclidean import review_scores
    p = torch.tensor([[0., 1.], [4., 2.], [1000., 1000.]])
    q = torch.tensor([[0., 2.], [2., 1.], [5., 2.]])
    results = review_scores(p, q, shot, ('P0', 'E2', 'UNIT', 'SOFTK'))
    r = results['P0']['logits_before'].softmax(-1)
    torch.testing.assert_close(r.sum(-1), torch.ones(3))
    expected = (shot*p+r.T@q)/(shot+r.sum(0)).unsqueeze(-1)
    torch.testing.assert_close(results['SOFTK']['prototypes_after'], expected)
    torch.testing.assert_close(results['SOFTK']['soft_mass'], r.sum(0))
    torch.testing.assert_close(results['SOFTK']['anchor_fraction'], shot/(shot+r.sum(0)))
    assert torch.isfinite(results['SOFTK']['logits_after']).all()
    if shot == 1:
        torch.testing.assert_close(results['UNIT']['logits_after'], results['E2']['logits_after'], rtol=0, atol=0)


@pytest.mark.parametrize('shot,ragged', [(1, False), (5, True)])
def test_shared_feature_interface_preserves_old_paths(shot, ragged, monkeypatch):
    from sac.model import EpisodeModel
    from sac.euclidean import review_scores
    from sac.vendor.pcia import euclidean_dist
    torch.set_num_threads(2)
    torch.manual_seed(71)
    model = EpisodeModel('P0', shot).eval()
    n = 5*shot
    clouds = [torch.randn(20+(i % 3) if ragged else 512, 3) for i in range(n+75)]
    args = (clouds, torch.arange(n), torch.arange(5).repeat_interleave(shot), torch.arange(n,n+75))
    with torch.no_grad():
        old_p0 = model(*args, backend='euler', refine=False)
        old_e2 = model(*args, backend='euler', refine=True)
        calls = []
        interact = model.pcia.interact
        def once(p, q):
            calls.append(1)
            return interact(p, q)
        monkeypatch.setattr(model.pcia, 'interact', once)
        features = model.episode_features(*args)
        results = review_scores(features['proto'], features['query'], shot, ('P0','E2','QONLY','UNIT','SOFTK'))
        assert len(calls) == 1
        torch.testing.assert_close(results['P0']['logits_after'], old_p0['logits_after'], rtol=0, atol=0)
        torch.testing.assert_close(results['E2']['logits_after'], old_e2['logits_after'], rtol=0, atol=0)
        order = torch.randperm(75)
        perm = model.episode_features(*args[:3], args[3][order])
        torch.testing.assert_close(-euclidean_dist(perm['query'], perm['proto']), old_p0['logits_before'][order], rtol=1e-4, atol=1e-5)
    model.train()
    with pytest.raises(ValueError, match='eval'):
        model.episode_features(*args)


def test_training_p0_outputs_and_gradients_match_original_computation():
    from copy import deepcopy
    from sac.model import EpisodeModel
    from sac.vendor.pcia import euclidean_dist
    # CPU scatter-add backward uses parallel accumulation; compare serially.
    torch.set_num_threads(1)
    torch.manual_seed(83)
    model = EpisodeModel('P0',1).train()
    original = deepcopy(model)
    clouds = [torch.randn(20,3) for _ in range(80)]
    si, sy, qi = torch.arange(5), torch.arange(5), torch.arange(5,80)
    actual = model(clouds,si,sy,qi,backend='euler',refine=False)
    xyz, point, glob = original.backbone(torch.stack(clouds))
    order = torch.cat((si,qi))
    p,q,_,_ = original.pcia(glob[si],glob[qi],point[order],xyz[order])
    expected = -euclidean_dist(q,p)
    torch.testing.assert_close(actual['logits_before'],expected,rtol=0,atol=0)
    target = torch.arange(5).repeat_interleave(15)
    torch.nn.functional.cross_entropy(actual['logits_before'],target).backward()
    torch.nn.functional.cross_entropy(expected,target).backward()
    for (name,a),(_,b) in zip(model.named_parameters(),original.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0,msg=name)
    for (name,a),(_,b) in zip(model.named_buffers(),original.named_buffers()):
        torch.testing.assert_close(a,b,rtol=0,atol=0,msg=name)


