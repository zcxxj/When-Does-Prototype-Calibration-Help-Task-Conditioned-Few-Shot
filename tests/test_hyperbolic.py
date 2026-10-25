"""Frozen HPR-D mathematics and gradient boundaries."""
import torch


def test_mapping_and_distance_reference():
    from sac.hyperbolic import to_ball, anchor_scores, weighted_mean
    from sac.vendor import hpr_pmath as pm
    torch.manual_seed(11)
    z = torch.randn(8, 9)
    expected = pm.project(pm.expmap0(z, c=.01), c=.01)
    torch.testing.assert_close(to_ball(z), expected)
    before, after = anchor_scores(z[:3], z[3:], shot=1, refine=False)
    torch.testing.assert_close(before, -pm.dist_matrix(expected[3:], expected[:3], c=.01))
    assert after is before
    assert torch.isfinite(before).all()
    torch.testing.assert_close(weighted_mean(expected, torch.ones(8)), pm.poincare_mean(expected, c=.01))
    large = to_ball(torch.full((2, 9), 100.))
    assert torch.isfinite(large).all()
    assert bool((large.norm(dim=-1) <= 9.99 + 1e-5).all())


def test_gradient_and_loss_boundary():
    from sac.hyperbolic import to_ball, anchor_scores, classification_loss
    from sac.vendor import hpr_pmath as pm
    torch.manual_seed(7)
    z = torch.randn(4, 8, requires_grad=True)
    mapped = pm.project(pm.expmap0(z, c=.01), c=.01)
    pm.RiemannianGradient.c = .01
    reference = torch.autograd.grad(pm.RiemannianGradient.apply(mapped).sum(), z)[0]
    torch.testing.assert_close(torch.autograd.grad(to_ball(z).sum(), z)[0], reference)
    # A second call with another curvature must not mutate the first backward.
    u = to_ball(z, c=.01)
    to_ball(z, c=.2)
    torch.testing.assert_close(torch.autograd.grad(u.sum(), z)[0], reference)
    p = torch.randn(5, 8, requires_grad=True)
    q = torch.randn(75, 8, requires_grad=True)
    y = torch.arange(5).repeat_interleave(15)
    before, after = anchor_scores(p, q, shot=5, refine=True)
    loss = classification_loss(before, y)
    grads = torch.autograd.grad(loss, (p, q))
    plain, _ = anchor_scores(p, q, shot=5, refine=False)
    grads_plain = torch.autograd.grad(classification_loss(plain, y), (p, q))
    assert before.shape == after.shape == (75, 5)
    assert not after.requires_grad
    for actual, expected in zip(grads, grads_plain):
        assert torch.isfinite(actual).all()
        assert actual.abs().sum() > 0
        torch.testing.assert_close(actual, expected)


def test_anchor_and_support_updates():
    from sac.hyperbolic import anchor_scores, support_scores, to_ball
    from sac.vendor import hpr_pmath as pm
    # Widely separated clusters leave class 2 empty; class 0 contains unequal supports.
    support = torch.tensor([[[.1, .2], [.7, .4]], [[-1., 0.], [-2., .2]],
                            [[0., 3.], [.1, 4.]]], requires_grad=True)
    query = torch.tensor([[.2, .1], [.4, .3], [-1.4, .1]], requires_grad=True)
    proto = support.mean(1)
    before, after = anchor_scores(proto, query, shot=2, refine=True)
    p, q = to_ball(proto), to_ball(query)
    pred = before.argmax(-1)
    assert set(pred.tolist()) == {0, 1}
    updated = []
    for j in range(3):
        chosen = q[pred == j]
        if len(chosen) == 0:
            updated.append(p[j])
            continue
        x = torch.cat((p[j:j+1], chosen))
        k = 2 * x / (1 + .01 * x.square().sum(-1, keepdim=True))
        gamma = (1 - .01 * k.square().sum(-1, keepdim=True)).rsqrt()
        weight = x.new_ones(len(x), 1)
        weight[0] = 2
        mean = (weight * gamma * k).sum(0) / (weight * gamma).sum(0)
        updated.append(mean / (1 + (1 - .01 * mean.square().sum()).sqrt()))
    torch.testing.assert_close(after, -pm.dist_matrix(q, torch.stack(updated), c=.01))
    torch.testing.assert_close(after[:, 2], before[:, 2])
    sb, sa = support_scores(support, query, refine=True)
    s = to_ball(support)
    initial = pm.poincare_mean(s, dim=1, c=.01)
    torch.testing.assert_close(sb, -pm.dist_matrix(q, initial, c=.01))
    expected = torch.stack([pm.poincare_mean(torch.cat((s[j], q[sb.argmax(-1) == j])), c=.01)
                            for j in range(3)])
    torch.testing.assert_close(sa, -pm.dist_matrix(q, expected, c=.01))
    assert not sa.requires_grad
    plain, _ = support_scores(support, query, refine=False)
    grads = torch.autograd.grad(sb.sum(), (support, query), retain_graph=True)
    plain_grads = torch.autograd.grad(plain.sum(), (support, query))
    for actual, reference in zip(grads, plain_grads):
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
        torch.testing.assert_close(actual, reference)
    perm = torch.tensor([2, 0, 1])
    for scorer, first, kwargs in [(anchor_scores, proto, {'shot': 2}),
                                   (support_scores, support, {})]:
        original = scorer(first, query, refine=True, **kwargs)
        shuffled = scorer(first, query[perm], refine=True, **kwargs)
        for a, b in zip(original, shuffled):
            torch.testing.assert_close(b, a[perm])

