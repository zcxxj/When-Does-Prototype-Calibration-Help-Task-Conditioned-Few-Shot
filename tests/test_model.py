"""Targeted CPU parity tests against the retained upstream PCIA."""
import torch
import pytest
from types import SimpleNamespace


@pytest.mark.parametrize('family', ['P0', 'J', 'H'])
@pytest.mark.parametrize('shot', [1, 5])
def test_hyperbolic_episode_families(family, shot):
    from sac.model import EpisodeModel
    from sac.hyperbolic import anchor_scores, support_scores
    torch.set_num_threads(2)
    torch.manual_seed(31)
    model = EpisodeModel(family, shot).eval()
    n = 5 * shot
    clouds = [torch.randn(20, 3) for _ in range(n + 75)]
    si, sy, qi = torch.arange(n), torch.arange(5).repeat_interleave(shot), torch.arange(n, n + 75)
    with torch.no_grad():
        actual = model(clouds, si, sy, qi, backend='hyperbolic', refine=True)
        xyz, point, glob = model.backbone(torch.stack(clouds))
        if family == 'H':
            assert model.pcia is None
            expected = support_scores(glob[si].reshape(5, shot, -1), glob[qi])
        else:
            p, q, _, _ = model.pcia(glob[si], glob[qi], point, xyz)
            expected = anchor_scores(p, q, shot)
        perm = torch.randperm(75)
        shuffled = model(clouds, si, sy, qi[perm], backend='hyperbolic', refine=True)
    for key, want in zip(('logits_before', 'logits_after'), expected):
        assert actual[key].shape == (75, 5) and torch.isfinite(actual[key]).all()
        torch.testing.assert_close(actual[key], want)
        torch.testing.assert_close(shuffled[key], actual[key][perm], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('family', ['J', 'H'])
def test_joint_encoder_gradient(family):
    from sac.model import EpisodeModel
    torch.set_num_threads(2)
    torch.manual_seed(37)
    model = EpisodeModel(family, 1).train()
    clouds = [torch.randn(20, 3) for _ in range(80)]
    si, sy, qi = torch.arange(5), torch.arange(5), torch.arange(5, 80)
    out = model(clouds, si, sy, qi, backend='hyperbolic', refine=True)
    loss = torch.nn.functional.cross_entropy(out['logits_before'], torch.arange(5).repeat_interleave(15))
    assert torch.isfinite(loss) and not out['logits_after'].requires_grad
    loss.backward()
    modules = [model.backbone.Econv1, model.backbone.MLP]
    if model.pcia is not None:
        modules += [model.pcia.Conv11, model.pcia.fuse1, model.pcia.Conv1]
    for module in modules:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0


@pytest.mark.parametrize('shot', [1, 5])
def test_ragged_eval_equivalence(shot):
    from sac.model import EpisodeModel
    torch.set_num_threads(2)
    torch.manual_seed(41)
    model = EpisodeModel('P0', shot).eval()
    n = 5 * shot
    clouds = [torch.randn(512, 3) for _ in range(n + 75)]
    si, sy, qi = torch.arange(n), torch.arange(5).repeat_interleave(shot), torch.arange(n, n + 75)
    with torch.no_grad():
        dense = model(clouds, si, sy, qi, backend='euler', refine=False)
        model.force_ragged_eval = True
        seen = []
        hook = model.backbone.register_forward_pre_hook(lambda m, inputs: seen.append(inputs[0].shape))
        bucketed = model(clouds, si, sy, qi, backend='euler', refine=False)
        hook.remove()
    assert len(seen) > 1 and all(shape[0] <= 8 for shape in seen)
    for key in ('logits_before', 'logits_after'):
        torch.testing.assert_close(bucketed[key], dense[key], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('family', ['P0', 'J', 'H'])
def test_ragged_real_points_and_episode_order(family, monkeypatch):
    from sac.model import EpisodeModel
    torch.set_num_threads(2)
    torch.manual_seed(47)
    model = EpisodeModel(family, 5).eval()
    # Single-object buckets exercise the SPF squeeze(1) fix; support spans buckets.
    clouds = [torch.randn(n, 3) for n in [512, 762] + [128] * 98]
    original = [c.clone() for c in clouds]
    si = torch.arange(25).flip(0)
    sy = torch.arange(5).repeat_interleave(5).flip(0)
    qi = torch.randperm(75) + 25
    observed = []
    def record(module, inputs, output):
        batch = inputs[0]
        observed.extend(batch.unbind())
    hook = model.backbone.register_forward_hook(record)
    interaction_calls = []
    if model.pcia is not None:
        interact = model.pcia.interact
        def once(s, q):
            interaction_calls.append((s.shape, q.shape))
            return interact(s, q)
        monkeypatch.setattr(model.pcia, 'interact', once)
    backend = 'euler' if family == 'P0' else 'hyperbolic'
    with torch.no_grad():
        actual = model(clouds, si, sy, qi, backend=backend, refine=family != 'P0')
    hook.remove()
    assert len(observed) == len(clouds)
    # No input point resampling, padding or changed order inside an object.
    assert sorted(tuple(c.shape) for c in observed) == sorted(tuple(c.shape) for c in clouds)
    for c in clouds:
        assert sum(torch.equal(c, encoded) for encoded in observed if c.shape == encoded.shape) == 1
    for c, orig in zip(clouds, original):
        torch.testing.assert_close(c, orig, rtol=0, atol=0)
    if model.pcia is not None:
        assert interaction_calls == [(torch.Size([5, 1024]), torch.Size([75, 1024]))]
    # Changing sampler order does not change class alignment or the query output order.
    perm = torch.randperm(100)
    inv = torch.argsort(perm)
    with torch.no_grad():
        shuffled = model([clouds[i] for i in perm], inv[si], sy, inv[qi],
                         backend=backend, refine=family != 'P0')
    for key in ('logits_before', 'logits_after'):
        assert actual[key].shape == (75, 5) and torch.isfinite(actual[key]).all()
        torch.testing.assert_close(shuffled[key], actual[key], rtol=1e-4, atol=1e-5)


def test_equal_length_corruption_uses_eval_buckets():
    from sac.model import EpisodeModel
    torch.set_num_threads(2)
    model = EpisodeModel('H', 1).eval()
    clouds = [torch.randn(128, 3) for _ in range(80)]
    seen = []
    hook = model.backbone.register_forward_pre_hook(lambda m, x: seen.append(x[0].shape))
    with torch.no_grad():
        model(clouds, torch.arange(5), torch.arange(5), torch.arange(5, 80),
              backend='hyperbolic', refine=True)
    hook.remove()
    assert len(seen) == 10 and all(shape == (8, 128, 3) for shape in seen)


def test_reject_unsupported_ragged_training_and_small_clouds():
    from sac.model import EpisodeModel
    model = EpisodeModel('P0', 1)
    clouds = [torch.randn(20, 3) for _ in range(80)]
    clouds[0] = torch.randn(21, 3)
    si, sy, qi = torch.arange(5), torch.arange(5), torch.arange(5, 80)
    with pytest.raises(ValueError, match='eval'):
        model(clouds, si, sy, qi, backend='euler', refine=False)
    model.eval()
    clouds[0] = torch.randn(19, 3)
    with pytest.raises(ValueError, match='20'):
        model(clouds, si, sy, qi, backend='euler', refine=False)


def args(shot=1):
    return SimpleNamespace(way=5, shot=shot, query=15, emb_dims=1024,
                           SPFknn=16, SPFks=8, CIFk=20, CIFh=64)


@pytest.mark.parametrize('shot', [1, 5])
def test_p0_matches_upstream(shot):
    from sac.model import AdaptedPCIA, EpisodeModel
    from sac.vendor.pcia import PCIA, euclidean_dist
    torch.set_num_threads(2)
    torch.manual_seed(7)
    ref = PCIA(args(shot)).eval()
    adapted = AdaptedPCIA(args(shot)).eval()
    adapted.load_state_dict(ref.state_dict(), strict=True)
    n = 5 * shot
    s, q = torch.randn(n, 1024), torch.randn(75, 1024)
    points, xyz = torch.randn(n + 75, 1024, 20), torch.randn(n + 75, 3, 20)
    with torch.no_grad():
        expected = ref(s, q, points, xyz)
        actual = adapted(s, q, points, xyz)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)
    # Full model embeds the original sampler order, then reorders by support label.
    model = EpisodeModel('P0', shot).eval()
    model.pcia.load_state_dict(ref.state_dict(), strict=True)
    clouds = [torch.randn(20, 3) for _ in range(n + 75)]
    si = torch.randperm(n)
    sy = (torch.arange(5).repeat_interleave(shot) + 1)[si] * 10
    qi = torch.arange(n, n + 75)
    with torch.no_grad():
        coords, pf, gf = model.backbone(torch.stack(clouds))
        order = torch.cat((si[torch.argsort(sy, stable=True)], qi))
        ep = ref(gf[order[:n]], gf[qi], pf[order], coords[order])
        result = model(clouds, si, sy, qi, backend='euler', refine=False)
    torch.testing.assert_close(result['logits_before'], -euclidean_dist(ep[1], ep[0]))
    torch.testing.assert_close(result['logits_after'], result['logits_before'])
    assert result['class_ids'].tolist() == [10, 20, 30, 40, 50]


@pytest.mark.parametrize('shot', [1, 5])
def test_e2_preserves_p0_before(shot):
    from sac.model import EpisodeModel
    from sac.euclidean import anchor_scores
    torch.set_num_threads(2)
    torch.manual_seed(53)
    model = EpisodeModel('P0', shot).eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    n = 5 * shot
    clouds = [torch.randn(20, 3) for _ in range(n + 75)]
    si, sy, qi = torch.arange(n), torch.arange(5).repeat_interleave(shot), torch.arange(n, n + 75)
    with torch.no_grad():
        p0 = model(clouds, si, sy, qi, backend='euler', refine=False)
        e2 = model(clouds, si, sy, qi, backend='euler', refine=True)
        _, (proto, query) = model._ragged_features(clouds, si, qi)
        expected = anchor_scores(proto, query, shot)
    torch.testing.assert_close(e2['logits_before'], p0['logits_before'], rtol=0, atol=0)
    for key, want in zip(('logits_before', 'logits_after'), expected):
        torch.testing.assert_close(e2[key], want, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


@pytest.mark.parametrize('shot', [1, 5])
def test_e2_ragged_and_permutation(shot):
    from sac.model import EpisodeModel
    torch.set_num_threads(2)
    torch.manual_seed(59)
    model = EpisodeModel('P0', shot).eval()
    n = 5 * shot
    si, sy, qi = torch.arange(n), torch.arange(5).repeat_interleave(shot), torch.arange(n, n + 75)
    clouds = [torch.randn(512, 3) for _ in range(n + 75)]
    with torch.no_grad():
        dense = model(clouds, si, sy, qi, backend='euler', refine=True)
        model.force_ragged_eval = True
        bucketed = model(clouds, si, sy, qi, backend='euler', refine=True)
        for key in ('logits_before', 'logits_after'):
            torch.testing.assert_close(bucketed[key], dense[key], rtol=1e-4, atol=1e-5)
        clouds = [torch.randn(size, 3) for size in [512, 762] + [128] * (n + 73)]
        original = [c.clone() for c in clouds]
        seen = []
        hook = model.backbone.register_forward_pre_hook(lambda m, x: seen.extend(x[0].unbind()))
        mixed = model(clouds, si, sy, qi, backend='euler', refine=True)
        hook.remove()
        assert len(seen) == len(clouds)
        for cloud, copy in zip(clouds, original):
            assert sum(torch.equal(cloud, item) for item in seen if item.shape == cloud.shape) == 1
            torch.testing.assert_close(cloud, copy, rtol=0, atol=0)
        order = torch.randperm(75)
        shuffled = model(clouds, si, sy, qi[order], backend='euler', refine=True)
        for key in ('logits_before', 'logits_after'):
            torch.testing.assert_close(shuffled[key], mixed[key][order], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('family,training,refine', [('P0', True, True), ('J', False, True),
                                                  ('H', False, True), ('H', False, False)])
def test_e2_restricted_to_p0_eval(family, training, refine):
    from sac.model import EpisodeModel
    model = EpisodeModel(family, 1).train(training)
    with pytest.raises(ValueError, match='Euler'):
        model([], torch.arange(5), torch.arange(5), torch.arange(5, 80),
              backend='euler', refine=refine)

