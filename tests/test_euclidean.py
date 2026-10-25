import pytest
import torch


@pytest.mark.parametrize('shot', [1, 5])
def test_anchor_mass_empty_class_and_once(shot):
    from sac.euclidean import anchor_scores
    from sac.vendor.pcia import euclidean_dist
    p = torch.tensor([[0.], [10.], [100.]], requires_grad=True)
    q = torch.tensor([[0.], [0.], [0.], [4.9], [5.1]], requires_grad=True)
    p0, q0 = p.detach().clone(), q.detach().clone()
    before, after = anchor_scores(p, q, shot)
    expected_p = torch.stack([(shot*p[0] + q[:4].sum(0))/(shot+4),
                              (shot*p[1] + q[4])/(shot+1), p[2]])
    torch.testing.assert_close(before, -euclidean_dist(q, p), rtol=0, atol=0)
    torch.testing.assert_close(after, -euclidean_dist(q, expected_p))
    assert before.argmax(-1)[3].item() == 0
    assert after.argmax(-1)[3].item() == 1
    assert not after.requires_grad
    torch.testing.assert_close(p, p0)
    torch.testing.assert_close(q, q0)


def test_disable_and_query_permutation():
    from sac.euclidean import anchor_scores
    p = torch.tensor([[0., 0.], [2., 2.]])
    q = torch.tensor([[0., 1.], [2., 1.], [3., 2.]])
    before, after = anchor_scores(p, q, 1, refine=False)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    _, refined = anchor_scores(p, q, 1)
    order = torch.tensor([2, 0, 1])
    _, permuted = anchor_scores(p, q[order], 1)
    torch.testing.assert_close(permuted, refined[order])
