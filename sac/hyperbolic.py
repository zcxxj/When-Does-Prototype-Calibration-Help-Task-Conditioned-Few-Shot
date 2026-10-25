"""HPR-D adapter: FP32, c=.01, multiplier=1, temperature=1.

Only unrectified scores train the encoder. Hard pseudo-label rectification
uses the entire query pool once, without query ground truth or gradients.
"""
import torch
import torch.nn.functional as F
from .vendor import hpr_pmath as pm


class _Riemannian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c):
        ctx.save_for_backward(x)
        ctx.c = c
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        scale = (1 - ctx.c * x.square().sum(-1, keepdim=True)).square() / 4
        return grad * scale, None


def to_ball(z, c=.01):
    return _Riemannian.apply(pm.project(pm.expmap0(z.float(), c=c), c=c), c)


def weighted_mean(points, weights, c=.01):
    klein = pm.p2k(points, c)
    gamma = pm.lorenz_factor(klein, c=c, keepdim=True)
    mass = weights.reshape(-1, 1) * gamma
    return pm.k2p((mass * klein).sum(0) / mass.sum(0), c)


def anchor_scores(proto, query, shot, refine=True, c=.01, temperature=1.):
    p, q = to_ball(proto, c), to_ball(query, c)
    before = -pm.dist_matrix(q, p, c=c) / temperature
    if not refine:
        return before, before
    with torch.no_grad():
        pred = before.argmax(-1)
        updated = []
        for j in range(p.shape[0]):
            chosen = q[pred == j]
            if chosen.shape[0] == 0:
                updated.append(p[j])
                continue
            points = torch.cat((p[j:j+1], chosen), dim=0)
            weights = points.new_ones(points.shape[0])
            weights[0] = shot
            updated.append(weighted_mean(points, weights, c))
        after = -pm.dist_matrix(q, torch.stack(updated), c=c) / temperature
    return before, after


def support_scores(support, query, refine=True, c=.01, temperature=1.):
    """Independent HPR-D: retain every support, shaped [way, shot, dims]."""
    s, q = to_ball(support, c), to_ball(query, c)
    p = pm.poincare_mean(s, dim=1, c=c)
    before = -pm.dist_matrix(q, p, c=c) / temperature
    if not refine:
        return before, before
    with torch.no_grad():
        pred = before.argmax(-1)
        updated = []
        for j in range(s.shape[0]):
            chosen = q[pred == j]
            if chosen.shape[0] == 0:
                updated.append(p[j])
                continue
            points = torch.cat((s[j], chosen), dim=0)
            updated.append(pm.poincare_mean(points, dim=0, c=c))
        after = -pm.dist_matrix(q, torch.stack(updated), c=c) / temperature
    return before, after


def classification_loss(logits_before, targets):
    return F.cross_entropy(logits_before, targets)
