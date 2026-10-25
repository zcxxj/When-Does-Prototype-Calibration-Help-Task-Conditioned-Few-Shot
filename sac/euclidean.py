"""One hard-assignment Euclidean update with shot-mass PCIA anchors."""
import torch
from .vendor.pcia import euclidean_dist


def anchor_scores(proto, query, shot, refine=True):
    before = -euclidean_dist(query, proto)
    if not refine:
        return before, before
    with torch.no_grad():
        pred = before.argmax(-1)
        updated = []
        for j in range(proto.shape[0]):
            chosen = query[pred == j]
            updated.append(proto[j] if len(chosen) == 0 else
                           (shot * proto[j] + chosen.sum(0)) / (shot + len(chosen)))
        after = -euclidean_dist(query, torch.stack(updated))
    return before, after


@torch.no_grad()
def review_scores(proto, query, shot, methods=('P0', 'E2', 'QONLY', 'UNIT', 'SOFTK')):
    """Single-step controls sharing the original PCIA logits and hard groups.

    No targets enter this interface. Returned features are transient, not a
    persistence format; external diagnostics retain only per-class statistics.
    """
    if shot not in (1, 5):
        raise ValueError('Expected shot 1 or 5')
    if not methods or len(set(methods)) != len(methods):
        raise ValueError('Expected unique nonempty methods')
    if set(methods) - {'P0', 'E2', 'QONLY', 'UNIT', 'SOFTK'}:
        raise ValueError('Unsupported review method')
    before = -euclidean_dist(query, proto)
    pred = before.argmax(-1)
    groups = [query[pred == j] for j in range(len(proto))]
    counts = torch.tensor([len(q) for q in groups], device=proto.device)
    results = {}
    for method in methods:
        soft_mass = torch.full_like(proto[:, 0], float('nan'))
        if method == 'P0':
            updated, after = proto, before
            fraction = torch.ones_like(counts, dtype=proto.dtype)
        elif method == 'SOFTK':
            responsibility = before.softmax(dim=-1)
            soft_mass = responsibility.sum(0)
            updated = (shot * proto + responsibility.T @ query) / (shot + soft_mass).unsqueeze(-1)
            after = -euclidean_dist(query, updated)
            fraction = shot / (shot + soft_mass)
        else:
            mass = {'QONLY': 0, 'UNIT': 1, 'E2': shot}[method]
            updated = torch.stack([proto[j] if len(q) == 0 else
                                   (mass * proto[j] + q.sum(0)) / (mass + len(q))
                                   for j, q in enumerate(groups)])
            after = -euclidean_dist(query, updated)
            fraction = torch.where(counts > 0, mass / (mass + counts).clamp_min(1),
                                   torch.ones_like(counts)).to(proto.dtype)
        results[method] = dict(logits_before=before, logits_after=after,
                               initial_pred=pred, group_size=counts,
                               prototypes_after=updated, anchor_fraction=fraction,
                               displacement=(updated - proto).norm(dim=-1),
                               soft_mass=soft_mass)
    return results
