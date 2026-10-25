"""PCIA protocol episode model; query labels never enter this interface."""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .vendor.dgcnn import DGCNN
from .vendor.pcia import (PCIA, euclidean_dist, cos_similarity_batch,
                          index_points, knn_point, top_cos_similarity)
from .hyperbolic import anchor_scores, support_scores
from .euclidean import anchor_scores as euclidean_anchor_scores


class AdaptedPCIA(PCIA):
    """Unchanged PCIA parameters; split point-local SPF from episode interaction."""

    def spf_local(self, p_fea, guide):
        """Each object uses its own query guide or the FULL support-class mean."""
        b, dims = guide.shape
        p_fea = p_fea.transpose(2, 1)
        g_fea = guide.view(b, 1, dims)
        center_idx = cos_similarity_batch(g_fea, p_fea, nk=self.args.SPFknn).squeeze(1)
        c_fea = index_points(p_fea, center_idx.long())
        local_idx = knn_point(self.args.SPFks, p_fea, c_fea)
        grouped_fea = index_points(p_fea, local_idx)
        grouped_fea = grouped_fea.view(-1, self.args.SPFks, dims).unsqueeze(2)
        g_fea = g_fea.expand(b, self.args.SPFknn, dims)
        g_fea = g_fea.reshape(-1, 1, dims).unsqueeze(2)
        g_l_feature = torch.cat((grouped_fea, g_fea), dim=1)
        mean_part_fea = self.Conv11(g_l_feature).view(b, self.args.SPFknn, dims)
        return mean_part_fea.max(dim=1)[0], center_idx, local_idx

    def spf_merge(self, local, s_global, q_global):
        way, shot, dims = self.args.way, self.args.shot, self.args.emb_dims
        s_mean = s_global.view(way, shot, dims).mean(1)
        s = (local[:way * shot].view(way, shot, dims).mean(1) + s_mean) / 2.0
        q = (local[way * shot:] + q_global) / 2.0
        return s, q

    def forward(self, s, q, p_fea, coords):
        way, shot, dims = self.args.way, self.args.shot, self.args.emb_dims
        s_mean = s.view(way, shot, dims).mean(1, keepdim=True)
        guide = torch.cat((s_mean.expand(way, shot, dims).reshape(way * shot, dims), q))
        local, centers, neighbors = self.spf_local(p_fea, guide)
        s, q = self.spf_merge(local, s, q)
        s, q = self.interact(s, q)
        return s, q, centers, neighbors

    def interact(self, s, q):
        way, query, dims = self.args.way, self.args.query, self.args.emb_dims
        # SCIp
        s_meta = self.fuse1(s.view(way, 1, 1, dims).transpose(0, 1)).view(1, 1, dims)
        s_cat = torch.cat((s.view(way, 1, -1), s_meta.repeat(way, 1, 1)), dim=1)
        q_cat = torch.cat((q.view(way * query, 1, -1), s_meta.repeat(way * query, 1, 1)), dim=1)

        s_fuse = self.fuse2(s_cat.view(way, 2, 1, dims)).view(-1, 32, dims)
        q_fuse = self.fuse2(q_cat.view(way * query, 2, 1, dims)).view(-1, 32, dims)

        s_fuse_q = self.FC1(s_fuse.transpose(2, 1))
        s_fuse_k = self.FC2(s_fuse.transpose(2, 1))
        s_fuse_v = self.FC3(s_fuse.transpose(2, 1))

        q_fuse_q = self.FC1(q_fuse.transpose(2, 1))
        q_fuse_k = self.FC2(q_fuse.transpose(2, 1))
        q_fuse_v = self.FC3(q_fuse.transpose(2, 1))

        s_att_map = torch.bmm(s_fuse_q, s_fuse_k.transpose(2, 1)) / np.power(32, 0.5)
        q_att_map = torch.bmm(q_fuse_q, q_fuse_k.transpose(2, 1)) / np.power(32, 0.5)

        s_att_map = F.softmax(s_att_map, dim=2)
        q_att_map = F.softmax(q_att_map, dim=2)

        s_atten = torch.bmm(s_att_map, s_fuse_v)
        q_atten = torch.bmm(q_att_map, q_fuse_v)

        s_atten = self.fuse3(s_atten.view(-1, 1024, 32, 1).transpose(2, 1)).squeeze()
        q_atten = self.fuse3(q_atten.view(-1, 1024, 32, 1).transpose(2, 1)).squeeze()

        s = (s_atten + s) / 2.0
        q = (q_atten + q) / 2.0

        # CIFp
        s_init = s
        q_init = q

        sq_emd_att_map = torch.mm(s, q.transpose(0, 1)) / np.power(dims, 0.5)
        qs_emd_att_map = torch.mm(q, s.transpose(0, 1)) / np.power(dims, 0.5)

        sq_emd_att_map = F.softmax(sq_emd_att_map, dim=-1)
        qs_emd_att_map = F.softmax(qs_emd_att_map, dim=-1)

        sq_atten = torch.mm(sq_emd_att_map, q)
        qs_atten = torch.mm(qs_emd_att_map, s)

        idxsq = top_cos_similarity(s, q, self.args.CIFk)
        idxqs = top_cos_similarity(q, s, s.size(0))

        sq_a = torch.cat((s.view(way, 1, 1, dims), q[idxsq].view(way, self.args.CIFk, 1, dims)), dim=1)
        qs_a = torch.cat((s[idxqs].view(way * query, way, 1, dims), q.view(way * query, 1, 1, dims)), dim=1)

        sq_aa = F.softmax(self.Conv3(self.Conv1(sq_a)).squeeze(), dim=1)
        qs_aa = F.softmax(self.Conv4(self.Conv2(qs_a)).squeeze(), dim=1)

        s_att = torch.mul(sq_aa, sq_a.squeeze()).sum(dim=1)
        q_att = torch.mul(qs_aa, qs_a.squeeze()).sum(dim=1)

        s = (s_att + s_init + sq_atten) / 3.0
        q = (q_att + q_init + qs_atten) / 3.0

        idxsq = top_cos_similarity(s, q, 15)
        sq_a = torch.cat((s.view(way, 1, 1, dims), q[idxsq].view(way, 15, 1, dims)), dim=1)
        sq_aa = F.softmax(self.Conv6(self.Conv5(sq_a)).squeeze(), dim=1)
        s_att = torch.mul(sq_aa, sq_a.squeeze()).sum(dim=1)

        s = (s_att + s) / 2.0

        return s, q


class EpisodeModel(nn.Module):
    def __init__(self, family, shot):
        super().__init__()
        if family not in ('P0', 'J', 'H', 'D') or shot not in (1, 5):
            raise ValueError('Supported families P0/J/H/D and shots 1/5 only')
        self.family, self.shot = family, shot
        self.args = SimpleNamespace(way=5, shot=shot, query=15, emb_dims=1024,
                                    SPFknn=16, SPFks=8, CIFk=20, CIFh=64)
        # Construct common backbone first so matched seeds give matched weights.
        self.backbone = DGCNN(k=20, emb_dims=1024)
        self.pcia = AdaptedPCIA(self.args) if family in ('P0', 'J') else None
        self.force_ragged_eval = False  # parity-test switch; not a checkpoint parameter

    def _ragged_features(self, clouds, sidx, qidx):
        # Buckets split only point-dependent operations, never an episode.
        buckets = {}
        for i, cloud in enumerate(clouds):
            buckets.setdefault(cloud.shape[0], []).append(i)
        global_parts, positions, point_parts = [], [], []
        for indices in buckets.values():
            for start in range(0, len(indices), 8):
                ids = indices[start:start + 8]
                _, point, glob = self.backbone(torch.stack([clouds[i] for i in ids]))
                global_parts.append(glob)
                positions.extend(ids)
                if self.pcia is not None:
                    point_parts.append((ids, point))
        device = global_parts[0].device
        restore = torch.argsort(torch.tensor(positions, device=device))
        glob = torch.cat(global_parts)[restore]
        if self.pcia is None:
            return glob, None
        support = glob[sidx].view(5, self.shot, -1)
        guide = glob.clone()
        guide[sidx] = support.mean(1, keepdim=True).expand_as(support).reshape(len(sidx), -1)
        local_parts = [self.pcia.spf_local(point, guide[ids])[0] for ids, point in point_parts]
        local = torch.cat(local_parts)[restore]
        order = torch.cat((sidx, qidx))
        s, q = self.pcia.spf_merge(local[order], glob[sidx], glob[qidx])
        return glob, self.pcia.interact(s, q)

    def _episode_features(self, clouds, support_indices, support_y, query_indices):
        class_ids, counts = torch.unique(support_y, sorted=True, return_counts=True)
        if class_ids.numel() != 5 or not bool((counts == self.shot).all()) or query_indices.numel() != 75:
            raise ValueError('Expected a balanced 5-way K-shot, 75-query episode')
        sizes = [cloud.shape[0] for cloud in clouds]
        if any(cloud.ndim != 2 or cloud.shape[1] != 3 for cloud in clouds) or min(sizes) < 20:
            raise ValueError('Each cloud must have shape [N, 3] with N >= 20')
        ragged = (len(set(sizes)) != 1 or self.force_ragged_eval
                  or (not self.training and sizes[0] != 512))
        if ragged and self.training:
            raise ValueError('Ragged point buckets are supported in eval mode only')
        sidx = support_indices[torch.argsort(support_y, stable=True)]
        order = torch.cat((sidx, query_indices))
        if ragged:
            glob, features = self._ragged_features(clouds, sidx, query_indices)
        else:
            # Preserve sampler batch order and training BN statistics.
            xyz, point, glob = self.backbone(torch.stack(clouds))
            features = None
            if self.pcia is not None:
                proto, query, _, _ = self.pcia(glob[sidx], glob[query_indices], point[order], xyz[order])
                features = proto, query
        return glob, features, sidx, class_ids

    @torch.no_grad()
    def episode_features(self, clouds, support_indices, support_y, query_indices):
        """Frozen full-episode PCIA p/q, computed once for review controls."""
        if self.family != 'P0' or any(module.training for module in self.modules()):
            raise ValueError('Shared PCIA features require P0 eval mode')
        _, (proto, query), _, class_ids = self._episode_features(
            clouds, support_indices, support_y, query_indices)
        return dict(proto=proto, query=query, class_ids=class_ids)

    def forward(self, clouds, support_indices, support_y, query_indices, *, backend, refine):
        if backend not in ('euler', 'hyperbolic'):
            raise ValueError('Supported backends: euler, hyperbolic')
        if self.family == 'D':
            if backend != 'euler':
                raise ValueError('D requires euler')
            if refine and self.training:
                raise ValueError('D refinement requires eval mode')
            glob, _, sidx, class_ids = self._episode_features(
                clouds, support_indices, support_y, query_indices)
            proto = glob[sidx].reshape(5, self.shot, -1).mean(1)
            before, after = euclidean_anchor_scores(
                proto, glob[query_indices], self.shot, refine=refine)
            return dict(logits_before=before, logits_after=after, class_ids=class_ids)
        if backend == 'euler' and self.family == 'H':
            raise ValueError('Euler requires PCIA')
        if backend == 'euler' and refine and (self.family != 'P0' or self.training):
            raise ValueError('Euler refinement requires P0 eval mode')
        glob, features, sidx, class_ids = self._episode_features(
            clouds, support_indices, support_y, query_indices)
        if self.pcia is None:
            before, after = support_scores(glob[sidx].reshape(5, self.shot, -1),
                                           glob[query_indices], refine=refine)
        else:
            proto, query = features
            if backend == 'euler':
                if refine:
                    before, after = euclidean_anchor_scores(proto, query, self.shot)
                else:
                    before = after = -euclidean_dist(query, proto)
            else:
                before, after = anchor_scores(proto, query, self.shot, refine=refine)
        return dict(logits_before=before, logits_after=after, class_ids=class_ids)
