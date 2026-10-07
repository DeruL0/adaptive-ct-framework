"""Edge-preserving penalty between neighbouring leaves (fixed tree, leaf types and step geometry).

R(c) = sum over shared-face sample points p of  area_p * phi(f_i(p) - f_j(p))
  f_i, f_j : the two leaves' functions at p (linear in the stored values c, nonnegative basis weights)
  phi(t)   : t^2 / 2 for |t| <= delta (small jumps, e.g. seams between leaves, are smoothed),
             delta * (|t| - delta / 2) beyond (large jumps, i.e. real edges, are penalised only linearly)
Each face patch between two leaves is sampled at 2 x 2 points from the finer leaf (equal sizes: once).
Separable curvature bound for the solver: a_p = b_i(p) - b_j(p) has |a_p| summing to 2 (two nonnegative
weight sets summing to 1 each) and phi'' <= 1, so  sum_p area_p a_p a_p^T <= diag(sum_p 2 area_p |a_p|).
Stored per sample: the two leaf indices (int32), both leaf-local points (fp16: dyadic, exact) and the area,
24 bytes; the basis weights are evaluated in chunks at every use from the model's current types, values and
step geometry (so the same samples serve a geometry update).
"""

from __future__ import annotations

import torch

from adaptive_ct.representation.octree import VALUE_COUNT, step_H, trilinear_weights


def basis_rows(model, leaf_ids: torch.Tensor, u: torch.Tensor, kind_all=None, offsets_all=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, 8) basis weights of each leaf's function at leaf-local points u (n, 3) and the value index per
    weight (padding weights are 0 and point at the leaf's first value).  kind_all / offsets_all: the model's
    leaf_type() / value_offsets(), when already at hand."""
    u = u.clamp(0.0, 1.0)
    kind = (model.leaf_type() if kind_all is None else kind_all)[leaf_ids]
    w = torch.zeros((leaf_ids.numel(), 8), device=u.device)
    w[kind == 0, 0] = 1.0
    w[kind == 1] = trilinear_weights(u[kind == 1])
    step = kind == 2
    H = step_H(model.geometry(leaf_ids[step]), u[step])
    w[step, 0], w[step, 1] = 1 - H, H
    count = torch.tensor(VALUE_COUNT, device=u.device)[kind]
    k = torch.arange(8, device=u.device)[None]
    idx = (model.value_offsets() if offsets_all is None else offsets_all)[leaf_ids, None] + torch.where(k < count[:, None], k, 0)
    return w, idx


def face_samples(model):
    """Shared-face sample points, in 24 chunks: (finer leaf i, other leaf j, u_i, u_j leaf-local, area)."""
    device = model.leaf_levels.device
    N = int(model.leaf_levels.numel())
    shapes = torch.tensor(model.level_shapes, dtype=torch.float32, device=device)
    size = 2.0 / shapes[model.leaf_levels]  # (N, 3) box units
    origin = -1.0 + model.leaf_coords.float() * size
    eps = 1e-4 * float((2.0 / shapes[-1]).min())
    q = torch.tensor([0.25, 0.75], device=device)
    for axis in range(3):
        t1, t2 = [a for a in range(3) if a != axis]
        for side in (0.0, 1.0):
            for a in q:
                for b in q:
                    u = torch.zeros((N, 3), device=device)
                    u[:, axis], u[:, t1], u[:, t2] = side, a, b
                    p = origin + u * size
                    outside = p.clone()
                    outside[:, axis] += eps if side == 1.0 else -eps
                    j = model.resolve_leaf_ids(outside)
                    keep = j >= 0
                    lj = torch.where(keep, model.leaf_levels[j.clamp_min(0)], 0)
                    keep &= (lj < model.leaf_levels) | ((lj == model.leaf_levels) & (side == 1.0))
                    i_ids, j_ids = torch.nonzero(keep).flatten(), j[keep]
                    yield i_ids, j_ids, u[keep], (p[keep] - origin[j_ids]) / size[j_ids], size[i_ids, t1] * size[i_ids, t2] / 4.0


class EdgePenalty:
    CHUNK = 1 << 21

    def __init__(self, model, delta: float):
        self.model, self.delta = model, float(delta)
        parts = [(i.int(), j.int(), ui.half(), uj.half(), a) for i, j, ui, uj, a in face_samples(model)]
        self.leaf_i, self.leaf_j, self.u_i, self.u_j, self.area = (torch.cat(x) for x in zip(*parts))

    @property
    def samples(self) -> int:
        return int(self.area.numel())

    def _chunks(self):
        """(slice, w_i, idx_i, w_j, idx_j) per chunk of samples, at the model's current state."""
        kind, off = self.model.leaf_type(), self.model.value_offsets()
        for s in range(0, self.samples, self.CHUNK):
            sl = slice(s, s + self.CHUNK)
            wi, ii = basis_rows(self.model, self.leaf_i[sl].long(), self.u_i[sl].float(), kind, off)
            wj, jj = basis_rows(self.model, self.leaf_j[sl].long(), self.u_j[sl].float(), kind, off)
            yield sl, wi, ii, wj, jj

    def differences(self, c: torch.Tensor) -> torch.Tensor:
        return torch.cat([(wi * c[ii]).sum(1) - (wj * c[jj]).sum(1) for _, wi, ii, wj, jj in self._chunks()])

    def value_and_gradient(self, c: torch.Tensor) -> tuple[float, torch.Tensor]:
        d = self.delta
        value = 0.0
        g = torch.zeros_like(c)
        for sl, wi, ii, wj, jj in self._chunks():
            t = (wi * c[ii]).sum(1) - (wj * c[jj]).sum(1)
            area = self.area[sl]
            value += float((area * torch.where(t.abs() <= d, 0.5 * t * t, d * (t.abs() - 0.5 * d))).sum())
            slope = (area * t.clamp(-d, d))[:, None]
            g.index_add_(0, ii.reshape(-1), (slope * wi).reshape(-1))
            g.index_add_(0, jj.reshape(-1), (-slope * wj).reshape(-1))
        return value, g

    def curvature(self) -> torch.Tensor:
        d = torch.zeros(int(self.model.values.numel()), device=self.area.device)
        for sl, wi, ii, wj, jj in self._chunks():
            a = 2.0 * self.area[sl, None]
            d.index_add_(0, ii.reshape(-1), (a * wi).reshape(-1))
            d.index_add_(0, jj.reshape(-1), (a * wj).reshape(-1))
        return d
