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
step geometry (so the same samples serve a geometry update).  freeze() evaluates them once instead and keeps them
(two (n, 8) float32 sets and the first value index of each side, 72 bytes more per sample) for a solve, where tree,
types and geometry do not change and the weights were 78 % of every value-and-gradient call.

Ghost penalty (cut-cell stabilisation, Burman 2010): a step leaf's plane may cut off a small side whose value c_k few
rays see; the data then hardly constrain it.  Each side k of each step leaf L is extended to the whole leaf (the
"ghost" part) and tied to the neighbours' values on its side of the plane:
  R_ghost = ghost * sum_L sum_k A_L phi(c_k - T_k),   T_k = sum_p beta_k(p) f_O(p)
over L's face samples p (f_O the neighbour's function there), beta_k(p) proportional to area_p sigmoid(+-s(p) / TAU)
(s = signed distance to the plane in leaf units, TAU = a quarter leaf: samples near side k count most) and summing
to 1, A_L = L's sampled face area.  The weight of every side is A_L whatever the cut, so a value the data cannot see
is held at its neighbours' (the same edge-preserving phi: a real edge at the leaf's face is penalised only
linearly).  The side weights beta are evaluated from the step geometry when the penalty is built (and kept).
Curvature bound: t_k = c_k - T_k has |coefficients| summing to 2 (beta and the basis weights each sum to 1).
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
    TAU = 0.25  # width of the side weights of the ghost penalty, leaf units

    def __init__(self, model, delta: float, ghost: float):
        self.model, self.delta, self.ghost = model, float(delta), float(ghost)
        parts = [(i.int(), j.int(), ui.half(), uj.half(), a) for i, j, ui, uj, a in face_samples(model)]
        self.leaf_i, self.leaf_j, self.u_i, self.u_j, self.area = (torch.cat(x) for x in zip(*parts))
        self.w_i = self.w_j = self.first_i = self.first_j = None
        self._ghost_weights()

    def _ghost_weights(self):
        """beta (samples, 2): weights of each sample for the sides 0 / 1 of leaf i and of leaf j (0 where that leaf is
        not a step); the steps' sampled face areas (N,) and the value index of each side (N, 2)."""
        model, N = self.model, self.model.leaf_count
        kind, off = model.leaf_type(), model.value_offsets()
        self.ghost_area = torch.zeros(N, device=self.area.device)
        self.beta_i, self.beta_j = (torch.zeros((self.samples, 2), device=self.area.device) for _ in range(2))
        self.ghost_index = (off[:, None] + torch.arange(2, device=off.device)[None]).clamp_max(int(model.values.numel()) - 1)
        if self.ghost <= 0 or not bool((kind == 2).any()):
            return
        total = torch.zeros((N, 2), device=self.area.device)
        for leaf, u, beta in ((self.leaf_i, self.u_i, self.beta_i), (self.leaf_j, self.u_j, self.beta_j)):
            leaf = leaf.long()
            step = torch.nonzero(kind[leaf] == 2).flatten()
            g = model.geometry(leaf[step])
            dist = ((u[step].float() - 0.5) * g[:, :3]).sum(1) - g[:, 3]
            beta[step] = self.area[step, None] * torch.sigmoid(torch.stack([-dist, dist], 1) / self.TAU)
            total.index_add_(0, leaf[step], beta[step])
            self.ghost_area.index_add_(0, leaf[step], self.area[step])
        for leaf, beta in ((self.leaf_i, self.beta_i), (self.leaf_j, self.beta_j)):
            beta /= total[leaf.long()].clamp_min(1e-30)

    @property
    def samples(self) -> int:
        return int(self.area.numel())

    def freeze(self):
        """Keep the basis weights at the model's current types and step geometry until release()."""
        self.release()  # _chunks evaluates from the model while nothing is frozen
        n, off = self.samples, self.model.value_offsets()
        w_i, w_j = (torch.empty((n, 8), device=self.area.device) for _ in range(2))
        for sl, wi, _, wj, _ in self._chunks():
            w_i[sl], w_j[sl] = wi, wj
        self.w_i, self.w_j = w_i, w_j
        self.first_i, self.first_j = off[self.leaf_i.long()].int(), off[self.leaf_j.long()].int()
        return self

    def release(self):
        self.w_i = self.w_j = self.first_i = self.first_j = None

    def _chunks(self):
        """(slice, w_i, idx_i, w_j, idx_j) per chunk of samples, at the model's current state (or from freeze())."""
        kind, off = self.model.leaf_type(), self.model.value_offsets()
        frozen = self.w_i is not None
        last, k = int(self.model.values.numel()) - 1, torch.arange(8, device=self.area.device)[None]
        for s in range(0, self.samples, self.CHUNK):
            sl = slice(s, s + self.CHUNK)
            if frozen:  # padding weights are 0: their index only has to stay inside the value array
                yield (sl, self.w_i[sl], (self.first_i[sl, None] + k).clamp_max(last).long(),
                       self.w_j[sl], (self.first_j[sl, None] + k).clamp_max(last).long())
                continue
            wi, ii = basis_rows(self.model, self.leaf_i[sl].long(), self.u_i[sl].float(), kind, off)
            wj, jj = basis_rows(self.model, self.leaf_j[sl].long(), self.u_j[sl].float(), kind, off)
            yield sl, wi, ii, wj, jj

    def differences(self, c: torch.Tensor) -> torch.Tensor:
        return torch.cat([(wi * c[ii]).sum(1) - (wj * c[jj]).sum(1) for _, wi, ii, wj, jj in self._chunks()])

    def _phi(self, t):
        d = self.delta
        return torch.where(t.abs() <= d, 0.5 * t * t, d * (t.abs() - 0.5 * d))

    def value_and_gradient(self, c: torch.Tensor) -> tuple[float, torch.Tensor]:
        d = self.delta
        value = torch.zeros((), dtype=torch.float64, device=c.device)  # on the device: one sync at the end, not one per chunk
        g = torch.zeros_like(c)
        ghost = self.ghost > 0 and bool((self.ghost_area > 0).any())
        T = torch.zeros((self.model.leaf_count, 2), device=c.device)
        for sl, wi, ii, wj, jj in self._chunks():
            fi, fj = (wi * c[ii]).sum(1), (wj * c[jj]).sum(1)
            t = fi - fj
            area = self.area[sl]
            value += (area * self._phi(t)).sum().double()
            slope = (area * t.clamp(-d, d))[:, None]
            g.index_add_(0, ii.reshape(-1), (slope * wi).reshape(-1))
            g.index_add_(0, jj.reshape(-1), (-slope * wj).reshape(-1))
            if ghost:  # targets: each side's neighbour values, weighted by beta
                T.index_add_(0, self.leaf_i[sl].long(), self.beta_i[sl] * fj[:, None])
                T.index_add_(0, self.leaf_j[sl].long(), self.beta_j[sl] * fi[:, None])
        if ghost:
            A = self.ghost * self.ghost_area[:, None]  # 0 for leaves that are not steps
            t = c[self.ghost_index] - T
            value += (A * self._phi(t)).sum().double()
            s = A * t.clamp(-d, d)  # d/dc_k; the neighbours get -s beta_k(p) times their basis weights at p
            g.index_add_(0, self.ghost_index.reshape(-1), s.reshape(-1))
            for sl, wi, ii, wj, jj in self._chunks():
                si = -(s[self.leaf_i[sl].long()] * self.beta_i[sl]).sum(1, keepdim=True)  # sides of leaf i: neighbour j
                sj = -(s[self.leaf_j[sl].long()] * self.beta_j[sl]).sum(1, keepdim=True)
                g.index_add_(0, jj.reshape(-1), (si * wj).reshape(-1))
                g.index_add_(0, ii.reshape(-1), (sj * wi).reshape(-1))
        return float(value), g

    def curvature(self) -> torch.Tensor:
        d = torch.zeros(int(self.model.values.numel()), device=self.area.device)
        A = 2.0 * self.ghost * self.ghost_area[:, None].expand(-1, 2)  # 2 x weight of each ghost term (|coefficients| sum to 2)
        if self.ghost > 0:
            d.index_add_(0, self.ghost_index.reshape(-1), A.reshape(-1))
        for sl, wi, ii, wj, jj in self._chunks():
            a = 2.0 * self.area[sl, None]
            d.index_add_(0, ii.reshape(-1), (a * wi).reshape(-1))
            d.index_add_(0, jj.reshape(-1), (a * wj).reshape(-1))
            if self.ghost > 0:  # the neighbours' share of the ghost terms of leaf i's / leaf j's sides
                gi = (A[self.leaf_i[sl].long()] * self.beta_i[sl]).sum(1, keepdim=True)
                gj = (A[self.leaf_j[sl].long()] * self.beta_j[sl]).sum(1, keepdim=True)
                d.index_add_(0, jj.reshape(-1), (gi * wj).reshape(-1))
                d.index_add_(0, ii.reshape(-1), (gj * wi).reshape(-1))
        return d
