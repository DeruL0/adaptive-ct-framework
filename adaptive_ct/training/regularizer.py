"""Edge-preserving penalty between neighbouring leaves (fixed tree, leaf types and step geometry).

R(c) = sum over shared faces of  integral h phi([f] / h) dS  ~  sum over face samples p of  area_p h_p phi(t_p / h_p)
  t_p = f_i(p) - f_j(p): the jump of the two leaves' functions (linear in the stored values c, nonnegative weights)
  h_p : the distance between the two leaves' centres along the face normal (half the sum of their sizes)
  phi(x): Huber, x^2 / 2 for |x| <= delta, delta (|x| - delta / 2) beyond; delta is a gradient (value per box unit).
The jump is priced as a gradient t / h over the distance between the leaves (two-point gradient of finite volumes):
small jumps cost area t^2 / (2 h) (the gradient energy, symmetric interior penalty weight 1 / h), large ones about
area delta |t| (total variation, independent of h): coarse and fine leaves are held to the same smoothness.
Each face patch between two leaves is sampled at the 2 x 2 Gauss-Legendre points of the finer leaf's face
(1/2 +- 1/(2 sqrt 3), a quarter of the area each: exact for the squared jump of constant / trilinear leaves).
Separable curvature bound for the solver: a_p = b_i(p) - b_j(p) has |a_p| summing to 2 (two nonnegative
weight sets summing to 1 each) and (h phi(t / h))'' <= 1 / h, so  <= diag(sum_p 2 area_p / h_p |a_p|).
Stored per sample: the two leaf indices (int32), both leaf-local points (fp16, error <= 2.5e-4 leaf) and area / h,
24 bytes; the basis weights are evaluated in chunks at every use from the model's current types, values and
step geometry (so the same samples serve a geometry update).  freeze() evaluates them once instead and keeps them
(two (n, 8) float32 sets and the first value index of each side, 72 bytes more per sample) for a solve, where tree,
types and geometry do not change and the weights were 78 % of every value-and-gradient call.
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


def face_samples(model, merge_constant_pairs: bool):
    """Shared-face sample points, in 24 chunks: (finer leaf i, other leaf j, u_i, u_j leaf-local, area, h = distance of
    the two leaves' centres along the face normal).  merge_constant_pairs: a patch between two constant leaves has the
    same jump at its 4 points: one point with the patch's area (73 % of the points of a 4.6 M-leaf tree; exact for the
    penalty, not for statistics that need the points' positions, e.g. the decision's child corrections)."""
    device = model.leaf_levels.device
    kind = model.leaf_type()
    N = int(model.leaf_levels.numel())
    shapes = torch.tensor(model.level_shapes, dtype=torch.float32, device=device)
    size = 2.0 / shapes[model.leaf_levels]  # (N, 3) box units
    origin = -1.0 + model.leaf_coords.float() * size
    eps = 1e-4 * float((2.0 / shapes[-1]).min())
    q = 0.5 + torch.tensor([-0.5, 0.5], device=device) / 3 ** 0.5  # Gauss-Legendre
    for axis in range(3):
        t1, t2 = [a for a in range(3) if a != axis]
        for side in (0.0, 1.0):
            for ia, a in enumerate(q):
                for ib, b in enumerate(q):
                    u = torch.zeros((N, 3), device=device)
                    u[:, axis], u[:, t1], u[:, t2] = side, a, b
                    p = origin + u * size
                    outside = p.clone()
                    outside[:, axis] += eps if side == 1.0 else -eps
                    j = model.resolve_leaf_ids(outside)
                    keep = j >= 0
                    lj = torch.where(keep, model.leaf_levels[j.clamp_min(0)], 0)
                    keep &= (lj < model.leaf_levels) | ((lj == model.leaf_levels) & (side == 1.0))
                    area = torch.full((N,), 0.25, device=device)
                    if merge_constant_pairs:
                        both = (kind == 0) & (kind[j.clamp_min(0)] == 0)
                        if ia == 0 and ib == 0:
                            area = torch.where(both, 1.0, area)
                        else:
                            keep &= ~both
                    i_ids, j_ids = torch.nonzero(keep).flatten(), j[keep]
                    yield (i_ids, j_ids, u[keep], (p[keep] - origin[j_ids]) / size[j_ids], size[i_ids, t1] * size[i_ids, t2] * area[keep],
                           0.5 * (size[i_ids, axis] + size[j_ids, axis]))


@torch.no_grad()
def face_jump_energy(model, delta: float, chunk: int = 1 << 21) -> torch.Tensor:
    """(N,) float64 edge-penalty energy on each leaf's faces, sum of area h phi(t / h) over its face samples (each sample
    counted for both leaves): a face-jump error indicator (the jump terms of residual a posteriori estimators)."""
    N = int(model.leaf_levels.numel())
    kind, off, c = model.leaf_type(), model.value_offsets(), model.values.detach()
    out = torch.zeros(N, dtype=torch.float64, device=c.device)
    for i_ids, j_ids, ui, uj, area, h in face_samples(model, merge_constant_pairs=True):
        for a in range(0, int(area.numel()), chunk):
            sl = slice(a, a + chunk)
            wi, ii = basis_rows(model, i_ids[sl], ui[sl], kind, off)
            wj, jj = basis_rows(model, j_ids[sl], uj[sl], kind, off)
            x = ((wi * c[ii]).sum(1) - (wj * c[jj]).sum(1)) / h[sl]
            e = (area[sl] * h[sl] * torch.where(x.abs() <= delta, 0.5 * x * x, delta * (x.abs() - 0.5 * delta))).double()
            out.index_add_(0, i_ids[sl], e)
            out.index_add_(0, j_ids[sl], e)
    return out


class EdgePenalty:
    CHUNK = 1 << 21

    def __init__(self, model, delta: float):
        self.model, self.delta = model, float(delta)
        parts = [(i.int(), j.int(), ui.half(), uj.half(), a / h, h) for i, j, ui, uj, a, h in face_samples(model, merge_constant_pairs=True)]
        self.leaf_i, self.leaf_j, self.u_i, self.u_j, self.weight, self.h = (torch.cat(x) for x in zip(*parts))  # weight = area / h
        self.rows_i = self.rows_j = self.table = self.offsets = None

    @property
    def samples(self) -> int:
        return int(self.weight.numel())

    def freeze(self):
        """Keep the basis weights at the model's current types and step geometry until release(): one row of a table of
        8 weights for every sample side in a trilinear or step leaf (a constant leaf's weight is 1), 4 bytes per side."""
        self.release()  # _chunks evaluates from the model while nothing is frozen
        kind, dev = self.model.leaf_type(), self.weight.device
        nc_i, nc_j = kind[self.leaf_i.long()] != 0, kind[self.leaf_j.long()] != 0
        n_i = int(nc_i.sum())
        rows_i, rows_j = (torch.full((self.samples,), -1, dtype=torch.int32, device=dev) for _ in range(2))
        rows_i[nc_i] = torch.arange(n_i, dtype=torch.int32, device=dev)
        rows_j[nc_j] = torch.arange(n_i, n_i + int(nc_j.sum()), dtype=torch.int32, device=dev)
        table = torch.empty((n_i + int(nc_j.sum()), 8), device=dev)
        for sl, wi, _, wj, _ in self._chunks():
            ri, rj = rows_i[sl].long(), rows_j[sl].long()
            table[ri[ri >= 0]], table[rj[rj >= 0]] = wi[ri >= 0], wj[rj >= 0]
        self.rows_i, self.rows_j, self.table, self.offsets = rows_i, rows_j, table, self.model.value_offsets()
        return self

    def release(self):
        self.rows_i = self.rows_j = self.table = self.offsets = None

    def _chunks(self):
        """(slice, w_i, idx_i, w_j, idx_j) per chunk of samples, at the model's current state (or from freeze())."""
        frozen = self.table is not None
        kind, off = (None, self.offsets) if frozen else (self.model.leaf_type(), self.model.value_offsets())
        last, k = int(self.model.values.numel()) - 1, torch.arange(8, device=self.weight.device)[None]

        def from_table(rows):
            w = torch.zeros((rows.numel(), 8), device=self.weight.device)
            w[:, 0] = (rows < 0).float()  # a constant leaf
            w[rows >= 0] = self.table[rows[rows >= 0]]
            return w

        for s in range(0, self.samples, self.CHUNK):
            sl = slice(s, s + self.CHUNK)
            if frozen:  # padding weights are 0: their index only has to stay inside the value array
                yield (sl, from_table(self.rows_i[sl].long()), (off[self.leaf_i[sl].long(), None] + k).clamp_max(last),
                       from_table(self.rows_j[sl].long()), (off[self.leaf_j[sl].long(), None] + k).clamp_max(last))
                continue
            wi, ii = basis_rows(self.model, self.leaf_i[sl].long(), self.u_i[sl].float(), kind, off)
            wj, jj = basis_rows(self.model, self.leaf_j[sl].long(), self.u_j[sl].float(), kind, off)
            yield sl, wi, ii, wj, jj

    def differences(self, c: torch.Tensor) -> torch.Tensor:
        return torch.cat([(wi * c[ii]).sum(1) - (wj * c[jj]).sum(1) for _, wi, ii, wj, jj in self._chunks()])

    def value_and_gradient(self, c: torch.Tensor) -> tuple[float, torch.Tensor]:
        d = self.delta
        value = torch.zeros((), dtype=torch.float64, device=c.device)  # on the device: one sync at the end, not one per chunk
        g = torch.zeros_like(c)
        for sl, wi, ii, wj, jj in self._chunks():
            t = (wi * c[ii]).sum(1) - (wj * c[jj]).sum(1)
            w, h = self.weight[sl], self.h[sl]  # area / h, h
            x = t / h
            value += (w * h * h * torch.where(x.abs() <= d, 0.5 * x * x, d * (x.abs() - 0.5 * d))).sum().double()
            slope = (w * h * x.clamp(-d, d))[:, None]
            g.index_add_(0, ii.reshape(-1), (slope * wi).reshape(-1))
            g.index_add_(0, jj.reshape(-1), (-slope * wj).reshape(-1))
        return float(value), g

    def geometry_statistics(self, c: torch.Tensor) -> torch.Tensor:
        """(n_step, 20) Gauss-Newton statistics of R with respect to each step leaf's geometry theta = (n, d, w), in the
        layout of training/geometry.py: gradient (5), then the upper triangle of the 5 x 5 matrix.  Per sample
        t = f_i - f_j, x = t / h; a step side contributes  +-area phi'(x) df/dtheta  to the gradient and
        (area / h) alpha df/dtheta df/dtheta^T  to the matrix, alpha = min(1, delta / |x|): the quadratic that touches
        h phi(t / h) at t and lies above it (as in the decision's penalty statistics).  The other leaf is held fixed (Jacobi).
        df/dtheta = (c1 - c0) H' (s / w)' with s = n . (u - 1/2) - d, H = sigmoid(s / w):  ((u - 1/2) / w, -1 / w, -s / w^2)."""
        model, d = self.model, self.delta
        n_step = int(model.n_step)
        out = torch.zeros((n_step, 20), device=c.device)
        if n_step == 0:
            return out
        kind, off = model.leaf_type(), model.value_offsets()
        iu, ju = torch.triu_indices(5, 5, device=c.device)
        for sl, wi, ii, wj, jj in self._chunks():
            t = (wi * c[ii]).sum(1) - (wj * c[jj]).sum(1)
            w, h = self.weight[sl], self.h[sl]  # area / h, h
            x = t / h
            slope, alpha = w * h * x.clamp(-d, d), w * (d / x.abs().clamp_min(1e-30)).clamp(max=1.0)
            for leaf, u, sign in ((self.leaf_i[sl].long(), self.u_i[sl].float(), 1.0), (self.leaf_j[sl].long(), self.u_j[sl].float(), -1.0)):
                m = torch.nonzero(kind[leaf] == 2).flatten()
                g = model.geometry(leaf[m])
                w = g[:, 4]
                m, g, w, um = m[w > 0], g[w > 0], w[w > 0], u[m][w > 0] - 0.5  # hard steps have no derivative
                s = (um * g[:, :3]).sum(1) - g[:, 3]
                H = torch.sigmoid(s / w)
                k = (c[off[leaf[m]] + 1] - c[off[leaf[m]]]) * H * (1 - H) / w
                D = torch.cat([k[:, None] * um, -k[:, None], (-k * s / w)[:, None]], 1)
                rows = torch.cat([sign * slope[m, None] * D, alpha[m, None] * D[:, iu] * D[:, ju]], 1)
                out.index_add_(0, leaf[m] - model.n_tri, rows)
        return out

    def curvature(self) -> torch.Tensor:
        d = torch.zeros(int(self.model.values.numel()), device=self.weight.device)
        for sl, wi, ii, wj, jj in self._chunks():
            a = 2.0 * self.weight[sl, None]
            d.index_add_(0, ii.reshape(-1), (a * wi).reshape(-1))
            d.index_add_(0, jj.reshape(-1), (a * wj).reshape(-1))
        return d
