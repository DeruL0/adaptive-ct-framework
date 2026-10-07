"""Structure and leaf-type decision on a solved tree.

For every leaf the options are refitted in the same background (all other leaves fixed):
  keep · change type (constant / trilinear / step) · split into 8 children of the same type.
Rays: every pixel of a set of evenly spaced training views; alternate views form two halves.  Each option's
values are fitted on one half and its objective  sum (prediction - data)^2  is measured on the other half
(and the other way round), so extra parameters only win when they predict unseen view directions better.
With the solve's edge-preserving penalty (weight beta) each half's objective is  sum (prediction - data)^2 +
beta * R  (the whole objective  1/2 ||Ac - y||^2 + beta R  doubled and shared between the halves); R near the
leaf is replaced by the quadratic that touches it at the current values and lies above it (penalty_statistics),
so the penalty part of a gain never exceeds the true one.
Gain = objective(keep) - objective(option); every refit respects the constraint of the solve (attenuation >= 0),
so gains that would need negative values are not counted.  A leaf whose type changed in the previous round may only be
kept or split (no immediate change back).  Trilinear and step leaves are never downgraded to constant: the
detail it drops is largely invisible in held-out projections but not in the volume (run fabed08: projections
+0.7 dB, volume -0.6 dB with 472k downgrades).

Choice: each leaf takes the option maximising  gain - lambda * extra_bytes ; lambda >= 0 is the smallest
price of a byte that keeps this round's additions within `action_share` of the remaining budget (byte-saving
options are taken when the lost gain is below the bytes they free, and pay for this round's additions).
All chosen actions are executed; the budget fills geometrically over the rounds, and the tree is re-solved
before the next decision.
"""

from __future__ import annotations

import math

import torch

from adaptive_ct.codec.budget import leaf_bytes, model_bytes, split_bytes
from adaptive_ct.core.geometry import parallel_rays, traversal_grid
from adaptive_ct.core.projector import bernstein_octree_decision_accumulate
from adaptive_ct.representation.octree import TYPE_NAMES, pad_geometry, step_integral, trilinear_weights
from adaptive_ct.training.regularizer import basis_rows, face_samples

OPTIONS = ("constant", "trilinear", "step", "split")  # column order of gains / extra bytes
_DIMS = {"constant": 1, "trilinear": 8, "step": 2, "split": 8}
# Accumulated statistics per leaf and ray half (same layout as segment_decision in native/bernstein_octree.cu):
# G upper triangle (row major) and h per option, then sum b^2 and sum residual^2.
_SLOTS = {"constant": (0, 1, 2), "trilinear": (2, 38, 46), "step": (46, 49, 51), "split": (51, 87, 95)}  # G, h, end
STRIDE = 97


def resolvable_mask(model, *, detector_h: int, detector_w: int, min_footprint_px: float) -> torch.Tensor:
    """Leaves whose children still span at least `min_footprint_px` detector pixels along every axis
    (axes 0/1 project onto detector columns, axis 2 onto detector rows)."""
    shapes = torch.tensor(model.level_shapes, dtype=torch.float32, device=model.leaf_levels.device)
    child_width = 1.0 / shapes[model.leaf_levels]  # 2 / (2 * shape)
    pixel = torch.tensor([2.0 / detector_w, 2.0 / detector_w, 2.0 / detector_h], device=model.leaf_levels.device)
    fine_enough = torch.all(child_width >= float(min_footprint_px) * pixel[None, :] - 1.0e-9, dim=1)
    return fine_enough & (model.leaf_levels + 1 < len(model.level_shapes))


@torch.no_grad()
def candidate_step_geometry(model, *, detector_w: int, grid: int = 5, chunk: int = 20000) -> torch.Tensor:
    """(N, GEOMETRY_WIDTH) step geometry (n, d, w) per leaf from the current reconstruction: n = main direction of the
    gradient second-moment matrix over the leaf and a quarter-leaf margin, d = gradient-weighted centre along
    n, w = half a detector pixel (in leaf units)."""
    device = model.leaf_levels.device
    N = int(model.leaf_levels.shape[0])
    shapes = torch.tensor(model.level_shapes, dtype=torch.float32, device=device)[model.leaf_levels]
    lin = torch.linspace(-0.25, 1.25, grid, device=device)
    uu = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    out = torch.zeros((N, 5), device=device)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        size = 2.0 / shapes[a:b]
        pts = (-1.0 + model.leaf_coords[a:b].float() * size)[:, None, :] + uu[None] * size[:, None, :]
        f = model.forward_mu(pts.clamp(-1.0, 1.0).reshape(-1, 3)).reshape(b - a, grid, grid, grid)
        g = torch.stack(torch.gradient(f, spacing=float(lin[1] - lin[0]), dim=(1, 2, 3)), -1).reshape(b - a, -1, 3)
        n = torch.linalg.eigh(torch.einsum("npi,npj->nij", g, g).double())[1][:, :, -1].float()
        s = ((uu[None] - 0.5) * n[:, None, :]).sum(-1)
        weight = (g * n[:, None, :]).sum(-1).abs()
        extent = 0.5 * n.abs().sum(1)  # max |n . (u - 1/2)| over the cell: the plane stays well inside the leaf
        d = (weight * s).sum(1) / weight.sum(1).clamp_min(1e-12)
        d = torch.maximum(torch.minimum(d, 0.8 * extent), -0.8 * extent)
        out[a:b] = torch.cat([n, d[:, None], (0.5 * shapes[a:b, 0] / detector_w)[:, None]], 1)
    return pad_geometry(out)


def decision_views(split, count: int, offset: int) -> list[int]:
    """`count` training views evenly spread in angle (shifted by `offset` from round to round)."""
    order = sorted(range(int(split.angles.numel())), key=lambda v: float(split.angles[v]))
    step = max(1, len(order) // max(1, count))
    return order[offset % step :: step][:count]


def _pack(L, tri, step, child, contribution, residual) -> torch.Tensor:
    """(P, STRIDE) statistics of (ray, leaf) pairs (PyTorch reference of segment_decision)."""
    iu, ju = torch.triu_indices(8, 8, device=L.device)
    b = contribution - residual
    return torch.cat([(L * L)[:, None], (L * b)[:, None], tri[:, iu] * tri[:, ju], tri * b[:, None],
                      (step[:, 0] * step[:, 0])[:, None], (step[:, 0] * step[:, 1])[:, None], (step[:, 1] * step[:, 1])[:, None],
                      step * b[:, None], child[:, iu] * child[:, ju], -child * residual[:, None], (b * b)[:, None],
                      (residual * residual)[:, None]], 1)


def _accumulate_reference(model, batch, residual, fold, candidate, acc):
    """PyTorch reference of the CUDA accumulation (used on CPU)."""
    device = model.leaf_levels.device
    N = int(model.leaf_levels.shape[0])
    kind = model.leaf_type()
    coeff = model.values.detach()
    offs = model.value_offsets()
    ray_ids, leaf_ids, iv = model._parallel_ray_segments(batch)
    u0 = model.segment_local_coords(batch, ray_ids, leaf_ids, iv[:, 0])
    u1 = model.segment_local_coords(batch, ray_ids, leaf_ids, iv[:, 1])
    L = iv[:, 1] - iv[:, 0]
    S = L.numel()
    tri = sum(0.5 * L[:, None] * trilinear_weights(u0 + (u1 - u0) * q) for q in (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0)))

    def step_weights(g):
        I = step_integral(g, ((u0 - 0.5) * g[:, :3]).sum(1), ((u1 - 0.5) * g[:, :3]).sum(1), L)
        return torch.stack([L - I, I], 1)

    # Segments are cut at every finest-level boundary, so none crosses its leaf's mid-planes.
    um = 0.5 * (u0 + u1)
    child = torch.zeros((S, 8), device=device)
    child[torch.arange(S, device=device), (um[:, 0] > 0.5).long() * 4 + (um[:, 1] > 0.5).long() * 2 + (um[:, 2] > 0.5).long()] = L
    k = kind[leaf_ids]
    cur = torch.zeros(S, device=device)
    m = k == 0
    cur[m] = coeff[offs[leaf_ids[m]]] * L[m]
    m = k == 1
    cur[m] = (tri[m] * coeff[offs[leaf_ids[m], None] + torch.arange(8, device=device)]).sum(1)
    m = k == 2
    cur[m] = (step_weights(model.geometry(leaf_ids))[m] * coeff[offs[leaf_ids[m], None] + torch.arange(2, device=device)]).sum(1)
    uniq, inv = torch.unique(ray_ids * N + leaf_ids, return_inverse=True)
    feat = torch.zeros((uniq.numel(), 20), device=device).index_add_(
        0, inv, torch.cat([L[:, None], tri, step_weights(candidate[leaf_ids]), child, cur[:, None]], 1))
    pair_ray, pair_leaf = uniq // N, uniq % N
    acc[fold].index_add_(0, pair_leaf, _pack(feat[:, 0], feat[:, 1:9], feat[:, 9:11], feat[:, 11:19], feat[:, 19], residual[pair_ray]))


def step_side_fraction(geometry: torch.Tensor, samples: int = 4) -> torch.Tensor:
    """Volume fraction of the smaller side of each step (mean of H or 1 - H over samples^3 points of the leaf)."""
    q = (torch.arange(samples, device=geometry.device, dtype=torch.float32) + 0.5) / samples
    u = torch.stack(torch.meshgrid(q, q, q, indexing="ij"), -1).reshape(-1, 3)
    s = ((u[None] - 0.5) * geometry[:, None, :3]).sum(-1) - geometry[:, 3:4]
    w = geometry[:, 4:5]
    mean_h = torch.where(w > 0, torch.sigmoid(s / w.clamp_min(1e-30)), (s > 0).float()).mean(1)
    return torch.minimum(mean_h, 1 - mean_h)


@torch.no_grad()
def penalty_statistics(model, delta: float, candidate: torch.Tensor) -> torch.Tensor:
    """(N, STRIDE) edge-penalty terms of every option, in the layout of the ray statistics.
    Each shared-face sample p of leaf L (other side g = neighbour value, fixed) enters with weight
    alpha = area * min(1, delta / |t0|), t0 = f_L(p) - g: the quadratic that touches the penalty's
    phi(t) at the current t0 and lies above it everywhere (phi <= alpha t^2 / 2 + const).  Summed:
      option values c (basis a at p):  sum alpha (a.c - g)^2   -> G, h, b2 slots (target b = g)
      split corrections e (child of p): sum alpha (e - (g - f_L))^2  -> split G, h, r2 slots (residual = t0)
    plus the faces between the 8 children (current jump 0, alpha = area)."""
    device = model.leaf_levels.device
    N = int(model.leaf_levels.shape[0])
    coeff = model.values.detach()
    acc = torch.zeros((N, STRIDE), device=device)

    def value(ids, u):
        w, idx = basis_rows(model, ids, u)
        return (w * coeff[idx]).sum(1)

    for i_ids, j_ids, ui, uj, area in face_samples(model):
        fi, fj = value(i_ids, ui), value(j_ids, uj)
        s = (area * (float(delta) / (fi - fj).abs().clamp_min(1e-30)).clamp(max=1.0)).sqrt()
        for ids, u, f, g in ((i_ids, ui, fi, fj), (j_ids, uj, fj, fi)):
            tri = trilinear_weights(u)
            cg = candidate[ids]
            H = torch.sigmoid((((u - 0.5) * cg[:, :3]).sum(1) - cg[:, 3]) / cg[:, 4].clamp_min(1e-30))
            child = torch.nn.functional.one_hot((u[:, 0] > 0.5).long() * 4 + (u[:, 1] > 0.5).long() * 2 + (u[:, 2] > 0.5).long(), 8).float()
            acc.index_add_(0, ids, _pack(s, s[:, None] * tri, s[:, None] * torch.stack([1 - H, H], 1), s[:, None] * child, s * f, s * (f - g)))
    iu, ju = torch.triu_indices(8, 8, device=device)
    size = 2.0 / torch.tensor(model.level_shapes, dtype=torch.float32, device=device)[model.leaf_levels]
    for axis in range(3):
        bit = 4 >> axis  # child index = 4 x + 2 y + z
        M = torch.zeros((8, 8), device=device)
        for a in range(8):
            if not a & bit:
                e = torch.zeros(8, device=device)
                e[a], e[a | bit] = 1.0, -1.0
                M += torch.outer(e, e)
        t1, t2 = [b for b in range(3) if b != axis]
        acc[:, 51:87] += (size[:, t1] * size[:, t2] / 4.0)[:, None] * M[iu, ju][None]
    return acc


@torch.no_grad()
def local_statistics(model, split, views: list[int], candidate: torch.Tensor, prior: torch.Tensor | None = None) -> dict:
    """Per leaf and ray half f (alternate views): normal equations G[f] (N, m, m), h[f] (N, m) of every option,
    b2[f] = sum b^2 with b = data minus the other leaves (target of a type refit), r2[f] = sum residual^2.
    `prior` (N, STRIDE), e.g. the weighted penalty_statistics, is added to both halves."""
    device = model.leaf_levels.device
    N = int(model.leaf_levels.shape[0])
    acc = torch.zeros((2, N, STRIDE), device=device) if prior is None else prior[None].repeat(2, 1, 1)
    h, w = split.projections.shape[1:]
    rows, cols = traversal_grid(h, w, device)
    for position, v in enumerate(views):
        batch = parallel_rays(split.angles[v].expand(rows.numel()), rows, cols, detector_h=h, detector_w=w,
                              targets=split.projections[v, rows, cols])
        residual = model.integrate_ray_batch(batch).detach() - batch.target
        if model.uses_cuda():
            bernstein_octree_decision_accumulate(model.values, model.step_geometry, **model.tree_arguments(), rays=batch, candidate_geometry=candidate,
                                                 residual=residual, fold=torch.full((rows.numel(),), position % 2, device=device), accum=acc)
        else:
            _accumulate_reference(model, batch, residual, position % 2, candidate, acc)
    out = {"b2": acc[:, :, 95].double(), "r2": acc[:, :, 96].double()}
    for o, (g0, h0, end) in _SLOTS.items():
        m = _DIMS[o]
        iu, ju = torch.triu_indices(m, m, device=device)
        G = torch.zeros((2, N, m, m), device=device)
        G[:, :, iu, ju] = acc[:, :, g0:h0]
        G[:, :, ju, iu] = acc[:, :, g0:h0]
        out[f"G_{o}"], out[f"h_{o}"] = G.reshape(2, N, m * m), acc[:, :, h0:end]
    return out


def _fit(G: torch.Tensor, h: torch.Tensor, lower: torch.Tensor, iterations: int = 100, chunk: int = 65536) -> torch.Tensor:
    """Least squares G c = h per leaf (tiny ridge) subject to c >= lower (per leaf and unknown), by projected
    gradient.  Processed in chunks of leaves to keep the GPU memory peak small."""
    if h.shape[0] > chunk:
        return torch.cat([_fit(G[a : a + chunk], h[a : a + chunk], lower[a : a + chunk], iterations, chunk) for a in range(0, h.shape[0], chunk)])
    m = h.shape[1]
    A = G.reshape(-1, m, m).double()
    A = A + 1e-6 * A.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-12)[:, None, None] * torch.eye(m, dtype=A.dtype, device=A.device)
    hd, lo = h.double(), lower.double()
    step = 1.0 / torch.linalg.eigvalsh(A).amax(1).clamp_min(1e-30)
    c = torch.maximum(torch.linalg.solve(A, hd), lo)
    for _ in range(iterations):
        c = torch.maximum(c - step[:, None] * (torch.bmm(A, c[:, :, None])[:, :, 0] - hd), lo)
    return c.float()


def _objective(G, h, base, c):
    m = h.shape[1]
    return base - 2 * (h.double() * c.double()).sum(1) + torch.einsum("nm,nmk,nk->n", c.double(), G.reshape(-1, m, m).double(), c.double())


@torch.no_grad()
def child_minimum(model, inset: float = 1e-4) -> torch.Tensor:
    """(N, 8) minimum of the current leaf function over each child cell (over its 8 corners: exact for constant
    and trilinear leaves)."""
    device = model.leaf_levels.device
    N = int(model.leaf_levels.numel())
    g = torch.tensor([inset, 0.5, 1.0 - inset], device=device)
    lattice = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)  # 3 x 3 x 3 points, i-major
    shapes = torch.tensor(model.level_shapes, dtype=torch.float32, device=device)[model.leaf_levels]
    size = 2.0 / shapes
    out = torch.empty((N, 8), device=device)
    child_corners = torch.tensor([[(cx + i) * 9 + (cy + j) * 3 + (cz + k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
                                  for cx in (0, 1) for cy in (0, 1) for cz in (0, 1)], device=device)  # (8 children, 8 corners)
    for a in range(0, N, 100000):
        b = min(N, a + 100000)
        pts = (-1 + model.leaf_coords[a:b].float() * size[a:b])[:, None, :] + lattice[None] * size[a:b, None, :]
        f = model.forward_mu(pts.reshape(-1, 3)).reshape(b - a, 27)
        out[a:b] = f[:, child_corners].min(-1).values
    return out


def evaluate_options(acc: dict, split_lower: torch.Tensor) -> torch.Tensor:
    """Held-out gains (N, 4) for OPTIONS.  Every fit respects what the solve can reach: type values >= 0, and
    split corrections >= split_lower (= -child minimum, so the children stay nonnegative)."""
    gains = []
    for o in OPTIONS:
        G, h = acc[f"G_{o}"], acc[f"h_{o}"]
        base = acc["r2"] if o == "split" else acc["b2"]
        lower = split_lower if o == "split" else torch.zeros_like(h[0])
        gain = torch.zeros_like(acc["r2"][0])
        for fit_half, eval_half in ((0, 1), (1, 0)):
            c = _fit(G[fit_half], h[fit_half], lower)
            gain += acc["r2"][eval_half] - _objective(G[eval_half], h[eval_half], base[eval_half], c)
        gains.append(gain)
    gains = torch.stack(gains, 1)
    # gains within floating-point noise of the residual energy (the leaf's, or the average leaf's) count as zero
    r2 = acc["r2"][0] + acc["r2"][1]
    return torch.where(gains.abs() <= 1e-6 * (r2 + r2.mean())[:, None], torch.zeros_like(gains), gains)


def choose(gains: torch.Tensor, extra: torch.Tensor, allowed: torch.Tensor, allowance: float):
    """Leaves and options (index into OPTIONS) taking the best net benefit under the smallest byte price lambda
    whose total extra bytes stay within `allowance`."""
    net_at = lambda lam: torch.where(allowed, gains - lam * extra, torch.full_like(gains, -math.inf))

    def pick(lam):
        best, idx = net_at(lam).max(1)
        idx = torch.where(best > 0, idx, -1)
        spent = float(extra.gather(1, idx.clamp_min(0)[:, None])[:, 0][idx >= 0].sum())
        return idx, best, spent

    lam = 0.0
    idx, best, spent = pick(0.0)
    if spent > allowance:
        lo, hi = 0.0, float(gains.clamp_min(0).max()) + 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if pick(mid)[2] > allowance:
                lo = mid
            else:
                hi = mid
        lam = hi
        idx, best, spent = pick(lam)
    chosen = torch.nonzero(idx >= 0).flatten()
    return chosen, idx[chosen], lam


@torch.no_grad()
def project_onto_types(model, ids: torch.Tensor, types: torch.Tensor, geometry: torch.Tensor, samples: int = 4) -> torch.Tensor:
    """Values (concatenated in leaf order) of the new-type functions closest, in least squares over
    samples^3 points inside each leaf, to the current leaf functions; nonnegative."""
    device = model.leaf_levels.device
    g = (torch.arange(samples, device=device, dtype=torch.float32) + 0.5) / samples
    u = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)  # (P, 3) leaf-local
    shapes = torch.tensor(model.level_shapes, dtype=torch.float32, device=device)[model.leaf_levels[ids]]
    size = 2.0 / shapes
    points = (-1.0 + model.leaf_coords[ids].float() * size)[:, None, :] + u[None] * size[:, None, :]
    f = torch.cat([model.forward_mu(points[a : a + 32768].reshape(-1, 3)).reshape(-1, u.shape[0])  # (n, P), in chunks of leaves
                   for a in range(0, ids.numel(), 32768)])
    tri = trilinear_weights(u)
    counts = torch.tensor([1, 8, 2], device=device)[types]
    starts = torch.cumsum(counts, 0) - counts
    out = torch.zeros(int(counts.sum()), device=device)
    for t in range(3):
        sel = torch.nonzero(types == t).flatten()
        if sel.numel() == 0:
            continue
        if t == 2:
            gs = geometry[sel]
            s = ((u[None] - 0.5) * gs[:, None, :3]).sum(-1) - gs[:, 3:4]
            H = torch.sigmoid(s / gs[:, 4:5].clamp_min(1e-30))
            B = torch.stack([1 - H, H], -1)  # (n, P, 2)
            G = torch.einsum("npi,npj->nij", B, B).reshape(sel.numel(), -1)
            h = torch.einsum("npi,np->ni", B, f[sel])
        else:  # the same basis in every leaf
            B = torch.ones((samples ** 3, 1), device=device) if t == 0 else tri
            G = (B.T @ B).reshape(1, -1).expand(sel.numel(), -1)
            h = f[sel] @ B
        c = _fit(G, h, torch.zeros_like(h))
        out[(starts[sel, None] + torch.arange(c.shape[1], device=device)).reshape(-1)] = c.reshape(-1)
    return out


@torch.no_grad()
def decide_and_apply(model, split, views, *, detector_h, detector_w, byte_budget, action_share, min_footprint_px,
                     penalty_weight, penalty_delta, frozen=None) -> tuple[dict, torch.Tensor]:
    """One structure / type decision on a solved model; applies the chosen batch.  `frozen` (N,) marks leaves
    that may not change type.  Returns a report and the cells (level, x, y, z) whose type changed."""
    device = model.leaf_levels.device
    N = int(model.leaf_levels.shape[0])
    kind = model.leaf_type()
    candidate = candidate_step_geometry(model, detector_w=detector_w)
    prior = 0.5 * float(penalty_weight) * penalty_statistics(model, penalty_delta, candidate) if penalty_weight > 0 else None
    acc = local_statistics(model, split, views, candidate, prior)
    del prior
    gains = evaluate_options(acc, -child_minimum(model))
    del acc  # free the statistics before the edits
    per_leaf = leaf_bytes().to(device)
    extra = torch.stack([per_leaf[0] - per_leaf[kind], per_leaf[1] - per_leaf[kind], per_leaf[2] - per_leaf[kind],
                         split_bytes(kind)], 1)
    type_allowed = torch.tensor([t in model.leaf_types for t in TYPE_NAMES], device=device)
    allowed = torch.zeros((N, 4), dtype=torch.bool, device=device)
    allowed[:, :3] = type_allowed[None, :] & (torch.arange(3, device=device)[None, :] != kind[:, None])
    allowed[:, 2] |= type_allowed[2] & (kind == 2)  # a step leaf may take the newly estimated geometry
    allowed[:, 2] &= step_side_fraction(candidate) >= 0.02  # a step must split its leaf, not shave a corner
    allowed[:, 0] &= kind == 0  # no downgrade to constant: what it drops is largely invisible in projections but not in the volume
    if frozen is not None:
        allowed[:, :3] &= ~frozen[:, None]
    allowed[:, 3] = resolvable_mask(model, detector_h=detector_h, detector_w=detector_w, min_footprint_px=min_footprint_px)
    bytes_now = model_bytes(model)
    remaining = float(byte_budget - bytes_now)
    ids, option, lam = choose(gains, extra, allowed, action_share * max(remaining, 0.0))
    if model.max_leaf_count is not None:  # leaf-count cap: drop splits beyond it
        room = (int(model.max_leaf_count) - N) // 7
        is_split = option == 3
        drop = is_split & (torch.cumsum(is_split.long(), 0) > room)
        ids, option = ids[~drop], option[~drop]
    report = {
        "byte_price": lam,
        "bytes_before": bytes_now,
        "candidates_with_positive_gain": {o: int(((gains[:, i] > 0) & allowed[:, i]).sum()) for i, o in enumerate(OPTIONS)},
        "actions": {o: int((option == i).sum()) for i, o in enumerate(OPTIONS)},
        "held_out_gain_of_batch": float(gains[ids, option].sum()) if ids.numel() else 0.0,
    }
    if ids.numel() == 0:
        report["bytes_after"] = bytes_now
        return report, torch.zeros((0, 4), dtype=torch.long, device=device)
    # 1. type changes, initialised with the closest function of the new type to the current leaf function
    #    (refits of many leaves made in isolation do not fit together; the next solve starts near the solved state)
    change = option < 3
    t_ids, t_new = ids[change], option[change]
    changed_cells = torch.cat([model.leaf_levels[t_ids, None], model.leaf_coords[t_ids]], 1)
    split_cells = model.cells[ids[~change]].long()  # leaf indices change with every edit; cells do not
    if t_ids.numel():
        geometry = torch.where((t_new == 2)[:, None], candidate[t_ids], torch.zeros_like(candidate[t_ids]))
        vals = project_onto_types(model, t_ids, t_new, geometry)
        model.set_types(t_ids, t_new, vals, geometry)
    # 2. splits (function preserving)
    if split_cells.numel():
        model.split_leaves_batch(model.ids_of_cells(split_cells))
    report["bytes_after"] = model_bytes(model)
    report["leaf_type_counts"] = dict(zip(TYPE_NAMES, torch.bincount(model.leaf_type(), minlength=3).tolist()))
    return report, changed_cells
