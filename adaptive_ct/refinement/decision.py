"""Structure and leaf-type decision on a solved tree.

For every leaf the options are refitted in the same background (all other leaves fixed):
  keep · change type (constant / trilinear / step) · split into 8 children of the same type.
Rays: every pixel of a set of evenly spaced training views; alternate views form two halves.  Each option's
values are fitted on one half and its objective  sum (prediction - data)^2  is measured on the other half
(and the other way round), so extra parameters only win when they predict unseen view directions better.
Gain = objective(keep) - objective(option).  A leaf whose type changed in the previous round may only be
kept or split (no immediate change back).

Choice: each leaf takes the option maximising  gain - lambda * extra_bytes ; lambda >= 0 is the smallest
price of a byte that keeps the model within the byte budget (byte-saving options are taken when the lost
gain is below the bytes they free).  Per round, in order of net benefit, byte-saving actions are executed
and byte-adding ones until they spend `action_share` of the remaining bytes; the tree is re-solved before
the next decision.
"""

from __future__ import annotations

import math

import torch

from adaptive_ct.codec.budget import leaf_bytes, model_bytes, split_bytes
from adaptive_ct.core.geometry import detector_grid, parallel_rays
from adaptive_ct.core.projector import bernstein_octree_decision_accumulate
from adaptive_ct.representation.octree import TYPE_NAMES, step_integral, type_ids

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
    """(N, 5) step geometry (n, d, w) per leaf from the current reconstruction: n = main direction of the
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
        d = ((weight * s).sum(1) / weight.sum(1).clamp_min(1e-12)).clamp(-0.85, 0.85)
        out[a:b] = torch.cat([n, d[:, None], (0.5 * shapes[a:b, 0] / detector_w)[:, None]], 1)
    return out


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
    kind = type_ids(model.leaf_degrees)
    coeff = model.coefficients().detach()
    offs = model.coefficient_offsets
    ray_ids, leaf_ids, iv = model._parallel_ray_segments(batch)
    u0 = model.segment_local_coords(batch, ray_ids, leaf_ids, iv[:, 0])
    u1 = model.segment_local_coords(batch, ray_ids, leaf_ids, iv[:, 1])
    L = iv[:, 1] - iv[:, 0]
    S = L.numel()
    tri = torch.zeros((S, 8), device=device)
    for qq in (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0)):  # exact: cubic along the segment
        u = u0 + (u1 - u0) * qq
        p = torch.stack([1 - u, u], -1)
        tri += 0.5 * L[:, None] * (p[:, 0, :, None, None] * p[:, 1, None, :, None] * p[:, 2, None, None, :]).reshape(S, 8)

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
    cur[m] = (step_weights(model.step_geometry[leaf_ids])[m] * coeff[offs[leaf_ids[m], None] + torch.arange(2, device=device)]).sum(1)
    uniq, inv = torch.unique(ray_ids * N + leaf_ids, return_inverse=True)
    feat = torch.zeros((uniq.numel(), 20), device=device).index_add_(
        0, inv, torch.cat([L[:, None], tri, step_weights(candidate[leaf_ids]), child, cur[:, None]], 1))
    pair_ray, pair_leaf = uniq // N, uniq % N
    acc[fold].index_add_(0, pair_leaf, _pack(feat[:, 0], feat[:, 1:9], feat[:, 9:11], feat[:, 11:19], feat[:, 19], residual[pair_ray]))


@torch.no_grad()
def local_statistics(model, split, views: list[int], candidate: torch.Tensor) -> dict:
    """Per leaf and ray half f (alternate views): normal equations G[f] (N, m, m), h[f] (N, m) of every option,
    b2[f] = sum b^2 with b = data minus the other leaves (target of a type refit), r2[f] = sum residual^2."""
    device = model.leaf_levels.device
    N = int(model.leaf_levels.shape[0])
    acc = torch.zeros((2, N, STRIDE), device=device)
    h, w = split.projections.shape[1:]
    rows, cols = detector_grid(h, w, device)
    for position, v in enumerate(views):
        batch = parallel_rays(split.angles[v].expand(rows.numel()), rows, cols, detector_h=h, detector_w=w,
                              targets=split.projections[v].reshape(-1))
        residual = model.integrate_ray_batch(batch).detach() - batch.target
        if model._can_use_cuda_integrator(batch):
            bernstein_octree_decision_accumulate(**model._cuda_integrator_arguments(batch), candidate_geometry=candidate, residual=residual,
                                                 fold=torch.full((rows.numel(),), position % 2, device=device), accum=acc)
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


def _fit(G: torch.Tensor, h: torch.Tensor, nonneg: bool, iterations: int = 100) -> torch.Tensor:
    """Least squares G c = h per leaf (tiny ridge); nonnegative by projected gradient when asked."""
    m = h.shape[1]
    A = G.reshape(-1, m, m).double()
    A = A + 1e-6 * A.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-12)[:, None, None] * torch.eye(m, dtype=A.dtype, device=A.device)
    hd = h.double()
    c = torch.linalg.solve(A, hd)
    if nonneg:
        step = 1.0 / torch.linalg.eigvalsh(A).amax(1).clamp_min(1e-30)
        c = c.clamp_min(0)
        for _ in range(iterations):
            c = (c - step[:, None] * (torch.bmm(A, c[:, :, None])[:, :, 0] - hd)).clamp_min(0)
    return c.float()


def _objective(G, h, base, c):
    m = h.shape[1]
    return base - 2 * (h.double() * c.double()).sum(1) + torch.einsum("nm,nmk,nk->n", c.double(), G.reshape(-1, m, m).double(), c.double())


@torch.no_grad()
def evaluate_options(acc: dict) -> tuple[torch.Tensor, dict]:
    """Held-out gains (N, 4) for OPTIONS and the values refitted on all rays (split: per-child offsets)."""
    gains, values = [], {}
    for o in OPTIONS:
        G, h = acc[f"G_{o}"], acc[f"h_{o}"]
        base = acc["r2"] if o == "split" else acc["b2"]
        gain = torch.zeros_like(acc["r2"][0])
        for fit_half, eval_half in ((0, 1), (1, 0)):
            c = _fit(G[fit_half], h[fit_half], nonneg=o != "split")
            gain += acc["r2"][eval_half] - _objective(G[eval_half], h[eval_half], base[eval_half], c)
        gains.append(gain)
        values[o] = _fit(G[0] + G[1], h[0] + h[1], nonneg=o != "split")
    return torch.stack(gains, 1), values


def choose(gains: torch.Tensor, extra: torch.Tensor, allowed: torch.Tensor, remaining: float, share: float):
    """Per leaf option index (-1 keep) under the byte price lambda, then the executed batch."""
    net_at = lambda lam: torch.where(allowed, gains - lam * extra, torch.full_like(gains, -math.inf))

    def pick(lam):
        best, idx = net_at(lam).max(1)
        idx = torch.where(best > 0, idx, -1)
        spent = float(extra.gather(1, idx.clamp_min(0)[:, None])[:, 0][idx >= 0].sum())
        return idx, best, spent

    lam = 0.0
    idx, best, spent = pick(0.0)
    if spent > remaining:
        lo, hi = 0.0, float(gains.clamp_min(0).max()) + 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if pick(mid)[2] > remaining:
                lo = mid
            else:
                hi = mid
        lam = hi
        idx, best, spent = pick(lam)
    chosen = torch.nonzero(idx >= 0).flatten()
    if chosen.numel() == 0:
        return chosen, idx[chosen], lam
    # Executed batch: in order of net benefit, byte-saving actions always, byte-adding ones while they stay
    # within `share` of the remaining bytes (the budget fills geometrically over the rounds).
    order = chosen[torch.argsort(best[chosen], descending=True)]
    added = extra[order, idx[order]]
    spent_so_far = torch.cumsum(added.clamp_min(0), 0)
    order = order[(added <= 0) | (spent_so_far <= share * max(remaining, 0.0))]
    return order, idx[order], lam


@torch.no_grad()
def decide_and_apply(model, split, views, *, detector_h, detector_w, byte_budget, quantization="float16",
                     action_share=0.5, min_footprint_px=1.0, frozen=None) -> tuple[dict, torch.Tensor]:
    """One structure / type decision on a solved model; applies the chosen batch.  `frozen` (N,) marks leaves
    that may not change type.  Returns a report and the cells (level, x, y, z) whose type changed."""
    device = model.leaf_levels.device
    N = int(model.leaf_levels.shape[0])
    kind = type_ids(model.leaf_degrees)
    candidate = candidate_step_geometry(model, detector_w=detector_w)
    acc = local_statistics(model, split, views, candidate)
    gains, values = evaluate_options(acc)
    per_leaf = leaf_bytes(quantization).to(device)
    extra = torch.stack([per_leaf[0] - per_leaf[kind], per_leaf[1] - per_leaf[kind], per_leaf[2] - per_leaf[kind],
                         split_bytes(kind, quantization)], 1).double()
    type_allowed = torch.tensor([t in model.leaf_types for t in TYPE_NAMES], device=device)
    allowed = torch.zeros((N, 4), dtype=torch.bool, device=device)
    allowed[:, :3] = type_allowed[None, :] & (torch.arange(3, device=device)[None, :] != kind[:, None])
    allowed[:, 2] |= type_allowed[2] & (kind == 2)  # a step leaf may take the newly estimated geometry
    if frozen is not None:
        allowed[:, :3] &= ~frozen[:, None]
    allowed[:, 3] = resolvable_mask(model, detector_h=detector_h, detector_w=detector_w, min_footprint_px=min_footprint_px)
    bytes_now = model_bytes(model, quantization)
    remaining = float(byte_budget - bytes_now)
    ids, option, lam = choose(gains, extra, allowed, remaining, action_share)
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
    # 1. type changes, initialised with the refitted values
    change = option < 3
    t_ids, t_new = ids[change], option[change]
    changed_cells = torch.cat([model.leaf_levels[t_ids, None], model.leaf_coords[t_ids]], 1)
    if t_ids.numel():
        counts = torch.tensor([1, 8, 2], device=device)[t_new]
        starts = torch.cumsum(counts, 0) - counts
        vals = torch.zeros(int(counts.sum()), device=device)  # concatenated in leaf order
        for t in range(3):
            sel = t_new == t
            if sel.any():
                v = values[TYPE_NAMES[t]][t_ids[sel]]
                vals[(starts[sel, None] + torch.arange(v.shape[1], device=device)).reshape(-1)] = v.reshape(-1)
        geometry = torch.where((t_new == 2)[:, None], candidate[t_ids], torch.zeros_like(candidate[t_ids]))
        model.set_types(t_ids, t_new, vals, geometry)
    # 2. splits (function preserving); leaf ids shifted by the replaced leaves
    s_ids = ids[~change]
    if s_ids.numel():
        removed = torch.sort(t_ids).values
        model.split_leaves_batch(s_ids - torch.searchsorted(removed, s_ids))
    report["bytes_after"] = model_bytes(model, quantization)
    report["leaf_type_counts"] = dict(zip(TYPE_NAMES, torch.bincount(type_ids(model.leaf_degrees), minlength=3).tolist()))
    return report, changed_cells
