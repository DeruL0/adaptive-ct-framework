"""Step-leaf geometry update with the values fixed (one damped Gauss-Newton step per call).

Objective as in the solve:  J(theta) = 1/2 ||A(theta) c - y||^2 + beta R(theta)  (theta = n, d, w of every step leaf).
Per step leaf L, with D_ray = d(prediction of the ray)/d theta_L (5 numbers, CUDA segment_geometry):
  g_L = sum residual * D,   G_L = sum D D^T   (rays through several step leaves: their coupling is ignored),
  delta_L = -(G_L + 0.1 diag(G_L) + eps I)^-1 g_L, each part capped (|dn| <= 0.25, |dd| <= 0.1, w within [w/2, 2w]).
Constraints after the step: n renormalised (d, w scaled with it, same function), w >= a quarter detector pixel,
the plane inside the leaf, and both sides >= 2% of the leaf (otherwise the leaf keeps its geometry).
The whole step is accepted only if J decreases; otherwise it is halved, up to three tries.
"""

from __future__ import annotations

import time

import torch

from adaptive_ct.core.projector import bernstein_octree_geometry_accumulate
from adaptive_ct.training.regularizer import EdgePenalty
from adaptive_ct.training.solve import binned, ray_batches

STATS = 20


def _side_fraction(g: torch.Tensor, samples: int = 4) -> torch.Tensor:
    q = (torch.arange(samples, device=g.device, dtype=torch.float32) + 0.5) / samples
    u = torch.stack(torch.meshgrid(q, q, q, indexing="ij"), -1).reshape(-1, 3)
    s = ((u[None] - 0.5) * g[:, None, :3]).sum(-1) - g[:, 3:4]
    h = torch.sigmoid(s / g[:, 4:5].clamp_min(1e-30)).mean(1)
    return torch.minimum(h, 1 - h)


def _reference_accumulate(model, batch, residual, accum):
    """PyTorch reference of the CUDA accumulation (used on CPU)."""
    from adaptive_ct.representation.octree import step_integral

    ray_ids, leaf, iv = model._parallel_ray_segments(batch)
    step = (leaf >= model.n_tri) & (leaf < model.n_tri + model.n_step)
    ray_ids, leaf, iv = ray_ids[step], leaf[step], iv[step]
    j = leaf - model.n_tri
    u0 = model.segment_local_coords(batch, ray_ids, leaf, iv[:, 0])
    u1 = model.segment_local_coords(batch, ray_ids, leaf, iv[:, 1])
    g = model.step_geometry[j, :5].detach().clone().requires_grad_(True)
    length = iv[:, 1] - iv[:, 0]
    I = step_integral(g, ((u0 - 0.5) * g[:, :3]).sum(1), ((u1 - 0.5) * g[:, :3]).sum(1), length)
    off = model.value_offsets()[leaf]
    v = model.values.detach()
    (v[off] * (length - I) + v[off + 1] * I).sum().backward()
    keep = model.step_geometry[j, 4] > 0  # hard steps have no derivative
    key, inv = torch.unique((ray_ids * model.n_step + j)[keep], return_inverse=True)
    D = torch.zeros((key.numel(), 5), device=g.device).index_add_(0, inv, g.grad[keep])
    iu, ju = torch.triu_indices(5, 5, device=g.device)
    rows = torch.cat([residual[key // model.n_step, None] * D, D[:, iu] * D[:, ju]], 1)
    accum.index_add_(0, key % model.n_step, rows)


@torch.no_grad()
def data_term(model, angles, projections, views) -> float:
    return sum(0.5 * float(((model.integrate_ray_batch(b) - b.target) ** 2).sum()) for b in ray_batches(angles, projections, views))


@torch.no_grad()
def statistics(model, angles, projections, views) -> tuple[torch.Tensor, float]:
    """(n_step, STATS) Gauss-Newton statistics at the current values and geometry, and the data term."""
    accum = torch.zeros((model.n_step, STATS), device=model.values.device)
    J = 0.0
    for batch in ray_batches(angles, projections, views):
        r = model.integrate_ray_batch(batch) - batch.target
        J += 0.5 * float((r * r).sum())
        if model.uses_cuda():
            bernstein_octree_geometry_accumulate(model.values, model.step_geometry, **model.tree_arguments(), rays=batch, residual=r, accum=accum)
        else:
            with torch.enable_grad():
                _reference_accumulate(model, batch, r, accum)
    return accum, J


@torch.no_grad()
def geometry_step(model, split, *, detector_bin: int, penalty_weight: float, penalty_delta: float, tries: int = 3) -> dict:
    t0 = time.perf_counter()
    if model.n_step == 0:
        return {"step_leaves": 0}
    angles, projections = split.angles, binned(split, detector_bin)
    views = list(range(int(angles.numel())))
    beta = float(penalty_weight) / detector_bin**2

    penalty = EdgePenalty(model, penalty_delta) if beta > 0 else None  # samples fixed; weights follow the geometry
    R = lambda: penalty.value_and_gradient(model.values.detach())[0] if penalty is not None else 0.0
    objective = lambda: data_term(model, angles, projections, views) + beta * R()
    accum, J_data = statistics(model, angles, projections, views)
    J0 = J_data + beta * R()
    iu, ju = torch.triu_indices(5, 5, device=accum.device)
    G = torch.zeros((model.n_step, 5, 5), dtype=torch.float64, device=accum.device)
    G[:, iu, ju] = accum[:, 5:].double()
    G[:, ju, iu] = accum[:, 5:].double()
    diag = G.diagonal(dim1=1, dim2=2)
    damp = 0.1 * diag + 1e-6 * diag.mean(1, keepdim=True) + 1e-30
    delta = -torch.linalg.solve(G + torch.diag_embed(damp), accum[:, :5].double()[..., None])[..., 0].float()
    old = model.step_geometry.clone()
    w = old[:, 4]
    delta[:, :3] *= (0.25 / delta[:, :3].norm(dim=1).clamp_min(1e-30)).clamp(max=1.0)[:, None]
    delta[:, 3] = delta[:, 3].clamp(-0.1, 0.1)
    delta[:, 4] = torch.maximum(torch.minimum(delta[:, 4], w), -0.5 * w)
    shapes = torch.tensor(model.level_shapes, dtype=torch.float32, device=old.device)[model.leaf_levels[model.n_tri : model.n_tri + model.n_step]]
    w_min = 0.25 * shapes[:, 0] / split.projections.shape[2]  # a quarter detector pixel, in leaf units
    report = {"step_leaves": model.n_step, "objective_before": J0, "accepted_step": 0.0}
    for k in range(tries):
        alpha = 0.5**k
        g = old[:, :5] + alpha * delta
        s = g[:, :3].norm(dim=1, keepdim=True).clamp_min(1e-30)
        g = g / s
        g[:, 4] = torch.maximum(g[:, 4], w_min)
        extent = 0.5 * g[:, :3].abs().sum(1)
        g[:, 3] = torch.maximum(torch.minimum(g[:, 3], 0.98 * extent), -0.98 * extent)
        ok = (_side_fraction(g) >= 0.02) & (old[:, 4] > 0)
        model.step_geometry[:, :5] = torch.where(ok[:, None], g, old[:, :5])
        J = objective()
        if J < J0:
            report.update(accepted_step=alpha, objective_after=J, leaves_moved=int(ok.sum()))
            break
        model.step_geometry.copy_(old)
    else:
        report["objective_after"] = J0
    report["seconds"] = time.perf_counter() - t0
    return report
