"""Fixed-tree solver for  min_{c >= 0}  J(c) = 1/2 * || A c - y ||^2  over the physical leaf coefficients c.

A is the exact projector of the current octree (linear in c = model.values, nonnegative entries).
  gradient     g = A^T (A c - y)                       (back-projection of the residual)
  step scale   d = A^T A 1                             (bounds the curvature per coefficient, so
                                                        c - g/d never increases J)
  update       c <- max(0, c - g / d)                  (truncation = the legal projection onto c >= 0)
Speed-ups that keep the same solution:
  groups       views split into G angle-interleaved groups; each update uses one group (gradient x G)
  extrapolate  evaluate the gradient at  z = max(0, c + beta (c - c_prev))  with beta from the standard
               (t-1)/t_next sequence; drop the extrapolation when the step turns against the last move
Convergence is measured by the optimality residual ||z - max(0, z - g/d)|| / ||c|| at the feasible point z;
the solve stops when it is <= tol, or after `passes` passes (then `converged` is False).
detector_bin b > 1 solves against b x b averaged detector pixels (one ray through each block centre):
16x fewer rays for b = 4, an approximation that is only used while the leaves span several blocks.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from adaptive_ct.core.geometry import parallel_rays, traversal_grid

CHUNK = 1 << 20  # rays per kernel call (one 1024x1024 view, or several binned views)


def _set(model, c):
    with torch.no_grad():
        model.values.copy_(c)


def binned(split, b):
    """(views, H/b, W/b) block-averaged projections."""
    return split.projections if b == 1 else F.avg_pool2d(split.projections[:, None], b)[:, 0]


def ray_batches(angles, projections, views):
    """All pixels of `views`, CHUNK rays per batch (one ray through each (binned) pixel centre)."""
    h, w = projections.shape[1:]
    rows, cols = traversal_grid(h, w, projections.device)
    n = rows.numel()
    views = torch.as_tensor(views, device=projections.device)
    for s in range(0, views.numel() * n, CHUNK):
        idx = torch.arange(s, min(s + CHUNK, views.numel() * n), device=projections.device)
        v, p = views[idx // n], idx % n
        yield parallel_rays(angles[v], rows[p], cols[p], detector_h=h, detector_w=w, targets=projections[v, rows[p], cols[p]])


def _sweep(model, angles, projections, views, residual):
    """One pass over `views`: accumulates into model.values.grad the gradient of 1/2 sum residual(pred, target)^2."""
    model.values.grad = None
    total = 0.0
    for batch in ray_batches(angles, projections, views):
        e = residual(model.integrate_ray_batch(batch), batch.target)
        loss = 0.5 * (e * e).sum()
        loss.backward()
        total += float(loss.detach())
    return total


def curvature(model, angles, projections, views):
    """d = A^T A 1 for the current tree (one pass)."""
    c_old = model.values.detach().clone()
    _set(model, torch.ones_like(c_old))
    _sweep(model, angles, projections, views, lambda pred, target: pred)
    d = model.values.grad.clone()
    _set(model, c_old)
    return d.clamp_min(1e-12)


def gradient(model, c, angles, projections, views):
    _set(model, c)
    J = _sweep(model, angles, projections, views, lambda pred, target: pred - target)
    return model.values.grad.clone(), J


def solve(model, split, *, passes, tol=0.0, groups=1, group_passes=0, extrapolate=True, detector_bin=1, on_pass=None,
          penalty=None, penalty_weight=0.0):
    """Run `group_passes` passes of grouped updates, then full-data (optionally extrapolated) updates until the
    optimality residual is <= tol or `passes` full-data-equivalent passes are spent.
    With `penalty` (training.regularizer.EdgePenalty) the objective is  J + penalty_weight / bin^2 * R  (the data
    term has bin^2 fewer rays when binned); its gradient joins g and its curvature bound joins d.
    Returns (coefficients, history, converged)."""
    angles, projections = split.angles, binned(split, detector_bin)
    views = list(range(int(angles.numel())))
    order = sorted(views, key=lambda v: float(angles[v]))
    view_groups = [order[g::groups] for g in range(groups)]
    model.values.requires_grad_(True)
    d = curvature(model, angles, projections, views)
    beta = float(penalty_weight) / detector_bin**2 if penalty is not None else 0.0
    if beta > 0:
        d = d + beta * penalty.curvature()
    c = model.values.detach().clone()

    def penalised(g, J, at):
        if beta == 0:
            return g, J, 0.0
        R, gR = penalty.value_and_gradient(at)
        return g + beta * gR, J + beta * R, R

    history, spent, t0, converged = [], 1.0, time.time(), False  # the curvature pass counts as one pass

    while spent < 1 + group_passes:
        for grp in view_groups:
            g, _ = gradient(model, c, angles, projections, grp)
            g, _, _ = penalised(groups * g, 0.0, c)
            c = (c - g / d).clamp_min(0)
        spent += 1
        if on_pass:
            history.append({"pass": spent, "phase": "grouped", "seconds": time.time() - t0, **on_pass(c)})

    t_k, z = 1.0, c.clone()
    while spent < passes:
        g, J = gradient(model, z, angles, projections, views)
        g, J, R = penalised(g, J, z)
        c_new = (z - g / d).clamp_min(0)
        resid = float((z - c_new).norm() / c_new.norm().clamp_min(1e-30))
        if extrapolate:
            if float(((z - c_new) * (c_new - c)).sum()) > 0:  # the step turned against the last move
                t_k = 1.0
            t_next = (1 + math.sqrt(1 + 4 * t_k * t_k)) / 2
            z = (c_new + ((t_k - 1) / t_next) * (c_new - c)).clamp_min(0)
            t_k = t_next
        else:
            z = c_new
        c = c_new
        spent += 1
        row = {"pass": spent, "phase": "full", "objective_at_eval_point": J, "optimality_residual": resid,
               "seconds": time.time() - t0, "detector_bin": detector_bin, "penalty": R}
        if on_pass:
            row.update(on_pass(c))
        history.append(row)
        print(row, flush=True)
        if resid <= tol:
            converged = True
            break
    _set(model, c)
    model.values.requires_grad_(False)
    return c, history, converged
