"""Fixed-tree solver for  min_{c >= 0}  J(c) = 1/2 * || A c - y ||^2  over the physical leaf coefficients c.

A is the exact projector of the current octree (linear in c, nonnegative entries).
  gradient     g = A^T (A c - y)                       (back-projection of the residual)
  step scale   d = A^T A 1                             (bounds the curvature per coefficient, so
                                                        c - g/d never increases J)
  update       c <- max(0, c - g / d)                  (truncation = the legal projection onto c >= 0)
Speed-ups that keep the same solution:
  groups       views split into G angle-interleaved groups; each update uses one group (gradient x G)
  extrapolate  evaluate the gradient at  z = c + beta (c - c_prev)  with beta from the standard
               (t-1)/t_next sequence; drop the extrapolation when the step turns against the last move
Convergence is measured by the optimality residual ||z - max(0, z - g/d)|| / ||c||.
"""

from __future__ import annotations

import math
import time

import torch

from adaptive_ct.core.geometry import detector_grid, make_parallel_ray_points

CHUNK = 262_144


def _set(model, c):
    with torch.no_grad():
        model.coefficient_logits.copy_(model._coefficients_to_logits(c.clamp_min(1e-12)))


def _physical_grad(model):
    s = torch.sigmoid(model.coefficient_logits.detach() + model.attenuation_shift)
    return model.coefficient_logits.grad / s.clamp_min(1e-30)


def _sweep(model, split, views, detector_h, detector_w, residual):
    """One pass over `views`: accumulates d/dlogits of 1/2 sum residual(pred, view, rows, cols)^2."""
    rows, cols = detector_grid(detector_h, detector_w, split.projections.device)
    model.coefficient_logits.grad = None
    total = 0.0
    for v in views:
        angle = split.angles[v]
        for s in range(0, rows.numel(), CHUNK):
            r, cc = rows[s : s + CHUNK], cols[s : s + CHUNK]
            batch = make_parallel_ray_points(angle.expand(r.numel()), r, cc, detector_h=detector_h, detector_w=detector_w,
                                             samples_per_ray=1024, targets=split.projections[v, r, cc], materialize_points=False)
            pred = model.integrate_ray_batch(batch)
            e = residual(pred, batch.target)
            loss = 0.5 * (e * e).sum()
            loss.backward()
            total += float(loss.detach())
    return total


def curvature(model, split, views, detector_h, detector_w):
    """d = A^T A 1 for the current tree (one pass)."""
    c_old = model.coefficients().detach().clone()
    _set(model, torch.ones_like(c_old))
    _sweep(model, split, views, detector_h, detector_w, lambda pred, target: pred)
    d = _physical_grad(model).clone()
    _set(model, c_old)
    return d.clamp_min(1e-12)


def gradient(model, c, split, views, detector_h, detector_w):
    _set(model, c)
    J = _sweep(model, split, views, detector_h, detector_w, lambda pred, target: pred - target)
    return _physical_grad(model).clone(), J


def solve(model, split, *, detector_h, detector_w, passes, groups=1, group_passes=0, extrapolate=True, on_pass=None):
    """Run `group_passes` passes of grouped updates, then full-data (optionally extrapolated) updates
    until `passes` full-data-equivalent passes are spent. Returns (c, history)."""
    views = list(range(int(split.angles.numel())))
    order = sorted(views, key=lambda v: float(split.angles[v]))
    view_groups = [order[g::groups] for g in range(groups)]
    model.coefficient_logits.requires_grad_(True)
    d = curvature(model, split, views, detector_h, detector_w)
    c = model.coefficients().detach().clone()
    history, spent, t0 = [], 1.0, time.time()  # the curvature pass counts as one pass

    while spent < 1 + group_passes:
        for grp in view_groups:
            g, _ = gradient(model, c, split, grp, detector_h, detector_w)
            c = (c - groups * g / d).clamp_min(0)
        spent += 1
        if on_pass:
            history.append({"pass": spent, "phase": "grouped", "seconds": time.time() - t0, **on_pass(c)})

    t_k, z = 1.0, c.clone()
    while spent < passes:
        g, J = gradient(model, z, split, views, detector_h, detector_w)
        c_new = (z - g / d).clamp_min(0)
        resid = float((z - c_new).norm() / c_new.norm().clamp_min(1e-30))
        if extrapolate:
            if float(((z - c_new) * (c_new - c)).sum()) > 0:  # the step turned against the last move
                t_k = 1.0
            t_next = (1 + math.sqrt(1 + 4 * t_k * t_k)) / 2
            z = c_new + ((t_k - 1) / t_next) * (c_new - c)
            t_k = t_next
        else:
            z = c_new
        c = c_new
        spent += 1
        row = {"pass": spent, "phase": "full", "objective_at_eval_point": J, "optimality_residual": resid, "seconds": time.time() - t0}
        if on_pass:
            row.update(on_pass(c))
        history.append(row)
        print(row, flush=True)
    _set(model, c)
    model.coefficient_logits.requires_grad_(False)
    return c, history
