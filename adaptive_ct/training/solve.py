"""Fixed-tree solver for  min_{c >= 0}  J(c) = 1/2 |A c - y|^2 + beta R(c)  over the leaf values c (model.values):
relaxed ordered-subsets linearized augmented Lagrangian method (Nien & Fessler 2016, arXiv:1512.04564, Algorithm 1, W = I).

A is the exact projector of the current octree (nonnegative entries), R the edge-preserving penalty between neighbouring
leaves (training/regularizer.py).  D_L = diag(A^T A 1) and D_R = beta * (the penalty's separable curvature bound) majorize
the curvature of the two terms.  The views are split into M angle-interleaved subsets L_m; one epoch visits them all,
spaced apart in angle.  Per subset (x the values, g and h the running gradient estimate and its offset):
    s = rho (D_L x - h) + (1 - rho) g
    x+ = max(0, x - (rho D_L + D_R)^-1 (s + beta grad R(x)))                 (grad R at the current x, every subset)
    zeta = M grad L_m(x+)
    g+ = rho/(rho + 1) (alpha zeta + (1 - alpha) g) + g/(rho + 1) ;   h+ = alpha (D_L x+ - zeta) + (1 - alpha) h
rho decays per subset: rho_0 = 1, rho_k = pi / (alpha (k + 1)) sqrt(1 - (pi / (2 alpha (k + 1)))^2), alpha = 1.999.
Measured against the previous solver on the same trees: 6 epochs instead of 62 passes to the same residual, a lower objective.
The subset count must stay small (10-12; 24 ripples) and grad R must be recomputed every subset (once per epoch diverges).
Convergence: the objective estimate of an epoch (the subsets' losses taken right after their updates, plus beta R) is free;
the solve has converged when it changed by no more than `tol` (relative) in the last epoch, after at least `min_epochs`.
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
ALPHA = 1.999  # relaxation of the augmented Lagrangian (the paper's choice)


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
        total = total + loss.detach()  # on the device: a host read per batch stalls the GPU queue
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


def _rho(k: int) -> float:
    """Penalty parameter of subset step k (counted over all epochs of a tree)."""
    return 1.0 if k == 0 else (math.pi / (ALPHA * (k + 1))) * math.sqrt(1 - (math.pi / (2 * ALPHA * (k + 1))) ** 2)


DIVERGED = 2.0  # objective estimate above this multiple of the best one so far: the run is diverging


def _subsets(angles, subsets):
    """Angle-interleaved view subsets, and a visiting order that puts consecutive subsets far apart in angle."""
    order = sorted(range(int(angles.numel())), key=lambda v: float(angles[v]))
    groups = [order[g::subsets] for g in range(subsets)]
    stride = next(s for s in range(max(1, subsets // 3), subsets + 1) if math.gcd(s, subsets) == 1)
    return groups, [(i * stride) % subsets for i in range(subsets)]


def solve(model, split, *, subsets, tol, max_epochs, min_epochs, detector_bin=1, penalty=None, penalty_weight=0.0, state=None):
    """Epochs of the relaxed OS-LALM until the objective estimate changed by <= tol (relative, either sign) in the last
    epoch (and at least `min_epochs` ran, counted over the whole tree) or `max_epochs` of this call are spent.  The
    objective is J + penalty_weight / bin^2 * R (the data term has bin^2 fewer rays when binned).
    The method has no guarantee for many small subsets (seen: 12 views per subset diverge on a toy problem, 10 views ripple
    on the real data): when the estimate exceeds DIVERGED times the best one, the values go back to the best ones and the
    subset count is halved (down to 1 = the full batch, which is monotone), restarting rho and the running gradient.
    `state` (a dict): filled at the end of the call; a later call with the same tree, geometry, binning and penalty and the
    same dict continues from it (same denominators and running gradient, rho keeps decaying).
    Returns (values, history, converged)."""
    angles, projections = split.angles, binned(split, detector_bin)
    beta = float(penalty_weight) / detector_bin**2 if penalty is not None else 0.0
    model.values.requires_grad_(True)

    def begin(x, subsets):  # running gradient estimate at x for the given subset count
        groups, visit = _subsets(angles, subsets)
        zeta = float(subsets) * gradient(model, x, angles, projections, groups[visit[-1]])[0]
        return groups, visit, zeta.clone(), d_l * x - zeta

    if state:
        d_l, d_r, x, g, h, k, J_prev, epochs, subsets, best_J, best_x = (state[n] for n in (
            "d_l", "d_r", "x", "g", "h", "k", "J_prev", "epochs", "subsets", "best_J", "best_x"))
        groups, visit = _subsets(angles, subsets)
    else:
        d_l = curvature(model, angles, projections, list(range(int(angles.numel()))))
        d_r = beta * penalty.curvature() if beta > 0 else torch.zeros_like(d_l)
        x = model.values.detach().clone()
        groups, visit, g, h = begin(x, subsets)
        k, J_prev, epochs, best_J, best_x = 0, None, 0, math.inf, x.clone()
    history, converged, t0 = [], False, time.time()
    for _ in range(max_epochs):
        M, J_data, R, rho = float(subsets), 0.0, 0.0, 1.0
        for m in visit:
            rho, k = _rho(k), k + 1
            gR = 0.0
            if beta > 0:
                R, gR = penalty.value_and_gradient(x)
                gR = beta * gR
            s = rho * (d_l * x - h) + (1 - rho) * g
            x = (x - (s + gR) / (rho * d_l + d_r)).clamp_min(0)
            gm, Jm = gradient(model, x, angles, projections, groups[m])
            zeta, J_data = M * gm, J_data + Jm  # the loss stays on the device until the epoch is over
            g = (rho / (rho + 1)) * (ALPHA * zeta + (1 - ALPHA) * g) + g / (rho + 1)
            h = ALPHA * (d_l * x - zeta) + (1 - ALPHA) * h
        epochs += 1
        J = float(J_data) + beta * R
        row = {"epoch": epochs, "objective_estimate": J, "rho": rho, "subsets": subsets, "seconds": time.time() - t0, "detector_bin": detector_bin}
        if not math.isfinite(J) or (J > DIVERGED * best_J and subsets > 1):
            subsets = max(1, subsets // 2)
            x = best_x.clone()
            groups, visit, g, h = begin(x, subsets)
            k, J_prev = 0, None
            history.append({**row, "relative_decrease": None, "restart_with_subsets": subsets})
            print(history[-1], flush=True)
            continue
        if J < best_J:
            best_J, best_x = J, x.clone()
        decrease = None if J_prev is None else (J_prev - J) / J
        J_prev = J
        history.append({**row, "relative_decrease": decrease})
        print(history[-1], flush=True)
        if decrease is not None and epochs >= min_epochs and abs(decrease) <= tol:
            converged = True
            break
    _set(model, x)
    model.values.requires_grad_(False)
    if state is not None:
        state.update(d_l=d_l, d_r=d_r, x=x, g=g, h=h, k=k, J_prev=J_prev, epochs=epochs, subsets=subsets, best_J=best_J, best_x=best_x)
    return x, history, converged
