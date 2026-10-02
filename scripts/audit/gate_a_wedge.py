"""Gate A: capacity oracle for interface leaves (offline, uses the GT volume; never part of training).

On the uniform L1 grid (96^3 cells over the normalized cube), every cell is fitted to the GT
voxels it contains with
  p0      one constant                          5 B/leaf
  p1      trilinear least squares                11 B/leaf (latent d=4 codec)
  wedge   plane + two constants (search)        13 B/leaf (fp16 normal angles, offset, 2 values)
  p0_l2   split into 8 constant children       104 B
  p1_l2   split into 8 trilinear children      152 B
Non-surface cells use p1 in every option, so only the surface treatment differs.
Bytes follow codec/budget.py: values * 2 B + 3 B tag per leaf, 8 nodes * 8 B per split.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
GT = np.load(ROOT / "output/figure473_dicom_orig_r2_convergence/vol_gt.npy", mmap_mode="r")
D, H, W = GT.shape
C = 96
TAU = 0.1
import os

DIRS = int(os.environ.get("GATE_DIRS", 256))
STRIDE = int(os.environ.get("GATE_STRIDE", 1))  # >1: subsample cell layers
CHUNK = 96
dev = torch.device("cuda")
BYTES = {"p0": 5, "p1": 11, "wedge": 13, "best_p1_wedge": 13, "p0_l2": 104, "p1_l2": 152}


def axis_groups(n):
    idx = torch.arange(n, dtype=torch.float64)
    cell = torch.floor((idx + 0.5) * C / n).long()
    m = int(torch.bincount(cell, minlength=C).max())
    gather = torch.full((C, m), -1, dtype=torch.long)
    local = torch.zeros((C, m), dtype=torch.float64)
    for c in range(C):
        ids = torch.nonzero(cell == c).flatten()
        gather[c, : ids.numel()] = ids
        local[c, : ids.numel()] = (ids.double() + 0.5) * C / n - c
    return gather.to(dev), local.float().to(dev)


gz, uz = axis_groups(D)
gy, uy = axis_groups(H)
gx, ux = axis_groups(W)
Mz, My, Mx = gz.shape[1], gy.shape[1], gx.shape[1]


def fib_hemisphere(k):
    i = torch.arange(k, dtype=torch.float64) + 0.5
    z = i / k  # z in (0,1): hemisphere, n and -n are the same plane
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    r = torch.sqrt(1 - z * z)
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=1).float().to(dev)


NORMALS = fib_hemisphere(DIRS)


def trilinear_basis(u):  # u (..., 3) -> (..., 8)
    a = torch.stack([1 - u[..., 0], u[..., 0]], -1)
    b = torch.stack([1 - u[..., 1], u[..., 1]], -1)
    c = torch.stack([1 - u[..., 2], u[..., 2]], -1)
    return (a[..., :, None, None] * b[..., None, :, None] * c[..., None, None, :]).flatten(-3)


def lsq_sse(basis, v, w):  # basis (B,n,k), v,w (B,n) -> SSE (B,)
    bw = basis * w[..., None]
    a = bw.transpose(1, 2) @ basis + 1e-6 * torch.eye(basis.shape[-1], device=dev)
    rhs = (bw * v[..., None]).sum(1)
    coef = torch.linalg.solve(a.double(), rhs.double()).float()
    r = (basis @ coef[..., None])[..., 0] - v
    return (w * r * r).sum(1)


def wedge_sse(coords, v, w):  # coords (B,n,3), v,w (B,n) -> best SSE over normals and offsets
    best = torch.full((v.shape[0],), float("inf"), device=dev, dtype=torch.float64)
    p = coords @ NORMALS.T  # (B,n,K)
    p = torch.where(w[..., None] > 0, p, torch.full_like(p, float("inf")))
    order = torch.argsort(p.transpose(1, 2), dim=-1)  # (B,K,n)
    vs = torch.gather(v[:, None, :].expand(-1, DIRS, -1), 2, order).double()
    ws = torch.gather(w[:, None, :].expand(-1, DIRS, -1), 2, order).double()
    s1 = torch.cumsum(vs * ws, -1)
    s2 = torch.cumsum(vs * vs * ws, -1)
    nl = torch.cumsum(ws, -1)
    t1, t2, tn = s1[..., -1:], s2[..., -1:], nl[..., -1:]
    nr = tn - nl
    sse = (s2 - s1 * s1 / nl.clamp_min(1)) + ((t2 - s2) - (t1 - s1) ** 2 / nr.clamp_min(1))
    sse = torch.where((nl >= 1) & (nr >= 1), sse, torch.full_like(sse, float("inf")))
    return torch.minimum(best, sse.amin(dim=(1, 2)))


acc = {k: 0.0 for k in BYTES}
acc_nonsurface_p1 = 0.0
n_surface = 0
gain_hist = []  # per surface cell: (sse_p1, sse_wedge)
t0 = time.time()
for cz in range(0, C, STRIDE):
    zid = gz[cz]
    zmask = zid >= 0
    slab = torch.from_numpy(np.ascontiguousarray(GT[int(zid[zmask].min()) : int(zid[zmask].max()) + 1])).to(dev)
    zl = (zid - zid[zmask].min()).clamp_min(0)
    vol = slab[zl][:, gy.clamp_min(0)][:, :, :, gx.clamp_min(0)]  # (Mz, C, My, C, Mx)
    vol = vol.permute(1, 3, 0, 2, 4).reshape(C * C, -1)  # cells x voxels
    wz = zmask.float()[None, None, :, None, None]
    wy = (gy >= 0).float()[:, None, None, :, None]
    wx = (gx >= 0).float()[None, :, None, None, :]
    w = (wz * wy * wx).reshape(C * C, -1)
    cu = torch.stack(
        torch.broadcast_tensors(
            uz[cz][None, None, :, None, None].expand(C, C, Mz, My, Mx),
            uy[:, None, None, :, None].expand(C, C, Mz, My, Mx),
            ux[None, :, None, None, :].expand(C, C, Mz, My, Mx),
        ),
        -1,
    ).reshape(C * C, -1, 3)
    big = torch.where(w > 0, vol, torch.full_like(vol, -1.0)).amax(1)
    small = torch.where(w > 0, vol, torch.full_like(vol, 2.0)).amin(1)
    surface = (small < TAU) & (big >= TAU)
    basis = trilinear_basis(cu)
    sse_p1_all = lsq_sse(basis, vol, w)
    acc_nonsurface_p1 += float(sse_p1_all[~surface].sum())
    sid = torch.nonzero(surface).flatten()
    n_surface += int(sid.numel())
    for s in range(0, sid.numel(), CHUNK):
        ids = sid[s : s + CHUNK]
        v, ww, uu = vol[ids], w[ids], cu[ids]
        sp1 = sse_p1_all[ids]
        mean = (v * ww).sum(1) / ww.sum(1)
        sp0 = (ww * (v - mean[:, None]) ** 2).sum(1)
        child = (uu >= 0.5).long()
        cid = child[..., 0] * 4 + child[..., 1] * 2 + child[..., 2]
        onehot = torch.nn.functional.one_hot(cid, 8).float() * ww[..., None]
        cnt = onehot.sum(1).clamp_min(1e-9)
        cmean = (onehot * v[..., None]).sum(1) / cnt
        sp0l2 = (ww * (v - torch.gather(cmean, 1, cid)) ** 2).sum(1)
        u2 = uu * 2 - child.float()
        b2 = trilinear_basis(u2)
        b64 = (b2[..., None, :] * torch.nn.functional.one_hot(cid, 8).float()[..., :, None]).flatten(-2)
        sp1l2 = lsq_sse(b64, v, ww)
        swg = wedge_sse(uu, v, ww).float()
        acc["p0"] += float(sp0.sum())
        acc["p1"] += float(sp1.sum())
        acc["wedge"] += float(swg.sum())
        acc["best_p1_wedge"] += float(torch.minimum(sp1, swg).sum())
        acc["p0_l2"] += float(sp0l2.sum())
        acc["p1_l2"] += float(sp1l2.sum())
        gain_hist.append(torch.stack([sp1, swg, sp1l2], 1).cpu())
    if cz % 16 == 0:
        print(f"layer {cz}/{C} surface cells so far {n_surface} ({time.time() - t0:.0f}s)", flush=True)

nvox = D * H * W
g = torch.cat(gain_hist).double()
ratio = g[:, 1] / g[:, 0].clamp_min(1e-12)
out = {
    "grid": f"L1 uniform {C}^3, cell ~{D / C:.2f}x{H / C:.2f}x{W / C:.2f} voxels",
    "surface_cells": n_surface,
    "normals_searched": DIRS,
    "options": {},
    "per_cell_wedge_over_p1_sse": {
        "median": float(ratio.median()),
        "frac_wedge_better": float((ratio < 1).double().mean()),
        "frac_wedge_halves_error": float((ratio < 0.5).double().mean()),
        "frac_wedge_worse": float((ratio > 1).double().mean()),
    },
}
for k, sse in acc.items():
    total = sse + acc_nonsurface_p1
    out["options"][k] = {
        "surface_bytes_MB": n_surface * BYTES[k] / 1e6,
        "surface_sse": sse,
        "surface_sse_vs_p1": sse / acc["p1"],
        "volume_psnr_db": 10 * math.log10(1.0 / (total / nvox)),
    }
# matched-bytes allocation: spend extra bytes over P1 either on splits only or on splits + wedges.
# Greedy on per-cell lower convex hulls of (bytes, sse) = multiple-choice knapsack LP bound.
a, b, c = g[:, 0].numpy(), g[:, 1].numpy(), g[:, 2].numpy()
cost_w, cost_s = BYTES["wedge"] - BYTES["p1"], BYTES["p1_l2"] - BYTES["p1"]


def curve(increments):
    eff = np.concatenate([gain / cost for gain, cost in increments])
    gain = np.concatenate([gain for gain, _ in increments])
    cost = np.concatenate([np.full_like(gn, cs, dtype=np.float64) if np.isscalar(cs) else cs for gn, cs in increments])
    keep = gain > 0
    order = np.argsort(-eff[keep])
    return np.cumsum(cost[keep][order]), np.cumsum(gain[keep][order])


split_only = curve([(a - c, cost_s)])
use_w = b < a
e1 = (a - b) / cost_w
e2 = (b - c) / (cost_s - cost_w)
convex = use_w & (e2 < e1)
inc = [
    ((a - b)[convex], cost_w),
    ((b - c)[convex], cost_s - cost_w),
    ((a - c)[~convex], cost_s),
]
split_wedge = curve(inc)
base_total = acc["p1"] + acc_nonsurface_p1
rd = []
for extra_mb in (0.25, 0.5, 1, 2, 4, 8, 16, 32):
    row = {"extra_MB": extra_mb}
    for name, (cb, cg) in (("split_only", split_only), ("split_plus_wedge", split_wedge)):
        k = np.searchsorted(cb, extra_mb * 1e6, side="right")
        gained = cg[k - 1] if k > 0 else 0.0
        row[name + "_psnr"] = 10 * math.log10(nvox / (base_total - gained))
    rd.append(row)
out["matched_bytes_rd"] = rd
out["p1_baseline_volume_psnr"] = 10 * math.log10(nvox / base_total)
out["seconds"] = time.time() - t0
print(json.dumps(out, indent=1))
(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output/_stage7_logs/gate_a.json").write_text(json.dumps(out, indent=1))
