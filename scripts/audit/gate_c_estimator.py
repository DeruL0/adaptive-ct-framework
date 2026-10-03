"""Gate C: does the refinement indicator pick the leaves whose split actually helps? (offline, uses GT)

For every leaf of a trained octree:
  e_K   current model's volume SSE inside the leaf (where the error is)
  p_K   SSE of the best trilinear fit of the leaf (no split)
  c_K   SSE of the best trilinear fits of its 8 children (after split)
  gain  = e_K - c_K   (error removed if the leaf is split and the children fit GT well)
and the estimator eta (refinement.estimator, same rays/weights as training).
Reports Spearman correlation and the share of the oracle gain captured when the top-k leaves by
each score are split (all splits cost the same bytes, so top-k is the budget-optimal choice).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adaptive_ct.core.config import load_config
from adaptive_ct.core.loss import projection_weight_kwargs
from adaptive_ct.data.dataset import load_r2_dataset
from adaptive_ct.data.rays import sample_random_rays
from adaptive_ct.refinement.estimator import attributed_excess_residual
from adaptive_ct.representation.factory import build_model
from adaptive_ct.training.loop import _split_dev_views

RUN = sys.argv[1] if len(sys.argv) > 1 else "stage7_dorfler_p1_256v"
dev = torch.device("cuda")
GT = np.load(ROOT / "output/figure473_dicom_orig_r2_convergence/vol_gt.npy", mmap_mode="r")
D, H, W = GT.shape

ck = torch.load(ROOT / "output" / RUN / "checkpoint.pt", map_location="cpu")
cfg = ck["config"]
model = build_model(cfg)
model.prepare_sparse_from_state_dict(ck["model"])
model.load_state_dict(ck["model"], strict=True)
model = model.to(dev).eval()
N = int(model.leaf_levels.shape[0])
shapes = torch.tensor(model.level_shapes, dtype=torch.float32, device=dev)


def trilinear(u):
    a = torch.stack([1 - u[:, 0], u[:, 0]], -1)
    b = torch.stack([1 - u[:, 1], u[:, 1]], -1)
    c = torch.stack([1 - u[:, 2], u[:, 2]], -1)
    return (a[:, :, None, None] * b[:, None, :, None] * c[:, None, None, :]).reshape(-1, 8)


# ---------------------------------------------------------------- volume-side accumulation
e = torch.zeros(N, dtype=torch.float64, device=dev)
pa = torch.zeros(N, 64, dtype=torch.float32, device=dev)
pb = torch.zeros(N, 8, dtype=torch.float32, device=dev)
ca = torch.zeros(N * 8, 64, dtype=torch.float32, device=dev)
cb = torch.zeros(N * 8, 8, dtype=torch.float32, device=dev)
s2 = torch.zeros(N, dtype=torch.float64, device=dev)
s2c = torch.zeros(N * 8, dtype=torch.float64, device=dev)
nvox = torch.zeros(N, dtype=torch.float64, device=dev)
ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
SLAB, CH = 4, 2_000_000
for z0 in range(0, D, SLAB):
    z1 = min(D, z0 + SLAB)
    gt_all = torch.from_numpy(np.ascontiguousarray(GT[z0:z1])).to(dev).reshape(-1)
    zz = torch.arange(z0, z1, device=dev)[:, None, None].expand(-1, H, W)
    pts_all = torch.stack([-1 + (zz.float() + 0.5) * 2 / D, -1 + (ys[None].expand(z1 - z0, -1, -1).float() + 0.5) * 2 / H, -1 + (xs[None].expand(z1 - z0, -1, -1).float() + 0.5) * 2 / W], -1).reshape(-1, 3)
    for s in range(0, pts_all.shape[0], CH):
        pts, v = pts_all[s : s + CH], gt_all[s : s + CH]
        with torch.no_grad():
            pred, ids = model.forward_mu_with_leaf_ids(pts)
        size = 2.0 / shapes[model.leaf_levels[ids]]
        u = ((pts + 1.0) - model.leaf_coords[ids].float() * size) / size
        u = u.clamp(0.0, 1.0)
        child = (u >= 0.5).long()
        cid = ids * 8 + child[:, 0] * 4 + child[:, 1] * 2 + child[:, 2]
        bp = trilinear(u)
        bc = trilinear(u * 2 - child.float())
        e.index_add_(0, ids, ((pred - v) ** 2).double())
        s2.index_add_(0, ids, (v * v).double())
        s2c.index_add_(0, cid, (v * v).double())
        nvox.index_add_(0, ids, torch.ones_like(v, dtype=torch.float64))
        pa.index_add_(0, ids, (bp[:, :, None] * bp[:, None, :]).reshape(-1, 64))
        pb.index_add_(0, ids, bp * v[:, None])
        ca.index_add_(0, cid, (bc[:, :, None] * bc[:, None, :]).reshape(-1, 64))
        cb.index_add_(0, cid, bc * v[:, None])


def lsq_sse(a, b, s2v):  # min ||B c - v||^2 = s2 - b^T c* at the optimum
    out = torch.empty(a.shape[0], dtype=torch.float64, device=dev)
    for s in range(0, a.shape[0], 500_000):
        aa = a[s : s + 500_000].reshape(-1, 8, 8).double()
        scale = aa.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-12)
        aa = aa + 1e-6 * scale[:, None, None] * torch.eye(8, device=dev, dtype=torch.float64)
        bb = b[s : s + 500_000].double()
        coef = torch.linalg.solve(aa, bb)
        out[s : s + 500_000] = (s2v[s : s + 500_000] - (coef * bb).sum(1)).clamp_min(0)
    return out


p = lsq_sse(pa, pb, s2)
c = lsq_sse(ca, cb, s2c).reshape(N, 8).sum(1)
gain = (e - c).cpu().numpy()
e_np, p_np, c_np = e.cpu().numpy(), p.cpu().numpy(), c.cpu().numpy()

# ---------------------------------------------------------------- projection-side estimator
dataset = load_r2_dataset(cfg["dataset"]["root"], device=dev, load_volume=False)
dataset = _split_dev_views(dataset, cfg.get("dataset", {}) or {}, int(cfg["training"].get("seed", 0)))
dh, dw = dataset.detector_shape
wk = projection_weight_kwargs(cfg["training"], iteration=1)
scores = {}
for rays_n in (2_097_152, 8_388_608):
    rays = sample_random_rays(dataset.train, ray_count=rays_n, detector_h=dh, detector_w=dw, samples_per_ray=1024, seed=7, chunk_rays=65536, materialize_points=False)
    eta, exposure, _ = attributed_excess_residual(model, rays, noise_variance=0.0, weight_kwargs=wk)
    scores[f"eta_{rays_n // 1_000_000}M"] = eta.cpu().numpy()
    scores[f"eta_density_{rays_n // 1_000_000}M"] = (eta / exposure.clamp_min(1e-12)).cpu().numpy()
vol = nvox.cpu().numpy()


# ---------------------------------------------------------------- signed-residual (Gauss-Newton) scores
GAUSS = torch.tensor([-1.0 / 3**0.5, 1.0 / 3**0.5], device=dev)


def gn_stats(m, rays, residual_model):
    """Per-leaf gradient g_k = sum_r r * a_rk and diag Hessian H_k = sum_r a_rk^2 of the 8 trilinear DOF."""
    n = int(m.leaf_levels.shape[0])
    g = torch.zeros(n, 8, dtype=torch.float64, device=dev)
    hd = torch.zeros(n, 8, dtype=torch.float64, device=dev)
    sh = torch.tensor(m.level_shapes, dtype=torch.float32, device=dev)
    from adaptive_ct.core.projector import bernstein_segment_ray_capacity
    from adaptive_ct.data.rays import ray_batch
    chunk = min(rays.chunk_rays, bernstein_segment_ray_capacity(m.level_shapes, 512.0))
    for start in range(0, rays.observation_count, chunk):
        batch = ray_batch(rays, start, min(start + chunk, rays.observation_count))
        with torch.no_grad():
            r = (residual_model.integrate_ray_batch(batch) - batch.target).double()
            ray_ids, leaf_ids, iv = m._parallel_ray_segments(batch)
            ang = -batch.angles.to(dev).float()
            dirs = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
            uax = torch.stack([-torch.sin(ang), torch.cos(ang)], 1)
            uu = -1.0 + (batch.cols.to(dev).float() + 0.5) * 2.0 / batch.detector_w
            zz = -1.0 + (batch.rows.to(dev).float() + 0.5) * 2.0 / batch.detector_h
            base = uax * uu[:, None]
            half = 0.5 * (iv[:, 1] - iv[:, 0])
            mid = 0.5 * (iv[:, 1] + iv[:, 0])
            size = 2.0 / sh[m.leaf_levels[leaf_ids]]
            origin = -1.0 + m.leaf_coords[leaf_ids].float() * size
            a = torch.zeros(leaf_ids.shape[0], 8, device=dev)
            for q in GAUSS:
                t = mid + half * q
                xy = base[ray_ids] + dirs[ray_ids] * t[:, None]
                pts = torch.stack([xy[:, 0], xy[:, 1], zz[ray_ids]], 1)
                a += trilinear(((pts - origin) / size).clamp(0, 1)) * half[:, None]
            a = a.double()
            g.index_add_(0, leaf_ids, a * r[ray_ids, None])
            hd.index_add_(0, leaf_ids, a * a)
    return g, hd


def gn_gain(g, hd, delta):
    pos = hd[hd > 0]
    lam = delta * (pos.median() if pos.numel() else 1.0)
    return (g * g / (hd + lam)).sum(1)


with torch.no_grad():
    children = build_model(cfg)
    children.prepare_sparse_from_state_dict(ck["model"])
    children.load_state_dict(ck["model"], strict=True)
    children = children.to(dev).eval()
    children.split_leaves_batch(torch.arange(N, device=dev))
    assert torch.equal(torch.div(children.leaf_coords.view(N, 8, 3), 2, rounding_mode="floor"), model.leaf_coords[:, None, :].expand(-1, 8, -1))
rays = sample_random_rays(dataset.train, ray_count=2_097_152, detector_h=dh, detector_w=dw, samples_per_ray=1024, seed=7, chunk_rays=65536, materialize_points=False)
gp, hp = gn_stats(model, rays, model)
gc, hc = gn_stats(children, rays, model)
for delta in (1e-2, 1.0):
    parent_gain = gn_gain(gp, hp, delta)
    child_gain = gn_gain(gc, hc, delta).view(N, 8).sum(1)
    scores[f"gn_child_d{delta:g}"] = child_gain.cpu().numpy()
    scores[f"gn_split_d{delta:g}"] = (child_gain - parent_gain).cpu().numpy()
scores["gradnorm_child"] = (gc * gc).sum(1).view(N, 8).sum(1).cpu().numpy()
scores["gradnorm_parent"] = (gp * gp).sum(1).cpu().numpy()
del children
torch.cuda.empty_cache()
scores["oracle_error_e"] = e_np
scores["random"] = np.random.default_rng(0).random(N)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


best_order = np.argsort(-gain)
out = {
    "run": RUN,
    "leaves": N,
    "total_volume_sse": float(e_np.sum()),
    "decomposition": {
        "fit_deficit_sum(e-p)": float((e_np - p_np).sum()),
        "split_capacity_sum(p-c)": float((p_np - c_np).sum()),
        "child_residual_sum(c)": float(c_np.sum()),
    },
    "budgets": {},
}
for k in (5_000, 18_000, 27_000, 50_000, 100_000):
    oracle = float(np.clip(gain[best_order[:k]], 0, None).sum())
    row = {"oracle_gain": oracle}
    for name, sc in scores.items():
        sel = np.argsort(-sc)[:k]
        row[name] = round(float(gain[sel].sum()) / oracle, 4)
    out["budgets"][str(k)] = row
out["spearman_vs_gain"] = {name: round(spearman(sc, gain), 4) for name, sc in scores.items()}
print(json.dumps(out, indent=1))
(ROOT / "output/_stage7_logs/gate_c.json").write_text(json.dumps(out, indent=1))
