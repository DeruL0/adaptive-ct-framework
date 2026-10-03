"""Error-chain diagnostic: where does the GT -> deployed gap come from?  (offline, uses GT; not part of training)

  s0      operator check: same function, exact integrator vs the dense projector that generated the data
  ls_eval our tree + GT least-squares coefficients                        (2: selection, no inversion loss)
  oracle  best tree under the same byte budget (Lagrangian DP, L0..L2) + GT LS coefficients   (1: capacity)
  train   fixed trees trained from projections only: oracle tree (3a), our tree (3b)          (3: inversion)
  views   oracle tree trained on 3x more views                                               (5: data)

usage: python gate_chain.py <run> <stage> [<stage> ...]     results -> output/_chain/results.json
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adaptive_ct.codec.budget import codec_bytes
from adaptive_ct.core.geometry import random_training_rays
from adaptive_ct.core.loss import projection_weight_kwargs, weighted_projection_loss
from adaptive_ct.core.projector import project_dense_parallel
from adaptive_ct.data.dataset import ProjectionSplit, load_r2_dataset
from adaptive_ct.evaluation.metrics import projection_metrics
from adaptive_ct.evaluation.render import render_split
from adaptive_ct.representation.factory import build_model
from adaptive_ct.training.loop import _initialize_constant_mu, _split_dev_views

RUN = sys.argv[1]
STAGES = sys.argv[2:]
OUT = ROOT / "output/_chain"
OUT.mkdir(parents=True, exist_ok=True)
RES_PATH = OUT / "results.json"
RES = json.loads(RES_PATH.read_text()) if RES_PATH.exists() else {}
dev = torch.device("cuda")
GT = np.load(ROOT / "output/figure473_dicom_orig_r2_convergence/vol_gt.npy", mmap_mode="r")
D, H, W = GT.shape
CK = torch.load(ROOT / "output" / RUN / "checkpoint.pt", map_location="cpu")
CFG = CK["config"]
CODEC = {"type": "latent_p1", "latent_dim": 4, "quantization": "float16"}
BUDGET = 8_388_608
SPLIT_BYTES = 8 * 8 + 7 * 11  # codec.budget.split_delta_bytes for latent d=4
TAU, BAND_R = 0.1, 3


def save(key, value):
    RES[key] = value
    RES_PATH.write_text(json.dumps(RES, indent=1, default=float))
    print(key, json.dumps(value, default=float), flush=True)


def trilinear(u):
    a = torch.stack([1 - u[:, 0], u[:, 0]], -1)
    b = torch.stack([1 - u[:, 1], u[:, 1]], -1)
    c = torch.stack([1 - u[:, 2], u[:, 2]], -1)
    return (a[:, :, None, None] * b[:, None, :, None] * c[:, None, None, :]).reshape(-1, 8)


def ls_solve(A, B, s2):
    """Per-row least squares: coefficients and SSE from accumulated normal equations."""
    n = A.shape[0]
    coef = torch.empty(n, 8, device=dev)
    sse = torch.empty(n, dtype=torch.float64, device=dev)
    for s in range(0, n, 400_000):
        a = A[s : s + 400_000].reshape(-1, 8, 8).double()
        a = a + 1e-6 * a.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-12)[:, None, None] * torch.eye(8, device=dev, dtype=torch.float64)
        b = B[s : s + 400_000].double()
        c = torch.linalg.solve(a, b)
        coef[s : s + 400_000] = c.float()
        sse[s : s + 400_000] = (s2[s : s + 400_000] - (c * b).sum(1)).clamp_min(0)
    return coef, sse


# ------------------------------------------------------------------ volume evaluation (full-res, surface band)
def boundary(gt):
    mask = (gt >= TAU).float()
    pad = F.pad(mask[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    c = pad[1:-1, 1:-1, 1:-1]
    diff = torch.zeros_like(c, dtype=torch.bool)
    for s in ((2, 1, 1), (0, 1, 1), (1, 2, 1), (1, 0, 1), (1, 1, 2), (1, 1, 0)):
        diff |= pad[s[0] : s[0] + c.shape[0], s[1] : s[1] + c.shape[1], s[2] : s[2] + c.shape[2]] != c
    return diff


YS, XS = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")


def slab_points(z0, z1):
    zz = torch.arange(z0, z1, device=dev)[:, None, None].expand(-1, H, W)
    return torch.stack([-1 + (zz.float() + 0.5) * 2 / D, -1 + (YS[None].expand(z1 - z0, -1, -1).float() + 0.5) * 2 / H, -1 + (XS[None].expand(z1 - z0, -1, -1).float() + 0.5) * 2 / W], -1).reshape(-1, 3)


@torch.no_grad()
def evaluate_volume(model):
    model = model.eval()
    se = dict(band=0.0, far=0.0)
    cnt = dict(band=0, far=0)
    halo = BAND_R + 1
    for z0 in range(0, D, 8):
        z1 = min(D, z0 + 8)
        a, b = max(0, z0 - halo), min(D, z1 + halo)
        ext = torch.from_numpy(np.ascontiguousarray(GT[a:b])).to(dev)
        band = (F.max_pool3d(boundary(ext).float()[None, None], 2 * BAND_R + 1, stride=1, padding=BAND_R)[0, 0] > 0)[z0 - a : z1 - a]
        gt = ext[z0 - a : z1 - a]
        pts = slab_points(z0, z1)
        pred = torch.cat([model.forward_mu(pts[s : s + 4_000_000]) for s in range(0, pts.shape[0], 4_000_000)]).reshape(gt.shape)
        e2 = ((pred - gt).double()) ** 2
        for k, m in (("band", band), ("far", ~band)):
            se[k] += float(e2[m].sum())
            cnt[k] += int(m.sum())
    tot = se["band"] + se["far"]
    return {
        "volume_psnr_db": 10 * math.log10(D * H * W / tot),
        "band_voxel_share": cnt["band"] / (D * H * W),
        "band_sse_share": se["band"] / tot,
        "band_rmse": math.sqrt(se["band"] / cnt["band"]),
        "far_rmse": math.sqrt(se["far"] / cnt["far"]),
        "leaves": int(model.leaf_levels.shape[0]),
        "codec_bytes": codec_bytes(model, CODEC),
    }


# ------------------------------------------------------------------ model construction
def base_model(dataset_train):
    m = build_model(CFG).to(dev)
    _initialize_constant_mu(m, (dataset_train.projections.mean() * 0.5).clamp_min(1e-8))
    m.elevate_leaves_isotropic_batch(torch.arange(m.leaf_levels.shape[0], device=dev), (1, 1, 1))
    return m


def lin_ids(m, level_mask, size):
    c = m.leaf_coords[level_mask]
    return (c[:, 0] * size + c[:, 1]) * size + c[:, 2]


def oracle_model(dataset_train, s0, s1):
    m = base_model(dataset_train)
    ids0 = torch.nonzero(torch.isin(lin_ids(m, m.leaf_levels == 0, 48), s0)).flatten()
    m.split_leaves_batch(torch.nonzero(m.leaf_levels == 0).flatten()[ids0])
    l1 = torch.nonzero(m.leaf_levels == 1).flatten()
    ids1 = torch.nonzero(torch.isin(lin_ids(m, m.leaf_levels == 1, 96), s1)).flatten()
    m.split_leaves_batch(l1[ids1])
    return m


def our_tree_model(dataset_train):
    m = build_model(CFG)
    m.prepare_sparse_from_state_dict(CK["model"])
    m.load_state_dict(CK["model"], strict=True)
    return m.to(dev)


def set_coefficients(m, coef):  # coef (N, 8), leaf order == model leaf order, 8 contiguous per leaf
    assert int(m.coefficient_logits.numel()) == coef.shape[0] * 8
    with torch.no_grad():
        m.coefficient_logits.copy_(m._coefficients_to_logits(coef.clamp_min(1e-6)).reshape(-1))


def get_dataset():
    ds = load_r2_dataset(CFG["dataset"]["root"], device=dev, load_volume=False)
    return _split_dev_views(ds, CFG.get("dataset", {}) or {}, int(CFG["training"].get("seed", 0)))


# ------------------------------------------------------------------ stages
def stage_s0(ds):
    model = our_tree_model(ds.train).eval()
    dh, dw = ds.detector_shape
    dense = torch.empty(D, H, W, device=dev)
    pts_chunks = 0
    with torch.no_grad():
        for z0 in range(0, D, 8):
            z1 = min(D, z0 + 8)
            pts = slab_points(z0, z1)
            dense[z0:z1] = torch.cat([model.forward_mu(pts[s : s + 4_000_000]) for s in range(0, pts.shape[0], 4_000_000)]).reshape(z1 - z0, H, W)
        pd = project_dense_parallel(dense, ds.test.angles, detector_h=dh, detector_w=dw, samples_per_ray=1024)
        pe = render_split(model, ds.test, detector_h=dh, detector_w=dw, samples_per_ray=1024, ray_chunk=4096)
    data = ds.test.projections
    mse = lambda x, y: float(((x - y) ** 2).mean())
    ps = lambda m_: 10 * math.log10(float(data.max()) ** 2 / m_)
    save("s0_operator", {
        "mse_exact_vs_data": mse(pe, data), "mse_dense_vs_data": mse(pd, data), "mse_exact_vs_dense_same_function": mse(pe, pd),
        "psnr_exact_vs_data": ps(mse(pe, data)), "psnr_dense_vs_data": ps(mse(pd, data)), "psnr_exact_vs_dense": ps(mse(pe, pd)),
    })


@torch.no_grad()
def stage_ls_eval(ds):
    m = our_tree_model(ds.train).eval()
    N = int(m.leaf_levels.shape[0])
    A = torch.zeros(N, 64, device=dev)
    B = torch.zeros(N, 8, device=dev)
    s2 = torch.zeros(N, dtype=torch.float64, device=dev)
    shapes = torch.tensor(m.level_shapes, dtype=torch.float32, device=dev)
    for z0 in range(0, D, 2):
        z1 = min(D, z0 + 2)
        gt = torch.from_numpy(np.ascontiguousarray(GT[z0:z1])).to(dev).reshape(-1)
        pts = slab_points(z0, z1)
        for s in range(0, pts.shape[0], 2_000_000):
            p, v = pts[s : s + 2_000_000], gt[s : s + 2_000_000]
            _, ids = m.forward_mu_with_leaf_ids(p)
            size = 2.0 / shapes[m.leaf_levels[ids]]
            u = (((p + 1.0) - m.leaf_coords[ids].float() * size) / size).clamp(0, 1)
            b = trilinear(u)
            A.index_add_(0, ids, (b[:, :, None] * b[:, None, :]).reshape(-1, 64))
            B.index_add_(0, ids, b * v[:, None])
            s2.index_add_(0, ids, (v * v).double())
    coef, _ = ls_solve(A, B, s2)
    save("our_tree_current", evaluate_volume(m))
    set_coefficients(m, coef)
    save("our_tree_gt_ls", evaluate_volume(m))


@torch.no_grad()
def stage_oracle(ds):
    levels = (0, 1, 2)
    S = [48 << L for L in levels]
    A = [torch.zeros(s**3, 64, device=dev) for s in S]
    B = [torch.zeros(s**3, 8, device=dev) for s in S]
    s2 = [torch.zeros(s**3, dtype=torch.float64, device=dev) for s in S]
    t0 = time.time()
    for z0 in range(0, D, 2):
        z1 = min(D, z0 + 2)
        gt = torch.from_numpy(np.ascontiguousarray(GT[z0:z1])).to(dev).reshape(-1)
        iz = torch.arange(z0, z1, device=dev)
        for li, s in enumerate(S):
            pz = (iz.float() + 0.5) * s / D
            py = (torch.arange(H, device=dev).float() + 0.5) * s / H
            px = (torch.arange(W, device=dev).float() + 0.5) * s / W
            cz, cy, cx = pz.floor().clamp(0, s - 1), py.floor().clamp(0, s - 1), px.floor().clamp(0, s - 1)
            lin = ((cz[:, None, None] * s + cy[None, :, None]) * s + cx[None, None, :]).long().reshape(-1)
            u = torch.stack(torch.broadcast_tensors((pz - cz)[:, None, None], (py - cy)[None, :, None], (px - cx)[None, None, :]), -1).reshape(-1, 3)
            for q in range(0, lin.numel(), 1_000_000):
                b = trilinear(u[q : q + 1_000_000])
                v = gt[q : q + 1_000_000]
                A[li].index_add_(0, lin[q : q + 1_000_000], (b[:, :, None] * b[:, None, :]).reshape(-1, 64))
                B[li].index_add_(0, lin[q : q + 1_000_000], b * v[:, None])
                s2[li].index_add_(0, lin[q : q + 1_000_000], (v * v).double())
        if z0 % 64 == 0:
            print(f"accumulate z={z0}/{D} {time.time() - t0:.0f}s", flush=True)
    coefs, sses = [], []
    for li in range(3):
        c, e = ls_solve(A[li], B[li], s2[li])
        coefs.append(c)
        sses.append(e)
        A[li] = B[li] = s2[li] = None
    torch.cuda.empty_cache()
    base = base_model(ds.train)
    k_splits = (BUDGET - codec_bytes(base, CODEC)) // SPLIT_BYTES
    sse0, sse1, sse2 = sses

    def dp(lam):
        c1 = sse2.view(96, 2, 96, 2, 96, 2).sum((1, 3, 5))
        split1 = c1 + lam * SPLIT_BYTES
        j1 = torch.minimum(sse1.view(96, 96, 96), split1)
        d1 = split1 < sse1.view(96, 96, 96)
        c0 = j1.view(48, 2, 48, 2, 48, 2).sum((1, 3, 5))
        split0 = c0 + lam * SPLIT_BYTES
        d0 = split0 < sse0.view(48, 48, 48)
        d0_up = d0[:, None, :, None, :, None].expand(48, 2, 48, 2, 48, 2).reshape(96, 96, 96)
        return d0, d1 & d0_up

    lo, hi = 1e-6, 1e6
    for _ in range(80):
        lam = math.sqrt(lo * hi)
        d0, d1 = dp(lam)
        if int(d0.sum()) + int(d1.sum()) > k_splits:
            lo = lam
        else:
            hi = lam
    d0, d1 = dp(hi)
    s0 = torch.nonzero(d0.reshape(-1)).flatten()
    s1 = torch.nonzero(d1.reshape(-1)).flatten()
    m = oracle_model(ds.train, s0, s1)
    # assign GT-LS coefficients leaf by leaf
    N = int(m.leaf_levels.shape[0])
    coef = torch.empty(N, 8, device=dev)
    for L in (0, 1, 2):
        mask = m.leaf_levels == L
        if bool(mask.any()):
            coef[mask] = coefs[L][lin_ids(m, mask, 48 << L)]
    set_coefficients(m, coef)
    ev = evaluate_volume(m)
    ev.update(splits_L0=int(s0.numel()), splits_L1=int(s1.numel()), level_leaves=torch.bincount(m.leaf_levels, minlength=4).tolist(), lambda_=hi)
    save("oracle_tree_gt_ls", ev)
    torch.save({"s0": s0.cpu(), "s1": s1.cpu(), "coef": coef.cpu()}, OUT / "oracle_tree.pt")


def train_fixed(m, split, iters, tag, ds, eval_every=0):
    dh, dw = ds.detector_shape
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=float(CFG["training"].get("lr", 0.015)))
    wk = projection_weight_kwargs(CFG["training"], iteration=1)
    compact = m.prefer_compact_ray_batch()
    t0 = time.time()
    hist = []
    for it in range(1, iters + 1):
        rb = random_training_rays(split, batch_rays=int(CFG["training"].get("batch_rays", 65536)), detector_h=dh, detector_w=dw, samples_per_ray=1024, materialize_points=not compact)
        loss = weighted_projection_loss(m.integrate_ray_batch(rb), rb.target, **wk)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 5000 == 0:
            print(f"{tag} it {it}/{iters} loss {float(loss):.3e} {time.time() - t0:.0f}s", flush=True)
        if eval_every and it % eval_every == 0:
            e = evaluate_volume(m)
            hist.append({"it": it, "train_loss": float(loss), "volume_psnr_db": e["volume_psnr_db"], "band_rmse": e["band_rmse"], "far_rmse": e["far_rmse"]})
            print(tag, hist[-1], flush=True)
    with torch.no_grad():
        pm = projection_metrics(render_split(m, ds.test, detector_h=dh, detector_w=dw, samples_per_ray=1024, ray_chunk=4096), ds.test.projections)
    ev = evaluate_volume(m)
    ev["test_projection_psnr"] = pm["psnr"]
    if hist:
        ev["history"] = hist
    return ev


def oracle_tree_for(ds):
    t = torch.load(OUT / "oracle_tree.pt")
    return oracle_model(ds.train, t["s0"].to(dev), t["s1"].to(dev))


def stage_train(ds, iters=30000):
    save("train_oracle_tree", train_fixed(oracle_tree_for(ds), ds.train, iters, "oracle", ds))
    m = our_tree_model(ds.train)
    _initialize_constant_mu(m, (ds.train.projections.mean() * 0.5).clamp_min(1e-8))
    save("train_our_tree", train_fixed(m, ds.train, iters, "ourtree", ds))


def stage_warm(ds, iters=10000):
    """Start the oracle tree at its GT least-squares coefficients, then train from projections only."""
    m = oracle_tree_for(ds)
    set_coefficients(m, torch.load(OUT / "oracle_tree.pt")["coef"].to(dev))
    save("warm_oracle_tree", train_fixed(m, ds.train, iters, "warm", ds, eval_every=2000))


def stage_views(ds, iters=30000, extra=480):
    dh, dw = ds.detector_shape
    angles = torch.tensor([math.pi * (i + 0.37) / extra for i in range(extra)], dtype=torch.float32, device=dev)
    vol = torch.from_numpy(np.ascontiguousarray(GT)).to(dev)
    projs = torch.cat([project_dense_parallel(vol, angles[s : s + 16], detector_h=dh, detector_w=dw, samples_per_ray=1024) for s in range(0, extra, 16)])
    del vol
    torch.cuda.empty_cache()
    big = ProjectionSplit(angles=torch.cat([ds.train.angles, angles]), projections=torch.cat([ds.train.projections, projs]), paths=[])
    save("train_oracle_tree_3x_views", {**train_fixed(oracle_tree_for(ds), big, iters, "views", ds), "train_views": int(big.angles.numel())})


if __name__ == "__main__":
    ds = get_dataset()
    for st in STAGES:
        {"s0": stage_s0, "ls_eval": stage_ls_eval, "oracle": stage_oracle, "train": stage_train, "views": stage_views, "warm": stage_warm}[st](ds)
