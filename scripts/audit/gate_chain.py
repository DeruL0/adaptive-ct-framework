"""Offline diagnostics of the main pipeline (use the GT volume; never part of training).

  ls_eval     the tree of output/<run> + per-leaf GT least-squares values        (how good is the chosen tree)
  oracle      best tree under the byte budget (Lagrangian DP, L0..L2) + GT LS values (capacity of the representation)
  speed       solver from a cold start on the oracle tree, to the stopping tolerance
  profile     capacity of the adjustable-width step leaf vs constant / trilinear leaves (GT fits)
  main_eval   volume metrics and median slice of output/<run>

usage: python gate_chain.py <run> <stage> [<stage> ...]     results -> output/_chain/results.json
Every model-producing stage saves its checkpoint and a median slice figure (GT | model | reference | error).
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
from adaptive_ct.codec.budget import codec_bytes, split_delta_bytes
from adaptive_ct.data.dataset import load_r2_dataset
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
CODEC = {"type": "p1", "quantization": "float16"}  # what the main pipeline stores
BUDGET = 8_388_608
SPLIT_BYTES = split_delta_bytes(CODEC)
TAU, BAND_R = 0.1, 3


def save(key, value):
    RES[key] = value
    RES_PATH.write_text(json.dumps(RES, indent=1, default=float))
    print(key, json.dumps(value, default=float), flush=True)


_REFERENCE = {}


def report(key, model, value):
    """Every model-producing stage: save checkpoint + median slice figure, then the metrics."""
    torch.save({"model": model.state_dict(), "config": CFG}, OUT / f"{key}.pt")
    if "ours" not in _REFERENCE:
        _REFERENCE["ours"] = our_tree_model(None).eval()
    value = dict(value)
    value.update(slice_figure([model, _REFERENCE["ours"]], OUT / f"{key}_slice.png"))
    value["checkpoint"] = str(OUT / f"{key}.pt")
    value["slice_png"] = str(OUT / f"{key}_slice.png")
    save(key, value)


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
    m.to_trilinear(torch.arange(m.leaf_levels.shape[0], device=dev))
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
    report("our_tree_gt_ls", m, evaluate_volume(m))


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
    report("oracle_tree_gt_ls", m, ev)
    torch.save({"s0": s0.cpu(), "s1": s1.cpu(), "coef": coef.cpu()}, OUT / "oracle_tree.pt")


def oracle_tree_for(ds):
    t = torch.load(OUT / "oracle_tree.pt")
    return oracle_model(ds.train, t["s0"].to(dev), t["s1"].to(dev))


def slice_figure(models, path, z=None, crop=(440, 560, 256)):
    """Median axial slice: GT | each model | |GT - first model|, full slice (downsampled) + zoomed crop."""
    from PIL import Image
    z = D // 2 if z is None else z
    gt = np.asarray(GT[z], dtype=np.float32)
    pts = torch.stack([torch.full((H, W), -1 + (z + 0.5) * 2 / D, device=dev), -1 + (YS.float() + 0.5) * 2 / H, -1 + (XS.float() + 0.5) * 2 / W], -1).reshape(-1, 3)
    preds = []
    with torch.no_grad():
        for m in models:
            preds.append(m.eval().forward_mu(pts).reshape(H, W).cpu().numpy())
    img = lambda a, hi=0.8: (np.clip(a / hi, 0, 1) * 255).astype(np.uint8)
    cy, cx, s = crop
    sl = (slice(cy - s // 2, cy + s // 2), slice(cx - s // 2, cx + s // 2))
    full = [img(gt)] + [img(p) for p in preds] + [img(np.abs(preds[0] - gt), 0.2)]
    zoom = [np.kron(img(a[sl]) if i < len(full) - 1 else img(np.abs(preds[0] - gt)[sl], 0.2), np.ones((4, 4), np.uint8)) for i, a in enumerate([gt] + preds + [gt])]
    top = np.concatenate([f[::1, ::1] for f in full], 1)
    bottom = np.concatenate(zoom, 1)
    Image.fromarray(np.concatenate([top, bottom], 0)).save(path)
    return {"slice_z": z, "crop_rows_cols": [cy - s // 2, cy + s // 2, cx - s // 2, cx + s // 2]}


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu")
    m = build_model(ck["config"])
    m.prepare_sparse_from_state_dict(ck["model"])
    m.load_state_dict(ck["model"], strict=True)
    return m.to(dev).eval()


def stage_main_eval(ds):
    """output/<RUN>/checkpoint.pt (deployed, fp16): volume metrics; slice figure next to the oracle tree solved from projections."""
    run_dir = ROOT / "output" / RUN
    deployed = load_checkpoint(run_dir / "checkpoint.pt")
    reference = load_checkpoint(OUT / "speed_full_tol1e-4.pt")
    out = {"deployed": evaluate_volume(deployed), "reference_oracle_tree_solved": evaluate_volume(reference)}
    out.update(slice_figure([deployed, reference], run_dir / "slice_vs_oracle_tree.png"))
    out["slice_png"] = str(run_dir / "slice_vs_oracle_tree.png")
    out["panel_order"] = "GT | this run | oracle tree solved from projections | |error of this run|"
    save(f"main_eval_{RUN}", out)


def stage_speed(ds):
    """Accelerated solver on the oracle tree, cold start: tolerance stop, with / without a 2x2-binned warm-up.
    Reference: 80 fixed passes, 57 min, 36.58 dB (solve_C_groups24_then_extrapolated_80)."""
    from adaptive_ct.training.solve import solve
    dh, dw = ds.detector_shape
    n_rays = int(ds.train.angles.numel()) * dh * dw
    for name, steps in {
        "speed_full_tol1e-4": [dict(passes=150, tol=1e-4, groups=24, group_passes=4)],
        "speed_bin2_then_full_tol1e-4": [dict(passes=30, tol=1e-3, groups=24, group_passes=4, detector_bin=2), dict(passes=150, tol=1e-4, groups=24)],
    }.items():
        m = oracle_tree_for(ds)
        t0, hist, out = time.time(), [], {}
        for i, kw in enumerate(steps):
            _, h, conv = solve(m, ds.train, **kw)
            hist += [{**x, "step": i} for x in h]
            out[f"step{i}"] = {"passes": h[-1]["pass"], "converged": conv, "seconds": time.time() - t0, **{k: kw[k] for k in ("tol", "detector_bin") if k in kw}}
        last = hist[-1]
        with torch.no_grad():
            pm = projection_metrics(render_split(m, ds.test, detector_h=dh, detector_w=dw), ds.test.projections)
        report(name, m, {**evaluate_volume(m), **out, "seconds": time.time() - t0, "train_mse": 2 * last["objective_at_eval_point"] / n_rays,
                         "optimality_residual": last["optimality_residual"], "test_projection_psnr": pm["psnr"], "history": hist})


@torch.no_grad()
def stage_profile(ds):
    """Capacity check of the adjustable-width step leaf  (1 - H) c1 + H c2,  H = sigmoid((n.(u - 1/2) - d) / w)
    (w = 0: hard step), against constant / trilinear leaves: every leaf of the oracle tree fitted to GT
    (values >= 0, parameters rounded to fp16), then the best leaf type per leaf under the 8 MiB budget,
    direct fp16 bytes: constant 5, trilinear 19, step 15, split 64. Same tree for every variant."""
    m = oracle_tree_for(ds).eval()
    N = int(m.leaf_levels.shape[0])
    shapes = torch.tensor(m.level_shapes, dtype=torch.float32, device=dev)
    NB, NN = 48, 9  # distance bins along a direction; candidate directions per leaf
    edges = torch.linspace(-0.87, 0.87, NB + 1, device=dev)
    centres = 0.5 * (edges[1:] + edges[:-1])
    box_scale = torch.tensor([D / 2, H / 2, W / 2], device=dev)  # voxel -> box-coordinate gradient

    def slabs(z_step=2, halo=0):
        for z0 in range(0, D, z_step):
            z1 = min(D, z0 + z_step)
            a, b = max(0, z0 - halo), min(D, z1 + halo)
            ext = torch.from_numpy(np.ascontiguousarray(GT[a:b])).to(dev)
            pts = slab_points(z0, z1)
            _, ids = m.forward_mu_with_leaf_ids(pts)
            size = 2.0 / shapes[m.leaf_levels[ids]]
            u = (((pts + 1.0) - m.leaf_coords[ids].float() * size) / size).clamp(0, 1)
            yield z0, z1, a, ext, ids, u

    # pass 1: constant / trilinear normal equations and the gradient second-moment matrix per leaf
    cnt = torch.zeros(N, device=dev, dtype=torch.float64)
    s1 = torch.zeros(N, device=dev, dtype=torch.float64)
    s2 = torch.zeros(N, device=dev, dtype=torch.float64)
    A = torch.zeros(N, 64, device=dev)
    B = torch.zeros(N, 8, device=dev)
    T = torch.zeros(N, 9, device=dev)
    for z0, z1, a, ext, ids, u in slabs(halo=1):
        v = ext[z0 - a : z1 - a].reshape(-1)
        g = torch.stack(torch.gradient(ext), -1)[z0 - a : z1 - a].reshape(-1, 3) * box_scale
        b = trilinear(u)
        cnt.index_add_(0, ids, torch.ones_like(v, dtype=torch.float64))
        s1.index_add_(0, ids, v.double())
        s2.index_add_(0, ids, (v * v).double())
        A.index_add_(0, ids, (b[:, :, None] * b[:, None, :]).reshape(-1, 64))
        B.index_add_(0, ids, b * v[:, None])
        T.index_add_(0, ids, (g[:, :, None] * g[:, None, :]).reshape(-1, 9))
    c0 = (s1 / cnt.clamp_min(1)).float()
    c8, _ = ls_solve(A, B, s2)
    Am = A.view(N, 8, 8)
    step = 1.0 / torch.linalg.eigvalsh(Am.double()).amax(1).float().clamp_min(1e-12)
    c8 = c8.clamp_min(0)
    for _ in range(300):  # projected gradient: nonnegative least squares
        c8 = (c8 - step[:, None] * (torch.bmm(Am, c8[:, :, None])[:, :, 0] - B)).clamp_min(0)
    del A, Am, B
    # candidate directions: dominant gradient direction + 8 directions tilted by 12 degrees around it
    n0 = torch.linalg.eigh(T.view(N, 3, 3).double())[1][:, :, -1].float()
    t1 = torch.linalg.cross(n0, torch.where(n0[:, :1].abs() < 0.9, torch.tensor([1.0, 0, 0], device=dev), torch.tensor([0, 1.0, 0], device=dev)).expand_as(n0))
    t1 = t1 / t1.norm(dim=1, keepdim=True)
    t2 = torch.linalg.cross(n0, t1)
    phi = torch.arange(8, device=dev) * (math.pi / 4)
    tilt = math.tan(math.radians(12))
    normals = torch.cat([n0[:, None], n0[:, None] + tilt * (phi.cos()[None, :, None] * t1[:, None] + phi.sin()[None, :, None] * t2[:, None])], 1)
    normals = normals / normals.norm(dim=2, keepdim=True)  # (N, 9, 3)
    del T

    # pass 2: per leaf, per direction, voxel count and value sum in distance bins
    bc = torch.zeros(N * NN * NB, device=dev)
    bs = torch.zeros(N * NN * NB, device=dev)
    for z0, z1, a, ext, ids, u in slabs():
        v = ext[z0 - a : z1 - a].reshape(-1)
        s = torch.einsum("pkc,pc->pk", normals[ids], u - 0.5)
        flat = (ids[:, None] * NN + torch.arange(NN, device=dev)) * NB + torch.bucketize(s, edges[1:-1])
        bc.index_add_(0, flat.reshape(-1), torch.ones(flat.numel(), device=dev))
        bs.index_add_(0, flat.reshape(-1), v[:, None].expand_as(s).reshape(-1))
    bc, bs = bc.view(N, NN, NB), bs.view(N, NN, NB)

    # geometry search on the bins: positions = bin edges, widths in voxels (xy voxel of the leaf's level)
    widths_vox = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0], device=dev)
    best = dict(sse=torch.full((N,), float("inf"), device=dev), k=torch.zeros(N, dtype=torch.long, device=dev),
                d=torch.zeros(N, device=dev), w=torch.zeros(N, device=dev), c=torch.zeros(N, 2, device=dev))
    for L in range(len(m.level_shapes)):
        sel = torch.nonzero(m.leaf_levels == L).flatten()
        if sel.numel() == 0:
            continue
        ws = widths_vox * float(shapes[L, 1]) / W  # in leaf units
        dd, ww = torch.meshgrid(edges, ws, indexing="ij")
        dd, ww = dd.reshape(-1), ww.reshape(-1)
        Hm = torch.where(ww[:, None] > 0, torch.sigmoid((centres[None] - dd[:, None]) / ww[:, None].clamp_min(1e-9)), (centres[None] > dd[:, None]).float())  # (G, NB)
        for q in range(0, sel.numel(), 6000):
            li = sel[q : q + 6000]
            C, S = bc[li], bs[li]
            a11, a12, a22 = C @ ((1 - Hm) ** 2).T, C @ (Hm * (1 - Hm)).T, C @ (Hm**2).T
            b1, b2 = S @ (1 - Hm).T, S @ Hm.T
            ss = s2[li].float()[:, None, None]
            det = a11 * a22 - a12**2
            u1, u2 = (a22 * b1 - a12 * b2) / det.clamp_min(1e-12), (a11 * b2 - a12 * b1) / det.clamp_min(1e-12)
            cand = []
            for x1, x2 in ((u1, u2), ((b1 / a11.clamp_min(1e-12)).clamp_min(0), torch.zeros_like(u1)), (torch.zeros_like(u1), (b2 / a22.clamp_min(1e-12)).clamp_min(0))):
                e = ss - 2 * (x1 * b1 + x2 * b2) + x1 * x1 * a11 + 2 * x1 * x2 * a12 + x2 * x2 * a22
                ok = (x1 >= 0) & (x2 >= 0) & (x1 <= 2) & (x2 <= 2) & (det > 1e-9 * (a11 * a22).clamp_min(1e-30) if x1 is u1 else True)
                cand.append((torch.where(ok, e, torch.full_like(e, float("inf"))), x1, x2))
            e = torch.stack([c[0] for c in cand])  # (3, n, NN, G)
            which = e.argmin(0)
            e = e.min(0).values
            x1 = torch.stack([c[1] for c in cand]).gather(0, which[None])[0]
            x2 = torch.stack([c[2] for c in cand]).gather(0, which[None])[0]
            flat = e.reshape(li.numel(), -1)
            j = flat.argmin(1)
            k, g = j // dd.numel(), j % dd.numel()
            best["sse"][li], best["k"][li], best["d"][li], best["w"][li] = flat.gather(1, j[:, None])[:, 0], k, dd[g], ww[g]
            best["c"][li] = torch.stack([x1.reshape(li.numel(), -1).gather(1, j[:, None])[:, 0], x2.reshape(li.numel(), -1).gather(1, j[:, None])[:, 0]], 1)
    del bc, bs
    half = lambda x: x.half().float()
    n_best = normals[torch.arange(N, device=dev), best["k"]]
    n_best = half(n_best)
    n_best = n_best / n_best.norm(dim=1, keepdim=True)
    d_best, w_best, c_step, c0, c8 = half(best["d"]), half(best["w"]), half(best["c"]), half(c0), half(c8)

    # pass 3: exact per-voxel errors with the fp16 parameters (total and surface band), and the median slice
    types = ("constant", "trilinear", "step")
    sse = torch.zeros(3, N, device=dev, dtype=torch.float64)
    band_sse = torch.zeros(3, N, device=dev, dtype=torch.float64)
    z_mid, slice_pred = D // 2, None
    for z0, z1, a, ext, ids, u in slabs(z_step=2, halo=BAND_R + 1):
        gt = ext[z0 - a : z1 - a].reshape(-1)
        band = (F.max_pool3d(boundary(ext).float()[None, None], 2 * BAND_R + 1, stride=1, padding=BAND_R)[0, 0] > 0)[z0 - a : z1 - a].reshape(-1)
        s = (n_best[ids] * (u - 0.5)).sum(1)
        wv = w_best[ids]
        Hv = torch.where(wv > 0, torch.sigmoid((s - d_best[ids]) / wv.clamp_min(1e-9)), (s > d_best[ids]).float())
        preds = torch.stack([c0[ids], (trilinear(u) * c8[ids]).sum(1), (1 - Hv) * c_step[ids, 0] + Hv * c_step[ids, 1]])
        e2 = ((preds - gt) ** 2).double()
        for t in range(3):
            sse[t].index_add_(0, ids, e2[t])
            band_sse[t].index_add_(0, ids, e2[t] * band)
        if z0 <= z_mid < z1:
            slice_pred = preds.view(3, z1 - z0, H, W)[:, z_mid - z0]

    # best leaf type per leaf under the byte budget (bisection on the price of one byte)
    splits = (N - int(np.prod(m.level_shapes[0]))) // 7
    leaf_budget = BUDGET - 64 * splits
    nbytes = torch.tensor([5.0, 19.0, 15.0], device=dev, dtype=torch.float64)

    def choose(allowed):
        cost = sse[allowed]
        lo, hi = 0.0, 1.0
        for _ in range(100):
            lam = 0.5 * (lo + hi)
            pick = (cost + lam * nbytes[allowed][:, None]).argmin(0)
            if float(nbytes[allowed][pick].sum()) > leaf_budget:
                lo = lam
            else:
                hi = lam
        pick = (cost + hi * nbytes[allowed][:, None]).argmin(0)
        t = torch.tensor(allowed, device=dev)[pick]
        tot, bnd = float(sse.gather(0, t[None])[0].sum()), float(band_sse.gather(0, t[None])[0].sum())
        return t, {"volume_psnr_db": 10 * math.log10(D * H * W / tot), "sse": tot, "band_sse": bnd, "far_sse": tot - bnd,
                   "bytes": int(nbytes[t].sum()) + 64 * splits, "leaf_types": {types[i]: int((t == i).sum()) for i in range(3)}}

    t_old, old = choose([0, 1])
    t_new, new = choose([0, 1, 2])
    per_type = {types[i]: {"volume_psnr_db": 10 * math.log10(D * H * W / float(sse[i].sum())), "band_sse": float(band_sse[i].sum()),
                           "bytes_all_leaves": int(nbytes[i]) * N + 64 * splits} for i in range(3)}
    surface = band_sse[1] > 0.5 * sse[1]  # leaves whose trilinear error sits mostly in the surface band
    ratio = (sse[2][surface] / sse[1][surface].clamp_min(1e-30)).float()
    widths = torch.bucketize(w_best[t_new == 2] * W / shapes[m.leaf_levels[t_new == 2], 1], widths_vox + 0.05)
    out = {
        "leaves": N, "splits": splits, "budget_bytes": BUDGET,
        "every_leaf_one_type": per_type,
        "mix_constant_trilinear": old,
        "mix_constant_trilinear_step": new,
        "band_sse_change": new["band_sse"] / old["band_sse"] - 1,
        "total_sse_change": new["sse"] / old["sse"] - 1,
        "pass_criterion": "band SSE -20% or better and total SSE not worse",
        "passed": new["band_sse"] <= 0.8 * old["band_sse"] and new["sse"] <= old["sse"],
        "surface_leaves_step_over_trilinear_sse": {"median": float(ratio.median()), "share_halved": float((ratio < 0.5).float().mean()), "count": int(surface.sum())},
        "chosen_step_width_vox_histogram": {f"{float(w):g}": int((widths == i).sum()) for i, w in enumerate(widths_vox)},
    }
    torch.save({"types": t_new.cpu(), "constant": c0.cpu(), "trilinear": c8.cpu(), "step_values": c_step.cpu(), "step_normal": n_best.cpu(),
                "step_position": d_best.cpu(), "step_width": w_best.cpu(), "config": CFG}, OUT / "profile_capacity.pt")

    class Slice:  # slice_figure reads the median slice through forward_mu
        def __init__(self, img):
            self.img = img

        def eval(self):
            return self

        def forward_mu(self, pts):
            return self.img.reshape(-1)

    ids_mid = None
    for z0, z1, a, ext, ids, u in slabs(z_step=1):
        if z0 == z_mid:
            ids_mid = ids
            break
    pick = lambda t: slice_pred.reshape(3, -1).gather(0, t[ids_mid][None])[0]
    out.update(slice_figure([Slice(pick(t_new)), Slice(pick(t_old))], OUT / "profile_capacity_slice.png"))
    out["slice_png"] = str(OUT / "profile_capacity_slice.png")
    out["panel_order"] = "GT | constant/trilinear/step mix | constant/trilinear mix | |error of the step mix|"
    save("profile_capacity", out)


if __name__ == "__main__":
    ds = get_dataset()
    for st in STAGES:
        {"ls_eval": stage_ls_eval, "oracle": stage_oracle, "speed": stage_speed, "profile": stage_profile, "main_eval": stage_main_eval}[st](ds)
