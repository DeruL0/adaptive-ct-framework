import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Projects\7.CT Framework")
from adaptive_ct.representation.factory import build_model

ROOT = Path(r"D:\Projects\7.CT Framework")
GT = np.load(ROOT / "output/figure473_dicom_orig_r2_convergence/vol_gt.npy", mmap_mode="r")
D, H, W = GT.shape
TAU = 0.1
R = 4  # band radius in voxels
SLAB = 12
dev = torch.device("cuda")


def load(run, codec):
    ck = torch.load(ROOT / "output" / run / ("checkpoint_codec.pt" if codec else "checkpoint.pt"), map_location="cpu")
    cfg = dict(ck["config"])
    if codec:
        cfg["model"] = dict(cfg["model"], representation="latent_p1_octree", latent_dim=4)
    m = build_model(cfg)
    m.prepare_sparse_from_state_dict(ck["model"])
    m.load_state_dict(ck["model"], strict=True)
    return m.to(dev).eval()


CLASSES = ["air_far", "air_near_surface", "object_near_surface", "object_far"]


def evaluate(model, name):
    acc = {c: dict(n=0, se=0.0, ae=0.0, bias=0.0, levels=np.zeros(8)) for c in CLASSES}
    tot = dict(n=0, se=0.0)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    t0 = time.time()
    for z0 in range(0, D, SLAB):
        z1 = min(D, z0 + SLAB)
        a, b = max(0, z0 - R - 1), min(D, z1 + R + 1)
        gt_ext = torch.from_numpy(np.ascontiguousarray(GT[a:b])).to(dev)
        mask = (gt_ext >= TAU).float()
        # boundary voxels: mask differs from any 6-neighbour
        pad = F.pad(mask[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
        c = pad[1:-1, 1:-1, 1:-1]
        diff = torch.zeros_like(c, dtype=torch.bool)
        for s in ((2, 1, 1), (0, 1, 1), (1, 2, 1), (1, 0, 1), (1, 1, 2), (1, 1, 0)):
            nb = pad[s[0] : s[0] + c.shape[0], s[1] : s[1] + c.shape[1], s[2] : s[2] + c.shape[2]]
            diff |= nb != c
        band = F.max_pool3d(diff.float()[None, None], 2 * R + 1, stride=1, padding=R)[0, 0] > 0
        lo, hi = z0 - a, z1 - a
        gt = gt_ext[lo:hi]
        bnd = band[lo:hi]
        zs = torch.arange(z0, z1, device=dev)
        zz = zs[:, None, None].expand(-1, H, W)
        pts = torch.stack(
            [
                -1 + (zz.float() + 0.5) * 2 / D,
                -1 + (ys[None].expand(z1 - z0, -1, -1).float() + 0.5) * 2 / H,
                -1 + (xs[None].expand(z1 - z0, -1, -1).float() + 0.5) * 2 / W,
            ],
            dim=-1,
        ).reshape(-1, 3)
        preds, lvls = [], []
        with torch.no_grad():
            for s in range(0, pts.shape[0], 2_000_000):
                v, ids = model.forward_mu_with_leaf_ids(pts[s : s + 2_000_000])
                preds.append(v)
                lvls.append(model.leaf_levels[ids.clamp_min(0)])
        pred = torch.cat(preds).reshape(gt.shape)
        lvl = torch.cat(lvls).reshape(gt.shape)
        err = pred - gt
        obj = gt >= TAU
        cls = {
            "air_far": (~obj) & (~bnd),
            "air_near_surface": (~obj) & bnd,
            "object_near_surface": obj & bnd,
            "object_far": obj & (~bnd),
        }
        for k, m in cls.items():
            n = int(m.sum())
            if n == 0:
                continue
            e = err[m]
            acc[k]["n"] += n
            acc[k]["se"] += float((e.double() ** 2).sum())
            acc[k]["ae"] += float(e.abs().double().sum())
            acc[k]["bias"] += float(e.double().sum())
            acc[k]["levels"] += torch.bincount(lvl[m], minlength=8).cpu().numpy()[:8]
        tot["n"] += gt.numel()
        tot["se"] += float((err.double() ** 2).sum())
    n_all = tot["n"]
    mse = tot["se"] / n_all
    out = {"name": name, "psnr_global": 10 * np.log10(1.0 / mse), "mae_global": sum(v["ae"] for v in acc.values()) / n_all, "seconds": time.time() - t0}
    sse_total = sum(v["se"] for v in acc.values())
    for k, v in acc.items():
        if v["n"] == 0:
            continue
        out[k] = {
            "voxel_fraction": v["n"] / n_all,
            "sse_share": v["se"] / sse_total,
            "rmse": float(np.sqrt(v["se"] / v["n"])),
            "mae": v["ae"] / v["n"],
            "bias": v["bias"] / v["n"],
            "level_share": (v["levels"] / max(v["levels"].sum(), 1)).round(4).tolist(),
        }
    return out


results = []
for run, codec, label in [
    ("stage7_dorfler_p1_256v", True, "p1_latent_deployed"),
    ("stage7_dorfler_p1_256v", False, "p1_trained"),
    ("stage7_dorfler_p0_longft_256v", False, "p0_longft"),
]:
    model = load(run, codec)
    r = evaluate(model, label)
    results.append(r)
    print(json.dumps(r, indent=1, default=float), flush=True)
    del model
    torch.cuda.empty_cache()
json.dump(results, open(Path(sys.argv[1]), "w"), indent=1, default=float)
