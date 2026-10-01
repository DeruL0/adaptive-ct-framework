import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, r"D:\Projects\7.CT Framework")
from adaptive_ct.representation.factory import build_model

ROOT = Path(r"D:\Projects\7.CT Framework")
SP = Path(sys.argv[1])
GT = np.load(ROOT / "output/figure473_dicom_orig_r2_convergence/vol_gt.npy", mmap_mode="r")
D, H, W = GT.shape
dev = torch.device("cuda")
SLAB = 12
ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")


def load(run, codec):
    ck = torch.load(ROOT / "output" / run / ("checkpoint_codec.pt" if codec else "checkpoint.pt"), map_location="cpu")
    cfg = dict(ck["config"])
    if codec:
        cfg["model"] = dict(cfg["model"], representation="latent_p1_octree", latent_dim=4)
    m = build_model(cfg)
    m.prepare_sparse_from_state_dict(ck["model"])
    m.load_state_dict(ck["model"], strict=True)
    return m.to(dev).eval()


def points(z0, z1):
    zz = torch.arange(z0, z1, device=dev)[:, None, None].expand(-1, H, W)
    return torch.stack(
        [-1 + (zz.float() + 0.5) * 2 / D, -1 + (ys[None].expand(z1 - z0, -1, -1).float() + 0.5) * 2 / H, -1 + (xs[None].expand(z1 - z0, -1, -1).float() + 0.5) * 2 / W],
        dim=-1,
    ).reshape(-1, 3)


def leaf_ids(model, pts):
    out = []
    with torch.no_grad():
        for s in range(0, pts.shape[0], 4_000_000):
            out.append(model.resolve_leaf_ids(pts[s : s + 4_000_000]))
    return torch.cat(out)


def oracle_p0(model):
    """Best piecewise-constant fit on this topology: leaf mean of GT."""
    n = model.leaf_levels.shape[0]
    s = torch.zeros(n, dtype=torch.float64, device=dev)
    c = torch.zeros(n, dtype=torch.float64, device=dev)
    for z0 in range(0, D, SLAB):
        z1 = min(D, z0 + SLAB)
        gt = torch.from_numpy(np.ascontiguousarray(GT[z0:z1])).to(dev).reshape(-1).double()
        ids = leaf_ids(model, points(z0, z1))
        s.index_add_(0, ids, gt)
        c.index_add_(0, ids, torch.ones_like(gt))
    mean = s / c.clamp_min(1)
    se = 0.0
    se_by_level = np.zeros(8)
    cnt_by_level = np.zeros(8)
    for z0 in range(0, D, SLAB):
        z1 = min(D, z0 + SLAB)
        gt = torch.from_numpy(np.ascontiguousarray(GT[z0:z1])).to(dev).reshape(-1).double()
        ids = leaf_ids(model, points(z0, z1))
        e = (mean[ids] - gt) ** 2
        se += float(e.sum())
        lv = model.leaf_levels[ids]
        se_by_level += torch.bincount(lv, weights=e, minlength=8).cpu().numpy()[:8]
        cnt_by_level += torch.bincount(lv, minlength=8).cpu().numpy()[:8]
    return 10 * np.log10(1.0 / (se / (D * H * W))), se_by_level / se, cnt_by_level / (D * H * W)


# ---- GT noise floor: residual of GT against its 3x3x3 box mean (air_far-like regions)
def noise_floor():
    res2, n = 0.0, 0
    air_vals = []
    for z0 in range(1, D - 1, 24):
        z1 = min(D - 1, z0 + 24)
        g = torch.from_numpy(np.ascontiguousarray(GT[z0 - 1 : z1 + 1])).to(dev)[None, None]
        box = F.avg_pool3d(g, 3, stride=1, padding=0)[0, 0]
        c = g[0, 0, 1:-1, 1:-1, 1:-1]
        box = box
        # restrict to flat air: box mean tiny
        m = box < 0.02
        d = (c - box)[m]
        res2 += float((d.double() ** 2).sum())
        n += int(m.sum())
        air_vals.append(c[m].flatten()[:200000].cpu())
    air = torch.cat(air_vals)
    return {"air_voxels": n, "highfreq_rmse_in_air": float(np.sqrt(res2 / max(n, 1))), "air_mean": float(air.mean()), "air_std": float(air.std())}


out = {"gt_noise_floor": noise_floor()}
print(json.dumps(out["gt_noise_floor"]), flush=True)
for run, codec, label in [("stage7_dorfler_p1_256v", False, "p1_trained_topology"), ("stage7_dorfler_p0_longft_256v", False, "p0_topology")]:
    m = load(run, codec)
    psnr, se_lv, cnt_lv = oracle_p0(m)
    out[label] = {"leaves": int(m.leaf_levels.shape[0]), "oracle_p0_leaf_mean_psnr_db": float(psnr), "sse_share_by_level": se_lv.round(4).tolist()[:5], "voxel_share_by_level": cnt_lv.round(4).tolist()[:5]}
    print(label, json.dumps(out[label]), flush=True)
    del m
    torch.cuda.empty_cache()
json.dump(out, open(SP / "audit_oracle.json", "w"), indent=1)

# ---- slice figure: GT / P1 deployed / |error|, crop around an interface-rich region
model = load("stage7_dorfler_p1_256v", True)
z = D // 2
gt = np.asarray(GT[z])
pts = torch.stack([torch.full((H, W), -1 + (z + 0.5) * 2 / D, device=dev), -1 + (ys.float() + 0.5) * 2 / H, -1 + (xs.float() + 0.5) * 2 / W], dim=-1).reshape(-1, 3)
with torch.no_grad():
    pred = model.forward_mu(pts).reshape(H, W).cpu().numpy()
ids = leaf_ids(model, pts).reshape(H, W)
lvl = model.leaf_levels[ids].cpu().numpy()
cy, cx, s = 440, 560, 256
crop = (slice(cy - s // 2, cy + s // 2), slice(cx - s // 2, cx + s // 2))


def to_img(a, lo=0.0, hi=0.8):
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


err = np.abs(pred - gt)
panels = [to_img(gt[crop]), to_img(pred[crop]), to_img(err[crop], 0, 0.2)]
# leaf boundaries overlay: mark pixels where leaf id changes
idc = ids.cpu().numpy()
edge = np.zeros((H, W), bool)
edge[:, 1:] |= idc[:, 1:] != idc[:, :-1]
edge[1:, :] |= idc[1:, :] != idc[:-1, :]
ov = np.stack([to_img(gt[crop])] * 3, axis=-1)
ov[edge[crop]] = [255, 60, 60]
rgb = [np.stack([p] * 3, axis=-1) for p in panels] + [ov]
img = np.concatenate(rgb, axis=1)
img = np.kron(img, np.ones((2, 2, 1), dtype=np.uint8))
Image.fromarray(img).save(SP / "audit_slice.png")
print("slice z", z, "crop", crop, "levels in crop", np.bincount(lvl[crop].ravel(), minlength=5)[:5] / lvl[crop].size)
