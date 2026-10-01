import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Projects\7.CT Framework")
from adaptive_ct.representation.factory import build_model

ROOT = Path(r"D:\Projects\7.CT Framework")
GT = np.load(ROOT / "output/figure473_dicom_orig_r2_convergence/vol_gt.npy", mmap_mode="r")
D, H, W = GT.shape
dev = torch.device("cuda")
TAU = 0.1
SLAB = 8
EDGES = [1, 3, 7, 15]  # distance bins: <=1, <=3, <=7, <=15, >15 voxels
HALO = 16
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


def boundary(gt_ext):
    mask = (gt_ext >= TAU).float()
    pad = F.pad(mask[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    c = pad[1:-1, 1:-1, 1:-1]
    diff = torch.zeros_like(c, dtype=torch.bool)
    for s in ((2, 1, 1), (0, 1, 1), (1, 2, 1), (1, 0, 1), (1, 1, 2), (1, 1, 0)):
        diff |= pad[s[0] : s[0] + c.shape[0], s[1] : s[1] + c.shape[1], s[2] : s[2] + c.shape[2]] != c
    return diff


def dist_bin(diff):
    """bin index per voxel: 0..len(EDGES) from successive dilations of the boundary."""
    b = torch.full(diff.shape, len(EDGES), dtype=torch.int8, device=dev)
    for i in reversed(range(len(EDGES))):
        r = EDGES[i]
        dil = F.max_pool3d(diff.float()[None, None], 2 * r + 1, stride=1, padding=r)[0, 0] > 0
        b[dil] = i
    return b


def run_error(model, label):
    nb = len(EDGES) + 1
    se = np.zeros(nb); ae = np.zeros(nb); n = np.zeros(nb)
    for z0 in range(0, D, SLAB):
        z1 = min(D, z0 + SLAB)
        a, b = max(0, z0 - HALO - 1), min(D, z1 + HALO + 1)
        gt_ext = torch.from_numpy(np.ascontiguousarray(GT[a:b])).to(dev)
        bins = dist_bin(boundary(gt_ext))[z0 - a : z1 - a]
        gt = gt_ext[z0 - a : z1 - a]
        zz = torch.arange(z0, z1, device=dev)[:, None, None].expand(-1, H, W)
        pts = torch.stack([-1 + (zz.float() + 0.5) * 2 / D, -1 + (ys[None].expand(z1 - z0, -1, -1).float() + 0.5) * 2 / H, -1 + (xs[None].expand(z1 - z0, -1, -1).float() + 0.5) * 2 / W], dim=-1).reshape(-1, 3)
        with torch.no_grad():
            pred = torch.cat([model.forward_mu(pts[s : s + 2_000_000]) for s in range(0, pts.shape[0], 2_000_000)]).reshape(gt.shape)
        e = (pred - gt).double()
        for k in range(nb):
            m = bins == k
            c = int(m.sum())
            if c:
                n[k] += c; se[k] += float((e[m] ** 2).sum()); ae[k] += float(e[m].abs().sum())
    tot = se.sum()
    names = ["<=1", "2-3", "4-7", "8-15", ">15"]
    return {label: {names[k]: {"vox%": round(100 * n[k] / n.sum(), 2), "SSE%": round(100 * se[k] / tot, 2), "RMSE": round(float(np.sqrt(se[k] / max(n[k], 1))), 5)} for k in range(nb)}}


def surface_cover():
    """Leaves needed to cover the whole surface (boundary voxels) at each octree level, in bytes."""
    shape0 = (48, 48, 48)
    res = {}
    for L in range(0, 5):
        sh = 48 * 2 ** L
        ids = []
        for z0 in range(0, D, 16):
            z1 = min(D, z0 + 16)
            a, b = max(0, z0 - 1), min(D, z1 + 1)
            gt_ext = torch.from_numpy(np.ascontiguousarray(GT[a:b])).to(dev)
            diff = boundary(gt_ext)[z0 - a : z1 - a]
            idx = torch.nonzero(diff, as_tuple=False)
            cz = ((idx[:, 0] + z0).double() + 0.5) / D * sh
            cy = (idx[:, 1].double() + 0.5) / H * sh
            cx = (idx[:, 2].double() + 0.5) / W * sh
            lin = (cz.floor().long() * sh + cy.floor().long()) * sh + cx.floor().long()
            ids.append(torch.unique(lin).cpu())
        u = torch.unique(torch.cat(ids))
        res[L] = int(u.numel())
    return res


out = {}
for run, codec, label in [("stage7_dorfler_p1_256v", True, "p1_latent_deployed"), ("stage7_dorfler_p0_longft_256v", False, "p0_longft")]:
    m = load(run, codec)
    out.update(run_error(m, label))
    print(json.dumps({label: out[label]}), flush=True)
    del m
    torch.cuda.empty_cache()
cover = surface_cover()
vox_nm = {}
out["surface_cover_leaves"] = cover
print("surface cover leaves by level", cover, flush=True)
json.dump(out, open(Path(sys.argv[1]) / "audit_distance.json", "w"), indent=1)
