"""Detail-aware evaluation of a trained octree (the GT volume is read for evaluation only, never for training).

One pass over the volume in slabs of 16 z-slices (halo R = 8) computes, from the deployed model output/<run>/checkpoint.pt:
  volume      PSNR, RMSE in the surface band (within 3 voxels of the GT boundary at TAU) and far from it
  detail      D(x) = x - Gauss_2(x): on voxels with |D(GT)| > 0.03 the RMSE of D(pred) - D(GT) and the retained share
              <D(pred), D(GT)> / <D(GT), D(GT)> (1 = all fine detail kept)
  surface     boundaries of the masks Gauss_1(GT) >= TAU_S and Gauss_1(pred) >= TAU_S (TAU_S = 0.2, midway between the
              low-density medium ~0.07 and the material ~0.33: at TAU = 0.1 the GT boundary is mostly grain); distance (voxels, L-infinity, capped at R) from each GT
              boundary voxel to the predicted boundary and back: mean, p95, p99 and the share beyond 2 voxels
  thin        the same GT -> pred distance on boundary voxels next to thin structures (cracks / walls narrower than
              about 5 voxels: closing(mask) - mask and mask - opening(mask), radius 2)
  specks      inside the material (the TAU_S mask eroded by 3) the share of voxels with |D(pred)| > 0.03 where |D(GT)| < 0.015:
              detail the reconstruction invents in quiet parts of the object
  seams       in the same quiet interior, mean |pred(v + e) - pred(v)| over neighbouring voxels in different leaves
              divided by that within one leaf (1 = leaf boundaries invisible), over the three axes
  step seams  anywhere, mean |pred(v + e) - pred(v)| over neighbouring voxels in two different step leaves (planes that
              do not meet show here)
Writes output/<run>/detail_eval.json and output/<run>/zoom.png (the zoomed panel of slice_vs_oracle_tree.png).
--deblock: the same metrics after the decoder-side deblocking (adaptive_ct/evaluation/deblock.py, delta of the run's
edge penalty) -> detail_eval_deblock.json, and zoom_deblock.png: GT | decoded | deblocked on the crop of the slice
figure (median z, rows 312:568, columns 432:688, x4).
usage: python scripts/audit/detail_eval.py [--deblock] <run> [<run> ...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adaptive_ct.evaluation.deblock import deblock
from adaptive_ct.representation.factory import build_model

dev = torch.device("cuda")
GT = np.load(ROOT / "output/figure473_dicom_orig_r2_convergence/vol_gt.npy", mmap_mode="r")
D, H, W = GT.shape
TAU, TAU_S, BAND_R, R, SLAB = 0.1, 0.2, 3, 8, 16
KR = 6


def _kernel(sigma):
    k = torch.exp(-0.5 * (torch.arange(-KR, KR + 1, device=dev, dtype=torch.float32) / sigma) ** 2)
    return k / k.sum()


K2, K1 = _kernel(2.0), _kernel(1.0)
YS, XS = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")


def points(z0, z1):
    zz = torch.arange(z0, z1, device=dev)[:, None, None].expand(-1, H, W)
    return torch.stack([-1 + (zz.float() + 0.5) * 2 / D, -1 + (YS[None].expand(z1 - z0, -1, -1).float() + 0.5) * 2 / H,
                        -1 + (XS[None].expand(z1 - z0, -1, -1).float() + 0.5) * 2 / W], -1).reshape(-1, 3)


def blur(v, k=K2):
    x = v[None, None]
    for dim in range(3):
        shape, pad = [1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]
        shape[2 + dim] = 2 * KR + 1
        pad[2 * (2 - dim)] = pad[2 * (2 - dim) + 1] = KR
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k.reshape(shape))
    return x[0, 0]


def dilate(m, r):
    return F.max_pool3d(m.float()[None, None], 2 * r + 1, stride=1, padding=r)[0, 0] > 0


def erode(m, r):
    return ~dilate(~m, r)


def boundary(m):
    return dilate(m, 1) & ~erode(m, 1)


def distance_to(target):
    """L-infinity distance (voxels) to the nearest True of `target`, R + 1 beyond R."""
    dist = torch.full(target.shape, R + 1, dtype=torch.uint8, device=dev)
    cur = target
    for k in range(R + 1):
        dist[cur & (dist > k)] = k
        cur = dilate(cur, 1)
    return dist


def quantiles(hist):
    c = hist.cumsum(0) / hist.sum()
    q = lambda p: int(torch.searchsorted(c, torch.tensor(p, device=dev, dtype=c.dtype)))
    v = torch.arange(hist.numel(), device=dev, dtype=torch.float64)
    return {"mean": float((hist * v).sum() / hist.sum()), "p95": q(0.95), "p99": q(0.99), "share_gt2": float(hist[3:].sum() / hist.sum())}


@torch.no_grad()
def evaluate(run, with_deblock=False, decoder=None, tag=""):
    """decoder(model) -> points -> (values, leaf ids): another decoding of the same model (offline tests); tag: suffix
    of the output files."""
    ck = torch.load(ROOT / "output" / run / "checkpoint.pt", map_location="cpu", weights_only=False)
    m = build_model(ck["config"])
    m.prepare_sparse_from_state_dict(ck["model"])
    m.load_state_dict(ck["model"])
    m = m.to(dev).eval()
    leaf_size = 2.0 / torch.tensor([s[0] for s in m.level_shapes], dtype=torch.float32, device=dev)[m.leaf_levels]
    is_step = m.leaf_type() == 2
    delta = float(ck["config"]["training"]["solver"]["edge_penalty"]["delta"])
    crop = None
    acc = {k: 0.0 for k in ("sse", "band_se", "band_n", "far_se", "far_n", "d_se", "d_dot", "d_gg", "d_n", "spk", "quiet",
                            "seam_cross", "seam_cross_n", "seam_in", "seam_in_n", "step_seam", "step_seam_n")}
    decode = decoder(m) if decoder is not None else m.forward_mu_with_leaf_ids
    h_gp, h_pg, h_thin = (torch.zeros(R + 2, dtype=torch.float64, device=dev) for _ in range(3))
    for z0 in range(0, D, SLAB):
        z1 = min(D, z0 + SLAB)
        a, b = max(0, z0 - R), min(D, z1 + R)
        gt = torch.from_numpy(np.ascontiguousarray(GT[a:b])).to(dev).float()
        pts = points(a, b)
        pred, ids = zip(*[decode(pts[s:s + 4_000_000]) for s in range(0, pts.shape[0], 4_000_000)])
        pred, ids = torch.cat(pred).reshape(gt.shape), torch.cat(ids).reshape(gt.shape)
        if with_deblock:
            raw = pred
            pred = deblock(pred, ids, leaf_size, is_step, delta)
            if z0 <= D // 2 < z1:
                k = D // 2 - a
                crop = [x[k, 312:568, 432:688].cpu().numpy() for x in (gt, raw, pred)]
        core = slice(z0 - a, z1 - a)
        band = dilate(boundary(gt >= TAU), BAND_R)[core]
        mg, mp = blur(gt, K1) >= TAU_S, blur(pred, K1) >= TAU_S
        bg, bp = boundary(mg), boundary(mp)
        dg, dp = gt - blur(gt), pred - blur(pred)
        # volume
        e2 = ((pred - gt).double() ** 2)[core]
        acc["sse"] += float(e2.sum())
        acc["band_se"] += float(e2[band].sum()); acc["band_n"] += int(band.sum())
        acc["far_se"] += float(e2[~band].sum()); acc["far_n"] += int((~band).sum())
        # detail
        sel = dg[core].abs() > 0.03
        dgc, dpc = dg[core][sel].double(), dp[core][sel].double()
        acc["d_se"] += float(((dpc - dgc) ** 2).sum()); acc["d_dot"] += float((dpc * dgc).sum())
        acc["d_gg"] += float((dgc ** 2).sum()); acc["d_n"] += int(sel.sum())
        # surface distances (both directions) and thin structures
        to_p, to_g = distance_to(bp)[core], distance_to(bg)[core]
        bgc, bpc = bg[core], bp[core]
        h_gp += torch.bincount(to_p[bgc].long(), minlength=R + 2).double()
        h_pg += torch.bincount(to_g[bpc].long(), minlength=R + 2).double()
        thin = ((erode(dilate(mg, 2), 2) & ~mg) | (mg & ~dilate(erode(mg, 2), 2)))
        tb = (dilate(thin, 1) & bg)[core]
        h_thin += torch.bincount(to_p[tb].long(), minlength=R + 2).double()
        # specks and seams in the quiet interior
        quiet = erode(mg, 3) & (dg.abs() < 0.015)
        qc = quiet[core]
        acc["quiet"] += int(qc.sum())
        acc["spk"] += int((qc & (dp[core].abs() > 0.03)).sum())
        for dim in range(3):
            sl0 = [slice(None)] * 3; sl1 = [slice(None)] * 3
            sl0[dim], sl1[dim] = slice(0, -1), slice(1, None)
            if dim == 0:  # the z pairs of this slab's core (plus the one into the next slab)
                sl0[0], sl1[0] = slice(z0 - a, min(z1 - a, gt.shape[0] - 1)), slice(z0 - a + 1, min(z1 - a + 1, gt.shape[0]))
                q2 = quiet[tuple(sl0)] & quiet[tuple(sl1)]
            else:
                q2 = (quiet[tuple(sl0)] & quiet[tuple(sl1)])[core]
            dd = (pred[tuple(sl1)] - pred[tuple(sl0)]).abs()
            cross = ids[tuple(sl1)] != ids[tuple(sl0)]
            if dim != 0:
                dd, cross = dd[core], cross[core]
            acc["seam_cross"] += float(dd[q2 & cross].double().sum()); acc["seam_cross_n"] += int((q2 & cross).sum())
            acc["seam_in"] += float(dd[q2 & ~cross].double().sum()); acc["seam_in_n"] += int((q2 & ~cross).sum())
            if dim == 0:
                both = cross & is_step[ids[tuple(sl0)]] & is_step[ids[tuple(sl1)]]
            else:
                both = cross & (is_step[ids[tuple(sl0)]] & is_step[ids[tuple(sl1)]])[core]
            acc["step_seam"] += float(dd[both].double().sum()); acc["step_seam_n"] += int(both.sum())
    n = D * H * W
    out = {"run": run, "volume_psnr_db": 10 * np.log10(n / acc["sse"]), "band_rmse": (acc["band_se"] / acc["band_n"]) ** 0.5,
           "far_rmse": (acc["far_se"] / acc["far_n"]) ** 0.5, "detail_rmse": (acc["d_se"] / acc["d_n"]) ** 0.5,
           "detail_retained": acc["d_dot"] / acc["d_gg"], "surface_gt_to_pred": quantiles(h_gp), "surface_pred_to_gt": quantiles(h_pg),
           "thin_gt_to_pred": quantiles(h_thin), "speck_share": acc["spk"] / acc["quiet"],
           "seam_ratio": (acc["seam_cross"] / acc["seam_cross_n"]) / (acc["seam_in"] / acc["seam_in_n"]),
           "step_seam": acc["step_seam"] / max(acc["step_seam_n"], 1), "leaves": m.leaf_count}
    from PIL import Image
    if with_deblock:
        (ROOT / "output" / run / f"detail_eval_deblock{tag}.json").write_text(json.dumps(out, indent=1))
        img = lambda x: np.kron((np.clip(x / 0.8, 0, 1) * 255).astype(np.uint8), np.ones((4, 4), np.uint8))
        Image.fromarray(np.concatenate([img(x) for x in crop], 1)).save(ROOT / "output" / run / f"zoom_deblock{tag}.png")
        return out
    (ROOT / "output" / run / f"detail_eval{tag}.json").write_text(json.dumps(out, indent=1))
    fig = ROOT / "output" / run / "slice_vs_oracle_tree.png"
    if fig.exists():
        im = Image.open(fig)
        w, h = im.size
        im.crop((w // 4, h // 2, w // 2, h)).save(ROOT / "output" / run / "zoom.png")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    flag = "--deblock" in args
    for run in [a for a in args if a != "--deblock"]:
        print(json.dumps(evaluate(run, flag)), flush=True)
