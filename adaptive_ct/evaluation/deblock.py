"""Decoder-side deblocking of a decoded voxel block (rendering and evaluation only; training and the stored bytes do not
change), after the weak deblocking filter of block video codecs (H.264 / HEVC) on the octree's leaf faces.

Along each axis in turn, a pair of neighbouring voxels p0 | q0 in different leaves is filtered when neither leaf is a
step (a step's plane is an edge the model placed) and the jump is below delta * h (h = the mean size of the two leaves
in box units, delta the edge penalty's gradient threshold: the same scale that separates seams from edges there):
    D = (9 (q0 - p0) - 3 (q1 - p1)) / 16, clipped to half the jump;  p0 += D,  q0 -= D
(p1, q1 the next voxels outward).  Larger jumps are left alone, so edges stay sharp.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def deblock(values: torch.Tensor, leaf_ids: torch.Tensor, leaf_size: torch.Tensor, is_step: torch.Tensor, delta: float) -> torch.Tensor:
    """values, leaf_ids: (Z, Y, X) decoded block and the leaf of every voxel; leaf_size (N,) box units; is_step (N,)."""
    out = values.clone()
    for dim in range(3):
        n = out.shape[dim]
        if n < 2:
            continue
        p0, q0 = out.narrow(dim, 0, n - 1), out.narrow(dim, 1, n - 1)
        p1 = torch.cat([out.narrow(dim, 0, 1), out.narrow(dim, 0, n - 2)], dim) if n > 2 else p0
        q1 = torch.cat([out.narrow(dim, 2, n - 2), out.narrow(dim, n - 1, 1)], dim) if n > 2 else q0
        a, b = leaf_ids.narrow(dim, 0, n - 1), leaf_ids.narrow(dim, 1, n - 1)
        jump = q0 - p0
        ok = (a != b) & ~is_step[a] & ~is_step[b] & (jump.abs() < float(delta) * 0.5 * (leaf_size[a] + leaf_size[b]))
        d = ((9 * jump - 3 * (q1 - p1)) / 16)
        d = torch.maximum(torch.minimum(d, 0.5 * jump.abs()), -0.5 * jump.abs()) * ok
        new = out.clone()
        new.narrow(dim, 0, n - 1).add_(d)
        new.narrow(dim, 1, n - 1).sub_(d)
        out = new
    return out
