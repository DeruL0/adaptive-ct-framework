"""Storage precision: values on the grid k q, q = 2 tolerance (|error| <= tolerance), coded per block (root cell) as a
16-bit base code plus a b-bit offset per value (codec/budget.py, codec/artifact.py); step geometry packed into 32 bits:
  direction: octahedral map of the unit sphere onto [-1, 1]^2, 2 x 10 bits
  d:         8 bits over [-D_MAX, D_MAX] (|d| <= half the leaf's extent along n <= sqrt(3)/2)
  w:         4 bits, 0 = hard step, k = 1..15 -> w = 2^((k - 15) / 2)  (1/128 .. 1, steps of sqrt 2)
(d, w relative to a unit direction)."""

from __future__ import annotations

import torch

from adaptive_ct.representation.octree import VALUE_COUNT

D_MAX = 0.875


def _sign(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, 1.0, -1.0)


def pack_geometry(g: torch.Tensor) -> torch.Tensor:
    """(n, >= 5) n, d, w -> (n,) int64 codes < 2^32."""
    norm = g[:, :3].norm(dim=1).clamp_min(1e-30)
    n, d, w = g[:, :3] / norm[:, None], g[:, 3] / norm, g[:, 4] / norm
    o = n / n.abs().sum(1, keepdim=True)
    u, v = o[:, 0], o[:, 1]
    neg = o[:, 2] < 0
    u, v = torch.where(neg, (1 - v.abs()) * _sign(u), u), torch.where(neg, (1 - u.abs()) * _sign(v), v)
    qu = torch.round((u + 1) * 0.5 * 1023).long().clamp(0, 1023)
    qv = torch.round((v + 1) * 0.5 * 1023).long().clamp(0, 1023)
    qd = torch.round((d / D_MAX + 1) * 0.5 * 255).long().clamp(0, 255)
    qw = torch.where(w > 0, torch.round(2 * torch.log2(w.clamp_min(1e-30)) + 15).long().clamp(1, 15), 0)
    return qu | (qv << 10) | (qd << 20) | (qw << 28)


def unpack_geometry(code: torch.Tensor) -> torch.Tensor:
    """(n,) codes -> (n, 5) n (unit), d, w."""
    u = (code & 1023).float() / 1023 * 2 - 1
    v = ((code >> 10) & 1023).float() / 1023 * 2 - 1
    d = (((code >> 20) & 255).float() / 255 * 2 - 1) * D_MAX
    k = (code >> 28) & 15
    z = 1 - u.abs() - v.abs()
    neg = z < 0
    x, y = torch.where(neg, (1 - v.abs()) * _sign(u), u), torch.where(neg, (1 - u.abs()) * _sign(v), v)
    n = torch.stack([x, y, z], 1)
    n = n / n.norm(dim=1, keepdim=True)
    w = torch.where(k > 0, torch.pow(2.0, (k.float() - 15) / 2), 0.0)
    return torch.cat([n, d[:, None], w[:, None]], 1)


MAX_CODE = 65535  # 16-bit base codes


@torch.no_grad()
def block_codes(model, tolerance: float):
    """Codes k = round(v / q) of the stored values (model order), and per block (root cell, raster order) the base
    code (smallest k) and the width b (bits of the largest k - base); the block of every leaf."""
    q = 2.0 * float(tolerance)
    v = model.values.detach()
    k = torch.round(v / q).long()
    if bool((k < 0).any()) or bool((k > MAX_CODE).any()) or not bool(torch.isfinite(v).all()):
        raise ValueError(f"stored values outside [0, {MAX_CODE * q:g}]: the solve left unconstrained values")
    shape = model.l0_shape
    root = model.leaf_coords >> model.leaf_levels[:, None]
    block_of_leaf = (root[:, 0] * shape[1] + root[:, 1]) * shape[2] + root[:, 2]
    count = torch.tensor(VALUE_COUNT, device=v.device)[model.leaf_type()]
    block = torch.repeat_interleave(block_of_leaf, count)  # values are stored leaf after leaf
    nb = shape[0] * shape[1] * shape[2]
    base = torch.full((nb,), MAX_CODE, dtype=torch.long, device=v.device).scatter_reduce(0, block, k, "amin")
    top = torch.zeros(nb, dtype=torch.long, device=v.device).scatter_reduce(0, block, k, "amax")
    span = (top - base).clamp_min(0)
    width = torch.where(span > 0, torch.floor(torch.log2(span.double())).long() + 1, torch.zeros_like(span))
    return k, base, width, block_of_leaf


def quantize_roundtrip_(model, tolerance: float) -> None:
    """Round the model in place to exactly what the stored file reproduces."""
    with torch.no_grad():
        k, _, _, _ = block_codes(model, tolerance)
        model.values.copy_(k.float() * (2.0 * float(tolerance)))
        model.step_geometry[:, :5] = unpack_geometry(pack_geometry(model.step_geometry))
