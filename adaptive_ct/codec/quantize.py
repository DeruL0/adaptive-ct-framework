"""Storage precision: values in fp16; step geometry packed into 32 bits per leaf:
  direction: octahedral map of the unit sphere onto [-1, 1]^2, 2 x 10 bits
  d:         8 bits over [-D_MAX, D_MAX] (|d| <= half the leaf's extent along n <= sqrt(3)/2)
  w:         4 bits, 0 = hard step, k = 1..15 -> w = 2^((k - 15) / 2)  (1/128 .. 1, steps of sqrt 2)
(d, w relative to a unit direction)."""

from __future__ import annotations

import torch

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


def quantize_roundtrip_(model) -> None:
    """Round the model in place to exactly what the stored file reproduces."""
    with torch.no_grad():
        if not bool(torch.isfinite(model.values.half()).all()):
            raise ValueError("stored values overflow fp16 (> 65504): the solve left unconstrained values")
        model.values.copy_(model.values.half().float())
        model.step_geometry[:, :5] = unpack_geometry(pack_geometry(model.step_geometry))
