"""Bytes of the stored format (codec/artifact.py):
  header 64 B · one bit per root cell (leaf or split) · one 8-bit child mask per internal node · a 2-bit type per leaf ·
  step geometry packed into 4 B per step leaf · values on the grid k q (q = 2 tolerance, codec/quantize.py), per block
  (root cell) a 16-bit base code and a 4-bit width b, then every value of the block's leaves as b bits (k - base).
A leaf in a block of width b costs 0.25 + n b / 8 B (+ 4 B for a step; n = 1 / 8 / 2 values for constant / trilinear /
step); a split adds 1 B + 7 children of the parent's type in the same block (widths as the block has them now)."""

from __future__ import annotations

import math

import torch

from adaptive_ct.codec.quantize import block_codes
from adaptive_ct.representation.octree import VALUE_COUNT

HEADER_BYTES = 64
BLOCK_BITS = 20  # 16-bit base code + 4-bit width per root cell
GEOMETRY_BYTES = 4


def leaf_type_bytes(model, tolerance: float) -> torch.Tensor:
    """(N, 3) bytes each leaf would cost as a constant / trilinear / step leaf at its block's current value width."""
    _, _, width, block_of_leaf = block_codes(model, tolerance)
    b = width[block_of_leaf].double()[:, None]
    n = torch.tensor(VALUE_COUNT, dtype=torch.float64, device=b.device)[None]
    return 0.25 + n * b / 8 + torch.tensor([0.0, 0.0, GEOMETRY_BYTES], dtype=torch.float64, device=b.device)[None]


def model_bytes(model, tolerance: float) -> int:
    root = math.prod(model.l0_shape)
    _, _, width, block_of_leaf = block_codes(model, tolerance)
    value_bits = float((width[block_of_leaf].double() * torch.tensor(VALUE_COUNT, device=width.device)[model.leaf_type()]).sum())
    return int(HEADER_BYTES + math.ceil(root / 8) + model.child_table.shape[0] + math.ceil(2 * model.leaf_count / 8)
               + GEOMETRY_BYTES * model.n_step + math.ceil(BLOCK_BITS * root / 8) + math.ceil(value_bits / 8))
