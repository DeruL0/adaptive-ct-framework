"""Bytes of the stored format (codec/artifact.py):
  header 64 B · one bit per root cell (leaf or split) · one 8-bit child mask per internal node ·
  a 2-bit type per leaf · values in fp16: constant 2 B, trilinear 16 B, step 4 B + 4 B packed geometry.
So a leaf costs 2.25 / 16.25 / 8.25 B (constant / trilinear / step) and a split adds 1 B + 7 children."""

from __future__ import annotations

import math

import torch

HEADER_BYTES = 64
PAYLOAD_BYTES = (2, 16, 8)  # by type id


def leaf_bytes() -> torch.Tensor:
    """Bytes per leaf (payload + 2-bit type) for type ids 0 constant, 1 trilinear, 2 step."""
    return torch.tensor(PAYLOAD_BYTES, dtype=torch.float64) + 0.25


def split_bytes(types: torch.Tensor) -> torch.Tensor:
    """Bytes added by splitting leaves of the given types (parent -> 8 children of the same type)."""
    return 1.0 + 7.0 * leaf_bytes().to(types.device)[types]


def model_bytes(model) -> int:
    root = math.prod(model.l0_shape)
    return int(HEADER_BYTES + math.ceil(root / 8) + model.child_table.shape[0] + math.ceil(2 * model.leaf_count / 8)
               + PAYLOAD_BYTES[0] * model.n_const + PAYLOAD_BYTES[1] * model.n_tri + PAYLOAD_BYTES[2] * model.n_step)
