"""Bytes of what is stored: per leaf its values (a step leaf also its geometry: two direction angles, d, w)
plus a 3-byte type tag; per split eight packed hierarchy nodes (two int32 each); plus a fixed header.
With fp16: constant 5 B, trilinear 19 B, step 15 B per leaf; a split adds 64 B + 7 children."""

from __future__ import annotations

import torch

from adaptive_ct.representation.octree import type_ids

_STRUCTURE_BYTES_PER_SPLIT = 8 * 8
_NUMBERS = (1, 8, 2 + 4)  # stored numbers per leaf: constant, trilinear, step


def _bytes_per_float(quantization: str) -> int:
    return 4 if str(quantization).lower() == "float32" else 2


def leaf_bytes(quantization: str = "float16") -> torch.Tensor:
    """Bytes per leaf for type ids 0 constant, 1 trilinear, 2 step."""
    return torch.tensor([n * _bytes_per_float(quantization) + 3 for n in _NUMBERS], dtype=torch.long)


def split_bytes(types: torch.Tensor, quantization: str = "float16") -> torch.Tensor:
    """Bytes added by splitting leaves of the given types (parent -> 8 children of the same type)."""
    return _STRUCTURE_BYTES_PER_SPLIT + 7 * leaf_bytes(quantization).to(types.device)[types]


def model_bytes(model, quantization: str = "float16") -> int:
    nodes = int(model.node_child_base.numel())
    leaves = int(leaf_bytes(quantization).to(model.leaf_degrees.device)[type_ids(model.leaf_degrees)].sum())
    return int(nodes * 8 + leaves + 4 + len(model.level_shapes) * 3 * 2 + 3 + 4)
