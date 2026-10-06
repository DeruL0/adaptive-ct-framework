"""Bytes of what is stored: per leaf its physical coefficients (fp16) + a 3-byte degree tag, per split
eight packed hierarchy nodes (two int32 each), plus a fixed header."""

from __future__ import annotations

import torch

_STRUCTURE_BYTES_PER_SPLIT = 8 * 8
_VALUES = {"p0": 1, "p1": 8}


def _bytes_per_float(codec: dict) -> int:
    return 4 if str(codec.get("quantization", "float16")).lower() == "float32" else 2


def _per_leaf_bytes(codec: dict) -> int:
    return _VALUES[str(codec["type"]).lower()] * _bytes_per_float(codec) + 3


def split_delta_bytes(codec: dict) -> int:
    """Bytes added by one octree split (parent leaf -> 8 children of the same type)."""
    return _STRUCTURE_BYTES_PER_SPLIT + 7 * _per_leaf_bytes(codec)


def codec_bytes(model, codec: dict) -> int:
    """Payload bytes of ``model`` stored with ``codec``."""
    kind = str(codec["type"]).lower()
    if bool(torch.any(model.leaf_degrees != (0 if kind == "p0" else 1))):
        raise ValueError(f"codec {kind!r} requires every leaf to have degree {0 if kind == 'p0' else 1}.")
    leaf_count = int(model.leaf_levels.shape[0])
    nodes = int(model.node_child_base.numel())
    return int(nodes * 8 + leaf_count * _per_leaf_bytes(codec) + 4 + len(model.level_shapes) * 3 * 2 + 3 + 4)
