"""Deterministic raw-byte model of the deployed codec."""

from __future__ import annotations

import torch

_STRUCTURE_BYTES_PER_SPLIT = 8 * 8  # eight new packed-hierarchy nodes, two int32 each


def _bytes_per_float(codec: dict) -> int:
    return 4 if str(codec.get("quantization", "float16")).lower() == "float32" else 2


def _per_leaf_bytes(codec: dict) -> int:
    kind = str(codec.get("type", "p0")).lower()
    if kind == "p0":
        values = 1
    elif kind == "p1":
        values = 8
    elif kind == "latent_p1":
        values = int(codec.get("latent_dim", 4))
    else:
        raise ValueError(f"Unknown codec type {kind!r}.")
    return values * _bytes_per_float(codec) + 3  # values + degree tag


def split_delta_bytes(codec: dict) -> int:
    """Raw packed bytes added by one octree split (parent leaf -> 8 children)."""
    return _STRUCTURE_BYTES_PER_SPLIT + 7 * _per_leaf_bytes(codec)


def codec_bytes(model, codec: dict) -> int:
    """Deterministic raw payload bytes of ``model`` stored with ``codec``."""
    kind = str(codec.get("type", "p0")).lower()
    degrees = model.leaf_degrees
    if kind == "p0" and bool(torch.any(degrees != 0)):
        raise ValueError("codec 'p0' requires every leaf to have degree 0.")
    if kind in {"p1", "latent_p1"} and bool(torch.any(degrees != 1)):
        raise ValueError(f"codec {kind!r} requires every leaf to be trilinear (degree 1).")
    leaf_count = int(model.leaf_levels.shape[0])
    nodes = int(model.node_child_base.numel())
    total = nodes * 8 + leaf_count * _per_leaf_bytes(codec)
    total += 4 + len(model.level_shapes) * 3 * 2 + 3 + 4
    if kind == "latent_p1":
        total += (8 * int(codec.get("latent_dim", 4)) + 8) * _bytes_per_float(codec)
    return int(total)
