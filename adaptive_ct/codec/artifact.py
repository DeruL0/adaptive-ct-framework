"""Compact file of the deployed octree: packed hierarchy (node arrays) + physical coefficients in fp16."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

SCHEMA = "compact_bernstein_octree_packed_v3"


@dataclass(frozen=True)
class CompactOctreeSummary:
    leaf_count: int
    coefficient_count: int
    quantization: str
    raw_payload_bytes: int
    file_bytes: int


def _all_coords_np(shape) -> np.ndarray:
    axes = [np.arange(value, dtype=np.int64) for value in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _children_np(parent_coords: np.ndarray) -> np.ndarray:
    offsets = _all_coords_np((2, 2, 2))
    return (parent_coords[:, None, :] * 2 + offsets[None, :, :]).reshape(-1, 3)


@torch.no_grad()
def export_compact_octree_artifact(model, path: str | Path, *, quantization: str = "float16") -> CompactOctreeSummary:
    quantization = str(quantization).lower()
    if quantization not in {"float16", "float32"}:
        raise ValueError("compact octree quantization must be 'float16' or 'float32'.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients = model.coefficients().detach().float().cpu().numpy()
    arrays = {
        "node_child_base": model.node_child_base.detach().cpu().numpy().astype(np.int32),
        "node_leaf_id": model.node_leaf_id.detach().cpu().numpy().astype(np.int32),
        "leaf_degrees": model.leaf_degrees.detach().cpu().numpy().astype(np.uint8),
        "leaf_count": np.asarray([int(model.leaf_levels.shape[0])], dtype=np.uint32),
        "level_shapes": np.asarray(model.level_shapes, dtype=np.uint16),
        "max_degree": np.asarray([model.max_degree], dtype=np.uint8),
        "attenuation_shift": np.asarray([float(model.attenuation_shift)], dtype=np.float32),
        "coefficients": coefficients.astype(np.float16 if quantization == "float16" else np.float32),
    }
    metadata = {"schema": SCHEMA, "quantization": quantization}
    arrays["metadata_json"] = np.asarray([json.dumps(metadata, sort_keys=True)], dtype="<U1024")
    np.savez_compressed(output_path, **arrays)
    return CompactOctreeSummary(
        leaf_count=int(model.leaf_levels.shape[0]),
        coefficient_count=int(coefficients.size),
        quantization=quantization,
        raw_payload_bytes=int(sum(value.nbytes for value in arrays.values())),
        file_bytes=int(output_path.stat().st_size),
    )


def load_compact_octree_state_dict(model, path: str | Path, *, device=None) -> tuple[dict[str, torch.Tensor], dict]:
    """Reconstruct a BernsteinOctree state_dict from a compact `.npz` file."""
    target_device = torch.device(device) if device is not None else model.coefficient_logits.device
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"][0]))
        if metadata.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported compact octree schema: {metadata.get('schema')!r}")
        shapes = [tuple(int(v) for v in row) for row in payload["level_shapes"].tolist()]
        if shapes != [tuple(s) for s in model.level_shapes]:
            raise ValueError(f"Compact octree levels {shapes} do not match model config {model.level_shapes}.")
        if int(payload["max_degree"][0]) != model.max_degree:
            raise ValueError("Compact octree max_degree does not match model config.")
        model.attenuation_shift = float(payload["attenuation_shift"][0])
        node_child_base = payload["node_child_base"].astype(np.int64)
        node_leaf_id = payload["node_leaf_id"].astype(np.int64)
        leaf_count = int(payload["leaf_count"][0])
        leaf_degrees = payload["leaf_degrees"].astype(np.int64)
        coefficients = payload["coefficients"].astype(np.float32)

    # Walk the packed hierarchy level by level to recover each leaf's level and coordinates.
    leaf_levels = np.full((leaf_count,), -1, dtype=np.int64)
    leaf_coords = np.full((leaf_count, 3), -1, dtype=np.int64)
    node_ids = np.arange(int(np.prod(shapes[0])), dtype=np.int64)
    coords = _all_coords_np(shapes[0])
    for level in range(len(shapes)):
        leaves = node_leaf_id[node_ids]
        is_leaf = leaves >= 0
        leaf_levels[leaves[is_leaf]] = level
        leaf_coords[leaves[is_leaf]] = coords[is_leaf]
        if is_leaf.all():
            break
        if level + 1 >= len(shapes):
            raise ValueError("Packed hierarchy continues beyond configured levels.")
        bases = node_child_base[node_ids[~is_leaf]]
        node_ids = (bases[:, None] + np.arange(8)[None, :]).reshape(-1)
        coords = _children_np(coords[~is_leaf])
    if np.any(leaf_levels < 0):
        raise ValueError("Packed hierarchy did not reconstruct every leaf.")
    counts = np.prod(leaf_degrees + 1, axis=1)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    if int(offsets[-1]) != coefficients.size:
        raise ValueError("Compact octree coefficient count does not match the leaf degrees.")
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device=target_device)
    state_dict = {
        "coefficient_logits": model._coefficients_to_logits(t(coefficients)).contiguous(),
        "leaf_levels": t(leaf_levels),
        "leaf_coords": t(leaf_coords),
        "leaf_degrees": t(leaf_degrees),
        "coefficient_offsets": t(offsets.astype(np.int64)),
        "coefficient_leaf_ids": t(np.repeat(np.arange(leaf_count), counts)),
    }
    return state_dict, metadata
