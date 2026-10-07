"""Stored file of the deployed octree (the format codec/budget.py counts).

Breadth-first order: root cells in raster order, then the children of every internal node (slot 4 x + 2 y + z)
level by level.  Stored: one bit per root cell (1 = internal), one 8-bit mask per internal node (bit j = child j
is internal), a 2-bit type per leaf, and the leaves' values by type in fp16 (step geometry packed, codec/quantize.py).
No pointers or coordinates: the decoder recovers them from the order."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from adaptive_ct.codec.budget import model_bytes
from adaptive_ct.codec.quantize import pack_geometry, unpack_geometry
from adaptive_ct.representation.octree import GEOMETRY_WIDTH, VALUE_COUNT, pad_geometry

SCHEMA = "typed_octree_bfs_v5"


@dataclass(frozen=True)
class ArtifactSummary:
    leaf_count: int
    value_count: int
    stored_bytes: int
    file_bytes: int


def _pack2(t: np.ndarray) -> np.ndarray:
    t = np.concatenate([t, np.zeros((-len(t)) % 4, dtype=np.uint8)]).reshape(-1, 4)
    return (t[:, 0] | (t[:, 1] << 2) | (t[:, 2] << 4) | (t[:, 3] << 6)).astype(np.uint8)


def _unpack2(b: np.ndarray, n: int) -> np.ndarray:
    return np.stack([(b >> s) & 3 for s in (0, 2, 4, 6)], 1).reshape(-1)[:n]


@torch.no_grad()
def export_octree_artifact(model, path: str | Path) -> ArtifactSummary:
    entries = model.root.long()
    root_bits = (entries < 0).cpu().numpy()
    masks, order = [], []
    while entries.numel():
        order.append(entries[entries >= 0])
        rows = model.child_table[-1 - entries[entries < 0]].long()
        masks.append(((rows < 0).long() << torch.arange(8, device=rows.device)).sum(1))
        entries = rows.reshape(-1)
    order = torch.cat(order)
    types = model.leaf_type()[order]
    off = model.value_offsets()[order]
    v = model.values.detach()
    tri, step, const = order[types == 1], off[types == 2], off[types == 0]
    arrays = {
        "root_bits": np.packbits(root_bits),
        "masks": torch.cat(masks).to(torch.uint8).cpu().numpy(),
        "types": _pack2(types.to(torch.uint8).cpu().numpy()),
        "trilinear": v[(8 * tri)[:, None] + torch.arange(8, device=v.device)].half().cpu().numpy(),
        "step": torch.stack([v[step], v[step + 1]], 1).half().cpu().numpy(),
        "step_geometry": pack_geometry(model.step_geometry[order[types == 2] - model.n_tri]).to(torch.int64).cpu().numpy().astype(np.uint32),
        "constant": v[const].half().cpu().numpy(),
        "header": np.asarray([json.dumps({"schema": SCHEMA, "levels": [list(s) for s in model.level_shapes], "leaves": model.leaf_count})]),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return ArtifactSummary(model.leaf_count, int(model.values.numel()), model_bytes(model), path.stat().st_size)


@torch.no_grad()
def load_octree_artifact(model, path: str | Path) -> dict:
    """Fill `model` (same level shapes) from a stored file; returns the header."""
    with np.load(Path(path), allow_pickle=False) as f:
        header = json.loads(str(f["header"][0]))
        if header["schema"] != SCHEMA:
            raise ValueError(f"Unsupported octree file schema: {header['schema']!r}")
        if [tuple(s) for s in header["levels"]] != [tuple(s) for s in model.level_shapes]:
            raise ValueError(f"Octree file levels {header['levels']} do not match the model {model.level_shapes}.")
        a = {k: f[k] for k in f.files}
    shape = model.l0_shape
    coords = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    internal = np.unpackbits(a["root_bits"])[: coords.shape[0]].astype(bool)
    masks, used = a["masks"], 0
    slots = np.stack(np.meshgrid(np.arange(2), np.arange(2), np.arange(2), indexing="ij"), -1).reshape(-1, 3)
    levels, leaf_coords, level = [], [], 0
    while coords.shape[0]:
        levels.append(np.full((~internal).sum(), level))
        leaf_coords.append(coords[~internal])
        parents = coords[internal]
        bits = masks[used : used + parents.shape[0]]
        used += parents.shape[0]
        coords = (parents[:, None] * 2 + slots[None]).reshape(-1, 3)
        internal = ((bits[:, None] >> np.arange(8)) & 1).astype(bool).reshape(-1)
        level += 1
    levels, leaf_coords = np.concatenate(levels), np.concatenate(leaf_coords)
    types = _unpack2(a["types"], levels.shape[0]).astype(np.int64)
    counts = np.asarray(VALUE_COUNT)[types]
    starts = np.cumsum(counts) - counts
    values = np.zeros(int(counts.sum()), dtype=np.float32)
    for t, block in ((0, a["constant"][:, None]), (1, a["trilinear"]), (2, a["step"])):
        idx = starts[types == t][:, None] + np.arange(VALUE_COUNT[t])
        values[idx.reshape(-1)] = block.astype(np.float32).reshape(-1)
    geometry = torch.zeros((types.shape[0], GEOMETRY_WIDTH))
    geometry[torch.from_numpy(types == 2)] = pad_geometry(unpack_geometry(torch.from_numpy(a["step_geometry"].astype(np.int64))))
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x))
    model._assign(t(levels), t(leaf_coords), t(types), t(values), geometry)
    return header
