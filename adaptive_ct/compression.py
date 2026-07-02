from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .projection_domain import coefficient_diagnostics


@dataclass(frozen=True)
class MACTSummary:
    leaf_count: int
    material_clusters: int
    template_groups: int
    original_coefficient_bytes: int
    compressed_payload_bytes: int
    compression_ratio: float


@dataclass(frozen=True)
class CompactOctreeSummary:
    leaf_count: int
    coefficient_count: int
    quantization: str
    raw_payload_bytes: int
    file_bytes: int
    checkpoint_bytes: int | None
    model_bytes: int
    file_ratio_vs_model_bytes: float


def _kmeans_1d(values: torch.Tensor, cluster_count: int, iterations: int = 25) -> tuple[torch.Tensor, torch.Tensor]:
    cluster_count = max(1, min(int(cluster_count), int(values.numel())))
    if cluster_count == 1:
        return torch.zeros_like(values, dtype=torch.long), values.mean()[None]
    quantiles = torch.linspace(0.0, 1.0, cluster_count, device=values.device)
    centres = torch.quantile(values, quantiles).clone()
    assignments = torch.zeros_like(values, dtype=torch.long)
    for _ in range(int(iterations)):
        assignments = torch.argmin(torch.abs(values[:, None] - centres[None, :]), dim=1)
        updated = centres.clone()
        for cluster_id in range(cluster_count):
            selected = values[assignments == cluster_id]
            if selected.numel() > 0:
                updated[cluster_id] = selected.mean()
        if torch.allclose(updated, centres):
            break
        centres = updated
    return assignments, centres


@torch.no_grad()
def build_mact_artifact(
    model,
    *,
    material_clusters: int = 4,
    variance_retained: float = 0.99,
    max_rank: int | None = None,
) -> tuple[dict, MACTSummary]:
    """Build coefficient-only material/orientation templates and PCA weights."""
    diagnostics = coefficient_diagnostics(model)
    coefficients = model.coefficients().detach()
    dc = coefficients[model.coefficient_offsets[:-1]]
    materials, material_centres = _kmeans_1d(dc, material_clusters)
    orientations = diagnostics.principal_axis
    groups = []
    compressed_bytes = int(material_centres.numel() * material_centres.element_size())

    unique_degrees = torch.unique(model.leaf_degrees, dim=0)
    for degree_tensor in unique_degrees:
        degree = tuple(int(value) for value in degree_tensor.tolist())
        degree_mask = torch.all(model.leaf_degrees == degree_tensor, dim=1)
        coefficient_count = 1
        for value in degree:
            coefficient_count *= value + 1
        for material_id in range(int(material_centres.numel())):
            for orientation in range(3):
                mask = degree_mask & (materials == material_id) & (orientations == orientation)
                leaf_ids = torch.nonzero(mask, as_tuple=False).reshape(-1)
                if leaf_ids.numel() == 0:
                    continue
                local_ids = torch.arange(coefficient_count, device=coefficients.device)
                coefficient_ids = model.coefficient_offsets[leaf_ids, None] + local_ids[None, :]
                matrix = coefficients[coefficient_ids]
                mean = matrix.mean(dim=0)
                centred = matrix - mean
                if matrix.shape[0] <= 1 or coefficient_count <= 1:
                    basis = matrix.new_empty((0, coefficient_count))
                    weights = matrix.new_empty((matrix.shape[0], 0))
                else:
                    _, singular_values, vh = torch.linalg.svd(centred, full_matrices=False)
                    energy = torch.square(singular_values)
                    if float(energy.sum().item()) <= 0.0:
                        rank = 0
                    else:
                        cumulative = torch.cumsum(energy, dim=0) / energy.sum()
                        rank = int(torch.searchsorted(cumulative, torch.tensor(float(variance_retained), device=cumulative.device)).item()) + 1
                    if max_rank is not None:
                        rank = min(rank, int(max_rank))
                    rank = min(rank, int(vh.shape[0]))
                    basis = vh[:rank]
                    weights = centred @ basis.T
                group = {
                    "degree": degree,
                    "material_id": material_id,
                    "orientation": orientation,
                    "leaf_ids": leaf_ids.detach().cpu(),
                    "mean": mean.detach().cpu(),
                    "basis": basis.detach().cpu(),
                    "weights": weights.detach().cpu(),
                }
                groups.append(group)
                compressed_bytes += sum(
                    int(tensor.numel() * tensor.element_size())
                    for tensor in (group["leaf_ids"], group["mean"], group["basis"], group["weights"])
                )

    original_bytes = int(coefficients.numel() * coefficients.element_size())
    summary = MACTSummary(
        leaf_count=int(model.leaf_levels.shape[0]),
        material_clusters=int(material_centres.numel()),
        template_groups=len(groups),
        original_coefficient_bytes=original_bytes,
        compressed_payload_bytes=compressed_bytes,
        compression_ratio=float(original_bytes / max(compressed_bytes, 1)),
    )
    artifact = {
        "representation": "mact_bernstein_templates",
        "material_centres": material_centres.detach().cpu(),
        "material_assignments": materials.detach().cpu(),
        "orientation_assignments": orientations.detach().cpu(),
        "groups": groups,
        "summary": summary.__dict__,
    }
    return artifact, summary


def export_mact_artifact(model, path: str | Path, **kwargs) -> MACTSummary:
    artifact, summary = build_mact_artifact(model, **kwargs)
    torch.save(artifact, Path(path))
    return summary


def _shape3_np(value) -> tuple[int, int, int]:
    if np.isscalar(value):
        return (int(value),) * 3
    shape = tuple(int(component) for component in value)
    if len(shape) != 3:
        raise ValueError(f"Expected a scalar or 3-value shape, got {value!r}.")
    return shape


def _model_level_shapes(model) -> list[tuple[int, int, int]]:
    if hasattr(model, "level_shapes"):
        return [_shape3_np(value) for value in model.level_shapes]
    return [_shape3_np(value) for value in model.level_resolutions]


def _payload_level_shapes(payload: np.lib.npyio.NpzFile) -> list[tuple[int, int, int]]:
    if "level_shapes" in payload:
        return [_shape3_np(value) for value in payload["level_shapes"].tolist()]
    return [_shape3_np(value) for value in payload["level_resolutions"].tolist()]


def _level_shape_arrays(model) -> dict[str, np.ndarray]:
    shapes = _model_level_shapes(model)
    arrays = {"level_shapes": np.asarray(shapes, dtype=np.uint16)}
    if all(nx == ny == nz for nx, ny, nz in shapes):
        arrays["level_resolutions"] = np.asarray([shape[0] for shape in shapes], dtype=np.uint16)
    return arrays


def _all_coords_np(resolution: int | tuple[int, int, int]) -> np.ndarray:
    shape = _shape3_np(resolution)
    axes = [np.arange(value, dtype=np.uint16) for value in shape]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def _children_np(parent_coords: np.ndarray) -> np.ndarray:
    offsets = _all_coords_np(2)
    if parent_coords.size == 0:
        return np.empty((0, 3), dtype=np.uint16)
    return (parent_coords[:, None, :].astype(np.uint32) * 2 + offsets[None, :, :].astype(np.uint32)).reshape(-1, 3).astype(
        np.uint16
    )


def _quantize_coefficients(coefficients: np.ndarray, quantization: str) -> tuple[dict[str, np.ndarray], dict[str, str | float]]:
    normalized_quantization = str(quantization).lower()
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, str | float] = {"quantization": normalized_quantization}
    if normalized_quantization == "float16":
        arrays["coefficients"] = coefficients.astype(np.float16)
    elif normalized_quantization in {"uint8", "uint16"}:
        finite = coefficients[np.isfinite(coefficients)]
        coeff_min = float(np.min(finite)) if finite.size else 0.0
        coeff_max = float(np.max(finite)) if finite.size else 1.0
        if coeff_max <= coeff_min:
            coeff_max = coeff_min + 1.0
        levels = 255.0 if normalized_quantization == "uint8" else 65535.0
        dtype = np.uint8 if normalized_quantization == "uint8" else np.uint16
        array_name = "coefficients_uint8" if normalized_quantization == "uint8" else "coefficients_uint16"
        scale = (coeff_max - coeff_min) / levels
        arrays[array_name] = np.rint(np.clip((coefficients - coeff_min) / scale, 0.0, levels)).astype(dtype)
        arrays["coefficient_min"] = np.asarray([coeff_min], dtype=np.float32)
        arrays["coefficient_scale"] = np.asarray([scale], dtype=np.float32)
        metadata["coefficient_min"] = coeff_min
        metadata["coefficient_scale"] = scale
    else:
        raise ValueError("compact octree quantization must be 'uint8', 'uint16', or 'float16'.")
    return arrays, metadata


def _decode_coefficients_from_payload(payload: np.lib.npyio.NpzFile) -> np.ndarray:
    if "coefficients" in payload:
        return payload["coefficients"].astype(np.float32)
    if "coefficients_uint16" in payload:
        coeff_min = float(payload["coefficient_min"][0])
        coeff_scale = float(payload["coefficient_scale"][0])
        return coeff_min + payload["coefficients_uint16"].astype(np.float32) * coeff_scale
    if "coefficients_uint8" in payload:
        coeff_min = float(payload["coefficient_min"][0])
        coeff_scale = float(payload["coefficient_scale"][0])
        return coeff_min + payload["coefficients_uint8"].astype(np.float32) * coeff_scale
    raise ValueError("Compact octree artifact is missing coefficients.")


def _build_split_mask_topology(model) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    """Build split-mask topology arrays and coefficients in canonical leaf order.

    This format is only lossless for the current h-only constant-leaf octree:
    every active leaf has degree 0, therefore each leaf owns one coefficient and
    offsets/owners are implicit. Mixed p-refinement falls back to the explicit
    v1 topology.
    """
    if not torch.all(model.leaf_degrees == 0):
        return None
    coefficients = model.coefficients().detach().float().cpu().numpy()
    if int(coefficients.size) != int(model.leaf_levels.shape[0]):
        return None
    levels = model.leaf_levels.detach().cpu().numpy().astype(np.int64)
    coords = model.leaf_coords.detach().cpu().numpy().astype(np.uint16)
    shapes = _model_level_shapes(model)
    if any(nx != ny or ny != nz for nx, ny, nz in shapes):
        return None
    resolutions = [shape[0] for shape in shapes]
    if len(resolutions) > 3:
        return None

    key_to_coeff = {
        (int(level), int(coord[0]), int(coord[1]), int(coord[2])): float(coeff)
        for level, coord, coeff in zip(levels.tolist(), coords.tolist(), coefficients.tolist())
    }

    root_coords = _all_coords_np(resolutions[0])
    active_l0 = np.zeros((resolutions[0], resolutions[0], resolutions[0]), dtype=bool)
    l0_coords = coords[levels == 0]
    if l0_coords.size:
        active_l0[l0_coords[:, 0], l0_coords[:, 1], l0_coords[:, 2]] = True
    split_l0 = ~active_l0
    split_l0_flat = split_l0.reshape(-1)
    split_root_coords = root_coords[split_l0_flat]

    l1_generated = _children_np(split_root_coords) if len(resolutions) >= 2 else np.empty((0, 3), dtype=np.uint16)
    if len(resolutions) < 2 and split_root_coords.size:
        return None
    active_l1 = np.zeros((resolutions[1], resolutions[1], resolutions[1]), dtype=bool) if len(resolutions) >= 2 else None
    if len(resolutions) >= 2:
        l1_coords = coords[levels == 1]
        if l1_coords.size:
            active_l1[l1_coords[:, 0], l1_coords[:, 1], l1_coords[:, 2]] = True
        split_l1 = ~active_l1[l1_generated[:, 0], l1_generated[:, 1], l1_generated[:, 2]]
    else:
        split_l1 = np.zeros((0,), dtype=bool)
    if len(resolutions) < 3 and np.any(split_l1):
        return None
    l2_generated = _children_np(l1_generated[split_l1]) if len(resolutions) >= 3 else np.empty((0, 3), dtype=np.uint16)

    canonical_keys: list[tuple[int, int, int, int]] = []
    canonical_keys.extend((0, int(x), int(y), int(z)) for x, y, z in root_coords[~split_l0_flat].tolist())
    canonical_keys.extend((1, int(x), int(y), int(z)) for x, y, z in l1_generated[~split_l1].tolist())
    canonical_keys.extend((2, int(x), int(y), int(z)) for x, y, z in l2_generated.tolist())
    if len(canonical_keys) != int(model.leaf_levels.shape[0]):
        return None
    try:
        ordered_coefficients = np.asarray([key_to_coeff[key] for key in canonical_keys], dtype=np.float32)
    except KeyError:
        return None

    arrays = {
        "split_l0_bits": np.packbits(split_l0_flat.astype(np.uint8)),
        "split_l1_bits": np.packbits(split_l1.astype(np.uint8)),
        "split_l0_size": np.asarray([split_l0_flat.size], dtype=np.uint32),
        "split_l1_size": np.asarray([split_l1.size], dtype=np.uint32),
        "leaf_count": np.asarray([len(canonical_keys)], dtype=np.uint32),
    }
    return arrays, ordered_coefficients


def _reconstruct_split_mask_topology(payload: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    resolutions = [int(value) for value in payload["level_resolutions"].tolist()]
    split_l0_size = int(payload["split_l0_size"][0])
    split_l0 = np.unpackbits(payload["split_l0_bits"], count=split_l0_size).astype(bool)
    root_coords = _all_coords_np(resolutions[0])
    split_root_coords = root_coords[split_l0]
    l1_generated = _children_np(split_root_coords) if len(resolutions) >= 2 else np.empty((0, 3), dtype=np.uint16)
    split_l1_size = int(payload["split_l1_size"][0])
    split_l1 = np.unpackbits(payload["split_l1_bits"], count=split_l1_size).astype(bool)
    if split_l1.shape[0] != l1_generated.shape[0]:
        raise ValueError(
            "Split-mask compact topology is inconsistent: "
            f"split_l1 has {split_l1.shape[0]} entries, expected {l1_generated.shape[0]}."
        )
    if np.any(split_l1) and len(resolutions) < 3:
        raise ValueError("Split-mask compact topology contains level-2 leaves but config has fewer than 3 levels.")
    l2_generated = _children_np(l1_generated[split_l1]) if len(resolutions) >= 3 else np.empty((0, 3), dtype=np.uint16)

    leaf_levels = np.concatenate(
        [
            np.zeros(int(np.sum(~split_l0)), dtype=np.int64),
            np.ones(int(np.sum(~split_l1)), dtype=np.int64),
            np.full(int(l2_generated.shape[0]), 2, dtype=np.int64),
        ]
    )
    leaf_coords = np.concatenate(
        [root_coords[~split_l0], l1_generated[~split_l1], l2_generated],
        axis=0,
    ).astype(np.int64)
    expected_leaf_count = int(payload["leaf_count"][0]) if "leaf_count" in payload else leaf_levels.shape[0]
    if int(leaf_levels.shape[0]) != expected_leaf_count:
        raise ValueError(
            "Split-mask compact topology leaf count mismatch: "
            f"{int(leaf_levels.shape[0])} vs {expected_leaf_count}."
        )
    return leaf_levels, leaf_coords


def _build_packed_hierarchy_topology(model) -> dict[str, np.ndarray]:
    """Serialize the arbitrary-depth hierarchy without dense per-level maps."""
    return {
        "node_child_base": model.node_child_base.detach().cpu().numpy().astype(np.int32),
        "node_leaf_id": model.node_leaf_id.detach().cpu().numpy().astype(np.int32),
        "leaf_degrees": model.leaf_degrees.detach().cpu().numpy().astype(np.uint8),
        "leaf_count": np.asarray([int(model.leaf_levels.shape[0])], dtype=np.uint32),
    }


def _reconstruct_packed_hierarchy_topology(
    payload: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shapes = _payload_level_shapes(payload)
    node_child_base = payload["node_child_base"].astype(np.int64)
    node_leaf_id = payload["node_leaf_id"].astype(np.int64)
    if node_child_base.ndim != 1 or node_child_base.shape != node_leaf_id.shape:
        raise ValueError("Packed hierarchy node arrays must be equal-length vectors.")
    leaf_count = int(payload["leaf_count"][0])
    leaf_levels = np.full((leaf_count,), -1, dtype=np.int64)
    leaf_coords = np.full((leaf_count, 3), -1, dtype=np.int64)
    node_ids = np.arange(int(np.prod(shapes[0], dtype=np.int64)), dtype=np.int64)
    coords = _all_coords_np(shapes[0]).astype(np.int64)
    for level in range(len(shapes)):
        if node_ids.size == 0:
            break
        if int(node_ids.max(initial=-1)) >= node_leaf_id.size:
            raise ValueError("Packed hierarchy contains an out-of-range node id.")
        leaves = node_leaf_id[node_ids]
        leaf_mask = leaves >= 0
        if np.any(leaves[leaf_mask] >= leaf_count):
            raise ValueError("Packed hierarchy contains an out-of-range leaf id.")
        leaf_levels[leaves[leaf_mask]] = level
        leaf_coords[leaves[leaf_mask]] = coords[leaf_mask]
        internal = ~leaf_mask
        if not np.any(internal):
            node_ids = np.empty((0,), dtype=np.int64)
            break
        if level + 1 >= len(shapes):
            raise ValueError("Packed hierarchy continues beyond configured levels.")
        bases = node_child_base[node_ids[internal]]
        if np.any(bases < 0):
            raise ValueError("Packed hierarchy has an unassigned internal node.")
        offsets = np.arange(8, dtype=np.int64)
        node_ids = (bases[:, None] + offsets[None, :]).reshape(-1)
        coords = _children_np(coords[internal].astype(np.uint16)).astype(np.int64)
    if np.any(leaf_levels < 0) or np.any(leaf_coords < 0):
        raise ValueError("Packed hierarchy did not reconstruct every leaf.")
    leaf_degrees = payload["leaf_degrees"].astype(np.int64)
    if leaf_degrees.shape != (leaf_count, 3):
        raise ValueError("Packed hierarchy leaf_degrees has an invalid shape.")
    counts = np.prod(leaf_degrees + 1, axis=1, dtype=np.int64)
    coefficient_offsets = np.concatenate([np.zeros((1,), dtype=np.int64), np.cumsum(counts, dtype=np.int64)])
    return leaf_levels, leaf_coords, leaf_degrees, coefficient_offsets


@torch.no_grad()
def export_compact_octree_artifact(
    model,
    path: str | Path,
    *,
    quantization: str = "uint16",
    topology: str = "explicit",
    checkpoint_path: str | Path | None = None,
) -> CompactOctreeSummary:
    """Write a storage-oriented octree artifact.

    Training checkpoints keep topology tensors as int64 and include redundant
    offsets/owners for autograd-friendly packing.  The inference artifact stores
    the same active tree in narrow types and quantizes physical attenuation
    coefficients.  This is storage compression only; it does not blur or decode
    through a dense volume.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coefficients = model.coefficients().detach().float().cpu().numpy()
    normalized_quantization = str(quantization).lower()
    topology_mode = str(topology).lower()
    if topology_mode in {"packed_hierarchy", "packed", "hierarchy", "auto"}:
        arrays: dict[str, np.ndarray] = {
            **_build_packed_hierarchy_topology(model),
            **_level_shape_arrays(model),
            "max_degree": np.asarray(model.max_degree, dtype=np.uint8),
            "attenuation_shift": np.asarray([float(model.attenuation_shift)], dtype=np.float32),
        }
        metadata: dict[str, str | float | int] = {
            "schema": "compact_bernstein_octree_packed_v3",
            "representation": getattr(model, "representation", "bernstein_octree"),
            "topology": "packed_hierarchy",
        }
        coefficient_arrays, coefficient_metadata = _quantize_coefficients(coefficients, normalized_quantization)
        arrays.update(coefficient_arrays)
        metadata.update(coefficient_metadata)
        arrays["metadata_json"] = np.asarray([json.dumps(metadata, sort_keys=True)], dtype="<U1024")
        np.savez_compressed(output_path, **arrays)
        raw_payload_bytes = int(sum(value.nbytes for value in arrays.values() if hasattr(value, "nbytes")))
        file_bytes = int(output_path.stat().st_size)
        stats = model.stats()
        checkpoint_bytes = (
            int(Path(checkpoint_path).stat().st_size)
            if checkpoint_path is not None and Path(checkpoint_path).exists()
            else None
        )
        return CompactOctreeSummary(
            leaf_count=int(model.leaf_levels.shape[0]),
            coefficient_count=int(coefficients.size),
            quantization=normalized_quantization,
            raw_payload_bytes=raw_payload_bytes,
            file_bytes=file_bytes,
            checkpoint_bytes=checkpoint_bytes,
            model_bytes=int(stats.model_bytes),
            file_ratio_vs_model_bytes=float(file_bytes / max(int(stats.model_bytes), 1)),
        )
    if topology_mode in {"split_masks", "split-mask", "mask", "auto"}:
        split_mask_payload = _build_split_mask_topology(model)
        if split_mask_payload is not None:
            topology_arrays, coefficients = split_mask_payload
            arrays: dict[str, np.ndarray] = {
                **topology_arrays,
                **_level_shape_arrays(model),
                "max_degree": np.asarray(model.max_degree, dtype=np.uint8),
                "attenuation_shift": np.asarray([float(model.attenuation_shift)], dtype=np.float32),
            }
            metadata: dict[str, str | float | int] = {
                "schema": "compact_bernstein_octree_splitmask_v2",
                "representation": getattr(model, "representation", "bernstein_octree"),
                "topology": "split_masks",
                "degree": 0,
            }
            coefficient_arrays, coefficient_metadata = _quantize_coefficients(coefficients, normalized_quantization)
            arrays.update(coefficient_arrays)
            metadata.update(coefficient_metadata)
            arrays["metadata_json"] = np.asarray([json.dumps(metadata, sort_keys=True)], dtype="<U1024")
            np.savez_compressed(output_path, **arrays)

            raw_payload_bytes = int(sum(value.nbytes for value in arrays.values() if hasattr(value, "nbytes")))
            file_bytes = int(output_path.stat().st_size)
            stats = model.stats()
            checkpoint_bytes = (
                int(Path(checkpoint_path).stat().st_size)
                if checkpoint_path is not None and Path(checkpoint_path).exists()
                else None
            )
            return CompactOctreeSummary(
                leaf_count=int(model.leaf_levels.shape[0]),
                coefficient_count=int(coefficients.size),
                quantization=normalized_quantization,
                raw_payload_bytes=raw_payload_bytes,
                file_bytes=file_bytes,
                checkpoint_bytes=checkpoint_bytes,
                model_bytes=int(stats.model_bytes),
                file_ratio_vs_model_bytes=float(file_bytes / max(int(stats.model_bytes), 1)),
            )
        if topology_mode != "auto":
            raise ValueError(
                "split_masks compact topology requires all Bernstein leaves to have degree 0 "
                "and one coefficient per leaf."
            )

    if any(nx != ny or ny != nz for nx, ny, nz in _model_level_shapes(model)):
        raise ValueError(
            "Anisotropic levels require topology='packed_hierarchy'; "
            "the legacy explicit v1 schema is cubic-only."
        )
    arrays = {
        "leaf_levels": model.leaf_levels.detach().cpu().numpy().astype(np.uint8),
        "leaf_coords": model.leaf_coords.detach().cpu().numpy().astype(np.uint16),
        "leaf_degrees": model.leaf_degrees.detach().cpu().numpy().astype(np.uint8),
        "coefficient_offsets": model.coefficient_offsets.detach().cpu().numpy().astype(np.uint32),
        **_level_shape_arrays(model),
        "max_degree": np.asarray(model.max_degree, dtype=np.uint8),
        "attenuation_shift": np.asarray([float(model.attenuation_shift)], dtype=np.float32),
    }
    metadata: dict[str, str | float | int] = {
        "schema": "compact_bernstein_octree_v1",
        "representation": getattr(model, "representation", "bernstein_octree"),
        "quantization": normalized_quantization,
    }

    coefficient_arrays, coefficient_metadata = _quantize_coefficients(coefficients, normalized_quantization)
    arrays.update(coefficient_arrays)
    metadata.update(coefficient_metadata)

    arrays["metadata_json"] = np.asarray([json.dumps(metadata, sort_keys=True)], dtype="<U1024")
    np.savez_compressed(output_path, **arrays)

    raw_payload_bytes = int(sum(value.nbytes for value in arrays.values() if hasattr(value, "nbytes")))
    file_bytes = int(output_path.stat().st_size)
    stats = model.stats()
    checkpoint_bytes = int(Path(checkpoint_path).stat().st_size) if checkpoint_path is not None and Path(checkpoint_path).exists() else None
    return CompactOctreeSummary(
        leaf_count=int(model.leaf_levels.shape[0]),
        coefficient_count=int(coefficients.size),
        quantization=normalized_quantization,
        raw_payload_bytes=raw_payload_bytes,
        file_bytes=file_bytes,
        checkpoint_bytes=checkpoint_bytes,
        model_bytes=int(stats.model_bytes),
        file_ratio_vs_model_bytes=float(file_bytes / max(int(stats.model_bytes), 1)),
    )


def load_compact_octree_state_dict(
    model,
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Reconstruct a BernsteinOctree state_dict from a compact `.npz` artifact."""
    artifact_path = Path(path)
    target_device = torch.device(device) if device is not None else model.coefficient_logits.device
    with np.load(artifact_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"][0])) if "metadata_json" in payload else {}
        schema = metadata.get("schema")
        if schema not in {
            "compact_bernstein_octree_v1",
            "compact_bernstein_octree_splitmask_v2",
            "compact_bernstein_octree_packed_v3",
        }:
            raise ValueError(f"Unsupported compact octree schema: {metadata.get('schema')!r}")

        level_shapes = _payload_level_shapes(payload)
        model_level_shapes = _model_level_shapes(model)
        if model_level_shapes != level_shapes:
            raise ValueError(
                f"Compact octree levels {level_shapes} do not match model config {model_level_shapes}."
            )
        max_degree = tuple(int(value) for value in payload["max_degree"].tolist())
        if tuple(model.max_degree) != max_degree:
            raise ValueError(f"Compact octree max_degree {max_degree} does not match model config {model.max_degree}.")
        if "attenuation_shift" in payload:
            model.attenuation_shift = float(payload["attenuation_shift"][0])

        if schema == "compact_bernstein_octree_packed_v3":
            leaf_levels_np, leaf_coords_np, leaf_degrees_np, coefficient_offsets_np = (
                _reconstruct_packed_hierarchy_topology(payload)
            )
            leaf_levels = torch.from_numpy(leaf_levels_np).to(device=target_device)
            leaf_coords = torch.from_numpy(leaf_coords_np).to(device=target_device)
            leaf_degrees = torch.from_numpy(leaf_degrees_np).to(device=target_device)
            coefficient_offsets = torch.from_numpy(coefficient_offsets_np).to(device=target_device)
            coefficients_np = _decode_coefficients_from_payload(payload)
        elif schema == "compact_bernstein_octree_splitmask_v2":
            leaf_levels_np, leaf_coords_np = _reconstruct_split_mask_topology(payload)
            coefficients_np = _decode_coefficients_from_payload(payload)
            if int(coefficients_np.size) != int(leaf_levels_np.shape[0]):
                raise ValueError(
                    "Split-mask compact coefficients must contain exactly one coefficient per leaf: "
                    f"{int(coefficients_np.size)} vs {int(leaf_levels_np.shape[0])}."
                )
            leaf_levels = torch.from_numpy(leaf_levels_np).to(device=target_device)
            leaf_coords = torch.from_numpy(leaf_coords_np).to(device=target_device)
            leaf_degrees = torch.zeros((leaf_levels_np.shape[0], 3), dtype=torch.long, device=target_device)
            coefficient_offsets = torch.arange(leaf_levels_np.shape[0] + 1, dtype=torch.long, device=target_device)
        else:
            leaf_levels = torch.from_numpy(payload["leaf_levels"].astype(np.int64)).to(device=target_device)
            leaf_coords = torch.from_numpy(payload["leaf_coords"].astype(np.int64)).to(device=target_device)
            leaf_degrees = torch.from_numpy(payload["leaf_degrees"].astype(np.int64)).to(device=target_device)
            coefficient_offsets = torch.from_numpy(payload["coefficient_offsets"].astype(np.int64)).to(device=target_device)
            coefficients_np = _decode_coefficients_from_payload(payload)
        coefficients = torch.from_numpy(coefficients_np.astype(np.float32)).to(device=target_device)

    if int(coefficient_offsets[0].item()) != 0:
        raise ValueError("Compact octree coefficient_offsets must start at zero.")
    if int(coefficient_offsets[-1].item()) != int(coefficients.numel()):
        raise ValueError(
            "Compact octree coefficient_offsets[-1] does not match coefficient count: "
            f"{int(coefficient_offsets[-1].item())} vs {int(coefficients.numel())}."
        )
    counts = coefficient_offsets[1:] - coefficient_offsets[:-1]
    if torch.any(counts <= 0):
        raise ValueError("Compact octree contains a leaf with no coefficients.")
    coefficient_leaf_ids = torch.repeat_interleave(
        torch.arange(leaf_levels.shape[0], dtype=torch.long, device=target_device),
        counts,
    )
    state_dict = {
        "coefficient_logits": model._coefficients_to_logits(coefficients).contiguous(),
        "leaf_levels": leaf_levels.contiguous(),
        "leaf_coords": leaf_coords.contiguous(),
        "leaf_degrees": leaf_degrees.contiguous(),
        "coefficient_offsets": coefficient_offsets.contiguous(),
        "coefficient_leaf_ids": coefficient_leaf_ids.contiguous(),
    }
    return state_dict, metadata
