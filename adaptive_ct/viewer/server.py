from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import time
import webbrowser
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from adaptive_ct.backend import project_dense_parallel
from adaptive_ct.compression import load_compact_octree_state_dict
from adaptive_ct.config import load_config
from adaptive_ct.data import ProjectionSplit, load_r2_dataset
from adaptive_ct.metrics import mae, projection_metrics, psnr
from adaptive_ct.model import build_model
from adaptive_ct.render import render_split


def _resolve_default_checkpoint(config: dict[str, Any]) -> Path:
    output_dir = Path(config["output"]["dir"])
    return (output_dir / "checkpoint.pt").resolve()


def _resolve_workspace_path(workspace_root: Path, value: str | Path, *, suffixes: set[str]) -> Path:
    raw = Path(value).expanduser()
    path = raw.resolve() if raw.is_absolute() else (workspace_root / raw).resolve()
    try:
        path.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside viewer workspace: {workspace_root}") from exc
    if path.suffix.lower() not in suffixes:
        allowed = ", ".join(sorted(suffixes))
        raise ValueError(f"Unsupported file type for {path.name}; expected one of {allowed}.")
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    return path


def _discover_viewer_sources(workspace_root: Path) -> dict[str, Any]:
    def entries(paths: list[Path]) -> list[dict[str, Any]]:
        result = []
        for path in sorted(set(paths), key=lambda item: item.stat().st_mtime, reverse=True):
            stat = path.stat()
            result.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(workspace_root)),
                    "size_bytes": int(stat.st_size),
                    "modified_time": float(stat.st_mtime),
                }
            )
        return result

    config_root = workspace_root / "configs"
    output_root = workspace_root / "output"
    config_paths = list(config_root.rglob("*.yaml")) + list(config_root.rglob("*.yml")) if config_root.exists() else []
    checkpoint_paths: list[Path] = []
    if output_root.exists():
        checkpoint_paths.extend(output_root.rglob("checkpoint.pt"))
        checkpoint_paths.extend(output_root.rglob("compact_octree*.npz"))
    return {
        "workspace": str(workspace_root),
        "configs": entries(config_paths),
        "checkpoints": entries(checkpoint_paths),
    }


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_torch_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _tensor_stats(value: torch.Tensor | np.ndarray | None) -> dict[str, float | int | None]:
    if value is None:
        return {"count": 0, "min": None, "max": None, "mean": None, "p95": None}
    tensor = torch.as_tensor(value).detach().float().reshape(-1)
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return {"count": int(tensor.numel()), "min": None, "max": None, "mean": None, "p95": None}
    return {
        "count": int(tensor.numel()),
        "min": float(torch.min(finite).item()),
        "max": float(torch.max(finite).item()),
        "mean": float(torch.mean(finite).item()),
        "p95": float(torch.quantile(finite, 0.95).item()),
    }


def _histogram(values: torch.Tensor | np.ndarray | None, *, bins: int | None = None) -> list[int]:
    if values is None:
        return []
    tensor = torch.as_tensor(values).detach().long().reshape(-1)
    if tensor.numel() == 0:
        return []
    if bins is None:
        bins = int(torch.max(tensor).item()) + 1
    if bins <= 0:
        return []
    return [int(v) for v in torch.bincount(tensor.clamp_min(0), minlength=int(bins))[: int(bins)].tolist()]


def _summarize_mact_artifact(path: Path) -> dict[str, Any]:
    payload = _load_torch_if_exists(path)
    result: dict[str, Any] = {
        "exists": path.exists(),
        "path": path,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "error": None,
    }
    if payload is None:
        return result
    try:
        groups = list(payload.get("groups", []))
        top_groups = []
        for group in sorted(groups, key=lambda item: int(torch.as_tensor(item.get("leaf_ids", [])).numel()), reverse=True)[:8]:
            leaf_ids = torch.as_tensor(group.get("leaf_ids", []))
            basis = torch.as_tensor(group.get("basis", []))
            top_groups.append(
                {
                    "degree": tuple(int(v) for v in group.get("degree", ())),
                    "material_id": int(group.get("material_id", 0)),
                    "orientation": int(group.get("orientation", 0)),
                    "leaf_count": int(leaf_ids.numel()),
                    "rank": int(basis.shape[0]) if basis.ndim >= 1 else 0,
                }
            )
        material_centres = torch.as_tensor(payload.get("material_centres", []), dtype=torch.float32)
        result.update(
            {
                "representation": payload.get("representation"),
                "summary": payload.get("summary", {}),
                "material_centres": [float(v) for v in material_centres.tolist()],
                "material_histogram": _histogram(payload.get("material_assignments"), bins=int(material_centres.numel())),
                "orientation_histogram": _histogram(payload.get("orientation_assignments"), bins=3),
                "top_groups": top_groups,
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic path for user-provided artifacts.
        result["error"] = repr(exc)
    return result


def _summarize_surface_artifact(path: Path) -> dict[str, Any]:
    payload = _load_torch_if_exists(path)
    result: dict[str, Any] = {
        "exists": path.exists(),
        "path": path,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "error": None,
    }
    if payload is None:
        return result
    try:
        surface_points = payload.get("surface_points")
        coefficient_std = payload.get("coefficient_std")
        surface_std = payload.get("surface_std")
        result.update(
            {
                "threshold": payload.get("threshold"),
                "surface_point_count": int(torch.as_tensor(surface_points).shape[0]) if surface_points is not None else 0,
                "coefficient_std": _tensor_stats(coefficient_std),
                "surface_std": _tensor_stats(surface_std),
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic path for user-provided artifacts.
        result["error"] = repr(exc)
    return result


def _summarize_compact_octree_artifact(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.exists(),
        "path": path,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "error": None,
    }
    if not path.exists():
        return result
    try:
        with np.load(path, allow_pickle=False) as payload:
            metadata_json = str(payload["metadata_json"][0]) if "metadata_json" in payload else "{}"
            result.update(
                {
                    "metadata": json.loads(metadata_json),
                    "leaf_count": (
                        int(payload["leaf_levels"].shape[0])
                        if "leaf_levels" in payload
                        else int(payload["leaf_count"][0])
                        if "leaf_count" in payload
                        else 0
                    ),
                    "coefficient_count": (
                        int(payload["coefficients"].shape[0])
                        if "coefficients" in payload
                        else int(payload["coefficients_uint16"].shape[0])
                        if "coefficients_uint16" in payload
                        else int(payload["coefficients_uint8"].shape[0])
                        if "coefficients_uint8" in payload
                        else 0
                    ),
                    "arrays": {
                        name: {
                            "shape": list(payload[name].shape),
                            "dtype": str(payload[name].dtype),
                            "bytes": int(payload[name].nbytes),
                        }
                        for name in payload.files
                        if name != "metadata_json"
                    },
                }
            )
    except Exception as exc:  # pragma: no cover - diagnostic path for user-provided artifacts.
        result["error"] = repr(exc)
    return result


def _artifact_payload(output_dir: Path, *, checkpoint_path: Path | None, report_path: Path | None) -> dict[str, Any]:
    checkpoint_exists = checkpoint_path is not None and checkpoint_path.exists()
    compact_path = (
        checkpoint_path
        if checkpoint_exists and checkpoint_path is not None and checkpoint_path.suffix.lower() == ".npz"
        else output_dir / "compact_octree.npz"
    )
    return {
        "kind": "rd_cvf_artifacts",
        "output": output_dir,
        "checkpoint": {
            "exists": bool(checkpoint_exists),
            "path": checkpoint_path,
            "size_bytes": checkpoint_path.stat().st_size if checkpoint_exists and checkpoint_path is not None else 0,
        },
        "report": {
            "exists": bool(report_path is not None and report_path.exists()),
            "path": report_path,
            "size_bytes": report_path.stat().st_size if report_path is not None and report_path.exists() else 0,
        },
        "mact": _summarize_mact_artifact(output_dir / "mact.pt"),
        "compact_octree": _summarize_compact_octree_artifact(compact_path),
        "surface": _summarize_surface_artifact(output_dir / "surface_uncertainty.pt"),
    }


def _leaf_structure_payload(model) -> dict[str, Any]:
    raw_levels = getattr(model, "level_shapes", getattr(model, "level_resolutions", []))
    resolutions = [
        [int(v), int(v), int(v)] if isinstance(v, (int, np.integer)) else [int(c) for c in v]
        for v in raw_levels
    ]
    leaf_levels = getattr(model, "leaf_levels", None)
    if not resolutions or leaf_levels is None:
        return {"levels": [], "size_range": None}
    levels = []
    counts = []
    for level, resolution in enumerate(resolutions):
        count = int(torch.sum(leaf_levels == level).item())
        counts.append(count)
        voxel_size = [2.0 / float(value) for value in resolution]
        voxel_size_geometric_mean = float(np.prod(voxel_size) ** (1.0 / 3.0))
        levels.append(
            {
                "level": level,
                "resolution": resolution[0] if resolution[0] == resolution[1] == resolution[2] else resolution,
                "count": count,
                "voxel_size": voxel_size_geometric_mean,
                "voxel_size_xyz": voxel_size,
                "voxel_size_geometric_mean": voxel_size_geometric_mean,
            }
        )
    active_sizes = [entry["voxel_size_geometric_mean"] for entry in levels if entry["count"] > 0]
    size_range = None
    if active_sizes:
        size_range = {
            "min": float(min(active_sizes)),
            "max": float(max(active_sizes)),
        }
    return {
        "levels": levels,
        "size_range": size_range,
        "leaf_cells": int(sum(counts)),
    }


def _leaf_geometry_payload(model, *, max_leaves: int | None = None, min_mu: float = 0.0) -> dict[str, Any]:
    """Export active leaf voxels as renderable boxes (binary base64).

    Each leaf becomes an axis-aligned box centred at its cell with the side
    length of its level. This is the adaptive structure the dense decode hides:
    coarse leaves stay big, refined leaves shrink to L1/L2 size.
    """
    raw_levels = getattr(model, "level_shapes", getattr(model, "level_resolutions", []))
    resolutions = [
        [int(v), int(v), int(v)] if isinstance(v, (int, np.integer)) else [int(c) for c in v]
        for v in raw_levels
    ]
    leaf_levels = getattr(model, "leaf_levels", None)
    leaf_coords = getattr(model, "leaf_coords", None)
    if not resolutions or leaf_levels is None or leaf_coords is None:
        return {
            "kind": "leaves",
            "count": 0,
            "total": 0,
            "level_counts": [],
            "level_sizes": [],
            "level_sizes_xyz": [],
            "resolutions": [],
            "mu_range": {"min": 0.0, "max": 1.0},
            "size_range": None,
            "encoding": "float32_base64",
            "positions": "",
            "sizes": "",
            "sizes_xyz": "",
            "mu": "",
            "levels": "",
        }

    levels_np = leaf_levels.detach().cpu().numpy().astype(np.int64)
    coords_np = leaf_coords.detach().cpu().numpy().astype(np.float64)
    mu_np = model._leaf_mu().detach().cpu().numpy().astype(np.float32)
    res_np = np.asarray(resolutions, dtype=np.float64)

    level_counts = [int(np.count_nonzero(levels_np == level)) for level in range(len(resolutions))]
    level_sizes_xyz = [[2.0 / float(value) for value in shape] for shape in resolutions]
    level_sizes = [float(np.prod(size) ** (1.0 / 3.0)) for size in level_sizes_xyz]

    total = int(levels_np.shape[0])
    keep = mu_np >= float(min_mu)
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        idx = np.arange(total)
    if max_leaves is not None and int(max_leaves) > 0:
        limit = int(max_leaves)
        if idx.size > limit:
            order = np.argpartition(mu_np[idx], -limit)[-limit:]
            idx = idx[order]

    sel_levels = levels_np[idx]
    sel_coords = coords_np[idx]
    sel_mu = mu_np[idx]
    sel_res = res_np[sel_levels]
    sizes_xyz_ct = (2.0 / sel_res).astype(np.float32)
    sizes = (np.prod(sizes_xyz_ct, axis=1) ** (1.0 / 3.0)).astype(np.float32)
    # Cell centre in [-1, 1]^3 at the leaf's own resolution.
    centres = (-1.0 + (sel_coords + 0.5) * 2.0 / sel_res).astype(np.float32)
    # Three.js uses Y as the vertical axis; map CT z to vertical (matches _volume3d_payload).
    positions = np.stack([centres[:, 0], centres[:, 2], centres[:, 1]], axis=1).astype(np.float32)
    sizes_xyz = np.stack(
        [sizes_xyz_ct[:, 0], sizes_xyz_ct[:, 2], sizes_xyz_ct[:, 1]],
        axis=1,
    ).astype(np.float32)

    finite_mu = sel_mu[np.isfinite(sel_mu)]
    if finite_mu.size:
        mu_min = float(np.min(finite_mu))
        mu_max = float(np.percentile(finite_mu, 99.5))
        if not math.isfinite(mu_max) or mu_max <= mu_min:
            mu_max = float(np.max(finite_mu))
        if mu_max <= mu_min:
            mu_max = mu_min + 1.0
    else:
        mu_min, mu_max = 0.0, 1.0

    active_sizes = [level_sizes[level] for level in range(len(resolutions)) if level_counts[level] > 0]
    size_range = None
    if active_sizes:
        size_range = {"min": float(min(active_sizes)), "max": float(max(active_sizes))}

    return {
        "kind": "leaves",
        "count": int(idx.size),
        "total": total,
        "level_counts": level_counts,
        "level_sizes": level_sizes,
        "level_sizes_xyz": level_sizes_xyz,
        "resolutions": resolutions,
        "mu_range": {"min": mu_min, "max": mu_max},
        "size_range": size_range,
        "min_mu": float(min_mu),
        "encoding": "float32_base64",
        "positions": base64.b64encode(np.ascontiguousarray(positions).tobytes()).decode("ascii"),
        "sizes": base64.b64encode(np.ascontiguousarray(sizes).tobytes()).decode("ascii"),
        "sizes_xyz": base64.b64encode(np.ascontiguousarray(sizes_xyz).tobytes()).decode("ascii"),
        "mu": base64.b64encode(np.ascontiguousarray(sel_mu.astype(np.float32)).tobytes()).decode("ascii"),
        "levels": base64.b64encode(np.ascontiguousarray(sel_levels.astype(np.uint8)).tobytes()).decode("ascii"),
    }


def _leaf_corner_values(model, mu_np: np.ndarray) -> np.ndarray | None:
    """Per-leaf 8-corner mu values in (i,j,k) order (k fastest), i.e. the same
    corner ordering as `bernstein._all_coords(2)`.

    Returns None when every leaf is exactly constant (degree (0,0,0), or the
    model has no degree concept at all): in that case the flat per-leaf mean
    already renders correctly as a single-color box and there is no reason to
    pay 8x the bandwidth. Only leaves whose degree is exactly (1,1,1) get real
    corner values -- pipeline v5's p0/p1 leaves are the only non-constant
    case today -- pulled directly from their stored Bernstein coefficients
    (a degree-(1,1,1) block's 8 coefficients *are* its corner values, no
    fitting needed). Any other, currently unused, degree combination falls
    back to the flat mean for that leaf rather than guessing.
    """
    leaf_degrees = getattr(model, "leaf_degrees", None)
    if leaf_degrees is None:
        return None
    degrees_np = leaf_degrees.detach().cpu().numpy()
    is_p1 = np.all(degrees_np == 1, axis=1)
    if not np.any(is_p1):
        return None
    corners = np.repeat(mu_np[:, None], 8, axis=1).astype(np.float32)
    p1_ids = np.nonzero(is_p1)[0]
    physical = model.coefficients().detach().cpu().numpy().astype(np.float32)
    offsets_np = model.coefficient_offsets.detach().cpu().numpy()
    starts = offsets_np[p1_ids]
    corners[p1_ids] = physical[starts[:, None] + np.arange(8)[None, :]]
    return corners


def _leaf_geometry_binary(model, *, max_leaves: int | None = None, min_mu: float = 0.0) -> bytes:
    """Compact GPU-ready leaf stream.

    Coordinates stay uint16 and are converted to positions/sizes in the vertex
    shader. Compared with the JSON endpoint this removes base64, float position
    and size arrays, and all server-side box construction. When any leaf is a
    non-constant (p1) Bernstein block, its 8 corner values ride along too, so
    the renderer can interpolate the real function instead of flat-shading a
    single box (pipeline v5 step 9: "viewer must evaluate p1, not block-fill it").
    """
    raw_levels = getattr(model, "level_shapes", getattr(model, "level_resolutions", []))
    resolutions = [
        [int(v), int(v), int(v)] if isinstance(v, (int, np.integer)) else [int(c) for c in v]
        for v in raw_levels
    ]
    leaf_levels = getattr(model, "leaf_levels", None)
    leaf_coords = getattr(model, "leaf_coords", None)
    if not resolutions or leaf_levels is None or leaf_coords is None:
        raise ValueError("Model does not expose adaptive leaf geometry.")
    if max(max(shape) for shape in resolutions) > np.iinfo(np.uint16).max:
        raise ValueError("Binary viewer coordinate encoding supports resolutions up to uint16.")

    levels_np = leaf_levels.detach().cpu().numpy().astype(np.uint8, copy=False)
    coords_np = leaf_coords.detach().cpu().numpy().astype(np.uint16, copy=False)
    mu_np = model._leaf_mu().detach().cpu().numpy().astype(np.float32, copy=False)
    corners_np = _leaf_corner_values(model, mu_np)
    total = int(levels_np.shape[0])
    idx = np.flatnonzero(mu_np >= float(min_mu))
    if idx.size == 0:
        idx = np.arange(total)
    if max_leaves is not None and int(max_leaves) > 0 and idx.size > int(max_leaves):
        limit = int(max_leaves)
        order = np.argpartition(mu_np[idx], -limit)[-limit:]
        idx = idx[order]

    selected_coords = np.ascontiguousarray(coords_np[idx])
    selected_mu = np.ascontiguousarray(mu_np[idx])
    selected_levels = np.ascontiguousarray(levels_np[idx])
    count = int(idx.size)
    level_counts = [int(np.count_nonzero(levels_np == level)) for level in range(len(resolutions))]
    level_sizes_xyz = [[2.0 / float(value) for value in shape] for shape in resolutions]
    level_sizes = [float(np.prod(size) ** (1.0 / 3.0)) for size in level_sizes_xyz]
    finite_mu = selected_mu[np.isfinite(selected_mu)]

    coords_bytes = selected_coords.tobytes()
    coords_padding = (-len(coords_bytes)) % 4
    mu_offset = len(coords_bytes) + coords_padding
    mu_bytes = selected_mu.tobytes()
    levels_offset = mu_offset + len(mu_bytes)
    body = coords_bytes + (b"\0" * coords_padding) + mu_bytes + selected_levels.tobytes()

    corners_offset = None
    if corners_np is not None:
        levels_padding = (-len(selected_levels.tobytes())) % 4
        corners_offset = levels_offset + len(selected_levels.tobytes()) + levels_padding
        selected_corners = np.ascontiguousarray(corners_np[idx])
        body += (b"\0" * levels_padding) + selected_corners.tobytes()

    header = {
        "kind": "leaves",
        "encoding": "actleaf1",
        "count": count,
        "total": total,
        "level_counts": level_counts,
        "level_sizes": level_sizes,
        "level_sizes_xyz": level_sizes_xyz,
        "resolutions": resolutions,
        "mu_range": {
            "min": float(np.min(finite_mu)) if finite_mu.size else 0.0,
            "max": float(np.max(finite_mu)) if finite_mu.size else 1.0,
        },
        "size_range": {
            "min": float(min(level_sizes)),
            "max": float(max(level_sizes)),
        } if level_sizes else None,
        "min_mu": float(min_mu),
        "coords_offset": 0,
        "mu_offset": mu_offset,
        "levels_offset": levels_offset,
        "has_corners": corners_offset is not None,
        "corners_offset": corners_offset,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((-len(header_bytes)) % 4)
    return b"ACTLEAF1" + struct.pack("<I", len(header_bytes)) + header_bytes + body


def _global_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    pred_f = np.asarray(pred, dtype=np.float64)
    target_f = np.asarray(target, dtype=np.float64)
    if pred_f.size == 0:
        return 0.0
    data_range = max(float(np.max(target_f) - np.min(target_f)), float(np.max(pred_f) - np.min(pred_f)), 1e-6)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = float(np.mean(pred_f))
    mu_y = float(np.mean(target_f))
    var_x = float(np.var(pred_f))
    var_y = float(np.var(target_f))
    cov_xy = float(np.mean((pred_f - mu_x) * (target_f - mu_y)))
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if denominator <= 0.0:
        return 0.0
    return float(max(-1.0, min(1.0, numerator / denominator)))


def _array_stats(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_t = torch.from_numpy(np.array(pred, dtype=np.float32, copy=True))
    target_t = torch.from_numpy(np.array(target, dtype=np.float32, copy=True))
    diff = pred_t - target_t
    return {
        "psnr": psnr(pred_t, target_t),
        "mae": mae(pred_t, target_t),
        "ssim": _global_ssim(pred, target),
        "max_abs_error": float(torch.max(torch.abs(diff)).item()),
        "pred_min": float(pred_t.min().item()),
        "pred_max": float(pred_t.max().item()),
        "target_min": float(target_t.min().item()),
        "target_max": float(target_t.max().item()),
    }


def _percentile_range(*arrays: np.ndarray) -> tuple[float, float]:
    merged = np.concatenate([np.asarray(arr, dtype=np.float32).reshape(-1) for arr in arrays])
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _encode_gray_png(image: np.ndarray, *, vmin: float, vmax: float) -> str:
    arr = np.asarray(image, dtype=np.float32)
    scaled = np.clip((arr - float(vmin)) / max(float(vmax) - float(vmin), 1e-8), 0.0, 1.0)
    uint8_image = np.round(scaled * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(uint8_image, mode="L")
    handle = BytesIO()
    pil_image.save(handle, format="PNG")
    return "data:image/png;base64," + base64.b64encode(handle.getvalue()).decode("ascii")


def _encode_error_png(error: np.ndarray) -> str:
    arr = np.asarray(error, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    scale = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
    scale = max(scale, 1e-8)
    normalized = np.clip(arr / scale, -1.0, 1.0)
    rgb = np.ones((*normalized.shape, 3), dtype=np.float32)
    positive = normalized > 0.0
    negative = normalized < 0.0
    pos = normalized[positive]
    neg = -normalized[negative]
    rgb[positive, 0] = 1.0
    rgb[positive, 1] = 1.0 - 0.62 * pos
    rgb[positive, 2] = 1.0 - 0.86 * pos
    rgb[negative, 0] = 1.0 - 0.82 * neg
    rgb[negative, 1] = 1.0 - 0.48 * neg
    rgb[negative, 2] = 1.0
    uint8_image = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(uint8_image, mode="RGB")
    handle = BytesIO()
    pil_image.save(handle, format="PNG")
    return "data:image/png;base64," + base64.b64encode(handle.getvalue()).decode("ascii")


def _slice_from_volume(volume: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "x":
        return np.asarray(volume[index, :, :], dtype=np.float32).T
    if axis == "y":
        return np.asarray(volume[:, index, :], dtype=np.float32).T
    if axis == "z":
        return np.asarray(volume[:, :, index], dtype=np.float32).T
    raise ValueError("axis must be one of x, y, z")


def _volume3d_payload_from_arrays(
    pred_volume: np.ndarray,
    target_volume: np.ndarray,
    *,
    source: str,
    threshold: float,
    max_points: int,
) -> dict[str, Any]:
    pred = np.asarray(pred_volume, dtype=np.float32)
    target = np.asarray(target_volume, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} vs {target.shape}.")

    normalized_source = str(source).lower()
    if normalized_source not in {"prediction", "target", "error"}:
        raise ValueError("source must be prediction, target, or error")

    max_points = max(1, int(max_points))
    threshold = float(threshold)
    shape = pred.shape
    if normalized_source == "prediction":
        values = pred
        scores = pred
        mask = pred >= threshold
    elif normalized_source == "target":
        values = target
        scores = target
        mask = target >= threshold
    else:
        values = pred - target
        scores = np.abs(values)
        mask = (pred >= threshold) | (target >= threshold)

    candidate = np.flatnonzero(mask.reshape(-1))
    if candidate.size == 0:
        finite_scores = np.where(np.isfinite(scores.reshape(-1)), scores.reshape(-1), 0.0)
        candidate = np.argsort(finite_scores)[-min(max_points, finite_scores.size) :]

    flat_scores = np.where(np.isfinite(scores.reshape(-1)[candidate]), scores.reshape(-1)[candidate], 0.0)
    if candidate.size > max_points:
        keep = np.argpartition(flat_scores, -max_points)[-max_points:]
        candidate = candidate[keep]
        flat_scores = flat_scores[keep]
    order = np.argsort(flat_scores)
    candidate = candidate[order]

    x, y, z = np.unravel_index(candidate, shape)
    scale = np.asarray(shape, dtype=np.float32)
    coords = np.stack([x, y, z], axis=1).astype(np.float32)
    normalized = -1.0 + (coords + 0.5) * 2.0 / scale
    # Three.js uses Y as the vertical axis; map CT z to vertical while keeping x/y horizontal.
    positions = np.stack([normalized[:, 0], normalized[:, 2], normalized[:, 1]], axis=1).astype(np.float32)

    selected_values = values.reshape(-1)[candidate].astype(np.float32)
    finite_values = selected_values[np.isfinite(selected_values)]
    if finite_values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(finite_values, 1.0))
        vmax = float(np.percentile(finite_values, 99.5))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.min(finite_values))
            vmax = float(np.max(finite_values))
        if vmax <= vmin:
            vmax = vmin + 1.0
    t = np.clip((selected_values - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)

    if normalized_source == "prediction":
        colors = np.stack([0.35 + 0.65 * t, 0.55 + 0.45 * t, 0.50 + 0.50 * t], axis=1)
    elif normalized_source == "target":
        colors = np.stack([0.45 + 0.55 * t, 0.70 + 0.30 * t, 0.95 + 0.05 * t], axis=1)
    else:
        abs_values = np.abs(selected_values)
        max_abs = float(np.percentile(abs_values[np.isfinite(abs_values)], 99.0)) if np.any(np.isfinite(abs_values)) else 1.0
        strength = np.clip(abs_values / max(max_abs, 1e-8), 0.0, 1.0)
        colors = np.ones((selected_values.shape[0], 3), dtype=np.float32) * 0.86
        positive = selected_values >= 0.0
        colors[positive, 0] = 1.0
        colors[positive, 1] = 0.86 - 0.58 * strength[positive]
        colors[positive, 2] = 0.78 - 0.68 * strength[positive]
        colors[~positive, 0] = 0.40 - 0.24 * strength[~positive]
        colors[~positive, 1] = 0.62 - 0.34 * strength[~positive]
        colors[~positive, 2] = 1.0
    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)

    finite_pred = pred[np.isfinite(pred)]
    finite_target = target[np.isfinite(target)]
    return {
        "kind": "volume3d",
        "source": normalized_source,
        "threshold": threshold,
        "max_points": max_points,
        "shape": list(shape),
        "candidate_count": int(candidate.size),
        "material_count": int(np.count_nonzero(((pred >= threshold) | (target >= threshold)).reshape(-1))),
        "value_range": {"min": vmin, "max": vmax},
        "volume_range": {
            "prediction_min": float(np.min(finite_pred)) if finite_pred.size else 0.0,
            "prediction_max": float(np.max(finite_pred)) if finite_pred.size else 0.0,
            "target_min": float(np.min(finite_target)) if finite_target.size else 0.0,
            "target_max": float(np.max(finite_target)) if finite_target.size else 0.0,
        },
        "positions": positions.reshape(-1).tolist(),
        "colors": colors.reshape(-1).tolist(),
    }


def _volume_texture_payload_from_arrays(
    pred_volume: np.ndarray,
    target_volume: np.ndarray,
    *,
    source: str,
) -> dict[str, Any]:
    pred = np.asarray(pred_volume, dtype=np.float32)
    target = np.asarray(target_volume, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} vs {target.shape}.")

    normalized_source = str(source).lower()
    if normalized_source == "prediction":
        values = pred
        finite = values[np.isfinite(values)]
        vmin = 0.0
        vmax = float(np.percentile(finite, 99.7)) if finite.size else 1.0
    elif normalized_source == "target":
        values = target
        finite = values[np.isfinite(values)]
        vmin = 0.0
        vmax = float(np.percentile(finite, 99.7)) if finite.size else 1.0
    elif normalized_source == "error":
        values = pred - target
        finite = values[np.isfinite(values)]
        max_abs = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
        max_abs = max(max_abs, 1e-6)
        vmin = -max_abs
        vmax = max_abs
    else:
        raise ValueError("source must be prediction, target, or error")

    if not math.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    normalized = np.clip((values - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    quantized = np.round(normalized * 255.0).astype(np.uint8)
    # Data3DTexture expects x to be the fastest-varying coordinate:
    # offset = x + width * (y + height * z).
    texture_order = np.ascontiguousarray(quantized.transpose(2, 1, 0))
    return {
        "kind": "volume_texture",
        "source": normalized_source,
        "shape": list(pred.shape),
        "width": int(pred.shape[0]),
        "height": int(pred.shape[1]),
        "depth": int(pred.shape[2]),
        "value_range": {"min": float(vmin), "max": float(vmax)},
        "encoding": "uint8_base64_x_fastest",
        "data": base64.b64encode(texture_order.reshape(-1).tobytes()).decode("ascii"),
    }


def _volume_projection_payload_from_arrays(
    pred_projection: np.ndarray,
    target_projection: np.ndarray,
    *,
    source: str,
    angle_rad: float,
    elapsed_ms: float | None = None,
) -> dict[str, Any]:
    pred = np.asarray(pred_projection, dtype=np.float32)
    target = np.asarray(target_projection, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction and target projections differ: {pred.shape} vs {target.shape}.")
    normalized_source = str(source).lower()
    if normalized_source not in {"prediction", "target", "error"}:
        raise ValueError("source must be prediction, target, or error")
    vmin, vmax = _percentile_range(pred, target)
    return {
        "kind": "volume_projection",
        "source": normalized_source,
        "angle_rad": float(angle_rad),
        "elapsed_ms": None if elapsed_ms is None else float(elapsed_ms),
        "metrics": _array_stats(pred, target),
        "images": {
            "prediction": _encode_gray_png(pred, vmin=vmin, vmax=vmax),
            "target": _encode_gray_png(target, vmin=vmin, vmax=vmax),
            "error": _encode_error_png(pred - target),
        },
    }


@dataclass(frozen=True)
class ViewerPaths:
    config: Path
    checkpoint: Path | None
    dataset: Path
    output: Path
    report: Path | None


class ViewerState:
    def __init__(self, *, config_path: Path, checkpoint_path: Path | None) -> None:
        self.config_path = config_path.resolve()
        self.config = load_config(self.config_path)
        self.device = torch.device(self.config.get("device", "cuda"))
        # The dynamic leaf-voxel paths (state, leaves, slices, projection render)
        # are pure torch and run fine on CPU. Only /api/volume_projection uses the
        # native CUDA projector, which the current frontend no longer calls. So
        # fall back to CPU instead of refusing to start when CUDA is unavailable.
        if self.device.type == "cuda" and not torch.cuda.is_available():
            print("[viewer] CUDA unavailable; falling back to CPU (native volume_projection disabled).")
            self.device = torch.device("cpu")

        # Projections are used by the renderer, but the original-resolution GT
        # volume can be several GiB. Keep it memory-mapped on CPU and read only
        # the requested slice instead of permanently occupying GPU memory.
        self.dataset = load_r2_dataset(self.config["dataset"]["root"], device=self.device, load_volume=False)
        metadata_path = self.dataset.root / "meta_data.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        volume_path = Path(metadata["vol"])
        if not volume_path.is_absolute():
            volume_path = self.dataset.root / volume_path
        self.target_volume_path = volume_path.resolve()
        self.target_volume = np.load(self.target_volume_path, mmap_mode="r")
        if tuple(int(v) for v in self.target_volume.shape) != tuple(int(v) for v in self.dataset.volume_shape):
            raise ValueError(
                f"GT volume shape {self.target_volume.shape} does not match scanner nVoxel {self.dataset.volume_shape}."
            )
        self.detector_h, self.detector_w = self.dataset.detector_shape
        self.samples_per_ray = int(self.config["geometry"].get("samples_per_ray", self.dataset.volume_shape[0]))
        self.ray_chunk = int(self.config.get("viewer", {}).get("ray_chunk", self.config["training"].get("eval_ray_chunk", 4096)))
        self.model = build_model(self.config).to(device=self.device)
        self.checkpoint_path = checkpoint_path.resolve() if checkpoint_path is not None else _resolve_default_checkpoint(self.config)
        self.checkpoint_loaded = False
        self.checkpoint_error: str | None = None
        self._load_checkpoint()
        self.model.eval()

        output_dir = Path(self.config["output"]["dir"]).resolve()
        report_path = output_dir / "training_report.json"
        self.output_dir = output_dir
        self.report_path = report_path if report_path.exists() else None
        self.paths = ViewerPaths(
            config=self.config_path,
            checkpoint=self.checkpoint_path if self.checkpoint_loaded else None,
            dataset=self.dataset.root,
            output=output_dir,
            report=self.report_path,
        )
        self.training_report = _load_json_if_exists(report_path)
        self._decoded_volume: np.ndarray | None = None
        self._decoded_volume_tensor: torch.Tensor | None = None
        self._decoded_volume_time_sec: float | None = None
        self._lock = Lock()

    def _load_checkpoint(self) -> None:
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            self.checkpoint_error = f"Checkpoint not found: {self.checkpoint_path}"
            return
        try:
            if self.checkpoint_path.suffix.lower() == ".npz":
                state_dict, _metadata = load_compact_octree_state_dict(
                    self.model,
                    self.checkpoint_path,
                    device=self.device,
                )
            else:
                payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                state_dict = payload.get("model", payload)
                checkpoint_config = payload.get("config") if isinstance(payload, dict) else None
                if isinstance(checkpoint_config, dict):
                    checkpoint_representation = checkpoint_config.get("model", {}).get("representation")
                    if checkpoint_representation != self.config.get("model", {}).get("representation"):
                        self.model = build_model(checkpoint_config).to(device=self.device)
            self.model.prepare_sparse_from_state_dict(state_dict)
            self.model.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            self.checkpoint_error = f"Could not load checkpoint: {exc}"
            self.checkpoint_loaded = False
            return
        self.checkpoint_loaded = True

    def state_payload(self) -> dict[str, Any]:
        stats = asdict(self.model.stats())
        report_projection = None
        report_material = None
        boundary_sharpness = None
        ray_integration = None
        training_objective = None
        exports = None
        adaptive_events = None
        rate_distortion_curve = None
        elapsed_sec = None
        growth_events = None
        history = None
        if self.training_report is not None:
            report_projection = self.training_report.get("projection_test")
            report_material = self.training_report.get("material_volume")
            boundary_sharpness = self.training_report.get("boundary_sharpness")
            ray_integration = self.training_report.get("ray_integration")
            training_objective = self.training_report.get("training_objective")
            exports = self.training_report.get("exports")
            elapsed_sec = self.training_report.get("elapsed_sec")
            growth_events = self.training_report.get("growth_events")
            adaptive_events = self.training_report.get("adaptive_events")
            rate_distortion_curve = self.training_report.get("rate_distortion_curve")
            history = self.training_report.get("history")
        return _to_jsonable(
            {
                "name": "Projection-domain RD-CVF",
                "device": str(self.device),
                "checkpoint_loaded": self.checkpoint_loaded,
                "checkpoint_error": self.checkpoint_error,
                "paths": asdict(self.paths),
                "volume_shape": self.dataset.volume_shape,
                "detector_shape": self.dataset.detector_shape,
                "train_views": int(self.dataset.train.projections.shape[0]),
                "test_views": int(self.dataset.test.projections.shape[0]),
                "train_angles": self.dataset.train.angles.detach().float().cpu().numpy(),
                "test_angles": self.dataset.test.angles.detach().float().cpu().numpy(),
                "samples_per_ray": self.samples_per_ray,
                "ray_chunk": self.ray_chunk,
                "material_threshold": float(self.config.get("metrics", {}).get("material_threshold", 0.1)),
                "model": stats,
                "leaf_structure": _leaf_structure_payload(self.model),
                "report": {
                    "elapsed_sec": elapsed_sec,
                    "projection_test": report_projection,
                    "material_volume": report_material,
                    "boundary_sharpness": boundary_sharpness,
                    "ray_integration": ray_integration,
                    "training_objective": training_objective,
                    "exports": exports,
                    "growth_events": growth_events,
                    "adaptive_events": adaptive_events,
                    "rate_distortion_curve": rate_distortion_curve,
                    "history": history,
                },
            }
        )

    def artifacts_payload(self) -> dict[str, Any]:
        return _to_jsonable(_artifact_payload(self.output_dir, checkpoint_path=self.checkpoint_path, report_path=self.report_path))

    def _split(self, split_name: str) -> ProjectionSplit:
        if split_name == "train":
            return self.dataset.train
        if split_name == "test":
            return self.dataset.test
        raise ValueError("split must be train or test")

    def render_projection_payload(self, *, split_name: str, view_index: int) -> dict[str, Any]:
        split = self._split(split_name)
        view_index = max(0, min(int(view_index), int(split.projections.shape[0]) - 1))
        single = ProjectionSplit(
            angles=split.angles[view_index : view_index + 1],
            projections=split.projections[view_index : view_index + 1],
            paths=split.paths[view_index : view_index + 1],
        )
        with self._lock, torch.no_grad():
            start = time.perf_counter()
            pred_t = render_split(
                self.model,
                single,
                detector_h=self.detector_h,
                detector_w=self.detector_w,
                samples_per_ray=self.samples_per_ray,
                ray_chunk=self.ray_chunk,
            )[0]
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        pred = pred_t.detach().float().cpu().numpy()
        target = single.projections[0].detach().float().cpu().numpy()
        vmin, vmax = _percentile_range(pred, target)
        return {
            "kind": "projection",
            "split": split_name,
            "view": view_index,
            "angle_rad": float(single.angles[0].detach().cpu().item()),
            "source_path": str(single.paths[0]),
            "elapsed_ms": elapsed_ms,
            "device": str(self.device),
            "native_cuda_integrator": bool(
                hasattr(self.model, "prefer_compact_ray_batch") and self.model.prefer_compact_ray_batch()
            ),
            "metrics": _array_stats(pred, target),
            "images": {
                "prediction": _encode_gray_png(pred, vmin=vmin, vmax=vmax),
                "target": _encode_gray_png(target, vmin=vmin, vmax=vmax),
                "error": _encode_error_png(pred - target),
            },
        }

    def _decode_volume(self) -> np.ndarray:
        if self._decoded_volume is not None:
            return self._decoded_volume
        nx, ny, nz = self.dataset.volume_shape
        device = self.device
        xs = torch.arange(nx, dtype=torch.float32, device=device)
        ys = torch.arange(ny, dtype=torch.float32, device=device)
        zs = torch.arange(nz, dtype=torch.float32, device=device)
        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        scale = torch.tensor([nx, ny, nz], dtype=torch.float32, device=device)
        points = -1.0 + (coords + 0.5) * 2.0 / scale
        chunks: list[torch.Tensor] = []
        chunk_size = 65536
        with self._lock, torch.no_grad():
            start = time.perf_counter()
            for start_idx in range(0, points.shape[0], chunk_size):
                chunks.append(self.model.forward_mu(points[start_idx : start_idx + chunk_size]).detach().float().cpu())
            self._decoded_volume_time_sec = time.perf_counter() - start
        decoded = torch.cat(chunks, dim=0).reshape(nx, ny, nz).numpy()
        self._decoded_volume = decoded
        return decoded

    def _decode_slice(self, *, axis: str, index: int) -> np.ndarray:
        shape = tuple(int(v) for v in self.dataset.volume_shape)
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        free_axes = [dim for dim in range(3) if dim != axis_index]
        first = torch.arange(shape[free_axes[0]], dtype=torch.float32, device=self.device)
        second = torch.arange(shape[free_axes[1]], dtype=torch.float32, device=self.device)
        grid_first, grid_second = torch.meshgrid(first, second, indexing="ij")
        coords = torch.empty(
            (grid_first.numel(), 3),
            dtype=torch.float32,
            device=self.device,
        )
        coords[:, axis_index] = float(index)
        coords[:, free_axes[0]] = grid_first.reshape(-1)
        coords[:, free_axes[1]] = grid_second.reshape(-1)
        scale = torch.tensor(shape, dtype=torch.float32, device=self.device)
        points = -1.0 + (coords + 0.5) * 2.0 / scale

        chunks: list[torch.Tensor] = []
        chunk_size = 65536
        with self._lock, torch.no_grad():
            start = time.perf_counter()
            for start_idx in range(0, points.shape[0], chunk_size):
                chunks.append(self.model.forward_mu(points[start_idx : start_idx + chunk_size]).detach().float().cpu())
            elapsed = time.perf_counter() - start
        self._decoded_volume_time_sec = elapsed
        plane = torch.cat(chunks, dim=0).reshape(shape[free_axes[0]], shape[free_axes[1]]).numpy()
        # Match _slice_from_volume: the first free coordinate is horizontal,
        # the second free coordinate is vertical.
        return np.asarray(plane, dtype=np.float32).T

    def _decoded_volume_for_projection(self) -> torch.Tensor:
        if self._decoded_volume_tensor is None:
            self._decoded_volume_tensor = torch.from_numpy(self._decode_volume()).to(device=self.device, dtype=torch.float32)
        return self._decoded_volume_tensor

    def render_slice_payload(self, *, axis: str, index: int) -> dict[str, Any]:
        axis = str(axis).lower()
        dims = dict(zip(("x", "y", "z"), self.dataset.volume_shape))
        if axis not in dims:
            raise ValueError("axis must be x, y, or z")
        index = max(0, min(int(index), int(dims[axis]) - 1))
        pred = self._decode_slice(axis=axis, index=index)
        target = _slice_from_volume(self.target_volume, axis, index)
        vmin, vmax = _percentile_range(pred, target)
        return {
            "kind": "slice",
            "axis": axis,
            "index": index,
            "decoded_volume_time_sec": self._decoded_volume_time_sec,
            "elapsed_ms": (
                None if self._decoded_volume_time_sec is None else float(self._decoded_volume_time_sec) * 1000.0
            ),
            "device": str(self.device),
            "metrics": _array_stats(pred, target),
            "images": {
                "prediction": _encode_gray_png(pred, vmin=vmin, vmax=vmax),
                "target": _encode_gray_png(target, vmin=vmin, vmax=vmax),
                "error": _encode_error_png(pred - target),
            },
        }

    def leaf_geometry_payload(self, *, max_leaves: int | None, min_mu: float) -> dict[str, Any]:
        with self._lock, torch.no_grad():
            return _leaf_geometry_payload(self.model, max_leaves=max_leaves, min_mu=min_mu)

    def leaf_geometry_binary(self, *, max_leaves: int | None, min_mu: float) -> bytes:
        with self._lock, torch.no_grad():
            return _leaf_geometry_binary(self.model, max_leaves=max_leaves, min_mu=min_mu)

    def render_volume_texture_payload(self, *, source: str) -> dict[str, Any]:
        pred_volume = self._decode_volume()
        target_volume = np.asarray(self.target_volume, dtype=np.float32)
        return _volume_texture_payload_from_arrays(pred_volume, target_volume, source=source)

    def render_volume_projection_payload(self, *, angle_rad: float, source: str) -> dict[str, Any]:
        normalized_source = str(source).lower()
        if normalized_source not in {"prediction", "target", "error"}:
            raise ValueError("source must be prediction, target, or error")
        angle = torch.tensor([float(angle_rad)], dtype=torch.float32, device=self.device)
        pred_volume = self._decoded_volume_for_projection()
        target_volume = torch.from_numpy(np.asarray(self.target_volume, dtype=np.float32)).to(
            device=self.device,
            dtype=torch.float32,
        )
        with self._lock, torch.no_grad():
            start = time.perf_counter()
            pred_projection = project_dense_parallel(
                pred_volume,
                angle,
                detector_h=self.detector_h,
                detector_w=self.detector_w,
                samples_per_ray=self.samples_per_ray,
            )[0]
            target_projection = project_dense_parallel(
                target_volume,
                angle,
                detector_h=self.detector_h,
                detector_w=self.detector_w,
                samples_per_ray=self.samples_per_ray,
            )[0]
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        return _volume_projection_payload_from_arrays(
            pred_projection.detach().float().cpu().numpy(),
            target_projection.detach().float().cpu().numpy(),
            source=normalized_source,
            angle_rad=float(angle_rad),
            elapsed_ms=elapsed_ms,
        )


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "AdaptiveCTViewer/1.0"

    @property
    def viewer_state(self) -> ViewerState:
        return self.server.viewer_state  # type: ignore[attr-defined]

    def _send_bytes(self, body: bytes, *, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = (json.dumps(_to_jsonable(payload), ensure_ascii=False) + "\n").encode("utf-8")
        self._send_bytes(body, content_type="application/json; charset=utf-8", status=status)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json({"error": message, "status": status}, status=status)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > 65536:
            raise ValueError("Expected a JSON request body no larger than 64 KiB.")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON request body must be an object.")
        return payload

    _CONTENT_TYPES = {
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".ico": "image/x-icon",
        ".woff2": "font/woff2",
        ".map": "application/json; charset=utf-8",
    }

    @staticmethod
    def _dist_dir() -> Path:
        return Path(__file__).resolve().parent / "web" / "dist"

    def _try_serve_static(self, url_path: str) -> bool:
        """Serve the built Vite app from web/dist. Falls back to a build hint."""
        dist = self._dist_dir()
        index = dist / "index.html"
        if not index.exists():
            hint = (
                "<!doctype html><meta charset='utf-8'><title>Adaptive CT Viewer</title>"
                "<body style='font-family:system-ui;background:#0b0f14;color:#e8eef2;padding:40px'>"
                "<h1>Frontend not built</h1>"
                "<p>Build the viewer once before serving:</p>"
                "<pre style='background:#11171f;padding:14px;border-radius:8px'>"
                "cd adaptive_ct/viewer/web\nnpm install\nnpm run build</pre>"
                "<p>The API is live; only the static bundle is missing.</p></body>"
            )
            self._send_bytes(hint.encode("utf-8"), content_type="text/html; charset=utf-8", status=HTTPStatus.SERVICE_UNAVAILABLE)
            return True

        relative = url_path.lstrip("/")
        if not relative:
            target = index
        else:
            candidate = (dist / relative).resolve()
            try:
                candidate.relative_to(dist.resolve())
            except ValueError:
                self._send_error(HTTPStatus.FORBIDDEN, "Path outside dist.")
                return True
            # SPA fallback: unknown non-asset routes return index.html.
            target = candidate if candidate.is_file() else index

        content_type = self._CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
        self._send_bytes(target.read_bytes(), content_type=content_type)
        return True

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/" or not parsed.path.startswith("/api/") and parsed.path not in {"/healthz"}:
                if self._try_serve_static(parsed.path):
                    return
            if parsed.path == "/api/state":
                self._send_json(self.viewer_state.state_payload())
                return
            if parsed.path == "/api/artifacts":
                self._send_json(self.viewer_state.artifacts_payload())
                return
            if parsed.path == "/api/sources":
                self._send_json(_discover_viewer_sources(self.server.workspace_root))  # type: ignore[attr-defined]
                return
            if parsed.path == "/api/projection":
                split = query.get("split", ["test"])[0]
                view = int(query.get("view", ["0"])[0])
                self._send_json(self.viewer_state.render_projection_payload(split_name=split, view_index=view))
                return
            if parsed.path == "/api/slice":
                axis = query.get("axis", ["z"])[0]
                index = int(query.get("index", [str(self.viewer_state.dataset.volume_shape[2] // 2)])[0])
                self._send_json(self.viewer_state.render_slice_payload(axis=axis, index=index))
                return
            if parsed.path == "/api/leaves":
                max_leaves_raw = query.get("max_leaves", [None])[0]
                max_leaves = None if max_leaves_raw in {None, ""} else int(max_leaves_raw)
                min_mu = float(query.get("min_mu", ["0.0"])[0])
                self._send_json(self.viewer_state.leaf_geometry_payload(max_leaves=max_leaves, min_mu=min_mu))
                return
            if parsed.path == "/api/leaves.bin":
                max_leaves_raw = query.get("max_leaves", [None])[0]
                max_leaves = None if max_leaves_raw in {None, ""} else int(max_leaves_raw)
                min_mu = float(query.get("min_mu", ["0.0"])[0])
                self._send_bytes(
                    self.viewer_state.leaf_geometry_binary(max_leaves=max_leaves, min_mu=min_mu),
                    content_type="application/octet-stream",
                )
                return
            if parsed.path == "/api/volume_texture":
                source = query.get("source", ["prediction"])[0]
                self._send_json(self.viewer_state.render_volume_texture_payload(source=source))
                return
            if parsed.path == "/api/volume_projection":
                source = query.get("source", ["prediction"])[0]
                angle_rad = float(query.get("angle", ["0.0"])[0])
                self._send_json(self.viewer_state.render_volume_projection_payload(angle_rad=angle_rad, source=source))
                return
            if parsed.path == "/healthz":
                self._send_json({"status": "ok", "checkpoint_loaded": self.viewer_state.checkpoint_loaded})
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Not found.")
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - user-facing diagnostic path.
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path != "/api/load":
                self._send_error(HTTPStatus.NOT_FOUND, "Not found.")
                return
            payload = self._read_json_body()
            workspace_root: Path = self.server.workspace_root  # type: ignore[attr-defined]
            config_value = str(payload.get("config", "")).strip()
            checkpoint_value = str(payload.get("checkpoint", "")).strip()
            if not config_value:
                raise ValueError("A config file is required.")
            config_path = _resolve_workspace_path(workspace_root, config_value, suffixes={".yaml", ".yml"})
            checkpoint_path = (
                _resolve_workspace_path(workspace_root, checkpoint_value, suffixes={".pt", ".npz"})
                if checkpoint_value
                else None
            )
            new_state = ViewerState(config_path=config_path, checkpoint_path=checkpoint_path)
            if checkpoint_value and not new_state.checkpoint_loaded:
                raise ValueError(new_state.checkpoint_error or f"Could not load checkpoint: {checkpoint_path}")
            with self.server.viewer_state_lock:  # type: ignore[attr-defined]
                self.server.viewer_state = new_state  # type: ignore[attr-defined]
            self._send_json(
                {
                    "status": "ok",
                    "state": new_state.state_payload(),
                    "artifacts": new_state.artifacts_payload(),
                }
            )
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - user-facing diagnostic path.
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[viewer] {self.address_string()} - {fmt % args}")


def serve(*, config: Path, checkpoint: Path | None, host: str, port: int, open_browser: bool = False) -> None:
    resolved_config = config.resolve()
    cwd = Path.cwd().resolve()
    try:
        resolved_config.relative_to(cwd)
        workspace_root = cwd
    except ValueError:
        workspace_root = resolved_config.parent
    viewer_state = ViewerState(config_path=resolved_config, checkpoint_path=checkpoint)
    httpd = ThreadingHTTPServer((host, int(port)), ViewerHandler)
    httpd.viewer_state = viewer_state  # type: ignore[attr-defined]
    httpd.viewer_state_lock = Lock()  # type: ignore[attr-defined]
    httpd.workspace_root = workspace_root  # type: ignore[attr-defined]
    url = f"http://{host}:{port}/"
    print(json.dumps({"viewer": url, "checkpoint_loaded": viewer_state.checkpoint_loaded}, indent=2))
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the adaptive CT local viewer.")
    parser.add_argument("--config", default="configs/figure473_48v.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)

    serve(
        config=Path(args.config),
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        host=str(args.host),
        port=int(args.port),
        open_browser=bool(args.open),
    )


if __name__ == "__main__":
    main()
