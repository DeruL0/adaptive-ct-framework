from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import subprocess
import sys
import tempfile
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

from adaptive_ct.core.projector import project_dense_parallel
from adaptive_ct.codec.artifact import load_compact_octree_state_dict
from adaptive_ct.core.config import load_config
from adaptive_ct.data.dataset import ProjectionSplit, load_r2_dataset
from adaptive_ct.evaluation.metrics import mae, projection_metrics, psnr
from adaptive_ct.representation.factory import build_model
from adaptive_ct.evaluation.render import render_split


def _resolve_default_checkpoint(config: dict[str, Any]) -> Path:
    output_dir = Path(config["output"]["dir"])
    return (output_dir / "checkpoint.pt").resolve()


_MODEL_SUFFIXES = {".pt", ".npz"}
_CONFIG_SUFFIXES = {".yaml", ".yml"}


def _validate_model_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() not in _MODEL_SUFFIXES:
        raise ValueError(f"Unsupported model file: {path.name}. Choose a .pt or .npz file.")
    if not path.is_file():
        raise ValueError(f"Model file not found: {path}")
    return path


def _config_output_dir(config_path: Path) -> Path | None:
    try:
        config = load_config(config_path)
        value = config.get("output", {}).get("dir")
        return Path(value).expanduser().resolve() if value else None
    except Exception:
        return None


def _referenced_config_path(value: Any, *, model_dir: Path, workspace_root: Path) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [model_dir / raw, workspace_root / raw]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.suffix.lower() in _CONFIG_SUFFIXES and resolved.is_file():
            return resolved
    return None


def _report_config_candidates(model_dir: Path, workspace_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for report_path in sorted(model_dir.glob("*.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        for key in ("config", "config_path", "final_config", "reference_config"):
            candidate = _referenced_config_path(report.get(key), model_dir=model_dir, workspace_root=workspace_root)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _checkpoint_config_candidate(model_path: Path, workspace_root: Path) -> Path | None:
    if model_path.suffix.lower() != ".pt":
        return None
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    checkpoint_config = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(checkpoint_config, dict):
        return None
    model_dir = model_path.parent
    for key in ("_config_path", "config", "config_path"):
        candidate = _referenced_config_path(checkpoint_config.get(key), model_dir=model_dir, workspace_root=workspace_root)
        if candidate is not None:
            return candidate
    return None


def _resolve_model_config(model_path: str | Path, workspace_root: Path) -> Path | None:
    model = _validate_model_path(model_path)
    model_dir = model.parent
    workspace = workspace_root.resolve()

    adjacent = sorted(
        {
            *model_dir.glob("*.yaml"),
            *model_dir.glob("*.yml"),
        }
    )
    preferred_names = ("config.yaml", "config.yml", f"{model.stem}.yaml", f"{model.stem}.yml")
    for name in preferred_names:
        candidate = (model_dir / name).resolve()
        if candidate in adjacent:
            return candidate

    report_candidates = _report_config_candidates(model_dir, workspace)
    if report_candidates:
        return report_candidates[0]

    embedded_candidate = _checkpoint_config_candidate(model, workspace)
    if embedded_candidate is not None:
        return embedded_candidate

    config_root = workspace / "configs"
    workspace_configs = (
        sorted({*config_root.rglob("*.yaml"), *config_root.rglob("*.yml")})
        if config_root.exists()
        else []
    )
    output_matches = [path.resolve() for path in (*adjacent, *workspace_configs) if _config_output_dir(path) == model_dir]
    output_matches = list(dict.fromkeys(output_matches))
    if len(output_matches) == 1:
        return output_matches[0]
    if len(output_matches) > 1:
        names = ", ".join(str(path) for path in output_matches)
        raise ValueError(f"Multiple config files point to this model folder: {names}")
    if len(adjacent) == 1:
        return adjacent[0].resolve()
    if len(adjacent) > 1:
        names = ", ".join(path.name for path in adjacent)
        raise ValueError(f"Multiple YAML files found beside the model; keep one usable config: {names}")
    # No parameter YAML found. This is not an error anymore: the caller can fall
    # back to a model-only view synthesized from the model file itself, which
    # renders the adaptive leaf structure and model stats without a dataset.
    return None


def _synthesize_model_config(model_path: Path) -> dict[str, Any] | None:
    """Build a minimal, dataset-free config straight from the model file.

    Used when no parameter YAML can be resolved: compact `.npz` artifacts embed
    their octree geometry (level resolutions, max degree, attenuation shift) and
    `.pt` checkpoints usually embed their full training config. Either is enough
    to rebuild the model and inspect it standalone (voxel view + stats), just
    without the ground-truth dataset needed for slice/projection comparisons.
    Returns None when the file carries too little to reconstruct the model.
    """
    suffix = model_path.suffix.lower()
    if suffix == ".npz":
        try:
            with np.load(model_path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"][0])) if "metadata_json" in payload else {}
                if "level_resolutions" in payload:
                    levels = [int(v) for v in payload["level_resolutions"].tolist()]
                elif "level_shapes" in payload:
                    levels = [int(shape[0]) for shape in payload["level_shapes"].tolist()]
                else:
                    return None
                max_degree = (
                    [int(v) for v in payload["max_degree"].tolist()] if "max_degree" in payload else 3
                )
                attenuation_shift = (
                    float(payload["attenuation_shift"][0]) if "attenuation_shift" in payload else -3.0
                )
        except Exception:
            return None
        return {
            "device": "cpu",
            "model": {
                "representation": str(metadata.get("representation", "bernstein_octree")),
                "levels": levels,
                "max_degree": max_degree,
                "attenuation_shift": attenuation_shift,
                "topology": str(metadata.get("topology", "packed_hierarchy")),
                "cuda_integrator": False,
            },
        }
    if suffix == ".pt":
        try:
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception:
            return None
        embedded = payload.get("config") if isinstance(payload, dict) else None
        model_cfg = embedded.get("model") if isinstance(embedded, dict) else None
        if not isinstance(model_cfg, dict) or "levels" not in model_cfg:
            return None
        slim_model = dict(model_cfg)
        slim_model["cuda_integrator"] = False
        return {"device": "cpu", "model": slim_model}
    return None


def _open_model_file_dialog(initial_dir: Path) -> Path | None:
    initial = initial_dir.resolve() if initial_dir.is_dir() else Path.cwd().resolve()
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    executable = pythonw if pythonw.is_file() else Path(sys.executable)
    script = r"""
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog

result_path = Path(sys.argv[1])
initial_dir = sys.argv[2]
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
root.update_idletasks()
try:
    selected = filedialog.askopenfilename(
        parent=root,
        initialdir=initial_dir,
        title="Select Adaptive CT model",
        filetypes=(
            ("Adaptive CT model", "*.pt *.npz"),
            ("PyTorch checkpoint", "*.pt"),
            ("Compact NumPy model", "*.npz"),
        ),
    )
    result_path.write_text(selected, encoding="utf-8")
except Exception as exc:
    result_path.write_text("ERROR:" + repr(exc), encoding="utf-8")
    raise
finally:
    root.destroy()
"""
    with tempfile.TemporaryDirectory(prefix="adaptive_ct_picker_") as temporary:
        result_path = Path(temporary) / "selection.txt"
        completed = subprocess.run(
            [str(executable), "-c", script, str(result_path), str(initial)],
            check=False,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        result = result_path.read_text(encoding="utf-8") if result_path.is_file() else ""
    if completed.returncode != 0:
        detail = result.removeprefix("ERROR:") or "The system file dialog could not be opened."
        raise RuntimeError(detail)
    selected = result.strip()
    return _validate_model_path(selected) if selected else None


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
        "compact_octree": _summarize_compact_octree_artifact(compact_path),
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


def _leaf_corner_values(
    model,
    leaf_ids_np: np.ndarray,
    selected_mu_np: np.ndarray,
) -> np.ndarray | None:
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
    device = leaf_degrees.device
    leaf_ids = torch.as_tensor(leaf_ids_np, dtype=torch.long, device=device)
    degrees = leaf_degrees[leaf_ids]
    degrees_np = degrees.detach().cpu().numpy()
    is_p1 = np.all(degrees_np == 1, axis=1)
    if not np.any(is_p1):
        return None
    corners = np.repeat(selected_mu_np[:, None], 8, axis=1).astype(np.float32)
    p1_rows_np = np.nonzero(is_p1)[0]
    p1_rows = torch.as_tensor(p1_rows_np, dtype=torch.long, device=device)
    p1_leaf_ids = leaf_ids[p1_rows]
    local_rows = model.coefficient_offsets[p1_leaf_ids, None] + torch.arange(8, device=device)[None, :]
    values = model.coefficients()[local_rows]
    corners[p1_rows_np] = values.detach().float().cpu().numpy()
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
    corners_np = _leaf_corner_values(model, idx, selected_mu)
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
        body += (b"\0" * levels_padding) + np.ascontiguousarray(corners_np).tobytes()

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
    config: Path | None
    checkpoint: Path | None
    dataset: Path | None
    output: Path
    report: Path | None


class ViewerState:
    def __init__(
        self,
        *,
        config_path: Path | None,
        checkpoint_path: Path | None,
        config: dict[str, Any] | None = None,
        dataset_available: bool = True,
        dataset_note: str | None = None,
    ) -> None:
        if config is not None:
            # Model-only mode: a config dict synthesized from the model file, with
            # no dataset section. config_path may still point at a YAML we failed
            # to fully load (kept for display) or be None.
            self.config = config
            self.config_path = config_path.resolve() if config_path is not None else None
        else:
            if config_path is None:
                raise ValueError("ViewerState needs either a config path or a synthesized config.")
            self.config_path = config_path.resolve()
            self.config = load_config(self.config_path)
        self.dataset_available = bool(dataset_available)
        self.dataset_note = dataset_note
        self.device = torch.device(self.config.get("device", "cuda"))
        # The dynamic leaf-voxel paths (state, leaves, slices, projection render)
        # are pure torch and run fine on CPU. Only /api/volume_projection uses the
        # native CUDA projector, which the current frontend no longer calls. So
        # fall back to CPU instead of refusing to start when CUDA is unavailable.
        if self.device.type == "cuda" and not torch.cuda.is_available():
            print("[viewer] CUDA unavailable; falling back to CPU (native volume_projection disabled).")
            self.device = torch.device("cpu")

        if self.dataset_available:
            # Projections are used by the renderer, but the original-resolution GT
            # volume can be several GiB. Keep it memory-mapped on CPU and read only
            # the requested slice instead of permanently occupying GPU memory.
            self.dataset = load_r2_dataset(self.config["dataset"]["root"], device=self.device, load_volume=False)
            metadata_path = self.dataset.root / "meta_data.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            volume_path = Path(metadata["vol"])
            if not volume_path.is_absolute():
                volume_path = self.dataset.root / volume_path
            self.target_volume_path: Path | None = volume_path.resolve()
            self.target_volume = np.load(self.target_volume_path, mmap_mode="r")
            if tuple(int(v) for v in self.target_volume.shape) != tuple(int(v) for v in self.dataset.volume_shape):
                raise ValueError(
                    f"GT volume shape {self.target_volume.shape} does not match scanner nVoxel {self.dataset.volume_shape}."
                )
            self.detector_h, self.detector_w = self.dataset.detector_shape
            self.samples_per_ray = int(self.config["geometry"].get("samples_per_ray", self.dataset.volume_shape[0]))
        else:
            # Model-only: no ground-truth dataset, so slice/projection comparisons
            # are unavailable. Everything the voxel view and inspector need comes
            # straight from the model.
            self.dataset = None
            self.target_volume_path = None
            self.target_volume = None
            self.detector_h = self.detector_w = None
            self.samples_per_ray = None
        self.ray_chunk = int(
            self.config.get("viewer", {}).get(
                "ray_chunk", self.config.get("training", {}).get("eval_ray_chunk", 4096)
            )
        )
        self.model = build_model(self.config).to(device=self.device)
        self.checkpoint_path = checkpoint_path.resolve() if checkpoint_path is not None else _resolve_default_checkpoint(self.config)
        self.checkpoint_loaded = False
        self.checkpoint_error: str | None = None
        self._load_checkpoint()
        self.model.eval()

        if self.dataset_available:
            output_dir = Path(self.config["output"]["dir"]).resolve()
        elif self.checkpoint_path is not None:
            # Show artifacts that live beside the selected model file.
            output_dir = self.checkpoint_path.parent
        else:
            output_dir = Path.cwd().resolve()
        report_path = output_dir / "training_report.json"
        self.output_dir = output_dir
        self.report_path = report_path if report_path.exists() else None
        self.paths = ViewerPaths(
            config=self.config_path,
            checkpoint=self.checkpoint_path if self.checkpoint_loaded else None,
            dataset=self.dataset.root if self.dataset is not None else None,
            output=output_dir,
            report=self.report_path,
        )
        self.training_report = _load_json_if_exists(report_path)
        self._decoded_volume: np.ndarray | None = None
        self._decoded_volume_tensor: torch.Tensor | None = None
        self._decoded_volume_time_sec: float | None = None
        self._preview_volume_cache: dict[int, np.ndarray] = {}
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
        if self.dataset is not None:
            dataset_fields: dict[str, Any] = {
                "volume_shape": self.dataset.volume_shape,
                "detector_shape": self.dataset.detector_shape,
                "train_views": int(self.dataset.train.projections.shape[0]),
                "test_views": int(self.dataset.test.projections.shape[0]),
                "train_angles": self.dataset.train.angles.detach().float().cpu().numpy(),
                "test_angles": self.dataset.test.angles.detach().float().cpu().numpy(),
                "samples_per_ray": self.samples_per_ray,
            }
        else:
            # Model-only: no ground truth, so the slice/projection comparison
            # modes have nothing to compare against and the client hides them.
            dataset_fields = {
                "volume_shape": None,
                "detector_shape": None,
                "train_views": 0,
                "test_views": 0,
                "train_angles": [],
                "test_angles": [],
                "samples_per_ray": None,
            }
        return _to_jsonable(
            {
                "name": "Projection-domain RD-CVF",
                "device": str(self.device),
                "checkpoint_loaded": self.checkpoint_loaded,
                "checkpoint_error": self.checkpoint_error,
                "dataset_available": self.dataset_available,
                "dataset_error": self.dataset_note,
                "paths": asdict(self.paths),
                **dataset_fields,
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

    def _require_dataset(self) -> None:
        if self.dataset is None:
            raise ValueError(
                "This model was opened without a dataset, so slice and projection comparisons "
                "are unavailable. Open it alongside a parameter YAML to enable them."
            )

    def _split(self, split_name: str) -> ProjectionSplit:
        self._require_dataset()
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

    def _slice_points(
        self,
        *,
        axis: str,
        index: int,
    ) -> tuple[tuple[int, int, int], list[int], torch.Tensor]:
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
        return shape, free_axes, points

    def _decode_slice(self, *, axis: str, index: int) -> np.ndarray:
        shape, free_axes, points = self._slice_points(axis=axis, index=index)

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
        self._require_dataset()
        axis = str(axis).lower()
        dims = dict(zip(("x", "y", "z"), self.dataset.volume_shape))
        if axis not in dims:
            raise ValueError("axis must be x, y, or z")
        index = max(0, min(int(index), int(dims[axis]) - 1))
        pred = self._decode_slice(axis=axis, index=index)
        target = _slice_from_volume(self.target_volume, axis, index)
        vmin, vmax = _percentile_range(pred, target)
        images = {
            "prediction": _encode_gray_png(pred, vmin=vmin, vmax=vmax),
            "target": _encode_gray_png(target, vmin=vmin, vmax=vmax),
            "error": _encode_error_png(pred - target),
        }
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
            "images": images,
        }

    def leaf_geometry_payload(self, *, max_leaves: int | None, min_mu: float) -> dict[str, Any]:
        with self._lock, torch.no_grad():
            return _leaf_geometry_payload(self.model, max_leaves=max_leaves, min_mu=min_mu)

    def leaf_geometry_binary(self, *, max_leaves: int | None, min_mu: float) -> bytes:
        with self._lock, torch.no_grad():
            return _leaf_geometry_binary(self.model, max_leaves=max_leaves, min_mu=min_mu)

    def render_volume_texture_payload(self, *, resolution: int) -> dict[str, Any]:
        resolution = max(64, min(int(resolution), 256))
        cached = self._preview_volume_cache.get(resolution)
        if cached is None:
            with self._lock, torch.no_grad():
                decoded = self.model.decoded_at_resolution(
                    (resolution, resolution, resolution),
                    chunk=262144,
                )
                cached = decoded.detach().float().cpu().numpy()
            self._preview_volume_cache[resolution] = cached
        # The texture endpoint is model-intrinsic and also works when no GT
        # dataset was loaded. Passing the same array as target keeps the shared
        # quantizer focused on prediction without decoding the native volume.
        return _volume_texture_payload_from_arrays(cached, cached, source="prediction")

    def render_volume_projection_payload(self, *, angle_rad: float, source: str) -> dict[str, Any]:
        self._require_dataset()
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
            if parsed.path == "/api/projection":
                split = query.get("split", ["test"])[0]
                view = int(query.get("view", ["0"])[0])
                self._send_json(self.viewer_state.render_projection_payload(split_name=split, view_index=view))
                return
            if parsed.path == "/api/slice":
                axis = query.get("axis", ["z"])[0]
                dataset = self.viewer_state.dataset
                # Don't dereference the dataset to build a default index when it is
                # absent: render_slice_payload raises the friendly "no dataset"
                # error, but only if we reach it. index=0 is a safe placeholder.
                if "index" in query:
                    index = int(query["index"][0])
                elif dataset is not None:
                    index = int(dataset.volume_shape[2] // 2)
                else:
                    index = 0
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
                resolution = int(query.get("resolution", ["192"])[0])
                self._send_json(self.viewer_state.render_volume_texture_payload(resolution=resolution))
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

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            # Picking a file and loading a model are split into two requests on
            # purpose. Rebuilding the ViewerState re-reads the dataset and decodes
            # the model, which takes ~100 s on CPU; folding that into the picker
            # request left the browser blocked for a minute-plus with no feedback,
            # so the selection looked like it "never came back". Now the picker
            # returns the chosen path immediately and the client shows a real
            # "Loading..." state while it calls /api/load-model separately.
            if parsed.path == "/api/select-model":
                self._handle_select_model()
                return
            if parsed.path == "/api/load-model":
                self._handle_load_model()
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Not found.")
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - user-facing diagnostic path.
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))

    def _handle_select_model(self) -> None:
        workspace_root: Path = self.server.workspace_root  # type: ignore[attr-defined]
        dialog_lock: Lock = self.server.model_dialog_lock  # type: ignore[attr-defined]
        if not dialog_lock.acquire(blocking=False):
            self._send_error(HTTPStatus.CONFLICT, "A model file dialog is already open.")
            return
        try:
            current_checkpoint = self.viewer_state.checkpoint_path
            initial_dir = current_checkpoint.parent if current_checkpoint is not None else workspace_root / "output"
            checkpoint_path = _open_model_file_dialog(initial_dir)
        finally:
            dialog_lock.release()
        if checkpoint_path is None:
            self._send_json({"status": "cancelled"})
            return
        self._send_json({"status": "ok", "checkpoint": str(checkpoint_path)})

    def _handle_load_model(self) -> None:
        workspace_root: Path = self.server.workspace_root  # type: ignore[attr-defined]
        body = self._read_json_body()
        raw_checkpoint = body.get("checkpoint")
        if not isinstance(raw_checkpoint, str) or not raw_checkpoint.strip():
            raise ValueError("Request body must include the selected model 'checkpoint' path.")
        checkpoint_path = _validate_model_path(raw_checkpoint)
        # The parameter YAML is optional. If one resolves we open in full mode
        # (voxels + slice/projection comparisons); otherwise, or if the config's
        # dataset can't be loaded, we fall back to a model-only view built from
        # the model file itself (voxels + stats, no comparisons).
        config_path = _resolve_model_config(checkpoint_path, workspace_root)
        new_state: ViewerState | None = None
        dataset_note: str | None = None
        if config_path is not None:
            try:
                new_state = ViewerState(config_path=config_path, checkpoint_path=checkpoint_path)
            except Exception as exc:
                dataset_note = f"Config found but its dataset could not be loaded: {exc}"
        if new_state is None:
            model_config = _synthesize_model_config(checkpoint_path)
            if model_config is None:
                raise ValueError(
                    dataset_note
                    or (
                        "No parameter YAML could be resolved and the model file does not embed enough "
                        "metadata to inspect it standalone. Put config.yaml beside the model, or keep a "
                        "training report/config whose output.dir points to this folder."
                    )
                )
            new_state = ViewerState(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                config=model_config,
                dataset_available=False,
                dataset_note=dataset_note,
            )
        if not new_state.checkpoint_loaded:
            raise ValueError(new_state.checkpoint_error or f"Could not load checkpoint: {checkpoint_path}")
        with self.server.viewer_state_lock:  # type: ignore[attr-defined]
            self.server.viewer_state = new_state  # type: ignore[attr-defined]
        self._send_json(
            {
                "status": "ok",
                "config": str(config_path) if config_path is not None else None,
                "checkpoint": str(checkpoint_path),
                "dataset_available": new_state.dataset_available,
                "state": new_state.state_payload(),
                "artifacts": new_state.artifacts_payload(),
            }
        )

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
    httpd.model_dialog_lock = Lock()  # type: ignore[attr-defined]
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
    parser.add_argument("--config", default="configs/stage7/p1.yaml")
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
