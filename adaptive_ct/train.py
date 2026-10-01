from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import replace as dataclass_replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import load_config
from .compression import export_compact_octree_artifact, export_mact_artifact
from .data import load_r2_dataset
from .decision_rays import FixedDecisionRaySet, fixed_rss
from .dorfler_refinement import DorflerController, finetune_model, quantize_roundtrip_
from .latent_p1 import LatentP1Octree
from .geometry import (
    detector_grid,
    make_parallel_ray_points,
    random_projection_gradient_rays,
    random_training_rays,
)
from .backend import bernstein_segment_ray_capacity
from .bernstein import unique_degree_rows
from .losses import coefficient_face_continuity_loss, tv3
from .metrics import projection_metrics
from .model import build_model
from .projection_domain import (
    _tune_affected_coefficients,
    adaptive_projection_round,
    coefficient_diagnostics,
    projection_weights,
    split_projection_views,
    weighted_projection_gradient_loss,
    weighted_projection_loss,
)
from .refinement import build_refinement_score
from .render import render_split
from .surface import export_surface_artifact
from .volume_loss import VolumeSampler, volume_sample_loss
from .volume_metrics import boundary_sharpness_metrics, material_volume_metrics


def _make_optimizer(model, lr: float):
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)


def _projection_domain_representation(name: str) -> bool:
    return name in {"bernstein_octree", "bernstein", "rd_cvf", "latent_p1_octree", "latent_p1", "compact_neural_voxel_field"}


@torch.no_grad()
def _initialize_constant_mu(model, value: torch.Tensor) -> None:
    if hasattr(model, "initialize_constant_mu"):
        model.initialize_constant_mu(float(value.item()))
        return
    values = torch.full_like(model.coefficient_logits, float(value.item()))
    model.coefficient_logits.copy_(model._coefficients_to_logits(values))


@torch.no_grad()
def _export_center_slice_png(
    model,
    output_path: Path,
    volume_shape: tuple[int, int, int],
    *,
    axis: int = 2,
    chunk: int = 524288,
) -> dict:
    shape = tuple(int(value) for value in volume_shape)
    axis = int(axis)
    index = shape[axis] // 2
    dims = [dim for dim in range(3) if dim != axis]
    device = next(model.parameters()).device
    scale = torch.tensor(shape, dtype=torch.float32, device=device)
    grids = torch.meshgrid(
        torch.arange(shape[dims[0]], dtype=torch.float32, device=device),
        torch.arange(shape[dims[1]], dtype=torch.float32, device=device),
        indexing="ij",
    )
    coords = torch.zeros((grids[0].numel(), 3), dtype=torch.float32, device=device)
    coords[:, dims[0]] = grids[0].reshape(-1)
    coords[:, dims[1]] = grids[1].reshape(-1)
    coords[:, axis] = float(index)
    points = -1.0 + (coords + 0.5) * 2.0 / scale
    values = []
    for start in range(0, points.shape[0], int(chunk)):
        values.append(model.forward_mu(points[start : start + int(chunk)]).detach().float().cpu())
    image = torch.cat(values, dim=0).reshape(shape[dims[0]], shape[dims[1]]).numpy().T
    lo, hi = np.percentile(image, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    if hi <= lo:
        hi = lo + 1.0
    pixels = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    Image.fromarray((pixels * 255.0).astype(np.uint8), mode="L").save(output_path)
    return {
        "path": str(output_path.resolve()),
        "axis": axis,
        "index": index,
        "shape": list(image.shape),
        "display_percentiles": [float(lo), float(hi)],
        "source": "model_prediction_only",
    }


def _finest_model_shape(model, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    shapes = getattr(model, "level_shapes", None)
    if shapes:
        return tuple(int(value) for value in shapes[-1])
    resolutions = getattr(model, "level_resolutions", None)
    if resolutions:
        return (int(resolutions[-1]),) * 3
    return tuple(int(value) for value in fallback)


def _degree3(value) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),) * 3
    if value is None:
        return ()
    return tuple(int(item) for item in value)


def _structural_round_milestones(progressive: dict) -> dict[int, dict]:
    rounds_cfg = progressive.get("structural_rounds", {}) or {}
    if not bool(rounds_cfg.get("enabled", False)):
        return {}
    rounds = int(rounds_cfg.get("rounds", 0))
    if rounds < 1:
        raise ValueError("training.progressive.structural_rounds.rounds must be positive.")
    start = int(rounds_cfg.get("start_iteration", rounds_cfg.get("start", 1)))
    interval = int(rounds_cfg.get("interval", rounds_cfg.get("every", 1000)))
    if interval < 1:
        raise ValueError("training.progressive.structural_rounds.interval must be positive.")
    level_schedule = rounds_cfg.get("level_schedule", rounds_cfg.get("levels"))
    level_schedule = [int(value) for value in (level_schedule or [])]
    skip = {"enabled", "rounds", "start_iteration", "start", "interval", "every", "level_schedule", "levels"}
    template = {key: value for key, value in rounds_cfg.items() if key not in skip}
    milestones: dict[int, dict] = {}
    for index in range(rounds):
        spec = dict(template)
        if level_schedule:
            spec["level"] = level_schedule[min(index, len(level_schedule) - 1)]
        spec.setdefault("operation", "h_split")
        spec["structural_round_index"] = index + 1
        spec["structural_rounds"] = rounds
        milestones[start + index * interval] = spec
    return milestones


def _validate_first_paper_scope(config: dict) -> None:
    model = config.get("model", {}) or {}
    training = config.get("training", {}) or {}
    export = config.get("export", {}) or {}
    errors: list[str] = []
    if str(model.get("representation", "")).lower() != "latent_p1_octree":
        errors.append("model.representation must be latent_p1_octree")
    if str(model.get("topology", "")).lower() != "packed_hierarchy":
        errors.append("model.topology must be packed_hierarchy")
    if "max_degree" in model or "initial_leaf_degree" in model:
        errors.append("first-paper latent P1 fixes the local field degree; remove max_degree/initial_leaf_degree")
    if str(model.get("initialization", "projection_mean")).lower() == "fdk_volume":
        errors.append("model.initialization must not use a reconstructed volume")
    for key in ("tv_weight",):
        if float(training.get(key, 0.0) or 0.0) != 0.0:
            errors.append(f"training.{key} must be 0")
    for dotted in (
        ("volume_loss", "weight"),
        ("projection_gradient_loss", "weight"),
        ("coefficient_continuity", "weight"),
    ):
        value = (training.get(dotted[0], {}) or {}).get(dotted[1], 0.0)
        if float(value or 0.0) != 0.0:
            errors.append(f"training.{dotted[0]}.{dotted[1]} must be 0")
    adaptive = training.get("adaptive", {}) or {}
    if bool(adaptive.get("allow_p", False)):
        errors.append("training.adaptive.allow_p must be false")
    if bool(adaptive.get("enabled", False)) and not bool(adaptive.get("held_out_validation", True)):
        errors.append("training.adaptive.held_out_validation must stay true")
    progressive = training.get("progressive", {}) or {}
    milestones = list((progressive.get("convergence_milestones", []) or []))
    milestones.extend((progressive.get("milestones", {}) or {}).values())
    for index, milestone in enumerate(milestones):
        operation = str((milestone or {}).get("operation", (milestone or {}).get("action", "h_split"))).lower()
        if operation.startswith("p") or operation in {"degree", "degree_elevate", "hp", "hp_compete", "hp_rate_distortion"}:
            errors.append(f"progressive milestone {index} uses forbidden degree adaptation: {operation}")
    for key in ("mact", "surface"):
        if bool((export.get(key, {}) or {}).get("enabled", False)):
            errors.append(f"export.{key}.enabled must be false")
    if errors:
        raise ValueError("Invalid first-paper projection-only config:\n- " + "\n- ".join(errors))


def _validate_stage0_direct_p1_scope(config: dict) -> None:
    model = config.get("model", {}) or {}
    training = config.get("training", {}) or {}
    metrics = config.get("metrics", {}) or {}
    export = config.get("export", {}) or {}
    errors: list[str] = []
    if str(model.get("representation", "")).lower() != "bernstein_octree":
        errors.append("model.representation must be bernstein_octree")
    if str(model.get("topology", "")).lower() != "packed_hierarchy":
        errors.append("model.topology must be packed_hierarchy")
    if str(model.get("integration_mode", "")).lower() != "exact":
        errors.append("model.integration_mode must be exact")
    if _degree3(model.get("max_degree")) != (1, 1, 1):
        errors.append("model.max_degree must be [1, 1, 1]")
    if _degree3(model.get("initial_leaf_degree")) != (1, 1, 1):
        errors.append("model.initial_leaf_degree must be [1, 1, 1]")
    for key in ("latent_dim", "decoder_hidden", "hash_encoding", "fourier_features"):
        if key in model:
            errors.append(f"stage0 direct P1 must not set model.{key}")
    if str(model.get("initialization", "projection_mean")).lower() == "fdk_volume":
        errors.append("model.initialization must not use a reconstructed volume")
    if float(training.get("tv_weight", 0.0) or 0.0) != 0.0:
        errors.append("training.tv_weight must be 0")
    for dotted in (
        ("volume_loss", "weight"),
        ("projection_gradient_loss", "weight"),
        ("coefficient_continuity", "weight"),
    ):
        value = (training.get(dotted[0], {}) or {}).get(dotted[1], 0.0)
        if float(value or 0.0) != 0.0:
            errors.append(f"training.{dotted[0]}.{dotted[1]} must be 0")
    adaptive = training.get("adaptive", {}) or {}
    if bool(adaptive.get("enabled", True)):
        errors.append("training.adaptive.enabled must be false")
    progressive = training.get("progressive", {}) or {}
    active_convergence = [
        value for value in (progressive.get("convergence_milestones", []) or [])
        if bool((value or {}).get("enabled", True))
    ]
    active_milestones = [
        value for value in (progressive.get("milestones", {}) or {}).values()
        if bool((value or {}).get("enabled", True))
    ]
    if active_convergence or active_milestones:
        errors.append("stage0 direct P1 keeps topology fixed; remove progressive milestones")
    if bool(training.get("allow_test_during_training", True)):
        errors.append("training.allow_test_during_training must be false")
    if bool(metrics.get("compute_volume_metrics", False)) or bool(metrics.get("compute_boundary_metrics", False)):
        errors.append("stage0 direct P1 must not read GT volume metrics")
    for key in ("mact", "surface"):
        if bool((export.get(key, {}) or {}).get("enabled", False)):
            errors.append(f"export.{key}.enabled must be false")
    if errors:
        raise ValueError("Invalid stage0 direct-P1 baseline config:\n- " + "\n- ".join(errors))


def _validate_stage1_equal_topology_scope(config: dict) -> None:
    model = config.get("model", {}) or {}
    training = config.get("training", {}) or {}
    metrics = config.get("metrics", {}) or {}
    export = config.get("export", {}) or {}
    errors: list[str] = []
    if str(model.get("representation", "")).lower() != "latent_p1_octree":
        errors.append("model.representation must be latent_p1_octree")
    if str(model.get("topology", "")).lower() != "packed_hierarchy":
        errors.append("model.topology must be packed_hierarchy")
    if str(model.get("integration_mode", "")).lower() != "exact":
        errors.append("model.integration_mode must be exact")
    if int(model.get("latent_dim", 0) or 0) != 4:
        errors.append("model.latent_dim must be 4 for the first neural gate")
    for key in ("decoder_hidden", "max_degree", "initial_leaf_degree", "hash_encoding", "fourier_features"):
        if key in model:
            errors.append(f"stage1 equal-topology latent P1 must not set model.{key}")
    if str(model.get("initialization", "projection_mean")).lower() == "fdk_volume":
        errors.append("model.initialization must not use a reconstructed volume")
    if float(training.get("tv_weight", 0.0) or 0.0) != 0.0:
        errors.append("training.tv_weight must be 0")
    for dotted in (
        ("volume_loss", "weight"),
        ("projection_gradient_loss", "weight"),
        ("coefficient_continuity", "weight"),
    ):
        value = (training.get(dotted[0], {}) or {}).get(dotted[1], 0.0)
        if float(value or 0.0) != 0.0:
            errors.append(f"training.{dotted[0]}.{dotted[1]} must be 0")
    if bool((training.get("adaptive", {}) or {}).get("enabled", True)):
        errors.append("training.adaptive.enabled must be false for equal-topology comparison")
    progressive = training.get("progressive", {}) or {}
    if (progressive.get("convergence_milestones", []) or []) or (progressive.get("milestones", {}) or {}):
        errors.append("stage1 equal-topology keeps topology fixed; remove progressive milestones")
    if bool(training.get("allow_test_during_training", True)):
        errors.append("training.allow_test_during_training must be false")
    if bool(metrics.get("compute_volume_metrics", False)) or bool(metrics.get("compute_boundary_metrics", False)):
        errors.append("stage1 equal-topology must not read GT volume metrics")
    for key in ("mact", "surface"):
        if bool((export.get(key, {}) or {}).get("enabled", False)):
            errors.append(f"export.{key}.enabled must be false")
    if errors:
        raise ValueError("Invalid stage1 equal-topology latent-P1 config:\n- " + "\n- ".join(errors))


def _validate_stage3_crossfit_split_scope(config: dict) -> None:
    model = config.get("model", {}) or {}
    training = config.get("training", {}) or {}
    metrics = config.get("metrics", {}) or {}
    errors: list[str] = []
    if str(model.get("representation", "")).lower() != "latent_p1_octree":
        errors.append("model.representation must be latent_p1_octree")
    if str(model.get("topology", "")).lower() != "packed_hierarchy":
        errors.append("model.topology must be packed_hierarchy")
    if str(model.get("integration_mode", "")).lower() != "exact":
        errors.append("model.integration_mode must be exact")
    if "decoder_hidden" in model or "hash_encoding" in model or "fourier_features" in model:
        errors.append("stage3 uses only the shared linear latent decoder")
    if str(model.get("initialization", "projection_mean")).lower() == "fdk_volume":
        errors.append("model.initialization must not use a reconstructed volume")
    if float(training.get("tv_weight", 0.0) or 0.0) != 0.0:
        errors.append("training.tv_weight must be 0")
    for dotted in (
        ("volume_loss", "weight"),
        ("projection_gradient_loss", "weight"),
        ("coefficient_continuity", "weight"),
    ):
        value = (training.get(dotted[0], {}) or {}).get(dotted[1], 0.0)
        if float(value or 0.0) != 0.0:
            errors.append(f"training.{dotted[0]}.{dotted[1]} must be 0")
    if bool(training.get("allow_test_during_training", True)):
        errors.append("training.allow_test_during_training must be false")
    if bool(metrics.get("compute_volume_metrics", False)) or bool(metrics.get("compute_boundary_metrics", False)):
        errors.append("stage3 topology selection must not read GT volume metrics")
    progressive = training.get("progressive", {}) or {}
    milestones = list(progressive.get("convergence_milestones", []) or [])
    milestones.extend((progressive.get("milestones", {}) or {}).values())
    milestones.extend(_structural_round_milestones(progressive).values())
    active = [item or {} for item in milestones if bool((item or {}).get("enabled", True))]
    if not active:
        errors.append("stage3 needs at least one h_split milestone")
    for index, item in enumerate(active):
        operation = str(item.get("operation", item.get("action", "h_split"))).lower()
        if operation not in {"h_split", "h", "split", "global_h_split", "global_split"}:
            errors.append(f"milestone {index} must be h_split only, got {operation!r}")
        strategy = str(item.get("strategy", "")).lower()
        if strategy not in {"residual", "projection_residual", "projected_residual", "heldout", "held_out"}:
            errors.append(f"milestone {index} must use residual cheap proposal")
        if not bool((item.get("crossfit_acceptance", {}) or {}).get("enabled", False)):
            errors.append(f"milestone {index} must enable crossfit_acceptance")
    if errors:
        raise ValueError("Invalid stage3 cross-fit split config:\n- " + "\n- ".join(errors))


def _validate_stage4_structural_split_scope(config: dict) -> None:
    _validate_stage3_crossfit_split_scope(config)
    progressive = ((config.get("training", {}) or {}).get("progressive", {}) or {})
    fixed_rounds = bool((progressive.get("structural_rounds", {}) or {}).get("enabled", False))
    dynamic_rounds = any(
        bool((item or {}).get("enabled", True))
        for item in (progressive.get("convergence_milestones", []) or [])
    )
    if not fixed_rounds and not dynamic_rounds:
        raise ValueError(
            "Invalid stage4 structural split config:\n- enable fixed structural rounds or convergence-triggered refinement"
        )


def _validate_stage5_merge_scope(config: dict) -> None:
    model = config.get("model", {}) or {}
    training = config.get("training", {}) or {}
    metrics = config.get("metrics", {}) or {}
    errors: list[str] = []
    if str(model.get("representation", "")).lower() != "latent_p1_octree":
        errors.append("model.representation must be latent_p1_octree")
    if str(model.get("topology", "")).lower() != "packed_hierarchy":
        errors.append("model.topology must be packed_hierarchy")
    if str(model.get("integration_mode", "")).lower() != "exact":
        errors.append("model.integration_mode must be exact")
    if "decoder_hidden" in model or "hash_encoding" in model or "fourier_features" in model:
        errors.append("stage5 uses only the shared linear latent decoder")
    if str(model.get("initialization", "projection_mean")).lower() == "fdk_volume":
        errors.append("model.initialization must not use a reconstructed volume")
    if float(training.get("tv_weight", 0.0) or 0.0) != 0.0:
        errors.append("training.tv_weight must be 0")
    for dotted in (
        ("volume_loss", "weight"),
        ("projection_gradient_loss", "weight"),
        ("coefficient_continuity", "weight"),
    ):
        value = (training.get(dotted[0], {}) or {}).get(dotted[1], 0.0)
        if float(value or 0.0) != 0.0:
            errors.append(f"training.{dotted[0]}.{dotted[1]} must be 0")
    if bool(training.get("allow_test_during_training", True)):
        errors.append("training.allow_test_during_training must be false")
    if bool(metrics.get("compute_volume_metrics", False)) or bool(metrics.get("compute_boundary_metrics", False)):
        errors.append("stage5 topology selection must not read GT volume metrics")
    progressive = training.get("progressive", {}) or {}
    milestones = list(progressive.get("convergence_milestones", []) or [])
    milestones.extend((progressive.get("milestones", {}) or {}).values())
    milestones.extend(_structural_round_milestones(progressive).values())
    active = [item or {} for item in milestones if bool((item or {}).get("enabled", True))]
    if not active:
        errors.append("stage5 needs h_split/h_merge structural milestones")
    merge_seen = False
    for index, item in enumerate(active):
        operation = str(item.get("operation", item.get("action", "h_split"))).lower()
        if operation not in {"h_split", "h", "split", "global_h_split", "global_split", "h_merge", "merge"}:
            errors.append(f"milestone {index} must be h_split or h_merge only, got {operation!r}")
        if operation in {"h_merge", "merge"}:
            merge_seen = True
            if not bool((item.get("crossfit_acceptance", {}) or item.get("acceptance", {}) or {}).get("enabled", False)):
                errors.append(f"merge milestone {index} must enable crossfit_acceptance")
            continue
        strategy = str(item.get("strategy", "")).lower()
        if strategy not in {"residual", "projection_residual", "projected_residual", "heldout", "held_out"}:
            errors.append(f"split milestone {index} must use residual cheap proposal")
        if not bool((item.get("crossfit_acceptance", {}) or {}).get("enabled", False)):
            errors.append(f"split milestone {index} must enable crossfit_acceptance")
    if not merge_seen:
        errors.append("stage5 needs at least one h_merge milestone")
    if errors:
        raise ValueError("Invalid stage5 merge config:\n- " + "\n- ".join(errors))


def _convergence_window_summary(
    values,
    *,
    window: int,
    relative_improvement_threshold: float,
    max_relative_regression: float = 0.005,
) -> dict:
    """Summarize whether a smoothed stochastic loss window has plateaued."""
    window = max(4, int(window))
    if len(values) < window:
        return {
            "ready": False,
            "samples": len(values),
            "window": window,
            "relative_improvement": None,
        }
    recent = list(values)[-window:]
    edge = max(1, window // 4)
    start = sum(recent[:edge]) / float(edge)
    end = sum(recent[-edge:]) / float(edge)
    relative_improvement = (start - end) / max(abs(start), 1.0e-12)
    return {
        "ready": (
            relative_improvement >= -abs(float(max_relative_regression))
            and relative_improvement < float(relative_improvement_threshold)
        ),
        "samples": len(recent),
        "window": window,
        "start_ema": float(start),
        "end_ema": float(end),
        "relative_improvement": float(relative_improvement),
    }


def _dynamic_refinement_status(
    *,
    tested: int,
    accepted_gains: list[float],
    round_index: int,
    min_rounds: int,
    max_rounds: int,
    stop_acceptance_ratio: float,
    current_bytes: int,
    byte_budget: int | None,
    min_added_bytes: int,
) -> dict:
    accepted = len(accepted_gains)
    acceptance_ratio = accepted / max(int(tested), 1)
    mean_gain = sum(accepted_gains) / accepted if accepted else 0.0
    budget_exhausted = (
        byte_budget is not None
        and int(current_bytes) + max(1, int(min_added_bytes)) > int(byte_budget)
    )
    reasons = []
    if int(tested) == 0:
        reasons.append("no tested candidates")
    if int(round_index) >= int(min_rounds) and acceptance_ratio < float(stop_acceptance_ratio):
        reasons.append("acceptance ratio below threshold")
    if budget_exhausted:
        reasons.append("byte budget exhausted")
    if int(round_index) >= int(max_rounds):
        reasons.append("safety round limit reached")
    return {
        "round": int(round_index),
        "tested": int(tested),
        "accepted": accepted,
        "acceptance_ratio": float(acceptance_ratio),
        "mean_accepted_validation_gain": float(mean_gain),
        "stop_acceptance_ratio": float(stop_acceptance_ratio),
        "current_bytes": int(current_bytes),
        "byte_budget": int(byte_budget) if byte_budget is not None else None,
        "stop": bool(reasons),
        "stop_reasons": reasons,
    }


def _projection_weight_kwargs(
    train_cfg: dict,
    adaptive_cfg: dict,
    local_cfg: dict | None = None,
    *,
    iteration: int | None = None,
) -> dict:
    weight_cfg = dict(train_cfg.get("projection_weighting", {}) or {})
    schedule = train_cfg.get("projection_weighting_schedule", {}) or {}
    if schedule:
        current_iteration = 1 if iteration is None else int(iteration)
        stages = sorted((int(key), value or {}) for key, value in schedule.items())
        for start, stage_cfg in stages:
            if start > current_iteration:
                break
            weight_cfg.update(stage_cfg)
    if local_cfg:
        weight_cfg.update({key: value for key, value in local_cfg.items() if value is not None})
    return {
        "epsilon": float(weight_cfg.get("epsilon", adaptive_cfg.get("weight_epsilon", 1e-4))),
        "mode": str(weight_cfg.get("mode", "inverse")),
        "target_power": float(weight_cfg.get("target_power", 1.0)),
        "min_weight": weight_cfg.get("min_weight"),
        "max_weight": weight_cfg.get("max_weight"),
        "blend_alpha": float(weight_cfg.get("blend_alpha", 0.15)),
    }


def _projection_gradient_magnitude(projection: torch.Tensor) -> torch.Tensor:
    gx = torch.zeros_like(projection)
    gy = torch.zeros_like(projection)
    gx[:, 1:-1] = 0.5 * (projection[:, 2:] - projection[:, :-2])
    gy[1:-1, :] = 0.5 * (projection[2:, :] - projection[:-2, :])
    return torch.sqrt(gx * gx + gy * gy)


def _projection_gradient_effective_weight(config: dict, iteration: int) -> float:
    weight = float(config.get("weight", 0.0))
    start = int(config.get("start_iteration", 1))
    stop = config.get("stop_iteration")
    if weight <= 0.0 or int(iteration) < start or (stop is not None and int(iteration) > int(stop)):
        return 0.0
    ramp_steps = max(0, int(config.get("ramp_steps", 0)))
    if ramp_steps <= 0:
        return weight
    return weight * min(1.0, float(int(iteration) - start + 1) / float(ramp_steps))


def _projection_gradient_loss_kwargs(config: dict) -> dict:
    return {
        "edge_weight_power": float(config.get("edge_weight_power", 0.0)),
        "edge_min_weight": config.get("edge_min_weight"),
        "edge_max_weight": config.get("edge_max_weight"),
        "magnitude_weight": float(config.get("magnitude_weight", 0.0)),
        "magnitude_quantile": float(config.get("magnitude_quantile", 0.75)),
        "moment_weight": float(config.get("moment_weight", 0.0)),
        "moment_quantile": float(config.get("moment_quantile", 0.75)),
        "endpoint_weight": float(config.get("endpoint_weight", 0.0)),
        "endpoint_quantile": float(config.get("endpoint_quantile", 0.75)),
    }


def _sample_projection_gradient_batch(
    split,
    config: dict,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    materialize_points: bool,
    generator: torch.Generator | None = None,
):
    return random_projection_gradient_rays(
        split,
        edge_rays=int(config.get("edge_rays", 2048)),
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        strides=tuple(int(value) for value in config.get("strides", [1, 2, 4])),
        uniform_fraction=float(config.get("uniform_fraction", 0.5)),
        candidate_multiplier=int(config.get("candidate_multiplier", 8)),
        materialize_points=materialize_points,
        generator=generator,
)


def _model_level_shapes(model) -> list[tuple[int, int, int]]:
    levels = model.level_shapes if hasattr(model, "level_shapes") else model.level_resolutions
    shapes: list[tuple[int, int, int]] = []
    for value in levels:
        if isinstance(value, int):
            shapes.append((int(value),) * 3)
        else:
            shape = tuple(int(component) for component in value)
            if len(shape) != 3:
                raise ValueError(f"Invalid model level shape: {value!r}.")
            shapes.append(shape)
    return shapes


@torch.no_grad()
def _bernstein_refinement_score(model, *, level: int, strategy: str) -> torch.Tensor:
    """Build a projection-domain score grid for forced Bernstein h-refinement.

    This is a bootstrap path for experiments that must expose multiple octree
    levels. It uses only coefficient diagnostics and leaf means; no dense volume
    or volume-domain gradient is loaded.
    """
    parent_level = int(level) - 1
    level_shapes = _model_level_shapes(model)
    if parent_level < 0 or parent_level >= len(level_shapes):
        raise ValueError(f"Invalid Bernstein refinement level: {level}.")
    shape = tuple(int(value) for value in level_shapes[parent_level])
    device = model.coefficient_logits.device
    score = torch.zeros(shape, dtype=torch.float32, device=device)
    mask = model.leaf_levels == parent_level
    if not torch.any(mask):
        return score
    leaf_ids = torch.nonzero(mask, as_tuple=False).reshape(-1)
    coords = model.leaf_coords[leaf_ids].to(dtype=torch.long)
    normalized_strategy = str(strategy).lower()
    if normalized_strategy in {"uniform", "all"}:
        values = torch.ones((leaf_ids.numel(),), dtype=torch.float32, device=device)
    elif normalized_strategy in {"mu", "attenuation", "leaf_mu"}:
        values = model._leaf_mu()[leaf_ids].detach().float()
    elif normalized_strategy in {"coefficient", "coeff", "diagnostic", "projection", "jump", "interface"}:
        diagnostics = coefficient_diagnostics(model)
        values = torch.maximum(
            diagnostics.directional_variation[leaf_ids].amax(dim=1),
            diagnostics.interface_jump[leaf_ids].amax(dim=1),
        ).detach().float()
        # Early constant fields can have numerically flat diagnostics. Fall
        # back to attenuation so top-k selection remains deterministic and
        # still coefficient-only.
        if bool(torch.all(values <= 0.0).item()):
            values = model._leaf_mu()[leaf_ids].detach().float()
    else:
        raise ValueError(
            "Projection-domain Bernstein milestones support strategy "
            "'coefficient', 'mu', or 'uniform'."
        )
    score[coords[:, 0], coords[:, 1], coords[:, 2]] = values
    return score


@torch.no_grad()
def _select_coefficient_h_candidates(
    model,
    *,
    level: int,
    score: torch.Tensor,
    active_fraction: float,
    min_mu_threshold: float | None,
    max_fraction: float,
    reserve_leaf_count: int = 0,
    max_parent_count: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Union top coefficient diagnostics with a material-support threshold.

    The threshold prevents a small top-fraction gate from permanently locking
    material cells at a coarse ancestor. Threshold candidates are prioritized;
    the top-score candidates fill the remaining capacity. The global leaf
    budget remains a hard limit.
    """
    parent_level = int(level) - 1
    candidates = torch.nonzero(model.leaf_levels == parent_level, as_tuple=False).reshape(-1)
    if candidates.numel() == 0:
        return candidates, {
            "candidate_parent_count": 0,
            "top_fraction_count": 0,
            "material_threshold": min_mu_threshold,
            "material_threshold_count": 0,
            "selection_cap": 0,
        }
    level_shapes = _model_level_shapes(model)
    parent_shape = tuple(int(value) for value in level_shapes[parent_level])
    if tuple(score.shape) != parent_shape:
        score = F.interpolate(
            score[None, None].to(dtype=torch.float32),
            size=parent_shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0]
    coords = model.leaf_coords[candidates]
    candidate_scores = score[coords[:, 0], coords[:, 1], coords[:, 2]]
    safe_scores = torch.where(
        torch.isfinite(candidate_scores),
        candidate_scores,
        torch.full_like(candidate_scores, -float("inf")),
    )
    top_count = max(1, int(candidates.numel() * float(active_fraction)))
    top_positions = torch.topk(safe_scores, min(top_count, candidates.numel())).indices
    selected_mask = torch.zeros((candidates.numel(),), dtype=torch.bool, device=candidates.device)
    selected_mask[top_positions] = True

    leaf_mu = model._leaf_mu()[candidates].detach().float()
    threshold_mask = torch.zeros_like(selected_mask)
    if min_mu_threshold is not None:
        threshold_mask = torch.isfinite(leaf_mu) & (leaf_mu >= float(min_mu_threshold))
        selected_mask |= threshold_mask

    cap = max(0, int(candidates.numel() * float(max_fraction)))
    if max_parent_count is not None:
        cap = min(cap, max(0, int(max_parent_count)))
    if model.max_leaf_count is not None:
        cap = min(
            cap,
            max(
                0,
                (
                    int(model.max_leaf_count)
                    - int(model.leaf_levels.shape[0])
                    - max(0, int(reserve_leaf_count))
                )
                // 7,
            ),
        )
    threshold_positions = torch.nonzero(threshold_mask, as_tuple=False).reshape(-1)
    if threshold_positions.numel() >= cap:
        positions = (
            threshold_positions[torch.topk(leaf_mu[threshold_positions], cap).indices]
            if cap > 0
            else threshold_positions[:0]
        )
    else:
        positions = threshold_positions
        remaining = cap - int(positions.numel())
        fill_mask = selected_mask & ~threshold_mask
        fill_positions = torch.nonzero(fill_mask, as_tuple=False).reshape(-1)
        if remaining > 0 and fill_positions.numel():
            take = torch.topk(safe_scores[fill_positions], min(remaining, fill_positions.numel())).indices
            positions = torch.cat([positions, fill_positions[take]])
    if positions.numel():
        positions = positions[torch.argsort(safe_scores[positions], descending=True)]
    return candidates[positions], {
        "candidate_parent_count": int(candidates.numel()),
        "top_fraction_count": int(top_count),
        "material_threshold": float(min_mu_threshold) if min_mu_threshold is not None else None,
        "material_threshold_count": int(threshold_positions.numel()),
        "selection_cap": int(cap),
        "max_parent_count": int(max_parent_count) if max_parent_count is not None else None,
        "selected_parent_count": int(positions.numel()),
    }


@torch.no_grad()
def _bernstein_leaf_score(model, *, level: int, strategy: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return leaf ids and coefficient-domain scores for p-refinement."""
    device = model.coefficient_logits.device
    leaf_ids = torch.nonzero(model.leaf_levels == int(level), as_tuple=False).reshape(-1)
    if leaf_ids.numel() == 0:
        return leaf_ids, torch.empty((0,), dtype=torch.float32, device=device)
    normalized_strategy = str(strategy).lower()
    if normalized_strategy in {"uniform", "all"}:
        values = torch.ones((leaf_ids.numel(),), dtype=torch.float32, device=device)
    elif normalized_strategy in {"mu", "attenuation", "leaf_mu"}:
        values = model._leaf_mu()[leaf_ids].detach().float()
    elif normalized_strategy in {"coefficient", "coeff", "diagnostic", "variation"}:
        diagnostics = coefficient_diagnostics(model)
        values = diagnostics.directional_variation[leaf_ids].amax(dim=1).detach().float()
        if bool(torch.all(values <= 0.0).item()):
            values = model._leaf_mu()[leaf_ids].detach().float()
    else:
        raise ValueError("Bernstein p-refinement milestones support strategy 'mu', 'coefficient', or 'uniform'.")
    return leaf_ids, values


def _haar_detail_basis(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal 3D Haar detail basis for eight dyadic child constants."""
    child_ids = torch.arange(8, device=device, dtype=torch.long)
    x = torch.where((child_ids & 4) == 0, 1.0, -1.0)
    y = torch.where((child_ids & 2) == 0, 1.0, -1.0)
    z = torch.where((child_ids & 1) == 0, 1.0, -1.0)
    basis = torch.stack([x, y, z, x * y, x * z, y * z, x * y * z], dim=1)
    return basis.to(dtype=dtype) / (8.0**0.5)


@torch.no_grad()
def _virtual_haar_ray_features(model, ray_batch, ray_ids: torch.Tensor, leaf_ids: torch.Tensor) -> torch.Tensor:
    """Line integrals of seven virtual child-detail basis functions.

    The current function is not changed. Each selected degree-0 leaf is
    virtually divided into eight children and intersected analytically by the
    parallel ray. Multiplying the eight child path lengths by the zero-mean
    Haar basis gives the Jacobian of the ray integral with respect to the seven
    new h-detail degrees of freedom.
    """
    device = model.coefficient_logits.device
    dtype = torch.float32
    ray_ids = ray_ids.to(device=device, dtype=torch.long)
    leaf_ids = leaf_ids.to(device=device, dtype=torch.long)
    if ray_ids.numel() == 0:
        return torch.empty((0, 7), dtype=dtype, device=device)
    if ray_batch.angles is None or ray_batch.rows is None or ray_batch.cols is None:
        raise ValueError("Haar predicted gain requires parallel-ray metadata.")
    degrees = model.leaf_degrees[leaf_ids]
    if torch.any(degrees != 0):
        raise ValueError("Haar predicted gain currently requires degree-0 candidate leaves.")

    effective_angles = -ray_batch.angles.to(device=device, dtype=dtype)[ray_ids]
    ca = torch.cos(effective_angles)
    sa = torch.sin(effective_angles)
    directions = torch.stack([ca, sa], dim=1)
    u_axes = torch.stack([-sa, ca], dim=1)
    cols = ray_batch.cols.to(device=device, dtype=dtype)[ray_ids]
    rows = ray_batch.rows.to(device=device, dtype=dtype)[ray_ids]
    u = -1.0 + (cols + 0.5) * 2.0 / float(ray_batch.detector_w)
    z_world = -1.0 + (rows + 0.5) * 2.0 / float(ray_batch.detector_h)
    bases = u_axes * u[:, None]

    levels = model.leaf_levels[leaf_ids]
    level_shapes = _model_level_shapes(model)
    resolutions = torch.tensor(level_shapes, dtype=dtype, device=device)[levels]
    cell = 2.0 / resolutions
    coords = model.leaf_coords[leaf_ids].to(dtype=dtype)
    lower = -1.0 + coords * cell
    middle = lower + 0.5 * cell
    upper = lower + cell
    z_bit = (z_world >= middle[:, 2]).to(dtype=torch.long)
    z_valid = (z_world >= lower[:, 2] - 1.0e-7) & (z_world <= upper[:, 2] + 1.0e-7)
    features = torch.zeros((leaf_ids.numel(), 7), dtype=dtype, device=device)
    basis = _haar_detail_basis(device=device, dtype=dtype)
    eps = 1.0e-8
    for x_bit in (0, 1):
        for y_bit in (0, 1):
            lo_x = lower[:, 0] if x_bit == 0 else middle[:, 0]
            hi_x = middle[:, 0] if x_bit == 0 else upper[:, 0]
            lo_y = lower[:, 1] if y_bit == 0 else middle[:, 1]
            hi_y = middle[:, 1] if y_bit == 0 else upper[:, 1]
            t_min = torch.full_like(z_world, -1.0e20)
            t_max = torch.full_like(z_world, 1.0e20)
            valid = z_valid.clone()
            for axis, lo, hi in ((0, lo_x, hi_x), (1, lo_y, hi_y)):
                direction = directions[:, axis]
                base = bases[:, axis]
                parallel = direction.abs() < eps
                valid &= (~parallel) | ((base >= lo - eps) & (base <= hi + eps))
                denominator = torch.where(parallel, torch.ones_like(direction), direction)
                t0 = (lo - base) / denominator
                t1 = (hi - base) / denominator
                t_min = torch.where(parallel, t_min, torch.maximum(t_min, torch.minimum(t0, t1)))
                t_max = torch.where(parallel, t_max, torch.minimum(t_max, torch.maximum(t0, t1)))
            length = torch.where(valid & (t_max > t_min + eps), t_max - t_min, torch.zeros_like(t_min))
            child_ids = int(x_bit * 4 + y_bit * 2) + z_bit
            features += length[:, None] * basis[child_ids]
    return features


@torch.no_grad()
def _virtual_trilinear_ray_features(
    model,
    ray_batch,
    ray_ids: torch.Tensor,
    leaf_ids: torch.Tensor,
) -> torch.Tensor:
    """Jacobian of a degree-0 -> trilinear p action in its seven new modes.

    Degree elevation embeds the old constant coefficient along the all-ones
    direction of the eight trilinear Bernstein coefficients.  The orthogonal
    Haar columns span exactly the seven newly available function directions.
    Two-point Gauss quadrature is exact because a ray-restricted trilinear
    basis is quadratic in the ray parameter.
    """
    device = model.coefficient_logits.device
    dtype = torch.float32
    ray_ids = ray_ids.to(device=device, dtype=torch.long)
    leaf_ids = leaf_ids.to(device=device, dtype=torch.long)
    if ray_ids.numel() == 0:
        return torch.empty((0, 7), dtype=dtype, device=device)
    if ray_batch.angles is None or ray_batch.rows is None or ray_batch.cols is None:
        raise ValueError("Trilinear predicted gain requires parallel-ray metadata.")
    if torch.any(model.leaf_degrees[leaf_ids] != 0):
        raise ValueError("Trilinear predicted gain currently requires degree-0 candidate leaves.")

    effective_angles = -ray_batch.angles.to(device=device, dtype=dtype)[ray_ids]
    ca = torch.cos(effective_angles)
    sa = torch.sin(effective_angles)
    directions = torch.stack([ca, sa], dim=1)
    u_axes = torch.stack([-sa, ca], dim=1)
    cols = ray_batch.cols.to(device=device, dtype=dtype)[ray_ids]
    rows = ray_batch.rows.to(device=device, dtype=dtype)[ray_ids]
    u = -1.0 + (cols + 0.5) * 2.0 / float(ray_batch.detector_w)
    z_world = -1.0 + (rows + 0.5) * 2.0 / float(ray_batch.detector_h)
    bases = u_axes * u[:, None]

    level_shapes = _model_level_shapes(model)
    levels = model.leaf_levels[leaf_ids]
    resolutions = torch.tensor(level_shapes, dtype=dtype, device=device)[levels]
    cell = 2.0 / resolutions
    coords = model.leaf_coords[leaf_ids].to(dtype=dtype)
    lower = -1.0 + coords * cell
    upper = lower + cell

    eps = 1.0e-8
    t_min = torch.full_like(z_world, -1.0e20)
    t_max = torch.full_like(z_world, 1.0e20)
    valid = (z_world >= lower[:, 2] - eps) & (z_world <= upper[:, 2] + eps)
    for axis in range(2):
        direction = directions[:, axis]
        base = bases[:, axis]
        parallel = direction.abs() < eps
        valid &= (~parallel) | ((base >= lower[:, axis] - eps) & (base <= upper[:, axis] + eps))
        denominator = torch.where(parallel, torch.ones_like(direction), direction)
        t0 = (lower[:, axis] - base) / denominator
        t1 = (upper[:, axis] - base) / denominator
        t_min = torch.where(parallel, t_min, torch.maximum(t_min, torch.minimum(t0, t1)))
        t_max = torch.where(parallel, t_max, torch.minimum(t_max, torch.maximum(t0, t1)))
    valid &= t_max > t_min + eps

    half = 0.5 * (t_max - t_min)
    centre = 0.5 * (t_max + t_min)
    nodes = torch.tensor(
        [-0.5773502691896257, 0.5773502691896257],
        dtype=dtype,
        device=device,
    )
    t = centre[:, None] + half[:, None] * nodes[None, :]
    x = bases[:, 0, None] + directions[:, 0, None] * t
    y = bases[:, 1, None] + directions[:, 1, None] * t
    local_x = ((x - lower[:, 0, None]) / cell[:, 0, None]).clamp(0.0, 1.0)
    local_y = ((y - lower[:, 1, None]) / cell[:, 1, None]).clamp(0.0, 1.0)
    local_z = ((z_world - lower[:, 2]) / cell[:, 2]).clamp(0.0, 1.0)
    bx = torch.stack([1.0 - local_x, local_x], dim=2)
    by = torch.stack([1.0 - local_y, local_y], dim=2)
    bz = torch.stack([1.0 - local_z, local_z], dim=1)
    corner_integrals = (
        half[:, None, None, None]
        * torch.einsum("nqi,nqj,nk->nijk", bx, by, bz)
    ).reshape(-1, 8)
    corner_integrals = torch.where(valid[:, None], corner_integrals, torch.zeros_like(corner_integrals))
    return corner_integrals @ _haar_detail_basis(device=device, dtype=dtype)


@torch.no_grad()
def _bernstein_hp_gauss_newton_scores(
    model,
    split,
    *,
    levels: Sequence[int] | None,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    ray_chunk: int,
    max_views: int | None,
    max_rays_per_view: int | None,
    weight_kwargs: dict,
    fisher_damping: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Estimate comparable h and p gains from the same residual/Jacobian data."""
    device = split.projections.device
    eligible = torch.all(model.leaf_degrees == 0, dim=1)
    if levels is not None:
        requested = torch.tensor(list(levels), dtype=torch.long, device=device)
        eligible &= torch.any(model.leaf_levels[:, None] == requested[None, :], dim=1)
    candidate_ids = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    if candidate_ids.numel() == 0:
        empty = torch.empty((0,), dtype=torch.float32, device=device)
        return candidate_ids, empty, empty, {
            "candidate_count": 0,
            "score_views": 0,
            "score_rays": 0,
        }

    leaf_to_candidate = torch.full(
        (model.leaf_levels.numel(),),
        -1,
        dtype=torch.int32,
        device=device,
    )
    leaf_to_candidate[candidate_ids] = torch.arange(candidate_ids.numel(), dtype=torch.int32, device=device)
    gradient_h = torch.zeros((candidate_ids.numel(), 7), dtype=torch.float32, device=device)
    fisher_h = torch.zeros_like(gradient_h)
    gradient_p = torch.zeros_like(gradient_h)
    fisher_p = torch.zeros_like(gradient_h)

    rows_all, cols_all = detector_grid(detector_h, detector_w, device)
    if max_rays_per_view is not None and int(max_rays_per_view) < rows_all.numel():
        sample_ids = torch.linspace(
            0,
            rows_all.numel() - 1,
            max(1, int(max_rays_per_view)),
            device=device,
        ).round().long()
        rows_all = rows_all[sample_ids]
        cols_all = cols_all[sample_ids]
    all_view_count = int(split.angles.shape[0])
    if max_views is None or int(max_views) >= all_view_count:
        view_ids = torch.arange(all_view_count, device=device)
    else:
        view_ids = torch.linspace(
            0,
            all_view_count - 1,
            max(1, int(max_views)),
            device=device,
        ).round().long().unique(sorted=True)

    total_rays = 0
    for view_id_tensor in view_ids:
        view_id = int(view_id_tensor.item())
        angle = split.angles[view_id]
        for start in range(0, rows_all.numel(), max(1, int(ray_chunk))):
            rows = rows_all[start : start + int(ray_chunk)]
            cols = cols_all[start : start + int(ray_chunk)]
            targets = split.projections[view_id, rows, cols]
            ray_batch = make_parallel_ray_points(
                angle.expand(rows.shape[0]),
                rows,
                cols,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                targets=targets,
                materialize_points=not (
                    hasattr(model, "prefer_compact_ray_batch") and model.prefer_compact_ray_batch()
                ),
            )
            ray_ids, leaf_ids, contributions = model.ray_cell_contributions(ray_batch)
            prediction = contributions.new_zeros((ray_batch.num_rays,))
            prediction.scatter_add_(0, ray_ids, contributions)
            residual = prediction - targets
            weights = projection_weights(targets, **weight_kwargs)
            positions = leaf_to_candidate[leaf_ids].long()
            keep = positions >= 0
            if torch.any(keep):
                selected_rays = ray_ids[keep]
                selected_leaves = leaf_ids[keep]
                positions = positions[keep]
                h_features = _virtual_haar_ray_features(
                    model,
                    ray_batch,
                    selected_rays,
                    selected_leaves,
                )
                p_features = _virtual_trilinear_ray_features(
                    model,
                    ray_batch,
                    selected_rays,
                    selected_leaves,
                )
                weighted_residual = weights[selected_rays] * residual[selected_rays]
                selected_weights = weights[selected_rays]
                for detail in range(7):
                    h_phi = h_features[:, detail]
                    p_phi = p_features[:, detail]
                    gradient_h[:, detail].scatter_add_(0, positions, weighted_residual * h_phi)
                    fisher_h[:, detail].scatter_add_(0, positions, selected_weights * h_phi.square())
                    gradient_p[:, detail].scatter_add_(0, positions, weighted_residual * p_phi)
                    fisher_p[:, detail].scatter_add_(0, positions, selected_weights * p_phi.square())
            total_rays += int(targets.numel())

    damping = max(float(fisher_damping), 0.0)
    h_gain = 0.5 * torch.sum(gradient_h.square() / (fisher_h + damping), dim=1)
    p_gain = 0.5 * torch.sum(gradient_p.square() / (fisher_p + damping), dim=1)
    h_gain = torch.where(torch.any(fisher_h > 0.0, dim=1) & torch.isfinite(h_gain), h_gain, 0.0)
    p_gain = torch.where(torch.any(fisher_p > 0.0, dim=1) & torch.isfinite(p_gain), p_gain, 0.0)
    return candidate_ids, h_gain, p_gain, {
        "candidate_count": int(candidate_ids.numel()),
        "score_views": int(view_ids.numel()),
        "score_rays": total_rays,
        "fisher_damping": damping,
        "h_gain_mean": float(h_gain.mean().item()),
        "h_gain_max": float(h_gain.max().item()),
        "p_gain_mean": float(p_gain.mean().item()),
        "p_gain_max": float(p_gain.max().item()),
    }


@torch.no_grad()
def _apply_hp_gauss_newton_rate_distortion(
    model,
    split,
    *,
    levels: Sequence[int] | None,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    ray_chunk: int,
    max_views: int | None,
    max_rays_per_view: int | None,
    weight_kwargs: dict,
    fisher_damping: float,
    max_added_bytes: int,
    h_added_bytes: int = 99,
    p_added_bytes: int = 14,
) -> tuple[int, dict]:
    """Choose one h/p action per cell by predicted objective gain per byte."""
    candidate_ids, h_gain, p_gain, summary = _bernstein_hp_gauss_newton_scores(
        model,
        split,
        levels=levels,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=ray_chunk,
        max_views=max_views,
        max_rays_per_view=max_rays_per_view,
        weight_kwargs=weight_kwargs,
        fisher_damping=fisher_damping,
    )
    if candidate_ids.numel() == 0 or int(max_added_bytes) <= 0:
        return 0, {
            **summary,
            "h_selected": 0,
            "p_selected": 0,
            "predicted_added_bytes": 0,
            "predicted_gain": 0.0,
        }

    h_valid = model.leaf_levels[candidate_ids] + 1 < len(_model_level_shapes(model))
    p_valid = torch.all(
        model.leaf_degrees[candidate_ids]
        < torch.tensor(model.max_degree, dtype=torch.long, device=candidate_ids.device)[None, :],
        dim=1,
    )
    h_rate = torch.where(h_valid, h_gain / float(h_added_bytes), torch.full_like(h_gain, -float("inf")))
    p_rate = torch.where(p_valid, p_gain / float(p_added_bytes), torch.full_like(p_gain, -float("inf")))
    choose_h = h_rate >= p_rate
    best_rate = torch.where(choose_h, h_rate, p_rate)
    best_gain = torch.where(choose_h, h_gain, p_gain)
    order = torch.argsort(best_rate, descending=True)
    candidate_cpu = candidate_ids.detach().cpu().numpy()
    choose_h_cpu = choose_h.detach().cpu().numpy()
    best_rate_cpu = best_rate.detach().cpu().numpy()
    best_gain_cpu = best_gain.detach().cpu().numpy()

    remaining_bytes = int(max_added_bytes)
    remaining_h = (
        max(0, (int(model.max_leaf_count) - int(model.leaf_levels.numel())) // 7)
        if model.max_leaf_count is not None
        else int(candidate_ids.numel())
    )
    selected_h: list[int] = []
    selected_p: list[int] = []
    predicted_gain = 0.0
    for position in order.detach().cpu().tolist():
        rate = float(best_rate_cpu[position])
        if not math.isfinite(rate) or rate <= 0.0:
            break
        is_h = bool(choose_h_cpu[position])
        action_bytes = int(h_added_bytes if is_h else p_added_bytes)
        if action_bytes > remaining_bytes:
            continue
        leaf_id = int(candidate_cpu[position])
        if is_h:
            if remaining_h <= 0:
                continue
            selected_h.append(leaf_id)
            remaining_h -= 1
        else:
            selected_p.append(leaf_id)
        remaining_bytes -= action_bytes
        predicted_gain += float(best_gain_cpu[position])

    h_keys = [model.leaf_key(leaf_id) for leaf_id in selected_h]
    p_keys = [model.leaf_key(leaf_id) for leaf_id in selected_p]
    if p_keys:
        p_ids = [model.find_leaf(key) for key in p_keys]
        model.elevate_leaves_isotropic_batch(
            [leaf_id for leaf_id in p_ids if leaf_id is not None],
            (1, 1, 1),
        )
    if h_keys:
        h_ids = [model.find_leaf(key) for key in h_keys]
        model.split_leaves_batch([leaf_id for leaf_id in h_ids if leaf_id is not None])

    h_count = len(h_keys)
    p_count = len(p_keys)
    used_bytes = int(max_added_bytes) - remaining_bytes
    return h_count + p_count, {
        **summary,
        "h_selected": h_count,
        "p_selected": p_count,
        "predicted_added_bytes": used_bytes,
        "predicted_gain": predicted_gain,
        "h_added_bytes_per_action": int(h_added_bytes),
        "p_added_bytes_per_action": int(p_added_bytes),
        "selection": "greedy predicted Gauss-Newton gain per raw float16 packed byte",
    }


@torch.no_grad()
def _bernstein_haar_predicted_gain_leaf_score(
    model,
    split,
    *,
    level: int,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    ray_chunk: int,
    max_views: int | None,
    max_rays_per_view: int | None,
    weight_kwargs: dict,
    gradient_weight: float = 0.0,
    fisher_damping: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Predict h-split loss reduction in the seven-dimensional Haar detail space."""
    device = split.projections.device
    candidate_ids = torch.nonzero(model.leaf_levels == int(level), as_tuple=False).reshape(-1)
    if candidate_ids.numel() == 0:
        return candidate_ids, torch.empty((0,), dtype=torch.float32, device=device), {
            "haar_supported_leaves": 0,
            "residual_views": 0,
            "residual_rays": 0,
        }
    candidate_degrees = model.leaf_degrees[candidate_ids]
    if torch.any(candidate_degrees != 0):
        raise ValueError("Haar predicted gain currently requires degree-0 candidate leaves.")
    leaf_to_candidate = torch.full(
        (model.leaf_levels.numel(),),
        -1,
        dtype=torch.int32,
        device=device,
    )
    leaf_to_candidate[candidate_ids] = torch.arange(candidate_ids.numel(), dtype=torch.int32, device=device)
    gradient = torch.zeros((candidate_ids.numel(), 7), dtype=torch.float32, device=device)
    fisher = torch.zeros_like(gradient)

    rows_all, cols_all = detector_grid(detector_h, detector_w, device)
    if max_rays_per_view is not None and int(max_rays_per_view) < rows_all.numel():
        ids = torch.linspace(
            0,
            rows_all.numel() - 1,
            max(1, int(max_rays_per_view)),
            device=device,
        ).round().to(dtype=torch.long)
        rows_all = rows_all[ids]
        cols_all = cols_all[ids]
    all_view_count = int(split.angles.shape[0])
    if max_views is None or int(max_views) >= all_view_count:
        view_ids = torch.arange(all_view_count, device=device)
    else:
        view_ids = torch.linspace(
            0,
            all_view_count - 1,
            max(1, int(max_views)),
            device=device,
        ).round().long().unique(sorted=True)
    total_rays = 0
    ray_chunk = max(1, int(ray_chunk))
    for view_id_tensor in view_ids:
        view_id = int(view_id_tensor.item())
        angle = split.angles[view_id]
        gradient_values = None
        if float(gradient_weight) > 0.0:
            gradient_map = _projection_gradient_magnitude(split.projections[view_id])
            finite_gradient = gradient_map[torch.isfinite(gradient_map)]
            gradient_scale = (
                torch.quantile(finite_gradient, 0.95).clamp_min(1.0e-8)
                if finite_gradient.numel()
                else gradient_map.new_tensor(1.0)
            )
            gradient_values = torch.clamp(gradient_map / gradient_scale, 0.0, 1.0)
        for start in range(0, rows_all.numel(), ray_chunk):
            rows = rows_all[start : start + ray_chunk]
            cols = cols_all[start : start + ray_chunk]
            targets = split.projections[view_id, rows, cols]
            ray_batch = make_parallel_ray_points(
                angle.expand(rows.shape[0]),
                rows,
                cols,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                targets=targets,
                materialize_points=not (
                    hasattr(model, "prefer_compact_ray_batch") and model.prefer_compact_ray_batch()
                ),
            )
            ray_ids, leaf_ids, contributions = model.ray_cell_contributions(ray_batch)
            prediction = contributions.new_zeros((ray_batch.num_rays,))
            prediction.scatter_add_(0, ray_ids, contributions)
            residual = prediction - targets
            weights = projection_weights(targets, **weight_kwargs)
            if gradient_values is not None:
                weights = weights * (1.0 + float(gradient_weight) * gradient_values[rows, cols])
                weights = weights / weights.mean().clamp_min(float(weight_kwargs.get("epsilon", 1.0e-4)))

            candidate_positions = leaf_to_candidate[leaf_ids].to(dtype=torch.long)
            keep = candidate_positions >= 0
            if torch.any(keep):
                selected_rays = ray_ids[keep]
                selected_leaves = leaf_ids[keep]
                positions = candidate_positions[keep]
                features = _virtual_haar_ray_features(model, ray_batch, selected_rays, selected_leaves)
                weighted_residual = weights[selected_rays] * residual[selected_rays]
                selected_weights = weights[selected_rays]
                for detail in range(7):
                    phi = features[:, detail]
                    gradient[:, detail].scatter_add_(0, positions, weighted_residual * phi)
                    fisher[:, detail].scatter_add_(0, positions, selected_weights * torch.square(phi))
            total_rays += int(targets.numel())

    supported = torch.any(fisher > 0.0, dim=1)
    damping = max(float(fisher_damping), 0.0)
    score = 0.5 * torch.sum(torch.square(gradient) / (fisher + damping), dim=1)
    score = torch.where(supported & torch.isfinite(score), score, torch.zeros_like(score))
    finite_score = score[torch.isfinite(score)]
    return candidate_ids, score.detach(), {
        "haar_supported_leaves": int(torch.sum(supported).item()),
        "haar_score_mean": float(finite_score.mean().item()) if finite_score.numel() else 0.0,
        "haar_score_max": float(finite_score.max().item()) if finite_score.numel() else 0.0,
        "haar_fisher_damping": damping,
        "haar_ray_chunk": ray_chunk,
        "residual_views": int(view_ids.numel()),
        "residual_rays": total_rays,
    }


@torch.no_grad()
def _apply_h_gauss_newton_rate_distortion(
    model,
    split,
    *,
    level: int,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    ray_chunk: int,
    max_views: int | None,
    max_rays_per_view: int | None,
    weight_kwargs: dict,
    fisher_damping: float,
    max_added_bytes: int,
    h_added_bytes: int = 99,
) -> tuple[int, dict]:
    """Select h-splits by Haar-detail predicted gain under a byte budget."""
    candidate_ids, gains, summary = _bernstein_haar_predicted_gain_leaf_score(
        model,
        split,
        level=int(level),
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=ray_chunk,
        max_views=max_views,
        max_rays_per_view=max_rays_per_view,
        weight_kwargs=weight_kwargs,
        fisher_damping=fisher_damping,
    )
    byte_budget_count = max(0, int(max_added_bytes) // max(int(h_added_bytes), 1))
    leaf_budget_count = (
        max(0, (int(model.max_leaf_count) - int(model.leaf_levels.numel())) // 7)
        if model.max_leaf_count is not None
        else int(candidate_ids.numel())
    )
    count = min(int(candidate_ids.numel()), byte_budget_count, leaf_budget_count)
    finite_positive = torch.isfinite(gains) & (gains > 0.0)
    positive_ids = torch.nonzero(finite_positive, as_tuple=False).reshape(-1)
    count = min(count, int(positive_ids.numel()))
    if count > 0:
        selected_positions = positive_ids[
            torch.topk(gains[positive_ids], count, sorted=False).indices
        ]
        selected = candidate_ids[selected_positions]
        selected_gain = float(gains[selected_positions].sum().item())
        model.split_leaves_batch(selected)
    else:
        selected_gain = 0.0
    return count, {
        **summary,
        "h_selected": count,
        "predicted_added_bytes": count * int(h_added_bytes),
        "predicted_gain": selected_gain,
        "h_added_bytes_per_action": int(h_added_bytes),
        "selection": "Haar-detail predicted Gauss-Newton gain under raw float16 packed-byte budget",
    }


@torch.no_grad()
def _apply_h_jump_rate_distortion(
    model,
    *,
    max_added_bytes: int,
    h_added_bytes: int | None = None,
) -> tuple[int, dict]:
    """Globally split leaves by a scale-aware face-jump error indicator.

    |K| * sum_F [mu]_F^2 measures unresolved cross-cell structure for both
    p0 and p1 leaves. All levels compete in one queue; no support mask or
    level floor is used.
    """
    diagnostics = coefficient_diagnostics(model)
    level_shapes = _model_level_shapes(model)
    eligible = model.leaf_levels + 1 < len(level_shapes)
    candidate_ids = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    if candidate_ids.numel() == 0:
        return 0, {
            "candidate_count": 0,
            "h_selected": 0,
            "predicted_added_bytes": 0,
            "indicator_mean": 0.0,
            "indicator_max": 0.0,
        }
    shapes = torch.tensor(
        level_shapes,
        dtype=torch.float32,
        device=model.coefficient_logits.device,
    )[model.leaf_levels[candidate_ids]]
    cell_volume = torch.prod(2.0 / shapes, dim=1)
    jumps = diagnostics.interface_jump[candidate_ids].float()
    indicator = cell_volume * torch.sum(jumps.square(), dim=1)
    valid = torch.isfinite(indicator) & (indicator > 0.0)
    valid_positions = torch.nonzero(valid, as_tuple=False).reshape(-1)
    coefficient_counts = model.coefficient_counts()[candidate_ids].to(torch.long)
    # packed-v3 delta: parent leaf -> internal node plus eight child leaves.
    # Node slot=8, degree tag=3, float16 coefficient=2 bytes.
    dynamic_added_bytes = 85 + 14 * coefficient_counts
    if h_added_bytes is not None:
        dynamic_added_bytes = torch.full_like(dynamic_added_bytes, int(h_added_bytes))
    rates = indicator / dynamic_added_bytes.to(indicator.dtype)
    leaf_budget_count = (
        max(0, (int(model.max_leaf_count) - int(model.leaf_levels.numel())) // 7)
        if model.max_leaf_count is not None
        else int(candidate_ids.numel())
    )
    ranked_positions = valid_positions[
        torch.argsort(rates[valid_positions], descending=True)
    ]
    ranked_positions = ranked_positions[:leaf_budget_count]
    cumulative_bytes = torch.cumsum(dynamic_added_bytes[ranked_positions], dim=0)
    affordable = cumulative_bytes <= int(max_added_bytes)
    count = int(torch.sum(affordable).item())
    if count > 0:
        selected_positions = ranked_positions[:count]
        selected = candidate_ids[selected_positions]
        selected_indicator = float(indicator[selected_positions].sum().item())
        selected_levels = model.leaf_levels[selected]
        selected_by_level = [
            int(torch.sum(selected_levels == level).item())
            for level in range(len(level_shapes) - 1)
        ]
        model.split_leaves_batch(selected)
    else:
        selected_indicator = 0.0
        selected_by_level = [0 for _ in range(len(level_shapes) - 1)]
    finite = indicator[torch.isfinite(indicator)]
    return count, {
        "candidate_count": int(candidate_ids.numel()),
        "positive_indicator_count": int(valid_positions.numel()),
        "h_selected": count,
        "selected_by_parent_level": selected_by_level,
        "predicted_added_bytes": (
            int(dynamic_added_bytes[selected_positions].sum().item()) if count else 0
        ),
        "selected_indicator_sum": selected_indicator,
        "indicator_mean": float(finite.mean().item()) if finite.numel() else 0.0,
        "indicator_max": float(finite.max().item()) if finite.numel() else 0.0,
        "h_added_bytes_per_action": (
            int(h_added_bytes) if h_added_bytes is not None else "degree-dependent"
        ),
        "selection": "global volume-scaled face-jump indicator per raw float16 packed byte",
    }


@torch.no_grad()
def _bernstein_projection_residual_leaf_score(
    model,
    split,
    *,
    level: int | list[int] | tuple[int, ...],
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    ray_chunk: int,
    max_views: int | None,
    max_rays_per_view: int | None,
    weight_kwargs: dict,
    gradient_weight: float = 0.0,
    max_buffer_mb: float | None = None,
    score_mode: str = "attributed_energy",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Projection-domain residual attribution score for p-refinement.

    Each sampled ray residual is distributed to the active leaves it intersects
    in proportion to the leaf's absolute line-integral contribution.  This is
    a direct projection-domain signal; it never decodes the coefficient field
    to a dense volume.
    """
    device = split.projections.device
    requested_ray_chunk = int(ray_chunk)
    if max_buffer_mb is not None and hasattr(model, "level_resolutions"):
        levels_for_capacity = getattr(model, "level_shapes", model.level_resolutions)
        ray_chunk = min(
            requested_ray_chunk,
            bernstein_segment_ray_capacity(levels_for_capacity, float(max_buffer_mb)),
        )
    leaf_count = int(model.leaf_levels.shape[0])
    normalized_score_mode = str(score_mode).lower()
    squared_density_mode = normalized_score_mode in {
        "residual_squared_density",
        "squared_density",
    }
    energy = torch.zeros((leaf_count,), dtype=torch.float32, device=device)
    exposure = torch.zeros_like(energy)
    rows_all, cols_all = detector_grid(detector_h, detector_w, device)
    if max_rays_per_view is not None and int(max_rays_per_view) < rows_all.numel():
        ids = torch.linspace(
            0,
            rows_all.numel() - 1,
            max(1, int(max_rays_per_view)),
            device=device,
        ).round().to(dtype=torch.long)
        rows_all = rows_all[ids]
        cols_all = cols_all[ids]

    view_count = int(split.angles.shape[0])
    if max_views is not None:
        view_count = min(view_count, int(max_views))
    total_rays = 0
    for view_id in range(view_count):
        angle = split.angles[view_id]
        gradient_values = None
        if float(gradient_weight) > 0.0:
            gradient_map = _projection_gradient_magnitude(split.projections[view_id])
            finite_gradient = gradient_map[torch.isfinite(gradient_map)]
            gradient_scale = (
                torch.quantile(finite_gradient, 0.95).clamp_min(1e-8)
                if finite_gradient.numel()
                else gradient_map.new_tensor(1.0)
            )
            gradient_values = torch.clamp(gradient_map / gradient_scale, 0.0, 1.0)
        for start in range(0, rows_all.numel(), int(ray_chunk)):
            rows = rows_all[start : start + int(ray_chunk)]
            cols = cols_all[start : start + int(ray_chunk)]
            targets = split.projections[view_id, rows, cols]
            ray_batch = make_parallel_ray_points(
                angle.expand(rows.shape[0]),
                rows,
                cols,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                targets=targets,
                materialize_points=not (
                    hasattr(model, "prefer_compact_ray_batch") and model.prefer_compact_ray_batch()
                ),
            )
            prediction = model.integrate_ray_batch(ray_batch)
            residual = prediction - targets
            weights = projection_weights(targets, **weight_kwargs)
            if gradient_values is not None:
                weights = weights * (1.0 + float(gradient_weight) * gradient_values[rows, cols])
                weights = weights / weights.mean().clamp_min(float(weight_kwargs.get("epsilon", 1e-4)))
            if normalized_score_mode in {
                "residual_exposure",
                "exposure",
                "hk",
                "h_k",
                "residual_density",
                "density",
                "hk_density",
                "h_k_density",
                "residual_sqrt_exposure",
                "sqrt_exposure",
                "sqrt_density",
                "residual_squared_density",
                "squared_density",
            } and hasattr(model, "_parallel_ray_segments"):
                ray_ids, leaf_ids, intervals = model._parallel_ray_segments(ray_batch)
                segment_lengths = intervals[:, 1] - intervals[:, 0]
                ray_energy = weights * (
                    torch.square(residual) if squared_density_mode else torch.abs(residual)
                )
                energy.scatter_add_(0, leaf_ids, ray_energy[ray_ids] * segment_lengths)
                exposure.scatter_add_(0, leaf_ids, segment_lengths)
            else:
                ray_ids, leaf_ids, contributions = model.ray_cell_contributions(ray_batch)
                ray_energy = weights * torch.square(residual)
                abs_contrib = torch.abs(contributions)
                ray_exposure = contributions.new_zeros((ray_batch.num_rays,))
                ray_exposure.scatter_add_(0, ray_ids, abs_contrib)
                attributed = ray_energy[ray_ids] * abs_contrib / ray_exposure[ray_ids].clamp_min(1e-8)
                energy.scatter_add_(0, leaf_ids, attributed)
                exposure.scatter_add_(0, leaf_ids, abs_contrib)
            total_rays += int(targets.numel())

    target_levels = [int(level)] if isinstance(level, int) else [int(value) for value in level]
    target_tensor = torch.tensor(target_levels, dtype=model.leaf_levels.dtype, device=device)
    leaf_ids = torch.nonzero(torch.isin(model.leaf_levels, target_tensor), as_tuple=False).reshape(-1)
    if leaf_ids.numel() == 0:
        return leaf_ids, torch.empty((0,), dtype=torch.float32, device=device), {
            "projected_residual_mean": 0.0,
            "projected_residual_max": 0.0,
            "residual_level_leaf_count": 0,
            "residual_covered_count": 0,
            "residual_positive_count": 0,
            "residual_views": view_count,
            "residual_rays": total_rays,
        }
    if normalized_score_mode in {"residual_exposure", "exposure", "hk", "h_k"}:
        raw = energy[leaf_ids]
    elif normalized_score_mode in {"residual_sqrt_exposure", "sqrt_exposure", "sqrt_density"}:
        raw = energy[leaf_ids] / torch.sqrt(exposure[leaf_ids].clamp_min(1e-8))
    else:
        raw = energy[leaf_ids] / exposure[leaf_ids].clamp_min(1e-8)
    supported = exposure[leaf_ids] > 0
    scores = torch.where(supported, raw, torch.zeros_like(raw))
    finite_scores = scores[torch.isfinite(scores)]
    summary = {
        "projected_residual_mean": float(finite_scores.mean().item()) if finite_scores.numel() else 0.0,
        "projected_residual_max": float(finite_scores.max().item()) if finite_scores.numel() else 0.0,
        "residual_level_leaf_count": int(leaf_ids.numel()),
        "residual_covered_count": int(torch.sum(supported).item()),
        "residual_positive_count": int(torch.sum(torch.isfinite(scores) & (scores > 0)).item()),
        "residual_views": view_count,
        "residual_rays": total_rays,
        "residual_ray_chunk": int(ray_chunk),
        "residual_requested_ray_chunk": requested_ray_chunk,
        "residual_max_buffer_mb": float(max_buffer_mb) if max_buffer_mb is not None else None,
        "residual_score_mode": str(score_mode),
    }
    return leaf_ids, scores.detach().float(), summary


def _select_residual_h_candidates(
    model,
    leaf_ids: torch.Tensor,
    scores: torch.Tensor,
    *,
    active_fraction: float,
    threshold_statistic: str | None,
    threshold_multiplier: float,
    max_fraction: float,
    reserve_leaf_count: int = 0,
    min_mu_threshold: float | None = None,
) -> tuple[torch.Tensor, dict]:
    if leaf_ids.numel() == 0:
        return leaf_ids, {"selection_threshold": None, "selection_cap": 0}
    finite = torch.isfinite(scores)
    safe_scores = torch.where(finite, scores, torch.full_like(scores, -float("inf")))
    top_count = max(1, int(leaf_ids.numel() * float(active_fraction)))
    top_positions = torch.topk(safe_scores, min(top_count, leaf_ids.numel())).indices
    selected_mask = torch.zeros_like(finite)
    selected_mask[top_positions] = True
    material_mask = torch.zeros_like(selected_mask)
    leaf_mu = None
    if min_mu_threshold is not None:
        leaf_mu = model._leaf_mu()[leaf_ids].detach().float()
        material_mask = torch.isfinite(leaf_mu) & (leaf_mu >= float(min_mu_threshold))
        selected_mask |= material_mask
    threshold = None
    normalized_statistic = str(threshold_statistic or "").lower()
    positive = scores[finite & (scores > 0)]
    if positive.numel() and normalized_statistic in {"q95", "p95", "quantile95"}:
        threshold = float(torch.quantile(positive, 0.95).item()) * float(threshold_multiplier)
        selected_mask |= finite & (scores >= threshold)
    selected_positions = torch.nonzero(selected_mask, as_tuple=False).reshape(-1)
    cap = max(0, int(leaf_ids.numel() * float(max_fraction)))
    if model.max_leaf_count is not None:
        cap = min(
            cap,
            max(
                0,
                (
                    int(model.max_leaf_count)
                    - int(model.leaf_levels.shape[0])
                    - max(0, int(reserve_leaf_count))
                )
                // 7,
            ),
        )
    if selected_positions.numel() > cap:
        material_positions = torch.nonzero(material_mask, as_tuple=False).reshape(-1)
        if material_positions.numel() >= cap:
            selected_positions = (
                material_positions[torch.topk(leaf_mu[material_positions], cap).indices]
                if cap > 0 and leaf_mu is not None
                else material_positions[:0]
            )
        else:
            selected_positions = material_positions
            remaining = cap - int(selected_positions.numel())
            fill_positions = torch.nonzero(selected_mask & ~material_mask, as_tuple=False).reshape(-1)
            if remaining > 0 and fill_positions.numel():
                order = torch.topk(safe_scores[fill_positions], min(remaining, fill_positions.numel())).indices
                selected_positions = torch.cat([selected_positions, fill_positions[order]])
    elif selected_positions.numel():
        selected_positions = selected_positions[torch.argsort(safe_scores[selected_positions], descending=True)]
    return leaf_ids[selected_positions], {
        "selection_threshold": threshold,
        "selection_cap": cap,
        "top_fraction_count": top_count,
        "threshold_union_count": int(torch.sum(selected_mask).item()),
        "material_threshold": float(min_mu_threshold) if min_mu_threshold is not None else None,
        "material_threshold_count": int(torch.sum(material_mask).item()),
    }


def _select_global_residual_h_candidates(
    model,
    leaf_ids: torch.Tensor,
    scores: torch.Tensor,
    *,
    max_candidates: int,
    reserve_leaf_count: int = 0,
    min_mu_threshold: float | None = None,
) -> tuple[torch.Tensor, dict]:
    if leaf_ids.numel() == 0:
        return leaf_ids, {
            "global_candidate_cap": 0,
            "global_shortlisted_by_level": {},
            "proposal_accuracy_score_mean": 0.0,
            "proposal_accuracy_score_max": 0.0,
        }
    finite = torch.isfinite(scores)
    accuracy_scores = torch.where(
        finite,
        scores,
        torch.full_like(scores, -float("inf")),
    )
    selected_mask = finite & (accuracy_scores > 0)
    material_mask = torch.zeros_like(selected_mask)
    leaf_mu = None
    if min_mu_threshold is not None:
        leaf_mu = model._leaf_mu()[leaf_ids].detach().float()
        material_mask = torch.isfinite(leaf_mu) & (leaf_mu >= float(min_mu_threshold))
        selected_mask |= material_mask
    positions = torch.nonzero(selected_mask, as_tuple=False).reshape(-1)
    cap = max(0, int(max_candidates))
    if model.max_leaf_count is not None:
        cap = min(
            cap,
            max(0, (int(model.max_leaf_count) - int(model.leaf_levels.shape[0]) - max(0, int(reserve_leaf_count))) // 7),
        )
    if positions.numel() > cap:
        order = torch.topk(accuracy_scores[positions], cap).indices if cap > 0 else torch.empty((0,), dtype=torch.long, device=positions.device)
        positions = positions[order]
    elif positions.numel():
        positions = positions[torch.argsort(accuracy_scores[positions], descending=True)]
    selected = leaf_ids[positions]
    selected_levels = model.leaf_levels[selected].detach().cpu().tolist() if selected.numel() else []
    shortlisted_by_level: dict[str, int] = {}
    for value in selected_levels:
        key = str(int(value))
        shortlisted_by_level[key] = shortlisted_by_level.get(key, 0) + 1
    finite_selected = accuracy_scores[positions][torch.isfinite(accuracy_scores[positions])] if positions.numel() else accuracy_scores[:0]
    return selected, {
        "global_candidate_cap": cap,
        "global_shortlisted_by_level": shortlisted_by_level,
        "proposal_accuracy_score_mean": float(finite_selected.mean().item()) if finite_selected.numel() else 0.0,
        "proposal_accuracy_score_max": float(finite_selected.max().item()) if finite_selected.numel() else 0.0,
        "material_threshold": float(min_mu_threshold) if min_mu_threshold is not None else None,
        "material_threshold_count": int(torch.sum(material_mask).item()),
    }


def _select_p_residual_candidates(
    leaf_ids: torch.Tensor,
    scores: torch.Tensor,
    *,
    candidate_fraction: float,
    residual_quantile_range: list[float] | tuple[float, float] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Select p candidates either from the top tail or a positive residual band.

    A mid-quantile band is intended for coarse leaves whose constant model is
    insufficient but whose residual is not large enough to justify another
    spatial split. Quantiles are computed only over finite positive scores so
    unobserved/zero-residual leaves cannot enter the band.
    """

    if leaf_ids.numel() == 0:
        return leaf_ids, {
            "p_candidate_selection": "empty",
            "p_candidate_count": 0,
        }
    finite_positive = torch.isfinite(scores) & (scores > 0)
    if residual_quantile_range is not None:
        if len(residual_quantile_range) != 2:
            raise ValueError("residual_quantile_range must contain [low, high]")
        low_q, high_q = (float(value) for value in residual_quantile_range)
        if not 0.0 <= low_q < high_q <= 1.0:
            raise ValueError("residual_quantile_range must satisfy 0 <= low < high <= 1")
        positive_scores = scores[finite_positive]
        if positive_scores.numel() == 0:
            return leaf_ids[:0], {
                "p_candidate_selection": "residual_quantile_band",
                "p_candidate_quantiles": [low_q, high_q],
                "p_candidate_score_range": [None, None],
                "p_candidate_count": 0,
                "p_positive_score_count": 0,
            }
        low = torch.quantile(positive_scores, low_q)
        high = torch.quantile(positive_scores, high_q)
        selected = finite_positive & (scores >= low) & (scores <= high)
        selected_ids = leaf_ids[selected]
        return selected_ids, {
            "p_candidate_selection": "residual_quantile_band",
            "p_candidate_quantiles": [low_q, high_q],
            "p_candidate_score_range": [float(low.item()), float(high.item())],
            "p_candidate_count": int(selected_ids.numel()),
            "p_positive_score_count": int(positive_scores.numel()),
        }

    safe_scores = torch.where(
        torch.isfinite(scores),
        scores,
        torch.full_like(scores, -float("inf")),
    )
    candidate_count = max(1, int(leaf_ids.numel() * float(candidate_fraction)))
    selected_positions = torch.topk(safe_scores, min(candidate_count, leaf_ids.numel())).indices
    selected_ids = leaf_ids[selected_positions]
    return selected_ids, {
        "p_candidate_selection": "top_fraction",
        "p_candidate_fraction": float(candidate_fraction),
        "p_candidate_count": int(selected_ids.numel()),
        "p_positive_score_count": int(finite_positive.sum().item()),
    }


def _axis_degree_elevation_matrix(old_degree: int, target_degree: int, *, device, dtype) -> torch.Tensor:
    if target_degree < old_degree:
        raise ValueError("target degree must not be lower than the current degree")
    matrix = torch.eye(old_degree + 1, dtype=dtype, device=device)
    for degree in range(old_degree, target_degree):
        step = torch.zeros((degree + 2, degree + 1), dtype=dtype, device=device)
        step[0, 0] = 1.0
        step[-1, -1] = 1.0
        if degree > 0:
            ids = torch.arange(1, degree + 1, dtype=dtype, device=device)
            alpha = ids / float(degree + 1)
            step[ids.long(), ids.long() - 1] = alpha
            step[ids.long(), ids.long()] = 1.0 - alpha
        matrix = step @ matrix
    return matrix


def _full_degree_elevation_matrix(
    old_degree: tuple[int, int, int],
    target_degree: tuple[int, int, int],
    *,
    device,
    dtype,
) -> torch.Tensor:
    matrices = [
        _axis_degree_elevation_matrix(old, target, device=device, dtype=dtype)
        for old, target in zip(old_degree, target_degree)
    ]
    return torch.kron(torch.kron(matrices[0], matrices[1]), matrices[2])


def _stratified_training_ray_batch(
    split,
    *,
    batch_index: int,
    batch_rays: int,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    materialize_points: bool,
):
    device = split.projections.device
    num_views = int(split.angles.shape[0])
    ids = torch.arange(batch_rays, device=device, dtype=torch.long)
    view_ids = torch.remainder(ids + batch_index, num_views)
    generator = torch.Generator(device=device)
    generator.manual_seed(0x5A17 + int(batch_index))
    rows = torch.randint(0, detector_h, (batch_rays,), generator=generator, device=device)
    cols = torch.randint(0, detector_w, (batch_rays,), generator=generator, device=device)
    targets = split.projections[view_ids, rows, cols]
    return make_parallel_ray_points(
        split.angles[view_ids],
        rows,
        cols,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        targets=targets,
        materialize_points=materialize_points,
    )


def _apply_gradient_rate_distortion_p_refinement(
    model,
    candidate_leaf_ids: torch.Tensor,
    target_degree,
    *,
    training_split,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    gradient_batches: int,
    batch_rays: int,
    active_fraction: float,
    eligible_leaf_count: int,
    max_added_coefficients: int,
    learning_rate: float,
    weight_kwargs: dict,
    projection_gradient_cfg: dict | None = None,
    iteration: int = 1,
) -> tuple[int, dict]:
    if isinstance(target_degree, int):
        target = (int(target_degree),) * 3
    else:
        target = tuple(int(value) for value in target_degree)
    if len(target) != 3:
        raise ValueError("target_degree must contain three values")
    target_tensor = torch.tensor(target, dtype=torch.long, device=model.coefficient_logits.device)
    current = model.leaf_degrees[candidate_leaf_ids]
    candidate_leaf_ids = candidate_leaf_ids[torch.any(current < target_tensor[None, :], dim=1)]
    if candidate_leaf_ids.numel() == 0:
        return 0, {"strategy": "gradient_rate_distortion", "candidate_active": 0, "accepted": False}

    before_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    old_degrees = model.leaf_degrees[candidate_leaf_ids].detach().clone()
    old_counts = torch.prod(old_degrees + 1, dim=1)
    old_leaf_count = int(model.leaf_levels.shape[0])
    change = model.elevate_leaves_isotropic_batch(candidate_leaf_ids, target)
    trial_count = int(len(change.new_leaf_keys))
    if trial_count == 0:
        return 0, {"strategy": "gradient_rate_distortion", "candidate_active": 0, "accepted": False}
    trial_leaf_ids = torch.arange(
        int(model.leaf_levels.shape[0]) - trial_count,
        int(model.leaf_levels.shape[0]),
        dtype=torch.long,
        device=model.coefficient_logits.device,
    )
    model.zero_grad(set_to_none=True)
    materialize_points = not model.prefer_compact_ray_batch()
    batches = max(1, int(gradient_batches))
    gradient_cfg = projection_gradient_cfg or {}
    projection_gradient_weight = _projection_gradient_effective_weight(gradient_cfg, iteration)
    for batch_index in range(batches):
        ray_batch = _stratified_training_ray_batch(
            training_split,
            batch_index=batch_index,
            batch_rays=int(batch_rays),
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            materialize_points=materialize_points,
        )
        prediction = model.integrate_ray_batch(ray_batch)
        loss = weighted_projection_loss(prediction, ray_batch.target, **weight_kwargs)
        if projection_gradient_weight > 0.0:
            gradient_generator = torch.Generator(device=training_split.projections.device)
            gradient_generator.manual_seed(0x6D2B + int(batch_index))
            gradient_batch = _sample_projection_gradient_batch(
                training_split,
                gradient_cfg,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                materialize_points=materialize_points,
                generator=gradient_generator,
            )
            gradient_prediction = model.integrate_ray_batch(gradient_batch.ray_batch)
            loss = loss + projection_gradient_weight * weighted_projection_gradient_loss(
                gradient_prediction,
                gradient_batch,
                **weight_kwargs,
                **_projection_gradient_loss_kwargs(gradient_cfg),
            )
        (loss / float(batches)).backward()

    logit_gradient = model.coefficient_logits.grad.detach()
    softplus_jacobian = torch.sigmoid(
        model.coefficient_logits.detach() + float(model.attenuation_shift)
    ).clamp_min(1.0e-8)
    gradient = logit_gradient / softplus_jacobian
    scores = torch.zeros((trial_count,), dtype=torch.float32, device=gradient.device)
    added_counts = torch.zeros((trial_count,), dtype=torch.long, device=gradient.device)
    trial_degrees = model.leaf_degrees[trial_leaf_ids]
    for old_degree_tensor in unique_degree_rows(old_degrees):
        group = torch.all(old_degrees == old_degree_tensor, dim=1)
        positions = torch.nonzero(group, as_tuple=False).reshape(-1)
        if positions.numel() == 0:
            continue
        new_degree = tuple(int(value) for value in trial_degrees[positions[0]].tolist())
        old_degree = tuple(int(value) for value in old_degree_tensor.tolist())
        coefficient_count = int(torch.prod(trial_degrees[positions[0]] + 1).item())
        offsets = torch.arange(coefficient_count, dtype=torch.long, device=gradient.device)
        coefficient_ids = model.coefficient_offsets[trial_leaf_ids[positions], None] + offsets[None, :]
        blocks = gradient[coefficient_ids]
        elevation = _full_degree_elevation_matrix(
            old_degree,
            new_degree,
            device=gradient.device,
            dtype=gradient.dtype,
        )
        old_basis, _ = torch.linalg.qr(elevation, mode="reduced")
        old_energy = torch.sum(torch.square(blocks @ old_basis), dim=1)
        new_energy = torch.clamp(torch.sum(torch.square(blocks), dim=1) - old_energy, min=0.0)
        delta = coefficient_count - old_counts[positions]
        added_counts[positions] = delta
        delta_bytes = (2 * delta).clamp_min(1).to(dtype=new_energy.dtype)
        scores[positions] = float(learning_rate) * new_energy / delta_bytes

    max_leaves = min(
        trial_count,
        max(1, int(int(eligible_leaf_count) * float(active_fraction))),
    )
    order = torch.argsort(scores, descending=True)
    order = order[scores[order] > 0]
    cumulative = torch.cumsum(added_counts[order], dim=0)
    keep = cumulative <= int(max_added_coefficients)
    positions = order[keep][:max_leaves]
    selected_original_ids = candidate_leaf_ids[positions].detach().cpu()
    selected_scores = scores[positions].detach()

    model.prepare_sparse_from_state_dict(before_state)
    model.load_state_dict(before_state, strict=False)
    selected_original_ids = selected_original_ids.to(device=model.coefficient_logits.device)
    accepted_change = model.elevate_leaves_isotropic_batch(selected_original_ids, target)
    accepted = int(len(accepted_change.new_leaf_keys))
    return accepted, {
        "strategy": "gradient_rate_distortion",
        "candidate_active": trial_count,
        "active": accepted,
        "accepted": bool(accepted),
        "gradient_batches": batches,
        "eligible_leaf_count": int(eligible_leaf_count),
        "max_added_coefficients": int(max_added_coefficients),
        "added_coefficients": int(accepted_change.new_coefficient_count - accepted_change.old_coefficient_count),
        "score_mean": float(scores.mean().item()) if scores.numel() else 0.0,
        "score_max": float(scores.max().item()) if scores.numel() else 0.0,
        "accepted_score_min": float(selected_scores.min().item()) if selected_scores.numel() else 0.0,
        "rate_bytes_per_coefficient": 2,
        "gradient_space": "physical_bernstein_coefficients",
        "projection_gradient_weight": projection_gradient_weight,
        "old_leaf_count": old_leaf_count,
    }


def _clone_sparse_state(model) -> dict[str, torch.Tensor]:
    """Clone the sparse Bernstein state so a rejected trial can be rolled back."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _restore_sparse_state(model, state_dict: dict[str, torch.Tensor]) -> None:
    model.prepare_sparse_from_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)


def _select_acceptance_splits(dataset, source: str, *, validation_fraction: float, seed: int):
    normalized = str(source).lower()
    if normalized in {"train_holdout", "train_heldout", "heldout", "held_out"}:
        return split_projection_views(
            dataset.train,
            validation_fraction=float(validation_fraction),
            seed=int(seed),
        )
    if normalized in {"test", "projection_test"}:
        raise ValueError("test views are evaluation-only and cannot gate p-refinement")
    if normalized in {"train", "training"}:
        return dataset.train, dataset.train
    raise ValueError(
        "p-refinement acceptance validation_source must be "
        "'train_holdout' or 'train'; test views are evaluation-only."
    )


@torch.no_grad()
def _sampled_projection_loss(
    model,
    split,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    ray_chunk: int,
    max_views: int | None,
    max_rays_per_view: int | None,
    weight_kwargs: dict,
) -> float:
    """Weighted projection loss on a deterministic subset of rays.

    This is a projection-domain validation gate; it does not decode or inspect a
    dense volume.  It intentionally samples fixed detector coordinates so a
    before/after topology trial is compared on exactly the same rays.
    """
    device = split.projections.device
    rows_all, cols_all = detector_grid(detector_h, detector_w, device)
    if max_rays_per_view is not None and int(max_rays_per_view) < rows_all.numel():
        count = max(1, int(max_rays_per_view))
        ids = torch.linspace(0, rows_all.numel() - 1, count, device=device).round().to(dtype=torch.long)
        rows_all = rows_all[ids]
        cols_all = cols_all[ids]

    view_count = int(split.angles.shape[0])
    if max_views is not None:
        view_count = min(view_count, int(max_views))
    total = 0.0
    total_rays = 0
    for view_id in range(view_count):
        angle = split.angles[view_id]
        for start in range(0, rows_all.numel(), int(ray_chunk)):
            rows = rows_all[start : start + int(ray_chunk)]
            cols = cols_all[start : start + int(ray_chunk)]
            targets = split.projections[view_id, rows, cols]
            ray_batch = make_parallel_ray_points(
                angle.expand(rows.shape[0]),
                rows,
                cols,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                targets=targets,
                materialize_points=not (
                    hasattr(model, "prefer_compact_ray_batch") and model.prefer_compact_ray_batch()
                ),
            )
            prediction = model.integrate_ray_batch(ray_batch)
            loss = weighted_projection_loss(prediction, targets, **weight_kwargs)
            total += float(loss.detach().item()) * int(targets.numel())
            total_rays += int(targets.numel())
    return total / max(total_rays, 1)


@torch.no_grad()
def _estimate_projection_noise_variance(split, config: dict) -> dict:
    flat = split.projections.detach().float().abs().cpu().numpy().reshape(-1)
    max_samples = int(config.get("max_samples", 1048576))
    if flat.size > max_samples:
        ids = np.linspace(0, flat.size - 1, max_samples).round().astype(np.int64)
        flat = flat[ids]
    quantile = float(config.get("air_quantile", 0.10))
    cutoff = float(np.quantile(flat, min(1.0, max(0.0, quantile))))
    air = flat[flat <= cutoff]
    if air.size < 2:
        air = flat
    variance = max(float(np.var(air)), float(config.get("min_variance", 1.0e-8)))
    return {
        "enabled": True,
        "source": "low_attenuation_training_detector_pixels",
        "air_quantile": quantile,
        "air_cutoff": cutoff,
        "sample_count": int(air.size),
        "variance": variance,
        "frozen": True,
    }


def _compact_raw_payload_bytes(model, *, quantization: str = "float16") -> int:
    bytes_per_float = 4 if str(quantization).lower() == "float32" else 2
    leaf_count = int(model.leaf_levels.shape[0])
    node_count = int(getattr(model, "node_child_base", model.leaf_levels).numel())
    level_shape_count = len(getattr(model, "level_shapes", getattr(model, "level_resolutions", [])))
    total = node_count * 4 * 2  # node_child_base + node_leaf_id, int32
    total += leaf_count * 3  # leaf_degrees, uint8
    total += 4  # leaf_count, uint32
    total += max(level_shape_count, 1) * 3 * 2  # level_shapes, uint16
    if getattr(model, "representation", "") == "latent_p1_octree":
        total += 3 + 4  # max_degree uint8 + attenuation_shift float32
        total += int(model.leaf_latents.numel()) * bytes_per_float
        total += int(model.decoder.weight.numel() + model.decoder.bias.numel()) * bytes_per_float
        return int(total)
    total += 3 + 4
    total += int(model.coefficients().numel()) * bytes_per_float
    return int(total)


def _sample_fit_decision_rays(
    split,
    *,
    ray_count: int,
    fit_fraction: float,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    seed: int,
    chunk_rays: int,
    materialize_points: bool,
) -> tuple[FixedDecisionRaySet, FixedDecisionRaySet]:
    device = split.projections.device
    total = int(split.angles.numel()) * int(detector_h) * int(detector_w)
    count = min(max(2, int(ray_count)), total)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    ids = torch.randperm(total, generator=generator)[:count].to(device=device)
    fit_count = max(1, min(count - 1, int(round(count * float(fit_fraction)))))
    pixels = int(detector_h) * int(detector_w)

    def build(selected: torch.Tensor) -> FixedDecisionRaySet:
        view_ids = torch.div(selected, pixels, rounding_mode="floor")
        rem = selected - view_ids * pixels
        rows = torch.div(rem, int(detector_w), rounding_mode="floor")
        cols = rem - rows * int(detector_w)
        return FixedDecisionRaySet(
            split=split,
            view_ids=view_ids,
            rows=rows,
            cols=cols,
            detector_h=int(detector_h),
            detector_w=int(detector_w),
            samples_per_ray=int(samples_per_ray),
            chunk_rays=max(1, int(chunk_rays)),
            materialize_points=bool(materialize_points),
        )

    return build(ids[:fit_count]), build(ids[fit_count:])


def _fixed_ray_batch(rays: FixedDecisionRaySet, indices: torch.Tensor):
    view_ids = rays.view_ids[indices]
    rows = rays.rows[indices]
    cols = rays.cols[indices]
    targets = rays.split.projections[view_ids, rows, cols]
    return make_parallel_ray_points(
        rays.split.angles[view_ids],
        rows,
        cols,
        detector_h=rays.detector_h,
        detector_w=rays.detector_w,
        samples_per_ray=rays.samples_per_ray,
        targets=targets,
        materialize_points=rays.materialize_points,
    )


@torch.no_grad()
def _rays_intersecting_leaf_keys(model, rays: FixedDecisionRaySet, leaf_keys) -> FixedDecisionRaySet:
    leaf_ids = [model.find_leaf(key) for key in leaf_keys]
    leaf_ids = [int(leaf_id) for leaf_id in leaf_ids if leaf_id is not None]
    if not leaf_ids:
        return FixedDecisionRaySet(
            split=rays.split,
            view_ids=rays.view_ids[:0],
            rows=rays.rows[:0],
            cols=rays.cols[:0],
            detector_h=rays.detector_h,
            detector_w=rays.detector_w,
            samples_per_ray=rays.samples_per_ray,
            chunk_rays=rays.chunk_rays,
            materialize_points=rays.materialize_points,
        )
    wanted = torch.tensor(leaf_ids, dtype=torch.long, device=rays.view_ids.device)
    selected_chunks = []
    for start in range(0, rays.observation_count, rays.chunk_rays):
        stop = min(start + rays.chunk_rays, rays.observation_count)
        indices = torch.arange(start, stop, dtype=torch.long, device=rays.view_ids.device)
        ray_batch = _fixed_ray_batch(rays, indices)
        ray_ids, hit_leaf_ids, _ = model.ray_cell_contributions(ray_batch)
        hit = torch.isin(hit_leaf_ids, wanted)
        if torch.any(hit):
            selected_chunks.append(indices[torch.unique(ray_ids[hit])])
    selected = torch.cat(selected_chunks) if selected_chunks else rays.view_ids[:0]
    return FixedDecisionRaySet(
        split=rays.split,
        view_ids=rays.view_ids[selected],
        rows=rays.rows[selected],
        cols=rays.cols[selected],
        detector_h=rays.detector_h,
        detector_w=rays.detector_w,
        samples_per_ray=rays.samples_per_ray,
        chunk_rays=rays.chunk_rays,
        materialize_points=rays.materialize_points,
    )


@torch.no_grad()
def _fixed_ray_distortion(
    model,
    rays: FixedDecisionRaySet,
    *,
    weight_kwargs: dict,
    noise_variance: float,
) -> dict:
    weighted_rss = 0.0
    rss = 0.0
    count = 0
    for start in range(0, rays.observation_count, rays.chunk_rays):
        stop = min(start + rays.chunk_rays, rays.observation_count)
        indices = torch.arange(start, stop, dtype=torch.long, device=rays.view_ids.device)
        ray_batch = _fixed_ray_batch(rays, indices)
        residual = model.integrate_ray_batch(ray_batch) - ray_batch.target
        weights = projection_weights(ray_batch.target, **weight_kwargs)
        weighted_rss += float(torch.sum(weights * torch.square(residual)).item())
        rss += float(torch.sum(torch.square(residual)).item())
        count += int(residual.numel())
    variance = max(float(noise_variance), 1.0e-12)
    return {
        "ray_count": count,
        "rss": rss,
        "weighted_rss": weighted_rss,
        "noise_normalized_rss": weighted_rss / variance,
        "distortion": weighted_rss / max(count, 1) / variance,
    }


def _tune_affected_on_fixed_rays(
    model,
    rays: FixedDecisionRaySet,
    affected_leaf_keys,
    *,
    steps: int,
    batch_rays: int,
    learning_rate: float,
    weight_kwargs: dict,
) -> None:
    affected_leaf_ids = [model.find_leaf(key) for key in affected_leaf_keys]
    affected_leaf_ids = [int(leaf_id) for leaf_id in affected_leaf_ids if leaf_id is not None]
    if int(steps) <= 0 or not affected_leaf_ids or rays.observation_count <= 0:
        return
    batch_count = max(1, min(int(batch_rays), rays.observation_count))
    if getattr(model, "representation", "") == "latent_p1_octree":
        trainable_mask = torch.zeros((model.leaf_latents.shape[0],), dtype=torch.bool, device=model.leaf_latents.device)
        trainable_mask[torch.tensor(affected_leaf_ids, dtype=torch.long, device=model.leaf_latents.device)] = True
        old_requires_grad = [param.requires_grad for param in model.decoder.parameters()]
        for param in model.decoder.parameters():
            param.requires_grad_(False)
        optimizer = torch.optim.Adam([model.leaf_latents], lr=float(learning_rate))
        try:
            for step in range(int(steps)):
                start = (step * batch_count) % rays.observation_count
                indices = (torch.arange(batch_count, device=rays.view_ids.device) + start) % rays.observation_count
                ray_batch = _fixed_ray_batch(rays, indices)
                loss = weighted_projection_loss(model.integrate_ray_batch(ray_batch), ray_batch.target, **weight_kwargs)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if model.leaf_latents.grad is not None:
                    model.leaf_latents.grad[~trainable_mask] = 0.0
                optimizer.step()
        finally:
            for param, requires_grad in zip(model.decoder.parameters(), old_requires_grad):
                param.requires_grad_(requires_grad)
        return

    ids = torch.tensor(affected_leaf_ids, dtype=torch.long, device=model.coefficient_logits.device)
    trainable_mask = torch.isin(model.coefficient_leaf_ids, ids)
    optimizer = torch.optim.Adam([model.coefficient_logits], lr=float(learning_rate))
    for step in range(int(steps)):
        start = (step * batch_count) % rays.observation_count
        indices = (torch.arange(batch_count, device=rays.view_ids.device) + start) % rays.observation_count
        ray_batch = _fixed_ray_batch(rays, indices)
        loss = weighted_projection_loss(model.integrate_ray_batch(ray_batch), ray_batch.target, **weight_kwargs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if model.coefficient_logits.grad is not None:
            model.coefficient_logits.grad[~trainable_mask] = 0.0
        optimizer.step()


def _apply_h_split_with_crossfit_gate(
    model,
    selected: torch.Tensor,
    *,
    dataset,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    spec: dict,
    adaptive_cfg: dict,
    train_cfg: dict,
    iteration: int,
    materialize_ray_points: bool,
    noise_variance: float,
) -> tuple[int, dict]:
    gate_cfg = spec.get("crossfit_acceptance") or spec.get("acceptance") or {}
    max_candidates = int(gate_cfg.get("max_candidates", selected.numel()))
    candidate_keys = [model.leaf_key(int(leaf_id)) for leaf_id in selected[:max_candidates].detach().cpu().tolist()]
    if not candidate_keys:
        return 0, {"crossfit_enabled": True, "candidate_count": 0, "accepted_count": 0}
    fit_rays, decision_rays = _sample_fit_decision_rays(
        dataset.train,
        ray_count=int(gate_cfg.get("ray_count", 8192)),
        fit_fraction=float(gate_cfg.get("fit_fraction", 0.8)),
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        seed=int(gate_cfg.get("seed", iteration)),
        chunk_rays=int(gate_cfg.get("ray_chunk", train_cfg.get("eval_ray_chunk", 4096))),
        materialize_points=materialize_ray_points,
    )
    weight_kwargs = _projection_weight_kwargs(
        train_cfg,
        adaptive_cfg,
        gate_cfg.get("projection_weighting"),
        iteration=iteration,
    )
    min_gain = float(gate_cfg.get("min_gain", 0.0))
    quantization = str(gate_cfg.get("compact_quantization", (train_cfg.get("compact_quantization") or "float16")))
    acceptance_mode = str(gate_cfg.get("acceptance_mode", "sequential")).lower()
    byte_budget = gate_cfg.get("byte_budget", gate_cfg.get("max_total_bytes", spec.get("byte_budget")))
    evaluations = []
    accepted = 0
    if acceptance_mode in {"ranked_batch", "ranked", "batch"}:
        base_state = _clone_sparse_state(model)
        base_bytes = _compact_raw_payload_bytes(model, quantization=quantization)
        for key in candidate_keys:
            _restore_sparse_state(model, base_state)
            leaf_id = model.find_leaf(key)
            if leaf_id is None:
                continue
            candidate_fit_rays = _rays_intersecting_leaf_keys(model, fit_rays, [key])
            candidate_decision_rays = _rays_intersecting_leaf_keys(model, decision_rays, [key])
            if candidate_decision_rays.observation_count == 0:
                evaluations.append(
                    {
                        "leaf_key": list(key),
                        "candidate_active": 0,
                        "fit_ray_count_for_leaf": int(candidate_fit_rays.observation_count),
                        "decision_ray_count_for_leaf": 0,
                        "decision_before": {"ray_count": 0},
                        "decision_after": {"ray_count": 0},
                        "decision_gain_noise_normalized_rss": 0.0,
                        "rate_delta_bytes": 0,
                        "passed_gate": False,
                        "accepted": False,
                        "reason": "rejected: no decision rays intersect candidate",
                    }
                )
                continue
            before = _fixed_ray_distortion(
                model,
                candidate_decision_rays,
                weight_kwargs=weight_kwargs,
                noise_variance=noise_variance,
            )
            change = model.split_leaves_batch([leaf_id])
            candidate_active = int(len(change.new_leaf_keys))
            if candidate_active:
                _tune_affected_on_fixed_rays(
                    model,
                    candidate_fit_rays,
                    change.new_leaf_keys,
                    steps=int(gate_cfg.get("tune_steps", 30)),
                    batch_rays=int(gate_cfg.get("tune_batch_rays", 2048)),
                    learning_rate=float(gate_cfg.get("tune_learning_rate", train_cfg.get("lr", 1.0e-2))),
                    weight_kwargs=weight_kwargs,
                )
            after = _fixed_ray_distortion(
                model,
                candidate_decision_rays,
                weight_kwargs=weight_kwargs,
                noise_variance=noise_variance,
            )
            after_bytes = _compact_raw_payload_bytes(model, quantization=quantization)
            rate_delta = after_bytes - base_bytes
            gain = float(before["noise_normalized_rss"] - after["noise_normalized_rss"])
            passed = candidate_active > 0 and rate_delta > 0 and gain > min_gain
            evaluations.append(
                {
                    "leaf_key": list(key),
                    "candidate_active": candidate_active,
                    "fit_ray_count_for_leaf": int(candidate_fit_rays.observation_count),
                    "decision_ray_count_for_leaf": int(candidate_decision_rays.observation_count),
                    "decision_before": before,
                    "decision_after": after,
                    "decision_gain_noise_normalized_rss": gain,
                    "rate_delta_bytes": int(rate_delta),
                    "passed_gate": bool(passed),
                    "accepted": False,
                    "reason": (
                        "passed gate; waiting for ranked batch selection"
                        if passed
                        else "rejected: no held-out decision accuracy gain"
                    ),
                }
            )

        _restore_sparse_state(model, base_state)
        passed_evaluations = [item for item in evaluations if item.get("passed_gate")]
        passed_evaluations.sort(
            key=lambda item: float(item.get("decision_gain_noise_normalized_rss", 0.0)),
            reverse=True,
        )
        max_splits = int(gate_cfg.get("max_splits_per_round", gate_cfg.get("max_splits", len(passed_evaluations))))
        max_added_bytes = gate_cfg.get("max_added_bytes")
        remaining_added = int(max_added_bytes) if max_added_bytes is not None else None
        if byte_budget is not None:
            budget_remaining = max(0, int(byte_budget) - base_bytes)
            remaining_added = budget_remaining if remaining_added is None else min(remaining_added, budget_remaining)
        selected_keys = []
        for item in passed_evaluations:
            if len(selected_keys) >= max_splits:
                item["reason"] = "passed gate; skipped by per-round split cap"
                continue
            rate_delta = int(item.get("rate_delta_bytes", 0))
            if remaining_added is not None and rate_delta > remaining_added:
                item["reason"] = "passed gate; skipped by byte budget"
                continue
            item["accepted"] = True
            item["reason"] = "accepted: ranked held-out decision accuracy gain"
            selected_keys.append(tuple(item["leaf_key"]))
            if remaining_added is not None:
                remaining_added -= rate_delta
        selected_ids = [model.find_leaf(key) for key in selected_keys]
        selected_ids = [int(leaf_id) for leaf_id in selected_ids if leaf_id is not None]
        if selected_ids:
            selected_fit_rays = _rays_intersecting_leaf_keys(model, fit_rays, selected_keys)
            final_change = model.split_leaves_batch(selected_ids)
            _tune_affected_on_fixed_rays(
                model,
                selected_fit_rays,
                final_change.new_leaf_keys,
                steps=int(gate_cfg.get("accepted_refit_steps", gate_cfg.get("tune_steps", 30))),
                batch_rays=int(gate_cfg.get("accepted_refit_batch_rays", gate_cfg.get("tune_batch_rays", 2048))),
                learning_rate=float(gate_cfg.get("tune_learning_rate", train_cfg.get("lr", 1.0e-2))),
                weight_kwargs=weight_kwargs,
            )
        accepted = len(selected_ids)
        reason_counts = {}
        for item in evaluations:
            reason = str(item.get("reason", ""))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        fit_valid = sum(1 for item in evaluations if int(item.get("fit_ray_count_for_leaf", 0)) > 0)
        decision_valid = sum(1 for item in evaluations if int(item.get("decision_ray_count_for_leaf", 0)) > 0)
        tested = sum(1 for item in evaluations if int(item.get("candidate_active", 0)) > 0)
        return accepted, {
            "crossfit_enabled": True,
            "crossfit_acceptance_mode": acceptance_mode,
            "fit_ray_count": int(fit_rays.observation_count),
            "decision_ray_count": int(decision_rays.observation_count),
            "fit_fraction": float(gate_cfg.get("fit_fraction", 0.8)),
            "candidate_count": len(candidate_keys),
            "fit_ray_valid_count": int(fit_valid),
            "decision_ray_valid_count": int(decision_valid),
            "tested_count": int(tested),
            "passed_gate_count": len(passed_evaluations),
            "accepted_count": accepted,
            "max_splits_per_round": max_splits,
            "base_rate_bytes": int(base_bytes),
            "byte_budget": int(byte_budget) if byte_budget is not None else None,
            "local_selection_objective": "held-out decision accuracy gain",
            "noise_variance": float(noise_variance),
            "compact_quantization": quantization,
            "rejection_reason_counts": reason_counts,
            "evaluations": evaluations,
        }

    for key in candidate_keys:
        leaf_id = model.find_leaf(key)
        if leaf_id is None:
            continue
        candidate_fit_rays = _rays_intersecting_leaf_keys(model, fit_rays, [key])
        candidate_decision_rays = _rays_intersecting_leaf_keys(model, decision_rays, [key])
        if candidate_decision_rays.observation_count == 0:
            evaluations.append(
                {
                    "leaf_key": list(key),
                    "candidate_active": 0,
                    "fit_ray_count_for_leaf": int(candidate_fit_rays.observation_count),
                    "decision_ray_count_for_leaf": 0,
                    "decision_before": {"ray_count": 0},
                    "decision_after": {"ray_count": 0},
                    "decision_gain_noise_normalized_rss": 0.0,
                    "rate_delta_bytes": 0,
                    "accepted": False,
                    "reason": "rejected: no decision rays intersect candidate",
                }
            )
            continue
        before_state = _clone_sparse_state(model)
        before_bytes = _compact_raw_payload_bytes(model, quantization=quantization)
        before = _fixed_ray_distortion(
            model,
            candidate_decision_rays,
            weight_kwargs=weight_kwargs,
            noise_variance=noise_variance,
        )
        change = model.split_leaves_batch([leaf_id])
        candidate_active = int(len(change.new_leaf_keys))
        if candidate_active:
            _tune_affected_on_fixed_rays(
                model,
                candidate_fit_rays,
                change.new_leaf_keys,
                steps=int(gate_cfg.get("tune_steps", 30)),
                batch_rays=int(gate_cfg.get("tune_batch_rays", 2048)),
                learning_rate=float(gate_cfg.get("tune_learning_rate", train_cfg.get("lr", 1.0e-2))),
                weight_kwargs=weight_kwargs,
            )
        after = _fixed_ray_distortion(
            model,
            candidate_decision_rays,
            weight_kwargs=weight_kwargs,
            noise_variance=noise_variance,
        )
        after_bytes = _compact_raw_payload_bytes(model, quantization=quantization)
        rate_delta = after_bytes - before_bytes
        gain = float(before["noise_normalized_rss"] - after["noise_normalized_rss"])
        within_budget = byte_budget is None or after_bytes <= int(byte_budget)
        keep = candidate_active > 0 and rate_delta > 0 and gain > min_gain and within_budget
        evaluations.append(
            {
                "leaf_key": list(key),
                "candidate_active": candidate_active,
                "fit_ray_count_for_leaf": int(candidate_fit_rays.observation_count),
                "decision_ray_count_for_leaf": int(candidate_decision_rays.observation_count),
                "decision_before": before,
                "decision_after": after,
                "decision_gain_noise_normalized_rss": gain,
                "rate_delta_bytes": int(rate_delta),
                "accepted": bool(keep),
                "reason": (
                    "accepted: held-out decision accuracy improved"
                    if keep
                    else (
                        "rejected: byte budget reached"
                        if not within_budget
                        else "rejected: no held-out decision accuracy gain"
                    )
                ),
            }
        )
        if keep:
            accepted += 1
        else:
            _restore_sparse_state(model, before_state)
    reason_counts = {}
    for item in evaluations:
        reason = str(item.get("reason", ""))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    fit_valid = sum(1 for item in evaluations if int(item.get("fit_ray_count_for_leaf", 0)) > 0)
    decision_valid = sum(1 for item in evaluations if int(item.get("decision_ray_count_for_leaf", 0)) > 0)
    tested = sum(1 for item in evaluations if int(item.get("candidate_active", 0)) > 0)
    return accepted, {
        "crossfit_enabled": True,
        "fit_ray_count": int(fit_rays.observation_count),
        "decision_ray_count": int(decision_rays.observation_count),
        "fit_fraction": float(gate_cfg.get("fit_fraction", 0.8)),
        "candidate_count": len(candidate_keys),
        "fit_ray_valid_count": int(fit_valid),
        "decision_ray_valid_count": int(decision_valid),
        "tested_count": int(tested),
        "accepted_count": accepted,
        "byte_budget": int(byte_budget) if byte_budget is not None else None,
        "local_selection_objective": "held-out decision accuracy gain",
        "noise_variance": float(noise_variance),
        "compact_quantization": quantization,
        "rejection_reason_counts": reason_counts,
        "evaluations": evaluations,
    }


@torch.no_grad()
def _complete_sibling_groups(model, *, level: int, max_groups: int | None = None) -> list[tuple[tuple[int, int, int, int], ...]]:
    groups: dict[tuple[int, int, int, int], list[tuple[int, int, int, int]]] = {}
    for leaf_id in range(int(model.leaf_levels.shape[0])):
        key = model.leaf_key(leaf_id)
        child_level, x, y, z = key
        if int(child_level) != int(level):
            continue
        parent = (child_level - 1, x // 2, y // 2, z // 2)
        groups.setdefault(parent, []).append(key)
    siblings = [tuple(sorted(keys)) for keys in groups.values() if len(keys) == 8]
    siblings.sort()
    return siblings[:max_groups] if max_groups is not None else siblings


def _apply_h_merge_with_crossfit_gate(
    model,
    *,
    dataset,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    spec: dict,
    adaptive_cfg: dict,
    train_cfg: dict,
    iteration: int,
    materialize_ray_points: bool,
    noise_variance: float,
) -> tuple[int, dict]:
    gate_cfg = spec.get("crossfit_acceptance") or spec.get("acceptance") or {}
    if not bool(gate_cfg.get("enabled", False)):
        return 0, {"crossfit_enabled": False, "accepted_count": 0}
    level = int(spec.get("level", 1))
    max_candidates = int(gate_cfg.get("max_candidates", spec.get("max_candidates", 256)))
    candidate_groups = _complete_sibling_groups(model, level=level, max_groups=max_candidates)
    if not candidate_groups:
        return 0, {"crossfit_enabled": True, "candidate_count": 0, "accepted_count": 0}

    fit_rays, decision_rays = _sample_fit_decision_rays(
        dataset.train,
        ray_count=int(gate_cfg.get("ray_count", 8192)),
        fit_fraction=float(gate_cfg.get("fit_fraction", 0.8)),
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        seed=int(gate_cfg.get("seed", iteration)),
        chunk_rays=int(gate_cfg.get("ray_chunk", train_cfg.get("eval_ray_chunk", 4096))),
        materialize_points=materialize_ray_points,
    )
    weight_kwargs = _projection_weight_kwargs(
        train_cfg,
        adaptive_cfg,
        gate_cfg.get("projection_weighting"),
        iteration=iteration,
    )
    quantization = str(gate_cfg.get("compact_quantization", (train_cfg.get("compact_quantization") or "float16")))
    merge_lambda = float(gate_cfg.get("merge_lambda", gate_cfg.get("rate_lambda", adaptive_cfg.get("merge_lambda", 0.0))))
    max_penalty = gate_cfg.get("max_penalty")
    max_penalty = float(max_penalty) if max_penalty is not None else None
    base_state = _clone_sparse_state(model)
    base_bytes = _compact_raw_payload_bytes(model, quantization=quantization)
    evaluations = []

    for group in candidate_groups:
        _restore_sparse_state(model, base_state)
        candidate_fit_rays = _rays_intersecting_leaf_keys(model, fit_rays, group)
        candidate_decision_rays = _rays_intersecting_leaf_keys(model, decision_rays, group)
        if candidate_decision_rays.observation_count == 0:
            evaluations.append(
                {
                    "leaf_keys": [list(key) for key in group],
                    "decision_before": {"ray_count": 0},
                    "decision_after": {"ray_count": 0},
                    "bytes_saved": 0,
                    "decision_penalty_noise_normalized_rss": 0.0,
                    "cost_per_saved_byte": None,
                    "passed_gate": False,
                    "accepted": False,
                    "reason": "rejected: no decision rays intersect candidate",
                }
            )
            continue
        before = _fixed_ray_distortion(
            model,
            candidate_decision_rays,
            weight_kwargs=weight_kwargs,
            noise_variance=noise_variance,
        )
        child_ids = [model.find_leaf(key) for key in group]
        if any(leaf_id is None for leaf_id in child_ids):
            continue
        change = model.merge_siblings([int(leaf_id) for leaf_id in child_ids])
        if change.new_leaf_keys:
            _tune_affected_on_fixed_rays(
                model,
                candidate_fit_rays,
                change.new_leaf_keys,
                steps=int(gate_cfg.get("tune_steps", 30)),
                batch_rays=int(gate_cfg.get("tune_batch_rays", 2048)),
                learning_rate=float(gate_cfg.get("tune_learning_rate", train_cfg.get("lr", 1.0e-2))),
                weight_kwargs=weight_kwargs,
            )
        after = _fixed_ray_distortion(
            model,
            candidate_decision_rays,
            weight_kwargs=weight_kwargs,
            noise_variance=noise_variance,
        )
        after_bytes = _compact_raw_payload_bytes(model, quantization=quantization)
        bytes_saved = base_bytes - after_bytes
        penalty = float(after["noise_normalized_rss"] - before["noise_normalized_rss"])
        cost = max(penalty, 0.0) / max(bytes_saved, 1)
        passed = bytes_saved > 0 and cost <= merge_lambda and (max_penalty is None or penalty <= max_penalty)
        evaluations.append(
            {
                "leaf_keys": [list(key) for key in group],
                "parent_key": list(change.new_leaf_keys[0]) if change.new_leaf_keys else None,
                "fit_ray_count_for_group": int(candidate_fit_rays.observation_count),
                "decision_ray_count_for_group": int(candidate_decision_rays.observation_count),
                "decision_before": before,
                "decision_after": after,
                "bytes_saved": int(bytes_saved),
                "decision_penalty_noise_normalized_rss": penalty,
                "cost_per_saved_byte": cost,
                "passed_gate": bool(passed),
                "accepted": False,
                "reason": (
                    "passed gate; waiting for ranked merge selection"
                    if passed
                    else "rejected: decision penalty per saved byte is too high"
                ),
            }
        )

    _restore_sparse_state(model, base_state)
    passed_evaluations = [item for item in evaluations if item.get("passed_gate")]
    passed_evaluations.sort(key=lambda item: float(item.get("cost_per_saved_byte", float("inf"))))
    max_merges = int(gate_cfg.get("max_merges_per_round", gate_cfg.get("max_merges", len(passed_evaluations))))
    selected_groups = []
    for item in passed_evaluations:
        if len(selected_groups) >= max_merges:
            item["reason"] = "passed gate; skipped by per-round merge cap"
            continue
        item["accepted"] = True
        item["reason"] = "accepted: ranked decision penalty per saved byte"
        selected_groups.append(tuple(tuple(key) for key in item["leaf_keys"]))
    for group in selected_groups:
        current_ids = [model.find_leaf(key) for key in group]
        current_ids = [int(leaf_id) for leaf_id in current_ids if leaf_id is not None]
        if len(current_ids) != 8:
            continue
        candidate_fit_rays = _rays_intersecting_leaf_keys(model, fit_rays, group)
        change = model.merge_siblings(current_ids)
        _tune_affected_on_fixed_rays(
            model,
            candidate_fit_rays,
            change.new_leaf_keys,
            steps=int(gate_cfg.get("accepted_refit_steps", gate_cfg.get("tune_steps", 30))),
            batch_rays=int(gate_cfg.get("accepted_refit_batch_rays", gate_cfg.get("tune_batch_rays", 2048))),
            learning_rate=float(gate_cfg.get("tune_learning_rate", train_cfg.get("lr", 1.0e-2))),
            weight_kwargs=weight_kwargs,
        )

    reason_counts = {}
    for item in evaluations:
        reason = str(item.get("reason", ""))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    accepted = len(selected_groups)
    return accepted, {
        "crossfit_enabled": True,
        "fit_ray_count": int(fit_rays.observation_count),
        "decision_ray_count": int(decision_rays.observation_count),
        "fit_fraction": float(gate_cfg.get("fit_fraction", 0.8)),
        "candidate_count": len(candidate_groups),
        "passed_gate_count": len(passed_evaluations),
        "accepted_count": accepted,
        "max_merges_per_round": max_merges,
        "base_rate_bytes": int(base_bytes),
        "merge_lambda": merge_lambda,
        "noise_variance": float(noise_variance),
        "compact_quantization": quantization,
        "rejection_reason_counts": reason_counts,
        "evaluations": evaluations,
    }


def _apply_p_elevation_with_projection_gate(
    model,
    selected: torch.Tensor,
    target_degree,
    *,
    dataset,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    spec: dict,
    refinement_defaults: dict,
    adaptive_cfg: dict,
    train_cfg: dict,
    iteration: int,
) -> tuple[int, dict]:
    """Apply batch p-elevation, optionally guarded by held-out projection R-D."""
    acceptance_cfg = (
        spec.get("acceptance")
        or spec.get("heldout_acceptance")
        or spec.get("held_out_acceptance")
        or {}
    )
    if not bool(acceptance_cfg.get("enabled", False)):
        change = model.elevate_leaves_isotropic_batch(selected, target_degree)
        return int(len(change.new_leaf_keys)), {
            "old_coefficient_count": int(change.old_coefficient_count),
            "new_coefficient_count": int(change.new_coefficient_count),
            "acceptance_enabled": False,
        }

    group_leaf_count = int(acceptance_cfg.get("group_leaf_count", 0) or 0)
    if group_leaf_count > 0 and int(selected.numel()) > group_leaf_count:
        old_count = int(model.coefficient_logits.numel())
        groups = list(torch.split(selected, group_leaf_count))
        group_summaries: list[dict] = []
        total_active = 0
        candidate_active = 0
        candidate_rate_delta = 0
        for group_id, group in enumerate(groups):
            nested_acceptance = dict(acceptance_cfg)
            nested_acceptance["group_leaf_count"] = 0
            nested_spec = dict(spec)
            nested_spec["acceptance"] = nested_acceptance
            active, summary = _apply_p_elevation_with_projection_gate(
                model,
                group,
                target_degree,
                dataset=dataset,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                spec=nested_spec,
                refinement_defaults=refinement_defaults,
                adaptive_cfg=adaptive_cfg,
                train_cfg=train_cfg,
                iteration=iteration + group_id,
            )
            total_active += int(active)
            candidate_active += int(summary.get("candidate_active", 0))
            candidate_rate_delta += int(summary.get("acceptance_rate_delta_bytes", 0))
            group_summaries.append(
                {
                    "group": group_id,
                    "candidate_active": int(summary.get("candidate_active", 0)),
                    "active": int(active),
                    "accepted": bool(summary.get("accepted", active > 0)),
                    "validation_gain": float(summary.get("acceptance_validation_gain", 0.0)),
                    "relative_gain": float(summary.get("acceptance_relative_gain", 0.0)),
                    "score_per_byte": float(summary.get("acceptance_score_per_byte", 0.0)),
                    "reason": str(summary.get("acceptance_reason", "")),
                }
            )
        accepted_groups = sum(int(item["accepted"]) for item in group_summaries)
        return total_active, {
            "old_coefficient_count": old_count,
            "new_coefficient_count": int(model.coefficient_logits.numel()),
            "candidate_coefficient_count": int(old_count + candidate_rate_delta // 4),
            "candidate_active": candidate_active,
            "acceptance_enabled": True,
            "accepted": bool(total_active > 0),
            "acceptance_reason": (
                "one or more held-out projection groups passed R-D gate"
                if total_active > 0
                else "rejected: no p-refinement group passed held-out R-D gate"
            ),
            "acceptance_group_leaf_count": group_leaf_count,
            "acceptance_group_count": len(group_summaries),
            "acceptance_accepted_groups": accepted_groups,
            "acceptance_rejected_groups": len(group_summaries) - accepted_groups,
            "acceptance_group_summaries": group_summaries,
        }

    before_state = _clone_sparse_state(model)
    before_stats = model.stats()
    validation_source = str(acceptance_cfg.get("validation_source", "train_holdout"))
    probe_train_split, validation_split = _select_acceptance_splits(
        dataset,
        validation_source,
        validation_fraction=float(acceptance_cfg.get("validation_fraction", 0.2)),
        seed=int(acceptance_cfg.get("seed", iteration)),
    )
    weight_kwargs = _projection_weight_kwargs(
        train_cfg,
        adaptive_cfg,
        acceptance_cfg.get("projection_weighting") or {"epsilon": acceptance_cfg.get("weight_epsilon", None)},
        iteration=iteration,
    )
    validation_ray_chunk = int(
        acceptance_cfg.get(
            "validation_ray_chunk",
            refinement_defaults.get("validation_ray_chunk", train_cfg.get("eval_ray_chunk", 4096)),
        )
    )
    validation_views = acceptance_cfg.get("validation_views", refinement_defaults.get("validation_views", 1))
    validation_rays_per_view = acceptance_cfg.get(
        "validation_rays_per_view",
        refinement_defaults.get("validation_rays_per_view", 8192),
    )
    before_loss = _sampled_projection_loss(
        model,
        validation_split,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=validation_ray_chunk,
        max_views=validation_views,
        max_rays_per_view=validation_rays_per_view,
        weight_kwargs=weight_kwargs,
    )
    change = model.elevate_leaves_isotropic_batch(selected, target_degree)
    candidate_active = int(len(change.new_leaf_keys))
    tune_steps = int(acceptance_cfg.get("tune_steps", 0))
    if candidate_active > 0 and tune_steps > 0:
        _tune_affected_coefficients(
            model,
            probe_train_split,
            change.new_leaf_keys,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            steps=tune_steps,
            batch_rays=int(acceptance_cfg.get("tune_batch_rays", train_cfg.get("batch_rays", 4096))),
            learning_rate=float(acceptance_cfg.get("tune_learning_rate", train_cfg.get("lr", 2e-2))),
            weight_epsilon=float(weight_kwargs["epsilon"]),
            weight_kwargs=weight_kwargs,
            new_dof_axis=None,
        )
    after_loss = _sampled_projection_loss(
        model,
        validation_split,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=validation_ray_chunk,
        max_views=validation_views,
        max_rays_per_view=validation_rays_per_view,
        weight_kwargs=weight_kwargs,
    )
    after_stats = model.stats()
    validation_gain = float(before_loss - after_loss)
    relative_gain = validation_gain / max(float(before_loss), 1e-12)
    rate_delta_bytes = int(after_stats.model_bytes - before_stats.model_bytes)
    score_per_byte = validation_gain / max(rate_delta_bytes, 1)
    min_gain = float(acceptance_cfg.get("min_gain", 0.0))
    min_relative_gain = float(acceptance_cfg.get("min_relative_gain", 0.0))
    rate_lambda = float(acceptance_cfg.get("rate_lambda", 0.0))
    accepted = (
        candidate_active > 0
        and validation_gain > min_gain
        and relative_gain >= min_relative_gain
        and score_per_byte >= rate_lambda
    )
    if candidate_active <= 0:
        reason = "rejected: no candidate leaf changed degree"
    elif validation_gain <= min_gain:
        reason = "rejected: held-out gain below absolute threshold"
    elif relative_gain < min_relative_gain:
        reason = "rejected: held-out gain below relative threshold"
    elif score_per_byte < rate_lambda:
        reason = "rejected: held-out gain per byte below R-D threshold"
    else:
        reason = "held-out projection gain passed R-D gate"
    if not accepted:
        _restore_sparse_state(model, before_state)
    return (candidate_active if accepted else 0), {
        "old_coefficient_count": int(change.old_coefficient_count),
        "new_coefficient_count": int(model.coefficient_logits.numel()),
        "candidate_coefficient_count": int(change.new_coefficient_count),
        "candidate_active": candidate_active,
        "acceptance_enabled": True,
        "accepted": bool(accepted),
        "acceptance_reason": reason,
        "validation_source": validation_source,
        "acceptance_training_views": int(probe_train_split.angles.shape[0]),
        "acceptance_validation_views": int(validation_split.angles.shape[0]),
        "acceptance_validation_loss_before": before_loss,
        "acceptance_validation_loss_after": after_loss,
        "acceptance_validation_gain": validation_gain,
        "acceptance_relative_gain": relative_gain,
        "acceptance_rate_delta_bytes": rate_delta_bytes,
        "acceptance_score_per_byte": score_per_byte,
        "acceptance_min_gain": min_gain,
        "acceptance_min_relative_gain": min_relative_gain,
        "acceptance_rate_lambda": rate_lambda,
        "acceptance_tune_steps": tune_steps,
        "acceptance_validation_sampled_rays": int(max(0, validation_views or validation_split.angles.shape[0]))
        * int(validation_rays_per_view or detector_h * detector_w),
    }


_DORFLER_OPERATIONS = {"dorfler_h_split", "dorfler_split"}


def _validate_stage7_dorfler_scope(config: dict) -> None:
    errors = []
    model_cfg = config.get("model", {}) or {}
    train_cfg = config.get("training", {}) or {}
    progressive = train_cfg.get("progressive", {}) or {}
    if str(model_cfg.get("representation", "")).lower() not in {"bernstein_octree", "bernstein"}:
        errors.append("stage7 trains a Bernstein octree; latent codecs are applied at export only")
    if str(model_cfg.get("initialization", "projection_mean")).lower() != "projection_mean":
        errors.append("stage7 must initialize from projection_mean only (no FDK/reference volume)")
    if bool(train_cfg.get("allow_test_during_training", False)):
        errors.append("test views must not be used during training")
    if float(train_cfg.get("tv_weight", 0.0)) != 0.0:
        errors.append("tv_weight must be 0")
    for key in ("volume_loss", "projection_gradient_loss", "coefficient_continuity"):
        if float((train_cfg.get(key, {}) or {}).get("weight", 0.0)) != 0.0:
            errors.append(f"training.{key}.weight must be 0")
    milestones = [
        item for item in (progressive.get("convergence_milestones", []) or [])
        if bool((item or {}).get("enabled", True))
    ]
    if len(milestones) != 1 or str(milestones[0].get("operation", "")).lower() not in _DORFLER_OPERATIONS:
        errors.append("stage7 needs exactly one convergence milestone with operation dorfler_h_split")
    else:
        spec = milestones[0]
        if bool(spec.get("rollback", True)) and int((config.get("dataset", {}) or {}).get("dev_views", 0)) < 1:
            errors.append("rollback needs dataset.dev_views >= 1")
        if spec.get("byte_budget") is None:
            errors.append("dorfler_h_split needs a byte_budget")
    if progressive.get("milestones") or (progressive.get("structural_rounds", {}) or {}).get("enabled", False):
        errors.append("stage7 uses only the dorfler convergence milestone")
    if errors:
        raise ValueError("Invalid stage7 Dorfler config:\n- " + "\n- ".join(errors))


def run_training(config_path: str | Path) -> dict:
    run_start_time = time.perf_counter()
    config = load_config(config_path)
    research_scope = str((config.get("research", {}) or {}).get("scope", "")).lower()
    if research_scope == "first_paper":
        _validate_first_paper_scope(config)
    elif research_scope in {"stage0_direct_p1", "baseline_direct_p1"}:
        _validate_stage0_direct_p1_scope(config)
    elif research_scope in {"stage1_equal_topology", "stage1_latent_p1_equal_topology"}:
        _validate_stage1_equal_topology_scope(config)
    elif research_scope in {"stage3_crossfit_split", "stage3_projection_split"}:
        _validate_stage3_crossfit_split_scope(config)
    elif research_scope in {"stage4_structural_split", "stage4_projection_structural_split"}:
        _validate_stage4_structural_split_scope(config)
    elif research_scope in {"stage5_merge", "stage5_projection_merge", "stage6_compact_artifact"}:
        _validate_stage5_merge_scope(config)
    elif research_scope == "stage7_dorfler":
        _validate_stage7_dorfler_scope(config)
    training_seed = int((config.get("training", {}) or {}).get("seed", 0))
    torch.manual_seed(training_seed)
    np.random.seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    device = torch.device(config.get("device", "cuda"))
    out_dir = Path(config["output"]["dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_representation = str(config.get("model", {}).get("representation", "dynamic_leaf_voxel")).lower()
    projection_domain_requested = _projection_domain_representation(requested_representation)
    dataset = load_r2_dataset(
        config["dataset"]["root"],
        device=device,
        load_volume=not projection_domain_requested,
    )
    dev_views = int((config.get("dataset", {}) or {}).get("dev_views", 0))
    if dev_views > 0 and dataset.validation is None:
        train_views = int(dataset.train.angles.shape[0])
        train_split, dev_split = split_projection_views(
            dataset.train,
            validation_fraction=dev_views / float(train_views),
            seed=int((config.get("dataset", {}) or {}).get("dev_seed", training_seed)),
        )
        dataset = dataclass_replace(dataset, train=train_split, validation=dev_split)
    detector_h, detector_w = dataset.detector_shape
    samples_per_ray = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))
    model = build_model(config).to(device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    projection_domain_model = _projection_domain_representation(str(getattr(model, "representation", "")))
    if projection_domain_model:
        initialization = str(config["model"].get("initialization", "projection_mean")).lower()
        if initialization == "projection_mean":
            # For the normalized [-1, 1]^3 object box, the representative path
            # length is two. This initializes c000 directly from observations.
            initial_mu = (dataset.train.projections.mean() * 0.5).clamp_min(1e-8)
            _initialize_constant_mu(model, initial_mu)
        elif initialization == "fdk_volume":
            fdk_path = config["model"].get("fdk_path")
            if not fdk_path:
                raise ValueError("model.fdk_path is required for fdk_volume initialization.")
            fdk_volume = torch.from_numpy(np.load(Path(fdk_path)).astype(np.float32)).to(device=device)
            coarse = F.adaptive_avg_pool3d(
                fdk_volume[None, None],
                output_size=(model.l0_resolution,) * 3,
            )[0, 0].reshape(-1)
            with torch.no_grad():
                model.coefficient_logits.copy_(model._coefficients_to_logits(coarse.clamp_min(1e-8)))
            del fdk_volume, coarse
        elif initialization != "random":
            raise ValueError("Bernstein initialization must be 'projection_mean', 'fdk_volume', or 'random'.")
        initial_leaf_degree = config["model"].get("initial_leaf_degree")
        if initial_leaf_degree is not None:
            target_degree = tuple(int(value) for value in initial_leaf_degree)
            if target_degree != (1, 1, 1):
                raise ValueError(
                    "model.initial_leaf_degree currently supports only [1, 1, 1]."
                )
            if any(int(value) < 1 for value in model.max_degree):
                raise ValueError(
                    "model.max_degree must allow [1, 1, 1] when initial_leaf_degree is enabled."
                )
            model.elevate_leaves_isotropic_batch(
                torch.arange(model.leaf_levels.shape[0], device=device),
                target_degree,
            )
    materialize_ray_points = not (
        hasattr(model, "prefer_compact_ray_batch") and model.prefer_compact_ray_batch()
    )

    train_cfg = config["training"]
    optimizer = _make_optimizer(model, float(train_cfg.get("lr", 2e-2)))
    iterations = int(train_cfg.get("iterations", 300))
    batch_rays = int(train_cfg.get("batch_rays", 4096))
    tv_weight = float(train_cfg.get("tv_weight", 0.0))
    log_every = int(train_cfg.get("log_every", 25))
    eval_every = int(train_cfg.get("eval_every", 100))
    eval_views = int(train_cfg.get("eval_test_views", 3))
    compute_eval = bool(train_cfg.get("compute_eval", True))
    allow_test_during_training = bool(train_cfg.get("allow_test_during_training", True))
    intermediate_evaluation_split = dataset.validation
    intermediate_evaluation_name = "validation"
    if intermediate_evaluation_split is None and allow_test_during_training:
        intermediate_evaluation_split = dataset.test
        intermediate_evaluation_name = "test"
    progressive = train_cfg.get("progressive", {}) or {}
    structural_rounds_cfg = progressive.get("structural_rounds", {}) or {}
    structural_gate_cfg = structural_rounds_cfg.get("crossfit_acceptance", {}) or {}
    global_byte_budget = structural_gate_cfg.get("byte_budget", structural_rounds_cfg.get("byte_budget"))
    if global_byte_budget is None:
        for item in progressive.get("convergence_milestones", []) or []:
            gate = (item or {}).get("crossfit_acceptance", {}) or {}
            global_byte_budget = gate.get("byte_budget", (item or {}).get("byte_budget"))
            if global_byte_budget is not None:
                break
    global_byte_budget = int(global_byte_budget) if global_byte_budget is not None else None
    milestones = {
        int(k): v
        for k, v in (progressive.get("milestones", {}) or {}).items()
        if bool((v or {}).get("enabled", True))
    }
    for structural_iteration, structural_spec in _structural_round_milestones(progressive).items():
        if structural_iteration in milestones:
            raise ValueError(f"structural_rounds collides with progressive.milestones at iteration {structural_iteration}.")
        milestones[structural_iteration] = structural_spec
    convergence_milestones = [
        dict(value)
        for value in (progressive.get("convergence_milestones", []) or [])
        if bool((value or {}).get("enabled", True))
    ]
    convergence_index = 0
    convergence_last_trigger = 0
    convergence_ema = None
    convergence_ema_span = max(
        [int(value.get("ema_span", 50)) for value in convergence_milestones] or [50]
    )
    convergence_values = deque(
        maxlen=max([int(value.get("window", 200)) for value in convergence_milestones] or [200])
    )
    fixed_decision_cfg = train_cfg.get("fixed_decision_rays", {}) or {}
    fixed_decision_enabled = bool(fixed_decision_cfg.get("enabled", False))
    fixed_decision_interval = max(1, int(fixed_decision_cfg.get("interval", 1)))
    fixed_decision_use_for_plateau = bool(
        fixed_decision_cfg.get("use_for_plateau", fixed_decision_enabled)
    )
    fixed_decision_set = None
    fixed_decision_last = None
    if fixed_decision_enabled:
        fixed_decision_set = FixedDecisionRaySet.sampled(
            dataset.train,
            ray_count=int(fixed_decision_cfg.get("ray_count", batch_rays)),
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            seed=int(fixed_decision_cfg.get("seed", int(train_cfg.get("seed", 0)) + 104729)),
            chunk_rays=int(
                fixed_decision_cfg.get("chunk_rays", train_cfg.get("eval_ray_chunk", 65536))
            ),
            materialize_points=materialize_ray_points,
        )
    refinement_defaults = progressive.get("refinement", {}) or {}
    dorfler_controller = None
    for item in convergence_milestones:
        if str(item.get("operation", "")).lower() in _DORFLER_OPERATIONS:
            dorfler_controller = DorflerController(
                spec=item,
                estimator_split=dataset.train,
                dev_split=dataset.validation,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                materialize_points=materialize_ray_points,
                chunk_rays=int(train_cfg.get("eval_ray_chunk", 65536)),
                seed=training_seed,
                byte_budget=global_byte_budget,
                clone_state=_clone_sparse_state,
                restore_state=_restore_sparse_state,
            )
            break

    def refinement_projection_split(spec: dict):
        split_name = str(
            spec.get("selection_split", refinement_defaults.get("selection_split", "train"))
        ).lower()
        if split_name in {"train", "training"}:
            return dataset.train, "train"
        if split_name in {"validation", "val", "development", "dev"}:
            if dataset.validation is None:
                raise ValueError(
                    "Refinement requested the validation split, but the dataset has no proj_val entries."
                )
            return dataset.validation, "validation"
        if split_name in {"test", "testing"}:
            raise ValueError("Final test projections must never be used for topology or degree selection.")
        raise ValueError(f"Unknown refinement selection_split {split_name!r}.")
    volume_loss_cfg = train_cfg.get("volume_loss", {}) or {}
    volume_loss_weight = float(volume_loss_cfg.get("weight", 0.0))
    if projection_domain_model and (tv_weight > 0.0 or volume_loss_weight > 0.0):
        raise ValueError(
            "Bernstein RD-CVF training is projection-domain only: set tv_weight=0 and volume_loss.weight=0."
        )
    continuity_cfg = train_cfg.get("coefficient_continuity", {}) or {}
    continuity_weight = float(continuity_cfg.get("weight", 0.0))
    continuity_start = int(continuity_cfg.get("start_iteration", 1))
    continuity_stop = continuity_cfg.get("stop_iteration")
    if continuity_stop is not None:
        continuity_stop = int(continuity_stop)
    projection_gradient_cfg = train_cfg.get("projection_gradient_loss", {}) or {}
    projection_gradient_weight = float(projection_gradient_cfg.get("weight", 0.0))
    if projection_gradient_weight > 0.0 and not projection_domain_model:
        raise ValueError("training.projection_gradient_loss is supported only by projection-domain models")
    adaptive_cfg = train_cfg.get("adaptive", {}) or {}
    noise_cfg = train_cfg.get("noise_normalization", {}) or {}
    noise_info = (
        _estimate_projection_noise_variance(dataset.train, noise_cfg)
        if bool(noise_cfg.get("enabled", False))
        else {"enabled": False, "variance": 1.0, "frozen": True}
    )
    rate_lambda = float(adaptive_cfg.get("rate_lambda", train_cfg.get("rate_lambda", 0.0)))
    adaptive_enabled = projection_domain_model and bool(adaptive_cfg.get("enabled", True))
    adaptive_iterations: set[int] = set()
    if adaptive_enabled:
        adaptive_rounds = int(adaptive_cfg.get("rounds", 5))
        if adaptive_rounds < 1:
            raise ValueError("training.adaptive.rounds must be positive when adaptive training is enabled.")
        adaptive_start = int(adaptive_cfg.get("start_iteration", max(2, iterations // 5)))
        adaptive_stop = int(adaptive_cfg.get("stop_iteration", max(adaptive_start, (4 * iterations) // 5)))
        if adaptive_rounds == 1:
            adaptive_iterations = {adaptive_start}
        else:
            adaptive_iterations = {
                int(round(adaptive_start + index * (adaptive_stop - adaptive_start) / float(adaptive_rounds - 1)))
                for index in range(adaptive_rounds)
            }
    volume_sampler = None
    if volume_loss_weight > 0.0:
        volume_sampler = VolumeSampler(
            dataset.volume,
            material_threshold=float(config["metrics"].get("material_threshold", 0.1)),
            boundary_quantile=float(volume_loss_cfg.get("boundary_quantile", 0.9)),
        )
    volume_loss_samples = int(volume_loss_cfg.get("samples", batch_rays))
    volume_loss_start = int(volume_loss_cfg.get("start_iteration", 1))
    volume_loss_stop = volume_loss_cfg.get("stop_iteration")
    if volume_loss_stop is not None:
        volume_loss_stop = int(volume_loss_stop)

    history = []
    growth_events = []
    adaptive_events = []
    evaluation_history = []
    dynamic_stop_iteration = None
    completed_iterations = 0
    start_time = time.perf_counter()
    for iteration in range(1, iterations + 1):
        completed_iterations = iteration
        convergence_spec = None
        convergence_trigger_summary = {}
        if convergence_index < len(convergence_milestones):
            candidate = convergence_milestones[convergence_index]
            overrides = {}
            if dorfler_controller is not None and str(candidate.get("operation", "")).lower() in _DORFLER_OPERATIONS:
                overrides = {k: v for k, v in dorfler_controller.window_overrides().items() if v is not None}
            min_iteration = int(candidate.get("min_iteration", 1))
            min_after_previous = int(
                overrides.get("min_iterations_after_previous", candidate.get("min_iterations_after_previous", 0))
            )
            plateau_window = int(overrides.get("window", candidate.get("window", 200)))
            max_iteration = candidate.get("max_iteration")
            window_summary = _convergence_window_summary(
                convergence_values,
                window=plateau_window,
                relative_improvement_threshold=float(candidate.get("relative_improvement", 0.002)),
                max_relative_regression=float(candidate.get("max_relative_regression", 0.005)),
            )
            time_ready = (
                iteration >= min_iteration
                and iteration - convergence_last_trigger >= min_after_previous
            )
            forced = max_iteration is not None and iteration >= int(max_iteration)
            if time_ready and (window_summary["ready"] or forced):
                convergence_spec = candidate
                if not bool(candidate.get("repeat_until_stopped", False)):
                    convergence_index += 1
                convergence_last_trigger = iteration
                convergence_values.clear()
                convergence_trigger_summary = {
                    "trigger": "convergence" if window_summary["ready"] else "max_iteration",
                    "convergence": window_summary,
                    "convergence_min_iteration": min_iteration,
                    "convergence_max_iteration": int(max_iteration) if max_iteration is not None else None,
                }
        if iteration in adaptive_iterations:
            model, adaptive_event = adaptive_projection_round(
                model,
                dataset.train,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                config=adaptive_cfg,
                seed=int(adaptive_cfg.get("seed", 0)) + iteration,
            )
            model = model.to(device=device)
            optimizer = _make_optimizer(model, float(train_cfg.get("lr", 2e-2)))
            adaptive_event = {"iteration": iteration, **adaptive_event}
            adaptive_events.append(adaptive_event)
            print(json.dumps({"adaptive": adaptive_event}))
        elif convergence_spec is not None or iteration in milestones:
            progress_checkpoint = out_dir / "checkpoint_latest.pt"
            torch.save(
                {"model": model.state_dict(), "config": config, "iteration": iteration - 1},
                progress_checkpoint,
            )
            growth_start_time = time.perf_counter()
            spec = convergence_spec if convergence_spec is not None else milestones[iteration]
            operation = str(spec.get("operation", spec.get("action", "h_split"))).lower()
            milestone_levels = spec.get("levels")
            if milestone_levels is None and operation in {"global_h_split", "global_split", *_DORFLER_OPERATIONS}:
                milestone_levels = []
            elif milestone_levels is None:
                milestone_levels = [int(spec["level"])]
            else:
                milestone_levels = [int(value) for value in milestone_levels]
            level = int(spec.get("level", milestone_levels[0] if milestone_levels else 1))
            strategy = str(spec.get("strategy", refinement_defaults.get("strategy", "gradient"))).lower()
            active_fraction = float(spec.get("active_fraction", 1.0))
            halo = int(spec.get("halo", refinement_defaults.get("halo", 1)))
            if projection_domain_model and operation in {
                "hp",
                "hp_compete",
                "hp_rate_distortion",
                "gauss_newton_hp",
            }:
                selection_split, selection_split_name = refinement_projection_split(spec)
                active, hp_summary = _apply_hp_gauss_newton_rate_distortion(
                    model,
                    selection_split,
                    levels=milestone_levels if spec.get("levels") is not None else None,
                    detector_h=detector_h,
                    detector_w=detector_w,
                    samples_per_ray=samples_per_ray,
                    ray_chunk=int(
                        spec.get(
                            "score_ray_chunk",
                            refinement_defaults.get(
                                "residual_ray_chunk",
                                train_cfg.get("eval_ray_chunk", 65536),
                            ),
                        )
                    ),
                    max_views=spec.get("score_views", refinement_defaults.get("residual_views", 16)),
                    max_rays_per_view=spec.get(
                        "score_rays_per_view",
                        refinement_defaults.get("residual_rays_per_view", 32768),
                    ),
                    weight_kwargs=_projection_weight_kwargs(
                        train_cfg,
                        adaptive_cfg,
                        spec.get("projection_weighting") or refinement_defaults.get("projection_weighting"),
                        iteration=iteration,
                    ),
                    fisher_damping=float(spec.get("fisher_damping", 1.0e-8)),
                    max_added_bytes=int(spec.get("max_added_bytes", 1_000_000)),
                    h_added_bytes=int(spec.get("h_added_bytes", 99)),
                    p_added_bytes=int(spec.get("p_added_bytes", 14)),
                )
                growth_summary = {
                    "operation": "hp_gauss_newton_rate_distortion",
                    "levels": milestone_levels if spec.get("levels") is not None else "all",
                    "selection_split": selection_split_name,
                    **hp_summary,
                }
            elif projection_domain_model and operation in {"p", "p_refine", "p_elevate", "degree", "degree_elevate"}:
                if getattr(model, "representation", "") == "latent_p1_octree":
                    raise ValueError("latent_p1_octree first-paper scope forbids p-degree adaptation; use h_split/h_merge only.")
                if strategy in {"gradient_rate_distortion", "gradient_rd", "gradient-rd"}:
                    target_degree = spec.get("target_degree", spec.get("degree", 1))
                    selection_split, selection_split_name = refinement_projection_split(spec)
                    leaf_ids, leaf_scores, residual_summary = _bernstein_projection_residual_leaf_score(
                        model,
                        selection_split,
                        level=milestone_levels,
                        detector_h=detector_h,
                        detector_w=detector_w,
                        samples_per_ray=samples_per_ray,
                        ray_chunk=int(
                            spec.get(
                                "residual_ray_chunk",
                                refinement_defaults.get("residual_ray_chunk", train_cfg.get("eval_ray_chunk", 65536)),
                            )
                        ),
                        max_views=spec.get("residual_views", refinement_defaults.get("residual_views", 16)),
                        max_rays_per_view=spec.get(
                            "residual_rays_per_view",
                            refinement_defaults.get("residual_rays_per_view", 32768),
                        ),
                        weight_kwargs=_projection_weight_kwargs(
                            train_cfg,
                            adaptive_cfg,
                            spec.get("projection_weighting") or refinement_defaults.get("projection_weighting"),
                            iteration=iteration,
                        ),
                        gradient_weight=float(spec.get("gradient_weight", refinement_defaults.get("gradient_weight", 0.0))),
                        max_buffer_mb=float(
                            spec.get("residual_max_buffer_mb", refinement_defaults.get("residual_max_buffer_mb", 512))
                        ),
                    )
                    candidate_fraction = float(spec.get("candidate_fraction", 0.08))
                    selected, candidate_summary = _select_p_residual_candidates(
                        leaf_ids,
                        leaf_scores,
                        candidate_fraction=candidate_fraction,
                        residual_quantile_range=spec.get("residual_quantile_range"),
                    )
                    active, acceptance_summary = _apply_gradient_rate_distortion_p_refinement(
                        model,
                        selected,
                        target_degree,
                        training_split=selection_split,
                        detector_h=detector_h,
                        detector_w=detector_w,
                        samples_per_ray=samples_per_ray,
                        gradient_batches=int(spec.get("gradient_batches", 8)),
                        batch_rays=int(spec.get("gradient_batch_rays", train_cfg.get("batch_rays", 8192))),
                        active_fraction=active_fraction,
                        eligible_leaf_count=int(leaf_ids.numel()),
                        max_added_coefficients=int(spec.get("max_added_coefficients", 1000000)),
                        learning_rate=float(train_cfg.get("lr", 2e-2)),
                        weight_kwargs=_projection_weight_kwargs(train_cfg, adaptive_cfg, iteration=iteration),
                        projection_gradient_cfg=train_cfg.get("projection_gradient_loss", {}) or {},
                        iteration=iteration,
                    )
                    growth_summary = {
                        "operation": "p_elevate",
                        "levels": milestone_levels,
                        "target_degree": target_degree,
                        "projection_domain_seed": True,
                        "selection_split": selection_split_name,
                        **candidate_summary,
                        **residual_summary,
                        **acceptance_summary,
                    }
                elif strategy in {"residual", "projection_residual", "projected_residual", "heldout", "held_out"}:
                    residual_split_name = str(spec.get("residual_split", refinement_defaults.get("residual_split", "train"))).lower()
                    residual_split, residual_split_name = refinement_projection_split(
                        {**spec, "selection_split": residual_split_name}
                    )
                    leaf_ids, leaf_scores, residual_summary = _bernstein_projection_residual_leaf_score(
                        model,
                        residual_split,
                        level=level,
                        detector_h=detector_h,
                        detector_w=detector_w,
                        samples_per_ray=samples_per_ray,
                        ray_chunk=int(spec.get("residual_ray_chunk", refinement_defaults.get("residual_ray_chunk", train_cfg.get("eval_ray_chunk", 4096)))),
                        max_views=spec.get("residual_views", refinement_defaults.get("residual_views", 2)),
                        max_rays_per_view=spec.get("residual_rays_per_view", refinement_defaults.get("residual_rays_per_view", 65536)),
                        weight_kwargs=_projection_weight_kwargs(
                            train_cfg,
                            adaptive_cfg,
                            spec.get("projection_weighting") or refinement_defaults.get("projection_weighting"),
                            iteration=iteration,
                        ),
                        gradient_weight=float(spec.get("gradient_weight", refinement_defaults.get("gradient_weight", 0.0))),
                        max_buffer_mb=float(
                            spec.get("residual_max_buffer_mb", refinement_defaults.get("residual_max_buffer_mb", 512))
                        ),
                    )
                else:
                    leaf_ids, leaf_scores = _bernstein_leaf_score(model, level=level, strategy=strategy)
                    residual_summary = {}
                if strategy not in {"gradient_rate_distortion", "gradient_rd", "gradient-rd"}:
                    count = max(1, int(leaf_ids.numel() * active_fraction)) if leaf_ids.numel() else 0
                    selected = (
                        leaf_ids[torch.topk(leaf_scores, min(count, leaf_ids.numel())).indices]
                        if count > 0
                        else leaf_ids
                    )
                    target_degree = spec.get("target_degree", spec.get("degree", 1))
                    active, acceptance_summary = _apply_p_elevation_with_projection_gate(
                        model,
                        selected,
                        target_degree,
                        dataset=dataset,
                        detector_h=detector_h,
                        detector_w=detector_w,
                        samples_per_ray=samples_per_ray,
                        spec=spec,
                        refinement_defaults=refinement_defaults,
                        adaptive_cfg=adaptive_cfg,
                        train_cfg=train_cfg,
                        iteration=iteration,
                    )
                    growth_summary = {
                        "operation": "p_elevate",
                        "target_degree": target_degree,
                        "score_mean": float(leaf_scores.mean().detach().item()) if leaf_scores.numel() else 0.0,
                        "score_max": float(leaf_scores.max().detach().item()) if leaf_scores.numel() else 0.0,
                        "projection_domain_seed": True,
                        **acceptance_summary,
                        **residual_summary,
                    }
            elif projection_domain_model and operation in {
                "h_jump",
                "h_jump_rd",
                "h_jump_rate_distortion",
            }:
                active, jump_summary = _apply_h_jump_rate_distortion(
                    model,
                    max_added_bytes=int(spec.get("max_added_bytes", 1_000_000)),
                    h_added_bytes=(
                        int(spec["h_added_bytes"]) if "h_added_bytes" in spec else None
                    ),
                )
                growth_summary = {
                    "operation": "h_jump_rate_distortion",
                    "levels": "all eligible",
                    **jump_summary,
                }
            elif projection_domain_model and operation in {
                "h_gauss_newton",
                "h_gauss_newton_rd",
                "h_rate_distortion",
            }:
                if len(milestone_levels) != 1:
                    raise ValueError("Each h Gauss-Newton round must target exactly one parent level.")
                parent_level = int(milestone_levels[0])
                selection_split, selection_split_name = refinement_projection_split(spec)
                active, h_summary = _apply_h_gauss_newton_rate_distortion(
                    model,
                    selection_split,
                    level=parent_level,
                    detector_h=detector_h,
                    detector_w=detector_w,
                    samples_per_ray=samples_per_ray,
                    ray_chunk=int(
                        spec.get(
                            "score_ray_chunk",
                            refinement_defaults.get(
                                "residual_ray_chunk",
                                train_cfg.get("eval_ray_chunk", 65536),
                            ),
                        )
                    ),
                    max_views=spec.get("score_views", refinement_defaults.get("residual_views", 16)),
                    max_rays_per_view=spec.get(
                        "score_rays_per_view",
                        refinement_defaults.get("residual_rays_per_view", 32768),
                    ),
                    weight_kwargs=_projection_weight_kwargs(
                        train_cfg,
                        adaptive_cfg,
                        spec.get("projection_weighting") or refinement_defaults.get("projection_weighting"),
                        iteration=iteration,
                    ),
                    fisher_damping=float(spec.get("fisher_damping", 1.0e-8)),
                    max_added_bytes=int(spec.get("max_added_bytes", 1_000_000)),
                    h_added_bytes=int(spec.get("h_added_bytes", 99)),
                )
                growth_summary = {
                    "operation": "h_gauss_newton_rate_distortion",
                    "parent_level": parent_level,
                    "target_level": parent_level + 1,
                    "selection_split": selection_split_name,
                    **h_summary,
                }
            elif projection_domain_model and operation in {"h_merge", "merge"}:
                active, merge_summary = _apply_h_merge_with_crossfit_gate(
                    model,
                    dataset=dataset,
                    detector_h=detector_h,
                    detector_w=detector_w,
                    samples_per_ray=samples_per_ray,
                    spec=spec,
                    adaptive_cfg=adaptive_cfg,
                    train_cfg=train_cfg,
                    iteration=iteration,
                    materialize_ray_points=materialize_ray_points,
                    noise_variance=float(noise_info.get("variance", 1.0)),
                )
                growth_summary = {
                    "operation": "h_merge",
                    "level": level,
                    **merge_summary,
                }
            elif projection_domain_model and operation in _DORFLER_OPERATIONS:
                if dorfler_controller is None:
                    raise RuntimeError("dorfler_h_split milestone without a controller.")
                active, dorfler_summary = dorfler_controller.on_trigger(
                    model,
                    iteration=iteration,
                    noise_variance=float(noise_info.get("variance", 0.0)) if noise_info.get("enabled") else 0.0,
                    weight_kwargs=_projection_weight_kwargs(train_cfg, adaptive_cfg, iteration=iteration),
                )
                growth_summary = {
                    "operation": "dorfler_h_split",
                    "candidate_scope": "all_splittable_leaves",
                    "projection_domain_seed": True,
                    **dorfler_summary,
                }
            elif projection_domain_model and operation in {"global_h_split", "global_split"}:
                if strategy not in {
                    "residual",
                    "projection_residual",
                    "projected_residual",
                    "heldout",
                    "held_out",
                }:
                    raise ValueError("global_h_split currently supports residual cheap proposal only.")
                model_levels = getattr(model, "level_shapes", None) or getattr(model, "level_resolutions", [])
                max_split_level = int(spec.get("max_split_level", len(model_levels) - 1))
                parent_levels_cfg = spec.get("parent_levels", spec.get("levels"))
                if parent_levels_cfg is None:
                    parent_levels = list(range(max(0, max_split_level)))
                else:
                    parent_levels = [int(value) for value in parent_levels_cfg]
                max_parent_level = len(model_levels) - 2
                parent_levels = [level_id for level_id in parent_levels if 0 <= level_id <= max_parent_level]
                if not parent_levels:
                    raise ValueError("global_h_split has no splittable parent levels.")
                residual_split_name = str(
                    spec.get(
                        "selection_split",
                        spec.get("residual_split", refinement_defaults.get("selection_split", refinement_defaults.get("residual_split", "train"))),
                    )
                ).lower()
                residual_split, residual_split_name = refinement_projection_split(
                    {**spec, "selection_split": residual_split_name}
                )
                score_weight_kwargs = _projection_weight_kwargs(
                    train_cfg,
                    adaptive_cfg,
                    spec.get("projection_weighting") or refinement_defaults.get("projection_weighting"),
                    iteration=iteration,
                )
                leaf_ids, leaf_scores, residual_summary = _bernstein_projection_residual_leaf_score(
                    model,
                    residual_split,
                    level=parent_levels,
                    detector_h=detector_h,
                    detector_w=detector_w,
                    samples_per_ray=samples_per_ray,
                    ray_chunk=int(
                        spec.get(
                            "residual_ray_chunk",
                            refinement_defaults.get("residual_ray_chunk", train_cfg.get("eval_ray_chunk", 4096)),
                        )
                    ),
                    max_views=spec.get("residual_views", refinement_defaults.get("residual_views", 2)),
                    max_rays_per_view=spec.get(
                        "residual_rays_per_view",
                        refinement_defaults.get("residual_rays_per_view", 65536),
                    ),
                    weight_kwargs=score_weight_kwargs,
                    gradient_weight=float(spec.get("gradient_weight", refinement_defaults.get("gradient_weight", 0.0))),
                    max_buffer_mb=float(
                        spec.get("residual_max_buffer_mb", refinement_defaults.get("residual_max_buffer_mb", 512))
                    ),
                    score_mode=str(spec.get("residual_score_mode", "attributed_energy")),
                )
                pre_score_levels = model.leaf_levels[leaf_ids].detach().clone() if leaf_ids.numel() else model.leaf_levels[:0]
                eligible_by_level = {
                    str(level_id): int(torch.sum(pre_score_levels == level_id).item())
                    for level_id in parent_levels
                }
                exposure_positive_by_level = {
                    str(level_id): int(torch.sum((pre_score_levels == level_id) & torch.isfinite(leaf_scores) & (leaf_scores > 0)).item())
                    for level_id in parent_levels
                }
                crossfit_cfg = spec.get("crossfit_acceptance") or {}
                selected, selection_summary = _select_global_residual_h_candidates(
                    model,
                    leaf_ids,
                    leaf_scores,
                    max_candidates=int(spec.get("max_candidates", crossfit_cfg.get("max_candidates", 256))),
                    reserve_leaf_count=int(spec.get("balance_reserve_leaves", 0)),
                    min_mu_threshold=spec.get("min_mu_threshold", spec.get("material_threshold")),
                )
                if bool(crossfit_cfg.get("enabled", False)):
                    active, acceptance_summary = _apply_h_split_with_crossfit_gate(
                        model,
                        selected,
                        dataset=dataset,
                        detector_h=detector_h,
                        detector_w=detector_w,
                        samples_per_ray=samples_per_ray,
                        spec=spec,
                        adaptive_cfg=adaptive_cfg,
                        train_cfg=train_cfg,
                        iteration=iteration,
                        materialize_ray_points=materialize_ray_points,
                        noise_variance=float(noise_info.get("variance", 1.0)),
                    )
                    balance_summary = (
                        model.balance_2to1()
                        if active > 0 and model.balance_2to1_enabled
                        else {"enabled": False, "rounds": 0, "split_parents": 0, "complete": True}
                    )
                    new_leaf_count = int(active * 8)
                else:
                    change = model.split_leaves_batch(selected)
                    balance_summary = (
                        model.balance_2to1()
                        if model.balance_2to1_enabled
                        else {"enabled": False, "rounds": 0, "split_parents": 0, "complete": True}
                    )
                    active = int(selected.numel())
                    acceptance_summary = {"crossfit_enabled": False}
                    new_leaf_count = int(len(change.new_leaf_keys))
                finite_scores = leaf_scores[torch.isfinite(leaf_scores)]
                proposal_funnel = {
                    "parent_levels": [int(value) for value in parent_levels],
                    "level_leaf_total": int(residual_summary.get("residual_level_leaf_count", leaf_ids.numel())),
                    "eligible": int(leaf_ids.numel()),
                    "eligible_by_level": eligible_by_level,
                    "residual_covered": int(residual_summary.get("residual_covered_count", 0)),
                    "exposure_positive": int(residual_summary.get("residual_positive_count", 0)),
                    "exposure_positive_by_level": exposure_positive_by_level,
                    "shortlisted": int(selected.numel()),
                    "shortlisted_by_level": selection_summary.get("global_shortlisted_by_level", {}),
                    "fit_ray_valid": int(acceptance_summary.get("fit_ray_valid_count", 0)),
                    "decision_ray_valid": int(acceptance_summary.get("decision_ray_valid_count", 0)),
                    "tested": int(acceptance_summary.get("tested_count", 0)),
                    "accepted": int(acceptance_summary.get("accepted_count", active)),
                }
                growth_summary = {
                    "operation": "global_h_split",
                    "candidate_scope": "all_splittable_leaves",
                    "parent_levels": [int(value) for value in parent_levels],
                    "max_split_level": max_split_level,
                    "selected_parent_count": int(selected.numel()),
                    "new_leaf_count": new_leaf_count,
                    "score_mean": float(finite_scores.mean().detach().item()) if finite_scores.numel() else 0.0,
                    "score_max": float(finite_scores.max().detach().item()) if finite_scores.numel() else 0.0,
                    "projection_domain_seed": True,
                    "selection_split": residual_split_name,
                    "h_score_method": strategy,
                    "balance_2to1": balance_summary,
                    "proposal_funnel": proposal_funnel,
                    **acceptance_summary,
                    **selection_summary,
                    **residual_summary,
                }
            elif projection_domain_model:
                if strategy in {
                    "residual",
                    "projection_residual",
                    "projected_residual",
                    "heldout",
                    "held_out",
                    "haar_predicted_gain",
                    "haar_fisher",
                    "haar",
                }:
                    parent_level = level - 1
                    if parent_level < 0:
                        raise ValueError("Projection-domain h-split requires level >= 1.")
                    residual_split_name = str(
                        spec.get(
                            "selection_split",
                            spec.get("residual_split", refinement_defaults.get("selection_split", refinement_defaults.get("residual_split", "train"))),
                        )
                    ).lower()
                    residual_split, residual_split_name = refinement_projection_split(
                        {**spec, "selection_split": residual_split_name}
                    )
                    score_weight_kwargs = _projection_weight_kwargs(
                        train_cfg,
                        adaptive_cfg,
                        spec.get("projection_weighting") or refinement_defaults.get("projection_weighting"),
                        iteration=iteration,
                    )
                    score_views = spec.get("residual_views", refinement_defaults.get("residual_views", 2))
                    score_rays_per_view = spec.get(
                        "residual_rays_per_view",
                        refinement_defaults.get("residual_rays_per_view", 65536),
                    )
                    score_gradient_weight = float(
                        spec.get("gradient_weight", refinement_defaults.get("gradient_weight", 0.0))
                    )
                    if strategy in {"haar_predicted_gain", "haar_fisher", "haar"}:
                        leaf_ids, leaf_scores, residual_summary = _bernstein_haar_predicted_gain_leaf_score(
                            model,
                            residual_split,
                            level=parent_level,
                            detector_h=detector_h,
                            detector_w=detector_w,
                            samples_per_ray=samples_per_ray,
                            ray_chunk=int(spec.get("haar_ray_chunk", 4096)),
                            max_views=score_views,
                            max_rays_per_view=score_rays_per_view,
                            weight_kwargs=score_weight_kwargs,
                            gradient_weight=score_gradient_weight,
                            fisher_damping=float(spec.get("fisher_damping", 1.0e-8)),
                        )
                    else:
                        leaf_ids, leaf_scores, residual_summary = _bernstein_projection_residual_leaf_score(
                            model,
                            residual_split,
                            level=parent_level,
                            detector_h=detector_h,
                            detector_w=detector_w,
                            samples_per_ray=samples_per_ray,
                            ray_chunk=int(
                                spec.get(
                                    "residual_ray_chunk",
                                    refinement_defaults.get(
                                        "residual_ray_chunk",
                                        train_cfg.get("eval_ray_chunk", 4096),
                                    ),
                                )
                            ),
                            max_views=score_views,
                            max_rays_per_view=score_rays_per_view,
                            weight_kwargs=score_weight_kwargs,
                            gradient_weight=score_gradient_weight,
                            max_buffer_mb=float(
                                spec.get(
                                    "residual_max_buffer_mb",
                                    refinement_defaults.get("residual_max_buffer_mb", 512),
                                )
                            ),
                            score_mode=str(spec.get("residual_score_mode", "attributed_energy")),
                        )
                    selected, selection_summary = _select_residual_h_candidates(
                        model,
                        leaf_ids,
                        leaf_scores,
                        active_fraction=active_fraction,
                        threshold_statistic=spec.get("threshold_statistic"),
                        threshold_multiplier=float(spec.get("threshold_multiplier", 0.5)),
                        max_fraction=float(spec.get("max_fraction", 0.20 if parent_level < 2 else 0.10)),
                        reserve_leaf_count=int(spec.get("balance_reserve_leaves", 0)),
                        min_mu_threshold=spec.get("min_mu_threshold", spec.get("material_threshold")),
                    )
                    crossfit_cfg = spec.get("crossfit_acceptance") or {}
                    if bool(crossfit_cfg.get("enabled", False)):
                        active, acceptance_summary = _apply_h_split_with_crossfit_gate(
                            model,
                            selected,
                            dataset=dataset,
                            detector_h=detector_h,
                            detector_w=detector_w,
                            samples_per_ray=samples_per_ray,
                            spec=spec,
                            adaptive_cfg=adaptive_cfg,
                            train_cfg=train_cfg,
                            iteration=iteration,
                            materialize_ray_points=materialize_ray_points,
                            noise_variance=float(noise_info.get("variance", 1.0)),
                        )
                        balance_summary = (
                            model.balance_2to1()
                            if active > 0 and model.balance_2to1_enabled
                            else {"enabled": False, "rounds": 0, "split_parents": 0, "complete": True}
                        )
                        new_leaf_count = int(active * 8)
                    else:
                        change = model.split_leaves_batch(selected)
                        balance_summary = (
                            model.balance_2to1()
                            if model.balance_2to1_enabled
                            else {"enabled": False, "rounds": 0, "split_parents": 0, "complete": True}
                        )
                        active = int(torch.sum(model.leaf_levels == level).item())
                        acceptance_summary = {"crossfit_enabled": False}
                        new_leaf_count = int(len(change.new_leaf_keys))
                    finite_scores = leaf_scores[torch.isfinite(leaf_scores)]
                    proposal_funnel = {
                        "parent_level": int(parent_level),
                        "level_leaf_total": int(residual_summary.get("residual_level_leaf_count", leaf_ids.numel())),
                        "eligible": int(residual_summary.get("residual_level_leaf_count", leaf_ids.numel())),
                        "residual_covered": int(residual_summary.get("residual_covered_count", 0)),
                        "exposure_positive": int(residual_summary.get("residual_positive_count", 0)),
                        "top_fraction": int(selection_summary.get("top_fraction_count", 0)),
                        "proposal_threshold_union": int(selection_summary.get("threshold_union_count", 0)),
                        "shortlisted": int(selected.numel()),
                        "fit_ray_valid": int(acceptance_summary.get("fit_ray_valid_count", 0)),
                        "decision_ray_valid": int(acceptance_summary.get("decision_ray_valid_count", 0)),
                        "tested": int(acceptance_summary.get("tested_count", 0)),
                        "accepted": int(acceptance_summary.get("accepted_count", active)),
                    }
                    growth_summary = {
                        "operation": "h_split",
                        "selected_parent_count": int(selected.numel()),
                        "new_leaf_count": new_leaf_count,
                        "score_mean": float(finite_scores.mean().detach().item()) if finite_scores.numel() else 0.0,
                        "score_max": float(finite_scores.max().detach().item()) if finite_scores.numel() else 0.0,
                        "projection_domain_seed": True,
                        "selection_split": residual_split_name,
                        "h_score_method": strategy,
                        "balance_2to1": balance_summary,
                        "proposal_funnel": proposal_funnel,
                        **acceptance_summary,
                        **selection_summary,
                        **residual_summary,
                    }
                else:
                    score = _bernstein_refinement_score(model, level=level, strategy=strategy)
                    selected, selection_summary = _select_coefficient_h_candidates(
                        model,
                        level=level,
                        score=score,
                        active_fraction=active_fraction,
                        min_mu_threshold=spec.get("min_mu_threshold", spec.get("material_threshold")),
                        max_fraction=float(spec.get("max_fraction", 1.0)),
                        reserve_leaf_count=int(spec.get("balance_reserve_leaves", 0)),
                        max_parent_count=spec.get("max_parent_count"),
                    )
                    change = model.split_leaves_batch(selected)
                    balance_summary = (
                        model.balance_2to1()
                        if model.balance_2to1_enabled
                        else {"enabled": False, "rounds": 0, "split_parents": 0, "complete": True}
                    )
                    active = int(torch.sum(model.leaf_levels == level).item())
                    growth_summary = {
                        "operation": "h_split",
                        "score_mean": float(score.mean().detach().item()),
                        "score_max": float(score.max().detach().item()),
                        "new_leaf_count": int(len(change.new_leaf_keys)),
                        "projection_domain_seed": True,
                        "balance_2to1": balance_summary,
                        **selection_summary,
                    }
            elif strategy == "hybrid":
                score = build_refinement_score(
                    model,
                    dataset,
                    level=level,
                    mode="hybrid",
                    detector_h=detector_h,
                    detector_w=detector_w,
                    samples_per_ray=samples_per_ray,
                    ray_chunk=int(train_cfg.get("eval_ray_chunk", 4096)),
                    residual_split=str(spec.get("residual_split", refinement_defaults.get("residual_split", "train"))),
                    residual_views=spec.get("residual_views", refinement_defaults.get("residual_views", 8)),
                    alpha=float(spec.get("alpha", refinement_defaults.get("alpha", 1.0))),
                    beta=float(spec.get("beta", refinement_defaults.get("beta", 1.0))),
                    gamma=float(spec.get("gamma", refinement_defaults.get("gamma", 0.25))),
                )
                active = model.activate_level_from_score(level, score.score, active_fraction, halo=halo)
                growth_summary = dict(score.summary)
            else:
                active = model.activate_level_from_gradient(level, active_fraction, halo=halo)
                growth_summary = {}
            optimizer = _make_optimizer(model, float(train_cfg.get("lr", 2e-2)))
            event = {
                "iteration": iteration,
                "level": level,
                "operation": growth_summary.get("operation", operation),
                "strategy": strategy,
                "halo": halo,
                "active": active,
                "elapsed_sec": time.perf_counter() - growth_start_time,
                **convergence_trigger_summary,
                **growth_summary,
            }
            if "structural_round_index" in spec:
                event["structural_round_index"] = int(spec["structural_round_index"])
                event["structural_rounds"] = int(spec.get("structural_rounds", event["structural_round_index"]))
            if event["operation"] == "h_jump_rate_distortion":
                # This selector scores every eligible leaf in one global queue.
                # Reporting a default active_fraction=1.0 is misleading: it is
                # candidate-pool coverage, not a forced selection fraction.
                event["candidate_scope"] = "all_eligible"
                event["strategy"] = "global_volume_scaled_face_jump_per_byte"
            elif event["operation"] in {"global_h_split", "dorfler_h_split"}:
                event["candidate_scope"] = "all_splittable_leaves"
            else:
                event["active_fraction"] = active_fraction
            if (
                convergence_spec is not None
                and bool(spec.get("repeat_until_stopped", False))
                and event["operation"] == "dorfler_h_split"
            ):
                event["structural_round_index"] = int(event.get("dorfler_round", 0))
                event["dynamic_refinement"] = {
                    "round": int(event.get("dorfler_round", 0)),
                    "stop": bool(event.get("stop", False)),
                    "stop_reasons": list(event.get("stop_reasons", [])),
                }
                if event.get("stop", False):
                    convergence_index += 1
                    dynamic_stop_iteration = min(
                        iterations,
                        iteration + max(0, int(spec.get("final_finetune_iterations", 1000))),
                    )
            elif convergence_spec is not None and bool(spec.get("repeat_until_stopped", False)):
                round_index = 1 + sum("dynamic_refinement" in previous for previous in growth_events)
                evaluations = event.get("evaluations", [])
                accepted_gains = [
                    float(value.get("decision_gain_noise_normalized_rss", 0.0))
                    for value in evaluations
                    if bool(value.get("accepted", False))
                ]
                added_bytes = [
                    int(value.get("rate_delta_bytes", 0))
                    for value in evaluations
                    if int(value.get("rate_delta_bytes", 0)) > 0
                ]
                current_bytes = _compact_raw_payload_bytes(
                    model,
                    quantization=str(
                        ((spec.get("crossfit_acceptance") or {}).get("compact_quantization"))
                        or (train_cfg.get("compact_quantization") or "float16")
                    ),
                )
                dynamic_status = _dynamic_refinement_status(
                    tested=int(event.get("tested_count", 0)),
                    accepted_gains=accepted_gains,
                    round_index=round_index,
                    min_rounds=int(spec.get("min_rounds", 1)),
                    max_rounds=int(spec.get("max_rounds", 32)),
                    stop_acceptance_ratio=float(spec.get("stop_acceptance_ratio", 0.03)),
                    current_bytes=current_bytes,
                    byte_budget=global_byte_budget,
                    min_added_bytes=min(added_bytes, default=1),
                )
                event["structural_round_index"] = round_index
                event["dynamic_refinement"] = dynamic_status
                if dynamic_status["stop"]:
                    convergence_index += 1
                    dynamic_stop_iteration = min(
                        iterations,
                        iteration + max(0, int(spec.get("final_finetune_iterations", 1000))),
                    )
            growth_events.append(event)
            print(json.dumps({"grow": event}))
            torch.save(
                {"model": model.state_dict(), "config": config, "iteration": iteration},
                progress_checkpoint,
            )

        rb = random_training_rays(
            dataset.train,
            batch_rays=batch_rays,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            materialize_points=materialize_ray_points,
        )
        pred = model.integrate_ray_batch(rb)
        if projection_domain_model:
            loss_proj = weighted_projection_loss(
                pred,
                rb.target,
                **_projection_weight_kwargs(train_cfg, adaptive_cfg, iteration=iteration),
            )
        else:
            loss_proj = F.mse_loss(pred, rb.target)
        projection_mse = F.mse_loss(pred, rb.target)
        loss = loss_proj
        loss_projection_gradient = rb.target.new_zeros(())
        effective_projection_gradient_weight = _projection_gradient_effective_weight(
            projection_gradient_cfg,
            iteration,
        )
        if effective_projection_gradient_weight > 0.0:
            gradient_batch = _sample_projection_gradient_batch(
                dataset.train,
                projection_gradient_cfg,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                materialize_points=materialize_ray_points,
            )
            gradient_prediction = model.integrate_ray_batch(gradient_batch.ray_batch)
            loss_projection_gradient = weighted_projection_gradient_loss(
                gradient_prediction,
                gradient_batch,
                **_projection_weight_kwargs(train_cfg, adaptive_cfg, iteration=iteration),
                **_projection_gradient_loss_kwargs(projection_gradient_cfg),
            )
            loss = loss + effective_projection_gradient_weight * loss_projection_gradient
        if tv_weight > 0.0:
            loss = loss + tv_weight * tv3(model.decoded_l0())
        loss_volume = rb.target.new_zeros(())
        loss_continuity = rb.target.new_zeros(())
        volume_loss_active = (
            volume_sampler is not None
            and iteration >= volume_loss_start
            and (volume_loss_stop is None or iteration <= volume_loss_stop)
        )
        if volume_loss_active:
            vb = volume_sampler.sample(
                volume_loss_samples,
                material_fraction=float(volume_loss_cfg.get("material_fraction", 0.5)),
                boundary_fraction=float(volume_loss_cfg.get("boundary_fraction", 0.25)),
            )
            volume_pred = model.forward_mu(vb.points)
            loss_volume = volume_sample_loss(volume_pred, vb.target, str(volume_loss_cfg.get("type", "mse")))
            loss = loss + volume_loss_weight * loss_volume
        continuity_active = (
            projection_domain_model
            and continuity_weight > 0.0
            and iteration >= continuity_start
            and (continuity_stop is None or iteration <= continuity_stop)
        )
        if continuity_active:
            ramp_steps = max(0, int(continuity_cfg.get("ramp_steps", 0)))
            continuity_scale = (
                min(1.0, float(iteration - continuity_start + 1) / float(ramp_steps))
                if ramp_steps > 0
                else 1.0
            )
            loss_continuity = coefficient_face_continuity_loss(
                model,
                max_pairs=int(continuity_cfg.get("max_pairs", 131072)),
                huber_delta=float(continuity_cfg.get("huber_delta", 0.02)),
                include_cross_level=bool(continuity_cfg.get("include_cross_level", True)),
                face_quadrature_order=int(continuity_cfg.get("face_quadrature_order", 2)),
            )
            loss = loss + continuity_weight * continuity_scale * loss_continuity

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        projection_loss_value = float(loss_proj.detach().item())
        plateau_loss_value = projection_loss_value
        fixed_decision_measured = (
            fixed_decision_set is not None
            and (iteration == 1 or iteration % fixed_decision_interval == 0)
        )
        if fixed_decision_measured:
            fixed_decision_last = fixed_rss(model, fixed_decision_set)
        if fixed_decision_use_for_plateau:
            if fixed_decision_last is None:
                raise RuntimeError("fixed decision rays did not produce a plateau measurement")
            plateau_loss_value = float(fixed_decision_last["mse"])
        if not fixed_decision_use_for_plateau or fixed_decision_measured:
            ema_alpha = 2.0 / float(max(1, convergence_ema_span) + 1)
            convergence_ema = (
                plateau_loss_value
                if convergence_ema is None
                else (1.0 - ema_alpha) * convergence_ema + ema_alpha * plateau_loss_value
            )
            convergence_values.append(float(convergence_ema))

        if iteration == 1 or iteration % log_every == 0:
            stats = model.stats()
            rate_bytes = _compact_raw_payload_bytes(
                model,
                quantization=str((config.get("export", {}) or {}).get("compact_octree", {}).get("quantization", "float16")),
            )
            distortion_value = projection_loss_value / max(float(noise_info.get("variance", 1.0)), 1.0e-12)
            row = {
                "iteration": iteration,
                "loss": float(loss.detach().item()),
                "projection_loss": float(loss_proj.detach().item()),
                "projection_mse": float(projection_mse.detach().item()),
                "noise_normalized_distortion": float(distortion_value),
                "rate_bytes": int(rate_bytes),
                "plateau_loss": float(plateau_loss_value),
                "plateau_source": (
                    "fixed_decision_rays"
                    if fixed_decision_use_for_plateau
                    else "stochastic_training_batch"
                ),
                "projection_gradient_loss": float(loss_projection_gradient.detach().item()),
                "projection_gradient_weight": effective_projection_gradient_weight,
                "volume_sample_loss": float(loss_volume.detach().item()),
                "volume_loss_weight": volume_loss_weight,
                "coefficient_continuity_loss": float(loss_continuity.detach().item()),
                "coefficient_continuity_weight": continuity_weight,
                "parameter_count": stats.parameter_count,
                "model_bytes": stats.model_bytes,
                "leaf_cells": stats.leaf_cells,
                "l0_active": stats.l0_active,
                "l1_active": stats.l1_active,
                "l2_active": stats.l2_active,
                "l3_active": getattr(stats, "l3_active", 0),
                "active_by_level": list(getattr(stats, "active_by_level", (stats.l0_active, stats.l1_active, stats.l2_active))),
            }
            if global_byte_budget is not None:
                row.update(
                    {
                        "constrained_accuracy_objective": float(distortion_value),
                        "byte_budget": global_byte_budget,
                        "within_byte_budget": rate_bytes <= global_byte_budget,
                    }
                )
            else:
                row.update(
                    {
                        "rate_lambda": float(rate_lambda),
                        "rate_distortion_objective": float(distortion_value + rate_lambda * rate_bytes),
                    }
                )
            print(json.dumps(row))
            history.append(row)

        if (
            compute_eval
            and intermediate_evaluation_split is not None
            and (iteration % eval_every == 0 or iteration == iterations)
        ):
            split = intermediate_evaluation_split
            sub_split = type(split)(
                angles=split.angles[:eval_views],
                projections=split.projections[:eval_views],
                paths=split.paths[:eval_views],
            )
            rendered = render_split(
                model,
                sub_split,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                ray_chunk=int(train_cfg.get("eval_ray_chunk", 4096)),
            )
            metric = projection_metrics(rendered, sub_split.projections)
            evaluation_history.append({"iteration": iteration, **metric})
            print(json.dumps({"iteration": iteration, f"{intermediate_evaluation_name}_projection": metric}))

        if dynamic_stop_iteration is not None and iteration >= dynamic_stop_iteration:
            break

    elapsed = time.perf_counter() - start_time
    trained_model = model
    codec_summary = None
    export_codec_cfg = ((config.get("export", {}) or {}).get("codec")) or {}
    if projection_domain_model and export_codec_cfg:
        codec_type = str(export_codec_cfg.get("type", "p0")).lower()
        codec_weight_kwargs = _projection_weight_kwargs(train_cfg, adaptive_cfg, iteration=completed_iterations)

        def codec_test_metric(candidate):
            return projection_metrics(
                render_split(
                    candidate,
                    dataset.test,
                    detector_h=detector_h,
                    detector_w=detector_w,
                    samples_per_ray=samples_per_ray,
                    ray_chunk=int(train_cfg.get("eval_ray_chunk", 4096)),
                ),
                dataset.test.projections,
            )

        pre_codec = codec_test_metric(model)
        codec_summary = {"type": codec_type, "projection_test_pre_codec": pre_codec}
        if codec_type == "latent_p1":
            deployed = LatentP1Octree.from_bernstein(model, int(export_codec_cfg.get("latent_dim", 4)))
            codec_summary["projection_test_after_pca"] = codec_test_metric(deployed)
            codec_summary["finetune"] = finetune_model(
                deployed,
                dataset.train,
                steps=int(export_codec_cfg.get("finetune_iterations", 2000)),
                lr=float(export_codec_cfg.get("finetune_lr", 0.003)),
                batch_rays=batch_rays,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                weight_kwargs=codec_weight_kwargs,
                materialize_points=materialize_ray_points,
            )
        elif codec_type in {"p0", "p1"}:
            deployed = model
        else:
            raise ValueError(f"Unknown export.codec.type {codec_type!r}.")
        quantize_roundtrip_(deployed, export_codec_cfg)
        codec_summary["projection_test_deployed"] = codec_test_metric(deployed)
        codec_summary["quantization"] = str(export_codec_cfg.get("quantization", "float16"))
        codec_summary["quantization_applied_in_eval"] = True
        model = deployed
    stats = model.stats()
    metrics_cfg = config.get("metrics", {}) or {}
    if bool(metrics_cfg.get("compute_projection_metrics", True)):
        final_render = render_split(
            model,
            dataset.test,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            ray_chunk=int(train_cfg.get("eval_ray_chunk", 4096)),
        )
        projection_test = projection_metrics(final_render, dataset.test.projections)
    else:
        projection_test = {
            "skipped": True,
            "reason": "disabled by metrics.compute_projection_metrics",
        }
    needs_dense_evaluation = bool(metrics_cfg.get("compute_volume_metrics", True)) or bool(
        metrics_cfg.get("compute_boundary_metrics", True)
    )
    evaluation_dataset = dataset
    if projection_domain_model and needs_dense_evaluation:
        # Ground-truth volume is loaded only after optimization, for explicitly
        # separated evaluation metrics. It never enters training or adaptation.
        evaluation_dataset = load_r2_dataset(config["dataset"]["root"], device=device, load_volume=True)
    if bool(metrics_cfg.get("compute_volume_metrics", True)):
        material_volume = material_volume_metrics(
            model,
            evaluation_dataset,
            threshold=float(metrics_cfg.get("material_threshold", 0.1)),
        )
    else:
        material_volume = {
            "skipped": True,
            "reason": "disabled by metrics.compute_volume_metrics",
        }
    if bool(metrics_cfg.get("compute_boundary_metrics", True)):
        boundary_sharpness = boundary_sharpness_metrics(
            model,
            evaluation_dataset,
            threshold=float(metrics_cfg.get("material_threshold", 0.1)),
        )
    else:
        boundary_sharpness = {
            "skipped": True,
            "reason": "disabled by metrics.compute_boundary_metrics",
        }
    checkpoint_path = out_dir / "checkpoint.pt"
    final_checkpoint = {
        "model": trained_model.state_dict(),
        "config": config,
        "iteration": completed_iterations,
    }
    torch.save(final_checkpoint, checkpoint_path)
    torch.save(final_checkpoint, out_dir / "checkpoint_latest.pt")
    if model is not trained_model:
        torch.save(
            {"model": model.state_dict(), "config": config, "iteration": completed_iterations},
            out_dir / "checkpoint_codec.pt",
        )
    export_summary = {}
    if projection_domain_model:
        export_cfg = config.get("export", {}) or {}
        mact_cfg = export_cfg.get("mact", {}) or {}
        if bool(mact_cfg.get("enabled", True)):
            mact_summary = export_mact_artifact(
                model,
                out_dir / "mact.pt",
                material_clusters=int(mact_cfg.get("material_clusters", 4)),
                variance_retained=float(mact_cfg.get("variance_retained", 0.99)),
                max_rank=mact_cfg.get("max_rank"),
            )
            export_summary["mact"] = mact_summary.__dict__
        compact_cfg = export_cfg.get("compact_octree", {}) or {}
        if bool(compact_cfg.get("enabled", True)):
            compact_name = str(compact_cfg.get("artifact_name", "compact_octree.npz"))
            compact_summary = export_compact_octree_artifact(
                model,
                out_dir / compact_name,
                quantization=str(compact_cfg.get("quantization", "uint16")),
                topology=str(compact_cfg.get("topology", "explicit")),
                checkpoint_path=checkpoint_path,
            )
            export_summary["compact_octree"] = compact_summary.__dict__
        surface_cfg = export_cfg.get("surface", {}) or {}
        if bool(surface_cfg.get("enabled", False)):
            export_summary["surface"] = export_surface_artifact(
                model,
                dataset.train,
                out_dir / "surface_uncertainty.pt",
                threshold=float(surface_cfg.get("threshold", metrics_cfg.get("material_threshold", 0.1))),
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                uncertainty_rays=int(surface_cfg.get("uncertainty_rays", 64)),
                uncertainty_surface_points=int(surface_cfg.get("uncertainty_surface_points", 1024)),
            )
    export_summary["visual_slice"] = _export_center_slice_png(
        model,
        out_dir / "center_slice.png",
        dataset.volume_shape,
    )
    export_summary["visual_slice_fullres"] = _export_center_slice_png(
        model,
        out_dir / "center_slice_fullres.png",
        _finest_model_shape(model, dataset.volume_shape),
    )
    total_elapsed = time.perf_counter() - run_start_time
    report = {
        "config": str(Path(config_path).resolve()),
        "training_seed": training_seed,
        "configured_iterations": iterations,
        "completed_iterations": completed_iterations,
        "dynamic_stop_iteration": dynamic_stop_iteration,
        "dataset": str(dataset.root),
        "evaluation_protocol": {
            "training_views": int(dataset.train.angles.numel()),
            "validation_views": int(dataset.validation.angles.numel()) if dataset.validation is not None else 0,
            "test_views": int(dataset.test.angles.numel()),
            "intermediate_split": (
                intermediate_evaluation_name if intermediate_evaluation_split is not None else None
            ),
            "test_used_during_training": bool(
                intermediate_evaluation_split is dataset.test
            ),
            "test_role": "final evaluation only" if intermediate_evaluation_split is not dataset.test else "legacy intermediate and final evaluation",
        },
        "dorfler": (
            {
                "rounds": dorfler_controller.round_index,
                "stopped": dorfler_controller.stopped,
                "stop_reasons": list(dorfler_controller.stop_reasons),
                "rollback_enabled": dorfler_controller.rollback_enabled,
                "theta": dorfler_controller.theta,
                "dev_role": "rollback verdict and diagnostics only",
            }
            if dorfler_controller is not None
            else None
        ),
        "codec": codec_summary,
        "elapsed_sec": elapsed,
        "total_elapsed_sec": total_elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "projection_test": projection_test,
        "material_volume": material_volume,
        "boundary_sharpness": boundary_sharpness,
        "model": stats.__dict__,
        "ray_integration": {
            "compact_cuda_traversal": not materialize_ray_points,
            "samples_per_ray_config": samples_per_ray,
            "mode": getattr(model, "integration_mode", "native_or_sampled"),
        },
        "training_objective": {
            "global_objective": (
                {
                    "form": "minimize D(y, A_T[D_phi(Z)]) subject to compact_bytes(T,Z,phi) <= B_max",
                    "byte_budget": global_byte_budget,
                    "rate_bytes_source": "compact raw payload byte estimate",
                    "local_split_score": "held-out projection accuracy gain",
                    "noise_normalized_distortion": bool(noise_info.get("enabled", False)),
                }
                if global_byte_budget is not None
                else {
                    "form": "D(y, A_T[D_phi(Z)]) + lambda * compact_bytes(T,Z,phi)",
                    "rate_lambda": float(rate_lambda),
                    "rate_bytes_source": "compact raw payload byte estimate",
                    "noise_normalized_distortion": bool(noise_info.get("enabled", False)),
                }
            ),
            "noise_normalization": noise_info,
            "projection_loss": "weighted_least_squares" if projection_domain_model else "mse",
            "projection_gradient_loss": {
                "enabled": projection_gradient_weight > 0.0,
                "weight": projection_gradient_weight,
                "edge_rays": int(projection_gradient_cfg.get("edge_rays", 2048)),
                "strides": [int(value) for value in projection_gradient_cfg.get("strides", [1, 2, 4])],
                "uniform_fraction": float(projection_gradient_cfg.get("uniform_fraction", 0.5)),
                "candidate_multiplier": int(projection_gradient_cfg.get("candidate_multiplier", 8)),
                "edge_weight_power": float(projection_gradient_cfg.get("edge_weight_power", 0.0)),
                "edge_min_weight": projection_gradient_cfg.get("edge_min_weight"),
                "edge_max_weight": projection_gradient_cfg.get("edge_max_weight"),
                "magnitude_weight": float(projection_gradient_cfg.get("magnitude_weight", 0.0)),
                "magnitude_quantile": float(projection_gradient_cfg.get("magnitude_quantile", 0.75)),
                "moment_weight": float(projection_gradient_cfg.get("moment_weight", 0.0)),
                "moment_quantile": float(projection_gradient_cfg.get("moment_quantile", 0.75)),
                "endpoint_weight": float(projection_gradient_cfg.get("endpoint_weight", 0.0)),
                "endpoint_quantile": float(projection_gradient_cfg.get("endpoint_quantile", 0.75)),
                "start_iteration": int(projection_gradient_cfg.get("start_iteration", 1)),
                "stop_iteration": projection_gradient_cfg.get("stop_iteration"),
                "ramp_steps": int(projection_gradient_cfg.get("ramp_steps", 0)),
                "note": "Finite differences of actual paired-ray forward projections; no image post-processing.",
            },
            "tv_weight": tv_weight,
            "volume_loss": {
                "enabled": volume_loss_weight > 0.0,
                "weight": volume_loss_weight,
                "samples": volume_loss_samples,
                "type": str(volume_loss_cfg.get("type", "mse")),
                "material_fraction": float(volume_loss_cfg.get("material_fraction", 0.5)),
                "boundary_fraction": float(volume_loss_cfg.get("boundary_fraction", 0.25)),
                "boundary_quantile": float(volume_loss_cfg.get("boundary_quantile", 0.9)),
                "start_iteration": volume_loss_start,
                "stop_iteration": volume_loss_stop,
                "note": (
                    "Disabled for projection-domain RD-CVF training."
                    if projection_domain_model
                    else "Uses vol_gt.npy as explicit volume-domain supervision; keep separated from projection-only CT results."
                ),
            },
            "coefficient_continuity": {
                "enabled": continuity_weight > 0.0,
                "weight": continuity_weight,
                "max_pairs": int(continuity_cfg.get("max_pairs", 131072)),
                "huber_delta": float(continuity_cfg.get("huber_delta", 0.02)),
                "ramp_steps": int(continuity_cfg.get("ramp_steps", 0)),
                "include_cross_level": bool(continuity_cfg.get("include_cross_level", True)),
                "face_quadrature_order": int(continuity_cfg.get("face_quadrature_order", 2)),
                "start_iteration": continuity_start,
                "stop_iteration": continuity_stop,
                "note": "Coefficient-domain face continuity prior; no dense volume is decoded.",
            },
            "dense_volume_loaded_during_training": not projection_domain_model,
            "fixed_decision_rays": {
                "enabled": fixed_decision_enabled,
                "use_for_plateau": fixed_decision_use_for_plateau,
                "ray_count": (
                    int(fixed_decision_set.observation_count)
                    if fixed_decision_set is not None
                    else 0
                ),
                "interval": fixed_decision_interval,
                "last_mse": (
                    float(fixed_decision_last["mse"])
                    if fixed_decision_last is not None
                    else None
                ),
                "selection_split": "train",
                "note": (
                    "Frozen training rays are used only as a stable plateau ruler; "
                    "optimization still uses random rays."
                ),
            },
        },
        "history": history,
        "evaluation_history": evaluation_history,
        "growth_events": growth_events,
        "adaptive_events": adaptive_events,
        "rate_distortion_curve": [
            {
                "iteration": event["iteration"],
                "model_bytes": event.get("model_bytes"),
                "accepted_count": event["accepted_count"],
                "evaluations": [
                    {
                        "validation_gain": value["validation_gain"],
                        "rate_delta_bytes": value["rate_delta_bytes"],
                        "score_per_byte": value["score_per_byte"],
                        "accepted": value["accepted"],
                    }
                    for value in event["evaluations"]
                ],
            }
            for event in adaptive_events
        ],
        "exports": export_summary,
    }
    report_path = out_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_training(args.config)


if __name__ == "__main__":
    main()
