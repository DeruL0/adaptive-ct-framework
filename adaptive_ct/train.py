"""Projection-driven adaptive octree reconstruction: training entry point.

Pipeline: projections -> constant start -> exact-projector training ->
plateau-triggered Dorfler h-refinement (``dorfler_refinement``) -> codec
export (latent P1 / P0) -> evaluation and compact artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import replace as dataclass_replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .compression import export_compact_octree_artifact
from .config import load_config
from .data import load_r2_dataset
from .decision_rays import FixedDecisionRaySet, fixed_rss
from .dorfler_refinement import (
    DorflerController,
    codec_bytes,
    finetune_model,
    quantize_roundtrip_,
)
from .geometry import random_training_rays
from .latent_p1 import LatentP1Octree
from .metrics import projection_metrics
from .model import build_model
from .projection_domain import split_projection_views, weighted_projection_loss
from .render import render_split
from .volume_metrics import boundary_sharpness_metrics, material_volume_metrics

_OPERATION = "dorfler_h_split"


def _make_optimizer(model, lr: float):
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)


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


def _projection_weight_kwargs(train_cfg: dict, *, iteration: int | None = None) -> dict:
    weight_cfg = dict(train_cfg.get("projection_weighting", {}) or {})
    schedule = train_cfg.get("projection_weighting_schedule", {}) or {}
    if schedule:
        current_iteration = 1 if iteration is None else int(iteration)
        stages = sorted((int(key), value or {}) for key, value in schedule.items())
        for start, stage_cfg in stages:
            if start > current_iteration:
                break
            weight_cfg.update(stage_cfg)
    return {
        "epsilon": float(weight_cfg.get("epsilon", 1e-4)),
        "mode": str(weight_cfg.get("mode", "inverse")),
        "target_power": float(weight_cfg.get("target_power", 1.0)),
        "min_weight": weight_cfg.get("min_weight"),
        "max_weight": weight_cfg.get("max_weight"),
        "blend_alpha": float(weight_cfg.get("blend_alpha", 0.15)),
    }


def _clone_sparse_state(model) -> dict[str, torch.Tensor]:
    """Clone the sparse Bernstein state so a rejected trial can be rolled back."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _restore_sparse_state(model, state_dict: dict[str, torch.Tensor]) -> None:
    model.prepare_sparse_from_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)


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


def _validate_config(config: dict) -> None:
    errors = []
    model_cfg = config.get("model", {}) or {}
    train_cfg = config.get("training", {}) or {}
    progressive = train_cfg.get("progressive", {}) or {}
    if str(model_cfg.get("representation", "")).lower() not in {"bernstein_octree", "bernstein"}:
        errors.append("this framework trains a Bernstein octree; latent codecs are applied at export only")
    if str(model_cfg.get("initialization", "projection_mean")).lower() != "projection_mean":
        errors.append("this framework must initialize from projection_mean only (no FDK/reference volume)")
    if bool(train_cfg.get("allow_test_during_training", False)):
        errors.append("test views must not be used during training")
    if float(train_cfg.get("tv_weight", 0.0)) != 0.0:
        errors.append("tv_weight must be 0")
    for key in ("volume_loss", "projection_gradient_loss", "coefficient_continuity"):
        if float((train_cfg.get(key, {}) or {}).get("weight", 0.0)) != 0.0:
            errors.append(f"training.{key}.weight must be 0")
    milestones = [item for item in (progressive.get("convergence_milestones", []) or []) if item]
    if len(milestones) != 1 or str(milestones[0].get("operation", "")).lower() != _OPERATION:
        errors.append("this framework needs exactly one convergence milestone with operation dorfler_h_split")
    else:
        spec = milestones[0]
        if bool(spec.get("rollback", True)) and int((config.get("dataset", {}) or {}).get("dev_views", 0)) < 1:
            errors.append("rollback needs dataset.dev_views >= 1")
        if spec.get("byte_budget") is None:
            errors.append("dorfler_h_split needs a byte_budget")
    if not bool((train_cfg.get("fixed_decision_rays", {}) or {}).get("enabled", False)):
        errors.append("training.fixed_decision_rays must be enabled (it is the plateau ruler)")
    if progressive.get("milestones") or (progressive.get("structural_rounds", {}) or {}).get("enabled", False):
        errors.append("this framework uses only the dorfler convergence milestone")
    if errors:
        raise ValueError("Invalid refinement config:\n- " + "\n- ".join(errors))


def _active_milestone(config: dict) -> dict:
    progressive = (config.get("training", {}) or {}).get("progressive", {}) or {}
    for item in progressive.get("convergence_milestones", []) or []:
        if item:
            return dict(item)
    raise ValueError("config has no convergence milestone.")


def _split_dev_views(dataset, dataset_cfg: dict, seed: int):
    """Carve ``dataset.dev_views`` held-out views out of the training split."""
    dev_views = int(dataset_cfg.get("dev_views", 0))
    if dev_views <= 0 or dataset.validation is not None:
        return dataset
    train_views = int(dataset.train.angles.shape[0])
    train_split, dev_split = split_projection_views(
        dataset.train,
        validation_fraction=dev_views / float(train_views),
        seed=int(dataset_cfg.get("dev_seed", seed)),
    )
    return dataclass_replace(dataset, train=train_split, validation=dev_split)


def _plateau_ready(spec: dict, controller, values, iteration: int, last_trigger: int):
    """Return (ready, trigger_info) for the convergence-triggered refinement."""
    overrides = {k: v for k, v in controller.window_overrides().items() if v is not None}
    min_after_previous = int(
        overrides.get("min_iterations_after_previous", spec.get("min_iterations_after_previous", 0))
    )
    window = int(overrides.get("window", spec.get("window", 200)))
    summary = _convergence_window_summary(
        values,
        window=window,
        relative_improvement_threshold=float(spec.get("relative_improvement", 0.002)),
        max_relative_regression=float(spec.get("max_relative_regression", 0.005)),
    )
    max_iteration = spec.get("max_iteration")
    time_ready = (
        iteration >= int(spec.get("min_iteration", 1))
        and iteration - last_trigger >= min_after_previous
    )
    forced = max_iteration is not None and iteration >= int(max_iteration)
    if not (time_ready and (summary["ready"] or forced)):
        return False, {}
    return True, {
        "trigger": "convergence" if summary["ready"] else "max_iteration",
        "convergence": summary,
        "convergence_min_iteration": int(spec.get("min_iteration", 1)),
        "convergence_max_iteration": int(max_iteration) if max_iteration is not None else None,
    }


def _apply_export_codec(model, dataset, config: dict, *, batch_rays: int, detector_h: int, detector_w: int,
                        samples_per_ray: int, materialize_points: bool, weight_kwargs: dict):
    """Convert the trained octree to its deployed codec; returns (deployed_model, summary)."""
    codec_cfg = (config.get("export", {}) or {}).get("codec") or {}
    if not codec_cfg:
        return model, None
    train_cfg = config["training"]
    codec_type = str(codec_cfg.get("type", "p0")).lower()

    def test_metric(candidate):
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

    summary = {"type": codec_type, "projection_test_pre_codec": test_metric(model)}
    if codec_type == "latent_p1":
        deployed = LatentP1Octree.from_bernstein(model, int(codec_cfg.get("latent_dim", 4)))
        summary["projection_test_after_pca"] = test_metric(deployed)
        summary["finetune"] = finetune_model(
            deployed,
            dataset.train,
            steps=int(codec_cfg.get("finetune_iterations", 2000)),
            lr=float(codec_cfg.get("finetune_lr", 0.003)),
            batch_rays=batch_rays,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            weight_kwargs=weight_kwargs,
            materialize_points=materialize_points,
        )
    elif codec_type in {"p0", "p1"}:
        deployed = model
    else:
        raise ValueError(f"Unknown export.codec.type {codec_type!r}.")
    quantize_roundtrip_(deployed, codec_cfg)
    summary["projection_test_deployed"] = test_metric(deployed)
    summary["quantization"] = str(codec_cfg.get("quantization", "float16"))
    summary["quantization_applied_in_eval"] = True
    return deployed, summary


def run_training(config_path: str | Path) -> dict:
    run_start_time = time.perf_counter()
    config = load_config(config_path)
    _validate_config(config)
    training_seed = int((config.get("training", {}) or {}).get("seed", 0))
    torch.manual_seed(training_seed)
    np.random.seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    device = torch.device(config.get("device", "cuda"))
    out_dir = Path(config["output"]["dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_r2_dataset(config["dataset"]["root"], device=device, load_volume=False)
    dataset = _split_dev_views(dataset, config.get("dataset", {}) or {}, training_seed)
    detector_h, detector_w = dataset.detector_shape
    samples_per_ray = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))
    model = build_model(config).to(device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Normalized [-1, 1]^3 box: the representative path length is two, so the
    # constant start value comes directly from the projections.
    _initialize_constant_mu(model, (dataset.train.projections.mean() * 0.5).clamp_min(1e-8))
    initial_leaf_degree = config["model"].get("initial_leaf_degree")
    if initial_leaf_degree is not None:
        model.elevate_leaves_isotropic_batch(
            torch.arange(model.leaf_levels.shape[0], device=device),
            tuple(int(value) for value in initial_leaf_degree),
        )
    materialize_ray_points = not model.prefer_compact_ray_batch()

    train_cfg = config["training"]
    optimizer = _make_optimizer(model, float(train_cfg.get("lr", 2e-2)))
    iterations = int(train_cfg.get("iterations", 300))
    batch_rays = int(train_cfg.get("batch_rays", 4096))
    log_every = int(train_cfg.get("log_every", 25))
    eval_every = int(train_cfg.get("eval_every", 100))
    eval_views = int(train_cfg.get("eval_test_views", 3))
    compute_eval = bool(train_cfg.get("compute_eval", True))
    eval_split = dataset.validation
    eval_split_name = "validation"
    if eval_split is None and bool(train_cfg.get("allow_test_during_training", False)):
        eval_split, eval_split_name = dataset.test, "test"

    spec = _active_milestone(config)
    global_byte_budget = int(spec["byte_budget"])
    noise_cfg = train_cfg.get("noise_normalization", {}) or {}
    noise_info = (
        _estimate_projection_noise_variance(dataset.train, noise_cfg)
        if bool(noise_cfg.get("enabled", False))
        else {"enabled": False, "variance": 1.0, "frozen": True}
    )
    fixed_cfg = train_cfg["fixed_decision_rays"]
    fixed_interval = max(1, int(fixed_cfg.get("interval", 1)))
    fixed_set = FixedDecisionRaySet.sampled(
        dataset.train,
        ray_count=int(fixed_cfg.get("ray_count", batch_rays)),
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        seed=int(fixed_cfg.get("seed", training_seed + 104729)),
        chunk_rays=int(fixed_cfg.get("chunk_rays", train_cfg.get("eval_ray_chunk", 65536))),
        materialize_points=materialize_ray_points,
    )
    controller = DorflerController(
        spec=spec,
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

    ema_alpha = 2.0 / float(max(1, int(spec.get("ema_span", 50))) + 1)
    plateau_values = deque(maxlen=int(spec.get("window", 200)))
    plateau_ema = None
    fixed_last = None
    last_trigger = 0
    refining = True
    history, growth_events, evaluation_history = [], [], []
    dynamic_stop_iteration = None
    completed_iterations = 0
    start_time = time.perf_counter()
    for iteration in range(1, iterations + 1):
        completed_iterations = iteration
        if refining:
            ready, trigger_info = _plateau_ready(spec, controller, plateau_values, iteration, last_trigger)
            if ready:
                last_trigger = iteration
                plateau_values.clear()
                torch.save(
                    {"model": model.state_dict(), "config": config, "iteration": iteration - 1},
                    out_dir / "checkpoint_latest.pt",
                )
                growth_start = time.perf_counter()
                active, summary = controller.on_trigger(
                    model,
                    iteration=iteration,
                    noise_variance=float(noise_info.get("variance", 0.0)) if noise_info.get("enabled") else 0.0,
                    weight_kwargs=_projection_weight_kwargs(train_cfg, iteration=iteration),
                )
                optimizer = _make_optimizer(model, float(train_cfg.get("lr", 2e-2)))
                event = {
                    "iteration": iteration,
                    "operation": "dorfler_h_split",
                    "candidate_scope": "all_splittable_leaves",
                    "active": active,
                    "elapsed_sec": time.perf_counter() - growth_start,
                    **trigger_info,
                    **summary,
                    "structural_round_index": int(summary.get("dorfler_round", 0)),
                    "dynamic_refinement": {
                        "round": int(summary.get("dorfler_round", 0)),
                        "stop": bool(summary.get("stop", False)),
                        "stop_reasons": list(summary.get("stop_reasons", [])),
                    },
                }
                if summary.get("stop", False):
                    refining = False
                    dynamic_stop_iteration = min(
                        iterations, iteration + max(0, int(spec.get("final_finetune_iterations", 1000)))
                    )
                growth_events.append(event)
                print(json.dumps({"grow": event}))

        batch = random_training_rays(
            dataset.train,
            batch_rays=batch_rays,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            materialize_points=materialize_ray_points,
        )
        prediction = model.integrate_ray_batch(batch)
        loss = weighted_projection_loss(
            prediction, batch.target, **_projection_weight_kwargs(train_cfg, iteration=iteration)
        )
        projection_mse = F.mse_loss(prediction, batch.target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration == 1 or iteration % fixed_interval == 0:
            fixed_last = fixed_rss(model, fixed_set)
            plateau_ema = (
                fixed_last["mse"]
                if plateau_ema is None
                else (1.0 - ema_alpha) * plateau_ema + ema_alpha * fixed_last["mse"]
            )
            plateau_values.append(float(plateau_ema))

        if iteration == 1 or iteration % log_every == 0:
            stats = model.stats()
            rate_bytes = codec_bytes(model, controller.codec)
            distortion = float(fixed_last["mse"]) / max(float(noise_info.get("variance", 1.0)), 1.0e-12)
            row = {
                "iteration": iteration,
                "loss": float(loss.detach().item()),
                "projection_mse": float(projection_mse.detach().item()),
                "noise_normalized_distortion": distortion,
                "rate_bytes": int(rate_bytes),
                "plateau_loss": float(fixed_last["mse"]),
                "plateau_source": "fixed_decision_rays",
                "parameter_count": stats.parameter_count,
                "model_bytes": stats.model_bytes,
                "leaf_cells": stats.leaf_cells,
                "active_by_level": list(stats.active_by_level),
                "byte_budget": global_byte_budget,
                "within_byte_budget": rate_bytes <= global_byte_budget,
            }
            print(json.dumps(row))
            history.append(row)

        if compute_eval and eval_split is not None and (iteration % eval_every == 0 or iteration == iterations):
            sub_split = type(eval_split)(
                angles=eval_split.angles[:eval_views],
                projections=eval_split.projections[:eval_views],
                paths=eval_split.paths[:eval_views],
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
            print(json.dumps({"iteration": iteration, f"{eval_split_name}_projection": metric}))

        if dynamic_stop_iteration is not None and iteration >= dynamic_stop_iteration:
            break

    elapsed = time.perf_counter() - start_time
    trained_model = model
    model, codec_summary = _apply_export_codec(
        model,
        dataset,
        config,
        batch_rays=batch_rays,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        materialize_points=materialize_ray_points,
        weight_kwargs=_projection_weight_kwargs(train_cfg, iteration=completed_iterations),
    )
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
        projection_test = {"skipped": True, "reason": "disabled by metrics.compute_projection_metrics"}

    compute_volume = bool(metrics_cfg.get("compute_volume_metrics", True))
    compute_boundary = bool(metrics_cfg.get("compute_boundary_metrics", True))
    evaluation_dataset = dataset
    if compute_volume or compute_boundary:
        # The ground-truth volume is loaded only after optimization, for
        # evaluation. It never enters training or refinement.
        evaluation_dataset = load_r2_dataset(config["dataset"]["root"], device=device, load_volume=True)
    threshold = float(metrics_cfg.get("material_threshold", 0.1))
    material_volume = (
        material_volume_metrics(model, evaluation_dataset, threshold=threshold)
        if compute_volume
        else {"skipped": True, "reason": "disabled by metrics.compute_volume_metrics"}
    )
    boundary_sharpness = (
        boundary_sharpness_metrics(model, evaluation_dataset, threshold=threshold)
        if compute_boundary
        else {"skipped": True, "reason": "disabled by metrics.compute_boundary_metrics"}
    )

    checkpoint_path = out_dir / "checkpoint.pt"
    final_checkpoint = {"model": trained_model.state_dict(), "config": config, "iteration": completed_iterations}
    torch.save(final_checkpoint, checkpoint_path)
    torch.save(final_checkpoint, out_dir / "checkpoint_latest.pt")
    if model is not trained_model:
        torch.save(
            {"model": model.state_dict(), "config": config, "iteration": completed_iterations},
            out_dir / "checkpoint_codec.pt",
        )
    export_summary = {}
    compact_cfg = (config.get("export", {}) or {}).get("compact_octree", {}) or {}
    if bool(compact_cfg.get("enabled", True)):
        compact_summary = export_compact_octree_artifact(
            model,
            out_dir / str(compact_cfg.get("artifact_name", "compact_octree.npz")),
            quantization=str(compact_cfg.get("quantization", "float16")),
            topology=str(compact_cfg.get("topology", "packed_hierarchy")),
            checkpoint_path=checkpoint_path,
        )
        export_summary["compact_octree"] = compact_summary.__dict__
    export_summary["visual_slice"] = _export_center_slice_png(model, out_dir / "center_slice.png", dataset.volume_shape)
    export_summary["visual_slice_fullres"] = _export_center_slice_png(
        model, out_dir / "center_slice_fullres.png", _finest_model_shape(model, dataset.volume_shape)
    )

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
            "intermediate_split": eval_split_name if eval_split is not None else None,
            "test_used_during_training": eval_split is dataset.test,
            "test_role": "final evaluation only" if eval_split is not dataset.test else "legacy intermediate and final evaluation",
        },
        "elapsed_sec": elapsed,
        "total_elapsed_sec": time.perf_counter() - run_start_time,
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
            "global_objective": {
                "form": "minimize D(y, A_T[f]) subject to codec_bytes(T, f) <= B_max",
                "byte_budget": global_byte_budget,
                "rate_bytes_source": "deterministic codec byte model",
                "codec": controller.codec,
            },
            "noise_normalization": noise_info,
            "projection_loss": "weighted_least_squares",
            "dense_volume_loaded_during_training": False,
            "fixed_decision_rays": {
                "ray_count": int(fixed_set.observation_count),
                "interval": fixed_interval,
                "last_mse": float(fixed_last["mse"]) if fixed_last is not None else None,
                "selection_split": "train",
                "note": "Frozen training rays are the plateau ruler; optimization still uses random rays.",
            },
        },
        "dorfler": {
            "rounds": controller.round_index,
            "stopped": controller.stopped,
            "stop_reasons": list(controller.stop_reasons),
            "rollback_enabled": controller.rollback_enabled,
            "theta": controller.theta,
            "dev_role": "rollback verdict and diagnostics only",
        },
        "codec": codec_summary,
        "history": history,
        "evaluation_history": evaluation_history,
        "growth_events": growth_events,
        # Kept for viewer compatibility with older reports.
        "adaptive_events": [],
        "rate_distortion_curve": [],
        "exports": export_summary,
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_training(args.config)


if __name__ == "__main__":
    main()
