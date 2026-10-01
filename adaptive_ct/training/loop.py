"""Training entry point: training loop, evaluation, export, report."""

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

from adaptive_ct.codec.artifact import export_compact_octree_artifact
from adaptive_ct.codec.budget import codec_bytes
from adaptive_ct.core.config import load_config
from adaptive_ct.core.geometry import random_training_rays
from adaptive_ct.core.loss import projection_weight_kwargs, weighted_projection_loss
from adaptive_ct.data.dataset import load_r2_dataset
from adaptive_ct.data.rays import FixedDecisionRaySet, fixed_rss
from adaptive_ct.data.splits import split_projection_views
from adaptive_ct.evaluation.metrics import projection_metrics
from adaptive_ct.evaluation.render import render_split
from adaptive_ct.evaluation.volume import boundary_sharpness_metrics, material_volume_metrics
from adaptive_ct.refinement.controller import DorflerController
from adaptive_ct.representation.factory import build_model
from adaptive_ct.training.export import apply_export_codec, export_center_slice_png, finest_model_shape
from adaptive_ct.training.plateau import plateau_ready
from adaptive_ct.training.validate import active_milestone, validate_config


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


def run_training(config_path: str | Path) -> dict:
    run_start_time = time.perf_counter()
    config = load_config(config_path)
    validate_config(config)
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

    spec = active_milestone(config)
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
            ready, trigger_info = plateau_ready(spec, controller, plateau_values, iteration, last_trigger)
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
                    weight_kwargs=projection_weight_kwargs(train_cfg, iteration=iteration),
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
            prediction, batch.target, **projection_weight_kwargs(train_cfg, iteration=iteration)
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
    model, codec_summary = apply_export_codec(
        model,
        dataset,
        config,
        batch_rays=batch_rays,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        materialize_points=materialize_ray_points,
        weight_kwargs=projection_weight_kwargs(train_cfg, iteration=completed_iterations),
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
    export_summary["visual_slice"] = export_center_slice_png(model, out_dir / "center_slice.png", dataset.volume_shape)
    export_summary["visual_slice_fullres"] = export_center_slice_png(
        model, out_dir / "center_slice_fullres.png", finest_model_shape(model, dataset.volume_shape)
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
