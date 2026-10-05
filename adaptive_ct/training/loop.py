"""Training entry point: training loop, evaluation, export, report."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace as dataclass_replace
from pathlib import Path

import torch

from adaptive_ct.codec.artifact import export_compact_octree_artifact
from adaptive_ct.codec.budget import codec_bytes
from adaptive_ct.core.config import load_config
from adaptive_ct.core.loss import projection_weight_kwargs
from adaptive_ct.data.dataset import load_r2_dataset
from adaptive_ct.data.splits import split_projection_views
from adaptive_ct.evaluation.metrics import projection_metrics
from adaptive_ct.evaluation.render import render_split
from adaptive_ct.evaluation.volume import boundary_sharpness_metrics, material_volume_metrics
from adaptive_ct.refinement.controller import RefinementController
from adaptive_ct.representation.factory import build_model
from adaptive_ct.training.export import apply_export_codec, export_center_slice_png, finest_model_shape
from adaptive_ct.training.solve import solve
from adaptive_ct.training.validate import validate_config


@torch.no_grad()
def _initialize_constant_mu(model, value: torch.Tensor) -> None:
    if hasattr(model, "initialize_constant_mu"):
        model.initialize_constant_mu(float(value.item()))
        return
    values = torch.full_like(model.coefficient_logits, float(value.item()))
    model.coefficient_logits.copy_(model._coefficients_to_logits(values))


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
    """Rounds of: solve the fixed tree to convergence -> score leaves -> split a batch; then a final strict solve."""
    run_start_time = time.perf_counter()
    config = load_config(config_path)
    validate_config(config)
    training_seed = int((config.get("training", {}) or {}).get("seed", 0))
    torch.manual_seed(training_seed)
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
    solver_cfg = train_cfg["solver"]
    ref_cfg = train_cfg["refinement"]
    eval_chunk = int(train_cfg.get("eval_ray_chunk", 65536))
    weight_kwargs = projection_weight_kwargs(train_cfg, iteration=1)
    controller = RefinementController(
        spec=ref_cfg,
        estimator_split=dataset.train,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        materialize_points=materialize_ray_points,
        chunk_rays=eval_chunk,
        seed=training_seed,
        byte_budget=int(ref_cfg["byte_budget"]),
    )

    def held_out_metric():
        split = dataset.validation
        if split is None:
            return None
        with torch.no_grad():
            rendered = render_split(model, split, detector_h=detector_h, detector_w=detector_w, samples_per_ray=samples_per_ray, ray_chunk=eval_chunk)
        return projection_metrics(rendered, split.projections)

    rounds, history, growth_events = [], [], []
    max_rounds = int(ref_cfg.get("max_rounds", 32))
    start_time = time.perf_counter()
    for r in range(max_rounds + 1):
        final = controller.stopped or r == max_rounds
        passes = int(solver_cfg["final_passes"] if final else solver_cfg["round_passes"])
        tol = float(solver_cfg["final_tol"] if final else solver_cfg["round_tol"])
        group_passes = int(solver_cfg.get("first_group_passes", 4) if r == 0 else solver_cfg.get("group_passes", 2))
        # Detector binning while the finest leaf still spans >= min_leaf_px binned pixels; the final solve is unbinned.
        finest = model.level_shapes[int(model.leaf_levels.max())]
        leaf_px = min(detector_h / finest[2], detector_w / finest[0])
        detector_bin = 1 if final else max([1] + [b for b in (2, 4) if leaf_px / b >= float(solver_cfg.get("min_leaf_px", 2.5))])
        t0 = time.perf_counter()
        _, hist, converged = solve(model, dataset.train, passes=passes, tol=tol, groups=int(solver_cfg.get("groups", 24)),
                                   group_passes=group_passes, extrapolate=bool(solver_cfg.get("extrapolate", True)),
                                   detector_bin=detector_bin)
        last = hist[-1] if hist else {}
        entry = {
            "round": r,
            "final": final,
            "passes": int(last.get("pass", 0)),
            "converged": converged,
            "detector_bin": detector_bin,
            "solve_seconds": time.perf_counter() - t0,
            "objective": last.get("objective_at_eval_point"),
            "optimality_residual": last.get("optimality_residual"),
            "leaves": int(model.leaf_levels.shape[0]),
            "codec_bytes": codec_bytes(model, controller.codec),
            "held_out_projection": held_out_metric(),
        }
        history.extend({**h, "round": r} for h in hist)
        rounds.append(entry)
        print(json.dumps({"round": entry}, default=float), flush=True)
        if final:
            break
        _, event = controller.refine(model, weight_kwargs=weight_kwargs)
        growth_events.append(event)
        print(json.dumps({"grow": event}, default=float), flush=True)

    elapsed = time.perf_counter() - start_time
    trained_model = model
    model, codec_summary = apply_export_codec(
        model,
        dataset,
        config,
        batch_rays=int(((config.get("export", {}) or {}).get("codec") or {}).get("finetune_batch_rays", 65536)),
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        materialize_points=materialize_ray_points,
        weight_kwargs=weight_kwargs,
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
            ray_chunk=eval_chunk,
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
    final_checkpoint = {"model": trained_model.state_dict(), "config": config, "rounds": len(rounds)}
    torch.save(final_checkpoint, checkpoint_path)
    torch.save(final_checkpoint, out_dir / "checkpoint_latest.pt")
    if model is not trained_model:
        torch.save(
            {"model": model.state_dict(), "config": config, "rounds": len(rounds)},
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
        "dataset": str(dataset.root),
        "evaluation_protocol": {
            "training_views": int(dataset.train.angles.numel()),
            "validation_views": int(dataset.validation.angles.numel()) if dataset.validation is not None else 0,
            "test_views": int(dataset.test.angles.numel()),
            "held_out_role": "per-round projection check only (not used for any decision)",
            "test_role": "final evaluation only",
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
            "form": "min_{c>=0} 1/2 ||A c - y||^2 per fixed tree; tree grows until codec_bytes <= byte_budget",
            "byte_budget": int(ref_cfg["byte_budget"]),
            "codec": controller.codec,
            "solver": solver_cfg,
            "dense_volume_loaded_during_training": False,
        },
        "refinement": {
            "rounds": controller.round_index,
            "stop_reasons": list(controller.stop_reasons),
            "gain_share": controller.share,
        },
        "rounds": rounds,
        "codec": codec_summary,
        "history": history,
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
