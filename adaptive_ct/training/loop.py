"""Training entry point: training loop, evaluation, export, report."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace as dataclass_replace
from pathlib import Path

import torch

from adaptive_ct.codec.artifact import export_octree_artifact
from adaptive_ct.codec.budget import model_bytes
from adaptive_ct.core.config import load_config
from adaptive_ct.data.dataset import load_r2_dataset
from adaptive_ct.data.splits import split_projection_views
from adaptive_ct.evaluation.metrics import projection_metrics
from adaptive_ct.evaluation.render import render_split
from adaptive_ct.evaluation.volume import boundary_sharpness_metrics, material_volume_metrics
from adaptive_ct.refinement.controller import RefinementController
from adaptive_ct.representation.factory import build_model
from adaptive_ct.training.export import deploy, export_center_slice_png, finest_model_shape
from adaptive_ct.training.regularizer import EdgePenalty
from adaptive_ct.training.solve import solve
from adaptive_ct.training.validate import validate_config


@torch.no_grad()
def _initialize_constant_mu(model, value: torch.Tensor) -> None:
    model.values.fill_(float(value.item()))


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
    model = build_model(config).to(device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Normalized [-1, 1]^3 box: the representative path length is two, so the
    # constant start value comes directly from the projections.
    _initialize_constant_mu(model, (dataset.train.projections.mean() * 0.5).clamp_min(1e-8))
    if "trilinear" in model.leaf_types:  # main line: every root leaf starts trilinear
        model.to_trilinear(torch.arange(model.leaf_count, device=device))

    train_cfg = config["training"]
    solver_cfg = train_cfg["solver"]
    ref_cfg = train_cfg["refinement"]
    eval_chunk = int(train_cfg.get("eval_ray_chunk", 1 << 20))
    controller = RefinementController(
        spec=ref_cfg,
        edge_penalty=solver_cfg["edge_penalty"],
        estimator_split=dataset.train,
        detector_h=detector_h,
        detector_w=detector_w,
        seed=training_seed,
        byte_budget=int(ref_cfg["byte_budget"]),
    )

    def held_out_metric():
        split = dataset.validation
        if split is None:
            return None
        with torch.no_grad():
            rendered = render_split(model, split, detector_h=detector_h, detector_w=detector_w, ray_chunk=eval_chunk)
        return projection_metrics(rendered, split.projections)

    rounds, history, growth_events = [], [], []
    max_rounds = int(ref_cfg["max_rounds"])
    start_time = time.perf_counter()
    for r in range(max_rounds + 1):
        final = controller.stopped or r == max_rounds
        passes = int(solver_cfg["final_passes"] if final else solver_cfg["round_passes"])
        tol = float(solver_cfg["final_tol"] if final else solver_cfg["round_tol"])
        group_passes = int(solver_cfg["first_group_passes"] if r == 0 else solver_cfg["group_passes"])
        # Detector binning while the finest leaf still spans >= min_leaf_px binned pixels; the final solve is unbinned.
        finest = model.level_shapes[int(model.leaf_levels.max())]
        leaf_px = min(detector_h / finest[2], detector_w / finest[0])
        detector_bin = 1 if final else max([1] + [b for b in (2, 4) if leaf_px / b >= float(solver_cfg["min_leaf_px"])])
        t0 = time.perf_counter()
        edge = solver_cfg["edge_penalty"]
        weight = float(edge["weight"])
        penalty = EdgePenalty(model, float(edge["delta"])) if weight > 0 else None
        _, hist, converged = solve(model, dataset.train, passes=passes, tol=tol, groups=int(solver_cfg["groups"]),
                                   group_passes=group_passes, extrapolate=bool(solver_cfg["extrapolate"]),
                                   detector_bin=detector_bin, penalty=penalty, penalty_weight=weight)
        del penalty
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
            "leaves": model.leaf_count,
            "bytes": model_bytes(model),
            "leaf_type_counts": list(model.stats().leaf_type_counts),
            "held_out_projection": held_out_metric(),
            "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0,
        }
        history.extend({**h, "round": r} for h in hist)
        rounds.append(entry)
        print(json.dumps({"round": entry}, default=float), flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if final:
            break
        event = controller.refine(model)
        growth_events.append(event)
        print(json.dumps({"grow": event}, default=float), flush=True)

    elapsed = time.perf_counter() - start_time
    codec_summary = deploy(model, dataset, detector_h=detector_h, detector_w=detector_w, ray_chunk=eval_chunk)
    stats = model.stats()
    metrics_cfg = config.get("metrics", {}) or {}
    if bool(metrics_cfg.get("compute_projection_metrics", True)):
        final_render = render_split(
            model,
            dataset.test,
            detector_h=detector_h,
            detector_w=detector_w,
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

    checkpoint_path = out_dir / "checkpoint.pt"  # the deployed model (coefficients rounded to fp16)
    torch.save({"model": model.state_dict(), "config": config, "rounds": len(rounds)}, checkpoint_path)
    export_summary = {}
    export_summary["stored_file"] = export_octree_artifact(model, out_dir / "model.npz").__dict__  # what the budget counts
    export_summary["visual_slice"] = export_center_slice_png(model, out_dir / "center_slice.png", dataset.volume_shape)
    export_summary["visual_slice_fullres"] = export_center_slice_png(model, out_dir / "center_slice_fullres.png", finest_model_shape(model))

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
        "training_objective": {
            "form": "min_{c>=0} 1/2 ||A c - y||^2 + beta R(c) per fixed tree; tree and leaf types chosen by held-out refits of the same objective under the byte budget",
            "byte_budget": int(ref_cfg["byte_budget"]),
            "solver": solver_cfg,
            "dense_volume_loaded_during_training": False,
        },
        "refinement": {
            "rounds": controller.round_index,
            "stop_reasons": list(controller.stop_reasons),
            "action_share": float(ref_cfg["action_share"]),
        },
        "rounds": rounds,
        "codec": codec_summary,
        "history": history,
        "growth_events": growth_events,
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
