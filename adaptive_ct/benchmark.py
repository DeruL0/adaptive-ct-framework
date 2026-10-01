from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Any

import torch

from .config import load_config
from .data import load_r2_dataset
from .metrics import projection_metrics
from .model import build_model
from .render import render_split
from .volume_metrics import boundary_sharpness_metrics, material_projection_metrics, material_volume_metrics


def _load_model(config: dict[str, Any], checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    checkpoint_config = payload.get("config", config) if isinstance(payload, dict) else config
    model = build_model(checkpoint_config).to(device=device)
    state_dict = payload.get("model", payload)
    model.prepare_sparse_from_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_training_report(config: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(config["output"]["dir"]).resolve() / "training_report.json"
    if not path.exists():
        return None
    return _load_json(path)


def _training_objective_from_config(config: dict[str, Any], training_report: dict[str, Any] | None) -> dict[str, Any]:
    if training_report is not None and isinstance(training_report.get("training_objective"), dict):
        return training_report["training_objective"]
    train_cfg = config.get("training", {}) or {}
    volume_loss = train_cfg.get("volume_loss", {}) or {}
    weight = float(volume_loss.get("weight", 0.0))
    return {
        "projection_loss": "mse",
        "tv_weight": float(train_cfg.get("tv_weight", 0.0)),
        "volume_loss": {
            "enabled": weight > 0.0,
            "weight": weight,
            "samples": int(volume_loss.get("samples", train_cfg.get("batch_rays", 4096))),
            "type": str(volume_loss.get("type", "mse")),
            "material_fraction": float(volume_loss.get("material_fraction", 0.5)),
            "boundary_fraction": float(volume_loss.get("boundary_fraction", 0.25)),
            "boundary_quantile": float(volume_loss.get("boundary_quantile", 0.9)),
            "start_iteration": int(volume_loss.get("start_iteration", 1)),
            "stop_iteration": volume_loss.get("stop_iteration"),
            "note": "Derived from config when the training report predates objective reporting.",
        },
    }


def _collect_baseline_sources(patterns: list[str]) -> list[dict[str, Any]]:
    sources = []
    seen = set()
    for pattern in patterns:
        for match in glob.glob(pattern):
            path = Path(match).resolve()
            if path in seen or not path.exists():
                continue
            seen.add(path)
            payload = _load_json(path)
            sources.append(
                {
                    "path": str(path),
                    "dataset": payload.get("dataset") or payload.get("case") or payload.get("schema"),
                    "protocol": payload.get("protocol"),
                    "methods": payload.get("methods"),
                    "datasets": payload.get("datasets"),
                }
            )
    return sources


def _storage_context(config: dict[str, Any], dataset, stats) -> dict[str, Any]:
    model_cfg = config.get("model", {}) or {}
    levels = [int(v) for v in model_cfg.get("levels", [])]
    scalar_volume_bytes = int(dataset.volume.numel() * dataset.volume.element_size())
    dense_leaf_voxel_bytes = int(sum(level ** 3 for level in levels) * dataset.volume.element_size())
    l1_density = None
    l2_density = None
    if len(levels) > 1 and levels[1] > 0:
        l1_density = float(stats.l1_active) / float(levels[1] ** 3)
    if len(levels) > 2 and levels[2] > 0:
        l2_density = float(stats.l2_active) / float(levels[2] ** 3)
    return {
        "current_scalar_volume_float32_bytes": scalar_volume_bytes,
        "current_dense_multiscale_leaf_voxel_bytes": dense_leaf_voxel_bytes,
        "current_sparse_model_bytes": stats.model_bytes,
        "sparse_model_vs_scalar_volume": stats.model_bytes / max(float(scalar_volume_bytes), 1.0),
        "sparse_model_vs_dense_multiscale_leaf_voxels": stats.model_bytes / max(float(dense_leaf_voxel_bytes), 1.0),
        "active_density": {
            "l0": float(stats.l0_active) / float(levels[0] ** 3) if levels else None,
            "l1": l1_density,
            "l2": l2_density,
        },
        "interpretation": (
            "The 96^3 figure case is a correctness and protocol-validation case. "
            "Do not claim paper-scale compression from this small volume alone; use the active densities and runtime as scaling diagnostics."
        ),
    }


def run_benchmark(
    *,
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    baseline_patterns: list[str],
    skip_material_projection: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda"))
    dataset = load_r2_dataset(config["dataset"]["root"], device=device)
    detector_h, detector_w = dataset.detector_shape
    samples_per_ray = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))
    ray_chunk = int(config["training"].get("eval_ray_chunk", 4096))
    threshold = float(config["metrics"].get("material_threshold", 0.1))

    model = _load_model(config, checkpoint_path, device)
    stats = model.stats()

    with torch.no_grad():
        start = time.perf_counter()
        rendered = render_split(
            model,
            dataset.test,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            ray_chunk=ray_chunk,
        )
        projection_elapsed = time.perf_counter() - start

    projection_domain = {
        "protocol": "stored held-out projection images; prediction rendered by dynamic leaf-voxel ray integration",
        "views": int(dataset.test.projections.shape[0]),
        "metrics": projection_metrics(rendered, dataset.test.projections),
        "elapsed_sec": projection_elapsed,
        "projections_per_sec": float(dataset.test.projections.shape[0]) / max(projection_elapsed, 1e-8),
    }

    material_volume_domain = {
        "protocol": "decoded attenuation compared on GT material voxels only",
        "threshold": threshold,
        "metrics": material_volume_metrics(model, dataset, threshold=threshold),
        "boundary_sharpness": boundary_sharpness_metrics(model, dataset, threshold=threshold),
    }

    material_projection_domain = {
        "protocol": "project decoded material volume and GT material volume; report full detector and GT material-ray mask",
        "threshold": threshold,
        "metrics": None,
        "skipped": bool(skip_material_projection),
    }
    if not skip_material_projection:
        material_projection_domain["metrics"] = material_projection_metrics(
            model,
            dataset,
            threshold=threshold,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
        )

    training_report = _maybe_training_report(config)
    training_objective = _training_objective_from_config(config, training_report)
    report = {
        "schema": "adaptive_ct_balanced_benchmark_v1",
        "method": "dynamic_leaf_voxel",
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": str(dataset.root),
        "scanner": {
            "volume_shape": list(dataset.volume_shape),
            "detector_shape": list(dataset.detector_shape),
            "train_views": int(dataset.train.projections.shape[0]),
            "test_views": int(dataset.test.projections.shape[0]),
            "samples_per_ray": samples_per_ray,
        },
        "protocols": {
            "projection_domain": projection_domain,
            "material_volume_domain": material_volume_domain,
            "material_projection_domain": material_projection_domain,
        },
        "storage_runtime": {
            "parameter_count": stats.parameter_count,
            "model_bytes": stats.model_bytes,
            "l0_cells": stats.l0_cells,
            "l0_active": stats.l0_active,
            "l1_active": stats.l1_active,
            "l2_active": stats.l2_active,
            "l3_active": getattr(stats, "l3_active", 0),
            "active_by_level": list(getattr(stats, "active_by_level", (stats.l0_active, stats.l1_active, stats.l2_active))),
            "leaf_cells": stats.leaf_cells,
            "max_depth": stats.max_depth,
            "train_time_sec": None if training_report is None else training_report.get("elapsed_sec"),
            "projection_eval_time_sec": projection_elapsed,
            "projections_per_sec": projection_domain["projections_per_sec"],
        },
        "storage_context": _storage_context(config, dataset, stats),
        "training_objective": training_objective,
        "baselines": {
            "note": "Baseline payloads are carried as separated source records. Do not merge projection-domain and material-domain metrics without protocol labels.",
            "sources": _collect_baseline_sources(baseline_patterns),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write a balanced adaptive CT benchmark report.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--skip-material-projection", action="store_true")
    args = parser.parse_args(argv)
    report = run_benchmark(
        config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.out),
        baseline_patterns=list(args.baseline),
        skip_material_projection=bool(args.skip_material_projection),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
