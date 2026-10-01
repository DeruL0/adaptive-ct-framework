from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .ceiling_diagnostics import _save_native_slices, grid_sample_dense_project
from .config import load_config
from .data import load_r2_dataset
from .metrics import projection_metrics
from .projection_domain import weighted_projection_loss


def _weight_kwargs(training: dict, iteration: int) -> dict:
    schedule = training.get("projection_weighting_schedule", {}) or {}
    selected = {}
    for start, value in sorted(((int(key), value) for key, value in schedule.items()), key=lambda item: item[0]):
        if int(iteration) >= start:
            selected = dict(value or {})
    return {
        "mode": str(selected.get("mode", "uniform")),
        "epsilon": float(selected.get("epsilon", 1.0e-4)),
        "target_power": float(selected.get("target_power", 1.0)),
        "blend_alpha": float(selected.get("blend_alpha", 0.15)),
        "min_weight": selected.get("min_weight"),
        "max_weight": selected.get("max_weight"),
    }


@torch.no_grad()
def render_dense_split(
    volume: torch.Tensor,
    split,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    ray_chunk: int,
) -> torch.Tensor:
    pixels_per_view = int(detector_h) * int(detector_w)
    total = int(split.angles.numel()) * pixels_per_view
    rendered = torch.empty((total,), dtype=torch.float32, device=volume.device)
    for start in range(0, total, int(ray_chunk)):
        linear = torch.arange(start, min(start + int(ray_chunk), total), device=volume.device)
        view_ids = torch.div(linear, pixels_per_view, rounding_mode="floor")
        pixels = linear - view_ids * pixels_per_view
        rows = torch.div(pixels, int(detector_w), rounding_mode="floor")
        cols = pixels - rows * int(detector_w)
        rendered[start : start + linear.numel()] = grid_sample_dense_project(
            volume,
            split.angles[view_ids],
            rows,
            cols,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
        )
    return rendered.reshape(int(split.angles.numel()), detector_h, detector_w)


@torch.no_grad()
def resample_original_gt(gt: np.ndarray, resolution: int, device: torch.device) -> torch.Tensor:
    res = int(resolution)
    source_shape = tuple(int(value) for value in gt.shape)
    target = torch.empty((res, res, res), dtype=torch.float32, device=device)
    for x_index in range(res):
        source_x = (float(x_index) + 0.5) * float(source_shape[0]) / float(res) - 0.5
        base = math.floor(source_x)
        x0 = max(0, min(source_shape[0] - 1, int(base)))
        x1 = max(0, min(source_shape[0] - 1, x0 + 1))
        alpha = float(source_x - base)
        pair = np.stack([np.asarray(gt[x0]), np.asarray(gt[x1])], axis=0).copy()
        source = torch.from_numpy(pair).to(device=device, dtype=torch.float32)[:, None]
        resized = F.interpolate(source, size=(res, res), mode="bilinear", align_corners=False)[:, 0]
        target[x_index] = resized[0] * (1.0 - alpha) + resized[1] * alpha
    return target


def _volume_metrics(prediction: torch.Tensor, target: torch.Tensor, threshold: float) -> dict:
    error = prediction - target
    data_range = float((target.max() - target.min()).clamp_min(1.0e-8).item())
    mse = float(torch.mean(torch.square(error)).item())
    material = target > float(threshold)
    material_mse = float(torch.mean(torch.square(error[material])).item())
    return {
        "full": {
            "psnr": 20.0 * math.log10(data_range) - 10.0 * math.log10(max(mse, 1.0e-20)),
            "mae": float(torch.mean(torch.abs(error)).item()),
            "rmse": math.sqrt(mse),
        },
        "material": {
            "threshold": float(threshold),
            "voxel_count": int(material.sum().item()),
            "psnr": 20.0 * math.log10(data_range) - 10.0 * math.log10(max(material_mse, 1.0e-20)),
            "mae": float(torch.mean(torch.abs(error[material])).item()),
        },
        "prediction_range": [float(prediction.min().item()), float(prediction.max().item())],
        "target_range": [float(target.min().item()), float(target.max().item())],
    }


def run_dense_baseline(config_path: Path) -> dict:
    config = load_config(config_path)
    dense = config["dense_baseline"]
    training = config["training"]
    seed = int(training.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(config.get("device", "cuda"))
    if device.type != "cuda":
        raise ValueError("The original-resolution dense baseline requires CUDA")
    dataset = load_r2_dataset(config["dataset"]["root"], device=device, load_volume=False)
    detector_h, detector_w = dataset.detector_shape
    resolution = int(dense.get("resolution", 384))
    samples_per_ray = int(config.get("geometry", {}).get("samples_per_ray", resolution))
    iterations = int(training.get("iterations", 15000))
    batch_rays = int(training.get("batch_rays", 8192))
    learning_rate = float(training.get("lr", 0.01))
    log_every = int(training.get("log_every", 100))
    eval_every = int(training.get("eval_every", 1000))
    eval_ray_chunk = int(training.get("eval_ray_chunk", 8192))
    material_threshold = float(config.get("metrics", {}).get("material_threshold", 0.10))
    output_dir = Path(config["output"]["dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_mu = float((dataset.train.projections.mean() * 0.5).item())
    volume = torch.nn.Parameter(
        torch.full((resolution, resolution, resolution), initial_mu, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam([volume], lr=learning_rate)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    history = []
    evaluation_history = []
    pixels_per_view = detector_h * detector_w
    for iteration in range(1, iterations + 1):
        view_ids = torch.randint(0, int(dataset.train.angles.numel()), (batch_rays,), device=device)
        rows = torch.randint(0, detector_h, (batch_rays,), device=device)
        cols = torch.randint(0, detector_w, (batch_rays,), device=device)
        target = dataset.train.projections[view_ids, rows, cols]
        prediction = grid_sample_dense_project(
            volume,
            dataset.train.angles[view_ids],
            rows,
            cols,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
        )
        projection_loss = weighted_projection_loss(
            prediction,
            target,
            **_weight_kwargs(training, iteration),
        )
        projection_mse = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        projection_loss.backward()
        optimizer.step()
        with torch.no_grad():
            volume.clamp_(min=0.0)

        if iteration == 1 or iteration % log_every == 0:
            row = {
                "iteration": iteration,
                "projection_loss": float(projection_loss.detach().item()),
                "projection_mse": float(projection_mse.detach().item()),
                "volume_min": float(volume.detach().min().item()),
                "volume_max": float(volume.detach().max().item()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if eval_every > 0 and iteration % eval_every == 0:
            rendered = render_dense_split(
                volume.detach(),
                dataset.test,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                ray_chunk=eval_ray_chunk,
            )
            metrics = projection_metrics(rendered, dataset.test.projections)
            evaluation = {"iteration": iteration, **metrics}
            evaluation_history.append(evaluation)
            print(json.dumps({"evaluation": evaluation}), flush=True)
            del rendered

    training_elapsed = time.perf_counter() - started
    final_projection = render_dense_split(
        volume.detach(),
        dataset.test,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=eval_ray_chunk,
    )
    final_projection_metrics = projection_metrics(final_projection, dataset.test.projections)
    final_train_projection = render_dense_split(
        volume.detach(),
        dataset.train,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=eval_ray_chunk,
    )
    final_train_projection_metrics = projection_metrics(final_train_projection, dataset.train.projections)

    dataset_root = Path(config["dataset"]["root"]).resolve()
    metadata = json.loads((dataset_root / "meta_data.json").read_text(encoding="utf-8"))
    gt = np.load(dataset_root / metadata["vol"], mmap_mode="r")
    target_volume = resample_original_gt(gt, resolution, device)
    volume_metrics = _volume_metrics(volume.detach(), target_volume, material_threshold)
    oracle_projection = render_dense_split(
        target_volume,
        dataset.test,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=eval_ray_chunk,
    )
    oracle_metrics = projection_metrics(oracle_projection, dataset.test.projections)

    center = resolution // 2
    slice_rows = [
        ("x-center", volume.detach()[center].cpu().numpy(), target_volume[center].cpu().numpy()),
        ("y-center", volume.detach()[:, center, :].cpu().numpy(), target_volume[:, center, :].cpu().numpy()),
        ("z-center", volume.detach()[:, :, center].cpu().numpy(), target_volume[:, :, center].cpu().numpy()),
    ]
    slices_path = output_dir / "dense384_center_slices.png"
    _save_native_slices(slice_rows, slices_path)
    volume_path = output_dir / "dense_volume_float16.npy"
    np.save(volume_path, volume.detach().to(dtype=torch.float16).cpu().numpy())
    report = {
        "schema": "dense_trilinear_original_resolution_v1",
        "config": str(config_path.resolve()),
        "dataset": {
            "root": str(dataset_root),
            "native_volume_shape": list(gt.shape),
            "detector_shape": [detector_h, detector_w],
            "train_views": int(dataset.train.angles.numel()),
            "test_views": int(dataset.test.angles.numel()),
        },
        "protocol": {
            "resolution": [resolution, resolution, resolution],
            "samples_per_ray": samples_per_ray,
            "iterations": iterations,
            "batch_rays": batch_rays,
            "learning_rate": learning_rate,
            "initial_mu": initial_mu,
            "nonnegative_projection": "clamp after Adam step",
            "regularization": "none",
            "test_views_used_for_training": False,
        },
        "projection_test": final_projection_metrics,
        "projection_train": final_train_projection_metrics,
        "projection_oracle_from_resampled_gt": oracle_metrics,
        "volume_384": volume_metrics,
        "history": history,
        "evaluation_history": evaluation_history,
        "training_elapsed_sec": training_elapsed,
        "total_elapsed_sec": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "artifacts": {
            "volume_float16": str(volume_path.resolve()),
            "center_slices": str(slices_path.resolve()),
        },
    }
    report_path = output_dir / "dense_baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train a dense trilinear CT ceiling baseline")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_dense_baseline(args.config), indent=2))


if __name__ == "__main__":
    main()
