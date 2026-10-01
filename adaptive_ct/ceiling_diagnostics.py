from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .bernstein import unique_degree_rows
from .config import load_config
from .data import load_r2_dataset
from .metrics import mae, psnr
from .model import build_model
from .refinement import decoded_volume, gradient_magnitude


def _load_checkpoint(config_path: Path, checkpoint_path: Path, device: torch.device):
    config = load_config(config_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = payload.get("config", config)
    model = build_model(model_config).to(device=device)
    model.prepare_sparse_from_state_dict(payload["model"])
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return config, model


def _volume_metrics(prediction: torch.Tensor, target: torch.Tensor, material_threshold: float) -> dict:
    error = prediction - target
    material = target > float(material_threshold)
    target_gradient = gradient_magnitude(target)
    prediction_gradient = gradient_magnitude(prediction)
    finite_gradient = target_gradient[torch.isfinite(target_gradient)]
    boundary_threshold = torch.quantile(finite_gradient, 0.99)
    boundary = target_gradient >= boundary_threshold
    result = {
        "full": {
            "psnr": psnr(prediction, target),
            "mae": mae(prediction, target),
            "rmse": float(torch.sqrt(torch.mean(torch.square(error))).item()),
        },
        "material": {
            "threshold": float(material_threshold),
            "voxel_count": int(material.sum().item()),
            "fraction": float(material.float().mean().item()),
            "psnr": psnr(prediction[material], target[material]),
            "mae": mae(prediction[material], target[material]),
        },
        "high_gradient_top1": {
            "threshold": float(boundary_threshold.item()),
            "voxel_count": int(boundary.sum().item()),
            "psnr": psnr(prediction[boundary], target[boundary]),
            "mae": mae(prediction[boundary], target[boundary]),
            "gradient_ratio": float(
                (
                    prediction_gradient[boundary].mean()
                    / target_gradient[boundary].mean().clamp_min(1.0e-8)
                ).item()
            ),
        },
        "ranges": {
            "prediction": [float(prediction.min().item()), float(prediction.max().item())],
            "target": [float(target.min().item()), float(target.max().item())],
        },
    }
    return result


def _leaf_mean_coefficients(model) -> torch.Tensor:
    coefficients = model.coefficients().detach()
    means = torch.empty((model.leaf_levels.numel(),), dtype=coefficients.dtype, device=coefficients.device)
    for degree in unique_degree_rows(model.leaf_degrees):
        mask = torch.all(model.leaf_degrees == degree, dim=1)
        leaf_ids = torch.nonzero(mask, as_tuple=False).reshape(-1)
        count = int(torch.prod(degree + 1).item())
        offsets = torch.arange(count, dtype=torch.long, device=coefficients.device)
        coefficient_ids = model.coefficient_offsets[leaf_ids, None] + offsets[None, :]
        means[leaf_ids] = coefficients[coefficient_ids].mean(dim=1)
    return means


@torch.no_grad()
def _decode_native_center_slice(model, shape: tuple[int, int, int], axis: int, index: int, chunk: int):
    other_axes = [value for value in range(3) if value != int(axis)]
    first = torch.arange(shape[other_axes[0]], dtype=torch.float32, device=model.coefficient_logits.device)
    second = torch.arange(shape[other_axes[1]], dtype=torch.float32, device=model.coefficient_logits.device)
    aa, bb = torch.meshgrid(first, second, indexing="ij")
    coords = torch.empty((aa.numel(), 3), dtype=torch.float32, device=aa.device)
    coords[:, axis] = float(index)
    coords[:, other_axes[0]] = aa.reshape(-1)
    coords[:, other_axes[1]] = bb.reshape(-1)
    scale = torch.tensor(shape, dtype=torch.float32, device=aa.device)
    points = -1.0 + (coords + 0.5) * 2.0 / scale
    values = []
    for start in range(0, points.shape[0], int(chunk)):
        values.append(model.forward_mu(points[start : start + int(chunk)]).detach().cpu())
    return torch.cat(values).reshape(shape[other_axes[0]], shape[other_axes[1]]).numpy()


def _slice_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(np.square(error)))
    data_range = max(float(np.max(target) - np.min(target)), 1.0e-8)
    return {
        "psnr": 20.0 * math.log10(data_range) - 10.0 * math.log10(max(mse, 1.0e-20)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": math.sqrt(mse),
        "prediction_range": [float(np.min(prediction)), float(np.max(prediction))],
        "target_range": [float(np.min(target)), float(np.max(target))],
    }


def _save_native_slices(rows: list[tuple[str, np.ndarray, np.ndarray]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, plots = plt.subplots(len(rows), 3, figsize=(13, 4 * len(rows)), constrained_layout=True)
    error_limit = max(float(np.quantile(np.abs(pred - gt), 0.99)) for _, pred, gt in rows)
    for row, (name, pred, gt) in enumerate(rows):
        vmax = float(np.max(gt))
        plots[row, 0].imshow(pred.T, cmap="gray", vmin=0.0, vmax=vmax, origin="lower")
        plots[row, 0].set_title(f"Prediction {name}")
        plots[row, 1].imshow(gt.T, cmap="gray", vmin=0.0, vmax=vmax, origin="lower")
        plots[row, 1].set_title(f"Ground truth {name}")
        image = plots[row, 2].imshow(
            (pred - gt).T,
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
            origin="lower",
        )
        plots[row, 2].set_title(f"Error {name}")
        for plot in plots[row]:
            plot.axis("off")
    fig.colorbar(image, ax=plots[:, 2], shrink=0.7, label="prediction - ground truth")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _root_gt_statistics(
    gt: np.ndarray,
    root_resolution: int | tuple[int, int, int] | list[int],
) -> tuple[np.ndarray, np.ndarray]:
    root = (
        (int(root_resolution),) * 3
        if isinstance(root_resolution, int)
        else tuple(int(value) for value in root_resolution)
    )
    means = np.empty(root, dtype=np.float32)
    maxima = np.empty_like(means)
    source_x = int(gt.shape[0])
    for x_index in range(root[0]):
        start = int(math.floor(x_index * source_x / root[0]))
        stop = int(math.ceil((x_index + 1) * source_x / root[0]))
        slab = torch.from_numpy(np.asarray(gt[start:stop]).copy()).to(dtype=torch.float32)
        mean_plane = slab.mean(dim=0)[None, None]
        max_plane = slab.max(dim=0).values[None, None]
        means[x_index] = F.adaptive_avg_pool2d(mean_plane, root[1:])[0, 0].numpy()
        maxima[x_index] = F.adaptive_max_pool2d(max_plane, root[1:])[0, 0].numpy()
    return means, maxima


def _save_l0_spatial_map(
    l0_mean_grid: np.ndarray,
    gt_root_mean: np.ndarray,
    output_path: Path,
    air_threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    centers = tuple(value // 2 for value in l0_mean_grid.shape)
    slices = (
        ("x-center", l0_mean_grid[centers[0]], gt_root_mean[centers[0]]),
        ("y-center", l0_mean_grid[:, centers[1], :], gt_root_mean[:, centers[1], :]),
        ("z-center", l0_mean_grid[:, :, centers[2]], gt_root_mean[:, :, centers[2]]),
    )
    fig, plots = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
    for row, (name, l0_values, gt_values) in enumerate(slices):
        plots[row, 0].imshow(np.nan_to_num(l0_values, nan=0.0).T, cmap="magma", origin="lower", vmin=0.0, vmax=0.2)
        plots[row, 0].set_title(f"L0 leaf mean {name}")
        plots[row, 1].imshow((gt_values > float(air_threshold)).T, cmap="gray", origin="lower", vmin=0, vmax=1)
        plots[row, 1].set_title(f"GT root mean > {air_threshold:g}")
        trapped = np.isfinite(l0_values) & (gt_values > float(air_threshold))
        plots[row, 2].imshow(trapped.T, cmap="Reds", origin="lower", vmin=0, vmax=1)
        plots[row, 2].set_title("GT-selected root cell still represented by L0")
        for plot in plots[row]:
            plot.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def _resampled_384_metrics_and_slices(
    model,
    gt: np.ndarray,
    *,
    resolution: int,
    material_threshold: float,
    point_chunk: int,
) -> tuple[dict, list[tuple[str, np.ndarray, np.ndarray]]]:
    device = model.coefficient_logits.device
    res = int(resolution)
    source_shape = tuple(int(value) for value in gt.shape)
    axis = torch.arange(res, dtype=torch.float32, device=device)
    yy, zz = torch.meshgrid(axis, axis, indexing="ij")
    y_points = -1.0 + (yy.reshape(-1) + 0.5) * 2.0 / float(res)
    z_points = -1.0 + (zz.reshape(-1) + 0.5) * 2.0 / float(res)
    center = res // 2
    pred_x = gt_x = None
    pred_y = np.empty((res, res), dtype=np.float32)
    gt_y = np.empty((res, res), dtype=np.float32)
    pred_z = np.empty((res, res), dtype=np.float32)
    gt_z = np.empty((res, res), dtype=np.float32)
    sums = {"sq": 0.0, "abs": 0.0, "count": 0, "material_sq": 0.0, "material_abs": 0.0, "material_count": 0}
    target_min = float("inf")
    target_max = -float("inf")
    for x_index in range(res):
        x_point = -1.0 + (float(x_index) + 0.5) * 2.0 / float(res)
        points = torch.stack(
            [torch.full_like(y_points, x_point), y_points, z_points],
            dim=1,
        )
        pred_chunks = []
        for start in range(0, points.shape[0], int(point_chunk)):
            pred_chunks.append(model.forward_mu(points[start : start + int(point_chunk)]).detach())
        pred_plane = torch.cat(pred_chunks).reshape(res, res)

        source_x = (float(x_index) + 0.5) * float(source_shape[0]) / float(res) - 0.5
        x0 = max(0, min(source_shape[0] - 1, int(math.floor(source_x))))
        x1 = max(0, min(source_shape[0] - 1, x0 + 1))
        alpha = float(source_x - math.floor(source_x))
        source_pair = np.stack([np.asarray(gt[x0]), np.asarray(gt[x1])], axis=0).copy()
        source_tensor = torch.from_numpy(source_pair).to(device=device, dtype=torch.float32)[:, None]
        resized = F.interpolate(source_tensor, size=(res, res), mode="bilinear", align_corners=False)[:, 0]
        target_plane = resized[0] * (1.0 - alpha) + resized[1] * alpha

        error = pred_plane - target_plane
        sums["sq"] += float(torch.sum(torch.square(error)).item())
        sums["abs"] += float(torch.sum(torch.abs(error)).item())
        sums["count"] += int(error.numel())
        material = target_plane > float(material_threshold)
        if torch.any(material):
            sums["material_sq"] += float(torch.sum(torch.square(error[material])).item())
            sums["material_abs"] += float(torch.sum(torch.abs(error[material])).item())
            sums["material_count"] += int(material.sum().item())
        target_min = min(target_min, float(target_plane.min().item()))
        target_max = max(target_max, float(target_plane.max().item()))
        pred_cpu = pred_plane.cpu().numpy()
        target_cpu = target_plane.cpu().numpy()
        if x_index == center:
            pred_x, gt_x = pred_cpu.copy(), target_cpu.copy()
        pred_y[x_index] = pred_cpu[center]
        gt_y[x_index] = target_cpu[center]
        pred_z[x_index] = pred_cpu[:, center]
        gt_z[x_index] = target_cpu[:, center]

    mse = sums["sq"] / max(sums["count"], 1)
    material_mse = sums["material_sq"] / max(sums["material_count"], 1)
    data_range = max(target_max - target_min, 1.0e-8)
    metrics = {
        "protocol": f"model and original GT sampled at {res}^3 voxel centres",
        "source_gt_shape": list(source_shape),
        "resolution": [res, res, res],
        "full": {
            "psnr": 20.0 * math.log10(data_range) - 10.0 * math.log10(max(mse, 1.0e-20)),
            "mae": sums["abs"] / max(sums["count"], 1),
            "rmse": math.sqrt(mse),
        },
        "material": {
            "threshold": float(material_threshold),
            "voxel_count": sums["material_count"],
            "psnr": 20.0 * math.log10(data_range) - 10.0 * math.log10(max(material_mse, 1.0e-20)),
            "mae": sums["material_abs"] / max(sums["material_count"], 1),
        },
        "target_range": [target_min, target_max],
    }
    return metrics, [("x-center", pred_x, gt_x), ("y-center", pred_y, gt_y), ("z-center", pred_z, gt_z)]


@torch.no_grad()
def diagnose_original_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    training_report_path: Path,
    output_dir: Path,
    material_threshold: float,
    diagnostic_resolution: int = 384,
) -> dict:
    device = torch.device("cuda")
    config, model = _load_checkpoint(config_path, checkpoint_path, device)
    dataset_root = Path(config["dataset"]["root"]).resolve()
    metadata = json.loads((dataset_root / "meta_data.json").read_text(encoding="utf-8"))
    gt_path = dataset_root / metadata["vol"]
    gt = np.load(gt_path, mmap_mode="r")
    shape = tuple(int(value) for value in gt.shape)
    output_dir.mkdir(parents=True, exist_ok=True)

    native_rows = []
    native_metrics = {}
    for axis, name in enumerate(("x-center", "y-center", "z-center")):
        index = shape[axis] // 2
        prediction = _decode_native_center_slice(model, shape, axis, index, chunk=262144)
        target = np.take(gt, index, axis=axis)
        native_rows.append((name, prediction, np.asarray(target)))
        native_metrics[name] = _slice_metrics(prediction, np.asarray(target))
    native_slice_path = output_dir / "native_center_slices.png"
    _save_native_slices(native_rows, native_slice_path)

    resampled_metrics, resampled_rows = _resampled_384_metrics_and_slices(
        model,
        gt,
        resolution=int(diagnostic_resolution),
        material_threshold=float(material_threshold),
        point_chunk=262144,
    )
    resampled_slice_path = output_dir / f"volume_center_slices_{int(diagnostic_resolution)}.png"
    _save_native_slices(resampled_rows, resampled_slice_path)

    leaf_means = _leaf_mean_coefficients(model)
    l0_mask = model.leaf_levels == 0
    l0 = leaf_means[l0_mask]
    level_shapes = model.level_shapes if hasattr(model, "level_shapes") else [(int(value),) * 3 for value in model.level_resolutions]
    root_shape = tuple(int(value) for value in level_shapes[0])
    root_count = math.prod(root_shape)
    thresholds = (0.001, 0.005, 0.01, 0.02, 0.05, float(material_threshold))
    occupied = {}
    for threshold in thresholds:
        count = int(torch.sum(l0 > threshold).item())
        occupied[str(threshold)] = {
            "count": count,
            "fraction_of_l0": count / max(float(l0.numel()), 1.0),
            "fraction_of_all_root_cells": count / float(root_count),
        }
    l0_mean_grid = np.full(root_shape, np.nan, dtype=np.float32)
    l0_coords = model.leaf_coords[l0_mask].detach().cpu().numpy()
    l0_mean_grid[l0_coords[:, 0], l0_coords[:, 1], l0_coords[:, 2]] = l0.detach().cpu().numpy()
    gt_root_mean, gt_root_max = _root_gt_statistics(gt, root_shape)
    l0_grid_mask = np.isfinite(l0_mean_grid)
    gt_support = gt_root_mean > 0.01
    gt_material_mean = gt_root_mean > float(material_threshold)
    gt_material_intersection = gt_root_max > float(material_threshold)
    l0_spatial_path = output_dir / "l0_spatial_distribution.png"
    _save_l0_spatial_map(l0_mean_grid, gt_root_mean, l0_spatial_path, float(material_threshold))
    spatial = {
        "gt_support_root_cells": int(np.sum(gt_support)),
        "gt_support_still_l0": int(np.sum(gt_support & l0_grid_mask)),
        "gt_support_still_l0_fraction": float(np.sum(gt_support & l0_grid_mask) / max(np.sum(gt_support), 1)),
        "gt_material_mean_root_cells": int(np.sum(gt_material_mean)),
        "gt_material_mean_still_l0": int(np.sum(gt_material_mean & l0_grid_mask)),
        "gt_material_mean_still_l0_fraction": float(
            np.sum(gt_material_mean & l0_grid_mask) / max(np.sum(gt_material_mean), 1)
        ),
        "gt_material_intersection_root_cells": int(np.sum(gt_material_intersection)),
        "gt_material_intersection_still_l0": int(np.sum(gt_material_intersection & l0_grid_mask)),
        "gt_material_intersection_still_l0_fraction": float(
            np.sum(gt_material_intersection & l0_grid_mask) / max(np.sum(gt_material_intersection), 1)
        ),
    }
    report = {
        "schema": "ct_original_resolution_diagnostics_v1",
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": {
            "root": str(dataset_root),
            "gt_path": str(gt_path.resolve()),
            "volume_shape": list(shape),
            "detector_shape": list(metadata["scanner"]["nDetector"]),
            "train_views": len(metadata["proj_train"]),
            "test_views": len(metadata["proj_test"]),
        },
        "native_center_slice_metrics": native_metrics,
        "resampled_volume_metrics": resampled_metrics,
        "l0_occupancy": {
            "root_resolution": list(root_shape),
            "all_root_cells": root_count,
            "l0_leaf_count": int(l0.numel()),
            "split_root_cells": root_count - int(l0.numel()),
            "occupied_by_leaf_mean": occupied,
            "ground_truth_spatial": spatial,
            "p_active_by_level": [
                int(torch.sum((model.leaf_levels == level) & torch.any(model.leaf_degrees > 0, dim=1)).item())
                for level in range(len(level_shapes))
            ],
        },
        "convergence": _convergence(training_report_path),
        "artifacts": {
            "native_center_slices": str(native_slice_path.resolve()),
            "resampled_center_slices": str(resampled_slice_path.resolve()),
            "l0_spatial_distribution": str(l0_spatial_path.resolve()),
        },
    }
    report_path = output_dir / "original_checkpoint_diagnostics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _l0_occupancy(model, target: torch.Tensor, material_threshold: float) -> dict:
    l0_ids = torch.nonzero(model.leaf_levels == 0, as_tuple=False).reshape(-1)
    means = _leaf_mean_coefficients(model)[l0_ids]
    coords = model.leaf_coords[l0_ids].to(dtype=torch.long)
    level_shapes = model.level_shapes if hasattr(model, "level_shapes") else [(int(value),) * 3 for value in model.level_resolutions]
    root_resolution = tuple(int(value) for value in level_shapes[0])
    shape = tuple(int(value) for value in target.shape)
    if any(value % root != 0 for value, root in zip(shape, root_resolution)):
        raise ValueError("GT dimensions must be divisible by the root resolution for exact L0 diagnostics")
    stride = tuple(value // root for value, root in zip(shape, root_resolution))
    gt_mean = torch.empty_like(means)
    gt_max = torch.empty_like(means)
    gt_material_fraction = torch.empty_like(means)
    for index, coord in enumerate(coords.tolist()):
        x, y, z = coord
        block = target[
            x * stride[0] : (x + 1) * stride[0],
            y * stride[1] : (y + 1) * stride[1],
            z * stride[2] : (z + 1) * stride[2],
        ]
        gt_mean[index] = block.mean()
        gt_max[index] = block.max()
        gt_material_fraction[index] = (block > float(material_threshold)).float().mean()

    def count(mask: torch.Tensor) -> int:
        return int(mask.sum().item())

    total = max(int(l0_ids.numel()), 1)
    criteria = {
        "predicted_mean_gt_0.01": means > 0.01,
        "predicted_mean_gt_material_threshold": means > float(material_threshold),
        "gt_mean_gt_0.01": gt_mean > 0.01,
        "gt_any_material": gt_max > float(material_threshold),
        "gt_material_fraction_gt_0.10": gt_material_fraction > 0.10,
        "gt_material_fraction_gt_0.50": gt_material_fraction > 0.50,
    }
    occupied = {
        name: {"count": count(mask), "fraction_of_l0": count(mask) / float(total)}
        for name, mask in criteria.items()
    }
    return {
        "l0_leaf_count": int(l0_ids.numel()),
        "root_resolution": list(root_resolution),
        "gt_voxels_per_l0_cell": list(stride),
        "material_threshold": float(material_threshold),
        "occupied": occupied,
        "predicted_cell_mean_quantiles": {
            str(q): float(torch.quantile(means, q).item()) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
        "gt_cell_mean_quantiles": {
            str(q): float(torch.quantile(gt_mean, q).item()) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
        "p_active_by_level": [
            int(torch.sum((model.leaf_levels == level) & torch.any(model.leaf_degrees > 0, dim=1)).item())
            for level in range(len(level_shapes))
        ],
    }


def _convergence(training_report_path: Path) -> dict:
    report = json.loads(training_report_path.read_text(encoding="utf-8"))
    history = report.get("history", [])
    evaluations = report.get("evaluation_history", [])
    if not history:
        return {"available": False}

    def row_at_or_after(iteration: int):
        return min(history, key=lambda row: abs(int(row["iteration"]) - iteration))

    checkpoints = [row_at_or_after(iteration) for iteration in (10000, 12000, 14000, 15000)]
    unique = {int(row["iteration"]): row for row in checkpoints}
    projection_losses = {iteration: float(row["projection_loss"]) for iteration, row in sorted(unique.items())}
    first_iteration = min(projection_losses)
    last_iteration = max(projection_losses)
    first_loss = projection_losses[first_iteration]
    last_loss = projection_losses[last_iteration]
    eval_tail = evaluations[-5:]
    return {
        "available": True,
        "projection_loss_checkpoints": projection_losses,
        "relative_change_first_to_last": (last_loss - first_loss) / max(abs(first_loss), 1.0e-20),
        "evaluation_tail": eval_tail,
        "best_evaluation": max(evaluations, key=lambda row: float(row["psnr"])) if evaluations else None,
        "final_evaluation": evaluations[-1] if evaluations else None,
    }


def _save_slices(prediction: torch.Tensor, target: torch.Tensor, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    pred = prediction.detach().cpu().numpy()
    gt = target.detach().cpu().numpy()
    axes = ((0, "sagittal"), (1, "coronal"), (2, "axial"))
    fig, plots = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    vmax = float(gt.max())
    error_limit = float(np.quantile(np.abs(pred - gt), 0.99))
    for row, (axis, title) in enumerate(axes):
        index = gt.shape[axis] // 2
        pred_slice = np.take(pred, index, axis=axis)
        gt_slice = np.take(gt, index, axis=axis)
        error_slice = pred_slice - gt_slice
        plots[row, 0].imshow(pred_slice, cmap="gray", vmin=0.0, vmax=vmax)
        plots[row, 0].set_title(f"Prediction {title}")
        plots[row, 1].imshow(gt_slice, cmap="gray", vmin=0.0, vmax=vmax)
        plots[row, 1].set_title(f"Ground truth {title}")
        im = plots[row, 2].imshow(error_slice, cmap="coolwarm", vmin=-error_limit, vmax=error_limit)
        plots[row, 2].set_title(f"Error {title}")
        for plot in plots[row]:
            plot.axis("off")
    fig.colorbar(im, ax=plots[:, 2], shrink=0.7, label="prediction - ground truth")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def diagnose_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    training_report_path: Path,
    output_dir: Path,
    material_threshold: float,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, model = _load_checkpoint(config_path, checkpoint_path, device)
    dataset = load_r2_dataset(config["dataset"]["root"], device=device, load_volume=True)
    target = dataset.volume.detach()
    prediction = decoded_volume(model, tuple(int(value) for value in target.shape))
    output_dir.mkdir(parents=True, exist_ok=True)
    slices_path = output_dir / "volume_center_slices.png"
    _save_slices(prediction, target, slices_path)
    report = {
        "schema": "ct_ceiling_diagnostics_v1",
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": {
            "root": str(dataset.root),
            "volume_shape": list(dataset.volume_shape),
            "detector_shape": list(dataset.detector_shape),
            "train_views": int(dataset.train.angles.numel()),
            "test_views": int(dataset.test.angles.numel()),
        },
        "volume_metrics": _volume_metrics(prediction, target, material_threshold),
        "l0_occupancy": _l0_occupancy(model, target, material_threshold),
        "convergence": _convergence(training_report_path),
        "artifacts": {"center_slices": str(slices_path.resolve())},
    }
    report_path = output_dir / "checkpoint_diagnostics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def differentiable_dense_project(
    volume: torch.Tensor,
    angles: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
) -> torch.Tensor:
    """Differentiable counterpart of the native normalized dense projector."""

    angles = -angles.to(dtype=torch.float32)
    rows_f = rows.to(dtype=torch.float32)
    cols_f = cols.to(dtype=torch.float32)
    ca = torch.cos(angles)
    sa = torch.sin(angles)
    direction_x = ca
    direction_y = sa
    base_x = -sa * (-1.0 + (cols_f + 0.5) * 2.0 / float(detector_w))
    base_y = ca * (-1.0 + (cols_f + 0.5) * 2.0 / float(detector_w))
    z = -1.0 + (rows_f + 0.5) * 2.0 / float(detector_h)

    eps = 1.0e-8
    t_min = torch.full_like(base_x, -1.0e20)
    t_max = torch.full_like(base_x, 1.0e20)
    valid = torch.ones_like(base_x, dtype=torch.bool)
    for base, direction in ((base_x, direction_x), (base_y, direction_y)):
        parallel = torch.abs(direction) < eps
        valid &= (~parallel) | ((base >= -1.0) & (base <= 1.0))
        denominator = torch.where(parallel, torch.ones_like(direction), direction)
        t0 = (-1.0 - base) / denominator
        t1 = (1.0 - base) / denominator
        t_min = torch.where(parallel, t_min, torch.maximum(t_min, torch.minimum(t0, t1)))
        t_max = torch.where(parallel, t_max, torch.minimum(t_max, torch.maximum(t0, t1)))
    valid &= t_max > t_min
    t_min = torch.where(valid, t_min, torch.zeros_like(t_min))
    t_max = torch.where(valid, t_max, torch.zeros_like(t_max))
    step = (t_max - t_min) / float(samples_per_ray)
    sample_ids = torch.arange(samples_per_ray, dtype=torch.float32, device=volume.device)
    t = t_min[:, None] + (sample_ids[None, :] + 0.5) * step[:, None]
    x = base_x[:, None] + direction_x[:, None] * t
    y = base_y[:, None] + direction_y[:, None] * t
    z_samples = z[:, None].expand_as(x)
    nx, ny, nz = (int(value) for value in volume.shape)
    fx = (x + 1.0) * 0.5 * float(nx) - 0.5
    fy = (y + 1.0) * 0.5 * float(ny) - 0.5
    fz = (z_samples + 1.0) * 0.5 * float(nz) - 0.5
    valid_samples = (
        (fx >= 0.0)
        & (fy >= 0.0)
        & (fz >= 0.0)
        & (fx <= float(nx - 1))
        & (fy <= float(ny - 1))
        & (fz <= float(nz - 1))
    )
    x0 = torch.floor(fx).to(dtype=torch.long).clamp(0, nx - 1)
    y0 = torch.floor(fy).to(dtype=torch.long).clamp(0, ny - 1)
    z0 = torch.floor(fz).to(dtype=torch.long).clamp(0, nz - 1)
    x1 = (x0 + 1).clamp_max(nx - 1)
    y1 = (y0 + 1).clamp_max(ny - 1)
    z1 = (z0 + 1).clamp_max(nz - 1)
    tx = fx - x0.to(dtype=fx.dtype)
    ty = fy - y0.to(dtype=fy.dtype)
    tz = fz - z0.to(dtype=fz.dtype)
    flat = volume.reshape(-1)

    def gather(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
        return flat[ix * (ny * nz) + iy * nz + iz]

    c000 = gather(x0, y0, z0)
    c001 = gather(x0, y0, z1)
    c010 = gather(x0, y1, z0)
    c011 = gather(x0, y1, z1)
    c100 = gather(x1, y0, z0)
    c101 = gather(x1, y0, z1)
    c110 = gather(x1, y1, z0)
    c111 = gather(x1, y1, z1)
    c00 = c000 * (1.0 - tx) + c100 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    sampled = (c0 * (1.0 - tz) + c1 * tz) * valid_samples.to(dtype=volume.dtype)
    return torch.sum(sampled, dim=1) * step


def grid_sample_dense_project(
    volume: torch.Tensor,
    angles: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
) -> torch.Tensor:
    """Fast trilinear dense projector for the optimization ceiling baseline."""

    angles = -angles.to(dtype=torch.float32)
    rows_f = rows.to(dtype=torch.float32)
    cols_f = cols.to(dtype=torch.float32)
    ca = torch.cos(angles)
    sa = torch.sin(angles)
    direction_xy = torch.stack([ca, sa], dim=1)
    u_axis_xy = torch.stack([-sa, ca], dim=1)
    u = -1.0 + (cols_f + 0.5) * 2.0 / float(detector_w)
    z = -1.0 + (rows_f + 0.5) * 2.0 / float(detector_h)
    base_xy = u_axis_xy * u[:, None]

    t_min = torch.full_like(u, -1.0e20)
    t_max = torch.full_like(u, 1.0e20)
    valid = torch.ones_like(u, dtype=torch.bool)
    for axis in range(2):
        base = base_xy[:, axis]
        direction = direction_xy[:, axis]
        parallel = torch.abs(direction) < 1.0e-8
        valid &= (~parallel) | ((base >= -1.0) & (base <= 1.0))
        denominator = torch.where(parallel, torch.ones_like(direction), direction)
        t0 = (-1.0 - base) / denominator
        t1 = (1.0 - base) / denominator
        t_min = torch.where(parallel, t_min, torch.maximum(t_min, torch.minimum(t0, t1)))
        t_max = torch.where(parallel, t_max, torch.minimum(t_max, torch.maximum(t0, t1)))
    valid &= t_max > t_min
    t_min = torch.where(valid, t_min, torch.zeros_like(t_min))
    t_max = torch.where(valid, t_max, torch.zeros_like(t_max))
    step = (t_max - t_min) / float(samples_per_ray)
    sample_ids = torch.arange(samples_per_ray, device=volume.device, dtype=torch.float32)
    t = t_min[:, None] + (sample_ids[None, :] + 0.5) * step[:, None]
    xy = base_xy[:, None, :] + direction_xy[:, None, :] * t[:, :, None]
    z_values = z[:, None].expand(-1, samples_per_ray)
    # PyTorch's 5D grid is ordered W,H,D. The CT tensor is indexed X,Y,Z.
    grid = torch.stack([z_values, xy[:, :, 1], xy[:, :, 0]], dim=-1)
    grid = grid.reshape(1, int(angles.numel()), int(samples_per_ray), 1, 3)
    sampled = F.grid_sample(
        volume[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(int(angles.numel()), int(samples_per_ray))
    nx, ny, nz = (int(value) for value in volume.shape)
    inside_centres = (
        (xy[:, :, 0] >= -1.0 + 1.0 / nx)
        & (xy[:, :, 0] <= 1.0 - 1.0 / nx)
        & (xy[:, :, 1] >= -1.0 + 1.0 / ny)
        & (xy[:, :, 1] <= 1.0 - 1.0 / ny)
        & (z_values >= -1.0 + 1.0 / nz)
        & (z_values <= 1.0 - 1.0 / nz)
    )
    sampled = sampled * inside_centres.to(dtype=sampled.dtype)
    return torch.sum(sampled, dim=1) * step


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose CT volume quality and spatial allocation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--material-threshold", type=float, default=0.10)
    parser.add_argument("--original-resolution", action="store_true")
    parser.add_argument("--diagnostic-resolution", type=int, default=384)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.original_resolution:
        report = diagnose_original_checkpoint(
            args.config,
            args.checkpoint,
            args.training_report,
            args.out,
            args.material_threshold,
            diagnostic_resolution=args.diagnostic_resolution,
        )
    else:
        report = diagnose_checkpoint(
            args.config,
            args.checkpoint,
            args.training_report,
            args.out,
            args.material_threshold,
        )
    report["elapsed_sec"] = time.perf_counter() - started
    report_name = "original_checkpoint_diagnostics.json" if args.original_resolution else "checkpoint_diagnostics.json"
    (args.out / report_name).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
