from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from adaptive_ct.data import load_r2_dataset
from adaptive_ct.functional_compression import load_reference_model
from adaptive_ct.metrics import mae, psnr


def _slice_points(shape: tuple[int, int, int], axis: int, index: int, device: torch.device) -> torch.Tensor:
    free = [value for value in range(3) if value != axis]
    aa, bb = torch.meshgrid(
        torch.arange(shape[free[0]], device=device),
        torch.arange(shape[free[1]], device=device),
        indexing="ij",
    )
    coords = torch.empty((aa.numel(), 3), dtype=torch.float32, device=device)
    coords[:, axis] = float(index)
    coords[:, free[0]] = aa.reshape(-1)
    coords[:, free[1]] = bb.reshape(-1)
    scale = torch.tensor(shape, dtype=torch.float32, device=device)
    return -1.0 + (coords + 0.5) * 2.0 / scale


def _gradient(image: torch.Tensor) -> torch.Tensor:
    gx = image[1:, :-1] - image[:-1, :-1]
    gy = image[:-1, 1:] - image[:-1, :-1]
    return torch.sqrt(gx.square() + gy.square() + 1e-20)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate representative volume slices and high-gradient fidelity.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=1048576)
    args = parser.parse_args()

    device = torch.device(args.device)
    config, model = load_reference_model(args.config, args.checkpoint, device)
    dataset = load_r2_dataset(config["dataset"]["root"], device=device, load_volume=True)
    shape = tuple(int(value) for value in dataset.volume.shape)
    fractions = (0.25, 0.375, 0.5, 0.625, 0.75)
    rows = []
    for axis in range(3):
        for fraction in fractions:
            index = min(shape[axis] - 1, max(0, int(round((shape[axis] - 1) * fraction))))
            points = _slice_points(shape, axis, index, device)
            values = []
            with torch.no_grad():
                for start in range(0, points.shape[0], int(args.chunk)):
                    values.append(model.forward_mu(points[start : start + int(args.chunk)]))
            free = [value for value in range(3) if value != axis]
            pred = torch.cat(values).reshape(shape[free[0]], shape[free[1]])
            target = dataset.volume.select(axis, index)
            target_grad = _gradient(target)
            pred_grad = _gradient(pred)
            cutoff = torch.quantile(target_grad, 0.99)
            high = target_grad >= cutoff
            target_high = target[:-1, :-1][high]
            pred_high = pred[:-1, :-1][high]
            grad_target_mean = target_grad[high].mean().clamp_min(1e-12)
            grad_ratio = pred_grad[high].mean() / grad_target_mean
            rows.append(
                {
                    "axis": "xyz"[axis],
                    "index": index,
                    "psnr": psnr(pred, target),
                    "mae": mae(pred, target),
                    "high_gradient_psnr": psnr(pred_high, target_high),
                    "high_gradient_mae": mae(pred_high, target_high),
                    "gradient_ratio": float(grad_ratio.item()),
                    "gradient_ratio_distance": float(torch.abs(grad_ratio - 1.0).item()),
                }
            )
    aggregate = {}
    for key in (
        "psnr",
        "mae",
        "high_gradient_psnr",
        "high_gradient_mae",
        "gradient_ratio",
        "gradient_ratio_distance",
    ):
        aggregate[key] = float(sum(row[key] for row in rows) / len(rows))
    report = {
        "schema": "representative_slice_metrics_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "slice_count": len(rows),
        "aggregate": aggregate,
        "slices": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

