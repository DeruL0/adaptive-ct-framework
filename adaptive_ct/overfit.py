from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import load_config
from .data import load_r2_dataset
from .geometry import make_parallel_ray_points
from .model import build_model


def run_overfit(config_path: str | Path, *, iterations: int, batch_rays: int, min_drop: float) -> dict:
    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda"))
    dataset = load_r2_dataset(config["dataset"]["root"], device=device)
    detector_h, detector_w = dataset.detector_shape
    samples_per_ray = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))
    model = build_model(config).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"].get("lr", 2e-2)))
    angle = dataset.train.angles[0]
    losses = []
    for _ in range(int(iterations)):
        rows = torch.randint(0, detector_h, (batch_rays,), device=device)
        cols = torch.randint(0, detector_w, (batch_rays,), device=device)
        targets = dataset.train.projections[0, rows, cols]
        rb = make_parallel_ray_points(
            angle.expand(batch_rays),
            rows,
            cols,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            targets=targets,
        )
        pred = model.integrate_ray_batch(rb)
        loss = F.mse_loss(pred, rb.target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    result = {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "drop_ratio": losses[-1] / max(losses[0], 1e-20),
        "iterations": int(iterations),
        "batch_rays": int(batch_rays),
    }
    if result["drop_ratio"] > float(min_drop):
        raise RuntimeError(f"One-view overfit did not drop enough: {result}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--batch-rays", type=int, default=2048)
    parser.add_argument("--min-drop", type=float, default=0.95)
    args = parser.parse_args(argv)
    print(json.dumps(run_overfit(args.config, iterations=args.iterations, batch_rays=args.batch_rays, min_drop=args.min_drop), indent=2))


if __name__ == "__main__":
    main()

