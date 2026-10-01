from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .compression import export_compact_octree_artifact, load_compact_octree_state_dict
from .config import load_config
from .data import load_r2_dataset
from .metrics import mae, projection_metrics, psnr
from .model import build_model
from .render import render_split


def _projection_gradient(image: torch.Tensor) -> torch.Tensor:
    gx = torch.zeros_like(image)
    gy = torch.zeros_like(image)
    gx[..., :, 1:-1] = 0.5 * (image[..., :, 2:] - image[..., :, :-2])
    gy[..., 1:-1, :] = 0.5 * (image[..., 2:, :] - image[..., :-2, :])
    return torch.sqrt(torch.square(gx) + torch.square(gy))


def projection_accuracy_diagnostics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    target_gradient = _projection_gradient(target)
    prediction_gradient = _projection_gradient(prediction)
    finite = target_gradient[torch.isfinite(target_gradient)]
    threshold = torch.quantile(finite, 0.99) if finite.numel() else target_gradient.new_tensor(0.0)
    mask = target_gradient >= threshold
    target_values = target[mask]
    prediction_values = prediction[mask]
    target_gradient_values = target_gradient[mask]
    prediction_gradient_values = prediction_gradient[mask]
    gradient_ratio = float(
        (prediction_gradient_values.mean() / target_gradient_values.mean().clamp_min(1.0e-8)).item()
    )
    return {
        "full": projection_metrics(prediction, target),
        "high_gradient_top1": {
            "threshold": float(threshold.item()),
            "ray_count": int(mask.sum().item()),
            "psnr": psnr(prediction_values, target_values, data_range=float(target.max().item())),
            "mae": mae(prediction_values, target_values),
            "gradient_ratio": gradient_ratio,
            "gradient_ratio_distance": abs(gradient_ratio - 1.0),
        },
    }


def evaluate_quantizations(
    *,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    quantizations: tuple[str, ...] = ("uint8", "uint16", "float16"),
) -> dict:
    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda"))
    dataset = load_r2_dataset(config["dataset"]["root"], device=device, load_volume=False)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = payload.get("config", config)
    source_model = build_model(checkpoint_config).to(device=device)
    source_model.prepare_sparse_from_state_dict(payload["model"])
    source_model.load_state_dict(payload["model"], strict=False)
    source_model.eval()
    detector_h, detector_w = dataset.detector_shape
    samples_per_ray = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))
    ray_chunk = int(config.get("training", {}).get("eval_ray_chunk", 65536))
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for quantization in quantizations:
        artifact_path = output_dir / f"compact_octree_{quantization}.npz"
        export_summary = export_compact_octree_artifact(
            source_model,
            artifact_path,
            quantization=quantization,
            topology="auto",
            checkpoint_path=checkpoint_path,
        )
        model = build_model(checkpoint_config).to(device=device)
        state, metadata = load_compact_octree_state_dict(model, artifact_path, device=device)
        model.prepare_sparse_from_state_dict(state)
        model.load_state_dict(state, strict=False)
        model.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rendered = render_split(
            model,
            dataset.test,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            ray_chunk=ray_chunk,
        )
        results[quantization] = {
            "artifact": str(artifact_path.resolve()),
            "schema": metadata.get("schema"),
            "export": export_summary.__dict__,
            "metrics": projection_accuracy_diagnostics(rendered, dataset.test.projections),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        }
        del model, rendered
    report = {
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "test_views": int(dataset.test.angles.shape[0]),
        "results": results,
    }
    report_path = output_dir / "quantization_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = evaluate_quantizations(
        config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.out),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
