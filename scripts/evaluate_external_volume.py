from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates


def _resample_voxel_centres(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    axes = [
        (np.arange(size, dtype=np.float32) + 0.5) * (volume.shape[axis] / float(size)) - 0.5
        for axis, size in enumerate(shape)
    ]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    return map_coordinates(volume, [xx, yy, zz], order=1, mode="constant", cval=0.0).astype(np.float32)


def _psnr(prediction: np.ndarray, target: np.ndarray, data_range: float) -> float:
    mse = float(np.mean(np.square(prediction.astype(np.float64) - target.astype(np.float64))))
    return 20.0 * math.log10(max(data_range, 1.0e-8)) - 10.0 * math.log10(max(mse, 1.0e-20))


def _metrics(prediction: np.ndarray, target: np.ndarray, data_range: float) -> dict[str, float]:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    return {
        "psnr": _psnr(prediction, target, data_range),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
    }


def _best_axis_alignment(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    sample = tuple(slice(None, None, 4) for _ in range(3))
    target_sample = target[sample].astype(np.float64)
    target_centered = target_sample - target_sample.mean()
    target_norm = np.linalg.norm(target_centered)
    best = None
    for permutation in itertools.permutations(range(3)):
        candidate = np.transpose(prediction, permutation)
        for flips in itertools.product((False, True), repeat=3):
            aligned = candidate
            for axis, enabled in enumerate(flips):
                if enabled:
                    aligned = np.flip(aligned, axis=axis)
            aligned_sample = aligned[sample].astype(np.float64)
            centered = aligned_sample - aligned_sample.mean()
            correlation = float(np.dot(centered.ravel(), target_centered.ravel())) / max(
                float(np.linalg.norm(centered) * target_norm), 1.0e-20
            )
            if best is None or correlation > best[0]:
                best = (correlation, permutation, flips)
    assert best is not None
    correlation, permutation, flips = best
    aligned = np.transpose(prediction, permutation)
    for axis, enabled in enumerate(flips):
        if enabled:
            aligned = np.flip(aligned, axis=axis)
    return np.ascontiguousarray(aligned), {
        "permutation": list(permutation),
        "flips": list(flips),
        "sampled_correlation": correlation,
    }


def _save_slices(prediction: np.ndarray, target: np.ndarray, output_path: Path) -> None:
    error_limit = float(np.quantile(np.abs(prediction - target), 0.99))
    vmax = float(target.max())
    figure, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    for row, label in enumerate(("x", "y", "z")):
        index = target.shape[row] // 2
        pred_slice = np.take(prediction, index, axis=row)
        target_slice = np.take(target, index, axis=row)
        error = pred_slice - target_slice
        axes[row, 0].imshow(pred_slice, cmap="gray", vmin=0.0, vmax=vmax)
        axes[row, 0].set_title(f"Prediction {label}-center")
        axes[row, 1].imshow(target_slice, cmap="gray", vmin=0.0, vmax=vmax)
        axes[row, 1].set_title(f"Ground truth {label}-center")
        image = axes[row, 2].imshow(error, cmap="coolwarm", vmin=-error_limit, vmax=error_limit)
        axes[row, 2].set_title(f"Error {label}-center")
        for axis in axes[row]:
            axis.axis("off")
    figure.colorbar(image, ax=axes[:, 2], shrink=0.7, label="prediction - ground truth")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an exported raw float32 volume against CT ground truth")
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--material-threshold", type=float, default=0.1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    shape = tuple(int(value) for value in args.shape)
    raw = np.fromfile(args.volume, dtype="<f4")
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"Expected {np.prod(shape)} float32 values, found {raw.size}")
    prediction = raw.reshape(shape)
    ground_truth_native = np.load(args.ground_truth, mmap_mode="r")
    target = _resample_voxel_centres(ground_truth_native, shape)
    prediction, alignment = _best_axis_alignment(prediction, target)

    denominator = float(np.sum(np.square(prediction.astype(np.float64))))
    scale = float(np.sum(prediction.astype(np.float64) * target.astype(np.float64)) / max(denominator, 1.0e-20))
    scaled = prediction * scale
    target_range = float(target.max() - target.min())
    material = target > float(args.material_threshold)
    background = ~material

    args.out.mkdir(parents=True, exist_ok=True)
    slices_path = args.out / "center_slices.png"
    _save_slices(scaled, target, slices_path)
    report = {
        "schema": "external_volume_evaluation_v1",
        "volume": str(args.volume.resolve()),
        "ground_truth": str(args.ground_truth.resolve()),
        "shape": list(shape),
        "alignment": alignment,
        "least_squares_scale": scale,
        "raw_full": _metrics(prediction, target, target_range),
        "scaled_full": _metrics(scaled, target, target_range),
        "scaled_material": {
            "threshold": float(args.material_threshold),
            "voxel_count": int(material.sum()),
            **_metrics(scaled[material], target[material], target_range),
        },
        "scaled_background": {
            "voxel_count": int(background.sum()),
            "prediction_mean": float(scaled[background].mean()),
            "prediction_mae": float(np.mean(np.abs(scaled[background] - target[background]))),
        },
        "prediction_range": [float(scaled.min()), float(scaled.max())],
        "target_range": [float(target.min()), float(target.max())],
        "artifacts": {"center_slices": str(slices_path.resolve())},
    }
    report_path = args.out / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
