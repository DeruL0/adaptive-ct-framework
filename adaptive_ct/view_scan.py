from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .backend import project_dense_parallel


def uniform_train_angles(view_count: int) -> np.ndarray:
    count = int(view_count)
    if count < 1:
        raise ValueError("view_count must be positive")
    # Parallel-beam projections at theta + pi are detector-reversed copies of
    # theta. Sampling a full turn therefore wastes half of the acquisition
    # budget. Use one half-turn so every stored view is independent.
    return np.arange(count, dtype=np.float64) * (math.pi / float(count))


def common_test_angles(test_count: int = 4) -> np.ndarray:
    """Angles shared by the 48/128 scans and disjoint from both train grids."""

    count = int(test_count)
    if count < 1:
        raise ValueError("test_count must be positive")
    common_grid_half_step = math.pi / 384.0
    return common_grid_half_step + np.arange(count, dtype=np.float64) * (math.pi / float(count))


def interleaved_holdout_angles(
    view_count: int,
    *,
    train_views: int,
    phase: float,
) -> np.ndarray:
    """Uniform half-turn holdout grid shifted off the training angles.

    ``phase`` is measured in fractions of one training-view interval. Separate
    phases provide disjoint validation and final-test grids without changing
    the angular density of either holdout split.
    """

    count = int(view_count)
    training_count = int(train_views)
    if count < 1 or training_count < 1:
        raise ValueError("view_count and train_views must be positive")
    phase_value = float(phase)
    if not 0.0 < phase_value < 1.0:
        raise ValueError("phase must lie strictly between 0 and 1")
    angles = (
        np.arange(count, dtype=np.float64) * (math.pi / float(count))
        + phase_value * (math.pi / float(training_count))
    )
    return np.sort(np.mod(angles, math.pi))


def _assert_disjoint(train_angles: np.ndarray, test_angles: np.ndarray, tolerance: float = 1.0e-10) -> None:
    wrapped = np.abs(
        np.angle(np.exp(1j * (train_angles[:, None] - test_angles[None, :])))
    )
    if float(wrapped.min()) <= float(tolerance):
        raise ValueError("test angles overlap the training angle grid")


def generate_original_view_dataset(
    *,
    source_root: Path,
    output_root: Path,
    train_views: int,
    validation_views: int = 0,
    test_views: int = 4,
    samples_per_ray: int = 1024,
) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    metadata = json.loads((source_root / "meta_data.json").read_text(encoding="utf-8"))
    volume_path = (source_root / metadata["vol"]).resolve()
    volume_array = np.load(volume_path, mmap_mode="r")
    shape = tuple(int(value) for value in volume_array.shape)
    detector_h, detector_w = (int(value) for value in metadata["scanner"]["nDetector"])
    train_angles = uniform_train_angles(train_views)
    if int(validation_views) > 0:
        validation_angles = interleaved_holdout_angles(
            validation_views,
            train_views=train_views,
            phase=1.0 / 3.0,
        )
        test_angles = interleaved_holdout_angles(
            test_views,
            train_views=train_views,
            phase=2.0 / 3.0,
        )
        _assert_disjoint(train_angles, validation_angles)
        _assert_disjoint(validation_angles, test_angles)
    else:
        validation_angles = np.empty((0,), dtype=np.float64)
        test_angles = common_test_angles(test_views)
    _assert_disjoint(train_angles, test_angles)

    output_root.mkdir(parents=True, exist_ok=True)
    train_dir = output_root / "proj_train"
    validation_dir = output_root / "proj_val"
    test_dir = output_root / "proj_test"
    train_dir.mkdir(exist_ok=True)
    if validation_angles.size:
        validation_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    device = torch.device("cuda")
    volume = torch.from_numpy(np.asarray(volume_array)).to(device=device, dtype=torch.float32)

    def render_set(angles: np.ndarray, directory: Path, prefix: str) -> list[dict]:
        frames = []
        for index, angle in enumerate(angles):
            angle_tensor = torch.tensor([float(angle)], dtype=torch.float32, device=device)
            projection = project_dense_parallel(
                volume,
                angle_tensor,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=int(samples_per_ray),
            )[0]
            path = directory / f"{prefix}_{index:04d}.npy"
            np.save(path, projection.cpu().numpy().astype(np.float32, copy=False))
            frames.append(
                {
                    "file_path": str(path.relative_to(output_root)).replace("\\", "/"),
                    "angle": float(angle),
                }
            )
        return frames

    train_frames = render_set(train_angles, train_dir, "proj_train")
    validation_frames = (
        render_set(validation_angles, validation_dir, "proj_val")
        if validation_angles.size
        else []
    )
    test_frames = render_set(test_angles, test_dir, "proj_test")
    output_metadata = {
        "scanner": metadata["scanner"],
        "vol": str(volume_path),
        "proj_train": train_frames,
        "proj_val": validation_frames,
        "proj_test": test_frames,
        "source": {
            "base_dataset": str(source_root),
            "source_volume": str(volume_path),
            "shape_dhw": list(shape),
            "note": "Original Figure473 resolution; projections generated by adaptive_ct native dense projector.",
            "train_angle_protocol": f"{int(train_views)} independent parallel-beam angles over [0, pi)",
            "validation_angle_protocol": (
                f"{int(validation_views)} independent angles over [0, pi), shifted by one-third "
                "of a training interval"
                if validation_frames
                else "none"
            ),
            "test_angle_protocol": (
                f"{int(test_views)} independent angles over [0, pi), shifted by two-thirds "
                "of a training interval; final evaluation only"
                if validation_frames
                else (
                    f"{int(test_views)} independent angles over [0, pi) with pi/384 offset; "
                    "shared across view-count scans"
                )
            ),
            "samples_per_ray": int(samples_per_ray),
        },
    }
    metadata_path = output_root / "meta_data.json"
    metadata_path.write_text(json.dumps(output_metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output_root),
        "volume_shape": list(shape),
        "detector_shape": [detector_h, detector_w],
        "train_views": len(train_frames),
        "validation_views": len(validation_frames),
        "test_views": len(test_frames),
        "metadata": str(metadata_path),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate original-resolution Figure473 view scans")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-views", type=int, required=True)
    parser.add_argument("--validation-views", type=int, default=0)
    parser.add_argument("--test-views", type=int, default=4)
    parser.add_argument("--samples-per-ray", type=int, default=1024)
    args = parser.parse_args(argv)
    report = generate_original_view_dataset(
        source_root=args.source,
        output_root=args.out,
        train_views=args.train_views,
        validation_views=args.validation_views,
        test_views=args.test_views,
        samples_per_ray=args.samples_per_ray,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
