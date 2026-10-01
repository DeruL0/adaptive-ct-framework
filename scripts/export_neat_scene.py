"""Export the current adaptive-CT parallel-beam dataset to NeAT's scene format."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def as_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.drive:
        drive = resolved.drive.rstrip(":").lower()
        tail = resolved.as_posix().split(":", 1)[1]
        return f"/mnt/{drive}{tail}"
    return resolved.as_posix()


def rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    return x, y, z, w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    metadata = json.loads((source / "meta_data.json").read_text(encoding="utf-8"))
    frames = [("train", item) for item in metadata["proj_train"]]
    frames += [("test", item) for item in metadata["proj_test"]]

    image_dir = destination / "images"
    split_dir = destination / "figure473_128_unique"
    image_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    image_names: list[str] = []
    poses: list[str] = []
    for index, (_, item) in enumerate(frames):
        projection = np.load(source / item["file_path"]).astype(np.float32)
        encoded = np.rint(np.clip(projection, 0.0, 1.0) * 65535.0).astype(np.uint16)
        image_name = f"{index:04d}.png"
        Image.fromarray(encoded, mode="I;16").save(image_dir / image_name)
        image_names.append(image_name)

        angle = float(item["angle"])
        forward = np.array([math.cos(angle), -math.sin(angle), 0.0], dtype=np.float64)
        right = np.array([math.sin(angle), math.cos(angle), 0.0], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        rotation = np.column_stack((right, up, forward))
        x, y, z, w = rotation_to_quaternion(rotation)
        translation = -3.0 * forward
        poses.append(
            f"{x:.15e} {y:.15e} {z:.15e} {w:.15e} "
            f"{translation[0]:.15e} {translation[1]:.15e} {translation[2]:.15e}"
        )

    (destination / "images.txt").write_text("\n".join(image_names) + "\n", encoding="utf-8", newline="\n")
    (destination / "poses.txt").write_text("\n".join(poses) + "\n", encoding="utf-8", newline="\n")
    (destination / "camera_indices.txt").write_text("0\n" * len(frames), encoding="utf-8", newline="\n")
    (split_dir / "train.txt").write_text(
        "\n".join(map(str, range(128))) + "\n", encoding="utf-8", newline="\n"
    )
    (split_dir / "eval.txt").write_text(
        "\n".join(map(str, range(128, 132))) + "\n", encoding="utf-8", newline="\n"
    )

    width = int(metadata["scanner"]["nDetector"][1])
    height = int(metadata["scanner"]["nDetector"][0])
    camera_ini = (
        "[CameraBase]\n"
        f"w = {width}\n"
        f"h = {height}\n"
        f"K = {width / 2:.15e} {height / 2:.15e} "
        f"{(width - 1) / 2:.15e} {(height - 1) / 2:.15e} 0.000000000000000e+00\n"
    )
    (destination / "camera.ini").write_text(camera_ini, encoding="utf-8", newline="\n")
    dataset_ini = (
        "[DatasetParams]\n"
        f"image_dir = {as_wsl_path(image_dir)}\n"
        "mask_dir = \n"
        "projection_factor = 1\n"
        "vis_volume_intensity_factor = 1\n"
        "scene_scale = 1\n"
        "xray_min = 0\n"
        "xray_max = 65535\n"
        "volume_file = \n"
        "log_space_input = true\n"
        "use_log10_conversion = false\n"
        "parallel_beam = true\n"
    )
    (destination / "dataset.ini").write_text(dataset_ini, encoding="utf-8", newline="\n")
    print(f"Exported {len(frames)} projections to {destination}")


if __name__ == "__main__":
    main()
