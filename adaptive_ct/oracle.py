from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .backend import project_dense_parallel
from .config import load_config
from .data import load_r2_dataset
from .metrics import projection_metrics


def run_oracle(config_path: str | Path) -> dict:
    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda"))
    dataset = load_r2_dataset(config["dataset"]["root"], device=device)
    detector_h, detector_w = dataset.detector_shape
    samples = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))
    pred = project_dense_parallel(
        dataset.volume,
        dataset.test.angles,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples,
    )
    per_view = []
    for idx in range(pred.shape[0]):
        per_view.append(projection_metrics(pred[idx], dataset.test.projections[idx]))
    mean_psnr = sum(v["psnr"] for v in per_view) / len(per_view)
    mean_mae = sum(v["mae"] for v in per_view) / len(per_view)

    train0 = project_dense_parallel(
        dataset.volume,
        dataset.train.angles[:1],
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples,
    )[0]
    train0_metrics = projection_metrics(train0, dataset.train.projections[0])
    result = {
        "dataset": str(dataset.root),
        "samples_per_ray": samples,
        "test_mean_psnr": mean_psnr,
        "test_mean_mae": mean_mae,
        "test_per_view": per_view,
        "train_view0": train0_metrics,
        "note": "train_view0 checks the exact axis-sum convention; test_mean exposes rotated-view geometry agreement.",
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    result = run_oracle(args.config)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

