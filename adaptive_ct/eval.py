from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .data import load_r2_dataset
from .metrics import projection_metrics
from .model import build_model
from .render import render_split


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    device = torch.device(config.get("device", "cuda"))
    dataset = load_r2_dataset(config["dataset"]["root"], device=device)
    payload = torch.load(args.checkpoint, map_location=device)
    checkpoint_config = payload.get("config", config) if isinstance(payload, dict) else config
    model = build_model(checkpoint_config).to(device=device)
    model.prepare_sparse_from_state_dict(payload["model"])
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    detector_h, detector_w = dataset.detector_shape
    rendered = render_split(
        model,
        dataset.test,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0])),
    )
    print(json.dumps({"projection_test": projection_metrics(rendered, dataset.test.projections)}, indent=2))


if __name__ == "__main__":
    main()
