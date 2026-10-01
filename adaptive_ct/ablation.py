from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _representation(config_path: Path) -> str:
    try:
        config = load_config(config_path)
    except Exception:
        return "unknown"
    return str(config.get("model", {}).get("representation", "dynamic_leaf_voxel"))


def _volume_loss_weight(config_path: Path) -> float:
    try:
        config = load_config(config_path)
    except Exception:
        return 0.0
    training = config.get("training", {}) or {}
    volume_loss = training.get("volume_loss", {}) or {}
    return float(volume_loss.get("weight", 0.0))


def _strategy(report: dict[str, Any]) -> str:
    events = report.get("growth_events") or []
    if not events:
        return "dense"
    strategies = [str(event.get("strategy", "unknown")) for event in events]
    return "+".join(strategies)


def build_ablation_report(report_paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in report_paths:
        report = _load_json(path)
        config_path = Path(report.get("config", ""))
        projection = report.get("projection_test") or {}
        material = report.get("material_volume") or {}
        boundary = report.get("boundary_sharpness") or {}
        model = report.get("model") or {}
        rows.append(
            {
                "name": path.parent.name,
                "config": str(config_path),
                "representation": model.get("representation") or _representation(config_path),
                "volume_loss_weight": _volume_loss_weight(config_path),
                "activation_strategy": _strategy(report),
                "projection_psnr": projection.get("psnr"),
                "projection_mae": projection.get("mae"),
                "material_volume_psnr": material.get("psnr"),
                "material_volume_mae": material.get("mae"),
                "boundary_gradient_ratio": boundary.get("gradient_ratio"),
                "boundary_high_gradient_mae": boundary.get("high_gradient_mae"),
                "model_bytes": model.get("model_bytes"),
                "leaf_cells": model.get("leaf_cells"),
                "l0_active": model.get("l0_active"),
                "l1_active": model.get("l1_active"),
                "l2_active": model.get("l2_active"),
                "train_time_sec": report.get("elapsed_sec"),
            }
        )

    def best(metric: str, higher: bool = True):
        valid = [row for row in rows if isinstance(row.get(metric), (int, float))]
        if not valid:
            return None
        return max(valid, key=lambda row: row[metric]) if higher else min(valid, key=lambda row: row[metric])

    return {
        "schema": "adaptive_ct_ablation_report_v1",
        "note": "Projection, material-volume, and boundary sharpness are distinct objectives; use the per-metric winners instead of a single overall winner.",
        "rows": rows,
        "winners": {
            "projection_psnr": best("projection_psnr", higher=True),
            "material_volume_psnr": best("material_volume_psnr", higher=True),
            "boundary_gradient_ratio": best("boundary_gradient_ratio", higher=True),
            "boundary_high_gradient_mae": best("boundary_high_gradient_mae", higher=False),
            "model_bytes": best("model_bytes", higher=False),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize adaptive CT ablation training reports.")
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_ablation_report([Path(path) for path in args.report])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
