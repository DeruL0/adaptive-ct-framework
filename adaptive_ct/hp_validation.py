from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_hp_validation_report(
    h_training: dict[str, Any],
    h_quantization: dict[str, Any],
    hp_training: dict[str, Any],
    hp_quantization: dict[str, Any],
) -> dict[str, Any]:
    h = h_quantization["results"]["float16"]
    hp = hp_quantization["results"]["float16"]
    h_metrics = h["metrics"]
    hp_metrics = hp["metrics"]
    h_full = h_metrics["full"]
    hp_full = hp_metrics["full"]
    h_high = h_metrics["high_gradient_top1"]
    hp_high = hp_metrics["high_gradient_top1"]
    p_events = [
        event
        for event in hp_training.get("growth_events", [])
        if event.get("operation") == "p_elevate"
    ]
    p_event = p_events[-1] if p_events else None
    p_fraction = float(p_event.get("active_fraction", 0.0)) if p_event else 0.0
    coefficient_delta = int(hp_training["model"]["coefficient_count"]) - int(
        h_training["model"]["coefficient_count"]
    )
    deltas = {
        "full_psnr_db": hp_full["psnr"] - h_full["psnr"],
        "full_mae": hp_full["mae"] - h_full["mae"],
        "high_gradient_psnr_db": hp_high["psnr"] - h_high["psnr"],
        "high_gradient_mae": hp_high["mae"] - h_high["mae"],
        "gradient_ratio_distance_reduction": (
            h_high["gradient_ratio_distance"] - hp_high["gradient_ratio_distance"]
        ),
        "coefficient_count": coefficient_delta,
        "float16_artifact_bytes": hp["export"]["file_bytes"] - h["export"]["file_bytes"],
        "training_seconds": hp_training["total_elapsed_sec"] - h_training["total_elapsed_sec"],
    }
    measurable_gain = (
        deltas["full_psnr_db"] > 0.01
        or deltas["high_gradient_psnr_db"] > 0.01
        or deltas["gradient_ratio_distance_reduction"] > 0.01
    )
    return {
        "schema": "hp_sufficient_view_validation_v1",
        "protocol": {
            "train_views": int(p_event["residual_views"]) if p_event else None,
            "test_views": int(h_quantization["test_views"]),
            "iterations": 15000,
            "projection_schedule": "inverse -> inverse_target_blend(alpha=0.50)",
            "matched_except_p_refinement": True,
            "p_active_fraction": p_fraction,
        },
        "h_only": {
            "metrics": h_metrics,
            "leaf_count": h_training["model"]["leaf_cells"],
            "coefficient_count": h_training["model"]["coefficient_count"],
            "total_elapsed_sec": h_training["total_elapsed_sec"],
            "float16_artifact_bytes": h["export"]["file_bytes"],
        },
        "hp": {
            "metrics": hp_metrics,
            "leaf_count": hp_training["model"]["leaf_cells"],
            "coefficient_count": hp_training["model"]["coefficient_count"],
            "total_elapsed_sec": hp_training["total_elapsed_sec"],
            "float16_artifact_bytes": hp["export"]["file_bytes"],
            "p_event": p_event,
        },
        "hp_minus_h": deltas,
        "verdict": {
            "measurable_test_gain": measurable_gain,
            "summary": (
                "hp shows a measurable held-out test gain"
                if measurable_gain
                else (
                    f"the {100.0 * p_fraction:.0f}% gradient-RD p budget "
                    "shows no measurable held-out test gain"
                )
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched h-only and hp CT runs")
    parser.add_argument("--h-training", type=Path, required=True)
    parser.add_argument("--h-quantization", type=Path, required=True)
    parser.add_argument("--hp-training", type=Path, required=True)
    parser.add_argument("--hp-quantization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_hp_validation_report(
        _load(args.h_training),
        _load(args.h_quantization),
        _load(args.hp_training),
        _load(args.hp_quantization),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
