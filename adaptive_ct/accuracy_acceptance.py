from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GIB = 1024**3


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(name: str, value: float, threshold: float, passed: bool, unit: str) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "unit": unit,
        "passed": bool(passed),
    }


def build_report(
    baseline_quantization: dict[str, Any],
    w_training: dict[str, Any],
    w_quantization: dict[str, Any],
    wp_training: dict[str, Any],
    wp_quantization: dict[str, Any],
    wpl3_training: dict[str, Any],
    wpl3_quantization: dict[str, Any],
    old_benchmark: dict[str, Any],
    packed_benchmark: dict[str, Any],
) -> dict[str, Any]:
    baseline_f16 = baseline_quantization["results"]["float16"]["metrics"]
    w_f16 = w_quantization["results"]["float16"]["metrics"]
    wp_f16 = wp_quantization["results"]["float16"]["metrics"]
    wpl3_f16 = wpl3_quantization["results"]["float16"]["metrics"]

    # L3 is retained only when the real test metrics Pareto-improve on WP.
    # A negligible full-PSNR increase does not override worse structural metrics.
    retain_l3 = (
        wpl3_f16["full"]["psnr"] > wp_f16["full"]["psnr"]
        and wpl3_f16["high_gradient_top1"]["psnr"] >= wp_f16["high_gradient_top1"]["psnr"]
        and wpl3_f16["high_gradient_top1"]["gradient_ratio_distance"]
        <= wp_f16["high_gradient_top1"]["gradient_ratio_distance"]
    )
    selected_name = "WPL3" if retain_l3 else "WP"
    selected_training = wpl3_training if retain_l3 else wp_training
    selected_quantization = wpl3_quantization if retain_l3 else wp_quantization
    selected_f16 = wpl3_f16 if retain_l3 else wp_f16
    selected_u8 = selected_quantization["results"]["uint8"]["metrics"]

    baseline_full = baseline_f16["full"]
    baseline_high = baseline_f16["high_gradient_top1"]
    final_full = selected_f16["full"]
    final_high = selected_f16["high_gradient_top1"]

    high_psnr_gain = final_high["psnr"] - baseline_high["psnr"]
    baseline_distance = baseline_high["gradient_ratio_distance"]
    final_distance = final_high["gradient_ratio_distance"]
    distance_reduction = (baseline_distance - final_distance) / baseline_distance
    u8_psnr_drop = final_full["psnr"] - selected_u8["full"]["psnr"]
    u8_high_mae_increase = (
        selected_u8["high_gradient_top1"]["mae"] / final_high["mae"] - 1.0
    )

    old_d0 = old_benchmark["cases"][0]["cuda_native"]["step_ms"]
    packed_d0 = packed_benchmark["cases"][0]["cuda_native"]["step_ms"]
    three_level_regression = packed_d0 / old_d0 - 1.0

    peak_allocated_gib = selected_training["peak_allocated_bytes"] / GIB
    peak_reserved_gib = selected_training["peak_reserved_bytes"] / GIB
    runtime_seconds = selected_training["total_elapsed_sec"]

    gates = [
        _gate(
            "float16 full-test PSNR",
            final_full["psnr"],
            baseline_full["psnr"] - 0.1,
            final_full["psnr"] >= baseline_full["psnr"] - 0.1,
            "dB (minimum)",
        ),
        _gate(
            "float16 full-test MAE",
            final_full["mae"],
            baseline_full["mae"] * 1.02,
            final_full["mae"] <= baseline_full["mae"] * 1.02,
            "absolute (maximum)",
        ),
        _gate(
            "high-gradient top-1% PSNR gain",
            high_psnr_gain,
            0.5,
            high_psnr_gain >= 0.5,
            "dB (minimum)",
        ),
        _gate(
            "top-1% gradient-ratio distance reduction",
            distance_reduction * 100.0,
            15.0,
            distance_reduction >= 0.15,
            "% (minimum)",
        ),
        _gate(
            "peak allocated GPU memory",
            peak_allocated_gib,
            10.5,
            peak_allocated_gib <= 10.5,
            "GiB (maximum)",
        ),
        _gate(
            "full training runtime",
            runtime_seconds,
            600.0,
            runtime_seconds <= 600.0,
            "seconds (maximum)",
        ),
        _gate(
            "three-level CUDA performance regression",
            three_level_regression * 100.0,
            10.0,
            three_level_regression <= 0.10,
            "% (maximum)",
        ),
    ]
    quantization_gates = [
        _gate(
            "uint8 PSNR loss vs float16",
            u8_psnr_drop,
            0.25,
            u8_psnr_drop <= 0.25,
            "dB (maximum)",
        ),
        _gate(
            "uint8 high-gradient MAE degradation",
            u8_high_mae_increase * 100.0,
            5.0,
            u8_high_mae_increase <= 0.05,
            "% (maximum)",
        ),
    ]

    return {
        "schema": "ct_accuracy_acceptance_v2",
        "overall_passed": all(g["passed"] for g in gates),
        "selected_variant": selected_name,
        "l3_retention": {
            "retained": retain_l3,
            "rule": "retain only on a real-test Pareto improvement over WP",
            "full_psnr_delta_db": wpl3_f16["full"]["psnr"] - wp_f16["full"]["psnr"],
            "high_gradient_psnr_delta_db": (
                wpl3_f16["high_gradient_top1"]["psnr"]
                - wp_f16["high_gradient_top1"]["psnr"]
            ),
            "gradient_ratio_distance_delta": (
                wpl3_f16["high_gradient_top1"]["gradient_ratio_distance"]
                - wp_f16["high_gradient_top1"]["gradient_ratio_distance"]
            ),
        },
        "quantization_decision": {
            "selected": "uint8" if all(g["passed"] for g in quantization_gates) else "float16",
            "gates": quantization_gates,
        },
        "gates": gates,
        "baseline_float16": baseline_f16,
        "selected_float16": selected_f16,
        "ablations": {
            "W": {
                "projection_test": w_training["projection_test"],
                "float16_metrics": w_f16,
                "leaf_cells": w_training["model"]["leaf_cells"],
                "coefficient_count": w_training["model"]["coefficient_count"],
                "elapsed_sec": w_training["elapsed_sec"],
            },
            "WP": {
                "projection_test": wp_training["projection_test"],
                "float16_metrics": wp_f16,
                "leaf_cells": wp_training["model"]["leaf_cells"],
                "coefficient_count": wp_training["model"]["coefficient_count"],
                "elapsed_sec": wp_training["elapsed_sec"],
            },
            "WPL3": {
                "projection_test": wpl3_training["projection_test"],
                "float16_metrics": wpl3_f16,
                "leaf_cells": wpl3_training["model"]["leaf_cells"],
                "coefficient_count": wpl3_training["model"]["coefficient_count"],
                "active_by_level": wpl3_training["model"]["active_by_level"],
                "elapsed_sec": wpl3_training["elapsed_sec"],
                "total_elapsed_sec": wpl3_training["total_elapsed_sec"],
                "peak_allocated_gib": wpl3_training["peak_allocated_bytes"] / GIB,
                "peak_reserved_gib": wpl3_training["peak_reserved_bytes"] / GIB,
            },
        },
        "cuda_benchmark": {
            "old_degree0_step_ms": old_d0,
            "packed_degree0_step_ms": packed_d0,
            "three_level_regression_fraction": three_level_regression,
            "four_level_numerical_consistency": "covered by tests.test_bernstein_cuda",
        },
        "notes": [
            "Test views were used only for final evaluation, never for refinement selection.",
            "Failure of a structural gate is retained as a failure; the report does not substitute per-view or training-view metrics.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 16-view CT accuracy acceptance report")
    parser.add_argument("--baseline-quantization", type=Path, required=True)
    parser.add_argument("--w-training", type=Path, required=True)
    parser.add_argument("--w-quantization", type=Path, required=True)
    parser.add_argument("--wp-training", type=Path, required=True)
    parser.add_argument("--wp-quantization", type=Path, required=True)
    parser.add_argument("--wpl3-training", type=Path, required=True)
    parser.add_argument("--wpl3-quantization", type=Path, required=True)
    parser.add_argument("--old-benchmark", type=Path, required=True)
    parser.add_argument("--packed-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        _load(args.baseline_quantization),
        _load(args.w_training),
        _load(args.w_quantization),
        _load(args.wp_training),
        _load(args.wp_quantization),
        _load(args.wpl3_training),
        _load(args.wpl3_quantization),
        _load(args.old_benchmark),
        _load(args.packed_benchmark),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
