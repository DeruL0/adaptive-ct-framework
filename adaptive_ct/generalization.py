from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _case_row(name: str, output_dir: Path) -> dict[str, Any]:
    training = _load(output_dir / "training_report.json") or {}
    oracle = _load(output_dir / "oracle.json") or {}
    diagnostics = _load(output_dir / "oracle_diagnostics.json") or {}
    best_diag = diagnostics.get("best") or {}
    identity_diag = diagnostics.get("identity_native") or {}
    projection = training.get("projection_test") or {}
    material = training.get("material_volume") or {}
    boundary = training.get("boundary_sharpness") or {}
    model = training.get("model") or {}
    return {
        "case": name,
        "output_dir": str(output_dir),
        "oracle_test_mean_psnr": oracle.get("test_mean_psnr"),
        "oracle_train0_psnr": (oracle.get("train_view0") or {}).get("psnr"),
        "diagnostic_best_psnr": best_diag.get("psnr"),
        "diagnostic_best_transform": {
            "volume_transform": best_diag.get("volume_transform"),
            "angle_sign": best_diag.get("angle_sign"),
            "detector_transform": best_diag.get("detector_transform"),
        },
        "diagnostic_identity_psnr": identity_diag.get("psnr"),
        "projection_test_psnr": projection.get("psnr"),
        "projection_test_mae": projection.get("mae"),
        "material_volume_psnr": material.get("psnr"),
        "material_volume_mae": material.get("mae"),
        "boundary_gradient_ratio": boundary.get("gradient_ratio"),
        "model_bytes": model.get("model_bytes"),
        "representation": model.get("representation"),
        "leaf_cells": model.get("leaf_cells"),
        "l0_active": model.get("l0_active"),
        "l1_active": model.get("l1_active"),
        "l2_active": model.get("l2_active"),
        "train_time_sec": training.get("elapsed_sec"),
    }


def build_generalization_report(cases: dict[str, Path]) -> dict[str, Any]:
    rows = [_case_row(name, path) for name, path in cases.items()]
    return {
        "schema": "adaptive_ct_generalization_report_v1",
        "note": (
            "These are parallel-beam R2-style smoke/generalization runs. "
            "The train0 oracle checks axis convention; lower rotated-view oracle PSNR flags projector/protocol mismatch. "
            "Diagnostic fields search axis, flip, angle-sign, and detector transforms for convention debugging."
        ),
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize adaptive CT generalization runs.")
    parser.add_argument("--out", default="output/generalization_report.json")
    args = parser.parse_args(argv)
    report = build_generalization_report(
        {
            "fandisk": Path("output/fandisk_48v"),
            "rocker_arm": Path("output/rocker_arm_48v"),
            "bunny": Path("output/bunny"),
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
