from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(metrics: dict[str, Any] | None, key: str):
    if not isinstance(metrics, dict):
        return None
    return metrics.get(key)


def _supervision_label(benchmark: dict[str, Any]) -> str:
    objective = benchmark.get("training_objective") or {}
    volume_loss = objective.get("volume_loss") if isinstance(objective, dict) else None
    if isinstance(volume_loss, dict) and volume_loss.get("enabled"):
        return "projection_plus_vol_gt"
    return "projection_only"


def build_comparison_tables(benchmark: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    protocols = benchmark["protocols"]
    storage = benchmark["storage_runtime"]
    methods = baseline.get("methods") or []
    supervision = _supervision_label(benchmark)

    material_volume_rows = [
        {
            "method": benchmark["method"],
            "source": "adaptive_ct benchmark",
            "supervision": supervision,
            "psnr": _metric(protocols["material_volume_domain"]["metrics"], "psnr"),
            "mae": _metric(protocols["material_volume_domain"]["metrics"], "mae"),
            "count": _metric(protocols["material_volume_domain"]["metrics"], "count"),
            "model_bytes": storage.get("model_bytes"),
            "leaf_cells": storage.get("leaf_cells"),
        }
    ]
    for method in methods:
        volume = method.get("volume_material")
        if not isinstance(volume, dict):
            continue
        material_volume_rows.append(
            {
                "method": method.get("method"),
                "source": method.get("source"),
                "psnr": volume.get("psnr"),
                "mae": volume.get("mae"),
                "count": volume.get("count"),
                "num_primitives": method.get("num_primitives"),
            }
        )

    material_projection = protocols["material_projection_domain"].get("metrics") or {}
    masked = material_projection.get("material_ray_masked") or {}
    full = material_projection.get("full_detector") or {}
    material_projection_rows = [
        {
            "method": benchmark["method"],
            "source": "adaptive_ct benchmark",
            "supervision": supervision,
            "protocol": "PSNR(P(V * M), P(GT * M)); material-ray masked plus full detector",
            "masked_psnr": masked.get("psnr"),
            "masked_mae": masked.get("mae"),
            "full_detector_psnr": full.get("psnr"),
            "full_detector_mae": full.get("mae"),
            "mask_pixel_fraction": masked.get("mask_pixel_fraction"),
            "model_bytes": storage.get("model_bytes"),
            "leaf_cells": storage.get("leaf_cells"),
        }
    ]
    for method in methods:
        projection = method.get("projection_material_from_volume")
        if not isinstance(projection, dict):
            continue
        material_projection_rows.append(
            {
                "method": method.get("method"),
                "source": method.get("source"),
                "protocol": projection.get("definition"),
                "masked_psnr": None,
                "masked_mae": None,
                "full_detector_psnr": projection.get("psnr"),
                "full_detector_mae": projection.get("mae"),
                "num_primitives": method.get("num_primitives"),
            }
        )

    rendered_projection_full_rows = [
        {
            "method": benchmark["method"],
            "source": "adaptive_ct benchmark",
            "supervision": supervision,
            "protocol": protocols["projection_domain"]["protocol"],
            "psnr": protocols["projection_domain"]["metrics"].get("psnr"),
            "mae": protocols["projection_domain"]["metrics"].get("mae"),
            "projections_per_sec": protocols["projection_domain"].get("projections_per_sec"),
            "model_bytes": storage.get("model_bytes"),
        }
    ]
    rendered_projection_masked_rows = []
    for method in methods:
        rendered_full = method.get("projection_rendered_common_gt")
        if isinstance(rendered_full, dict):
            rendered_projection_full_rows.append(
                {
                    "method": method.get("method"),
                    "source": method.get("rendered_projection_sources") or method.get("source"),
                    "protocol": rendered_full.get("definition") or "full-detector rendered projection; see baseline source protocol",
                    "psnr": rendered_full.get("psnr"),
                    "mae": rendered_full.get("mae"),
                    "num_primitives": method.get("num_primitives"),
                }
            )
        rendered_masked = method.get("projection_masked")
        if isinstance(rendered_masked, dict):
            rendered_projection_masked_rows.append(
                {
                    "method": method.get("method"),
                    "source": method.get("rendered_projection_sources") or method.get("source"),
                    "protocol": rendered_masked.get("definition") or "material-ray masked rendered projection; see baseline source protocol",
                    "psnr": rendered_masked.get("psnr"),
                    "mae": rendered_masked.get("mae"),
                    "num_primitives": method.get("num_primitives"),
                }
            )

    return {
        "schema": "adaptive_ct_comparison_tables_v1",
        "dataset": {
            "adaptive_ct": benchmark.get("dataset"),
            "baseline": baseline.get("dataset"),
        },
        "note": "Each table has one protocol. Do not merge these rows across tables without carrying the protocol label.",
        "tables": {
            "material_volume_domain": {
                "protocol": protocols["material_volume_domain"]["protocol"],
                "rows": material_volume_rows,
            },
            "material_projection_from_volume_domain": {
                "protocol": protocols["material_projection_domain"]["protocol"],
                "rows": material_projection_rows,
            },
            "rendered_projection_full_detector_domain": {
                "protocol": "Native/rendered projection on the full detector. Air/background handling follows each method's rendered output.",
                "rows": rendered_projection_full_rows,
            },
            "rendered_projection_material_ray_masked_domain": {
                "protocol": "Native/rendered projection evaluated only on detector pixels intersecting GT material. Adaptive CT full-render masked metrics are not inserted unless computed under this same protocol.",
                "rows": rendered_projection_masked_rows,
            },
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build separated comparison tables from adaptive and baseline reports.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    tables = build_comparison_tables(_load(Path(args.benchmark)), _load(Path(args.baseline)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tables, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(tables, indent=2))


if __name__ == "__main__":
    main()
