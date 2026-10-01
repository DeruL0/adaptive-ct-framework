from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Callable

import torch

from .backend import project_dense_parallel
from .config import load_config
from .data import load_r2_dataset
from .metrics import projection_metrics


AXIS_NAMES = ("x", "y", "z")


def _detector_transforms() -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    return {
        "identity": lambda x: x,
        "flip_rows": lambda x: x.flip(1),
        "flip_cols": lambda x: x.flip(2),
        "flip_rows_cols": lambda x: x.flip(1).flip(2),
        "transpose": lambda x: x.transpose(1, 2),
        "transpose_flip_rows": lambda x: x.transpose(1, 2).flip(1),
        "transpose_flip_cols": lambda x: x.transpose(1, 2).flip(2),
        "transpose_flip_rows_cols": lambda x: x.transpose(1, 2).flip(1).flip(2),
    }


def _volume_variant(volume: torch.Tensor, perm: tuple[int, int, int], flips: tuple[bool, bool, bool]) -> torch.Tensor:
    transformed = volume.permute(perm)
    for axis, enabled in enumerate(flips):
        if enabled:
            transformed = transformed.flip(axis)
    return transformed.contiguous()


def _variant_name(perm: tuple[int, int, int], flips: tuple[bool, bool, bool]) -> str:
    axes = "".join(AXIS_NAMES[i] for i in perm)
    suffix = "".join(AXIS_NAMES[i] for i, enabled in enumerate(flips) if enabled)
    return f"perm_{axes}_flip_{suffix or 'none'}"


@torch.no_grad()
def run_oracle_diagnostics(
    config_path: str | Path,
    *,
    max_views: int = 4,
    top_k: int = 12,
    include_axis_search: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda"))
    dataset = load_r2_dataset(config["dataset"]["root"], device=device)
    detector_h, detector_w = dataset.detector_shape
    samples = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))
    view_count = min(int(max_views), int(dataset.test.projections.shape[0]))
    angles = dataset.test.angles[:view_count]
    target = dataset.test.projections[:view_count]
    detector_ops = _detector_transforms()

    permutations = list(itertools.permutations((0, 1, 2))) if include_axis_search else [(0, 1, 2)]
    flips_iter = list(itertools.product((False, True), repeat=3)) if include_axis_search else [(False, False, False)]
    rows = []
    for perm in permutations:
        for flips in flips_iter:
            volume = _volume_variant(dataset.volume, perm, flips)
            for angle_sign in (-1.0, 1.0):
                rendered = project_dense_parallel(
                    volume,
                    angles * angle_sign,
                    detector_h=detector_h,
                    detector_w=detector_w,
                    samples_per_ray=samples,
                )
                for detector_name, detector_fn in detector_ops.items():
                    candidate = detector_fn(rendered)
                    if tuple(candidate.shape[-2:]) != tuple(target.shape[-2:]):
                        continue
                    metric = projection_metrics(candidate, target)
                    rows.append(
                        {
                            "volume_transform": _variant_name(perm, flips),
                            "perm": list(perm),
                            "flips": list(flips),
                            "angle_sign": angle_sign,
                            "detector_transform": detector_name,
                            "psnr": metric["psnr"],
                            "mae": metric["mae"],
                        }
                    )

    rows.sort(key=lambda row: float(row["psnr"]), reverse=True)
    best = rows[0] if rows else None
    identity = [
        row
        for row in rows
        if row["volume_transform"] == "perm_xyz_flip_none"
        and row["angle_sign"] == 1.0
        and row["detector_transform"] == "identity"
    ]
    return {
        "schema": "adaptive_ct_oracle_diagnostics_v1",
        "config": str(Path(config_path).resolve()),
        "dataset": str(dataset.root),
        "views": view_count,
        "samples_per_ray": samples,
        "searched_axis_variants": include_axis_search,
        "variant_count": len(rows),
        "best": best,
        "identity_native": identity[0] if identity else None,
        "top": rows[: int(top_k)],
        "note": "Diagnostic only: searches convention transforms for stored projections; it does not change training configs.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Search projection convention variants for R2-style oracle alignment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--skip-axis-search", action="store_true")
    args = parser.parse_args(argv)
    result = run_oracle_diagnostics(
        args.config,
        max_views=int(args.max_views),
        top_k=int(args.top_k),
        include_axis_search=not bool(args.skip_axis_search),
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
