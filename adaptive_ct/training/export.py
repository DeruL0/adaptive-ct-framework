"""Post-training export: codec conversion and slice previews."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from adaptive_ct.codec.finetune import finetune_model
from adaptive_ct.codec.quantize import quantize_roundtrip_
from adaptive_ct.evaluation.metrics import projection_metrics
from adaptive_ct.evaluation.render import render_split
from adaptive_ct.representation.latent import LatentP1Octree


@torch.no_grad()
def export_center_slice_png(
    model,
    output_path: Path,
    volume_shape: tuple[int, int, int],
    *,
    axis: int = 2,
    chunk: int = 524288,
) -> dict:
    shape = tuple(int(value) for value in volume_shape)
    axis = int(axis)
    index = shape[axis] // 2
    dims = [dim for dim in range(3) if dim != axis]
    device = next(model.parameters()).device
    scale = torch.tensor(shape, dtype=torch.float32, device=device)
    grids = torch.meshgrid(
        torch.arange(shape[dims[0]], dtype=torch.float32, device=device),
        torch.arange(shape[dims[1]], dtype=torch.float32, device=device),
        indexing="ij",
    )
    coords = torch.zeros((grids[0].numel(), 3), dtype=torch.float32, device=device)
    coords[:, dims[0]] = grids[0].reshape(-1)
    coords[:, dims[1]] = grids[1].reshape(-1)
    coords[:, axis] = float(index)
    points = -1.0 + (coords + 0.5) * 2.0 / scale
    values = []
    for start in range(0, points.shape[0], int(chunk)):
        values.append(model.forward_mu(points[start : start + int(chunk)]).detach().float().cpu())
    image = torch.cat(values, dim=0).reshape(shape[dims[0]], shape[dims[1]]).numpy().T
    lo, hi = np.percentile(image, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    if hi <= lo:
        hi = lo + 1.0
    pixels = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    Image.fromarray((pixels * 255.0).astype(np.uint8), mode="L").save(output_path)
    return {
        "path": str(output_path.resolve()),
        "axis": axis,
        "index": index,
        "shape": list(image.shape),
        "display_percentiles": [float(lo), float(hi)],
        "source": "model_prediction_only",
    }


def finest_model_shape(model, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    shapes = getattr(model, "level_shapes", None)
    if shapes:
        return tuple(int(value) for value in shapes[-1])
    resolutions = getattr(model, "level_resolutions", None)
    if resolutions:
        return (int(resolutions[-1]),) * 3
    return tuple(int(value) for value in fallback)


def apply_export_codec(model, dataset, config: dict, *, batch_rays: int, detector_h: int, detector_w: int,
                        samples_per_ray: int, materialize_points: bool, weight_kwargs: dict):
    """Convert the trained octree to its deployed codec; returns (deployed_model, summary)."""
    codec_cfg = (config.get("export", {}) or {}).get("codec") or {}
    if not codec_cfg:
        return model, None
    train_cfg = config["training"]
    codec_type = str(codec_cfg.get("type", "p0")).lower()

    def test_metric(candidate):
        return projection_metrics(
            render_split(
                candidate,
                dataset.test,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                ray_chunk=int(train_cfg.get("eval_ray_chunk", 4096)),
            ),
            dataset.test.projections,
        )

    summary = {"type": codec_type, "projection_test_pre_codec": test_metric(model)}
    if codec_type == "latent_p1":
        deployed = LatentP1Octree.from_bernstein(model, int(codec_cfg.get("latent_dim", 4)))
        summary["projection_test_after_pca"] = test_metric(deployed)
        summary["finetune"] = finetune_model(
            deployed,
            dataset.train,
            steps=int(codec_cfg.get("finetune_iterations", 2000)),
            lr=float(codec_cfg.get("finetune_lr", 0.003)),
            batch_rays=batch_rays,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            weight_kwargs=weight_kwargs,
            materialize_points=materialize_points,
        )
    elif codec_type in {"p0", "p1"}:
        deployed = model
    else:
        raise ValueError(f"Unknown export.codec.type {codec_type!r}.")
    quantize_roundtrip_(deployed, codec_cfg)
    summary["projection_test_deployed"] = test_metric(deployed)
    summary["quantization"] = str(codec_cfg.get("quantization", "float16"))
    summary["quantization_applied_in_eval"] = True
    return deployed, summary
