"""Post-training export: storage rounding and slice previews."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from adaptive_ct.codec.quantize import quantize_roundtrip_
from adaptive_ct.evaluation.metrics import projection_metrics
from adaptive_ct.evaluation.render import render_split


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


def finest_model_shape(model) -> tuple[int, int, int]:
    return tuple(int(value) for value in model.level_shapes[-1])


def deploy(model, dataset, quantization: str, *, detector_h: int, detector_w: int, ray_chunk: int) -> dict:
    """Round what is stored (coefficients, step geometry) to the storage precision, in place; test projections before/after."""
    test = lambda: projection_metrics(render_split(model, dataset.test, detector_h=detector_h, detector_w=detector_w, ray_chunk=ray_chunk), dataset.test.projections)
    summary = {"quantization": quantization, "projection_test_solved": test()}
    quantize_roundtrip_(model, quantization)
    summary["projection_test_deployed"] = test()
    return summary
