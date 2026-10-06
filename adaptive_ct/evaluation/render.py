from __future__ import annotations

import torch

from adaptive_ct.core.geometry import detector_grid, parallel_rays


@torch.no_grad()
def render_split(model, split, *, detector_h: int, detector_w: int, ray_chunk: int = 1 << 20):
    """Predicted projections of every view in ``split``: (views, detector_h, detector_w)."""
    device = split.projections.device
    rows_all, cols_all = detector_grid(detector_h, detector_w, device)
    rendered = []
    for angle in split.angles:
        chunks = []
        for start in range(0, rows_all.numel(), ray_chunk):
            rows, cols = rows_all[start : start + ray_chunk], cols_all[start : start + ray_chunk]
            rb = parallel_rays(angle.expand(rows.shape[0]), rows, cols, detector_h=detector_h, detector_w=detector_w,
                               targets=torch.zeros(rows.shape[0], device=device))
            chunks.append(model.integrate_ray_batch(rb))
        rendered.append(torch.cat(chunks, dim=0).reshape(detector_h, detector_w))
    return torch.stack(rendered, dim=0)
