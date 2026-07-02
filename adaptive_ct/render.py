from __future__ import annotations

import torch

from .geometry import detector_grid, make_parallel_ray_points


@torch.no_grad()
def render_split(model, split, *, detector_h: int, detector_w: int, samples_per_ray: int, ray_chunk: int = 4096):
    device = split.projections.device
    rows_all, cols_all = detector_grid(detector_h, detector_w, device)
    rendered = []
    materialize_points = not (hasattr(model, "prefer_compact_ray_batch") and model.prefer_compact_ray_batch())
    for angle in split.angles:
        chunks = []
        for start in range(0, rows_all.numel(), ray_chunk):
            rows = rows_all[start : start + ray_chunk]
            cols = cols_all[start : start + ray_chunk]
            angles = angle.expand(rows.shape[0])
            targets = torch.zeros(rows.shape[0], dtype=torch.float32, device=device)
            rb = make_parallel_ray_points(
                angles,
                rows,
                cols,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                targets=targets,
                materialize_points=materialize_points,
            )
            chunks.append(model.integrate_ray_batch(rb))
        rendered.append(torch.cat(chunks, dim=0).reshape(detector_h, detector_w))
    return torch.stack(rendered, dim=0)
