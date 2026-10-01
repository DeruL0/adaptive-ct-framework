from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class RayBatch:
    points: Optional[torch.Tensor]
    step: Optional[torch.Tensor]
    target: torch.Tensor
    num_rays: int
    samples_per_ray: int
    angles: Optional[torch.Tensor] = None
    rows: Optional[torch.Tensor] = None
    cols: Optional[torch.Tensor] = None
    detector_h: Optional[int] = None
    detector_w: Optional[int] = None


def detector_grid(detector_h: int, detector_w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(detector_h, device=device, dtype=torch.long)
    cols = torch.arange(detector_w, device=device, dtype=torch.long)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    return rr.reshape(-1), cc.reshape(-1)


def ray_box_intersections(base_xy: torch.Tensor, direction_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-8
    tmin = torch.full((base_xy.shape[0],), -1.0e20, dtype=base_xy.dtype, device=base_xy.device)
    tmax = torch.full_like(tmin, 1.0e20)
    valid = torch.ones_like(tmin, dtype=torch.bool)
    for axis in range(2):
        base = base_xy[:, axis]
        direction = direction_xy[:, axis]
        parallel = direction.abs() < eps
        valid &= (~parallel) | ((base >= -1.0) & (base <= 1.0))
        denom = torch.where(parallel, torch.ones_like(direction), direction)
        t0 = (-1.0 - base) / denom
        t1 = (1.0 - base) / denom
        lo = torch.minimum(t0, t1)
        hi = torch.maximum(t0, t1)
        tmin = torch.where(parallel, tmin, torch.maximum(tmin, lo))
        tmax = torch.where(parallel, tmax, torch.minimum(tmax, hi))
    valid &= tmax > tmin
    tmin = torch.where(valid, tmin, torch.zeros_like(tmin))
    tmax = torch.where(valid, tmax, torch.zeros_like(tmax))
    return tmin, tmax


def make_parallel_ray_points(
    angles: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    targets: torch.Tensor,
    materialize_points: bool = True,
) -> RayBatch:
    # Match the R2/TIGRE convention used to generate the prepared projections.
    angles = -angles.to(dtype=torch.float32)
    rows_f = rows.to(dtype=torch.float32)
    cols_f = cols.to(dtype=torch.float32)
    ca = torch.cos(angles)
    sa = torch.sin(angles)
    direction_xy = torch.stack([ca, sa], dim=1)
    u_axis_xy = torch.stack([-sa, ca], dim=1)
    u = -1.0 + (cols_f + 0.5) * 2.0 / float(detector_w)
    z = -1.0 + (rows_f + 0.5) * 2.0 / float(detector_h)
    base_xy = u_axis_xy * u[:, None]
    tmin, tmax = ray_box_intersections(base_xy, direction_xy)
    step = (tmax - tmin) / float(samples_per_ray)
    if not materialize_points:
        return RayBatch(
            points=None,
            step=step.contiguous(),
            target=targets.to(dtype=torch.float32).contiguous(),
            num_rays=int(angles.shape[0]),
            samples_per_ray=int(samples_per_ray),
            angles=-angles.contiguous(),
            rows=rows.contiguous(),
            cols=cols.contiguous(),
            detector_h=int(detector_h),
            detector_w=int(detector_w),
        )
    sample_ids = torch.arange(samples_per_ray, device=angles.device, dtype=torch.float32)
    t = tmin[:, None] + (sample_ids[None, :] + 0.5) * step[:, None]
    xy = base_xy[:, None, :] + direction_xy[:, None, :] * t[:, :, None]
    z_values = z[:, None].expand(-1, samples_per_ray)
    points = torch.stack([xy[:, :, 0], xy[:, :, 1], z_values], dim=2).reshape(-1, 3)
    return RayBatch(
        points=points.contiguous(),
        step=step.contiguous(),
        target=targets.to(dtype=torch.float32).contiguous(),
        num_rays=int(angles.shape[0]),
        samples_per_ray=int(samples_per_ray),
        angles=-angles.contiguous(),
        rows=rows.contiguous(),
        cols=cols.contiguous(),
        detector_h=int(detector_h),
        detector_w=int(detector_w),
    )


def random_training_rays(
    split,
    *,
    batch_rays: int,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    materialize_points: bool = True,
) -> RayBatch:
    device = split.projections.device
    num_views = split.projections.shape[0]
    view_ids = torch.randint(0, num_views, (batch_rays,), device=device)
    rows = torch.randint(0, detector_h, (batch_rays,), device=device)
    cols = torch.randint(0, detector_w, (batch_rays,), device=device)
    targets = split.projections[view_ids, rows, cols]
    return make_parallel_ray_points(
        split.angles[view_ids],
        rows,
        cols,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        targets=targets,
        materialize_points=materialize_points,
    )


