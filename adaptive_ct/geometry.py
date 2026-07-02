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


@dataclass(frozen=True)
class ProjectionGradientBatch:
    """Paired detector rays used for a projection-domain Sobolev data term."""

    ray_batch: RayBatch
    edge_count: int
    target_derivative: torch.Tensor
    target_midpoint: torch.Tensor
    strides: torch.Tensor


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


def random_projection_gradient_rays(
    split,
    *,
    edge_rays: int,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    strides: tuple[int, ...] | list[int] = (1, 2, 4),
    uniform_fraction: float = 0.5,
    candidate_multiplier: int = 8,
    materialize_points: bool = True,
    generator: torch.Generator | None = None,
) -> ProjectionGradientBatch:
    """Sample adjacent detector-ray pairs from training projections.

    A fixed fraction is sampled uniformly and the remainder is selected from a
    larger random candidate pool by measured derivative magnitude. This keeps
    coverage of flat regions while ensuring that sparse, informative edges are
    represented in every stochastic batch. Test projections are never used.
    """

    count = int(edge_rays)
    if count < 1:
        raise ValueError("edge_rays must be positive")
    stride_values = tuple(int(value) for value in strides)
    if not stride_values or any(value < 1 for value in stride_values):
        raise ValueError("strides must contain positive integers")
    if max(stride_values) >= min(int(detector_h), int(detector_w)):
        raise ValueError("all gradient strides must fit inside the detector")
    fraction = float(uniform_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("uniform_fraction must be in [0, 1]")
    multiplier = max(1, int(candidate_multiplier))

    device = split.projections.device
    uniform_count = min(count, int(round(count * fraction)))
    biased_count = count - uniform_count
    candidate_count = uniform_count + max(biased_count, biased_count * multiplier)
    num_views = int(split.projections.shape[0])
    stride_table = torch.tensor(stride_values, dtype=torch.long, device=device)

    view_ids = torch.randint(0, num_views, (candidate_count,), device=device, generator=generator)
    directions = torch.randint(0, 2, (candidate_count,), device=device, generator=generator)
    stride_ids = torch.randint(
        0,
        len(stride_values),
        (candidate_count,),
        device=device,
        generator=generator,
    )
    sampled_strides = stride_table[stride_ids]
    vertical = directions == 0

    row_span = int(detector_h) - torch.where(vertical, sampled_strides, torch.zeros_like(sampled_strides))
    col_span = int(detector_w) - torch.where(vertical, torch.zeros_like(sampled_strides), sampled_strides)
    rows = torch.floor(
        torch.rand((candidate_count,), device=device, generator=generator) * row_span.to(torch.float32)
    ).to(torch.long)
    cols = torch.floor(
        torch.rand((candidate_count,), device=device, generator=generator) * col_span.to(torch.float32)
    ).to(torch.long)
    rows_neighbor = rows + torch.where(vertical, sampled_strides, torch.zeros_like(sampled_strides))
    cols_neighbor = cols + torch.where(vertical, torch.zeros_like(sampled_strides), sampled_strides)

    target_a = split.projections[view_ids, rows, cols]
    target_b = split.projections[view_ids, rows_neighbor, cols_neighbor]
    target_derivative = (target_b - target_a) / sampled_strides.to(dtype=target_a.dtype)

    selected = torch.arange(uniform_count, dtype=torch.long, device=device)
    if biased_count > 0:
        biased_pool = torch.arange(uniform_count, candidate_count, dtype=torch.long, device=device)
        top = torch.topk(torch.abs(target_derivative[biased_pool]), biased_count, sorted=False).indices
        selected = torch.cat([selected, biased_pool[top]], dim=0)

    view_ids = view_ids[selected]
    rows = rows[selected]
    cols = cols[selected]
    rows_neighbor = rows_neighbor[selected]
    cols_neighbor = cols_neighbor[selected]
    sampled_strides = sampled_strides[selected]
    target_a = target_a[selected]
    target_b = target_b[selected]
    target_derivative = target_derivative[selected]
    targets = torch.cat([target_a, target_b], dim=0)
    ray_batch = make_parallel_ray_points(
        torch.cat([split.angles[view_ids], split.angles[view_ids]], dim=0),
        torch.cat([rows, rows_neighbor], dim=0),
        torch.cat([cols, cols_neighbor], dim=0),
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        targets=targets,
        materialize_points=materialize_points,
    )
    return ProjectionGradientBatch(
        ray_batch=ray_batch,
        edge_count=count,
        target_derivative=target_derivative.contiguous(),
        target_midpoint=(0.5 * (target_a + target_b)).contiguous(),
        strides=sampled_strides.contiguous(),
    )
