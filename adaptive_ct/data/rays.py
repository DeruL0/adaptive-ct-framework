from __future__ import annotations

from dataclasses import dataclass

import torch

from adaptive_ct.core.geometry import make_parallel_ray_points
from adaptive_ct.data.dataset import ProjectionSplit


@dataclass(frozen=True)
class FixedDecisionRaySet:
    split: ProjectionSplit
    view_ids: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    detector_h: int
    detector_w: int
    samples_per_ray: int
    chunk_rays: int
    materialize_points: bool

    @property
    def observation_count(self) -> int:
        return int(self.rows.numel())

    @classmethod
    def sampled(
        cls,
        split: ProjectionSplit,
        *,
        ray_count: int,
        detector_h: int,
        detector_w: int,
        samples_per_ray: int,
        seed: int,
        chunk_rays: int,
        materialize_points: bool,
    ) -> "FixedDecisionRaySet":
        count = max(1, int(ray_count))
        device = split.projections.device
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        total = int(split.angles.numel()) * int(detector_h) * int(detector_w)
        ids = torch.randperm(total, generator=generator)[: min(count, total)].to(device=device)
        pixels = int(detector_h) * int(detector_w)
        view_ids = torch.div(ids, pixels, rounding_mode="floor")
        rem = ids - view_ids * pixels
        rows = torch.div(rem, int(detector_w), rounding_mode="floor")
        cols = rem - rows * int(detector_w)
        return cls(
            split=split,
            view_ids=view_ids,
            rows=rows,
            cols=cols,
            detector_h=int(detector_h),
            detector_w=int(detector_w),
            samples_per_ray=int(samples_per_ray),
            chunk_rays=max(1, int(chunk_rays)),
            materialize_points=bool(materialize_points),
        )


def ray_batch(rays: FixedDecisionRaySet, start: int, stop: int):
    view_ids = rays.view_ids[start:stop]
    rows = rays.rows[start:stop]
    cols = rays.cols[start:stop]
    return make_parallel_ray_points(
        rays.split.angles[view_ids],
        rows,
        cols,
        detector_h=rays.detector_h,
        detector_w=rays.detector_w,
        samples_per_ray=rays.samples_per_ray,
        targets=rays.split.projections[view_ids, rows, cols],
        materialize_points=rays.materialize_points,
    )


def sample_random_rays(
    split,
    *,
    ray_count: int,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    seed: int,
    chunk_rays: int,
    materialize_points: bool,
) -> FixedDecisionRaySet:
    """Uniform random rays over every view of ``split`` (with replacement)."""
    device = split.projections.device
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    count = max(1, int(ray_count))
    views = int(split.angles.numel())
    return FixedDecisionRaySet(
        split=split,
        view_ids=torch.randint(0, views, (count,), generator=generator).to(device),
        rows=torch.randint(0, int(detector_h), (count,), generator=generator).to(device),
        cols=torch.randint(0, int(detector_w), (count,), generator=generator).to(device),
        detector_h=int(detector_h),
        detector_w=int(detector_w),
        samples_per_ray=int(samples_per_ray),
        chunk_rays=max(1, int(chunk_rays)),
        materialize_points=bool(materialize_points),
    )


def head_rays(rays: FixedDecisionRaySet, count: int) -> FixedDecisionRaySet:
    count = max(1, min(int(count), rays.observation_count))
    return FixedDecisionRaySet(
        split=rays.split,
        view_ids=rays.view_ids[:count],
        rows=rays.rows[:count],
        cols=rays.cols[:count],
        detector_h=rays.detector_h,
        detector_w=rays.detector_w,
        samples_per_ray=rays.samples_per_ray,
        chunk_rays=rays.chunk_rays,
        materialize_points=rays.materialize_points,
    )


@torch.no_grad()
def fixed_rss(model, rays: FixedDecisionRaySet) -> dict[str, float]:
    rss = 0.0
    count = 0
    for start in range(0, rays.observation_count, rays.chunk_rays):
        batch = ray_batch(rays, start, min(start + rays.chunk_rays, rays.observation_count))
        residual = model.integrate_ray_batch(batch) - batch.target
        rss += float(torch.sum(torch.square(residual)).item())
        count += int(residual.numel())
    return {"rss": rss, "mse": rss / max(count, 1), "ray_count": float(count)}
