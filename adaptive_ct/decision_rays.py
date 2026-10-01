from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import ProjectionSplit
from .geometry import make_parallel_ray_points


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


@torch.no_grad()
def fixed_rss(model, rays: FixedDecisionRaySet) -> dict[str, float]:
    rss = 0.0
    count = 0
    for start in range(0, rays.observation_count, rays.chunk_rays):
        stop = min(start + rays.chunk_rays, rays.observation_count)
        view_ids = rays.view_ids[start:stop]
        rows = rays.rows[start:stop]
        cols = rays.cols[start:stop]
        angles = rays.split.angles[view_ids]
        target = rays.split.projections[view_ids, rows, cols]
        ray_batch = make_parallel_ray_points(
            angles,
            rows,
            cols,
            detector_h=rays.detector_h,
            detector_w=rays.detector_w,
            samples_per_ray=rays.samples_per_ray,
            targets=target,
            materialize_points=rays.materialize_points,
        )
        residual = model.integrate_ray_batch(ray_batch) - target
        rss += float(torch.sum(torch.square(residual)).item())
        count += int(residual.numel())
    return {"rss": rss, "mse": rss / max(count, 1), "ray_count": float(count)}
