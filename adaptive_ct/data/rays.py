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
