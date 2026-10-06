from __future__ import annotations

from dataclasses import dataclass

import torch

from adaptive_ct.core.geometry import parallel_rays
from adaptive_ct.data.dataset import ProjectionSplit


@dataclass(frozen=True)
class FixedDecisionRaySet:
    split: ProjectionSplit
    view_ids: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    detector_h: int
    detector_w: int
    chunk_rays: int

    @property
    def observation_count(self) -> int:
        return int(self.rows.numel())


def ray_batch(rays: FixedDecisionRaySet, start: int, stop: int):
    view_ids, rows, cols = rays.view_ids[start:stop], rays.rows[start:stop], rays.cols[start:stop]
    return parallel_rays(rays.split.angles[view_ids], rows, cols, detector_h=rays.detector_h, detector_w=rays.detector_w,
                         targets=rays.split.projections[view_ids, rows, cols])


def sample_random_rays(split, *, ray_count: int, detector_h: int, detector_w: int, seed: int, chunk_rays: int) -> FixedDecisionRaySet:
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
        chunk_rays=max(1, int(chunk_rays)),
    )
