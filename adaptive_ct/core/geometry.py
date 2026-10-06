from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RayBatch:
    """Parallel-beam rays: one per (angle, detector row, detector column); target = measured line integral."""

    angles: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    target: torch.Tensor
    detector_h: int
    detector_w: int

    @property
    def num_rays(self) -> int:
        return int(self.angles.shape[0])


def detector_grid(detector_h: int, detector_w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(detector_h, device=device, dtype=torch.long)
    cols = torch.arange(detector_w, device=device, dtype=torch.long)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    return rr.reshape(-1), cc.reshape(-1)


def parallel_rays(angles, rows, cols, *, detector_h: int, detector_w: int, targets) -> RayBatch:
    """Ray geometry follows the R2/TIGRE convention used to generate the prepared projections
    (see BernsteinOctree._parallel_ray_segments and the CUDA traversal)."""
    return RayBatch(
        angles=angles.to(dtype=torch.float32).contiguous(),
        rows=rows.to(dtype=torch.long).contiguous(),
        cols=cols.to(dtype=torch.long).contiguous(),
        target=targets.to(dtype=torch.float32).contiguous(),
        detector_h=int(detector_h),
        detector_w=int(detector_w),
    )
