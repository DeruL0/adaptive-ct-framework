"""Projection-residual error indicator and resolvability filter."""

from __future__ import annotations

import torch

from adaptive_ct.core.loss import projection_weights
from adaptive_ct.core.projector import bernstein_segment_ray_capacity
from adaptive_ct.data.rays import FixedDecisionRaySet, ray_batch


@torch.no_grad()
def attributed_residual_energy(
    model,
    rays: FixedDecisionRaySet,
    *,
    weight_kwargs: dict,
    max_buffer_mb: float = 512.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Residual energy per leaf.

    ``eta_K = sum_r w_r res_r^2 * l_rK / L_r``: each ray's squared residual is shared among the
    leaves it crosses in proportion to the chord length ``l_rK`` inside leaf K (``L_r`` = total
    length), so the leaf values add up to the sampled residual energy.  Also returns per-leaf
    exposure (total chord length) and a summary.
    """
    leaf_count = int(model.leaf_levels.shape[0])
    device = model.leaf_levels.device
    eta = torch.zeros((leaf_count,), dtype=torch.float32, device=device)
    exposure = torch.zeros_like(eta)
    chunk = min(
        rays.chunk_rays,
        bernstein_segment_ray_capacity(model.level_shapes, float(max_buffer_mb)),
    )
    rss = 0.0
    for start in range(0, rays.observation_count, chunk):
        stop = min(start + chunk, rays.observation_count)
        batch = ray_batch(rays, start, stop)
        residual = model.integrate_ray_batch(batch) - batch.target
        weights = projection_weights(batch.target, **weight_kwargs)
        energy = weights * torch.square(residual)
        ray_ids, leaf_ids, intervals = model._parallel_ray_segments(batch)
        segment = intervals[:, 1] - intervals[:, 0]
        path = torch.zeros((batch.num_rays,), dtype=segment.dtype, device=device)
        path.scatter_add_(0, ray_ids, segment)
        share = segment / path[ray_ids].clamp_min(1.0e-12)
        eta.scatter_add_(0, leaf_ids, energy[ray_ids] * share)
        exposure.scatter_add_(0, leaf_ids, segment)
        rss += float(energy.sum().item())
    return eta, exposure, {
        "estimator_rays": int(rays.observation_count),
        "estimator_weighted_rss": rss,
        "covered_fraction": float((exposure > 0).float().mean().item()) if leaf_count else 0.0,
    }


def resolvable_mask(model, *, detector_h: int, detector_w: int, min_footprint_px: float) -> torch.Tensor:
    """Leaves whose children stay resolvable on the detector.

    Axes 0/1 project onto detector columns, axis 2 onto detector rows (same
    convention as ``make_parallel_ray_points``).  A child cell must span at
    least ``min_footprint_px`` detector pixels along every axis.
    """
    shapes = torch.tensor(model.level_shapes, dtype=torch.float32, device=model.leaf_levels.device)
    child_width = 1.0 / shapes[model.leaf_levels]  # 2 / (2 * shape)
    pixel = torch.tensor(
        [2.0 / float(detector_w), 2.0 / float(detector_w), 2.0 / float(detector_h)],
        dtype=torch.float32,
        device=model.leaf_levels.device,
    )
    fine_enough = torch.all(child_width >= float(min_footprint_px) * pixel[None, :] - 1.0e-9, dim=1)
    return fine_enough & (model.leaf_levels + 1 < len(model.level_shapes))
