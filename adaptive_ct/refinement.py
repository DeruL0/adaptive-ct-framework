from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .data import ProjectionSplit
from .render import render_split


@dataclass(frozen=True)
class RefinementScore:
    score: torch.Tensor
    components: Dict[str, torch.Tensor]
    summary: Dict[str, float]


def target_resolution_for_level(model, level: int) -> int:
    if int(level) == 1:
        return int(model.l1_resolution)
    if int(level) == 2:
        return int(model.l2_resolution)
    raise ValueError(f"Unsupported sparse level {level}.")


@torch.no_grad()
def decoded_volume(model, resolution: int | tuple[int, int, int], *, chunk: int = 65536) -> torch.Tensor:
    if hasattr(model, "decoded_at_resolution"):
        return model.decoded_at_resolution(resolution, chunk=chunk).detach()
    if isinstance(resolution, int):
        shape = (int(resolution), int(resolution), int(resolution))
    else:
        shape = tuple(int(v) for v in resolution)
    device = next(model.parameters()).device
    axes = [torch.arange(n, dtype=torch.float32, device=device) for n in shape]
    xx, yy, zz = torch.meshgrid(*axes, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    scale = torch.tensor(shape, dtype=torch.float32, device=device)
    points = -1.0 + (coords + 0.5) * 2.0 / scale
    values = []
    for start in range(0, points.shape[0], int(chunk)):
        values.append(model.forward_mu(points[start : start + int(chunk)]).detach())
    return torch.cat(values, dim=0).reshape(shape)


def gradient_magnitude(volume: torch.Tensor) -> torch.Tensor:
    gx = torch.zeros_like(volume)
    gy = torch.zeros_like(volume)
    gz = torch.zeros_like(volume)
    gx[1:-1, :, :] = 0.5 * (volume[2:, :, :] - volume[:-2, :, :])
    gy[:, 1:-1, :] = 0.5 * (volume[:, 2:, :] - volume[:, :-2, :])
    gz[:, :, 1:-1] = 0.5 * (volume[:, :, 2:] - volume[:, :, :-2])
    return torch.sqrt(gx * gx + gy * gy + gz * gz)


def local_sigma(volume: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = int(kernel_size) // 2
    x = volume[None, None]
    mean = F.avg_pool3d(x, kernel_size=int(kernel_size), stride=1, padding=pad)
    mean_sq = F.avg_pool3d(x * x, kernel_size=int(kernel_size), stride=1, padding=pad)
    return torch.sqrt((mean_sq - mean * mean).clamp_min(0.0))[0, 0]


def normalize_score(score: torch.Tensor) -> torch.Tensor:
    score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
    flat = score.reshape(-1)
    lo = torch.quantile(flat, 0.01)
    hi = torch.quantile(flat, 0.99)
    if not torch.isfinite(hi) or float((hi - lo).detach().item()) < 1e-8:
        hi = flat.max()
        lo = flat.min()
    return ((score - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)


@torch.no_grad()
def projection_residuals(
    model,
    split: ProjectionSplit,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    ray_chunk: int,
    max_views: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_views is not None:
        view_count = min(int(max_views), int(split.projections.shape[0]))
        split = ProjectionSplit(
            angles=split.angles[:view_count],
            projections=split.projections[:view_count],
            paths=split.paths[:view_count],
        )
    rendered = render_split(
        model,
        split,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=ray_chunk,
    )
    return torch.abs(rendered - split.projections), split.angles


@torch.no_grad()
def backproject_residual_score(
    residuals: torch.Tensor,
    angles: torch.Tensor,
    *,
    resolution: int,
    voxel_chunk: int = 65536,
) -> torch.Tensor:
    device = residuals.device
    res = int(resolution)
    coords_1d = torch.arange(res, dtype=torch.float32, device=device)
    xx, yy, zz = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    points = -1.0 + (coords + 0.5) * 2.0 / float(res)
    residuals_4d = residuals[:, None].to(dtype=torch.float32)
    score = torch.zeros((points.shape[0],), dtype=torch.float32, device=device)
    effective_angles = -angles.to(device=device, dtype=torch.float32)
    for angle_idx, angle in enumerate(effective_angles):
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        u_axis = torch.stack([-sa, ca])
        image = residuals_4d[angle_idx : angle_idx + 1]
        for start in range(0, points.shape[0], int(voxel_chunk)):
            chunk = points[start : start + int(voxel_chunk)]
            u = chunk[:, 0] * u_axis[0] + chunk[:, 1] * u_axis[1]
            z = chunk[:, 2]
            grid = torch.stack([u, z], dim=-1).view(1, -1, 1, 2)
            sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            score[start : start + chunk.shape[0]] += sampled.reshape(-1)
    score /= max(int(angles.shape[0]), 1)
    return score.reshape(res, res, res)


@torch.no_grad()
def build_refinement_score(
    model,
    dataset,
    *,
    level: int,
    mode: str,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    ray_chunk: int,
    residual_split: str = "train",
    residual_views: int | None = 8,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.25,
) -> RefinementScore:
    resolution = target_resolution_for_level(model, int(level))
    volume = decoded_volume(model, resolution)
    grad = gradient_magnitude(volume)
    sigma = local_sigma(volume)
    components = {
        "gradient": normalize_score(grad),
        "sigma_consist": normalize_score(sigma),
    }
    normalized_mode = str(mode).lower()
    if normalized_mode == "hybrid":
        split = dataset.train if str(residual_split).lower() == "train" else dataset.test
        residual, angles = projection_residuals(
            model,
            split,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            ray_chunk=ray_chunk,
            max_views=residual_views,
        )
        components["projected_residual"] = normalize_score(
            backproject_residual_score(residual, angles, resolution=resolution)
        )
    elif normalized_mode != "gradient":
        raise ValueError("refinement strategy must be 'gradient' or 'hybrid'.")

    score = float(alpha) * components["gradient"] + float(gamma) * components["sigma_consist"]
    if "projected_residual" in components:
        score = score + float(beta) * components["projected_residual"]
    score = normalize_score(score)
    summary = {
        "resolution": float(resolution),
        "score_mean": float(score.mean().item()),
        "score_max": float(score.max().item()),
        "gradient_mean": float(components["gradient"].mean().item()),
        "sigma_mean": float(components["sigma_consist"].mean().item()),
    }
    if "projected_residual" in components:
        summary["projected_residual_mean"] = float(components["projected_residual"].mean().item())
    return RefinementScore(score=score, components=components, summary=summary)
