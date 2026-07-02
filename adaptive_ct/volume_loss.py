from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VolumeSampleBatch:
    points: torch.Tensor
    target: torch.Tensor


def _gradient_magnitude(volume: torch.Tensor) -> torch.Tensor:
    gx = torch.zeros_like(volume)
    gy = torch.zeros_like(volume)
    gz = torch.zeros_like(volume)
    gx[1:-1, :, :] = 0.5 * (volume[2:, :, :] - volume[:-2, :, :])
    gy[:, 1:-1, :] = 0.5 * (volume[:, 2:, :] - volume[:, :-2, :])
    gz[:, :, 1:-1] = 0.5 * (volume[:, :, 2:] - volume[:, :, :-2])
    return torch.sqrt(gx * gx + gy * gy + gz * gz)


def _sample_from_pool(pool: torch.Tensor, count: int) -> torch.Tensor:
    if int(count) <= 0:
        return pool.new_empty((0,))
    if pool.numel() == 0:
        raise ValueError("Cannot sample from an empty volume index pool.")
    ids = torch.randint(0, int(pool.numel()), (int(count),), device=pool.device)
    return pool[ids]


class VolumeSampler:
    """Biased voxel-center sampler for explicit volume-domain diagnostics."""

    def __init__(
        self,
        volume: torch.Tensor,
        *,
        material_threshold: float,
        boundary_quantile: float = 0.9,
    ):
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D volume, got shape {tuple(volume.shape)}.")
        self.volume = volume.detach().to(dtype=torch.float32)
        self.shape = tuple(int(v) for v in self.volume.shape)
        flat = self.volume.reshape(-1)
        all_indices = torch.arange(flat.numel(), dtype=torch.long, device=flat.device)
        material_mask = flat > float(material_threshold)
        self.all_indices = all_indices
        self.material_indices = all_indices[material_mask]
        self.background_indices = all_indices[~material_mask]

        grad = _gradient_magnitude(self.volume).reshape(-1)
        if self.material_indices.numel() > 0:
            material_grad = grad[self.material_indices]
            cutoff = torch.quantile(material_grad, float(boundary_quantile))
            boundary_mask = material_mask & (grad >= cutoff)
            self.boundary_indices = all_indices[boundary_mask]
        else:
            self.boundary_indices = all_indices.new_empty((0,))

    def sample(
        self,
        count: int,
        *,
        material_fraction: float = 0.5,
        boundary_fraction: float = 0.25,
    ) -> VolumeSampleBatch:
        count = int(count)
        if count <= 0:
            raise ValueError("Volume sample count must be positive.")
        boundary_count = min(count, max(0, int(round(count * float(boundary_fraction)))))
        material_count = min(count - boundary_count, max(0, int(round(count * float(material_fraction)))))
        remaining = count - boundary_count - material_count

        pools = []
        if boundary_count > 0 and self.boundary_indices.numel() > 0:
            pools.append(_sample_from_pool(self.boundary_indices, boundary_count))
        else:
            remaining += boundary_count
        if material_count > 0 and self.material_indices.numel() > 0:
            pools.append(_sample_from_pool(self.material_indices, material_count))
        else:
            remaining += material_count
        if remaining > 0:
            pool = self.background_indices if self.background_indices.numel() > 0 else self.all_indices
            pools.append(_sample_from_pool(pool, remaining))

        indices = torch.cat(pools, dim=0)
        if indices.numel() != count:
            pad = _sample_from_pool(self.all_indices, count - int(indices.numel()))
            indices = torch.cat([indices, pad], dim=0)

        nx, ny, nz = self.shape
        z = indices % nz
        y = torch.div(indices, nz, rounding_mode="floor") % ny
        x = torch.div(indices, ny * nz, rounding_mode="floor")
        coords = torch.stack([x, y, z], dim=1).to(dtype=torch.float32)
        scale = torch.tensor([nx, ny, nz], dtype=torch.float32, device=coords.device)
        points = -1.0 + (coords + 0.5) * 2.0 / scale
        target = self.volume.reshape(-1)[indices]
        return VolumeSampleBatch(points=points.contiguous(), target=target.contiguous())


def volume_sample_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    normalized = str(loss_type).lower()
    if normalized == "mse":
        return F.mse_loss(pred, target)
    if normalized in {"l1", "mae"}:
        return F.l1_loss(pred, target)
    if normalized in {"huber", "smooth_l1"}:
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unsupported volume loss type {loss_type!r}.")
