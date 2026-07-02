from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .backend import dynamic_voxel_integrate, has_native
from .bernstein import BernsteinOctree


@dataclass(frozen=True)
class ModelStats:
    parameter_count: int
    model_bytes: int
    l0_cells: int
    l0_active: int
    l1_active: int
    l2_active: int
    l3_active: int
    active_by_level: tuple[int, ...]
    leaf_cells: int
    max_depth: int
    representation: str


def _expand_mask_halo(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return mask
    pooled = F.max_pool3d(
        mask.to(dtype=torch.float32)[None, None],
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return pooled[0, 0] > 0.0


def _all_coords(resolution: int, *, device: torch.device | None = None) -> torch.Tensor:
    axes = [torch.arange(int(resolution), dtype=torch.long, device=device) for _ in range(3)]
    xx, yy, zz = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)


class DynamicVoxelGrid(nn.Module):
    """Adaptive leaf-voxel CT volume with dyadic split refinement.

    The model starts as a dense coarse voxel grid. Refinement replaces selected
    parent leaf voxels with children at the next configured resolution, so each
    point is decoded from exactly one active leaf rather than an additive stack
    of multi-resolution features.
    """

    representation = "dynamic_leaf_voxel"

    def __init__(
        self,
        *,
        levels: Iterable[int],
        attenuation_shift: float = -3.0,
        init_std: float = 1e-3,
        cuda_integrator: bool = True,
    ):
        super().__init__()
        self.level_resolutions = [int(v) for v in levels]
        if not self.level_resolutions:
            raise ValueError("At least one voxel resolution is required.")
        for resolution in self.level_resolutions:
            if resolution <= 0:
                raise ValueError(f"Voxel resolutions must be positive, got {resolution}.")
        for parent, child in zip(self.level_resolutions, self.level_resolutions[1:]):
            if child % parent != 0:
                raise ValueError(f"Each voxel level must be an integer refinement of the previous one: {parent}, {child}.")
        self.l0_resolution = self.level_resolutions[0]
        self.l1_resolution = self.level_resolutions[1] if len(self.level_resolutions) > 1 else self.l0_resolution
        self.l2_resolution = self.level_resolutions[2] if len(self.level_resolutions) > 2 else self.l1_resolution
        self.attenuation_shift = float(attenuation_shift)
        self.cuda_integrator = bool(cuda_integrator)

        base_coords = _all_coords(self.l0_resolution)
        self.leaf_logits = nn.Parameter(torch.empty(base_coords.shape[0], dtype=torch.float32))
        nn.init.normal_(self.leaf_logits, mean=0.0, std=float(init_std))
        self.register_buffer("leaf_levels", torch.zeros(base_coords.shape[0], dtype=torch.long), persistent=True)
        self.register_buffer("leaf_coords", base_coords, persistent=True)
        for level, resolution in enumerate(self.level_resolutions):
            self.register_buffer(
                self._index_name(level),
                torch.full((resolution, resolution, resolution), -1, dtype=torch.long),
                persistent=False,
            )
        self._rebuild_index_maps()

    @staticmethod
    def _index_name(level: int) -> str:
        return f"level_{int(level)}_index"

    def _level_index(self, level: int) -> torch.Tensor:
        return getattr(self, self._index_name(level))

    def _leaf_mu(self) -> torch.Tensor:
        return F.softplus(self.leaf_logits + self.attenuation_shift)

    @torch.no_grad()
    def _rebuild_index_maps(self) -> None:
        device = self.leaf_logits.device
        for level, resolution in enumerate(self.level_resolutions):
            index = torch.full((resolution, resolution, resolution), -1, dtype=torch.long, device=device)
            mask = self.leaf_levels == level
            if torch.any(mask):
                coords = self.leaf_coords[mask].to(device=device, dtype=torch.long)
                leaf_ids = torch.nonzero(mask, as_tuple=False).reshape(-1).to(device=device, dtype=torch.long)
                index[coords[:, 0], coords[:, 1], coords[:, 2]] = leaf_ids
            setattr(self, self._index_name(level), index)

    def forward_mu(self, points: torch.Tensor) -> torch.Tensor:
        original_shape = points.shape[:-1]
        flat_points = points.reshape(-1, 3)
        values = flat_points.new_zeros((flat_points.shape[0],), dtype=torch.float32)
        unresolved = torch.ones(flat_points.shape[0], dtype=torch.bool, device=flat_points.device)
        leaf_mu = self._leaf_mu()

        for level in range(len(self.level_resolutions) - 1, -1, -1):
            unresolved_ids = torch.nonzero(unresolved, as_tuple=False).reshape(-1)
            if unresolved_ids.numel() == 0:
                break
            resolution = self.level_resolutions[level]
            scaled = (flat_points[unresolved_ids] + 1.0) * 0.5 * float(resolution)
            coords = torch.floor(scaled).to(dtype=torch.long)
            valid = torch.all((coords >= 0) & (coords < resolution), dim=1)
            if not torch.any(valid):
                continue
            valid_ids = unresolved_ids[valid]
            valid_coords = coords[valid]
            leaf_ids = self._level_index(level)[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]]
            hit = leaf_ids >= 0
            if not torch.any(hit):
                continue
            hit_ids = valid_ids[hit]
            values[hit_ids] = leaf_mu[leaf_ids[hit]]
            unresolved[hit_ids] = False
        return values.reshape(original_shape)

    def integrate_ray_batch(self, ray_batch) -> torch.Tensor:
        if self._can_use_cuda_integrator(ray_batch):
            return dynamic_voxel_integrate(
                leaf_logits=self.leaf_logits,
                index_maps=[self._level_index(level) for level in range(len(self.level_resolutions))],
                angles=ray_batch.angles.to(device=self.leaf_logits.device, dtype=torch.float32),
                rows=ray_batch.rows.to(device=self.leaf_logits.device, dtype=torch.long),
                cols=ray_batch.cols.to(device=self.leaf_logits.device, dtype=torch.long),
                detector_h=int(ray_batch.detector_h),
                detector_w=int(ray_batch.detector_w),
                resolutions=self.level_resolutions,
                attenuation_shift=self.attenuation_shift,
            )
        if ray_batch.points is None or ray_batch.step is None:
            raise ValueError("RayBatch points are not materialized and the CUDA integrator is unavailable.")
        mu = self.forward_mu(ray_batch.points).reshape(ray_batch.num_rays, ray_batch.samples_per_ray)
        return torch.sum(mu * ray_batch.step[:, None], dim=1)

    def _can_use_cuda_integrator(self, ray_batch) -> bool:
        return (
            self.cuda_integrator
            and self.leaf_logits.is_cuda
            and has_native()
            and ray_batch.angles is not None
            and ray_batch.rows is not None
            and ray_batch.cols is not None
            and ray_batch.detector_h is not None
            and ray_batch.detector_w is not None
            and len(self.level_resolutions) <= 3
        )

    def prefer_compact_ray_batch(self) -> bool:
        return self.cuda_integrator and self.leaf_logits.is_cuda and has_native() and len(self.level_resolutions) <= 3

    def decoded_at_resolution(self, resolution: int | tuple[int, int, int], *, chunk: int | None = None) -> torch.Tensor:
        if isinstance(resolution, int):
            shape = (int(resolution), int(resolution), int(resolution))
        else:
            shape = tuple(int(v) for v in resolution)
        device = self.leaf_logits.device
        axes = [torch.arange(n, dtype=torch.float32, device=device) for n in shape]
        xx, yy, zz = torch.meshgrid(*axes, indexing="ij")
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        scale = torch.tensor(shape, dtype=torch.float32, device=device)
        points = -1.0 + (coords + 0.5) * 2.0 / scale
        if chunk is None:
            return self.forward_mu(points).reshape(shape)
        values = []
        for start in range(0, points.shape[0], int(chunk)):
            values.append(self.forward_mu(points[start : start + int(chunk)]))
        return torch.cat(values, dim=0).reshape(shape)

    def decoded_l0(self) -> torch.Tensor:
        return self.decoded_at_resolution(self.l0_resolution)

    @torch.no_grad()
    def _split_parent_leaf_ids(self, parent_level: int, parent_leaf_ids: torch.Tensor) -> int:
        parent_level = int(parent_level)
        child_level = parent_level + 1
        if child_level >= len(self.level_resolutions):
            raise ValueError(f"Cannot split beyond configured voxel level {parent_level}.")
        parent_leaf_ids = torch.unique(parent_leaf_ids.to(device=self.leaf_logits.device, dtype=torch.long))
        parent_leaf_ids = parent_leaf_ids[parent_leaf_ids >= 0]
        if parent_leaf_ids.numel() == 0:
            return int(torch.sum(self.leaf_levels == child_level).item())

        child_ratio = self.level_resolutions[child_level] // self.level_resolutions[parent_level]
        offsets = _all_coords(child_ratio, device=self.leaf_logits.device)
        parent_coords = self.leaf_coords[parent_leaf_ids].to(device=self.leaf_logits.device, dtype=torch.long)
        child_coords = (parent_coords[:, None, :] * child_ratio + offsets[None, :, :]).reshape(-1, 3)
        child_levels = torch.full((child_coords.shape[0],), child_level, dtype=torch.long, device=self.leaf_logits.device)
        child_logits = self.leaf_logits.detach()[parent_leaf_ids].repeat_interleave(offsets.shape[0])

        keep = torch.ones(self.leaf_logits.shape[0], dtype=torch.bool, device=self.leaf_logits.device)
        keep[parent_leaf_ids] = False
        kept_logits = self.leaf_logits.detach()[keep]
        kept_levels = self.leaf_levels[keep].detach()
        kept_coords = self.leaf_coords[keep].detach()

        self.leaf_logits = nn.Parameter(torch.cat([kept_logits, child_logits], dim=0).contiguous())
        self.leaf_levels = torch.cat([kept_levels, child_levels], dim=0).contiguous()
        self.leaf_coords = torch.cat([kept_coords, child_coords], dim=0).contiguous()
        self._rebuild_index_maps()
        return int(torch.sum(self.leaf_levels == child_level).item())

    @torch.no_grad()
    def activate_level_from_score(self, level: int, score: torch.Tensor, active_fraction: float, halo: int = 1) -> int:
        level = int(level)
        if level <= 0 or level >= len(self.level_resolutions):
            raise ValueError(f"Refinement level must be in 1..{len(self.level_resolutions) - 1}, got {level}.")
        target_res = self.level_resolutions[level]
        score = score.detach().to(device=self.leaf_logits.device, dtype=torch.float32)
        if tuple(score.shape) != (target_res, target_res, target_res):
            score = F.interpolate(
                score[None, None],
                size=(target_res, target_res, target_res),
                mode="trilinear",
                align_corners=False,
            )[0, 0]
        score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
        k = max(1, int(score.numel() * float(active_fraction)))
        top_ids = torch.topk(score.reshape(-1), k).indices
        selected = torch.zeros_like(score, dtype=torch.bool).reshape(-1)
        selected[top_ids] = True
        selected = _expand_mask_halo(selected.reshape_as(score), halo)
        target_coords = torch.nonzero(selected, as_tuple=False)
        if target_coords.numel() == 0:
            return int(torch.sum(self.leaf_levels == level).item())

        parent_level = level - 1
        parent_ratio = self.level_resolutions[level] // self.level_resolutions[parent_level]
        parent_coords = torch.unique(torch.div(target_coords, parent_ratio, rounding_mode="floor"), dim=0)
        parent_index = self._level_index(parent_level)
        parent_leaf_ids = parent_index[parent_coords[:, 0], parent_coords[:, 1], parent_coords[:, 2]]
        return self._split_parent_leaf_ids(parent_level, parent_leaf_ids)

    @torch.no_grad()
    def activate_level_from_gradient(self, level: int, active_fraction: float, halo: int = 1) -> int:
        level = int(level)
        volume = self.decoded_at_resolution(self.level_resolutions[level]).detach()
        gx = torch.zeros_like(volume)
        gy = torch.zeros_like(volume)
        gz = torch.zeros_like(volume)
        gx[1:-1, :, :] = 0.5 * (volume[2:, :, :] - volume[:-2, :, :])
        gy[:, 1:-1, :] = 0.5 * (volume[:, 2:, :] - volume[:, :-2, :])
        gz[:, :, 1:-1] = 0.5 * (volume[:, :, 2:] - volume[:, :, :-2])
        grad = torch.sqrt(gx * gx + gy * gy + gz * gz)
        return self.activate_level_from_score(level, grad, active_fraction, halo=halo)

    def stats(self) -> ModelStats:
        active_by_level = [
            int(torch.sum(self.leaf_levels == level).item()) for level in range(len(self.level_resolutions))
        ]
        param_count = sum(int(p.numel()) for p in self.parameters())
        model_bytes = sum(int(p.numel() * p.element_size()) for p in self.parameters())
        model_bytes += sum(int(b.numel() * b.element_size()) for b in (self.leaf_levels, self.leaf_coords))
        return ModelStats(
            parameter_count=param_count,
            model_bytes=model_bytes,
            l0_cells=self.l0_resolution ** 3,
            l0_active=active_by_level[0] if len(active_by_level) > 0 else 0,
            l1_active=active_by_level[1] if len(active_by_level) > 1 else 0,
            l2_active=active_by_level[2] if len(active_by_level) > 2 else 0,
            l3_active=active_by_level[3] if len(active_by_level) > 3 else 0,
            active_by_level=tuple(active_by_level),
            leaf_cells=int(self.leaf_logits.shape[0]),
            max_depth=max((level for level, count in enumerate(active_by_level) if count > 0), default=0),
            representation=self.representation,
        )

    def prepare_sparse_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if "leaf_logits" not in state_dict:
            raise ValueError("Checkpoint does not contain dynamic voxel leaf_logits.")
        device = self.leaf_logits.device
        self.leaf_logits = nn.Parameter(torch.empty_like(state_dict["leaf_logits"], device=device))
        self.leaf_levels = torch.empty_like(state_dict["leaf_levels"], device=device)
        self.leaf_coords = torch.empty_like(state_dict["leaf_coords"], device=device)

    def load_state_dict(self, state_dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._rebuild_index_maps()
        return result


def build_model(config: Dict) -> DynamicVoxelGrid | BernsteinOctree:
    model_cfg = config["model"]
    representation = str(model_cfg.get("representation", "dynamic_leaf_voxel")).lower()
    if representation in {"bernstein_octree", "rd_cvf", "bernstein"}:
        return BernsteinOctree(
            levels=model_cfg["levels"],
            max_degree=model_cfg.get("max_degree", 3),
            attenuation_shift=float(model_cfg.get("attenuation_shift", -3.0)),
            init_std=float(model_cfg.get("init_std", 1e-3)),
            cuda_integrator=bool(model_cfg.get("cuda_integrator", True)),
            integration_mode=str(model_cfg.get("integration_mode", "exact")),
            topology=str(model_cfg.get("topology", "packed_hierarchy")),
            balance_2to1=bool(model_cfg.get("balance_2to1", False)),
            max_leaf_count=model_cfg.get("max_leaf_count"),
        )
    if representation not in {"dynamic_leaf_voxel", "dynamic_voxel", "voxel", "octree"}:
        raise ValueError(f"Unsupported model representation {representation!r}.")
    levels = model_cfg["levels"]
    return DynamicVoxelGrid(
        levels=levels,
        attenuation_shift=float(model_cfg.get("attenuation_shift", -3.0)),
        init_std=float(model_cfg.get("init_std", 1e-3)),
        cuda_integrator=bool(model_cfg.get("cuda_integrator", True)),
    )
