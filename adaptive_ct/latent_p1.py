from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .bernstein import BernsteinOctree, BernsteinStats, TopologyChange, _all_coords


class LatentP1Octree(BernsteinOctree):
    """Packed octree with per-leaf latents and one shared P1 decoder."""

    representation = "latent_p1_octree"

    def __init__(
        self,
        *,
        levels,
        latent_dim: int = 4,
        attenuation_shift: float = -3.0,
        init_std: float = 1e-3,
        cuda_integrator: bool = True,
        integration_mode: str = "exact",
        topology: str = "packed_hierarchy",
        balance_2to1: bool = False,
        max_leaf_count: int | None = None,
    ):
        super().__init__(
            levels=levels,
            max_degree=(1, 1, 1),
            attenuation_shift=attenuation_shift,
            init_std=init_std,
            cuda_integrator=cuda_integrator,
            integration_mode=integration_mode,
            topology=topology,
            balance_2to1=balance_2to1,
            max_leaf_count=max_leaf_count,
        )
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        leaf_count = int(self.leaf_levels.numel())
        del self._parameters["coefficient_logits"]
        self.register_buffer("coefficient_logits", torch.empty((0,), dtype=torch.float32), persistent=False)
        self.leaf_latents = nn.Parameter(torch.empty((leaf_count, self.latent_dim), dtype=torch.float32))
        nn.init.normal_(self.leaf_latents, mean=0.0, std=float(init_std))
        self.decoder = nn.Linear(self.latent_dim, 8)
        self._set_all_p1_metadata(leaf_count, self.leaf_latents.device)

    def _set_all_p1_metadata(self, leaf_count: int, device: torch.device) -> None:
        self.leaf_degrees = torch.ones((leaf_count, 3), dtype=torch.long, device=device)
        self.coefficient_offsets = torch.arange(0, leaf_count * 8 + 1, 8, dtype=torch.long, device=device)
        self.coefficient_leaf_ids = torch.arange(leaf_count, dtype=torch.long, device=device).repeat_interleave(8)

    def local_coefficient_logits(self) -> torch.Tensor:
        return self.decoder(self.leaf_latents).reshape(-1)

    def coefficients(self) -> torch.Tensor:
        return F.softplus(self.local_coefficient_logits() + self.attenuation_shift)

    def master_coefficients(self) -> torch.Tensor:
        return self.coefficients()

    @torch.no_grad()
    def initialize_constant_mu(self, value: float) -> None:
        target = torch.full((self.leaf_latents.shape[0], 8), float(value), device=self.leaf_latents.device)
        target_logits = self._coefficients_to_logits(target).mean(dim=0)
        current_logits = self.decoder(self.leaf_latents).mean(dim=0)
        self.decoder.bias.add_(target_logits - current_logits)

    @torch.no_grad()
    def _child_latents_from_parent(self, leaf_ids: torch.Tensor) -> torch.Tensor:
        """Approximate each parent P1 field on its eight child cells."""
        parent_coefficients = self.coefficients().reshape(-1, 8)[leaf_ids].reshape(-1, 2, 2, 2)
        child_offsets = _all_coords(2, device=self.leaf_latents.device).to(dtype=parent_coefficients.dtype)
        child_corners = child_offsets
        parent_local = (child_offsets[:, None, :] + child_corners[None, :, :]) * 0.5
        basis_x = torch.stack((1.0 - parent_local[..., 0], parent_local[..., 0]), dim=-1)
        basis_y = torch.stack((1.0 - parent_local[..., 1], parent_local[..., 1]), dim=-1)
        basis_z = torch.stack((1.0 - parent_local[..., 2], parent_local[..., 2]), dim=-1)
        target_coefficients = torch.einsum(
            "cpi,cpj,cpk,nijk->ncp",
            basis_x,
            basis_y,
            basis_z,
            parent_coefficients,
        ).reshape(-1, 8)
        target_logits = self._coefficients_to_logits(target_coefficients)
        decoder_pinv = torch.linalg.pinv(self.decoder.weight)
        return (target_logits - self.decoder.bias) @ decoder_pinv.T

    @torch.no_grad()
    def split_leaves_batch(self, leaf_ids: torch.Tensor | Sequence[int]) -> TopologyChange:
        leaf_ids_tensor = torch.unique(
            torch.as_tensor(leaf_ids, dtype=torch.long, device=self.leaf_latents.device),
            sorted=True,
        )
        leaf_ids_tensor = leaf_ids_tensor[(leaf_ids_tensor >= 0) & (leaf_ids_tensor < self.leaf_levels.shape[0])]
        if self.max_leaf_count is not None:
            available = max(0, (int(self.max_leaf_count) - int(self.leaf_levels.shape[0])) // 7)
            leaf_ids_tensor = leaf_ids_tensor[:available]
        old_count = int(self.stats().parameter_count)
        if leaf_ids_tensor.numel() == 0:
            return TopologyChange("h_split_batch", tuple(), tuple(), old_count, old_count)
        levels = self.leaf_levels[leaf_ids_tensor]
        if torch.any(levels + 1 >= len(self.level_shapes)):
            raise ValueError("At least one selected leaf is already at maximum octree depth.")

        old_keys = [self.leaf_key(int(leaf_id)) for leaf_id in leaf_ids_tensor.tolist()]
        offsets = _all_coords(2, device=self.leaf_latents.device)
        new_coords = (self.leaf_coords[leaf_ids_tensor][:, None, :] * 2 + offsets[None, :, :]).reshape(-1, 3)
        new_levels = (levels[:, None] + 1).expand(-1, 8).reshape(-1)
        new_latents = self._child_latents_from_parent(leaf_ids_tensor)

        keep = torch.ones(self.leaf_levels.shape[0], dtype=torch.bool, device=self.leaf_latents.device)
        keep[leaf_ids_tensor] = False
        self.leaf_levels = torch.cat([self.leaf_levels[keep], new_levels], dim=0).contiguous()
        self.leaf_coords = torch.cat([self.leaf_coords[keep], new_coords], dim=0).contiguous()
        self.leaf_latents = nn.Parameter(torch.cat([self.leaf_latents.detach()[keep], new_latents], dim=0).contiguous())
        self._set_all_p1_metadata(int(self.leaf_levels.numel()), self.leaf_latents.device)
        self._rebuild_packed_topology()
        new_keys = [self.leaf_key(index) for index in range(int(self.leaf_levels.numel()) - int(new_levels.numel()), int(self.leaf_levels.numel()))]
        return TopologyChange("h_split_batch", tuple(old_keys), tuple(new_keys), old_count, int(self.stats().parameter_count))

    @torch.no_grad()
    def split_leaf(self, leaf_id: int) -> TopologyChange:
        return self.split_leaves_batch([int(leaf_id)])

    @torch.no_grad()
    def merge_siblings(self, child_ids: Sequence[int]) -> TopologyChange:
        ids = torch.unique(torch.as_tensor(list(child_ids), dtype=torch.long, device=self.leaf_latents.device), sorted=True)
        if ids.numel() != 8:
            raise ValueError("An octree merge requires exactly eight sibling leaves.")
        levels = self.leaf_levels[ids]
        if torch.any(levels != levels[0]) or int(levels[0].item()) <= 0:
            raise ValueError("Merge candidates must share one non-root level.")
        parent_coords = torch.div(self.leaf_coords[ids], 2, rounding_mode="floor")
        if torch.unique(parent_coords, dim=0).shape[0] != 1:
            raise ValueError("Merge candidates are not siblings.")

        old_count = int(self.stats().parameter_count)
        old_keys = [self.leaf_key(int(leaf_id)) for leaf_id in ids.tolist()]
        keep = torch.ones(self.leaf_levels.shape[0], dtype=torch.bool, device=self.leaf_latents.device)
        keep[ids] = False
        self.leaf_levels = torch.cat([self.leaf_levels[keep], (levels[0] - 1).reshape(1)], dim=0).contiguous()
        self.leaf_coords = torch.cat([self.leaf_coords[keep], parent_coords[:1]], dim=0).contiguous()
        self.leaf_latents = nn.Parameter(
            torch.cat([self.leaf_latents.detach()[keep], self.leaf_latents.detach()[ids].mean(dim=0, keepdim=True)], dim=0).contiguous()
        )
        self._set_all_p1_metadata(int(self.leaf_levels.numel()), self.leaf_latents.device)
        self._rebuild_packed_topology()
        return TopologyChange("h_merge", tuple(old_keys), (self.leaf_key(int(self.leaf_levels.numel()) - 1),), old_count, int(self.stats().parameter_count))

    @classmethod
    def from_bernstein(cls, model: BernsteinOctree, latent_dim: int = 4) -> "LatentP1Octree":
        """Fit a shared linear decoder to a trained all-P1 Bernstein octree.

        Each leaf's eight corner logits are projected on their top
        ``latent_dim`` principal axes (exact when ``latent_dim`` is 8).  This is
        a codec step only: topology is copied unchanged.
        """
        latent_dim = int(latent_dim)
        if not 1 <= latent_dim <= 8:
            raise ValueError("latent_dim must be in [1, 8].")
        if model.constrained_nodal:
            raise ValueError("Constrained nodal models are not supported.")
        leaf_count = int(model.leaf_levels.shape[0])
        device = model.leaf_levels.device
        if not bool(torch.all(model.leaf_degrees == 1)):
            raise ValueError("from_bernstein requires every leaf to be trilinear (degree 1).")
        offsets = torch.arange(0, leaf_count * 8 + 1, 8, dtype=torch.long, device=device)
        if not torch.equal(model.coefficient_offsets, offsets):
            raise ValueError("from_bernstein requires eight contiguous coefficients per leaf.")
        logits = model.coefficient_logits.detach().reshape(leaf_count, 8).double()
        bias = logits.mean(dim=0)
        centered = logits - bias
        _, axes = torch.linalg.eigh(centered.T @ centered)
        weight = axes[:, -latent_dim:].flip(-1).contiguous()
        latents = centered @ weight
        out = cls(
            levels=model.level_shapes,
            latent_dim=latent_dim,
            attenuation_shift=float(model.attenuation_shift),
            cuda_integrator=bool(model.cuda_integrator),
            integration_mode=str(model.integration_mode),
            topology="packed_hierarchy",
            balance_2to1=False,
            max_leaf_count=model.max_leaf_count,
        ).to(device=device)
        state = {
            "leaf_latents": latents.float().contiguous(),
            "decoder.weight": weight.float().contiguous(),
            "decoder.bias": bias.float().contiguous(),
            "leaf_levels": model.leaf_levels.detach().clone(),
            "leaf_coords": model.leaf_coords.detach().clone(),
        }
        out.prepare_sparse_from_state_dict(state)
        out.load_state_dict(state)
        return out

    def stats(self) -> BernsteinStats:
        stats = super().stats()
        parameter_count = sum(int(p.numel()) for p in self.parameters())
        model_bytes = sum(int(p.numel() * p.element_size()) for p in self.parameters())
        model_bytes += sum(
            int(buffer.numel() * buffer.element_size())
            for buffer in (self.leaf_levels, self.leaf_coords, self.leaf_degrees, self.coefficient_offsets, self.coefficient_leaf_ids)
        )
        return replace(
            stats,
            parameter_count=parameter_count,
            model_bytes=model_bytes,
            representation=self.representation,
            coefficient_count=int(self.leaf_levels.numel()) * 8,
            max_degree=(1, 1, 1),
        )

    def prepare_sparse_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        if "leaf_latents" not in state_dict:
            raise ValueError("Latent P1 checkpoint is missing leaf_latents.")
        device = self.leaf_latents.device
        self.leaf_latents = nn.Parameter(torch.empty_like(state_dict["leaf_latents"], device=device))
        for name in ("leaf_levels", "leaf_coords"):
            if name not in state_dict:
                raise ValueError(f"Latent P1 checkpoint is missing {name}.")
            setattr(self, name, torch.empty_like(state_dict[name], device=device))
        self._set_all_p1_metadata(int(state_dict["leaf_latents"].shape[0]), device)

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = state_dict.copy()
        leaf_count = int(state_dict["leaf_latents"].shape[0]) if "leaf_latents" in state_dict else int(self.leaf_latents.shape[0])
        device = self.leaf_latents.device
        state_dict.setdefault("leaf_degrees", torch.ones((leaf_count, 3), dtype=torch.long, device=device))
        state_dict.setdefault("coefficient_offsets", torch.arange(0, leaf_count * 8 + 1, 8, dtype=torch.long, device=device))
        state_dict.setdefault("coefficient_leaf_ids", torch.arange(leaf_count, dtype=torch.long, device=device).repeat_interleave(8))
        result = super().load_state_dict(state_dict, strict=False if strict else strict)
        self._set_all_p1_metadata(leaf_count, self.leaf_latents.device)
        self._rebuild_packed_topology()
        return result
