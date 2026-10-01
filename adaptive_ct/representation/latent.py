from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
import torch.nn.functional as F

from adaptive_ct.representation.octree import BernsteinOctree, BernsteinStats


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
