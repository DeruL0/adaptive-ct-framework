from __future__ import annotations

from typing import Dict

from .bernstein import BernsteinOctree
from .latent_p1 import LatentP1Octree


def build_model(config: Dict) -> BernsteinOctree | LatentP1Octree:
    model_cfg = config["model"]
    representation = str(model_cfg.get("representation", "bernstein_octree")).lower()
    common = dict(
        levels=model_cfg["levels"],
        attenuation_shift=float(model_cfg.get("attenuation_shift", -3.0)),
        init_std=float(model_cfg.get("init_std", 1e-3)),
        cuda_integrator=bool(model_cfg.get("cuda_integrator", True)),
        integration_mode=str(model_cfg.get("integration_mode", "exact")),
        topology=str(model_cfg.get("topology", "packed_hierarchy")),
        max_leaf_count=model_cfg.get("max_leaf_count"),
    )
    if representation in {"latent_p1_octree", "latent_p1"}:
        return LatentP1Octree(latent_dim=int(model_cfg.get("latent_dim", 4)), **common)
    if representation in {"bernstein_octree", "bernstein"}:
        return BernsteinOctree(max_degree=model_cfg.get("max_degree", 3), **common)
    raise ValueError(f"Unsupported model representation {representation!r}.")
