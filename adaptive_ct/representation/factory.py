from __future__ import annotations

from typing import Dict

from adaptive_ct.representation.octree import BernsteinOctree


def build_model(config: Dict) -> BernsteinOctree:
    model_cfg = config["model"]
    return BernsteinOctree(
        levels=model_cfg["levels"],
        max_degree=int(model_cfg["max_degree"]),
        attenuation_shift=float(model_cfg.get("attenuation_shift", -3.0)),
        cuda_integrator=bool(model_cfg.get("cuda_integrator", True)),
        max_leaf_count=model_cfg.get("max_leaf_count"),
    )
