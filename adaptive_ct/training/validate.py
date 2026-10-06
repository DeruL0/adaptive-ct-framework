"""Configuration checks for the projection-driven reconstruction pipeline."""

from __future__ import annotations


def validate_config(config: dict) -> None:
    errors = []
    train_cfg = config.get("training", {}) or {}
    types = (config.get("model", {}) or {}).get("leaf_types")
    if not types or any(t not in ("constant", "trilinear", "step") for t in types):
        errors.append("model.leaf_types must list leaf types from constant / trilinear / step")
    if not isinstance(train_cfg.get("solver"), dict):
        errors.append("training.solver is required (round_tol, round_passes, final_tol, final_passes, groups, group_passes)")
    ref = train_cfg.get("refinement")
    if not isinstance(ref, dict) or ref.get("byte_budget") is None:
        errors.append("training.refinement with byte_budget is required")
    if errors:
        raise ValueError("Invalid config:\n- " + "\n- ".join(errors))
