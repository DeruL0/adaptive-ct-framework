"""Configuration checks for the projection-driven reconstruction pipeline."""

from __future__ import annotations


def validate_config(config: dict) -> None:
    errors = []
    train_cfg = config.get("training", {}) or {}
    if int((config.get("model", {}) or {}).get("max_degree", -1)) not in (0, 1):
        errors.append("model.max_degree must be 0 (constant leaves) or 1 (trilinear leaves)")
    if not isinstance(train_cfg.get("solver"), dict):
        errors.append("training.solver is required (round_tol, round_passes, final_tol, final_passes, groups, group_passes)")
    ref = train_cfg.get("refinement")
    if not isinstance(ref, dict) or ref.get("byte_budget") is None or not isinstance(ref.get("codec"), dict):
        errors.append("training.refinement with byte_budget and codec is required")
    if not isinstance((config.get("export", {}) or {}).get("codec"), dict):
        errors.append("export.codec is required")
    if errors:
        raise ValueError("Invalid config:\n- " + "\n- ".join(errors))
