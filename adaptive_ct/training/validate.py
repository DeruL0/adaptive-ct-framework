"""Configuration checks for the projection-driven reconstruction pipeline."""

from __future__ import annotations


def validate_config(config: dict) -> None:
    errors = []
    model_cfg = config.get("model", {}) or {}
    train_cfg = config.get("training", {}) or {}
    if str(model_cfg.get("representation", "")).lower() not in {"bernstein_octree", "bernstein"}:
        errors.append("the pipeline solves for a Bernstein octree; latent codecs are applied at export only")
    if str(model_cfg.get("initialization", "projection_mean")).lower() != "projection_mean":
        errors.append("initialization must be projection_mean (no FDK / reference volume)")
    if bool(train_cfg.get("allow_test_during_training", False)):
        errors.append("test views must not be used during training")
    if not isinstance(train_cfg.get("solver"), dict):
        errors.append("training.solver is required (round_passes, final_passes, groups, group_passes)")
    ref = train_cfg.get("refinement")
    if not isinstance(ref, dict) or ref.get("byte_budget") is None:
        errors.append("training.refinement with a byte_budget is required")
    if errors:
        raise ValueError("Invalid config:\n- " + "\n- ".join(errors))
