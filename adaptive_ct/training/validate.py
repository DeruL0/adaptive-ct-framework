"""Configuration checks for the projection-driven refinement framework."""

from __future__ import annotations

OPERATION = "dorfler_h_split"


def validate_config(config: dict) -> None:
    errors = []
    model_cfg = config.get("model", {}) or {}
    train_cfg = config.get("training", {}) or {}
    progressive = train_cfg.get("progressive", {}) or {}
    if str(model_cfg.get("representation", "")).lower() not in {"bernstein_octree", "bernstein"}:
        errors.append("this framework trains a Bernstein octree; latent codecs are applied at export only")
    if str(model_cfg.get("initialization", "projection_mean")).lower() != "projection_mean":
        errors.append("this framework must initialize from projection_mean only (no FDK/reference volume)")
    if bool(train_cfg.get("allow_test_during_training", False)):
        errors.append("test views must not be used during training")
    if float(train_cfg.get("tv_weight", 0.0)) != 0.0:
        errors.append("tv_weight must be 0")
    for key in ("volume_loss", "projection_gradient_loss", "coefficient_continuity"):
        if float((train_cfg.get(key, {}) or {}).get("weight", 0.0)) != 0.0:
            errors.append(f"training.{key}.weight must be 0")
    milestones = [item for item in (progressive.get("convergence_milestones", []) or []) if item]
    if len(milestones) != 1 or str(milestones[0].get("operation", "")).lower() != OPERATION:
        errors.append("this framework needs exactly one convergence milestone with operation dorfler_h_split")
    else:
        spec = milestones[0]
        if bool(spec.get("rollback", True)) and int((config.get("dataset", {}) or {}).get("dev_views", 0)) < 1:
            errors.append("rollback needs dataset.dev_views >= 1")
        if spec.get("byte_budget") is None:
            errors.append("dorfler_h_split needs a byte_budget")
    if not bool((train_cfg.get("fixed_decision_rays", {}) or {}).get("enabled", False)):
        errors.append("training.fixed_decision_rays must be enabled (it is the plateau ruler)")
    if progressive.get("milestones") or (progressive.get("structural_rounds", {}) or {}).get("enabled", False):
        errors.append("this framework uses only the dorfler convergence milestone")
    if errors:
        raise ValueError("Invalid refinement config:\n- " + "\n- ".join(errors))


def active_milestone(config: dict) -> dict:
    progressive = (config.get("training", {}) or {}).get("progressive", {}) or {}
    for item in progressive.get("convergence_milestones", []) or []:
        if item:
            return dict(item)
    raise ValueError("config has no convergence milestone.")
