"""Configuration checks for the projection-driven reconstruction pipeline."""

from __future__ import annotations


def validate_config(config: dict) -> None:
    errors = []
    train_cfg = config.get("training", {}) or {}
    types = (config.get("model", {}) or {}).get("leaf_types")
    if not types or any(t not in ("constant", "trilinear", "step") for t in types):
        errors.append("model.leaf_types must list leaf types from constant / trilinear / step")
    # Every main-line setting is stated in the config: a missing key is an error, never a silent fallback.
    solver = train_cfg.get("solver")
    keys = ("round_tol", "round_passes", "final_tol", "final_passes", "groups", "first_group_passes", "group_passes",
            "min_leaf_px", "extrapolate", "edge_penalty")
    if not isinstance(solver, dict) or any(k not in solver for k in keys):
        errors.append("training.solver needs " + ", ".join(keys))
    elif not isinstance(solver["edge_penalty"], dict) or {"weight", "delta"} - set(solver["edge_penalty"]):
        errors.append("training.solver.edge_penalty needs weight and delta")
    ref = train_cfg.get("refinement")
    keys = ("byte_budget", "action_share", "stop_below_remaining", "decision_views", "min_footprint_px", "max_rounds")
    if not isinstance(ref, dict) or any(k not in ref for k in keys):
        errors.append("training.refinement needs " + ", ".join(keys))
    if errors:
        raise ValueError("Invalid config:\n- " + "\n- ".join(errors))
