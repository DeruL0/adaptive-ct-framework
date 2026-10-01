from __future__ import annotations

import torch


def projection_weights(
    target: torch.Tensor,
    epsilon: float = 1e-4,
    *,
    mode: str = "inverse",
    target_power: float = 1.0,
    min_weight: float | None = None,
    max_weight: float | None = None,
    blend_alpha: float = 0.15,
) -> torch.Tensor:
    """Per-ray projection weights.

    The legacy inverse-target mode is kept for compatibility.  For structure
    recovery, configs can use target_power/sqrt_target so high-attenuation rays
    are not suppressed relative to near-air pixels.
    """
    normalized = str(mode).lower()
    detached = target.detach().abs()
    if normalized in {"inverse", "inverse_target", "reciprocal"}:
        weights = torch.reciprocal(detached.clamp_min(float(epsilon)))
    elif normalized in {"uniform", "none", "mse"}:
        weights = torch.ones_like(detached)
    elif normalized in {"target", "target_power", "attenuation", "foreground"}:
        weights = torch.pow(detached + float(epsilon), float(target_power))
    elif normalized in {"sqrt_target", "sqrt_attenuation"}:
        weights = torch.sqrt(detached + float(epsilon))
    elif normalized in {
        "dynamic_range_balanced",
        "balanced",
        "inverse_target_blend",
        "inverse_edge_blend",
        "blend",
    }:
        inverse = torch.reciprocal(detached.clamp_min(float(epsilon)))
        inverse = inverse / inverse.mean().clamp_min(float(epsilon))
        edge = torch.pow(detached + float(epsilon), float(target_power))
        edge = edge / edge.mean().clamp_min(float(epsilon))
        alpha = min(1.0, max(0.0, float(blend_alpha)))
        weights = (1.0 - alpha) * inverse + alpha * edge
    else:
        raise ValueError(
            "projection weight mode must be 'inverse', 'uniform', "
            "'target_power', 'sqrt_target', or 'dynamic_range_balanced'."
        )
    weights = weights / weights.mean().clamp_min(float(epsilon))
    if min_weight is not None or max_weight is not None:
        low = float(min_weight) if min_weight is not None else 0.0
        high = float(max_weight) if max_weight is not None else float("inf")
        weights = weights.clamp(min=low, max=high)
        weights = weights / weights.mean().clamp_min(float(epsilon))
    return weights


def weighted_projection_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-4,
    *,
    mode: str = "inverse",
    target_power: float = 1.0,
    min_weight: float | None = None,
    max_weight: float | None = None,
    blend_alpha: float = 0.15,
) -> torch.Tensor:
    weights = projection_weights(
        target,
        epsilon=epsilon,
        mode=mode,
        target_power=target_power,
        min_weight=min_weight,
        max_weight=max_weight,
        blend_alpha=blend_alpha,
    )
    return torch.mean(weights * torch.square(prediction - target))


def projection_weight_kwargs(train_cfg: dict, *, iteration: int | None = None) -> dict:
    weight_cfg = dict(train_cfg.get("projection_weighting", {}) or {})
    schedule = train_cfg.get("projection_weighting_schedule", {}) or {}
    if schedule:
        current_iteration = 1 if iteration is None else int(iteration)
        stages = sorted((int(key), value or {}) for key, value in schedule.items())
        for start, stage_cfg in stages:
            if start > current_iteration:
                break
            weight_cfg.update(stage_cfg)
    return {
        "epsilon": float(weight_cfg.get("epsilon", 1e-4)),
        "mode": str(weight_cfg.get("mode", "inverse")),
        "target_power": float(weight_cfg.get("target_power", 1.0)),
        "min_weight": weight_cfg.get("min_weight"),
        "max_weight": weight_cfg.get("max_weight"),
        "blend_alpha": float(weight_cfg.get("blend_alpha", 0.15)),
    }
