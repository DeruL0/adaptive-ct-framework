from __future__ import annotations

from typing import Sequence

import torch

from .data import ProjectionSplit


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

def split_projection_views(
    split: ProjectionSplit,
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[ProjectionSplit, ProjectionSplit]:
    view_count = int(split.angles.shape[0])
    if view_count < 2:
        raise ValueError("Held-out projection validation requires at least two views.")
    validation_count = max(1, min(view_count - 1, int(round(view_count * float(validation_fraction)))))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(view_count, generator=generator).tolist()
    validation_ids = permutation[:validation_count]
    training_ids = permutation[validation_count:]

    def subset(ids: Sequence[int]) -> ProjectionSplit:
        tensor_ids = torch.tensor(ids, dtype=torch.long, device=split.angles.device)
        return ProjectionSplit(
            angles=split.angles[tensor_ids],
            projections=split.projections[tensor_ids],
            paths=[split.paths[index] for index in ids],
        )

    return subset(training_ids), subset(validation_ids)
