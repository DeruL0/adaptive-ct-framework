from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace
from typing import Iterable, Sequence

import torch

from .bernstein import unique_degree_rows
from .data import ProjectionSplit
from .geometry import (
    ProjectionGradientBatch,
    detector_grid,
    make_parallel_ray_points,
    random_training_rays,
)


@dataclass(frozen=True)
class CoefficientDiagnostics:
    directional_variation: torch.Tensor
    high_order_ratio: torch.Tensor
    interface_jump: torch.Tensor
    principal_axis: torch.Tensor


@dataclass(frozen=True)
class CandidateAction:
    action: str
    leaf_keys: tuple[tuple[int, int, int, int], ...]
    axis: int | None
    diagnostic_score: float


@dataclass(frozen=True)
class CandidateEvaluation:
    action: str
    leaf_keys: tuple[tuple[int, int, int, int], ...]
    axis: int | None
    diagnostic_score: float
    validation_before: float
    validation_after: float
    validation_gain: float
    rate_delta_bytes: int
    score_per_byte: float
    accepted: bool
    reason: str

    def to_dict(self) -> dict:
        value = asdict(self)
        value["leaf_keys"] = [list(key) for key in self.leaf_keys]
        return value


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


def weighted_projection_gradient_loss(
    prediction: torch.Tensor,
    batch: ProjectionGradientBatch,
    epsilon: float = 1e-4,
    *,
    mode: str = "inverse",
    target_power: float = 1.0,
    min_weight: float | None = None,
    max_weight: float | None = None,
    blend_alpha: float = 0.15,
    edge_weight_power: float = 0.0,
    edge_min_weight: float | None = None,
    edge_max_weight: float | None = None,
    magnitude_weight: float = 0.0,
    magnitude_quantile: float = 0.75,
    moment_weight: float = 0.0,
    moment_quantile: float = 0.75,
    endpoint_weight: float = 0.0,
    endpoint_quantile: float = 0.75,
) -> torch.Tensor:
    """Sobolev data fidelity on paired detector rays.

    This compares finite differences of the actual forward projection, rather
    than applying a display-space sharpening term. Dividing by the detector
    stride makes multiple scales share the same derivative units.
    """

    count = int(batch.edge_count)
    if prediction.numel() != 2 * count:
        raise ValueError("projection-gradient prediction must contain two endpoints per edge")
    derivative = (prediction[count:] - prediction[:count]) / batch.strides.to(prediction.dtype)
    residual = derivative - batch.target_derivative
    weights = projection_weights(
        batch.target_midpoint,
        epsilon=epsilon,
        mode=mode,
        target_power=target_power,
        min_weight=min_weight,
        max_weight=max_weight,
        blend_alpha=blend_alpha,
    )
    if float(edge_weight_power) > 0.0:
        magnitude = torch.abs(batch.target_derivative.detach())
        finite = magnitude[torch.isfinite(magnitude)]
        scale = torch.quantile(finite, 0.95).clamp_min(float(epsilon)) if finite.numel() else magnitude.new_tensor(1.0)
        edge_weights = torch.pow(magnitude / scale + float(epsilon), float(edge_weight_power))
        if edge_min_weight is not None or edge_max_weight is not None:
            low = float(edge_min_weight) if edge_min_weight is not None else 0.0
            high = float(edge_max_weight) if edge_max_weight is not None else float("inf")
            edge_weights = edge_weights.clamp(min=low, max=high)
        weights = weights * edge_weights
        weights = weights / weights.mean().clamp_min(float(epsilon))
    signed_loss = torch.mean(weights * torch.square(residual))
    total = signed_loss
    target_magnitude = torch.abs(batch.target_derivative.detach())
    if float(magnitude_weight) > 0.0:
        quantile = float(magnitude_quantile)
        if not 0.0 <= quantile < 1.0:
            raise ValueError("magnitude_quantile must be in [0, 1)")
        informative = target_magnitude >= torch.quantile(target_magnitude, quantile)
        if torch.any(informative):
            magnitude_weights = weights[informative]
            magnitude_weights = magnitude_weights / magnitude_weights.mean().clamp_min(float(epsilon))
            magnitude_residual = torch.abs(derivative[informative]) - target_magnitude[informative]
            magnitude_loss = torch.mean(magnitude_weights * torch.square(magnitude_residual))
            total = total + float(magnitude_weight) * magnitude_loss
    if float(moment_weight) > 0.0:
        quantile = float(moment_quantile)
        if not 0.0 <= quantile < 1.0:
            raise ValueError("moment_quantile must be in [0, 1)")
        informative = target_magnitude >= torch.quantile(target_magnitude, quantile)
        if torch.any(informative):
            moment_weights = weights[informative]
            moment_weights = moment_weights / moment_weights.sum().clamp_min(float(epsilon))
            prediction_moment = torch.sum(moment_weights * torch.abs(derivative[informative]))
            target_moment = torch.sum(moment_weights * target_magnitude[informative])
            total = total + float(moment_weight) * torch.square(prediction_moment - target_moment)
    if float(endpoint_weight) > 0.0:
        quantile = float(endpoint_quantile)
        if not 0.0 <= quantile < 1.0:
            raise ValueError("endpoint_quantile must be in [0, 1)")
        informative = target_magnitude >= torch.quantile(target_magnitude, quantile)
        if torch.any(informative):
            endpoint_weights = weights[informative]
            endpoint_weights = endpoint_weights / endpoint_weights.mean().clamp_min(float(epsilon))
            target_a = batch.ray_batch.target[:count][informative]
            target_b = batch.ray_batch.target[count:][informative]
            residual_a = prediction[:count][informative] - target_a
            residual_b = prediction[count:][informative] - target_b
            endpoint_loss = 0.5 * torch.mean(
                endpoint_weights * (torch.square(residual_a) + torch.square(residual_b))
            )
            total = total + float(endpoint_weight) * endpoint_loss
    return total


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


def _group_coefficient_blocks(model, leaf_ids: torch.Tensor, degree: tuple[int, int, int]) -> torch.Tensor:
    count = int(math.prod(value + 1 for value in degree))
    local_ids = torch.arange(count, device=model.coefficient_logits.device, dtype=torch.long)
    coefficient_ids = model.coefficient_offsets[leaf_ids, None] + local_ids[None, :]
    return model.coefficients()[coefficient_ids].reshape(leaf_ids.shape[0], *(value + 1 for value in degree))


@torch.no_grad()
def coefficient_diagnostics(model) -> CoefficientDiagnostics:
    """Compute all adaptive direction signals without constructing a volume."""
    device = model.coefficient_logits.device
    leaf_count = int(model.leaf_levels.shape[0])
    variation = torch.zeros((leaf_count, 3), dtype=torch.float32, device=device)
    ratio = torch.zeros_like(variation)
    face_means = torch.zeros((leaf_count, 3, 2), dtype=torch.float32, device=device)

    for degree_tensor in unique_degree_rows(model.leaf_degrees):
        degree = tuple(int(value) for value in degree_tensor.tolist())
        mask = torch.all(model.leaf_degrees == degree_tensor, dim=1)
        leaf_ids = torch.nonzero(mask, as_tuple=False).reshape(-1)
        coefficients = _group_coefficient_blocks(model, leaf_ids, degree)
        level_shapes = (
            model.level_shapes
            if hasattr(model, "level_shapes")
            else [(int(value),) * 3 for value in model.level_resolutions]
        )
        resolutions = torch.tensor(
            level_shapes,
            dtype=coefficients.dtype,
            device=device,
        )[model.leaf_levels[leaf_ids]]
        for axis in range(3):
            low_face = coefficients.select(axis + 1, 0).reshape(leaf_ids.shape[0], -1).mean(dim=1)
            high_face = coefficients.select(axis + 1, degree[axis]).reshape(leaf_ids.shape[0], -1).mean(dim=1)
            face_means[leaf_ids, axis, 0] = low_face
            face_means[leaf_ids, axis, 1] = high_face
            if degree[axis] <= 0:
                continue
            differences = torch.diff(coefficients, dim=axis + 1).abs()
            flat = differences.reshape(leaf_ids.shape[0], -1)
            # Cell width is 2 / resolution in normalized object coordinates.
            variation[leaf_ids, axis] = flat.amax(dim=1) * float(degree[axis]) * resolutions[:, axis] * 0.5
            low_difference = differences.select(axis + 1, 0).reshape(leaf_ids.shape[0], -1).amax(dim=1)
            high_difference = differences.select(axis + 1, degree[axis] - 1).reshape(leaf_ids.shape[0], -1).amax(dim=1)
            ratio[leaf_ids, axis] = high_difference / low_difference.clamp_min(1e-12)

    level_shapes = (
        model.level_shapes
        if hasattr(model, "level_shapes")
        else [(int(value),) * 3 for value in model.level_resolutions]
    )
    resolutions = torch.tensor(level_shapes, dtype=torch.float32, device=device)[model.leaf_levels]
    centres = -1.0 + (model.leaf_coords.to(dtype=torch.float32) + 0.5) * (2.0 / resolutions)
    half_width = torch.reciprocal(resolutions)
    jumps = torch.zeros_like(variation)
    epsilon = 1e-5
    for axis in range(3):
        for side, sign in ((0, -1.0), (1, 1.0)):
            query = centres.clone()
            query[:, axis] += sign * (half_width[:, axis] + epsilon)
            neighbor_ids = model.resolve_leaf_ids(query).reshape(-1)
            valid = neighbor_ids >= 0
            if not torch.any(valid):
                continue
            ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
            opposite_side = 1 - side
            difference = torch.abs(
                face_means[ids, axis, side] - face_means[neighbor_ids[ids], axis, opposite_side]
            )
            jumps[ids, axis] = torch.maximum(jumps[ids, axis], difference)

    # Constant leaves have no internal differences. Neighbor variation is the
    # projection-trained coefficient signal that seeds their first p/h action.
    directional = torch.maximum(variation, jumps * resolutions * 0.5)
    return CoefficientDiagnostics(
        directional_variation=directional,
        high_order_ratio=ratio,
        interface_jump=jumps,
        principal_axis=torch.argmax(directional, dim=1),
    )


@torch.no_grad()
def propose_forward_actions(
    model,
    diagnostics: CoefficientDiagnostics,
    *,
    variation_threshold: float,
    material_jump_threshold: float,
    p_ratio_threshold: float,
    max_candidates: int,
    allow_p: bool = True,
    allow_h: bool = True,
    isotropic_p: bool = False,
) -> list[CandidateAction]:
    candidates: list[CandidateAction] = []
    for leaf_id in range(int(model.leaf_levels.shape[0])):
        key = model.leaf_key(leaf_id)
        level = int(model.leaf_levels[leaf_id].item())
        jumps = diagnostics.interface_jump[leaf_id]
        strongest_jump, jump_axis_tensor = torch.max(jumps, dim=0)
        jump_axis = int(jump_axis_tensor.item())
        if (
            allow_h
            and
            level + 1 < len(model.level_resolutions)
            and float(strongest_jump.item()) > float(material_jump_threshold)
        ):
            candidates.append(
                CandidateAction("h_split", (key,), jump_axis, float(strongest_jump.item()))
            )
            continue

        if allow_p and isotropic_p:
            scores = []
            for axis in range(3):
                degree = int(model.leaf_degrees[leaf_id, axis].item())
                signal = float(diagnostics.directional_variation[leaf_id, axis].item())
                high_order_ratio = float(diagnostics.high_order_ratio[leaf_id, axis].item())
                if degree >= int(model.max_degree[axis]) or signal <= float(variation_threshold):
                    continue
                if degree > 0 and high_order_ratio < float(p_ratio_threshold):
                    continue
                scores.append(signal * (high_order_ratio if degree > 0 else 1.0))
            if scores:
                candidates.append(CandidateAction("p_elevate_isotropic", (key,), None, max(scores)))
            continue

        for axis in range(3):
            if not allow_p:
                break
            degree = int(model.leaf_degrees[leaf_id, axis].item())
            if degree >= int(model.max_degree[axis]):
                continue
            signal = float(diagnostics.directional_variation[leaf_id, axis].item())
            high_order_ratio = float(diagnostics.high_order_ratio[leaf_id, axis].item())
            if signal <= float(variation_threshold):
                continue
            if degree > 0 and high_order_ratio < float(p_ratio_threshold):
                continue
            candidates.append(
                CandidateAction(
                    "p_elevate",
                    (key,),
                    axis,
                    signal * (high_order_ratio if degree > 0 else 1.0),
                )
            )
    candidates.sort(key=lambda candidate: candidate.diagnostic_score, reverse=True)
    return candidates[: max(0, int(max_candidates))]


@torch.no_grad()
def propose_reverse_actions(model, *, max_candidates: int) -> list[CandidateAction]:
    candidates: list[CandidateAction] = []
    # Prefer reducing the weakest high-order coefficient differences first.
    diagnostics = coefficient_diagnostics(model)
    for leaf_id in range(int(model.leaf_levels.shape[0])):
        key = model.leaf_key(leaf_id)
        for axis in range(3):
            if int(model.leaf_degrees[leaf_id, axis].item()) > 0:
                score = float(diagnostics.directional_variation[leaf_id, axis].item())
                candidates.append(CandidateAction("p_reduce", (key,), axis, score))

    sibling_groups: dict[tuple[int, int, int, int], list[tuple[int, int, int, int]]] = {}
    for leaf_id in range(int(model.leaf_levels.shape[0])):
        level, x, y, z = model.leaf_key(leaf_id)
        if level <= 0:
            continue
        parent = (level - 1, x // 2, y // 2, z // 2)
        sibling_groups.setdefault(parent, []).append((level, x, y, z))
    for siblings in sibling_groups.values():
        if len(siblings) == 8:
            candidates.append(CandidateAction("h_merge", tuple(sorted(siblings)), None, 0.0))

    candidates.sort(key=lambda candidate: candidate.diagnostic_score)
    return candidates[: max(0, int(max_candidates))]


def _apply_candidate(model, candidate: CandidateAction):
    if candidate.action == "p_elevate":
        leaf_id = model.find_leaf(candidate.leaf_keys[0])
        if leaf_id is None:
            return None
        return model.elevate_degree(leaf_id, int(candidate.axis))
    if candidate.action == "p_elevate_isotropic":
        key = candidate.leaf_keys[0]
        old_count = int(model.coefficient_logits.numel())
        changed = False
        for axis in range(3):
            leaf_id = model.find_leaf(key)
            if leaf_id is None:
                return None
            if int(model.leaf_degrees[leaf_id, axis].item()) < int(model.max_degree[axis]):
                model.elevate_degree(leaf_id, axis)
                changed = True
        if not changed:
            return None
        return SimpleNamespace(
            old_leaf_keys=(key,),
            new_leaf_keys=(key,),
            old_coefficient_count=old_count,
            new_coefficient_count=int(model.coefficient_logits.numel()),
        )
    if candidate.action == "p_reduce":
        leaf_id = model.find_leaf(candidate.leaf_keys[0])
        if leaf_id is None:
            return None
        return model.reduce_degree(leaf_id, int(candidate.axis))
    if candidate.action == "h_split":
        leaf_id = model.find_leaf(candidate.leaf_keys[0])
        if leaf_id is None:
            return None
        return model.split_leaf(leaf_id)
    if candidate.action == "h_merge":
        leaf_ids = [model.find_leaf(key) for key in candidate.leaf_keys]
        if any(leaf_id is None for leaf_id in leaf_ids):
            return None
        return model.merge_siblings([int(leaf_id) for leaf_id in leaf_ids])
    raise ValueError(f"Unknown adaptive action {candidate.action!r}.")


def _selected_detector_rays(
    detector_h: int,
    detector_w: int,
    *,
    device: torch.device,
    max_rays: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = detector_grid(detector_h, detector_w, device)
    if max_rays is None or int(max_rays) >= rows.numel():
        return rows, cols
    count = max(1, int(max_rays))
    ids = torch.linspace(0, rows.numel() - 1, count, device=device).round().to(dtype=torch.long)
    return rows[ids], cols[ids]


@torch.no_grad()
def region_validation_energy(
    model,
    split: ProjectionSplit,
    leaf_keys: Iterable[Sequence[int]],
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    ray_chunk: int,
    max_views: int | None = None,
    max_rays_per_view: int | None = None,
) -> float:
    region_leaf_ids = [model.find_leaf(key) for key in leaf_keys]
    region_leaf_ids = [int(leaf_id) for leaf_id in region_leaf_ids if leaf_id is not None]
    if not region_leaf_ids:
        return float("inf")
    device = split.projections.device
    region_tensor = torch.tensor(region_leaf_ids, dtype=torch.long, device=device)
    view_count = int(split.angles.shape[0])
    if max_views is not None:
        view_count = min(view_count, int(max_views))
    rows_all, cols_all = _selected_detector_rays(
        detector_h,
        detector_w,
        device=device,
        max_rays=max_rays_per_view,
    )
    total_energy = 0.0
    total_rays = 0
    for view_id in range(view_count):
        angle = split.angles[view_id]
        for start in range(0, rows_all.numel(), int(ray_chunk)):
            rows = rows_all[start : start + int(ray_chunk)]
            cols = cols_all[start : start + int(ray_chunk)]
            targets = split.projections[view_id, rows, cols]
            ray_batch = make_parallel_ray_points(
                angle.expand(rows.shape[0]),
                rows,
                cols,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
            targets=targets,
            materialize_points=not (
                hasattr(model, "prefer_compact_ray_batch") and model.prefer_compact_ray_batch()
            ),
            )
            ray_ids, leaf_ids, contributions = model.ray_cell_contributions(ray_batch)
            prediction = contributions.new_zeros((ray_batch.num_rays,))
            prediction.scatter_add_(0, ray_ids, contributions)
            region = contributions.new_zeros((ray_batch.num_rays,))
            selected = torch.isin(leaf_ids, region_tensor)
            if torch.any(selected):
                region.scatter_add_(0, ray_ids[selected], contributions[selected])
            residual = prediction - targets
            attributed = residual * region / prediction.detach().clamp_min(1e-8)
            weights = projection_weights(targets)
            total_energy += float(torch.sum(weights * torch.square(attributed)).item())
            total_rays += int(targets.numel())
    return total_energy / max(total_rays, 1)


def _tune_affected_coefficients(
    model,
    training_split: ProjectionSplit,
    affected_leaf_keys: Iterable[Sequence[int]],
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    steps: int,
    batch_rays: int,
    learning_rate: float,
    weight_epsilon: float,
    weight_kwargs: dict | None = None,
    new_dof_axis: int | None = None,
) -> None:
    if int(steps) <= 0:
        return
    affected_leaf_ids = [model.find_leaf(key) for key in affected_leaf_keys]
    affected_leaf_ids = [int(leaf_id) for leaf_id in affected_leaf_ids if leaf_id is not None]
    if not affected_leaf_ids:
        return
    ids = torch.tensor(affected_leaf_ids, dtype=torch.long, device=model.coefficient_logits.device)
    trainable_mask = torch.isin(model.coefficient_leaf_ids, ids)
    optimizer = (
        torch.optim.SGD([model.coefficient_logits], lr=float(learning_rate))
        if new_dof_axis is not None
        else torch.optim.Adam([model.coefficient_logits], lr=float(learning_rate))
    )
    for _ in range(int(steps)):
        ray_batch = random_training_rays(
            training_split,
            batch_rays=int(batch_rays),
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            materialize_points=not (
                hasattr(model, "prefer_compact_ray_batch") and model.prefer_compact_ray_batch()
            ),
        )
        prediction = model.integrate_ray_batch(ray_batch)
        loss = weighted_projection_loss(
            prediction,
            ray_batch.target,
            **(weight_kwargs or {"epsilon": weight_epsilon}),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if model.coefficient_logits.grad is not None:
            model.coefficient_logits.grad[~trainable_mask] = 0.0
            if new_dof_axis is not None:
                axis = int(new_dof_axis)
                for leaf_id in affected_leaf_ids:
                    degree = int(model.leaf_degrees[leaf_id, axis].item())
                    if degree <= 0:
                        continue
                    old_degree = degree - 1
                    elevation = model.coefficient_logits.new_zeros((degree + 1, old_degree + 1))
                    elevation[0, 0] = 1.0
                    elevation[-1, -1] = 1.0
                    for index in range(1, degree):
                        alpha = float(index) / float(degree)
                        elevation[index, index - 1] = alpha
                        elevation[index, index] = 1.0 - alpha
                    # The one-dimensional orthogonal complement is the newly
                    # introduced Bernstein degree of freedom.
                    null_direction = torch.linalg.svd(elevation, full_matrices=True).U[:, -1]
                    start = int(model.coefficient_offsets[leaf_id].item())
                    stop = int(model.coefficient_offsets[leaf_id + 1].item())
                    shape = tuple(int(value) + 1 for value in model.leaf_degrees[leaf_id].tolist())
                    gradient = model.coefficient_logits.grad[start:stop].reshape(shape).movedim(axis, 0)
                    logits = model.coefficient_logits.detach()[start:stop].reshape(shape).movedim(axis, 0)
                    physical_jacobian = torch.sigmoid(logits + model.attenuation_shift).clamp_min(1e-8)
                    tangent = null_direction.reshape((-1,) + (1,) * (gradient.ndim - 1)) / physical_jacobian
                    flat_gradient = gradient.reshape(degree + 1, -1)
                    flat_tangent = tangent.reshape(degree + 1, -1)
                    scale = torch.sum(flat_gradient * flat_tangent, dim=0) / torch.sum(
                        flat_tangent * flat_tangent, dim=0
                    ).clamp_min(1e-12)
                    projected = (flat_tangent * scale[None, :]).reshape_as(gradient).movedim(0, axis)
                    model.coefficient_logits.grad[start:stop] = projected.reshape(-1)
        optimizer.step()


def _tune_all_coefficients(
    model,
    training_split: ProjectionSplit,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    steps: int,
    batch_rays: int,
    learning_rate: float,
    weight_epsilon: float,
) -> None:
    if int(steps) <= 0:
        return
    optimizer = torch.optim.Adam([model.coefficient_logits], lr=float(learning_rate))
    for _ in range(int(steps)):
        ray_batch = random_training_rays(
            training_split,
            batch_rays=int(batch_rays),
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            materialize_points=not (
                hasattr(model, "prefer_compact_ray_batch") and model.prefer_compact_ray_batch()
            ),
        )
        prediction = model.integrate_ray_batch(ray_batch)
        loss = weighted_projection_loss(prediction, ray_batch.target, epsilon=weight_epsilon)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate_candidate(
    model,
    candidate: CandidateAction,
    training_split: ProjectionSplit,
    validation_split: ProjectionSplit,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    tune_steps: int,
    tune_batch_rays: int,
    tune_learning_rate: float,
    validation_ray_chunk: int,
    validation_views: int | None,
    validation_rays_per_view: int | None,
    rate_lambda: float,
    merge_lambda: float,
    weight_epsilon: float,
    selection_mode: str = "rate_distortion",
    fixed_gain_threshold: float = 0.0,
) -> tuple[object, CandidateEvaluation]:
    before_energy = region_validation_energy(
        model,
        validation_split,
        candidate.leaf_keys,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=validation_ray_chunk,
        max_views=validation_views,
        max_rays_per_view=validation_rays_per_view,
    )
    trial = copy.deepcopy(model)
    before_bytes = int(trial.stats().model_bytes)
    change = _apply_candidate(trial, candidate)
    if change is None:
        evaluation = CandidateEvaluation(
            action=candidate.action,
            leaf_keys=candidate.leaf_keys,
            axis=candidate.axis,
            diagnostic_score=candidate.diagnostic_score,
            validation_before=before_energy,
            validation_after=before_energy,
            validation_gain=0.0,
            rate_delta_bytes=0,
            score_per_byte=0.0,
            accepted=False,
            reason="candidate topology no longer exists",
        )
        return model, evaluation

    is_forward = candidate.action in {"p_elevate", "p_elevate_isotropic", "h_split"}
    if is_forward:
        _tune_affected_coefficients(
            trial,
            training_split,
            change.new_leaf_keys,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            steps=tune_steps,
            batch_rays=tune_batch_rays,
            learning_rate=tune_learning_rate,
            weight_epsilon=weight_epsilon,
            new_dof_axis=int(candidate.axis) if candidate.action == "p_elevate" else None,
        )
    after_energy = region_validation_energy(
        trial,
        validation_split,
        change.new_leaf_keys,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=validation_ray_chunk,
        max_views=validation_views,
        max_rays_per_view=validation_rays_per_view,
    )
    after_bytes = int(trial.stats().model_bytes)
    rate_delta = after_bytes - before_bytes
    validation_gain = before_energy - after_energy

    if is_forward:
        score = validation_gain / max(rate_delta, 1)
        if str(selection_mode).lower() == "fixed_gain":
            accepted = validation_gain > float(fixed_gain_threshold) and rate_delta > 0
        elif str(selection_mode).lower() == "rate_distortion":
            accepted = validation_gain > 0.0 and rate_delta > 0 and score > float(rate_lambda)
        else:
            raise ValueError("selection_mode must be 'rate_distortion' or 'fixed_gain'.")
        reason = (
            "held-out gain passes selection threshold"
            if accepted
            else "no positive held-out gain" if validation_gain <= 0.0 else "gain per byte below rate threshold"
        )
    else:
        bytes_saved = -rate_delta
        penalty_per_byte = max(-validation_gain, 0.0) / max(bytes_saved, 1)
        score = -penalty_per_byte
        accepted = bytes_saved > 0 and penalty_per_byte < float(merge_lambda)
        reason = "validation penalty per saved byte is below merge threshold" if accepted else "reverse action costs too much validation error"

    evaluation = CandidateEvaluation(
        action=candidate.action,
        leaf_keys=candidate.leaf_keys,
        axis=candidate.axis,
        diagnostic_score=candidate.diagnostic_score,
        validation_before=before_energy,
        validation_after=after_energy,
        validation_gain=validation_gain,
        rate_delta_bytes=rate_delta,
        score_per_byte=score,
        accepted=accepted,
        reason=reason,
    )
    return (trial if accepted else model), evaluation


def adaptive_projection_round(
    model,
    split: ProjectionSplit,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    config: dict,
    seed: int,
) -> tuple[object, dict]:
    if bool(config.get("held_out_validation", True)):
        training_split, validation_split = split_projection_views(
            split,
            validation_fraction=float(config.get("validation_fraction", 0.2)),
            seed=int(seed),
        )
    else:
        training_split = split
        validation_split = split
    _tune_all_coefficients(
        model,
        training_split,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        steps=int(config.get("pre_tune_steps", 5)),
        batch_rays=int(config.get("pre_tune_batch_rays", config.get("tune_batch_rays", 512))),
        learning_rate=float(config.get("pre_tune_learning_rate", config.get("tune_learning_rate", 5e-3))),
        weight_epsilon=float(config.get("weight_epsilon", 1e-4)),
    )
    diagnostics = coefficient_diagnostics(model)
    candidates = propose_forward_actions(
        model,
        diagnostics,
        variation_threshold=float(config.get("variation_threshold", 1e-3)),
        material_jump_threshold=float(config.get("material_jump_threshold", 0.05)),
        p_ratio_threshold=float(config.get("p_ratio_threshold", 0.5)),
        max_candidates=int(config.get("max_candidates", 8)),
        allow_p=bool(config.get("allow_p", True)),
        allow_h=bool(config.get("allow_h", True)),
        isotropic_p=bool(config.get("isotropic_p", False)),
    )
    evaluations: list[CandidateEvaluation] = []
    kwargs = {
        "detector_h": detector_h,
        "detector_w": detector_w,
        "samples_per_ray": samples_per_ray,
        "tune_steps": int(config.get("tune_steps", 20)),
        "tune_batch_rays": int(config.get("tune_batch_rays", 512)),
        "tune_learning_rate": float(config.get("tune_learning_rate", 5e-3)),
        "validation_ray_chunk": int(config.get("validation_ray_chunk", 512)),
        "validation_views": config.get("validation_views"),
        "validation_rays_per_view": config.get("validation_rays_per_view"),
        "rate_lambda": float(config.get("rate_lambda", 0.0)),
        "merge_lambda": float(config.get("merge_lambda", 0.0)),
        "weight_epsilon": float(config.get("weight_epsilon", 1e-4)),
        "selection_mode": str(config.get("selection_mode", "rate_distortion")),
        "fixed_gain_threshold": float(config.get("fixed_gain_threshold", 0.0)),
    }
    for candidate in candidates:
        model, evaluation = evaluate_candidate(
            model,
            candidate,
            training_split,
            validation_split,
            **kwargs,
        )
        evaluations.append(evaluation)

    reverse_candidates = propose_reverse_actions(
        model,
        max_candidates=int(config.get("max_reverse_candidates", 4)),
    )
    reverse_kwargs = dict(kwargs)
    reverse_kwargs["tune_steps"] = 0
    for candidate in reverse_candidates:
        model, evaluation = evaluate_candidate(
            model,
            candidate,
            training_split,
            validation_split,
            **reverse_kwargs,
        )
        evaluations.append(evaluation)

    report = {
        "seed": int(seed),
        "held_out_validation": bool(config.get("held_out_validation", True)),
        "selection_mode": str(config.get("selection_mode", "rate_distortion")),
        "training_views": int(training_split.angles.shape[0]),
        "validation_views": int(validation_split.angles.shape[0]),
        "candidate_count": len(candidates),
        "reverse_candidate_count": len(reverse_candidates),
        "accepted_count": sum(int(evaluation.accepted) for evaluation in evaluations),
        "directional_variation_mean": diagnostics.directional_variation.mean(dim=0).detach().cpu().tolist(),
        "interface_jump_mean": diagnostics.interface_jump.mean(dim=0).detach().cpu().tolist(),
        "evaluations": [evaluation.to_dict() for evaluation in evaluations],
        "model_bytes": int(model.stats().model_bytes),
        "coefficient_count": int(model.stats().parameter_count),
    }
    return model, report
