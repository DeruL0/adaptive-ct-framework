from __future__ import annotations

from pathlib import Path

import torch

from .geometry import make_parallel_ray_points
from .projection_domain import coefficient_diagnostics, projection_weights


@torch.no_grad()
def extract_coefficient_surface_points(
    model,
    *,
    threshold: float,
    samples_per_leaf: int = 9,
    bisection_steps: int = 12,
) -> torch.Tensor:
    """Find isovalue roots on coefficient-selected leaf axes, without voxelizing."""
    diagnostics = coefficient_diagnostics(model)
    device = model.coefficient_logits.device
    roots = []
    sample_count = max(2, int(samples_per_leaf))
    local_axis = torch.linspace(0.0, 1.0, sample_count, device=device)
    nonconstant_leaf_ids = torch.nonzero(torch.any(model.leaf_degrees > 0, dim=1), as_tuple=False).reshape(-1)
    for leaf_id_tensor in nonconstant_leaf_ids:
        leaf_id = int(leaf_id_tensor.item())
        axis = int(diagnostics.principal_axis[leaf_id].item())
        other_axes = [value for value in range(3) if value != axis]
        level = int(model.leaf_levels[leaf_id].item())
        shape = model.level_shapes[level] if hasattr(model, "level_shapes") else (int(model.level_resolutions[level]),) * 3
        cell = torch.tensor([2.0 / float(value) for value in shape], dtype=torch.float32, device=device)
        coord = model.leaf_coords[leaf_id].to(dtype=torch.float32)
        cross_counts = [max(2, int(model.leaf_degrees[leaf_id, value].item()) + 1) for value in other_axes]
        cross_axes = [
            (torch.arange(count, dtype=torch.float32, device=device) + 0.5) / float(count)
            for count in cross_counts
        ]
        cross_a, cross_b = torch.meshgrid(*cross_axes, indexing="ij")
        cross_values = torch.stack([cross_a.reshape(-1), cross_b.reshape(-1)], dim=1)
        line_count = int(cross_values.shape[0])
        local = torch.full((line_count, sample_count, 3), 0.5, dtype=torch.float32, device=device)
        local[:, :, axis] = local_axis[None, :]
        local[:, :, other_axes[0]] = cross_values[:, 0, None]
        local[:, :, other_axes[1]] = cross_values[:, 1, None]
        points = -1.0 + (coord[None, None, :] + local) * cell
        values = model.forward_mu(points.reshape(-1, 3)).reshape(line_count, sample_count) - float(threshold)
        for line_id in range(line_count):
            line_values = values[line_id]
            exact_mask = torch.abs(line_values) <= 1e-7
            exact_ids = torch.nonzero(exact_mask, as_tuple=False).reshape(-1)
            for exact_id in exact_ids.tolist():
                roots.append(points[line_id, exact_id])
            crossing_mask = (
                (line_values[:-1] * line_values[1:] < 0.0)
                & ~exact_mask[:-1]
                & ~exact_mask[1:]
            )
            crossing_ids = torch.nonzero(crossing_mask, as_tuple=False).reshape(-1)
            for crossing_id in crossing_ids.tolist():
                low = local_axis[crossing_id].clone()
                high = local_axis[crossing_id + 1].clone()
                low_value = line_values[crossing_id].clone()
                fixed_local = local[line_id, 0].clone()
                for _ in range(int(bisection_steps)):
                    mid = 0.5 * (low + high)
                    mid_local = fixed_local.clone()
                    mid_local[axis] = mid
                    mid_point = -1.0 + (coord + mid_local) * cell
                    mid_value = model.forward_mu(mid_point[None, :])[0] - float(threshold)
                    if bool((low_value * mid_value <= 0.0).item()):
                        high = mid
                    else:
                        low = mid
                        low_value = mid_value
                root_local = fixed_local.clone()
                root_local[axis] = 0.5 * (low + high)
                roots.append(-1.0 + (coord + root_local) * cell)

    # Piecewise leaves can meet discontinuously. Add threshold-crossing face
    # samples directly from the two coefficient evaluations around each face.
    level_shapes = model.level_shapes if hasattr(model, "level_shapes") else [(int(value),) * 3 for value in model.level_resolutions]
    resolutions = torch.tensor(level_shapes, dtype=torch.float32, device=device)[model.leaf_levels]
    centres = -1.0 + (model.leaf_coords.to(dtype=torch.float32) + 0.5) * (2.0 / resolutions)
    half_width = torch.reciprocal(resolutions)
    face_centres = []
    inside_points = []
    outside_points = []
    for axis in range(3):
        centre = centres.clone()
        centre[:, axis] += half_width[:, axis]
        inside = centre.clone()
        outside = centre.clone()
        inside[:, axis] -= 1e-5
        outside[:, axis] += 1e-5
        face_centres.append(centre)
        inside_points.append(inside)
        outside_points.append(outside)
    face_centres_tensor = torch.cat(face_centres, dim=0)
    inside_values = model.forward_mu(torch.cat(inside_points, dim=0)) - float(threshold)
    outside_values = model.forward_mu(torch.cat(outside_points, dim=0)) - float(threshold)
    interface_crossing = inside_values * outside_values < 0.0
    if torch.any(interface_crossing):
        roots.extend(face_centres_tensor[interface_crossing])
    if not roots:
        return torch.empty((0, 3), dtype=torch.float32, device=device)
    points = torch.stack(list(roots), dim=0)
    return torch.unique(torch.round(points * 1e6) / 1e6, dim=0)


def estimate_wls_coefficient_uncertainty(
    model,
    split,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    max_rays: int = 64,
    damping: float = 1e-6,
) -> torch.Tensor:
    """Diagonal Gauss-Newton covariance from projection-domain WLS Jacobians."""
    device = model.coefficient_logits.device
    view_count = int(split.angles.shape[0])
    detector_pixels = detector_h * detector_w
    ray_count = max(1, min(int(max_rays), view_count * detector_pixels))
    global_ids = torch.linspace(0, view_count * detector_pixels - 1, ray_count, device=device).round().long()
    view_ids = torch.div(global_ids, detector_pixels, rounding_mode="floor")
    pixel_ids = torch.remainder(global_ids, detector_pixels)
    rows = torch.div(pixel_ids, detector_w, rounding_mode="floor")
    cols = torch.remainder(pixel_ids, detector_w)
    targets = split.projections[view_ids, rows, cols]
    weights = projection_weights(targets)
    hessian_diagonal = torch.zeros_like(model.coefficient_logits)
    for index in range(ray_count):
        ray_batch = make_parallel_ray_points(
            split.angles[view_ids[index : index + 1]],
            rows[index : index + 1],
            cols[index : index + 1],
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            targets=targets[index : index + 1],
            materialize_points=not (
                hasattr(model, "prefer_compact_ray_batch") and model.prefer_compact_ray_batch()
            ),
        )
        prediction = model.integrate_ray_batch(ray_batch)[0]
        gradient = torch.autograd.grad(prediction, model.coefficient_logits, retain_graph=False)[0]
        hessian_diagonal += weights[index] * torch.square(gradient.detach())
    variance = torch.reciprocal(hessian_diagonal + float(damping))
    return torch.sqrt(variance)


def propagate_surface_uncertainty(
    model,
    points: torch.Tensor,
    coefficient_std: torch.Tensor,
    *,
    max_points: int | None = None,
) -> torch.Tensor:
    if max_points is not None:
        points = points[: int(max_points)]
    uncertainties = []
    for point in points:
        value = model.forward_mu(point[None, :])[0]
        gradient = torch.autograd.grad(value, model.coefficient_logits, retain_graph=False)[0]
        uncertainties.append(torch.sqrt(torch.sum(torch.square(gradient * coefficient_std))))
    if not uncertainties:
        return points.new_empty((0,))
    return torch.stack(uncertainties)


def export_surface_artifact(
    model,
    split,
    path: str | Path,
    *,
    threshold: float,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    uncertainty_rays: int = 64,
    uncertainty_surface_points: int = 1024,
) -> dict:
    points = extract_coefficient_surface_points(model, threshold=threshold)
    coefficient_std = estimate_wls_coefficient_uncertainty(
        model,
        split,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        max_rays=uncertainty_rays,
    )
    surface_std = propagate_surface_uncertainty(
        model,
        points,
        coefficient_std,
        max_points=uncertainty_surface_points,
    )
    artifact = {
        "threshold": float(threshold),
        "surface_points": points.detach().cpu(),
        "coefficient_std": coefficient_std.detach().cpu(),
        "surface_std": surface_std.detach().cpu(),
    }
    torch.save(artifact, Path(path))
    return {
        "surface_point_count": int(points.shape[0]),
        "uncertainty_surface_point_count": int(surface_std.shape[0]),
        "coefficient_uncertainty_count": int(coefficient_std.numel()),
    }
