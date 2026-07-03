from __future__ import annotations

import torch


def tv3(volume: torch.Tensor) -> torch.Tensor:
    dx = torch.mean(torch.abs(volume[1:, :, :] - volume[:-1, :, :]))
    dy = torch.mean(torch.abs(volume[:, 1:, :] - volume[:, :-1, :]))
    dz = torch.mean(torch.abs(volume[:, :, 1:] - volume[:, :, :-1]))
    return dx + dy + dz


def _evaluate_leaf_polynomials(
    model,
    leaf_ids: torch.Tensor,
    points: torch.Tensor,
    *,
    coefficients: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate explicitly selected leaf polynomials at world-space points."""
    from .bernstein import bernstein_basis, unique_degree_rows

    original_shape = points.shape[:-1]
    flat_points = points.reshape(-1, 3)
    repeats = flat_points.shape[0] // max(int(leaf_ids.numel()), 1)
    flat_leaf_ids = leaf_ids[:, None].expand(-1, repeats).reshape(-1)
    values = flat_points.new_zeros((flat_points.shape[0],), dtype=torch.float32)
    degrees = model.leaf_degrees[flat_leaf_ids]
    coefficients = model.coefficients() if coefficients is None else coefficients
    for degree_tensor in unique_degree_rows(degrees):
        degree = tuple(int(value) for value in degree_tensor.tolist())
        mask = torch.all(degrees == degree_tensor, dim=1)
        point_ids = torch.nonzero(mask, as_tuple=False).reshape(-1)
        selected_leaf_ids = flat_leaf_ids[point_ids]
        level = model.leaf_levels[selected_leaf_ids]
        levels = model.level_shapes if hasattr(model, "level_shapes") else [(int(value),) * 3 for value in model.level_resolutions]
        resolutions = torch.tensor(
            levels,
            dtype=flat_points.dtype,
            device=flat_points.device,
        )[level]
        coords = model.leaf_coords[selected_leaf_ids].to(dtype=flat_points.dtype)
        local = (flat_points[point_ids] + 1.0) * 0.5 * resolutions - coords
        bx = bernstein_basis(degree[0], local[:, 0])
        by = bernstein_basis(degree[1], local[:, 1])
        bz = bernstein_basis(degree[2], local[:, 2])
        basis = torch.einsum("ni,nj,nk->nijk", bx, by, bz).reshape(point_ids.numel(), -1)
        offsets = torch.arange(basis.shape[1], dtype=torch.long, device=points.device)
        coefficient_ids = model.coefficient_offsets[selected_leaf_ids, None] + offsets[None, :]
        values[point_ids] = torch.sum(coefficients[coefficient_ids] * basis, dim=1)
    return values.reshape(original_shape)


@torch.no_grad()
def _continuity_pairs(
    model,
    *,
    max_pairs: int = 131072,
    include_cross_level: bool = True,
    face_quadrature_order: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = model.coefficient_logits.device
    dtype = model.coefficient_logits.dtype
    key = (
        int(getattr(model, "_topology_version", 0)),
        int(max_pairs),
        bool(include_cross_level),
        int(face_quadrature_order),
        str(device),
    )
    cache = getattr(model, "_continuity_pair_cache", {})
    if key in cache:
        return cache[key]
    source_chunks = []
    neighbor_chunks = []
    point_chunks = []
    area_chunks = []
    level_shapes = model.level_shapes if hasattr(model, "level_shapes") else [(int(value),) * 3 for value in model.level_resolutions]
    bucket_count = max(1, len(level_shapes) * 3 * 2)
    per_bucket = max(1, int(max_pairs) // bucket_count)
    order = max(1, int(face_quadrature_order))
    if order == 1:
        nodes = torch.tensor([0.5], dtype=dtype, device=device)
    else:
        import numpy as np

        nodes_np, _ = np.polynomial.legendre.leggauss(order)
        nodes = torch.tensor(0.5 * (nodes_np + 1.0), dtype=dtype, device=device)
    uu, vv = torch.meshgrid(nodes, nodes, indexing="ij")
    face_uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)

    for level in range(len(level_shapes)):
        level_ids = torch.nonzero(model.leaf_levels == level, as_tuple=False).reshape(-1)
        if level_ids.numel() == 0:
            continue
        if level_ids.numel() > per_bucket:
            take = torch.linspace(0, level_ids.numel() - 1, per_bucket, device=device).round().long()
            level_ids = level_ids[take]
        cell = torch.tensor(
            [2.0 / float(value) for value in level_shapes[level]],
            dtype=dtype,
            device=device,
        )
        coords = model.leaf_coords[level_ids].to(dtype=dtype)
        for axis in range(3):
            tangential = [value for value in range(3) if value != axis]
            for side in (0, 1):
                face_points = torch.empty(
                    (level_ids.numel(), face_uv.shape[0], 3), dtype=dtype, device=device
                )
                face_points[..., axis] = -1.0 + (coords[:, axis, None] + float(side)) * cell[axis]
                face_points[..., tangential[0]] = -1.0 + (
                    coords[:, tangential[0], None] + face_uv[None, :, 0]
                ) * cell[tangential[0]]
                face_points[..., tangential[1]] = -1.0 + (
                    coords[:, tangential[1], None] + face_uv[None, :, 1]
                ) * cell[tangential[1]]
                probe = face_points[:, 0].clone()
                probe[:, axis] += (-1.0 if side == 0 else 1.0) * cell[axis] * 1.0e-3
                inside = (probe[:, axis] > -1.0) & (probe[:, axis] < 1.0)
                if not torch.any(inside):
                    continue
                source_ids = level_ids[inside]
                neighbor_ids = model.resolve_leaf_ids(probe[inside]).reshape(-1)
                valid = neighbor_ids >= 0
                if not torch.any(valid):
                    continue
                source_ids = source_ids[valid]
                neighbor_ids = neighbor_ids[valid]
                selected_points = face_points[inside][valid]
                neighbor_levels = model.leaf_levels[neighbor_ids]
                if not include_cross_level:
                    valid_level = neighbor_levels == level
                else:
                    # Same-level pairs are emitted only from the high side.
                    # Cross-level pairs are emitted from the finer leaf so its
                    # face is exactly the shared sub-face.
                    valid_level = (neighbor_levels < level) | (
                        (neighbor_levels == level) & (side == 1)
                    )
                if not torch.any(valid_level):
                    continue
                source_ids = source_ids[valid_level]
                neighbor_ids = neighbor_ids[valid_level]
                selected_points = selected_points[valid_level]
                source_chunks.append(source_ids)
                neighbor_chunks.append(neighbor_ids)
                point_chunks.append(selected_points)
                face_area = cell[tangential[0]] * cell[tangential[1]]
                area_chunks.append(torch.full((source_ids.numel(),), face_area, dtype=dtype, device=device))
    if source_chunks:
        result = (
            torch.cat(source_chunks).contiguous(),
            torch.cat(neighbor_chunks).contiguous(),
            torch.cat(point_chunks).contiguous(),
            torch.cat(area_chunks).contiguous(),
        )
    else:
        result = (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0, order * order, 3), dtype=dtype, device=device),
            torch.empty((0,), dtype=dtype, device=device),
        )
    cache[key] = result
    model._continuity_pair_cache = cache
    return result


def coefficient_face_continuity_loss(
    model,
    *,
    max_pairs: int = 131072,
    huber_delta: float = 0.02,
    include_cross_level: bool = True,
    face_quadrature_order: int = 2,
) -> torch.Tensor:
    """Area-normalized edge-preserving continuity over cached leaf interfaces."""
    if not hasattr(model, "leaf_levels") or not hasattr(model, "resolve_leaf_ids"):
        return next(model.parameters()).new_zeros(())
    source_ids, neighbor_ids, points, areas = _continuity_pairs(
        model,
        max_pairs=max_pairs,
        include_cross_level=include_cross_level,
        face_quadrature_order=face_quadrature_order,
    )
    if source_ids.numel() == 0:
        return model.coefficient_logits.new_zeros(())
    coefficients = model.coefficients()
    left = _evaluate_leaf_polynomials(model, source_ids, points, coefficients=coefficients)
    right = _evaluate_leaf_polynomials(model, neighbor_ids, points, coefficients=coefficients)
    diff = left - right
    delta = float(huber_delta)
    abs_diff = torch.abs(diff)
    penalty = torch.where(
        abs_diff <= delta,
        0.5 * torch.square(diff),
        delta * (abs_diff - 0.5 * delta),
    ).mean(dim=1)
    return torch.sum(areas * penalty) / torch.sum(areas).clamp_min(1.0e-12)
