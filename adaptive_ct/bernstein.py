from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .backend import (
    bernstein_octree_integrate,
    bernstein_octree_ray_segments,
    has_bernstein_native,
)


@dataclass(frozen=True)
class BernsteinStats:
    parameter_count: int
    model_bytes: int
    l0_cells: int
    l0_active: int
    l1_active: int
    l2_active: int
    l3_active: int
    active_by_level: tuple[int, ...]
    leaf_cells: int
    max_depth: int
    representation: str
    coefficient_count: int
    max_degree: tuple[int, int, int]


@dataclass(frozen=True)
class TopologyChange:
    action: str
    old_leaf_keys: tuple[tuple[int, int, int, int], ...]
    new_leaf_keys: tuple[tuple[int, int, int, int], ...]
    old_coefficient_count: int
    new_coefficient_count: int


def _shape3(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value),) * 3
    shape = tuple(int(component) for component in value)
    if len(shape) != 3:
        raise ValueError(f"Expected an integer or a 3-value shape, got {value!r}.")
    return shape


def _all_coords(
    resolution: int | Sequence[int],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    shape = _shape3(resolution)
    axes = [torch.arange(value, dtype=torch.long, device=device) for value in shape]
    xx, yy, zz = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(torch.finfo(value.dtype).tiny)
    return value + torch.log(-torch.expm1(-value))


def bernstein_basis(degree: int, coordinate: torch.Tensor) -> torch.Tensor:
    """Evaluate all Bernstein basis functions of one degree on [0, 1]."""
    degree = int(degree)
    x = coordinate.clamp(0.0, 1.0)
    if degree == 0:
        return torch.ones((*x.shape, 1), dtype=x.dtype, device=x.device)
    values = []
    one_minus_x = 1.0 - x
    for index in range(degree + 1):
        values.append(
            float(math.comb(degree, index))
            * torch.pow(x, index)
            * torch.pow(one_minus_x, degree - index)
        )
    return torch.stack(values, dim=-1)


def _degree_elevate_tensor(coefficients: torch.Tensor, axis: int) -> torch.Tensor:
    axis = int(axis)
    source = coefficients.movedim(axis, 0)
    degree = source.shape[0] - 1
    elevated = source.new_empty((degree + 2, *source.shape[1:]))
    elevated[0] = source[0]
    elevated[-1] = source[-1]
    if degree > 0:
        indices = torch.arange(1, degree + 1, dtype=source.dtype, device=source.device)
        alpha = indices / float(degree + 1)
        alpha = alpha.reshape((-1,) + (1,) * (source.ndim - 1))
        elevated[1:-1] = alpha * source[:-1] + (1.0 - alpha) * source[1:]
    return elevated.movedim(0, axis)


def _subdivide_axis(coefficients: torch.Tensor, axis: int, location: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    axis = int(axis)
    source = coefficients.movedim(axis, 0)
    degree = source.shape[0] - 1
    triangle = [source]
    t = float(location)
    for _ in range(degree):
        previous = triangle[-1]
        triangle.append((1.0 - t) * previous[:-1] + t * previous[1:])
    left = torch.stack([triangle[row][0] for row in range(degree + 1)], dim=0)
    right = torch.stack([triangle[degree - row][row] for row in range(degree + 1)], dim=0)
    return left.movedim(0, axis), right.movedim(0, axis)


def subdivide_bernstein_tensor(coefficients: torch.Tensor) -> list[torch.Tensor]:
    """Split a trivariate Bernstein block into eight exact octants."""
    blocks = [coefficients]
    for axis in range(3):
        next_blocks = []
        for block in blocks:
            next_blocks.extend(_subdivide_axis(block, axis))
        blocks = next_blocks
    return blocks


class BernsteinOctree(nn.Module):
    """Sparse octree whose leaves store anisotropic Bernstein polynomials.

    Coefficients are packed into one learnable vector. Leaf topology, degree,
    and packed offsets are explicit persistent buffers, so storage accounting
    reflects the active representation instead of a padded maximum degree.
    """

    representation = "bernstein_octree"

    def __init__(
        self,
        *,
        levels: Iterable[int | Sequence[int]],
        max_degree: int | Sequence[int] = 3,
        attenuation_shift: float = -3.0,
        init_std: float = 1e-3,
        cuda_integrator: bool = True,
        integration_mode: str = "exact",
        topology: str = "packed_hierarchy",
        balance_2to1: bool = False,
        max_leaf_count: int | None = None,
    ):
        super().__init__()
        self.level_shapes = [_shape3(value) for value in levels]
        if not self.level_shapes:
            raise ValueError("At least one octree resolution is required.")
        if any(component <= 0 for shape in self.level_shapes for component in shape):
            raise ValueError("Octree resolutions must be positive.")
        for parent, child in zip(self.level_shapes, self.level_shapes[1:]):
            if any(child[axis] != 2 * parent[axis] for axis in range(3)):
                raise ValueError(
                    "Bernstein octree levels must use dyadic refinement; "
                    f"got parent={parent}, child={child}."
                )
        # Keep the historical public field stable for cubic checkpoints/configs.
        # An anisotropic entry remains a tuple so accidental scalar geometry
        # fails visibly instead of silently using one axis for all three.
        self.level_resolutions = [
            shape[0] if shape[0] == shape[1] == shape[2] else shape
            for shape in self.level_shapes
        ]
        if isinstance(max_degree, int):
            max_degree = (int(max_degree),) * 3
        self.max_degree = tuple(int(value) for value in max_degree)
        if len(self.max_degree) != 3 or any(value < 0 for value in self.max_degree):
            raise ValueError("max_degree must contain three nonnegative integers.")
        self.l0_shape = self.level_shapes[0]
        self.l0_resolution = self.level_resolutions[0]
        self.l1_resolution = self.level_resolutions[1] if len(self.level_resolutions) > 1 else self.l0_resolution
        self.l2_resolution = self.level_resolutions[2] if len(self.level_resolutions) > 2 else self.l1_resolution
        self.attenuation_shift = float(attenuation_shift)
        self.integration_mode = str(integration_mode).lower()
        if self.integration_mode not in {"exact", "sampled"}:
            raise ValueError("integration_mode must be 'exact' or 'sampled'.")
        self.cuda_integrator = bool(cuda_integrator)
        self.topology = str(topology).lower()
        if self.topology not in {"packed_hierarchy", "packed", "hierarchy"}:
            raise ValueError("BernsteinOctree topology must be 'packed_hierarchy'.")
        self.balance_2to1_enabled = bool(balance_2to1)
        self.max_leaf_count = int(max_leaf_count) if max_leaf_count is not None else None
        root_count = math.prod(self.l0_shape)
        if self.max_leaf_count is not None and self.max_leaf_count < root_count:
            raise ValueError("max_leaf_count cannot be smaller than the number of root cells.")

        base_coords = _all_coords(self.l0_shape)
        leaf_count = int(base_coords.shape[0])
        initial_logits = torch.empty(leaf_count, dtype=torch.float32)
        nn.init.normal_(initial_logits, mean=0.0, std=float(init_std))
        self.coefficient_logits = nn.Parameter(initial_logits)
        self.register_buffer("leaf_levels", torch.zeros(leaf_count, dtype=torch.long), persistent=True)
        self.register_buffer("leaf_coords", base_coords, persistent=True)
        self.register_buffer("leaf_degrees", torch.zeros((leaf_count, 3), dtype=torch.long), persistent=True)
        self.register_buffer("coefficient_offsets", torch.arange(leaf_count + 1, dtype=torch.long), persistent=True)
        self.register_buffer("coefficient_leaf_ids", torch.arange(leaf_count, dtype=torch.long), persistent=True)
        self.register_buffer("node_child_base", torch.empty((0,), dtype=torch.int32), persistent=False)
        self.register_buffer("node_leaf_id", torch.empty((0,), dtype=torch.int32), persistent=False)
        for level in range(len(self.level_shapes)):
            self.register_buffer(self._lookup_linear_name(level), torch.empty((0,), dtype=torch.long), persistent=False)
            self.register_buffer(self._lookup_leaf_name(level), torch.empty((0,), dtype=torch.long), persistent=False)
        self._rebuild_packed_topology()

    @staticmethod
    def _lookup_linear_name(level: int) -> str:
        return f"level_{int(level)}_leaf_linear"

    @staticmethod
    def _lookup_leaf_name(level: int) -> str:
        return f"level_{int(level)}_leaf_ids"

    def _level_leaf_lookup(self, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        return getattr(self, self._lookup_linear_name(level)), getattr(self, self._lookup_leaf_name(level))

    def coefficients(self) -> torch.Tensor:
        return F.softplus(self.coefficient_logits + self.attenuation_shift)

    def _coefficients_to_logits(self, coefficients: torch.Tensor) -> torch.Tensor:
        return _inverse_softplus(coefficients.clamp_min(1e-12)) - self.attenuation_shift

    def coefficient_counts(self) -> torch.Tensor:
        return torch.prod(self.leaf_degrees + 1, dim=1)

    def leaf_key(self, leaf_id: int) -> tuple[int, int, int, int]:
        leaf_id = int(leaf_id)
        level = int(self.leaf_levels[leaf_id].item())
        coord = self.leaf_coords[leaf_id].tolist()
        return level, int(coord[0]), int(coord[1]), int(coord[2])

    def find_leaf(self, key: Sequence[int]) -> int | None:
        level, x, y, z = (int(value) for value in key)
        if level < 0 or level >= len(self.level_shapes):
            return None
        nx, ny, nz = self.level_shapes[level]
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
            return None
        linear, leaf_ids = self._level_leaf_lookup(level)
        if linear.numel() == 0:
            return None
        target = (x * ny + y) * nz + z
        position = int(torch.searchsorted(linear, linear.new_tensor(target)).item())
        if position >= int(linear.numel()) or int(linear[position].item()) != target:
            return None
        return int(leaf_ids[position].item())

    def coefficient_block(self, leaf_id: int, *, physical: bool = True) -> torch.Tensor:
        leaf_id = int(leaf_id)
        start = int(self.coefficient_offsets[leaf_id].item())
        stop = int(self.coefficient_offsets[leaf_id + 1].item())
        values = self.coefficients()[start:stop] if physical else self.coefficient_logits[start:stop]
        shape = tuple(int(value) + 1 for value in self.leaf_degrees[leaf_id].tolist())
        return values.reshape(shape)

    def _leaf_mu(self) -> torch.Tensor:
        """Coefficient-domain leaf mean used by summaries and the viewer."""
        coefficients = self.coefficients()
        sums = coefficients.new_zeros((self.leaf_levels.shape[0],))
        sums.scatter_add_(0, self.coefficient_leaf_ids, coefficients)
        return sums / self.coefficient_counts().to(dtype=coefficients.dtype).clamp_min(1.0)

    @torch.no_grad()
    def _rebuild_packed_topology(self) -> None:
        self._topology_version = int(getattr(self, "_topology_version", 0)) + 1
        self._continuity_pair_cache = {}
        device = self.coefficient_logits.device
        root_count = math.prod(self.l0_shape)
        leaf_count = int(self.leaf_levels.shape[0])
        if leaf_count < root_count or (leaf_count - root_count) % 7 != 0:
            raise ValueError("Invalid full-octree topology: leaf count is inconsistent with eight-way splits.")
        internal_count = (leaf_count - root_count) // 7
        node_count = root_count + 8 * internal_count
        child_base = torch.full((node_count,), -1, dtype=torch.int32, device=device)
        node_leaf_id = torch.full((node_count,), -1, dtype=torch.int32, device=device)
        next_node = root_count
        previous_internal_linear = torch.empty((0,), dtype=torch.long, device=device)
        previous_internal_bases = torch.empty((0,), dtype=torch.long, device=device)

        for level, shape in enumerate(self.level_shapes):
            _, ny, nz = shape
            mask = self.leaf_levels == level
            if torch.any(mask):
                coords = self.leaf_coords[mask].to(device=device, dtype=torch.long)
                leaf_ids = torch.nonzero(mask, as_tuple=False).reshape(-1).to(device=device, dtype=torch.long)
                linear = (coords[:, 0] * ny + coords[:, 1]) * nz + coords[:, 2]
                linear, order = torch.sort(linear)
                leaf_ids = leaf_ids[order]
            else:
                linear = torch.empty((0,), dtype=torch.long, device=device)
                leaf_ids = torch.empty((0,), dtype=torch.long, device=device)
            setattr(self, self._lookup_linear_name(level), linear.contiguous())
            setattr(self, self._lookup_leaf_name(level), leaf_ids.contiguous())

            if level == 0:
                leaf_nodes = linear
            elif linear.numel():
                coords = self.leaf_coords[leaf_ids]
                _, parent_ny, parent_nz = self.level_shapes[level - 1]
                parent = torch.div(coords, 2, rounding_mode="floor")
                parent_linear = (parent[:, 0] * parent_ny + parent[:, 1]) * parent_nz + parent[:, 2]
                parent_positions = torch.searchsorted(previous_internal_linear, parent_linear)
                if torch.any(parent_positions >= previous_internal_linear.numel()) or torch.any(
                    previous_internal_linear[parent_positions] != parent_linear
                ):
                    raise ValueError("Invalid octree topology: leaf parent is not an internal node.")
                child_index = (coords[:, 0] & 1) * 4 + (coords[:, 1] & 1) * 2 + (coords[:, 2] & 1)
                leaf_nodes = previous_internal_bases[parent_positions] + child_index
            else:
                leaf_nodes = torch.empty((0,), dtype=torch.long, device=device)
            if leaf_nodes.numel():
                node_leaf_id[leaf_nodes] = leaf_ids.to(dtype=torch.int32)

            if level + 1 >= len(self.level_shapes):
                continue
            deeper = self.leaf_levels > level
            if torch.any(deeper):
                deeper_levels = self.leaf_levels[deeper]
                deeper_coords = self.leaf_coords[deeper]
                divisors = torch.pow(
                    torch.full_like(deeper_levels, 2), deeper_levels - level
                )[:, None]
                internal_coords = torch.div(deeper_coords, divisors, rounding_mode="floor")
                internal_linear = (
                    (internal_coords[:, 0] * ny + internal_coords[:, 1]) * nz
                    + internal_coords[:, 2]
                )
                internal_linear = torch.unique(internal_linear, sorted=True)
            else:
                internal_linear = torch.empty((0,), dtype=torch.long, device=device)

            if level == 0:
                internal_nodes = internal_linear
            elif internal_linear.numel():
                _, parent_ny, parent_nz = self.level_shapes[level - 1]
                x = torch.div(internal_linear, ny * nz, rounding_mode="floor")
                remainder = internal_linear - x * ny * nz
                y = torch.div(remainder, nz, rounding_mode="floor")
                z = remainder - y * nz
                parent_x, parent_y, parent_z = x // 2, y // 2, z // 2
                parent_linear = (parent_x * parent_ny + parent_y) * parent_nz + parent_z
                positions = torch.searchsorted(previous_internal_linear, parent_linear)
                if torch.any(positions >= previous_internal_linear.numel()) or torch.any(
                    previous_internal_linear[positions] != parent_linear
                ):
                    raise ValueError("Invalid octree topology: internal node parent is missing.")
                child_index = (x & 1) * 4 + (y & 1) * 2 + (z & 1)
                internal_nodes = previous_internal_bases[positions] + child_index
            else:
                internal_nodes = torch.empty((0,), dtype=torch.long, device=device)
            bases = torch.arange(
                next_node,
                next_node + 8 * int(internal_nodes.numel()),
                8,
                dtype=torch.long,
                device=device,
            )
            if internal_nodes.numel():
                child_base[internal_nodes] = bases.to(dtype=torch.int32)
            next_node += 8 * int(internal_nodes.numel())
            previous_internal_linear = internal_linear
            previous_internal_bases = bases

        if next_node != node_count:
            raise ValueError(f"Packed octree node count mismatch: built {next_node}, expected {node_count}.")
        if torch.any((child_base >= 0) & (node_leaf_id >= 0)):
            raise ValueError("Packed octree node cannot be both internal and a leaf.")
        if int(torch.sum(node_leaf_id >= 0).item()) != leaf_count:
            raise ValueError("Packed octree did not assign every leaf exactly once.")
        self.node_child_base = child_base.contiguous()
        self.node_leaf_id = node_leaf_id.contiguous()

    def resolve_leaf_ids(self, points: torch.Tensor) -> torch.Tensor:
        flat_points = points.reshape(-1, 3)
        leaf_ids = torch.full((flat_points.shape[0],), -1, dtype=torch.long, device=flat_points.device)
        valid = torch.all((flat_points >= -1.0) & (flat_points <= 1.0), dim=1)
        point_ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if point_ids.numel() == 0:
            return leaf_ids.reshape(points.shape[:-1])
        root_shape = torch.tensor(self.l0_shape, dtype=flat_points.dtype, device=flat_points.device)
        coords = torch.floor((flat_points[point_ids] + 1.0) * 0.5 * root_shape).long()
        coords = torch.minimum(torch.maximum(coords, torch.zeros_like(coords)), root_shape.long() - 1)
        nodes = (coords[:, 0] * self.l0_shape[1] + coords[:, 1]) * self.l0_shape[2] + coords[:, 2]
        for level in range(len(self.level_shapes)):
            candidates = self.node_leaf_id[nodes].long()
            hit = candidates >= 0
            if torch.any(hit):
                leaf_ids[point_ids[hit]] = candidates[hit]
            keep = ~hit
            if not torch.any(keep) or level + 1 >= len(self.level_shapes):
                break
            point_ids = point_ids[keep]
            nodes = nodes[keep]
            child_base = self.node_child_base[nodes].long()
            if torch.any(child_base < 0):
                raise RuntimeError("Packed octree traversal reached an unassigned node.")
            shape = torch.tensor(
                self.level_shapes[level + 1],
                dtype=flat_points.dtype,
                device=flat_points.device,
            )
            child_coords = torch.floor(
                (flat_points[point_ids] + 1.0) * 0.5 * shape
            ).long()
            child_coords = torch.minimum(
                torch.maximum(child_coords, torch.zeros_like(child_coords)),
                shape.long() - 1,
            )
            child_index = (child_coords[:, 0] & 1) * 4 + (child_coords[:, 1] & 1) * 2 + (child_coords[:, 2] & 1)
            nodes = child_base + child_index
        return leaf_ids.reshape(points.shape[:-1])

    def forward_mu_with_leaf_ids(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = points.shape[:-1]
        flat_points = points.reshape(-1, 3)
        leaf_ids = self.resolve_leaf_ids(flat_points).reshape(-1)
        values = flat_points.new_zeros((flat_points.shape[0],), dtype=torch.float32)
        valid = leaf_ids >= 0
        if not torch.any(valid):
            return values.reshape(original_shape), leaf_ids.reshape(original_shape)

        valid_point_ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
        valid_leaf_ids = leaf_ids[valid]
        point_degrees = self.leaf_degrees[valid_leaf_ids]
        for degree_tensor in torch.unique(point_degrees, dim=0):
            degree = tuple(int(value) for value in degree_tensor.tolist())
            degree_mask = torch.all(point_degrees == degree_tensor, dim=1)
            point_ids = valid_point_ids[degree_mask]
            group_leaf_ids = valid_leaf_ids[degree_mask]
            levels = self.leaf_levels[group_leaf_ids]
            resolutions = torch.tensor(
                self.level_shapes,
                dtype=flat_points.dtype,
                device=flat_points.device,
            )[levels]
            coords = self.leaf_coords[group_leaf_ids].to(dtype=flat_points.dtype)
            local = (flat_points[point_ids] + 1.0) * 0.5 * resolutions - coords
            bx = bernstein_basis(degree[0], local[:, 0])
            by = bernstein_basis(degree[1], local[:, 1])
            bz = bernstein_basis(degree[2], local[:, 2])
            weights = torch.einsum("ni,nj,nk->nijk", bx, by, bz).reshape(point_ids.shape[0], -1)
            offsets = torch.arange(weights.shape[1], device=flat_points.device, dtype=torch.long)
            coefficient_ids = self.coefficient_offsets[group_leaf_ids, None] + offsets[None, :]
            coefficients = self.coefficients()[coefficient_ids]
            values[point_ids] = torch.sum(coefficients * weights, dim=1)
        return values.reshape(original_shape), leaf_ids.reshape(original_shape)

    def forward_mu(self, points: torch.Tensor) -> torch.Tensor:
        return self.forward_mu_with_leaf_ids(points)[0]

    def _parallel_ray_segments(self, ray_batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized exact cell-boundary segmentation for parallel rays."""
        if (
            ray_batch.angles is None
            or ray_batch.rows is None
            or ray_batch.cols is None
            or ray_batch.detector_h is None
            or ray_batch.detector_w is None
        ):
            raise ValueError("Exact Bernstein integration requires parallel-ray metadata.")
        device = self.coefficient_logits.device
        dtype = self.coefficient_logits.dtype
        effective_angles = -ray_batch.angles.to(device=device, dtype=dtype)
        rows = ray_batch.rows.to(device=device, dtype=dtype)
        cols = ray_batch.cols.to(device=device, dtype=dtype)
        ca = torch.cos(effective_angles)
        sa = torch.sin(effective_angles)
        directions = torch.stack([ca, sa], dim=1)
        u_axes = torch.stack([-sa, ca], dim=1)
        u = -1.0 + (cols + 0.5) * 2.0 / float(ray_batch.detector_w)
        z = -1.0 + (rows + 0.5) * 2.0 / float(ray_batch.detector_h)
        bases = u_axes * u[:, None]

        eps = 1e-8
        t_min = torch.full((directions.shape[0],), -1e20, dtype=dtype, device=device)
        t_max = torch.full_like(t_min, 1e20)
        valid_ray = torch.ones_like(t_min, dtype=torch.bool)
        for axis in range(2):
            direction = directions[:, axis]
            base = bases[:, axis]
            parallel = direction.abs() < eps
            valid_ray &= (~parallel) | ((base >= -1.0) & (base <= 1.0))
            denominator = torch.where(parallel, torch.ones_like(direction), direction)
            t0 = (-1.0 - base) / denominator
            t1 = (1.0 - base) / denominator
            t_min = torch.where(parallel, t_min, torch.maximum(t_min, torch.minimum(t0, t1)))
            t_max = torch.where(parallel, t_max, torch.minimum(t_max, torch.maximum(t0, t1)))
        valid_ray &= t_max > t_min

        candidates = [t_min[:, None], t_max[:, None]]
        for axis in range(2):
            finest_resolution = self.level_shapes[-1][axis]
            boundaries = -1.0 + torch.arange(
                1,
                finest_resolution,
                dtype=dtype,
                device=device,
            ) * (2.0 / float(finest_resolution))
            direction = directions[:, axis]
            denominator = torch.where(direction.abs() < eps, torch.ones_like(direction), direction)
            crossings = (boundaries[None, :] - bases[:, axis, None]) / denominator[:, None]
            inside = (
                (direction.abs() >= eps)[:, None]
                & (crossings > t_min[:, None] + eps)
                & (crossings < t_max[:, None] - eps)
            )
            crossings = torch.where(inside, crossings, torch.full_like(crossings, float("inf")))
            candidates.append(crossings)
        sorted_t = torch.sort(torch.cat(candidates, dim=1), dim=1).values
        starts = sorted_t[:, :-1]
        stops = sorted_t[:, 1:]
        valid_segment = (
            valid_ray[:, None]
            & torch.isfinite(starts)
            & torch.isfinite(stops)
            & (stops > starts + eps)
        )
        ray_grid = torch.arange(ray_batch.num_rays, device=device)[:, None].expand_as(starts)
        ray_ids = ray_grid[valid_segment]
        starts = starts[valid_segment]
        stops = stops[valid_segment]
        mid = 0.5 * (starts + stops)
        xy = bases[ray_ids] + directions[ray_ids] * mid[:, None]
        points = torch.stack([xy[:, 0], xy[:, 1], z[ray_ids]], dim=1)
        leaf_ids = self.resolve_leaf_ids(points).reshape(-1)
        valid_leaf = leaf_ids >= 0
        return ray_ids[valid_leaf], leaf_ids[valid_leaf], torch.stack([starts[valid_leaf], stops[valid_leaf]], dim=1)

    def exact_ray_cell_contributions(self, ray_batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ray_ids, leaf_ids, intervals = self._parallel_ray_segments(ray_batch)
        segment_integrals = intervals.new_zeros((intervals.shape[0],))
        segment_degrees = self.leaf_degrees[leaf_ids]
        for degree_tensor in torch.unique(segment_degrees, dim=0):
            degree_mask = torch.all(segment_degrees == degree_tensor, dim=1)
            segment_ids = torch.nonzero(degree_mask, as_tuple=False).reshape(-1)
            degree = tuple(int(value) for value in degree_tensor.tolist())
            quadrature_order = max(1, int(math.ceil((sum(degree) + 1) / 2.0)))
            nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature_order)
            nodes = torch.tensor(nodes_np, dtype=intervals.dtype, device=intervals.device)
            weights = torch.tensor(weights_np, dtype=intervals.dtype, device=intervals.device)
            selected_intervals = intervals[segment_ids]
            half = 0.5 * (selected_intervals[:, 1] - selected_intervals[:, 0])
            centre = 0.5 * (selected_intervals[:, 1] + selected_intervals[:, 0])
            t = centre[:, None] + half[:, None] * nodes[None, :]

            effective_angles = -ray_batch.angles.to(device=intervals.device, dtype=intervals.dtype)
            ca = torch.cos(effective_angles)
            sa = torch.sin(effective_angles)
            directions = torch.stack([ca, sa], dim=1)
            u_axes = torch.stack([-sa, ca], dim=1)
            cols = ray_batch.cols.to(device=intervals.device, dtype=intervals.dtype)
            rows = ray_batch.rows.to(device=intervals.device, dtype=intervals.dtype)
            u = -1.0 + (cols + 0.5) * 2.0 / float(ray_batch.detector_w)
            z = -1.0 + (rows + 0.5) * 2.0 / float(ray_batch.detector_h)
            bases = u_axes * u[:, None]
            selected_rays = ray_ids[segment_ids]
            xy = bases[selected_rays, None, :] + directions[selected_rays, None, :] * t[:, :, None]
            points = torch.stack(
                [
                    xy[:, :, 0],
                    xy[:, :, 1],
                    z[selected_rays, None].expand_as(t),
                ],
                dim=2,
            )
            values = self.forward_mu(points.reshape(-1, 3)).reshape(segment_ids.shape[0], quadrature_order)
            segment_integrals[segment_ids] = half * torch.sum(values * weights[None, :], dim=1)

        keys = ray_ids * int(self.leaf_levels.shape[0]) + leaf_ids
        unique_keys, inverse = torch.unique(keys, return_inverse=True)
        totals = segment_integrals.new_zeros((unique_keys.shape[0],))
        totals.scatter_add_(0, inverse, segment_integrals)
        return (
            torch.div(unique_keys, int(self.leaf_levels.shape[0]), rounding_mode="floor"),
            torch.remainder(unique_keys, int(self.leaf_levels.shape[0])),
            totals,
        )

    def _can_use_cuda_integrator(self, ray_batch) -> bool:
        return (
            self.cuda_integrator
            and self.integration_mode == "exact"
            and self.coefficient_logits.is_cuda
            and has_bernstein_native()
            and max(self.max_degree) <= 3
            and ray_batch.angles is not None
            and ray_batch.rows is not None
            and ray_batch.cols is not None
            and ray_batch.detector_h is not None
            and ray_batch.detector_w is not None
        )

    def _cuda_integrator_arguments(self, ray_batch) -> dict:
        return {
            "coefficient_logits": self.coefficient_logits,
            "leaf_degrees": self.leaf_degrees,
            "coefficient_offsets": self.coefficient_offsets,
            "node_child_base": self.node_child_base,
            "node_leaf_id": self.node_leaf_id,
            "angles": ray_batch.angles.to(device=self.coefficient_logits.device, dtype=torch.float32),
            "rows": ray_batch.rows.to(device=self.coefficient_logits.device, dtype=torch.long),
            "cols": ray_batch.cols.to(device=self.coefficient_logits.device, dtype=torch.long),
            "detector_h": int(ray_batch.detector_h),
            "detector_w": int(ray_batch.detector_w),
            "level_shapes": self.level_shapes,
            "attenuation_shift": self.attenuation_shift,
        }

    def ray_cell_contributions(self, ray_batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sparse (ray, leaf, line-integral) tuples for projection decisions."""
        if self._can_use_cuda_integrator(ray_batch):
            return bernstein_octree_ray_segments(**self._cuda_integrator_arguments(ray_batch))
        if self.integration_mode == "exact":
            return self.exact_ray_cell_contributions(ray_batch)
        if ray_batch.points is None or ray_batch.step is None:
            raise ValueError("Bernstein projection requires materialized quadrature points.")
        values, leaf_ids = self.forward_mu_with_leaf_ids(ray_batch.points)
        leaf_ids = leaf_ids.reshape(-1)
        ray_ids = torch.arange(ray_batch.num_rays, device=values.device).repeat_interleave(ray_batch.samples_per_ray)
        step = ray_batch.step[ray_ids]
        valid = leaf_ids >= 0
        ray_ids = ray_ids[valid]
        leaf_ids = leaf_ids[valid]
        contributions = values.reshape(-1)[valid] * step[valid]
        keys = ray_ids * int(self.leaf_levels.shape[0]) + leaf_ids
        unique_keys, inverse = torch.unique(keys, return_inverse=True)
        totals = contributions.new_zeros((unique_keys.shape[0],))
        totals.scatter_add_(0, inverse, contributions)
        return (
            torch.div(unique_keys, int(self.leaf_levels.shape[0]), rounding_mode="floor"),
            torch.remainder(unique_keys, int(self.leaf_levels.shape[0])),
            totals,
        )

    def integrate_ray_batch(self, ray_batch) -> torch.Tensor:
        if self._can_use_cuda_integrator(ray_batch):
            return bernstein_octree_integrate(**self._cuda_integrator_arguments(ray_batch))
        if self.integration_mode == "exact":
            ray_ids, _, contributions = self.exact_ray_cell_contributions(ray_batch)
            prediction = contributions.new_zeros((ray_batch.num_rays,))
            prediction.scatter_add_(0, ray_ids, contributions)
            return prediction
        if ray_batch.points is None or ray_batch.step is None:
            raise ValueError("Bernstein projection requires materialized quadrature points.")
        values = self.forward_mu(ray_batch.points).reshape(ray_batch.num_rays, ray_batch.samples_per_ray)
        return torch.sum(values * ray_batch.step[:, None], dim=1)

    def prefer_compact_ray_batch(self) -> bool:
        return (
            self.cuda_integrator
            and self.integration_mode == "exact"
            and self.coefficient_logits.is_cuda
            and has_bernstein_native()
            and max(self.max_degree) <= 3
        )

    def decoded_at_resolution(self, resolution: int | tuple[int, int, int], *, chunk: int | None = None) -> torch.Tensor:
        if isinstance(resolution, int):
            shape = (int(resolution),) * 3
        else:
            shape = tuple(int(value) for value in resolution)
        device = self.coefficient_logits.device
        axes = [torch.arange(value, dtype=torch.float32, device=device) for value in shape]
        xx, yy, zz = torch.meshgrid(*axes, indexing="ij")
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        scale = torch.tensor(shape, dtype=torch.float32, device=device)
        points = -1.0 + (coords + 0.5) * 2.0 / scale
        if chunk is None:
            return self.forward_mu(points).reshape(shape)
        values = []
        for start in range(0, points.shape[0], int(chunk)):
            values.append(self.forward_mu(points[start : start + int(chunk)]))
        return torch.cat(values, dim=0).reshape(shape)

    def decoded_l0(self) -> torch.Tensor:
        return self.decoded_at_resolution(self.l0_shape)

    @torch.no_grad()
    def _replace_leaves(
        self,
        removed_leaf_ids: torch.Tensor,
        new_levels: torch.Tensor,
        new_coords: torch.Tensor,
        new_degrees: torch.Tensor,
        new_blocks: Sequence[torch.Tensor] | torch.Tensor,
    ) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
        device = self.coefficient_logits.device
        removed_leaf_ids = torch.unique(removed_leaf_ids.to(device=device, dtype=torch.long))
        removed_leaf_ids = removed_leaf_ids[(removed_leaf_ids >= 0) & (removed_leaf_ids < self.leaf_levels.shape[0])]
        # Batch topology changes can involve hundreds of thousands of leaves.
        # Calling leaf_key() here performs several scalar .item() synchronizations
        # per CUDA leaf and makes coarse-band p-refinement effectively serial.
        old_levels = self.leaf_levels[removed_leaf_ids].detach().cpu().tolist()
        old_coords = self.leaf_coords[removed_leaf_ids].detach().cpu().tolist()
        old_keys = [
            (int(level), int(coord[0]), int(coord[1]), int(coord[2]))
            for level, coord in zip(old_levels, old_coords)
        ]
        remove_leaf_mask = torch.zeros(self.leaf_levels.shape[0], dtype=torch.bool, device=device)
        remove_leaf_mask[removed_leaf_ids] = True
        keep_leaf_mask = ~remove_leaf_mask
        keep_coefficient_mask = ~remove_leaf_mask[self.coefficient_leaf_ids]

        physical = self.coefficients().detach()
        kept_coefficients = physical[keep_coefficient_mask]
        if isinstance(new_blocks, torch.Tensor):
            appended_coefficients = new_blocks.to(device=device, dtype=physical.dtype).reshape(-1)
        else:
            appended_coefficients = (
                torch.cat([block.reshape(-1).to(device=device, dtype=physical.dtype) for block in new_blocks], dim=0)
                if new_blocks
                else physical.new_empty((0,))
            )
        coefficients = torch.cat([kept_coefficients, appended_coefficients], dim=0)
        levels = torch.cat([self.leaf_levels[keep_leaf_mask], new_levels.to(device=device, dtype=torch.long)], dim=0)
        coords = torch.cat([self.leaf_coords[keep_leaf_mask], new_coords.to(device=device, dtype=torch.long)], dim=0)
        degrees = torch.cat([self.leaf_degrees[keep_leaf_mask], new_degrees.to(device=device, dtype=torch.long)], dim=0)
        counts = torch.prod(degrees + 1, dim=1)
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(counts, dim=0)],
            dim=0,
        )
        owners = torch.repeat_interleave(torch.arange(levels.shape[0], device=device), counts)

        self.coefficient_logits = nn.Parameter(self._coefficients_to_logits(coefficients).contiguous())
        self.leaf_levels = levels.contiguous()
        self.leaf_coords = coords.contiguous()
        self.leaf_degrees = degrees.contiguous()
        self.coefficient_offsets = offsets.contiguous()
        self.coefficient_leaf_ids = owners.contiguous()
        self._rebuild_packed_topology()
        new_keys = [
            (int(level), int(coord[0]), int(coord[1]), int(coord[2]))
            for level, coord in zip(new_levels.tolist(), new_coords.tolist())
        ]
        return old_keys, new_keys

    @torch.no_grad()
    def elevate_degree(self, leaf_id: int, axis: int) -> TopologyChange:
        leaf_id = int(leaf_id)
        axis = int(axis)
        degree = int(self.leaf_degrees[leaf_id, axis].item())
        if degree >= self.max_degree[axis]:
            raise ValueError(f"Leaf {leaf_id} already has maximum degree on axis {axis}.")
        old_count = int(self.coefficient_logits.numel())
        elevated = _degree_elevate_tensor(self.coefficient_block(leaf_id).detach(), axis)
        new_degree = self.leaf_degrees[leaf_id].detach().clone()
        new_degree[axis] += 1
        old_keys, new_keys = self._replace_leaves(
            torch.tensor([leaf_id], device=self.coefficient_logits.device),
            self.leaf_levels[leaf_id : leaf_id + 1].detach(),
            self.leaf_coords[leaf_id : leaf_id + 1].detach(),
            new_degree[None],
            [elevated],
        )
        return TopologyChange("p_elevate", tuple(old_keys), tuple(new_keys), old_count, int(self.coefficient_logits.numel()))

    @torch.no_grad()
    def reduce_degree(self, leaf_id: int, axis: int) -> TopologyChange:
        leaf_id = int(leaf_id)
        axis = int(axis)
        degree = int(self.leaf_degrees[leaf_id, axis].item())
        if degree <= 0:
            raise ValueError(f"Leaf {leaf_id} is already constant on axis {axis}.")
        old_count = int(self.coefficient_logits.numel())
        source = self.coefficient_block(leaf_id).detach().movedim(axis, 0)
        lower_degree = degree - 1
        elevation = source.new_zeros((degree + 1, lower_degree + 1))
        for basis_id in range(lower_degree + 1):
            unit = source.new_zeros((lower_degree + 1, 1, 1))
            unit[basis_id] = 1.0
            elevation[:, basis_id] = _degree_elevate_tensor(unit, 0)[:, 0, 0]
        flattened = source.reshape(degree + 1, -1)
        reduced = torch.linalg.lstsq(elevation, flattened).solution.clamp_min(1e-12)
        reduced = reduced.reshape((lower_degree + 1, *source.shape[1:])).movedim(0, axis)
        new_degree = self.leaf_degrees[leaf_id].detach().clone()
        new_degree[axis] -= 1
        old_keys, new_keys = self._replace_leaves(
            torch.tensor([leaf_id], device=self.coefficient_logits.device),
            self.leaf_levels[leaf_id : leaf_id + 1].detach(),
            self.leaf_coords[leaf_id : leaf_id + 1].detach(),
            new_degree[None],
            [reduced],
        )
        return TopologyChange("p_reduce", tuple(old_keys), tuple(new_keys), old_count, int(self.coefficient_logits.numel()))

    @torch.no_grad()
    def split_leaf(self, leaf_id: int) -> TopologyChange:
        leaf_id = int(leaf_id)
        level = int(self.leaf_levels[leaf_id].item())
        if level + 1 >= len(self.level_shapes):
            raise ValueError(f"Leaf {leaf_id} is already at maximum octree depth.")
        old_count = int(self.coefficient_logits.numel())
        blocks = subdivide_bernstein_tensor(self.coefficient_block(leaf_id).detach())
        offsets = _all_coords(2, device=self.coefficient_logits.device)
        coords = self.leaf_coords[leaf_id][None, :] * 2 + offsets
        levels = torch.full((8,), level + 1, dtype=torch.long, device=self.coefficient_logits.device)
        degrees = self.leaf_degrees[leaf_id][None, :].expand(8, -1).clone()
        old_keys, new_keys = self._replace_leaves(
            torch.tensor([leaf_id], device=self.coefficient_logits.device),
            levels,
            coords,
            degrees,
            blocks,
        )
        return TopologyChange("h_split", tuple(old_keys), tuple(new_keys), old_count, int(self.coefficient_logits.numel()))

    @torch.no_grad()
    def split_leaves_batch(self, leaf_ids: torch.Tensor | Sequence[int]) -> TopologyChange:
        leaf_ids_tensor = torch.unique(
            torch.as_tensor(leaf_ids, dtype=torch.long, device=self.coefficient_logits.device),
            sorted=True,
        )
        leaf_ids_tensor = leaf_ids_tensor[
            (leaf_ids_tensor >= 0) & (leaf_ids_tensor < self.leaf_levels.shape[0])
        ]
        if self.max_leaf_count is not None:
            available_parents = max(
                0,
                (int(self.max_leaf_count) - int(self.leaf_levels.shape[0])) // 7,
            )
            leaf_ids_tensor = leaf_ids_tensor[:available_parents]
        if leaf_ids_tensor.numel() == 0:
            old_count = int(self.coefficient_logits.numel())
            return TopologyChange("h_split_batch", tuple(), tuple(), old_count, old_count)
        levels = self.leaf_levels[leaf_ids_tensor]
        if torch.any(levels + 1 >= len(self.level_shapes)):
            raise ValueError("At least one selected leaf is already at maximum octree depth.")

        old_count = int(self.coefficient_logits.numel())
        child_offsets = _all_coords(2, device=self.coefficient_logits.device)
        coords = (self.leaf_coords[leaf_ids_tensor][:, None, :] * 2 + child_offsets[None, :, :]).reshape(-1, 3)
        new_levels = (levels[:, None] + 1).expand(-1, 8).reshape(-1)
        new_degrees = self.leaf_degrees[leaf_ids_tensor][:, None, :].expand(-1, 8, -1).reshape(-1, 3).clone()

        if torch.all(self.leaf_degrees[leaf_ids_tensor] == 0):
            physical = self.coefficients().detach()
            parent_coefficients = physical[self.coefficient_offsets[leaf_ids_tensor]]
            new_blocks: Sequence[torch.Tensor] | torch.Tensor = parent_coefficients.repeat_interleave(8)
        else:
            blocks: list[torch.Tensor] = []
            for leaf_id in leaf_ids_tensor.tolist():
                blocks.extend(subdivide_bernstein_tensor(self.coefficient_block(int(leaf_id)).detach()))
            new_blocks = blocks

        old_keys, new_keys = self._replace_leaves(
            leaf_ids_tensor,
            new_levels,
            coords,
            new_degrees,
            new_blocks,
        )
        return TopologyChange(
            "h_split_batch",
            tuple(old_keys),
            tuple(new_keys),
            old_count,
            int(self.coefficient_logits.numel()),
        )

    @torch.no_grad()
    def balance_2to1(self, *, point_chunk: int = 262144, max_rounds: int = 8) -> dict:
        """Split face-adjacent coarse leaves until neighboring depths differ by at most one."""
        rounds = 0
        split_parents = 0
        complete = True
        for _ in range(max(1, int(max_rounds))):
            violations: list[torch.Tensor] = []
            for fine_level in range(2, len(self.level_shapes)):
                fine_ids = torch.nonzero(self.leaf_levels == fine_level, as_tuple=False).reshape(-1)
                if fine_ids.numel() == 0:
                    continue
                cell = torch.tensor(
                    [2.0 / float(value) for value in self.level_shapes[fine_level]],
                    dtype=torch.float32,
                    device=self.coefficient_logits.device,
                )
                for start in range(0, int(fine_ids.numel()), int(point_chunk)):
                    batch_ids = fine_ids[start : start + int(point_chunk)]
                    coords = self.leaf_coords[batch_ids].to(dtype=torch.float32)
                    centers = -1.0 + (coords + 0.5) * cell[None, :]
                    for axis in range(3):
                        for side in (-1, 1):
                            points = centers.clone()
                            boundary_offset = 0.5 * cell[axis] * float(side)
                            points[:, axis] += boundary_offset + float(side) * cell[axis] * 1.0e-3
                            inside = (points[:, axis] > -1.0) & (points[:, axis] < 1.0)
                            if not torch.any(inside):
                                continue
                            neighbor_ids = self.resolve_leaf_ids(points[inside]).reshape(-1)
                            valid = neighbor_ids >= 0
                            if not torch.any(valid):
                                continue
                            neighbor_ids = neighbor_ids[valid]
                            too_coarse = self.leaf_levels[neighbor_ids] < fine_level - 1
                            if torch.any(too_coarse):
                                violations.append(neighbor_ids[too_coarse])
            if not violations:
                break
            selected = torch.unique(torch.cat(violations), sorted=True)
            before = int(self.leaf_levels.shape[0])
            change = self.split_leaves_batch(selected)
            changed = (int(change.new_coefficient_count) != int(change.old_coefficient_count)) or bool(change.new_leaf_keys)
            added_parents = (int(self.leaf_levels.shape[0]) - before) // 7
            split_parents += max(0, added_parents)
            rounds += 1
            if added_parents < int(selected.numel()) or not changed:
                complete = False
                break
        else:
            complete = False
        return {
            "enabled": True,
            "rounds": rounds,
            "split_parents": split_parents,
            "complete": complete,
            "leaf_count": int(self.leaf_levels.shape[0]),
        }

    @torch.no_grad()
    def elevate_leaves_isotropic_batch(
        self,
        leaf_ids: torch.Tensor | Sequence[int],
        target_degree: int | Sequence[int],
    ) -> TopologyChange:
        leaf_ids_tensor = torch.unique(
            torch.as_tensor(leaf_ids, dtype=torch.long, device=self.coefficient_logits.device),
            sorted=True,
        )
        leaf_ids_tensor = leaf_ids_tensor[
            (leaf_ids_tensor >= 0) & (leaf_ids_tensor < self.leaf_levels.shape[0])
        ]
        if isinstance(target_degree, int):
            target = torch.full((3,), int(target_degree), dtype=torch.long, device=self.coefficient_logits.device)
        else:
            target = torch.tensor(list(target_degree), dtype=torch.long, device=self.coefficient_logits.device)
        if target.numel() != 3:
            raise ValueError("target_degree must be an int or a 3-value sequence.")
        max_degree_tensor = torch.tensor(self.max_degree, dtype=torch.long, device=self.coefficient_logits.device)
        target = torch.minimum(torch.maximum(target, torch.zeros_like(target)), max_degree_tensor)
        if leaf_ids_tensor.numel() == 0:
            old_count = int(self.coefficient_logits.numel())
            return TopologyChange("p_elevate_batch", tuple(), tuple(), old_count, old_count)

        current_degrees = self.leaf_degrees[leaf_ids_tensor]
        new_degrees = torch.maximum(current_degrees, target[None, :])
        changed = torch.any(new_degrees != current_degrees, dim=1)
        leaf_ids_tensor = leaf_ids_tensor[changed]
        new_degrees = new_degrees[changed]
        if leaf_ids_tensor.numel() == 0:
            old_count = int(self.coefficient_logits.numel())
            return TopologyChange("p_elevate_batch", tuple(), tuple(), old_count, old_count)

        old_count = int(self.coefficient_logits.numel())
        selected_levels = self.leaf_levels[leaf_ids_tensor].detach()
        selected_coords = self.leaf_coords[leaf_ids_tensor].detach()

        if torch.all(self.leaf_degrees[leaf_ids_tensor] == 0) and torch.all(new_degrees == new_degrees[0]):
            physical = self.coefficients().detach()
            parent_coefficients = physical[self.coefficient_offsets[leaf_ids_tensor]]
            coefficient_count = int(torch.prod(new_degrees[0] + 1).item())
            new_blocks: Sequence[torch.Tensor] | torch.Tensor = (
                parent_coefficients[:, None].expand(-1, coefficient_count).contiguous()
            )
        else:
            blocks: list[torch.Tensor] = []
            for leaf_id, degree_tensor in zip(leaf_ids_tensor.tolist(), new_degrees):
                block = self.coefficient_block(int(leaf_id)).detach()
                current = list(int(value) for value in self.leaf_degrees[int(leaf_id)].tolist())
                target_tuple = tuple(int(value) for value in degree_tensor.tolist())
                for axis in range(3):
                    while current[axis] < target_tuple[axis]:
                        block = _degree_elevate_tensor(block, axis)
                        current[axis] += 1
                blocks.append(block)
            new_blocks = blocks

        old_keys, new_keys = self._replace_leaves(
            leaf_ids_tensor,
            selected_levels,
            selected_coords,
            new_degrees,
            new_blocks,
        )
        return TopologyChange(
            "p_elevate_batch",
            tuple(old_keys),
            tuple(new_keys),
            old_count,
            int(self.coefficient_logits.numel()),
        )

    def _fit_parent_block(self, child_ids: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        degree_tuple = tuple(int(value) for value in degree.tolist())
        samples_per_axis = tuple(max(2 * (value + 1), 4) for value in degree_tuple)
        axes = [
            (torch.arange(count, dtype=torch.float32, device=self.coefficient_logits.device) + 0.5) / float(count)
            for count in samples_per_axis
        ]
        xx, yy, zz = torch.meshgrid(*axes, indexing="ij")
        local = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        child_level = int(self.leaf_levels[int(child_ids[0])].item())
        parent_level = child_level - 1
        parent_coord = torch.div(self.leaf_coords[int(child_ids[0])], 2, rounding_mode="floor")
        cell = torch.tensor(
            [2.0 / float(value) for value in self.level_shapes[parent_level]],
            dtype=torch.float32,
            device=self.coefficient_logits.device,
        )
        points = -1.0 + (parent_coord.to(dtype=torch.float32)[None, :] + local) * cell[None, :]
        target = self.forward_mu(points).detach()
        bx = bernstein_basis(degree_tuple[0], local[:, 0])
        by = bernstein_basis(degree_tuple[1], local[:, 1])
        bz = bernstein_basis(degree_tuple[2], local[:, 2])
        design = torch.einsum("ni,nj,nk->nijk", bx, by, bz).reshape(local.shape[0], -1)
        return torch.linalg.lstsq(design, target[:, None]).solution[:, 0].clamp_min(1e-12).reshape(
            tuple(value + 1 for value in degree_tuple)
        )

    @torch.no_grad()
    def merge_siblings(self, child_ids: Sequence[int]) -> TopologyChange:
        child_ids_tensor = torch.unique(
            torch.tensor(list(child_ids), dtype=torch.long, device=self.coefficient_logits.device)
        )
        if child_ids_tensor.numel() != 8:
            raise ValueError("An octree merge requires exactly eight sibling leaves.")
        levels = self.leaf_levels[child_ids_tensor]
        if not torch.all(levels == levels[0]) or int(levels[0].item()) <= 0:
            raise ValueError("Merge candidates must share one non-root level.")
        parent_coords = torch.div(self.leaf_coords[child_ids_tensor], 2, rounding_mode="floor")
        if torch.unique(parent_coords, dim=0).shape[0] != 1:
            raise ValueError("Merge candidates are not siblings.")
        expected = parent_coords[0][None, :] * 2 + _all_coords(2, device=self.coefficient_logits.device)
        if set(map(tuple, self.leaf_coords[child_ids_tensor].tolist())) != set(map(tuple, expected.tolist())):
            raise ValueError("Merge candidates do not cover all eight octants.")
        old_count = int(self.coefficient_logits.numel())
        degree = torch.amax(self.leaf_degrees[child_ids_tensor], dim=0)
        parent_block = self._fit_parent_block(child_ids_tensor, degree)
        old_keys, new_keys = self._replace_leaves(
            child_ids_tensor,
            (levels[0] - 1)[None],
            parent_coords[:1],
            degree[None],
            [parent_block],
        )
        return TopologyChange("h_merge", tuple(old_keys), tuple(new_keys), old_count, int(self.coefficient_logits.numel()))

    @torch.no_grad()
    def activate_level_from_score(self, level: int, score: torch.Tensor, active_fraction: float, halo: int = 0) -> int:
        """Compatibility adapter for legacy milestone configs."""
        level = int(level)
        if level <= 0 or level >= len(self.level_shapes):
            raise ValueError(f"Refinement level must be in 1..{len(self.level_shapes) - 1}.")
        parent_level = level - 1
        candidates = torch.nonzero(self.leaf_levels == parent_level, as_tuple=False).reshape(-1)
        if candidates.numel() == 0:
            return int(torch.sum(self.leaf_levels == level).item())
        parent_shape = self.level_shapes[parent_level]
        if tuple(score.shape) != parent_shape:
            score = F.interpolate(
                score[None, None].to(dtype=torch.float32),
                size=parent_shape,
                mode="trilinear",
                align_corners=False,
            )[0, 0]
        coords = self.leaf_coords[candidates]
        candidate_scores = score[coords[:, 0], coords[:, 1], coords[:, 2]]
        count = max(1, int(candidates.numel() * float(active_fraction)))
        selected = candidates[torch.topk(candidate_scores, min(count, candidates.numel())).indices]
        self.split_leaves_batch(selected)
        return int(torch.sum(self.leaf_levels == level).item())

    @torch.no_grad()
    def activate_level_from_gradient(self, level: int, active_fraction: float, halo: int = 0) -> int:
        raise RuntimeError(
            "Dense-volume gradient refinement is disabled for BernsteinOctree; "
            "use coefficient diagnostics and held-out projection validation."
        )

    def stats(self) -> BernsteinStats:
        active_by_level = [
            int(torch.sum(self.leaf_levels == level).item()) for level in range(len(self.level_shapes))
        ]
        parameter_count = int(self.coefficient_logits.numel())
        model_bytes = int(self.coefficient_logits.numel() * self.coefficient_logits.element_size())
        model_bytes += sum(
            int(buffer.numel() * buffer.element_size())
            for buffer in (
                self.leaf_levels,
                self.leaf_coords,
                self.leaf_degrees,
                self.coefficient_offsets,
                self.coefficient_leaf_ids,
                self.node_child_base,
                self.node_leaf_id,
            )
        )
        return BernsteinStats(
            parameter_count=parameter_count,
            model_bytes=model_bytes,
            l0_cells=math.prod(self.l0_shape),
            l0_active=active_by_level[0] if active_by_level else 0,
            l1_active=active_by_level[1] if len(active_by_level) > 1 else 0,
            l2_active=active_by_level[2] if len(active_by_level) > 2 else 0,
            l3_active=active_by_level[3] if len(active_by_level) > 3 else 0,
            active_by_level=tuple(active_by_level),
            leaf_cells=int(self.leaf_levels.shape[0]),
            max_depth=max((level for level, count in enumerate(active_by_level) if count > 0), default=0),
            representation=self.representation,
            coefficient_count=parameter_count,
            max_degree=self.max_degree,
        )

    def prepare_sparse_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        required = {
            "coefficient_logits",
            "leaf_levels",
            "leaf_coords",
            "leaf_degrees",
            "coefficient_offsets",
            "coefficient_leaf_ids",
        }
        missing = sorted(required.difference(state_dict))
        if missing:
            raise ValueError(f"Bernstein checkpoint is missing fields: {missing}.")
        device = self.coefficient_logits.device
        self.coefficient_logits = nn.Parameter(torch.empty_like(state_dict["coefficient_logits"], device=device))
        for name in required.difference({"coefficient_logits"}):
            setattr(self, name, torch.empty_like(state_dict[name], device=device))

    def load_state_dict(self, state_dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._rebuild_packed_topology()
        return result
