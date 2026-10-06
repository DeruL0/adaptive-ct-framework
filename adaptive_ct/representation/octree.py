from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from adaptive_ct.core.projector import (
    bernstein_octree_integrate,
    bernstein_octree_ray_segments,
    has_bernstein_native,
)


# Leaf types, stored as degree triples (the CUDA traversal reads them):
#   constant  (0, 0, 0)  1 value
#   trilinear (1, 1, 1)  8 corner values
#   step      (1, 0, 0)  2 values c0, c1:  (1 - H) c0 + H c1,  H = sigmoid((n . (u - 1/2) - d) / w)
#             with geometry (n, d, w) in step_geometry; w = 0 is a hard step.
LEAF_TYPES = {"constant": (0, 0, 0), "trilinear": (1, 1, 1), "step": (1, 0, 0)}
TYPE_NAMES = ("constant", "trilinear", "step")  # type id = index


def type_ids(degrees: torch.Tensor) -> torch.Tensor:
    """0 constant, 1 trilinear, 2 step."""
    return torch.where(degrees[:, 0] == 0, 0, torch.where(degrees[:, 1] == 1, 1, 2))


def step_integral(geometry: torch.Tensor, s_start: torch.Tensor, s_stop: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """Exact integral of H over straight segments, given s = n . (u - 1/2) at both ends (same formula as the CUDA kernel)."""
    s0, s1, w = s_start - geometry[:, 3], s_stop - geometry[:, 3], geometry[:, 4]
    hard = torch.where((s0 > 0) & (s1 > 0), length, torch.where((s0 <= 0) & (s1 <= 0), torch.zeros_like(length),
                                                                 length * torch.maximum(s0, s1) / (s1 - s0).abs().clamp_min(1e-30)))
    ws = w.clamp_min(1e-30)
    a0, a1 = s0 / ws, s1 / ws
    close = (a1 - a0).abs() < 1e-3
    smooth = torch.where(close, length * torch.sigmoid(0.5 * (a0 + a1)),
                         length * (F.softplus(a1) - F.softplus(a0)) / torch.where(close, torch.ones_like(a0), a1 - a0))
    return torch.where(w > 0, smooth, hard)


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
    leaf_types: tuple[str, ...]
    leaf_type_counts: tuple[int, ...]


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


def unique_degree_rows(degrees: torch.Tensor) -> torch.Tensor:
    """Fast replacement for `torch.unique(degrees, dim=0)` on (N, 3) small
    nonnegative-integer degree tuples.

    PyTorch's dim=0 unique is a general row-wise comparison and is
    dramatically slower on CPU than the plain 1D unique -- for a few hundred
    thousand rows the difference is seconds vs. milliseconds. Since a Bernstein
    degree is a handful of small integers per axis, encode each row into one
    integer key, run the well-optimized 1D unique, and decode back.
    """
    if degrees.shape[0] == 0:
        return degrees
    base = max(int(degrees.max().item()) + 1, 2)
    keys = (degrees[:, 0] * base + degrees[:, 1]) * base + degrees[:, 2]
    unique_keys = torch.unique(keys)
    d0 = torch.div(unique_keys, base * base, rounding_mode="floor")
    remainder = unique_keys - d0 * base * base
    d1 = torch.div(remainder, base, rounding_mode="floor")
    d2 = remainder - d1 * base
    return torch.stack([d0, d1, d2], dim=1)


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


class BernsteinOctree(nn.Module):
    """Sparse octree whose leaves are constant, trilinear or step functions (see LEAF_TYPES).

    Physical coefficients c = softplus(logit + attenuation_shift) >= 0 are packed into one vector; every
    leaf value is a nonnegative weighted average of its coefficients, so the field is >= 0.  Leaf topology,
    type (degree triple), packed offsets and step geometry are explicit buffers.
    """

    representation = "bernstein_octree"

    def __init__(
        self,
        *,
        levels: Iterable[int | Sequence[int]],
        leaf_types: Sequence[str] = ("constant", "trilinear", "step"),
        attenuation_shift: float = -3.0,
        cuda_integrator: bool = True,
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
        self.leaf_types = tuple(str(t) for t in leaf_types)
        if not self.leaf_types or any(t not in LEAF_TYPES for t in self.leaf_types):
            raise ValueError(f"leaf_types must be a non-empty subset of {list(LEAF_TYPES)}.")
        self.l0_shape = self.level_shapes[0]
        self.attenuation_shift = float(attenuation_shift)
        self.cuda_integrator = bool(cuda_integrator)
        self.max_leaf_count = int(max_leaf_count) if max_leaf_count is not None else None
        root_count = math.prod(self.l0_shape)
        if self.max_leaf_count is not None and self.max_leaf_count < root_count:
            raise ValueError("max_leaf_count cannot be smaller than the number of root cells.")

        base_coords = _all_coords(self.l0_shape)
        leaf_count = int(base_coords.shape[0])
        self.coefficient_logits = nn.Parameter(torch.zeros(leaf_count, dtype=torch.float32))
        self.register_buffer("leaf_levels", torch.zeros(leaf_count, dtype=torch.long), persistent=True)
        self.register_buffer("leaf_coords", base_coords, persistent=True)
        self.register_buffer("leaf_degrees", torch.zeros((leaf_count, 3), dtype=torch.long), persistent=True)
        self.register_buffer("coefficient_offsets", torch.arange(leaf_count + 1, dtype=torch.long), persistent=True)
        self.register_buffer("coefficient_leaf_ids", torch.arange(leaf_count, dtype=torch.long), persistent=True)
        self.register_buffer("step_geometry", torch.zeros((leaf_count, 5)), persistent=True)  # n (3), d, w
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

    def _leaf_mu(self) -> torch.Tensor:
        """Coefficient-domain leaf mean used by summaries and the viewer."""
        coefficients = self.coefficients()
        sums = coefficients.new_zeros((self.leaf_levels.shape[0],))
        sums.scatter_add_(0, self.coefficient_leaf_ids, coefficients)
        return sums / self.coefficient_counts().to(dtype=coefficients.dtype).clamp_min(1.0)

    @torch.no_grad()
    def _rebuild_packed_topology(self) -> None:
        self._topology_version = int(getattr(self, "_topology_version", 0)) + 1
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
        # softplus(coefficient_logits) is a whole-model transform; compute it
        # once per call instead of once per distinct degree group, otherwise a
        # batch spanning both p0 and p1 leaves (any post-compression model)
        # redoes it multiple times per chunk, and a full-volume/slice decode
        # multiplies that by thousands of chunks.
        physical_coefficients = self.coefficients()
        for degree_tensor in unique_degree_rows(point_degrees):
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
            if degree == LEAF_TYPES["step"]:
                g = self.step_geometry[group_leaf_ids]
                s = ((local.clamp(0.0, 1.0) - 0.5) * g[:, :3]).sum(1) - g[:, 3]
                H = torch.where(g[:, 4] > 0, torch.sigmoid(s / g[:, 4].clamp_min(1e-30)), (s > 0).to(s.dtype))
                c = physical_coefficients[self.coefficient_offsets[group_leaf_ids, None] + torch.arange(2, device=flat_points.device)]
                values[point_ids] = (1 - H) * c[:, 0] + H * c[:, 1]
                continue
            bx = bernstein_basis(degree[0], local[:, 0])
            by = bernstein_basis(degree[1], local[:, 1])
            bz = bernstein_basis(degree[2], local[:, 2])
            weights = torch.einsum("ni,nj,nk->nijk", bx, by, bz).reshape(point_ids.shape[0], -1)
            offsets = torch.arange(weights.shape[1], device=flat_points.device, dtype=torch.long)
            coefficient_ids = self.coefficient_offsets[group_leaf_ids, None] + offsets[None, :]
            coefficients = physical_coefficients[coefficient_ids]
            values[point_ids] = torch.sum(coefficients * weights, dim=1)
        return values.reshape(original_shape), leaf_ids.reshape(original_shape)

    def forward_mu(self, points: torch.Tensor) -> torch.Tensor:
        return self.forward_mu_with_leaf_ids(points)[0]

    def _parallel_ray_segments(self, ray_batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized exact cell-boundary segmentation for parallel rays."""
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
        for degree_tensor in unique_degree_rows(segment_degrees):
            degree_mask = torch.all(segment_degrees == degree_tensor, dim=1)
            segment_ids = torch.nonzero(degree_mask, as_tuple=False).reshape(-1)
            degree = tuple(int(value) for value in degree_tensor.tolist())
            if degree == LEAF_TYPES["step"]:
                leaves, iv = leaf_ids[segment_ids], intervals[segment_ids]
                u0 = self.segment_local_coords(ray_batch, ray_ids[segment_ids], leaves, iv[:, 0])
                u1 = self.segment_local_coords(ray_batch, ray_ids[segment_ids], leaves, iv[:, 1])
                g = self.step_geometry[leaves]
                length = iv[:, 1] - iv[:, 0]
                I = step_integral(g, ((u0 - 0.5) * g[:, :3]).sum(1), ((u1 - 0.5) * g[:, :3]).sum(1), length)
                c = self.coefficients()[self.coefficient_offsets[leaves, None] + torch.arange(2, device=iv.device)]
                segment_integrals[segment_ids] = c[:, 0] * (length - I) + c[:, 1] * I
                continue
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

    def segment_local_coords(self, ray_batch, ray_ids: torch.Tensor, leaf_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Leaf-local coordinates u in [0, 1]^3 of the points at ray parameter t (same convention as _parallel_ray_segments)."""
        a = -ray_batch.angles.to(device=t.device, dtype=t.dtype)[ray_ids]
        u = -1.0 + (ray_batch.cols.to(device=t.device, dtype=t.dtype)[ray_ids] + 0.5) * 2.0 / float(ray_batch.detector_w)
        z = -1.0 + (ray_batch.rows.to(device=t.device, dtype=t.dtype)[ray_ids] + 0.5) * 2.0 / float(ray_batch.detector_h)
        points = torch.stack([-torch.sin(a) * u + torch.cos(a) * t, torch.cos(a) * u + torch.sin(a) * t, z], dim=1)
        shapes = torch.tensor(self.level_shapes, dtype=t.dtype, device=t.device)[self.leaf_levels[leaf_ids]]
        return (points + 1.0) * 0.5 * shapes - self.leaf_coords[leaf_ids].to(t.dtype)

    def _can_use_cuda_integrator(self, ray_batch) -> bool:
        return self.cuda_integrator and self.coefficient_logits.is_cuda and has_bernstein_native()

    def _cuda_integrator_arguments(self, ray_batch) -> dict:
        return {
            "coefficient_logits": self.coefficient_logits,
            "leaf_degrees": self.leaf_degrees,
            "coefficient_offsets": self.coefficient_offsets,
            "step_geometry": self.step_geometry,
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
        """Sparse (ray, leaf, line-integral) tuples."""
        if self._can_use_cuda_integrator(ray_batch):
            return bernstein_octree_ray_segments(**self._cuda_integrator_arguments(ray_batch))
        return self.exact_ray_cell_contributions(ray_batch)

    def integrate_ray_batch(self, ray_batch) -> torch.Tensor:
        """Exact line integrals (CUDA traversal, or the PyTorch reference implementation on CPU)."""
        if self._can_use_cuda_integrator(ray_batch):
            return bernstein_octree_integrate(**self._cuda_integrator_arguments(ray_batch))
        ray_ids, _, contributions = self.exact_ray_cell_contributions(ray_batch)
        prediction = contributions.new_zeros((ray_batch.num_rays,))
        prediction.scatter_add_(0, ray_ids, contributions)
        return prediction

    def decoded_at_resolution(self, resolution: int | tuple[int, int, int], *, chunk: int | None = None) -> torch.Tensor:
        if isinstance(resolution, int):
            shape = (int(resolution),) * 3
        else:
            shape = tuple(int(value) for value in resolution)
        device = self.coefficient_logits.device
        scale = torch.tensor(shape, dtype=torch.float32, device=device)
        if chunk is None:
            axes = [torch.arange(value, dtype=torch.float32, device=device) for value in shape]
            xx, yy, zz = torch.meshgrid(*axes, indexing="ij")
            coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
            points = -1.0 + (coords + 0.5) * 2.0 / scale
            return self.forward_mu(points).reshape(shape)

        # Do not materialize a resolution^3 coordinate mesh before chunking.
        # At 384^3, xx/yy/zz + stacked coords + points can require several
        # GiB even though each actual query batch is small.  Decode linear
        # indices into xyz coordinates inside each batch instead.
        nx, ny, nz = shape
        total = int(nx) * int(ny) * int(nz)
        values = []
        for start in range(0, total, int(chunk)):
            stop = min(start + int(chunk), total)
            linear = torch.arange(start, stop, dtype=torch.long, device=device)
            x = torch.div(linear, int(ny) * int(nz), rounding_mode="floor")
            remainder = linear - x * int(ny) * int(nz)
            y = torch.div(remainder, int(nz), rounding_mode="floor")
            z = remainder - y * int(nz)
            coords = torch.stack([x, y, z], dim=1).to(dtype=torch.float32)
            points = -1.0 + (coords + 0.5) * 2.0 / scale
            values.append(self.forward_mu(points))
        return torch.cat(values, dim=0).reshape(shape)


    @torch.no_grad()
    def _replace_leaves(
        self,
        removed_leaf_ids: torch.Tensor,
        new_levels: torch.Tensor,
        new_coords: torch.Tensor,
        new_degrees: torch.Tensor,
        new_blocks: Sequence[torch.Tensor] | torch.Tensor,
        new_geometry: torch.Tensor | None = None,
    ) -> None:
        device = self.coefficient_logits.device
        removed_leaf_ids = torch.unique(removed_leaf_ids.to(device=device, dtype=torch.long))
        removed_leaf_ids = removed_leaf_ids[(removed_leaf_ids >= 0) & (removed_leaf_ids < self.leaf_levels.shape[0])]
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
        if new_geometry is None:
            new_geometry = torch.zeros((new_levels.shape[0], 5), device=device)
        geometry = torch.cat([self.step_geometry[keep_leaf_mask], new_geometry.to(device=device, dtype=torch.float32)], dim=0)
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
        self.step_geometry = geometry.contiguous()
        self._rebuild_packed_topology()

    @torch.no_grad()
    def split_leaves_batch(self, leaf_ids) -> int:
        """Split leaves into 8 children of the same type, function preserving:
        constant children copy the parent; trilinear children take the parent's values at their corners;
        step children keep both values with geometry n' = n, d' = 2d - n . (o - 1/2), w' = 2w (o = child offset).
        Returns the number of split leaves (capped by max_leaf_count)."""
        device = self.coefficient_logits.device
        ids = torch.unique(torch.as_tensor(leaf_ids, dtype=torch.long, device=device), sorted=True)
        ids = ids[(ids >= 0) & (ids < self.leaf_levels.shape[0])]
        if self.max_leaf_count is not None:
            ids = ids[: max(0, (int(self.max_leaf_count) - int(self.leaf_levels.shape[0])) // 7)]
        if ids.numel() == 0:
            return 0
        if torch.any(self.leaf_levels[ids] + 1 >= len(self.level_shapes)):
            raise ValueError("At least one selected leaf is already at maximum octree depth.")
        kind = type_ids(self.leaf_degrees[ids])
        ids = torch.cat([ids[kind == 0], ids[kind == 1], ids[kind == 2]])  # grouped by type
        n0, n1 = int((kind == 0).sum()), int((kind == 1).sum())
        physical = self.coefficients().detach()
        child_offsets = _all_coords(2, device=device)
        constant_children = physical[self.coefficient_offsets[ids[:n0]]].repeat_interleave(8)
        parents = physical[self.coefficient_offsets[ids[n0 : n0 + n1], None] + torch.arange(8, device=device)].reshape(-1, 2, 2, 2)
        half = parents.new_tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])  # corner values at 0, 1/2, 1 per axis
        lattice = torch.einsum("ai,bj,ck,nijk->nabc", half, half, half, parents)
        trilinear_children = torch.stack(
            [lattice[:, x : x + 2, y : y + 2, z : z + 2] for x, y, z in child_offsets.tolist()], dim=1
        ).reshape(-1)
        step_ids = ids[n0 + n1 :]
        step_children = physical[self.coefficient_offsets[step_ids, None] + torch.arange(2, device=device)][:, None, :].expand(-1, 8, -1).reshape(-1)
        geometry = torch.zeros((ids.numel(), 8, 5), device=device)
        g = self.step_geometry[step_ids][:, None, :]
        shift = (g[..., :3] * (child_offsets[None].float() - 0.5)).sum(-1)
        geometry[n0 + n1 :] = torch.cat([g[..., :3].expand(-1, 8, -1), (2 * g[..., 3] - shift)[..., None], (2 * g[..., 4]).expand(-1, 8)[..., None]], -1)
        self._replace_leaves(
            ids,
            (self.leaf_levels[ids][:, None] + 1).expand(-1, 8).reshape(-1),
            (self.leaf_coords[ids][:, None, :] * 2 + child_offsets[None]).reshape(-1, 3),
            self.leaf_degrees[ids][:, None, :].expand(-1, 8, -1).reshape(-1, 3),
            torch.cat([constant_children, trilinear_children, step_children]),
            geometry.reshape(-1, 5),
        )
        return int(ids.numel())

    @torch.no_grad()
    def set_types(self, leaf_ids, types: torch.Tensor, values: torch.Tensor, geometry: torch.Tensor) -> None:
        """Replace leaves in place (same cell) by leaves of type ``types`` (0 constant, 1 trilinear, 2 step) with
        the given physical values (concatenated in leaf order) and step geometry (rows used only by step leaves)."""
        device = self.coefficient_logits.device
        ids = torch.as_tensor(leaf_ids, dtype=torch.long, device=device)
        types = torch.as_tensor(types, dtype=torch.long, device=device)
        if ids.numel() == 0:
            return
        allowed = torch.tensor([name in self.leaf_types for name in TYPE_NAMES], device=device)
        if not bool(allowed[types].all()):
            raise ValueError(f"leaf types outside {self.leaf_types}")
        degrees = torch.tensor([LEAF_TYPES[name] for name in TYPE_NAMES], device=device)[types]
        self._replace_leaves(ids, self.leaf_levels[ids], self.leaf_coords[ids], degrees, values, geometry)

    @torch.no_grad()
    def to_trilinear(self, leaf_ids) -> None:
        """Turn constant leaves into trilinear leaves with all 8 corners equal to the constant (same function)."""
        device = self.coefficient_logits.device
        ids = torch.unique(torch.as_tensor(leaf_ids, dtype=torch.long, device=device))
        ids = ids[type_ids(self.leaf_degrees[ids]) == 0]
        values = self.coefficients().detach()[self.coefficient_offsets[ids]]
        self.set_types(ids, torch.ones_like(ids), values[:, None].expand(-1, 8).reshape(-1), torch.zeros((ids.numel(), 5), device=device))

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
            leaf_types=self.leaf_types,
            leaf_type_counts=tuple(torch.bincount(type_ids(self.leaf_degrees), minlength=3).tolist()),
        )

    def prepare_sparse_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        required = {
            "coefficient_logits",
            "leaf_levels",
            "leaf_coords",
            "leaf_degrees",
            "coefficient_offsets",
            "coefficient_leaf_ids",
            "step_geometry",
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
