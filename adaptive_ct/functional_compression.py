"""Detail-preserving functional subtree compression (pipeline v5).

Given a frozen h-only reference field `mu_ref` (steps 1-3 of the plan: an
existing h-adaptive reconstruction that already has enough spatial resolution
for cracks, pores, thin walls and material interfaces), this module adds only
the new step: compress every internal subtree that does *not* need that
resolution into a single non-negative Bernstein leaf, either

  - p0 (degree 0,0,0): a constant, for uniform/air regions, or
  - p1 (degree 1,1,1): a trilinear block, for smoothly varying regions,

while leaving subtrees with real internal structure (material boundaries,
cracks, pores) untouched. The reconstruction framework itself is not
rewritten: this only adds function-preserving-at-the-leaf-level compression
on top of an already-trained BernsteinOctree checkpoint.

Pipeline correspondence:
  step 4 (bottom-up candidates)      -> build_candidate_bank
  step 5 (projection-residual distortion) -> score_projection_candidates
  step 6 (global R-D tree selection) -> run_compression (BFOS/CLG bisection)
  step 7 (fixed-topology joint refit)-> materialize_compressed_model
                                        + finetune_fixed_topology
  step 9 (compact export)            -> adaptive_ct.compression (reused, not
                                          duplicated: the packed_hierarchy_v3
                                          schema already supports mixed p0/p1
                                          leaves)

Nothing here touches `mu_ref` or the h-only training loop; it only reads the
frozen model to build merge candidates and, once a tree is selected, produces
a brand new independently-trainable BernsteinOctree.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .bernstein import BernsteinOctree, bernstein_basis
from .compression import export_compact_octree_artifact
from .config import load_config
from .data import ProjectionSplit, load_r2_dataset
from .geometry import detector_grid, make_parallel_ray_points, random_training_rays
from .metrics import projection_metrics
from .model import build_model
from .render import render_split

# compact_bernstein_octree_packed_v3 byte accounting (adaptive_ct/compression.py):
# every node (leaf or internal) occupies one int32 node_child_base + one int32
# node_leaf_id slot; every leaf additionally stores uint8 x3 leaf_degrees and
# float16 coefficients.
NODE_SLOT_BYTES = 8
LEAF_DEGREE_BYTES = 3
COEFF_BYTES = 2
P1_DEGREE = (1, 1, 1)
P1_COEFF_COUNT = 8
P0_BYTES = float(NODE_SLOT_BYTES + LEAF_DEGREE_BYTES + COEFF_BYTES * 1)
P1_BYTES = float(NODE_SLOT_BYTES + LEAF_DEGREE_BYTES + COEFF_BYTES * P1_COEFF_COUNT)
METADATA_JSON_BYTES = 4096

KEEP_ACTION = 0
P0_ACTION = 1
P1_ACTION = 2


def _packed_v3_fixed_payload_bytes(model: BernsteinOctree) -> int:
    """Raw compact-v3 bytes independent of the chosen tree.

    The R-D pass controls node slots, degree tags, and coefficients.  Compact
    v3 also writes a small fixed header; subtract it before tree selection so
    the exported raw payload, not only its variable part, respects R_max.
    """
    level_count = len(model.level_shapes)
    level_shapes_bytes = level_count * 3 * 2  # uint16[level, xyz]
    level_resolutions_bytes = (
        level_count * 2
        if all(nx == ny == nz for nx, ny, nz in model.level_shapes)
        else 0
    )
    return (
        4  # uint32 leaf_count
        + level_shapes_bytes
        + level_resolutions_bytes
        + len(model.max_degree)  # uint8 max_degree[xyz]
        + 4  # float32 attenuation_shift
        + METADATA_JSON_BYTES
    )


@dataclass
class CandidateBank:
    """Per-level p0/p1 merge candidates for every internal node, sampled once
    against the frozen reference field, plus the reused interface-jump detail
    energy of everything below that node."""

    linear: list[torch.Tensor]
    p0_value: list[torch.Tensor]
    p0_value_distortion: list[torch.Tensor]
    p0_detail_distortion: list[torch.Tensor]
    p1_coeffs: list[torch.Tensor]
    p1_value_distortion: list[torch.Tensor]
    p1_detail_distortion: list[torch.Tensor]


def load_reference_model(
    config_path: str | Path, checkpoint_path: str | Path, device: torch.device
) -> tuple[dict, BernsteinOctree]:
    config = load_config(Path(config_path))
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model_config = payload.get("config", config)
    model = build_model(model_config).to(device=device)
    model.prepare_sparse_from_state_dict(payload["model"])
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    if not isinstance(model, BernsteinOctree):
        raise ValueError("Functional compression requires a BernsteinOctree checkpoint.")
    return config, model


def _decode_linear(linear: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    _, ny, nz = shape
    x = torch.div(linear, ny * nz, rounding_mode="floor")
    remainder = linear - x * ny * nz
    y = torch.div(remainder, nz, rounding_mode="floor")
    z = remainder - y * nz
    return torch.stack([x, y, z], dim=1)


def _child_offsets(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[ox, oy, oz] for ox in (0, 1) for oy in (0, 1) for oz in (0, 1)],
        dtype=torch.long,
        device=device,
    )


@torch.no_grad()
def _internal_coords_by_level(model: BernsteinOctree) -> list[torch.Tensor]:
    """Unique linear coordinates of nodes whose subtree extends past `level`.

    Mirrors the internal-node bookkeeping in
    `BernsteinOctree._rebuild_packed_topology`, recomputed here read-only.
    """
    device = model.coefficient_logits.device
    result: list[torch.Tensor] = [torch.empty((0,), dtype=torch.long, device=device)] * len(model.level_shapes)
    for level in range(len(model.level_shapes) - 1, -1, -1):
        _, ny, nz = model.level_shapes[level]
        deeper = model.leaf_levels > level
        if not torch.any(deeper):
            result[level] = torch.empty((0,), dtype=torch.long, device=device)
            continue
        deeper_levels = model.leaf_levels[deeper]
        deeper_coords = model.leaf_coords[deeper]
        divisors = torch.pow(torch.full_like(deeper_levels, 2), deeper_levels - level)[:, None]
        internal_coords = torch.div(deeper_coords, divisors, rounding_mode="floor")
        linear = (internal_coords[:, 0] * ny + internal_coords[:, 1]) * nz + internal_coords[:, 2]
        result[level] = torch.unique(linear, sorted=True)
    return result


def _bernstein_p1_design(
    samples_per_axis: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = (torch.arange(samples_per_axis, device=device, dtype=dtype) + 0.5) / float(samples_per_axis)
    xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
    local = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    bx = bernstein_basis(1, local[:, 0])
    by = bernstein_basis(1, local[:, 1])
    bz = bernstein_basis(1, local[:, 2])
    design = torch.einsum("si,sj,sk->sijk", bx, by, bz).reshape(local.shape[0], P1_COEFF_COUNT)
    pinv = torch.linalg.pinv(design)
    return local, design, pinv


def _local_detail_distortion(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    samples_per_axis: int,
) -> torch.Tensor:
    """Candidate-specific detail loss on a common local sampling grid.

    Finite differences are derivatives with respect to local cell coordinates
    u/v/w in [0, 1].  A p0 candidate therefore pays for every smooth slope it
    removes, while a p1 candidate pays only for variation it cannot reproduce.
    A discontinuity leaves a large residual for both and is consequently kept
    as an h-subtree.  This is a compression distortion only; it imposes no
    continuity between neighboring leaves and is not a training regularizer.
    """
    count = int(samples_per_axis)
    if count < 2:
        raise ValueError("samples_per_axis must be at least 2 for detail-preserving compression")
    residual = (reference - prediction).reshape(-1, count, count, count)
    scale = float(count - 1)
    detail = residual.new_zeros((residual.shape[0],))
    for axis in range(1, 4):
        derivative = torch.diff(residual, dim=axis) * scale
        detail = detail + torch.mean(torch.square(derivative), dim=(1, 2, 3))
    return detail


@torch.no_grad()
def build_candidate_bank(
    model: BernsteinOctree, *, samples_per_axis: int = 4, node_chunk: int = 4096
) -> CandidateBank:
    """Sample the frozen field inside every internal node's full physical
    domain (regardless of how many levels of real structure lie beneath it)
    and fit the p0 mean (step 4.B) and the p1 least-squares Bernstein block
    (step 4.C) against it.  Value and detail distortion are both evaluated
    against each candidate, so p1 receives credit for smooth variation it
    actually explains instead of sharing p0's constant-field penalty.
    """
    device = model.coefficient_logits.device
    dtype = torch.float32
    local, design, pinv = _bernstein_p1_design(samples_per_axis, device, dtype)
    sample_count = int(local.shape[0])
    if int(node_chunk) <= 0:
        raise ValueError("node_chunk must be positive after workspace-based resolution.")
    internal_by_level = _internal_coords_by_level(model)
    reference_is_p0 = bool(torch.all(model.leaf_degrees == 0).item())
    reference_coefficients = model.coefficients().detach() if reference_is_p0 else None

    def evaluate_reference(points: torch.Tensor) -> torch.Tensor:
        if reference_coefficients is None:
            return model.forward_mu(points)
        # The formal reference checkpoints are h-only p0 trees.  Avoid the
        # mixed-degree basis construction in forward_mu: packed traversal plus
        # one coefficient gather is the exact same field and is materially
        # faster for the tens of millions of candidate samples.
        leaf_ids = model.resolve_leaf_ids(points).reshape(-1)
        if torch.any(leaf_ids < 0):
            raise RuntimeError("Candidate sampling unexpectedly left the reconstruction box.")
        coefficient_ids = model.coefficient_offsets[leaf_ids]
        return reference_coefficients[coefficient_ids].reshape(points.shape[:-1])

    bank_linear: list[torch.Tensor] = []
    bank_p0_value: list[torch.Tensor] = []
    bank_p0_distortion: list[torch.Tensor] = []
    bank_p0_detail: list[torch.Tensor] = []
    bank_p1_coeffs: list[torch.Tensor] = []
    bank_p1_distortion: list[torch.Tensor] = []
    bank_p1_detail: list[torch.Tensor] = []

    for level in range(len(model.level_shapes) - 1):
        linear = internal_by_level[level]
        bank_linear.append(linear)
        if linear.numel() == 0:
            bank_p0_value.append(torch.empty((0,), dtype=dtype, device=device))
            bank_p0_distortion.append(torch.empty((0,), dtype=dtype, device=device))
            bank_p0_detail.append(torch.empty((0,), dtype=dtype, device=device))
            bank_p1_coeffs.append(torch.empty((0, P1_COEFF_COUNT), dtype=dtype, device=device))
            bank_p1_distortion.append(torch.empty((0,), dtype=dtype, device=device))
            bank_p1_detail.append(torch.empty((0,), dtype=dtype, device=device))
            continue

        coords = _decode_linear(linear, model.level_shapes[level])
        shape = torch.tensor(model.level_shapes[level], dtype=dtype, device=device)
        cell = 2.0 / shape
        lo = -1.0 + coords.to(dtype=dtype) * cell[None, :]
        cell_volume = float((cell[0] * cell[1] * cell[2]).item())

        p0_values = torch.empty((linear.shape[0],), dtype=dtype, device=device)
        p0_dist = torch.empty_like(p0_values)
        p0_detail = torch.empty_like(p0_values)
        p1_coeffs = torch.empty((linear.shape[0], P1_COEFF_COUNT), dtype=dtype, device=device)
        p1_dist = torch.empty_like(p0_values)
        p1_detail = torch.empty_like(p0_values)

        for start in range(0, int(linear.shape[0]), int(node_chunk)):
            stop = min(start + int(node_chunk), int(linear.shape[0]))
            chunk_lo = lo[start:stop]
            points = chunk_lo[:, None, :] + local[None, :, :] * cell[None, None, :]
            values = evaluate_reference(points.reshape(-1, 3)).reshape(stop - start, sample_count)

            mean = values.mean(dim=1)
            p0_prediction = mean[:, None].expand_as(values)
            p0_values[start:stop] = mean
            p0_dist[start:stop] = torch.mean(torch.square(values - p0_prediction), dim=1)
            p0_detail[start:stop] = _local_detail_distortion(
                values,
                p0_prediction,
                samples_per_axis=samples_per_axis,
            )

            coeffs = (values @ pinv.T).clamp_min(1e-12)
            p1_coeffs[start:stop] = coeffs
            predicted = coeffs @ design.T
            p1_dist[start:stop] = torch.mean(torch.square(values - predicted), dim=1)
            p1_detail[start:stop] = _local_detail_distortion(
                values,
                predicted,
                samples_per_axis=samples_per_axis,
            )

        bank_p0_value.append(p0_values)
        bank_p0_distortion.append(p0_dist * cell_volume)
        bank_p0_detail.append(p0_detail * cell_volume)
        bank_p1_coeffs.append(p1_coeffs)
        bank_p1_distortion.append(p1_dist * cell_volume)
        bank_p1_detail.append(p1_detail * cell_volume)

    return CandidateBank(
        bank_linear,
        bank_p0_value,
        bank_p0_distortion,
        bank_p0_detail,
        bank_p1_coeffs,
        bank_p1_distortion,
        bank_p1_detail,
    )


@torch.no_grad()
def _candidate_cell_integrals(
    model: BernsteinOctree,
    ray_batch,
    ray_ids: torch.Tensor,
    *,
    level: int,
    coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact p0 lengths and p1 corner-basis integrals through candidate cells."""
    device = model.coefficient_logits.device
    dtype = torch.float32
    ray_ids = ray_ids.to(device=device, dtype=torch.long)
    coords = coords.to(device=device, dtype=dtype)
    if ray_ids.numel() == 0:
        return (
            torch.empty((0,), dtype=dtype, device=device),
            torch.empty((0, P1_COEFF_COUNT), dtype=dtype, device=device),
        )
    if ray_batch.angles is None or ray_batch.rows is None or ray_batch.cols is None:
        raise ValueError("Projection candidate scoring requires parallel-ray metadata.")

    shape = torch.tensor(model.level_shapes[int(level)], dtype=dtype, device=device)
    cell = 2.0 / shape
    lower = -1.0 + coords * cell
    upper = lower + cell

    effective_angles = -ray_batch.angles.to(device=device, dtype=dtype)[ray_ids]
    ca = torch.cos(effective_angles)
    sa = torch.sin(effective_angles)
    directions = torch.stack([ca, sa], dim=1)
    u_axes = torch.stack([-sa, ca], dim=1)
    cols = ray_batch.cols.to(device=device, dtype=dtype)[ray_ids]
    rows = ray_batch.rows.to(device=device, dtype=dtype)[ray_ids]
    u = -1.0 + (cols + 0.5) * 2.0 / float(ray_batch.detector_w)
    z_world = -1.0 + (rows + 0.5) * 2.0 / float(ray_batch.detector_h)
    bases = u_axes * u[:, None]

    eps = 1.0e-8
    t_min = torch.full_like(z_world, -1.0e20)
    t_max = torch.full_like(z_world, 1.0e20)
    valid = (z_world >= lower[:, 2] - eps) & (z_world <= upper[:, 2] + eps)
    for axis in range(2):
        direction = directions[:, axis]
        base = bases[:, axis]
        parallel = direction.abs() < eps
        valid &= (~parallel) | ((base >= lower[:, axis] - eps) & (base <= upper[:, axis] + eps))
        denominator = torch.where(parallel, torch.ones_like(direction), direction)
        t0 = (lower[:, axis] - base) / denominator
        t1 = (upper[:, axis] - base) / denominator
        t_min = torch.where(parallel, t_min, torch.maximum(t_min, torch.minimum(t0, t1)))
        t_max = torch.where(parallel, t_max, torch.minimum(t_max, torch.maximum(t0, t1)))
    valid &= t_max > t_min + eps
    length = torch.where(valid, t_max - t_min, torch.zeros_like(t_min))

    half = 0.5 * length
    centre = 0.5 * (t_max + t_min)
    nodes = torch.tensor([-0.5773502691896257, 0.5773502691896257], dtype=dtype, device=device)
    t = centre[:, None] + half[:, None] * nodes[None, :]
    x = bases[:, 0, None] + directions[:, 0, None] * t
    y = bases[:, 1, None] + directions[:, 1, None] * t
    local_x = ((x - lower[:, 0, None]) / cell[0]).clamp(0.0, 1.0)
    local_y = ((y - lower[:, 1, None]) / cell[1]).clamp(0.0, 1.0)
    local_z = ((z_world - lower[:, 2]) / cell[2]).clamp(0.0, 1.0)
    bx = torch.stack([1.0 - local_x, local_x], dim=2)
    by = torch.stack([1.0 - local_y, local_y], dim=2)
    bz = torch.stack([1.0 - local_z, local_z], dim=1)
    corner_integrals = (
        half[:, None, None, None] * torch.einsum("nqi,nqj,nk->nijk", bx, by, bz)
    ).reshape(-1, P1_COEFF_COUNT)
    corner_integrals = torch.where(valid[:, None], corner_integrals, torch.zeros_like(corner_integrals))
    return length, corner_integrals


@torch.no_grad()
def score_projection_candidates(
    model: BernsteinOctree,
    bank: CandidateBank,
    split: ProjectionSplit,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    ray_chunk: int = 8192,
    max_views: int | None = None,
    rays_per_view: int | None = 8192,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    """Predict the projection-loss change of replacing each subtree.

    For candidate K and sampled ray r, this accumulates
        2 * residual_r * delta_p_rK + delta_p_rK**2
    where delta_p is the exact p0/p1 candidate integral minus the reference
    subtree integral. Cross-candidate terms are intentionally omitted so the
    resulting costs remain additive for the tree DP. Only the training split
    is accepted by the public pipeline.
    """
    device = model.coefficient_logits.device
    p0_scores = [torch.zeros_like(values) for values in bank.p0_value_distortion]
    p1_scores = [torch.zeros_like(values) for values in bank.p1_value_distortion]
    rows_all, cols_all = detector_grid(int(detector_h), int(detector_w), device)
    if rays_per_view is not None and int(rays_per_view) < int(rows_all.numel()):
        sample_ids = torch.linspace(
            0,
            rows_all.numel() - 1,
            max(1, int(rays_per_view)),
            device=device,
        ).round().long().unique(sorted=True)
        rows_all = rows_all[sample_ids]
        cols_all = cols_all[sample_ids]
    view_count = int(split.angles.shape[0])
    if max_views is None or int(max_views) >= view_count:
        view_ids = torch.arange(view_count, device=device)
    else:
        view_ids = torch.linspace(
            0,
            view_count - 1,
            max(1, int(max_views)),
            device=device,
        ).round().long().unique(sorted=True)

    sampled_rays = 0
    for view_id_tensor in view_ids:
        view_id = int(view_id_tensor.item())
        angle = split.angles[view_id]
        for start in range(0, int(rows_all.numel()), max(1, int(ray_chunk))):
            rows = rows_all[start : start + int(ray_chunk)]
            cols = cols_all[start : start + int(ray_chunk)]
            targets = split.projections[view_id, rows, cols]
            ray_batch = make_parallel_ray_points(
                angle.expand(rows.shape[0]),
                rows,
                cols,
                detector_h=int(detector_h),
                detector_w=int(detector_w),
                samples_per_ray=int(samples_per_ray),
                targets=targets,
                materialize_points=not model.prefer_compact_ray_batch(),
            )
            segment_rays, segment_leaves, contributions = model.ray_cell_contributions(ray_batch)
            prediction = contributions.new_zeros((ray_batch.num_rays,))
            prediction.scatter_add_(0, segment_rays, contributions)
            residual = prediction - targets

            leaf_levels = model.leaf_levels[segment_leaves]
            leaf_coords = model.leaf_coords[segment_leaves]
            for level in range(len(bank.linear)):
                candidate_count = int(bank.linear[level].numel())
                if candidate_count == 0:
                    continue
                eligible = leaf_levels > int(level)
                if not torch.any(eligible):
                    continue
                selected_rays = segment_rays[eligible]
                selected_levels = leaf_levels[eligible]
                selected_coords = leaf_coords[eligible]
                divisors = torch.pow(
                    torch.full_like(selected_levels, 2),
                    selected_levels - int(level),
                )[:, None]
                ancestor_coords = torch.div(selected_coords, divisors, rounding_mode="floor")
                _, ny, nz = model.level_shapes[level]
                ancestor_linear = (ancestor_coords[:, 0] * ny + ancestor_coords[:, 1]) * nz + ancestor_coords[:, 2]
                candidate_ids = torch.searchsorted(bank.linear[level], ancestor_linear)
                found = (candidate_ids < candidate_count) & (
                    bank.linear[level][candidate_ids.clamp_max(candidate_count - 1)] == ancestor_linear
                )
                if not torch.any(found):
                    continue
                selected_rays = selected_rays[found]
                candidate_ids = candidate_ids[found]
                selected_contributions = contributions[eligible][found]
                keys = selected_rays * candidate_count + candidate_ids
                unique_keys, inverse = torch.unique(keys, return_inverse=True)
                reference_integral = selected_contributions.new_zeros((unique_keys.numel(),))
                reference_integral.scatter_add_(0, inverse, selected_contributions)
                unique_rays = torch.div(unique_keys, candidate_count, rounding_mode="floor")
                unique_candidates = torch.remainder(unique_keys, candidate_count)
                candidate_coords = _decode_linear(bank.linear[level][unique_candidates], model.level_shapes[level])
                lengths, corner_integrals = _candidate_cell_integrals(
                    model,
                    ray_batch,
                    unique_rays,
                    level=level,
                    coords=candidate_coords,
                )
                p0_integral = lengths * bank.p0_value[level][unique_candidates]
                p1_integral = torch.sum(
                    corner_integrals * bank.p1_coeffs[level][unique_candidates],
                    dim=1,
                )
                p0_delta = p0_integral - reference_integral
                p1_delta = p1_integral - reference_integral
                selected_residual = residual[unique_rays]
                p0_change = 2.0 * selected_residual * p0_delta + p0_delta.square()
                p1_change = 2.0 * selected_residual * p1_delta + p1_delta.square()
                p0_scores[level].scatter_add_(0, unique_candidates, p0_change)
                p1_scores[level].scatter_add_(0, unique_candidates, p1_change)
            sampled_rays += int(targets.numel())

    normalizer = float(max(sampled_rays, 1))
    p0_scores = [values / normalizer for values in p0_scores]
    p1_scores = [values / normalizer for values in p1_scores]
    return p0_scores, p1_scores, {
        "domain": "training_projection_residual",
        "views": int(view_ids.numel()),
        "rays_per_view": int(rows_all.numel()),
        "sampled_rays": sampled_rays,
        "cross_candidate_terms": "omitted_for_additive_tree_dp",
        "p0_negative_candidates": sum(int(torch.sum(values < 0.0).item()) for values in p0_scores),
        "p1_negative_candidates": sum(int(torch.sum(values < 0.0).item()) for values in p1_scores),
    }


@torch.no_grad()
def _leaf_bytes_by_level(model: BernsteinOctree) -> list[torch.Tensor]:
    counts_all = model.coefficient_counts()
    result = []
    for level in range(len(model.level_shapes)):
        _, leaf_ids = model._level_leaf_lookup(level)
        bytes_ = float(NODE_SLOT_BYTES + LEAF_DEGREE_BYTES) + float(COEFF_BYTES) * counts_all[leaf_ids].to(torch.float32)
        result.append(bytes_)
    return result


@torch.no_grad()
def _run_lagrangian_dp(
    model: BernsteinOctree,
    bank: CandidateBank,
    leaf_bytes_by_level: list[torch.Tensor],
    lam: float,
    gamma: float,
    allow_p0: bool = True,
) -> tuple[float, float, list[torch.Tensor]]:
    """Bottom-up BFOS/CLG pass for one lambda, choosing among
    {keep, p0, p1} at every internal node under
    D_K = D_value + gamma * D_detail, cost = D_K + lambda * bytes.
    Returns (total raw packed bytes, total D_K, per-level chosen action)."""
    device = model.coefficient_logits.device
    n_levels = len(model.level_shapes)

    resolved_linear: list[torch.Tensor] = [None] * n_levels  # type: ignore[list-item]
    resolved_distortion: list[torch.Tensor] = [None] * n_levels  # type: ignore[list-item]
    resolved_bytes: list[torch.Tensor] = [None] * n_levels  # type: ignore[list-item]
    actions: list[torch.Tensor] = [torch.empty((0,), dtype=torch.long, device=device) for _ in range(n_levels)]

    deepest = n_levels - 1
    linear, _leaf_ids = model._level_leaf_lookup(deepest)
    bytes_ = leaf_bytes_by_level[deepest]
    resolved_linear[deepest] = linear
    resolved_bytes[deepest] = bytes_
    resolved_distortion[deepest] = torch.zeros_like(bytes_)

    for level in range(n_levels - 2, -1, -1):
        leaf_linear, _leaf_ids = model._level_leaf_lookup(level)
        leaf_bytes = leaf_bytes_by_level[level]
        leaf_dist = torch.zeros_like(leaf_bytes)

        bank_linear = bank.linear[level]
        if bank_linear.numel() > 0:
            _, ny, nz = model.level_shapes[level + 1]
            coords = _decode_linear(bank_linear, model.level_shapes[level])
            offsets = _child_offsets(device)
            children = (coords[:, None, :] * 2 + offsets[None, :, :]).reshape(-1, 3)
            child_linear = ((children[:, 0] * ny + children[:, 1]) * nz + children[:, 2]).reshape(-1, 8)

            child_table = resolved_linear[level + 1]
            pos = torch.searchsorted(child_table, child_linear.reshape(-1))
            found_bytes = resolved_bytes[level + 1][pos].reshape(child_linear.shape)
            found_dist = resolved_distortion[level + 1][pos].reshape(child_linear.shape)

            keep_bytes = found_bytes.sum(dim=1) + float(NODE_SLOT_BYTES)
            keep_dist = found_dist.sum(dim=1)
            keep_cost = keep_dist + float(lam) * keep_bytes

            p0_dist = bank.p0_value_distortion[level] + float(gamma) * bank.p0_detail_distortion[level]
            p1_dist = bank.p1_value_distortion[level] + float(gamma) * bank.p1_detail_distortion[level]
            p0_cost = p0_dist + float(lam) * P0_BYTES
            if not allow_p0:
                p0_cost = torch.full_like(p0_cost, float("inf"))
            p1_cost = p1_dist + float(lam) * P1_BYTES

            stacked_cost = torch.stack([keep_cost, p0_cost, p1_cost], dim=1)
            bytes_candidates = torch.stack(
                [keep_bytes, torch.full_like(keep_bytes, P0_BYTES), torch.full_like(keep_bytes, P1_BYTES)], dim=1
            )
            dist_candidates = torch.stack([keep_dist, p0_dist, p1_dist], dim=1)

            best = torch.argmin(stacked_cost, dim=1)
            chosen_bytes = bytes_candidates.gather(1, best[:, None])[:, 0]
            chosen_dist = dist_candidates.gather(1, best[:, None])[:, 0]
            actions[level] = best

            level_linear = torch.cat([leaf_linear, bank_linear])
            level_dist = torch.cat([leaf_dist, chosen_dist])
            level_bytes = torch.cat([leaf_bytes, chosen_bytes])
        else:
            level_linear = leaf_linear
            level_dist = leaf_dist
            level_bytes = leaf_bytes

        order = torch.argsort(level_linear)
        resolved_linear[level] = level_linear[order]
        resolved_distortion[level] = level_dist[order]
        resolved_bytes[level] = level_bytes[order]

    total_bytes = float(resolved_bytes[0].sum().item())
    total_distortion = float(resolved_distortion[0].sum().item())
    return total_bytes, total_distortion, actions


@torch.no_grad()
def run_compression(
    model: BernsteinOctree,
    bank: CandidateBank,
    leaf_bytes_by_level: list[torch.Tensor],
    *,
    gamma: float,
    r_max_bytes: float,
    iterations: int = 40,
    allow_p0: bool = True,
) -> tuple[float, float, list[torch.Tensor], float]:
    """Step 6: bisect lambda so R(T) <= R_max (raw packed bytes), selecting
    {keep, p0, p1} per subtree under D_K = D_value + gamma*D_detail.
    Returns (bytes, total D_K, actions, lambda)."""

    def evaluate(lam: float):
        return _run_lagrangian_dp(
            model,
            bank,
            leaf_bytes_by_level,
            lam,
            gamma,
            allow_p0=allow_p0,
        )

    bytes_at_zero, distortion_at_zero, actions_at_zero = evaluate(0.0)
    if bytes_at_zero <= r_max_bytes:
        return bytes_at_zero, distortion_at_zero, actions_at_zero, 0.0

    lo, hi = 0.0, 1.0e-6
    bytes_hi, _, _ = evaluate(hi)
    expansions = 0
    while bytes_hi > r_max_bytes and expansions < 80:
        hi *= 2.0
        bytes_hi, _, _ = evaluate(hi)
        expansions += 1
    if bytes_hi > r_max_bytes:
        raise RuntimeError(
            f"Could not bracket lambda to reach the {r_max_bytes:.0f}-byte budget "
            f"(smallest achieved: {bytes_hi:.0f} bytes)."
        )

    hi_distortion, hi_actions = evaluate(hi)[1:]
    best_bytes, best_distortion, best_actions, best_lambda = (bytes_hi, hi_distortion, hi_actions, hi)
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        mid_bytes, mid_distortion, mid_actions = evaluate(mid)
        if mid_bytes > r_max_bytes:
            lo = mid
        else:
            hi = mid
            best_bytes, best_distortion, best_actions, best_lambda = mid_bytes, mid_distortion, mid_actions, mid
    return best_bytes, best_distortion, best_actions, best_lambda


@torch.no_grad()
def evaluate_pruned_field(
    model: BernsteinOctree,
    bank: CandidateBank,
    actions: list[torch.Tensor],
    points: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the DP-pruned tree at arbitrary points without mutating
    `model`. Points whose root-to-leaf path never enters an internal bank
    node fall straight through to the untouched original field. Used for
    quick diagnostics; the real artifact is `materialize_compressed_model`."""
    device = points.device
    dtype = torch.float32
    original_shape = points.shape[:-1]
    values = torch.zeros((int(points.reshape(-1, 3).shape[0]),), dtype=dtype, device=device)

    active_ids = torch.arange(values.shape[0], device=device)
    active_points = points.reshape(-1, 3).to(dtype=dtype)

    for level in range(len(model.level_shapes)):
        if active_ids.numel() == 0:
            break
        shape_t = torch.tensor(model.level_shapes[level], dtype=dtype, device=device)
        cell = 2.0 / shape_t
        coords = torch.floor((active_points + 1.0) * 0.5 * shape_t).long()
        coords = torch.minimum(torch.maximum(coords, torch.zeros_like(coords)), shape_t.long() - 1)
        _, ny, nz = model.level_shapes[level]
        linear = (coords[:, 0] * ny + coords[:, 1]) * nz + coords[:, 2]

        table = bank.linear[level] if level < len(bank.linear) else torch.empty((0,), dtype=torch.long, device=device)
        if table.numel() > 0:
            pos = torch.clamp(torch.searchsorted(table, linear), max=table.numel() - 1)
            is_internal = table[pos] == linear
        else:
            is_internal = torch.zeros_like(linear, dtype=torch.bool)
            pos = linear

        leaf_mask = ~is_internal
        if torch.any(leaf_mask):
            values[active_ids[leaf_mask]] = model.forward_mu(active_points[leaf_mask])
        if not torch.any(is_internal):
            active_ids = active_ids[:0]
            continue

        internal_local = torch.nonzero(is_internal, as_tuple=False).reshape(-1)
        internal_global_ids = active_ids[internal_local]
        internal_points = active_points[internal_local]
        internal_coords = coords[internal_local]
        internal_pos = pos[internal_local]
        action = actions[level][internal_pos]

        p0_mask = action == P0_ACTION
        p1_mask = action == P1_ACTION
        keep_mask = action == KEEP_ACTION

        if torch.any(p0_mask):
            values[internal_global_ids[p0_mask]] = bank.p0_value[level][internal_pos[p0_mask]]
        if torch.any(p1_mask):
            lo_pts = -1.0 + internal_coords[p1_mask].to(dtype=dtype) * cell[None, :]
            local = (internal_points[p1_mask] - lo_pts) / cell[None, :]
            bx = bernstein_basis(1, local[:, 0])
            by = bernstein_basis(1, local[:, 1])
            bz = bernstein_basis(1, local[:, 2])
            weights = torch.einsum("ni,nj,nk->nijk", bx, by, bz).reshape(-1, P1_COEFF_COUNT)
            coeffs = bank.p1_coeffs[level][internal_pos[p1_mask]]
            values[internal_global_ids[p1_mask]] = torch.sum(weights * coeffs, dim=1)

        active_ids = internal_global_ids[keep_mask]
        active_points = internal_points[keep_mask]

    return values.reshape(original_shape)


@torch.no_grad()
def materialize_compressed_model(
    model: BernsteinOctree,
    bank: CandidateBank,
    actions: list[torch.Tensor],
) -> BernsteinOctree:
    """Step 6/7 handoff: build a brand-new, independently-trainable
    BernsteinOctree whose topology is the DP-chosen compressed tree.

    Original leaves untouched by any merge are copied verbatim (same level,
    coord, degree, coefficient). Every internal node that chose p0/p1 becomes
    one new leaf at that node's own level with the fitted coefficients from
    `bank`. `model` is never modified.

    Reference leaves may be p0, p1, or mixed degree. Untouched leaf functions
    are copied exactly; a selected merge uses the fitted parent candidate.
    """
    device = model.coefficient_logits.device
    physical = model.coefficients().detach()

    new_levels: list[torch.Tensor] = []
    new_coords: list[torch.Tensor] = []
    new_degrees: list[torch.Tensor] = []
    new_coeffs: list[torch.Tensor] = []

    def _emit(level: int, coords: torch.Tensor, degrees: torch.Tensor, coeffs: torch.Tensor) -> None:
        n = int(coords.shape[0])
        if n == 0:
            return
        new_levels.append(torch.full((n,), level, dtype=torch.long, device=device))
        new_coords.append(coords)
        new_degrees.append(degrees)
        new_coeffs.append(coeffs)

    active_coords = _decode_linear(
        torch.arange(int(torch.prod(torch.tensor(model.l0_shape))), device=device), model.l0_shape
    )
    for level in range(len(model.level_shapes)):
        _, ny, nz = model.level_shapes[level]
        linear = (active_coords[:, 0] * ny + active_coords[:, 1]) * nz + active_coords[:, 2]

        table = bank.linear[level] if level < len(bank.linear) else torch.empty((0,), dtype=torch.long, device=device)
        if table.numel() > 0:
            pos = torch.clamp(torch.searchsorted(table, linear), max=table.numel() - 1)
            is_internal = table[pos] == linear
        else:
            is_internal = torch.zeros_like(linear, dtype=torch.bool)
            pos = linear

        leaf_mask = ~is_internal
        if torch.any(leaf_mask):
            leaf_linear_table, leaf_ids_table = model._level_leaf_lookup(level)
            leaf_pos = torch.searchsorted(leaf_linear_table, linear[leaf_mask])
            found_leaf_ids = leaf_ids_table[leaf_pos]
            found_degrees = model.leaf_degrees[found_leaf_ids]
            found_counts = model.coefficient_counts()[found_leaf_ids]
            if torch.all(found_counts == found_counts[0]):
                width = int(found_counts[0].item())
                gather = (
                    model.coefficient_offsets[found_leaf_ids, None]
                    + torch.arange(width, device=device)[None, :]
                )
                found_coefficients = physical[gather].reshape(-1)
            else:
                found_coefficients = torch.cat(
                    [model.coefficient_block(int(leaf_id)).detach() for leaf_id in found_leaf_ids.tolist()]
                )
            _emit(
                level,
                active_coords[leaf_mask],
                found_degrees,
                found_coefficients,
            )

        next_active: list[torch.Tensor] = []
        if torch.any(is_internal):
            internal_local = torch.nonzero(is_internal, as_tuple=False).reshape(-1)
            node_pos = pos[internal_local]
            coords_internal = active_coords[internal_local]
            action = actions[level][node_pos]

            p0_sel = action == P0_ACTION
            _emit(
                level,
                coords_internal[p0_sel],
                torch.zeros((int(p0_sel.sum().item()), 3), dtype=torch.long, device=device),
                bank.p0_value[level][node_pos[p0_sel]],
            )

            p1_sel = action == P1_ACTION
            n_p1 = int(p1_sel.sum().item())
            _emit(
                level,
                coords_internal[p1_sel],
                torch.tensor(P1_DEGREE, dtype=torch.long, device=device).expand(n_p1, 3),
                bank.p1_coeffs[level][node_pos[p1_sel]].reshape(-1),
            )

            keep_sel = action == KEEP_ACTION
            if torch.any(keep_sel):
                offsets = _child_offsets(device)
                children = (coords_internal[keep_sel][:, None, :] * 2 + offsets[None, :, :]).reshape(-1, 3)
                next_active.append(children)

        active_coords = (
            torch.cat(next_active, dim=0) if next_active else torch.empty((0, 3), dtype=torch.long, device=device)
        )
        if active_coords.shape[0] == 0:
            break

    all_levels = torch.cat(new_levels, dim=0)
    all_coords = torch.cat(new_coords, dim=0)
    all_degrees = torch.cat(new_degrees, dim=0)
    all_coeffs = torch.cat(new_coeffs, dim=0)
    counts = torch.prod(all_degrees + 1, dim=1)
    offsets = torch.cat([torch.zeros((1,), dtype=torch.long, device=device), torch.cumsum(counts, dim=0)], dim=0)
    owners = torch.repeat_interleave(torch.arange(all_levels.shape[0], device=device), counts)

    new_model = BernsteinOctree(
        levels=model.level_shapes,
        max_degree=model.max_degree,
        attenuation_shift=model.attenuation_shift,
        cuda_integrator=model.cuda_integrator,
        integration_mode=model.integration_mode,
        topology=model.topology,
        balance_2to1=model.balance_2to1_enabled,
        max_leaf_count=model.max_leaf_count,
    ).to(device=device)
    state_dict = {
        "coefficient_logits": new_model._coefficients_to_logits(all_coeffs).contiguous(),
        "leaf_levels": all_levels.contiguous(),
        "leaf_coords": all_coords.contiguous(),
        "leaf_degrees": all_degrees.contiguous(),
        "coefficient_offsets": offsets.contiguous(),
        "coefficient_leaf_ids": owners.contiguous(),
    }
    new_model.prepare_sparse_from_state_dict(state_dict)
    new_model.load_state_dict(state_dict, strict=False)
    return new_model


def finetune_fixed_topology(
    model: BernsteinOctree,
    dataset,
    *,
    iterations: int,
    batch_rays: int,
    lr: float,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    log_every: int = 200,
) -> dict[str, Any]:
    """Step 7: freeze the compressed topology (never call split/merge here)
    and jointly optimize every p0/p1 coefficient against the real sparse
    training projections with plain uniform WLS, matching stage 1's
    objective -- no continuity, no TV, no re-introduced structure search."""
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    materialize_points = not model.prefer_compact_ray_batch()
    history: list[dict[str, float]] = []
    for iteration in range(int(iterations)):
        ray_batch = random_training_rays(
            dataset.train,
            batch_rays=int(batch_rays),
            detector_h=int(detector_h),
            detector_w=int(detector_w),
            samples_per_ray=int(samples_per_ray),
            materialize_points=materialize_points,
        )
        prediction = model.integrate_ray_batch(ray_batch)
        loss = 0.5 * torch.mean(torch.square(prediction - ray_batch.target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration % max(1, int(log_every)) == 0 or iteration == int(iterations) - 1:
            row = {"iteration": iteration, "projection_loss": float(loss.detach().item())}
            history.append(row)
            print(json.dumps({"phase": "finetune", **row}), flush=True)
    return {"history": history, "iterations": int(iterations)}


def _resolve_node_chunk(
    *,
    requested: int,
    samples_per_axis: int,
    device: torch.device,
    max_workspace_mb: float,
) -> int:
    """Choose a conservative candidate batch from currently free CUDA memory.

    Packed point traversal keeps several ids, masks, coordinates and temporary
    arrays per sample at each octree level.  The 512-byte estimate intentionally
    overestimates that live footprint so automatic mode remains stable on
    laptop GPUs while still issuing millions of point queries per kernel batch.
    """
    if int(requested) > 0:
        return int(requested)
    sample_count = max(1, int(samples_per_axis) ** 3)
    if device.type != "cuda":
        return 4096
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    workspace_bytes = min(
        int(float(max_workspace_mb) * 1024.0 * 1024.0),
        int(float(free_bytes) * 0.25),
    )
    estimated_bytes_per_node = sample_count * 512
    return max(1024, min(65536, workspace_bytes // max(estimated_bytes_per_node, 1)))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _evaluate_projection_split(
    model: BernsteinOctree,
    split,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    ray_chunk: int,
) -> dict[str, float]:
    rendered = render_split(
        model,
        split,
        detector_h=int(detector_h),
        detector_w=int(detector_w),
        samples_per_ray=int(samples_per_ray),
        ray_chunk=int(ray_chunk),
    )
    return projection_metrics(rendered, split.projections)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline v5: compress a frozen h-only reference field into p0/p1 subtrees under a byte budget."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--r-max-bytes", type=float, required=True, help="raw packed byte budget for the compressed tree.")
    parser.add_argument("--samples-per-axis", type=int, default=4)
    parser.add_argument(
        "--node-chunk",
        type=int,
        default=0,
        help="candidate nodes per GPU batch; 0 selects from available workspace.",
    )
    parser.add_argument("--max-workspace-mb", type=float, default=2048.0)
    parser.add_argument("--finetune-iterations", type=int, default=0, help="0 disables step 7.")
    parser.add_argument("--finetune-batch-rays", type=int, default=65536)
    parser.add_argument("--finetune-lr", type=float, default=0.015)
    parser.add_argument("--finetune-log-every", type=int, default=200)
    parser.add_argument("--eval-ray-chunk", type=int, default=65536)
    parser.add_argument("--score-ray-chunk", type=int, default=8192)
    parser.add_argument("--score-views", type=int, default=0, help="0 uses every training view.")
    parser.add_argument("--score-rays-per-view", type=int, default=8192)
    parser.add_argument(
        "--p1-only",
        action="store_true",
        help="Only permit keep-subtree or merge-to-p1 actions; require an all-p1 reference and result.",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-checkpoint", default="")
    parser.add_argument("--out-compact", default="")
    parser.add_argument("--compact-quantization", default="float16")
    parser.add_argument("--out-report", default="")
    args = parser.parse_args(argv)

    torch_device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    seed = int(args.seed) if args.seed is not None else 0
    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(torch_device)

    timings: dict[str, float] = {}

    def timed(name: str, function):
        _sync(torch_device)
        started = time.perf_counter()
        result = function()
        _sync(torch_device)
        timings[name] = time.perf_counter() - started
        print(json.dumps({"phase": name, "seconds": timings[name]}), flush=True)
        return result

    config, model = load_reference_model(args.config, args.checkpoint, torch_device)
    if args.p1_only and not torch.all(model.leaf_degrees == 1):
        raise ValueError("--p1-only requires every reference leaf to have degree [1,1,1].")
    if args.seed is None:
        seed = int(config.get("training", {}).get("seed", 0))
        torch.manual_seed(seed)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    node_chunk = _resolve_node_chunk(
        requested=args.node_chunk,
        samples_per_axis=args.samples_per_axis,
        device=torch_device,
        max_workspace_mb=args.max_workspace_mb,
    )
    print(
        json.dumps(
            {
                "phase": "setup",
                "device": str(torch_device),
                "gpu": torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else None,
                "node_chunk": node_chunk,
                "samples_per_axis": int(args.samples_per_axis),
                "seed": seed,
            }
        ),
        flush=True,
    )

    # Projection data is the sole arbiter of compression actions.  It is
    # required even when final evaluation and coefficient finetuning are off.
    dataset = timed(
        "load_dataset",
        lambda: load_r2_dataset(config["dataset"]["root"], device=torch_device, load_volume=False),
    )
    detector_h = dataset.detector_shape[0]
    detector_w = dataset.detector_shape[1]
    samples_per_ray = int(
        config["geometry"].get(
            "samples_per_ray",
            dataset.volume_shape[0],
        )
    )

    evaluation: dict[str, Any] = {}
    if dataset is not None and not args.skip_eval:
        evaluation["reference_test"] = timed(
            "evaluate_reference",
            lambda: _evaluate_projection_split(
                model,
                dataset.test,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                ray_chunk=args.eval_ray_chunk,
            ),
        )

    bank = timed(
        "build_candidate_bank",
        lambda: build_candidate_bank(
            model,
            samples_per_axis=args.samples_per_axis,
            node_chunk=node_chunk,
        ),
    )
    p0_projection_distortion, p1_projection_distortion, projection_score_summary = timed(
        "score_projection_candidates",
        lambda: score_projection_candidates(
            model,
            bank,
            dataset.train,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            ray_chunk=args.score_ray_chunk,
            max_views=(None if int(args.score_views) <= 0 else int(args.score_views)),
            rays_per_view=args.score_rays_per_view,
        ),
    )
    # mu_ref constructs candidate functions, but only training projections
    # judge them.  Reuse the DP storage slots to avoid another million-entry
    # candidate bank allocation.
    bank.p0_value_distortion = p0_projection_distortion
    bank.p1_value_distortion = p1_projection_distortion
    bank.p0_detail_distortion = [torch.zeros_like(value) for value in p0_projection_distortion]
    bank.p1_detail_distortion = [torch.zeros_like(value) for value in p1_projection_distortion]
    leaf_bytes_by_level = _leaf_bytes_by_level(model)
    fixed_payload_bytes = _packed_v3_fixed_payload_bytes(model)
    tree_budget_bytes = float(args.r_max_bytes) - float(fixed_payload_bytes)
    if tree_budget_bytes <= 0:
        raise ValueError(
            f"R_max={args.r_max_bytes:.0f} is not larger than the "
            f"{fixed_payload_bytes}-byte compact-v3 fixed payload."
        )
    total_bytes, total_distortion, actions, lam = timed(
        "rate_distortion_dp",
        lambda: run_compression(
            model,
            bank,
            leaf_bytes_by_level,
            gamma=0.0,
            r_max_bytes=tree_budget_bytes,
            allow_p0=not args.p1_only,
        ),
    )
    action_counts = [
        {
            "level": level,
            "keep": int(torch.sum(level_actions == KEEP_ACTION).item()),
            "p0": int(torch.sum(level_actions == P0_ACTION).item()),
            "p1": int(torch.sum(level_actions == P1_ACTION).item()),
        }
        for level, level_actions in enumerate(actions)
        if level_actions.numel() > 0
    ]
    compressed = timed(
        "materialize_compressed_model",
        lambda: materialize_compressed_model(model, bank, actions),
    )
    if args.p1_only and not torch.all(compressed.leaf_degrees == 1):
        raise RuntimeError("p1-only compression produced a non-p1 leaf.")
    p0_leaf_count = int(torch.sum(torch.all(compressed.leaf_degrees == 0, dim=1)).item())
    p1_leaf_count = int(torch.sum(torch.all(compressed.leaf_degrees == 1, dim=1)).item())
    other_degree_leaf_count = int(compressed.leaf_degrees.shape[0]) - p0_leaf_count - p1_leaf_count

    if dataset is not None and not args.skip_eval:
        evaluation["compressed_before_finetune_test"] = timed(
            "evaluate_compressed_before_finetune",
            lambda: _evaluate_projection_split(
                compressed,
                dataset.test,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                ray_chunk=args.eval_ray_chunk,
            ),
        )

    # The candidate bank and the 454 MB training checkpoint representation are
    # not needed during fixed-topology optimization.  Release them before the
    # long CUDA run so ray batches get the largest possible workspace.
    del bank, actions, leaf_bytes_by_level, model
    gc.collect()
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()

    finetune_report = None
    if int(args.finetune_iterations) > 0:
        if dataset is None:
            raise RuntimeError("The training dataset was not loaded.")
        finetune_report = timed(
            "finetune_fixed_topology",
            lambda: finetune_fixed_topology(
                compressed,
                dataset,
                iterations=args.finetune_iterations,
                batch_rays=args.finetune_batch_rays,
                lr=args.finetune_lr,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                log_every=args.finetune_log_every,
            ),
        )

    if dataset is not None and not args.skip_eval:
        evaluation["compressed_after_finetune_test"] = timed(
            "evaluate_compressed_after_finetune",
            lambda: _evaluate_projection_split(
                compressed,
                dataset.test,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                ray_chunk=args.eval_ray_chunk,
            ),
        )

    out_checkpoint: Path | None = None
    if args.out_checkpoint:
        out_checkpoint = Path(args.out_checkpoint)
        out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        timed(
            "save_checkpoint",
            lambda: torch.save({"config": config, "model": compressed.state_dict()}, out_checkpoint),
        )

    compact_summary = None
    if args.out_compact:
        compact_summary = timed(
            "export_compact",
            lambda: export_compact_octree_artifact(
                compressed,
                Path(args.out_compact),
                quantization=args.compact_quantization,
                topology="packed_hierarchy",
                checkpoint_path=out_checkpoint,
            ),
        )

    report = {
        "schema": "functional_compression_v5_v1",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "r_max_bytes": float(args.r_max_bytes),
        "fixed_payload_bytes": fixed_payload_bytes,
        "selected_tree_bytes": total_bytes,
        "distortion_domain": "training_projection_residual",
        "p1_only": bool(args.p1_only),
        "projection_candidate_scoring": projection_score_summary,
        "lambda": lam,
        "raw_packed_bytes": total_bytes + float(fixed_payload_bytes),
        "total_distortion": total_distortion,
        "compressed_leaf_count": int(compressed.leaf_levels.shape[0]),
        "functional_leaf_counts": {
            "p0": p0_leaf_count,
            "p1": p1_leaf_count,
            "other": other_degree_leaf_count,
        },
        "compressed_stats": compressed.stats().__dict__,
        "action_counts": action_counts,
        "finetune": finetune_report,
        "evaluation": evaluation,
        "compact": compact_summary.__dict__ if compact_summary is not None else None,
        "device": str(torch_device),
        "gpu": torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else None,
        "node_chunk": node_chunk,
        "timings_seconds": timings,
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(torch_device) / (1024.0 * 1024.0))
            if torch_device.type == "cuda"
            else None
        ),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out_report:
        out = Path(args.out_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
