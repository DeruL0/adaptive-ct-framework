from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.autograd import Function

try:
    from adaptive_ct_native import _C as _NATIVE
except Exception:  # pragma: no cover
    native_root = Path(__file__).resolve().parents[2] / "native"
    if str(native_root) not in sys.path:
        sys.path.insert(0, str(native_root))
    try:
        from adaptive_ct_native import _C as _NATIVE
    except Exception as second_error:  # pragma: no cover
        _NATIVE = None
        _NATIVE_IMPORT_ERROR = second_error
    else:  # pragma: no cover
        _NATIVE_IMPORT_ERROR = None
else:  # pragma: no cover
    _NATIVE_IMPORT_ERROR = None


def has_bernstein_native() -> bool:
    return _NATIVE is not None and all(
        hasattr(_NATIVE, name)
        for name in (
            "bernstein_octree_integrate_forward",
            "bernstein_octree_integrate_backward",
            "bernstein_octree_segments_forward",
            "bernstein_octree_decision_accumulate",
        )
    )


def require_native() -> None:
    if _NATIVE is None:
        raise RuntimeError(f"adaptive_ct native extension is unavailable: {_NATIVE_IMPORT_ERROR!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("adaptive_ct native extension requires CUDA.")


def project_dense_parallel(volume: torch.Tensor, angles: torch.Tensor, *, detector_h: int, detector_w: int) -> torch.Tensor:
    """Projections of a dense voxel volume (viewer only: GT and decoded volumes at any angle)."""
    require_native()
    return _NATIVE.project_dense_parallel_forward(volume.contiguous(), angles.contiguous(), int(detector_h), int(detector_w), int(max(volume.shape)))


def _normalise_level_shapes(
    levels: list[int | tuple[int, int, int] | list[int]],
) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for value in levels:
        if isinstance(value, int):
            shape = (int(value),) * 3
        else:
            shape = tuple(int(component) for component in value)
            if len(shape) != 3:
                raise ValueError(f"Expected a scalar or 3-value level shape, got {value!r}.")
        shapes.append(shape)
    return shapes


def bernstein_segment_ray_capacity(
    levels: list[int | tuple[int, int, int] | list[int]],
    max_buffer_mb: float,
) -> int:
    """Maximum ray count whose dense segment tuple fits the requested budget."""
    shapes = _normalise_level_shapes(levels)
    if not shapes:
        raise ValueError("At least one octree level is required.")
    max_segments = int(shapes[-1][0]) + int(shapes[-1][1]) + 8
    bytes_per_ray = max_segments * (8 + 4)  # int64 leaf id + float32 contribution
    return max(1, int(float(max_buffer_mb) * 1024.0 * 1024.0) // bytes_per_ray)


class _BernsteinOctreeIntegrate(Function):
    @staticmethod
    def forward(
        ctx,
        coefficient_logits: torch.Tensor,
        leaf_degrees: torch.Tensor,
        coefficient_offsets: torch.Tensor,
        step_geometry: torch.Tensor,
        node_child_base: torch.Tensor,
        node_leaf_id: torch.Tensor,
        angles: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        detector_h: int,
        detector_w: int,
        num_levels: int,
        root_x: int,
        root_y: int,
        root_z: int,
        attenuation_shift: float,
    ) -> torch.Tensor:
        require_native()
        if not has_bernstein_native():
            raise RuntimeError("The loaded adaptive_ct native extension does not include Bernstein CUDA kernels.")
        ctx.save_for_backward(
            coefficient_logits,
            leaf_degrees,
            coefficient_offsets,
            step_geometry,
            node_child_base,
            node_leaf_id,
            angles,
            rows,
            cols,
        )
        ctx.detector_h = int(detector_h)
        ctx.detector_w = int(detector_w)
        ctx.num_levels = int(num_levels)
        ctx.root_shape = (int(root_x), int(root_y), int(root_z))
        ctx.attenuation_shift = float(attenuation_shift)
        return _NATIVE.bernstein_octree_integrate_forward(
            coefficient_logits.contiguous(),
            leaf_degrees.contiguous(),
            coefficient_offsets.contiguous(),
            step_geometry.contiguous(),
            node_child_base.contiguous(),
            node_leaf_id.contiguous(),
            angles.contiguous(),
            rows.contiguous(),
            cols.contiguous(),
            ctx.detector_h,
            ctx.detector_w,
            ctx.num_levels,
            ctx.root_shape[0],
            ctx.root_shape[1],
            ctx.root_shape[2],
            ctx.attenuation_shift,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            coefficient_logits,
            leaf_degrees,
            coefficient_offsets,
            step_geometry,
            node_child_base,
            node_leaf_id,
            angles,
            rows,
            cols,
        ) = ctx.saved_tensors
        grad_coefficients = _NATIVE.bernstein_octree_integrate_backward(
            coefficient_logits.contiguous(),
            leaf_degrees.contiguous(),
            coefficient_offsets.contiguous(),
            step_geometry.contiguous(),
            node_child_base.contiguous(),
            node_leaf_id.contiguous(),
            angles.contiguous(),
            rows.contiguous(),
            cols.contiguous(),
            grad_output.contiguous(),
            ctx.detector_h,
            ctx.detector_w,
            ctx.num_levels,
            ctx.root_shape[0],
            ctx.root_shape[1],
            ctx.root_shape[2],
            ctx.attenuation_shift,
        )
        return (grad_coefficients,) + (None,) * 15


def bernstein_octree_integrate(
    *,
    coefficient_logits: torch.Tensor,
    leaf_degrees: torch.Tensor,
    coefficient_offsets: torch.Tensor,
    step_geometry: torch.Tensor,
    node_child_base: torch.Tensor,
    node_leaf_id: torch.Tensor,
    angles: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    detector_h: int,
    detector_w: int,
    level_shapes: list[int | tuple[int, int, int] | list[int]],
    attenuation_shift: float,
) -> torch.Tensor:
    shapes = _normalise_level_shapes(level_shapes)
    if not shapes:
        raise ValueError("At least one octree level is required.")
    root_x, root_y, root_z = shapes[0]
    return _BernsteinOctreeIntegrate.apply(
        coefficient_logits,
        leaf_degrees,
        coefficient_offsets,
        step_geometry,
        node_child_base,
        node_leaf_id,
        angles,
        rows,
        cols,
        int(detector_h),
        int(detector_w),
        int(len(shapes)),
        root_x,
        root_y,
        root_z,
        float(attenuation_shift),
    )


def bernstein_octree_ray_segments(
    *,
    coefficient_logits: torch.Tensor,
    leaf_degrees: torch.Tensor,
    coefficient_offsets: torch.Tensor,
    step_geometry: torch.Tensor,
    node_child_base: torch.Tensor,
    node_leaf_id: torch.Tensor,
    angles: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    detector_h: int,
    detector_w: int,
    level_shapes: list[int | tuple[int, int, int] | list[int]],
    attenuation_shift: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require_native()
    if not has_bernstein_native():
        raise RuntimeError("The loaded adaptive_ct native extension does not include Bernstein CUDA kernels.")
    shapes = _normalise_level_shapes(level_shapes)
    if not shapes:
        raise ValueError("At least one octree level is required.")
    root_x, root_y, root_z = shapes[0]
    leaf_ids, contributions = _NATIVE.bernstein_octree_segments_forward(
        coefficient_logits.contiguous(),
        leaf_degrees.contiguous(),
        coefficient_offsets.contiguous(),
        step_geometry.contiguous(),
        node_child_base.contiguous(),
        node_leaf_id.contiguous(),
        angles.contiguous(),
        rows.contiguous(),
        cols.contiguous(),
        int(detector_h),
        int(detector_w),
        int(len(shapes)),
        root_x,
        root_y,
        root_z,
        float(attenuation_shift),
    )
    valid = leaf_ids >= 0
    ray_ids = torch.arange(leaf_ids.shape[0], device=leaf_ids.device)[:, None].expand_as(leaf_ids)[valid]
    return ray_ids, leaf_ids[valid], contributions[valid]


def bernstein_octree_decision_accumulate(*, coefficient_logits, leaf_degrees, coefficient_offsets, step_geometry, node_child_base,
                                         node_leaf_id, angles, rows, cols, detector_h, detector_w, level_shapes, attenuation_shift,
                                         candidate_geometry, residual, fold, accum) -> torch.Tensor:
    """Add the decision statistics of the rays (see segment_decision in bernstein_octree.cu) into accum (2, N, 97)."""
    require_native()
    shapes = _normalise_level_shapes(level_shapes)
    root_x, root_y, root_z = shapes[0]
    return _NATIVE.bernstein_octree_decision_accumulate(
        coefficient_logits.detach().contiguous(), leaf_degrees.contiguous(), coefficient_offsets.contiguous(),
        step_geometry.contiguous(), candidate_geometry.contiguous(), node_child_base.contiguous(), node_leaf_id.contiguous(),
        angles.contiguous(), rows.contiguous(), cols.contiguous(), residual.contiguous(), fold.to(torch.int32).contiguous(), accum,
        int(detector_h), int(detector_w), int(len(shapes)), root_x, root_y, root_z, float(attenuation_shift))
