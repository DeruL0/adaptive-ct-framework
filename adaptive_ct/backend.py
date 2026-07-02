from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.autograd import Function

try:
    from adaptive_ct_native import _C as _NATIVE
except Exception as import_error:  # pragma: no cover
    native_root = Path(__file__).resolve().parents[1] / "native"
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


def has_native() -> bool:
    return _NATIVE is not None


def has_bernstein_native() -> bool:
    return _NATIVE is not None and all(
        hasattr(_NATIVE, name)
        for name in (
            "bernstein_octree_integrate_forward",
            "bernstein_octree_integrate_backward",
            "bernstein_octree_segments_forward",
        )
    )


def require_native() -> None:
    if _NATIVE is None:
        raise RuntimeError(f"adaptive_ct native extension is unavailable: {_NATIVE_IMPORT_ERROR!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("adaptive_ct native extension requires CUDA.")


def project_dense_parallel(
    volume: torch.Tensor,
    angles: torch.Tensor,
    *,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
) -> torch.Tensor:
    require_native()
    return _NATIVE.project_dense_parallel_forward(
        volume.contiguous(),
        angles.contiguous(),
        int(detector_h),
        int(detector_w),
        int(samples_per_ray),
    )


class _DynamicVoxelIntegrate(Function):
    @staticmethod
    def forward(
        ctx,
        leaf_logits: torch.Tensor,
        index0: torch.Tensor,
        index1: torch.Tensor,
        index2: torch.Tensor,
        angles: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        detector_h: int,
        detector_w: int,
        num_levels: int,
        res0: int,
        res1: int,
        res2: int,
        attenuation_shift: float,
    ) -> torch.Tensor:
        require_native()
        ctx.save_for_backward(leaf_logits, index0, index1, index2, angles, rows, cols)
        ctx.detector_h = int(detector_h)
        ctx.detector_w = int(detector_w)
        ctx.num_levels = int(num_levels)
        ctx.res0 = int(res0)
        ctx.res1 = int(res1)
        ctx.res2 = int(res2)
        ctx.attenuation_shift = float(attenuation_shift)
        return _NATIVE.dynamic_voxel_integrate_forward(
            leaf_logits.contiguous(),
            index0.contiguous(),
            index1.contiguous(),
            index2.contiguous(),
            angles.contiguous(),
            rows.contiguous(),
            cols.contiguous(),
            ctx.detector_h,
            ctx.detector_w,
            ctx.num_levels,
            ctx.res0,
            ctx.res1,
            ctx.res2,
            ctx.attenuation_shift,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        leaf_logits, index0, index1, index2, angles, rows, cols = ctx.saved_tensors
        grad_leaf_logits = _NATIVE.dynamic_voxel_integrate_backward(
            leaf_logits.contiguous(),
            index0.contiguous(),
            index1.contiguous(),
            index2.contiguous(),
            angles.contiguous(),
            rows.contiguous(),
            cols.contiguous(),
            grad_output.contiguous(),
            ctx.detector_h,
            ctx.detector_w,
            ctx.num_levels,
            ctx.res0,
            ctx.res1,
            ctx.res2,
            ctx.attenuation_shift,
        )
        return grad_leaf_logits, None, None, None, None, None, None, None, None, None, None, None, None, None


def dynamic_voxel_integrate(
    *,
    leaf_logits: torch.Tensor,
    index_maps: list[torch.Tensor],
    angles: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    detector_h: int,
    detector_w: int,
    resolutions: list[int],
    attenuation_shift: float,
) -> torch.Tensor:
    require_native()
    if not resolutions:
        raise ValueError("At least one voxel level is required.")
    if len(resolutions) > 3:
        raise ValueError("The CUDA dynamic voxel integrator currently supports at most 3 levels.")
    if len(index_maps) != len(resolutions):
        raise ValueError("index_maps and resolutions must have the same length.")
    empty_index = torch.empty((0,), dtype=torch.long, device=leaf_logits.device)
    padded_maps = list(index_maps) + [empty_index] * (3 - len(index_maps))
    padded_res = [int(v) for v in resolutions] + [int(resolutions[-1])] * (3 - len(resolutions))
    return _DynamicVoxelIntegrate.apply(
        leaf_logits,
        padded_maps[0],
        padded_maps[1],
        padded_maps[2],
        angles,
        rows,
        cols,
        int(detector_h),
        int(detector_w),
        int(len(resolutions)),
        padded_res[0],
        padded_res[1],
        padded_res[2],
        float(attenuation_shift),
    )


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
        return (grad_coefficients,) + (None,) * 14


def bernstein_octree_integrate(
    *,
    coefficient_logits: torch.Tensor,
    leaf_degrees: torch.Tensor,
    coefficient_offsets: torch.Tensor,
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
