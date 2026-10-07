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
            "bernstein_octree_decision_accumulate",
            "bernstein_octree_geometry_accumulate",
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


def _tree_args(values, step_geometry, root, children, n_tri, n_step, level_shapes, rays):
    """Positional arguments of the octree kernels (layout: representation/octree.py)."""
    rx, ry, rz = level_shapes[0]
    device = values.device
    return (values.detach().contiguous(), step_geometry.detach().contiguous(), root, children, int(n_tri), int(n_step),
            len(level_shapes), int(rx), int(ry), int(rz), rays.angles.to(device, torch.float32).contiguous(),
            rays.rows.to(device, torch.long).contiguous(), rays.cols.to(device, torch.long).contiguous(),
            int(rays.detector_h), int(rays.detector_w))


class _OctreeIntegrate(Function):
    @staticmethod
    def forward(ctx, values, step_geometry, args):
        ctx.args = args
        return _NATIVE.bernstein_octree_integrate_forward(*args)

    @staticmethod
    def backward(ctx, grad_output):
        return _NATIVE.bernstein_octree_integrate_backward(*ctx.args, grad_output.contiguous()), None, None


def bernstein_octree_integrate(values, step_geometry, *, root, children, n_tri, n_step, level_shapes, rays) -> torch.Tensor:
    """Line integrals of the rays through the octree; differentiable in `values`."""
    require_native()
    return _OctreeIntegrate.apply(values, step_geometry, _tree_args(values, step_geometry, root, children, n_tri, n_step, level_shapes, rays))


def bernstein_octree_decision_accumulate(values, step_geometry, *, root, children, n_tri, n_step, level_shapes, rays,
                                         candidate_geometry, residual, accum) -> torch.Tensor:
    """Add the decision statistics of the rays (see segment_decision in bernstein_octree.cu) into accum (N, 102) (float32; one view per call)."""
    require_native()
    return _NATIVE.bernstein_octree_decision_accumulate(
        *_tree_args(values, step_geometry, root, children, n_tri, n_step, level_shapes, rays), candidate_geometry.contiguous(),
        residual.contiguous(), accum, int(candidate_geometry.shape[0]))


def bernstein_octree_geometry_accumulate(values, step_geometry, *, root, children, n_tri, n_step, level_shapes, rays, residual,
                                         accum) -> torch.Tensor:
    """Add the step-geometry Gauss-Newton statistics of the rays (see segment_geometry in bernstein_octree.cu) into accum (n_step, 20)."""
    require_native()
    return _NATIVE.bernstein_octree_geometry_accumulate(
        *_tree_args(values, step_geometry, root, children, n_tri, n_step, level_shapes, rays), residual.contiguous(), accum)
