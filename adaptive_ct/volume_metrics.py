from __future__ import annotations

from typing import Dict

import torch

from .backend import project_dense_parallel
from .metrics import mae, projection_metrics, psnr
from .refinement import decoded_volume, gradient_magnitude


@torch.no_grad()
def material_volume_metrics(model, dataset, *, threshold: float, chunk: int = 65536) -> Dict[str, float]:
    target = dataset.volume
    mask = target > float(threshold)
    if not torch.any(mask):
        return {}
    shape = tuple(int(v) for v in target.shape)
    coords = torch.nonzero(mask, as_tuple=False).to(dtype=torch.float32)
    scale = torch.tensor(shape, dtype=torch.float32, device=target.device)
    points = -1.0 + (coords + 0.5) * 2.0 / scale
    preds = []
    for start in range(0, points.shape[0], int(chunk)):
        preds.append(model.forward_mu(points[start : start + int(chunk)]).detach())
    pred = torch.cat(preds, dim=0)
    tgt = target[mask].detach()
    return {
        "count": int(tgt.numel()),
        "psnr": psnr(pred, tgt),
        "mae": mae(pred, tgt),
    }


@torch.no_grad()
def boundary_sharpness_metrics(
    model,
    dataset,
    *,
    threshold: float,
    boundary_quantile: float = 0.90,
) -> Dict[str, float]:
    target = dataset.volume.detach()
    pred = decoded_volume(model, tuple(int(v) for v in target.shape))
    gt_grad = gradient_magnitude(target)
    pred_grad = gradient_magnitude(pred)
    material = target > float(threshold)
    if not torch.any(material):
        return {}
    gt_material_grad = gt_grad[material]
    cutoff = torch.quantile(gt_material_grad, float(boundary_quantile))
    boundary = material & (gt_grad >= cutoff)
    if not torch.any(boundary):
        return {}
    pred_boundary = pred_grad[boundary]
    gt_boundary = gt_grad[boundary]
    pred_values = pred[boundary]
    target_values = target[boundary]
    return {
        "boundary_voxels": int(boundary.sum().item()),
        "gt_gradient_mean": float(gt_boundary.mean().item()),
        "pred_gradient_mean": float(pred_boundary.mean().item()),
        "gradient_ratio": float((pred_boundary.mean() / gt_boundary.mean().clamp_min(1e-8)).item()),
        "high_gradient_mae": mae(pred_values, target_values),
        "boundary_quantile": float(boundary_quantile),
    }


@torch.no_grad()
def material_projection_metrics(
    model,
    dataset,
    *,
    threshold: float,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
) -> Dict[str, object]:
    target = dataset.volume.detach()
    pred = decoded_volume(model, tuple(int(v) for v in target.shape))
    mask = (target > float(threshold)).to(dtype=torch.float32)
    pred_proj = project_dense_parallel(
        pred * mask,
        dataset.test.angles,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
    )
    target_proj = project_dense_parallel(
        target * mask,
        dataset.test.angles,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
    )
    mask_proj = project_dense_parallel(
        mask,
        dataset.test.angles,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
    ) > 1e-6
    if not torch.any(mask_proj):
        masked = {}
    else:
        masked = projection_metrics(pred_proj[mask_proj], target_proj[mask_proj])
        masked["mask_pixel_fraction"] = float(mask_proj.float().mean().item())
    return {
        "full_detector": projection_metrics(pred_proj, target_proj),
        "material_ray_masked": masked,
    }
