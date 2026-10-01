"""Plain projection-MSE fine-tuning of every trainable parameter."""

from __future__ import annotations

import torch

from adaptive_ct.core.geometry import random_training_rays
from adaptive_ct.core.loss import weighted_projection_loss


def finetune_model(
    model,
    split,
    *,
    steps: int,
    lr: float,
    batch_rays: int,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    weight_kwargs: dict,
    materialize_points: bool,
) -> dict:
    """Plain projection-MSE fine-tuning of every trainable parameter."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=float(lr))
    last = float("nan")
    for _ in range(int(steps)):
        batch = random_training_rays(
            split,
            batch_rays=int(batch_rays),
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            materialize_points=materialize_points,
        )
        loss = weighted_projection_loss(model.integrate_ray_batch(batch), batch.target, **weight_kwargs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last = float(loss.detach().item())
    return {"steps": int(steps), "lr": float(lr), "last_loss": last}
