"""Round what is stored through the storage precision: physical coefficients and step geometry."""

from __future__ import annotations

import torch


def direction_to_angles(n: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.atan2(n[:, 1], n[:, 0]), torch.acos(n[:, 2].clamp(-1.0, 1.0))], 1)


def angles_to_direction(a: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(a[:, 0]) * torch.sin(a[:, 1]), torch.sin(a[:, 0]) * torch.sin(a[:, 1]), torch.cos(a[:, 1])], 1)


def quantize_roundtrip_(model, quantization: str = "float16") -> None:
    if str(quantization).lower() == "float32":
        return
    with torch.no_grad():
        model.coefficient_logits.copy_(model._coefficients_to_logits(model.coefficients().half().float()))
        g = model.step_geometry
        g[:, :3] = angles_to_direction(direction_to_angles(g[:, :3]).half().float())
        g[:, 3:] = g[:, 3:].half().float()
