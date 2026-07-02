from __future__ import annotations

from math import log10
from typing import Dict

import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred.detach() - target.detach())).item())


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None) -> float:
    pred_f = pred.detach().to(dtype=torch.float32)
    target_f = target.detach().to(dtype=torch.float32)
    mse = torch.mean((pred_f - target_f) ** 2).clamp_min(1e-20)
    if data_range is None:
        data_range = float(target_f.max().clamp_min(1e-8).item())
    return 20.0 * log10(max(float(data_range), 1e-8)) - 10.0 * log10(float(mse.item()))


def projection_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {
        "psnr": psnr(pred, target),
        "mae": mae(pred, target),
    }
