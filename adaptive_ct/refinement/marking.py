"""Dorfler marking."""

from __future__ import annotations

import torch


def dorfler_mark(
    eta: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    delta_bytes: int,
    theta: float,
    remaining_bytes: float,
    max_parents: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Smallest rate-ranked prefix carrying ``theta`` of the indicator mass.

    ``eta`` holds the (positive) indicator of ``candidate_ids``.  The prefix is
    then truncated by the byte budget and by ``max_parents``.  Every split costs
    the same ``delta_bytes`` (uniform degree), so the rate ranking is the eta
    ranking.
    """
    count = int(candidate_ids.numel())
    if count == 0:
        return candidate_ids, {"k_theta": 0, "k_bytes": 0, "eta_fraction": 0.0}
    order = torch.argsort(eta, descending=True)
    cumulative = torch.cumsum(eta[order].double(), dim=0)
    total = float(cumulative[-1].item())
    target = float(theta) * total
    k_theta = int(torch.searchsorted(cumulative, cumulative.new_tensor(target)).item()) + 1
    k_theta = min(k_theta, count)
    k_bytes = count if remaining_bytes == float("inf") else max(0, int(remaining_bytes // max(1, int(delta_bytes))))
    k = min(k_theta, k_bytes, count)
    if max_parents is not None:
        k = min(k, max(0, int(max_parents)))
    fraction = float(cumulative[k - 1].item()) / max(total, 1.0e-30) if k > 0 else 0.0
    return candidate_ids[order[:k]], {"k_theta": k_theta, "k_bytes": k_bytes, "eta_fraction": fraction}
