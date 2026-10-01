"""Snapshot and restore the sparse (topology + parameters) state of an octree."""

from __future__ import annotations

import torch


def clone_sparse_state(model) -> dict[str, torch.Tensor]:
    """Clone the sparse Bernstein state so a rejected trial can be rolled back."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def restore_sparse_state(model, state_dict: dict[str, torch.Tensor]) -> None:
    model.prepare_sparse_from_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
