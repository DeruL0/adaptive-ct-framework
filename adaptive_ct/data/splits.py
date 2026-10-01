from __future__ import annotations

from typing import Sequence

import torch

from adaptive_ct.data.dataset import ProjectionSplit


def split_projection_views(
    split: ProjectionSplit,
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[ProjectionSplit, ProjectionSplit]:
    view_count = int(split.angles.shape[0])
    if view_count < 2:
        raise ValueError("Held-out projection validation requires at least two views.")
    validation_count = max(1, min(view_count - 1, int(round(view_count * float(validation_fraction)))))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(view_count, generator=generator).tolist()
    validation_ids = permutation[:validation_count]
    training_ids = permutation[validation_count:]

    def subset(ids: Sequence[int]) -> ProjectionSplit:
        tensor_ids = torch.tensor(ids, dtype=torch.long, device=split.angles.device)
        return ProjectionSplit(
            angles=split.angles[tensor_ids],
            projections=split.projections[tensor_ids],
            paths=[split.paths[index] for index in ids],
        )

    return subset(training_ids), subset(validation_ids)
