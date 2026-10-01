from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json

import numpy as np
import torch


@dataclass(frozen=True)
class ProjectionSplit:
    angles: torch.Tensor
    projections: torch.Tensor
    paths: List[Path]


@dataclass(frozen=True)
class R2Dataset:
    root: Path
    scanner: Dict
    volume: Optional[torch.Tensor]
    train: ProjectionSplit
    test: ProjectionSplit
    validation: Optional[ProjectionSplit] = None

    @property
    def detector_shape(self) -> tuple[int, int]:
        n_detector = self.scanner["nDetector"]
        return int(n_detector[0]), int(n_detector[1])

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        n_voxel = self.scanner["nVoxel"]
        return int(n_voxel[0]), int(n_voxel[1]), int(n_voxel[2])


def _load_split(root: Path, frames: list, device: torch.device) -> ProjectionSplit:
    angles = []
    projs = []
    paths = []
    for frame in frames:
        path = root / frame["file_path"]
        paths.append(path)
        angles.append(float(frame["angle"]))
        projs.append(np.load(path).astype(np.float32))
    return ProjectionSplit(
        angles=torch.tensor(angles, dtype=torch.float32, device=device),
        projections=torch.from_numpy(np.stack(projs, axis=0)).to(device=device, dtype=torch.float32),
        paths=paths,
    )


def load_r2_dataset(
    root: str | Path,
    device: str | torch.device = "cuda",
    *,
    load_volume: bool = True,
) -> R2Dataset:
    root_path = Path(root).resolve()
    with (root_path / "meta_data.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    scanner = dict(meta["scanner"])
    if scanner.get("mode") != "parallel":
        raise ValueError(f"Only parallel mode is implemented for M0/M1, got {scanner.get('mode')!r}.")
    dev = torch.device(device)
    volume = None
    if load_volume:
        volume = torch.from_numpy(np.load(root_path / meta["vol"]).astype(np.float32)).to(device=dev)
    return R2Dataset(
        root=root_path,
        scanner=scanner,
        volume=volume,
        train=_load_split(root_path, meta["proj_train"], dev),
        test=_load_split(root_path, meta["proj_test"], dev),
        validation=(
            _load_split(root_path, meta["proj_val"], dev)
            if meta.get("proj_val")
            else None
        ),
    )
