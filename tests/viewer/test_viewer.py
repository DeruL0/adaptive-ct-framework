from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from threading import Lock

import numpy as np
import torch

from adaptive_ct.viewer.server import (
    ViewerState,
    _resolve_model_config,
    _slice_from_volume,
    _validate_model_path,
)


class _CoordinateModel:
    def forward_mu(self, points: torch.Tensor) -> torch.Tensor:
        return points[:, 0] + 10.0 * points[:, 1] + 100.0 * points[:, 2]


class ViewerSliceTests(unittest.TestCase):
    def test_decode_slice_matches_full_coordinate_field_for_all_axes(self) -> None:
        shape = (3, 4, 5)
        state = ViewerState.__new__(ViewerState)
        state.dataset = SimpleNamespace(volume_shape=shape)
        state.device = torch.device("cpu")
        state.model = _CoordinateModel()
        state._lock = Lock()
        state._decoded_volume_time_sec = None

        coordinates = [
            -1.0 + (np.arange(size, dtype=np.float32) + 0.5) * 2.0 / size
            for size in shape
        ]
        xx, yy, zz = np.meshgrid(*coordinates, indexing="ij")
        full = xx + 10.0 * yy + 100.0 * zz

        for axis, index in (("x", 1), ("y", 2), ("z", 3)):
            with self.subTest(axis=axis):
                actual = state._decode_slice(axis=axis, index=index)
                expected = _slice_from_volume(full, axis, index)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


class ViewerSourceTests(unittest.TestCase):
    @staticmethod
    def _write_config(path: Path, output_dir: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"output:\n  dir: {output_dir.as_posix()}\n", encoding="utf-8")

    def test_adjacent_config_is_selected_for_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "external_run"
            checkpoint = model_dir / "checkpoint.pt"
            config = model_dir / "config.yaml"
            model_dir.mkdir()
            checkpoint.write_bytes(b"pt")
            self._write_config(config, model_dir)
            self.assertEqual(_resolve_model_config(checkpoint, root), config.resolve())

    def test_training_report_resolves_workspace_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "output" / "run"
            checkpoint = model_dir / "checkpoint.pt"
            config = root / "configs" / "case.yaml"
            model_dir.mkdir(parents=True)
            checkpoint.write_bytes(b"pt")
            self._write_config(config, model_dir)
            (model_dir / "training_report.json").write_text(
                json.dumps({"config": str(config.resolve())}),
                encoding="utf-8",
            )
            self.assertEqual(_resolve_model_config(checkpoint, root), config.resolve())

    def test_output_directory_match_resolves_compact_model_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "output" / "run"
            compact = model_dir / "compact_octree.npz"
            config = root / "configs" / "case.yaml"
            model_dir.mkdir(parents=True)
            compact.write_bytes(b"npz")
            self._write_config(config, model_dir)
            self.assertEqual(_resolve_model_config(compact, root), config.resolve())

    def test_checkpoint_metadata_resolves_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "copied_model"
            checkpoint = model_dir / "checkpoint.pt"
            config = root / "configs" / "case.yaml"
            model_dir.mkdir()
            self._write_config(config, root / "original_output")
            torch.save({"config": {"_config_path": str(config.resolve())}, "model": {}}, checkpoint)
            self.assertEqual(_resolve_model_config(checkpoint, root), config.resolve())

    def test_model_selection_rejects_unsupported_file_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "model.txt"
            invalid.write_text("not a model", encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate_model_path(invalid)


if __name__ == "__main__":
    unittest.main()
