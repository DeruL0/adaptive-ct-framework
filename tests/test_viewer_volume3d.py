import base64
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from adaptive_ct.bernstein import BernsteinOctree
from adaptive_ct.model import build_model
from adaptive_ct.viewer.server import (
    _artifact_payload,
    _leaf_geometry_binary,
    _leaf_geometry_payload,
    _leaf_structure_payload,
    _volume3d_payload_from_arrays,
    _volume_projection_payload_from_arrays,
    _volume_texture_payload_from_arrays,
)


class ViewerVolume3DTests(unittest.TestCase):
    def test_leaf_structure_payload_reports_leaf_sizes(self):
        model = build_model({"model": {"representation": "bernstein_octree", "max_degree": 0, "levels": [2, 4, 8], "init_std": 0.0, "cuda_integrator": False}})
        model.split_leaves_batch(torch.tensor([0]))
        payload = _leaf_structure_payload(model)
        self.assertEqual(payload["leaf_cells"], 15)
        self.assertEqual([level["count"] for level in payload["levels"]], [7, 8, 0])
        self.assertAlmostEqual(payload["levels"][0]["voxel_size"], 1.0)
        self.assertAlmostEqual(payload["levels"][1]["voxel_size"], 0.5)
        self.assertEqual(payload["size_range"], {"min": 0.5, "max": 1.0})

    def test_leaf_geometry_payload_exports_boxes_with_sizes(self):
        model = build_model({"model": {"representation": "bernstein_octree", "max_degree": 0, "levels": [2, 4, 8], "init_std": 0.0, "cuda_integrator": False}})
        model.split_leaves_batch(torch.tensor([0]))
        payload = _leaf_geometry_payload(model)
        self.assertEqual(payload["kind"], "leaves")
        self.assertEqual(payload["total"], 15)
        self.assertEqual(payload["count"], 15)
        self.assertEqual(payload["level_counts"], [7, 8, 0])
        # L0 voxel size 2/2 = 1.0, L1 size 2/4 = 0.5.
        self.assertAlmostEqual(payload["level_sizes"][0], 1.0)
        self.assertAlmostEqual(payload["level_sizes"][1], 0.5)

        positions = np.frombuffer(base64.b64decode(payload["positions"]), dtype=np.float32).reshape(-1, 3)
        sizes = np.frombuffer(base64.b64decode(payload["sizes"]), dtype=np.float32)
        levels = np.frombuffer(base64.b64decode(payload["levels"]), dtype=np.uint8)
        self.assertEqual(positions.shape, (15, 3))
        self.assertEqual(sizes.shape, (15,))
        self.assertEqual(levels.shape, (15,))
        # Coarse leaves keep size 1.0, refined leaves shrink to 0.5.
        self.assertAlmostEqual(float(sizes[levels == 0].max()), 1.0, places=5)
        self.assertAlmostEqual(float(sizes[levels == 1].max()), 0.5, places=5)
        # Positions stay inside the normalized cube.
        self.assertTrue(np.all(np.abs(positions) <= 1.0 + 1e-5))

    def test_leaf_geometry_payload_respects_budget(self):
        model = build_model({"model": {"representation": "bernstein_octree", "max_degree": 0, "levels": [4], "init_std": 0.0, "cuda_integrator": False}})
        payload = _leaf_geometry_payload(model, max_leaves=10)
        self.assertEqual(payload["total"], 64)
        self.assertEqual(payload["count"], 10)
        positions = np.frombuffer(base64.b64decode(payload["positions"]), dtype=np.float32)
        self.assertEqual(positions.size, 30)

    def test_leaf_geometry_binary_is_compact_and_decodable(self):
        model = build_model({"model": {"representation": "bernstein_octree", "max_degree": 0, "levels": [4], "init_std": 0.0, "cuda_integrator": False}})
        encoded = _leaf_geometry_binary(model)
        self.assertEqual(encoded[:8], b"ACTLEAF1")
        header_size = struct.unpack("<I", encoded[8:12])[0]
        header = json.loads(encoded[12 : 12 + header_size])
        payload_offset = 12 + header_size
        count = header["count"]
        coords = np.frombuffer(
            encoded,
            dtype=np.uint16,
            count=count * 3,
            offset=payload_offset + header["coords_offset"],
        )
        mu = np.frombuffer(
            encoded,
            dtype=np.float32,
            count=count,
            offset=payload_offset + header["mu_offset"],
        )
        levels = np.frombuffer(
            encoded,
            dtype=np.uint8,
            count=count,
            offset=payload_offset + header["levels_offset"],
        )
        self.assertEqual(count, 64)
        self.assertEqual(coords.shape, (64 * 3,))
        self.assertEqual(mu.shape, (64,))
        self.assertEqual(levels.shape, (64,))
        # Even for this tiny fixture (where the fixed JSON header dominates),
        # the stream is smaller than the old 33 raw bytes/leaf box payload.
        self.assertLess(len(encoded), 24 * count)
        self.assertFalse(header["has_corners"])
        self.assertIsNone(header["corners_offset"])

    def test_leaf_geometry_binary_carries_p1_corners_for_non_constant_leaves(self):
        model = BernsteinOctree(levels=[2, 4], max_degree=1, attenuation_shift=0.0)
        for axis in (0, 1, 2):
            # Each elevate_degree() call rebuilds the leaf arrays, so the
            # affected leaf's integer id must be re-resolved every time.
            leaf_id = model.find_leaf((0, 0, 0, 0))
            model.elevate_degree(leaf_id, axis=axis)
        leaf_id = model.find_leaf((0, 0, 0, 0))
        with torch.no_grad():
            corner_values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            start = int(model.coefficient_offsets[leaf_id])
            model.coefficient_logits[start : start + 8].copy_(model._coefficients_to_logits(corner_values))

        encoded = _leaf_geometry_binary(model)
        header_size = struct.unpack("<I", encoded[8:12])[0]
        header = json.loads(encoded[12 : 12 + header_size])
        payload_offset = 12 + header_size
        count = header["count"]
        self.assertTrue(header["has_corners"])
        corners = np.frombuffer(
            encoded,
            dtype=np.float32,
            count=count * 8,
            offset=payload_offset + header["corners_offset"],
        ).reshape(count, 8)

        levels = np.frombuffer(
            encoded, dtype=np.uint8, count=count, offset=payload_offset + header["levels_offset"]
        )
        coords = np.frombuffer(
            encoded, dtype=np.uint16, count=count * 3, offset=payload_offset + header["coords_offset"]
        ).reshape(count, 3)
        p1_row = np.flatnonzero((levels == 0) & (coords == [0, 0, 0]).all(axis=1))
        self.assertEqual(p1_row.size, 1)
        np.testing.assert_allclose(corners[p1_row[0]], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], atol=1e-5)

        # A constant (p0) leaf elsewhere in the same model must have all 8
        # corners equal to its own mean, not the p1 leaf's values.
        other_row = np.flatnonzero((levels == 0) & (coords == [1, 0, 0]).all(axis=1))
        self.assertEqual(other_row.size, 1)
        self.assertEqual(len(set(np.round(corners[other_row[0]], 6).tolist())), 1)

    def test_prediction_payload_respects_point_budget(self):
        volume = np.zeros((4, 4, 4), dtype=np.float32)
        volume[1:3, 1:3, 1:3] = 0.5
        payload = _volume3d_payload_from_arrays(
            volume,
            volume,
            source="prediction",
            threshold=0.1,
            max_points=5,
        )
        self.assertEqual(payload["source"], "prediction")
        self.assertLessEqual(len(payload["positions"]) // 3, 5)
        self.assertEqual(len(payload["positions"]), len(payload["colors"]))
        self.assertEqual(payload["shape"], [4, 4, 4])

    def test_error_payload_keeps_top_errors(self):
        pred = np.zeros((4, 4, 4), dtype=np.float32)
        target = np.zeros((4, 4, 4), dtype=np.float32)
        target[1:3, 1:3, 1:3] = 0.5
        payload = _volume3d_payload_from_arrays(
            pred,
            target,
            source="error",
            threshold=0.1,
            max_points=10,
        )
        self.assertEqual(payload["source"], "error")
        self.assertGreater(len(payload["positions"]), 0)
        self.assertEqual(len(payload["positions"]), len(payload["colors"]))

    def test_invalid_source_is_rejected(self):
        volume = np.zeros((2, 2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            _volume3d_payload_from_arrays(volume, volume, source="bad", threshold=0.1, max_points=10)

    def test_volume_texture_uses_x_fastest_order(self):
        volume = np.zeros((2, 3, 4), dtype=np.float32)
        volume[1, 2, 3] = 1.0
        payload = _volume_texture_payload_from_arrays(volume, volume, source="prediction")
        self.assertEqual(payload["shape"], [2, 3, 4])
        self.assertEqual(payload["width"], 2)
        self.assertEqual(payload["height"], 3)
        self.assertEqual(payload["depth"], 4)
        self.assertEqual(payload["encoding"], "uint8_base64_x_fastest")
        self.assertGreater(len(payload["data"]), 0)

    def test_volume_texture_rejects_unknown_source(self):
        volume = np.zeros((2, 2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            _volume_texture_payload_from_arrays(volume, volume, source="bad")

    def test_volume_projection_payload_reports_metrics_and_images(self):
        pred = np.array([[0.0, 0.5], [1.0, 0.5]], dtype=np.float32)
        target = np.array([[0.0, 0.25], [1.0, 0.75]], dtype=np.float32)
        payload = _volume_projection_payload_from_arrays(
            pred,
            target,
            source="prediction",
            angle_rad=0.5,
            elapsed_ms=12.0,
        )
        self.assertEqual(payload["kind"], "volume_projection")
        self.assertEqual(payload["source"], "prediction")
        self.assertEqual(payload["angle_rad"], 0.5)
        self.assertEqual(payload["elapsed_ms"], 12.0)
        self.assertIn("psnr", payload["metrics"])
        self.assertIn("ssim", payload["metrics"])
        self.assertIn("mae", payload["metrics"])
        self.assertIn("prediction", payload["images"])
        self.assertIn("target", payload["images"])
        self.assertIn("error", payload["images"])

    def test_volume_projection_rejects_unknown_source(self):
        projection = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            _volume_projection_payload_from_arrays(projection, projection, source="bad", angle_rad=0.0)

    def test_artifact_payload_reports_checkpoint_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            report = root / "training_report.json"
            checkpoint.write_bytes(b"checkpoint")
            report.write_text("{}", encoding="utf-8")
            payload = _artifact_payload(root, checkpoint_path=checkpoint, report_path=report)
            self.assertTrue(payload["checkpoint"]["exists"])
            self.assertTrue(payload["report"]["exists"])


if __name__ == "__main__":
    unittest.main()
