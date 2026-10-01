import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from adaptive_ct.representation.octree import BernsteinOctree
from adaptive_ct.codec.artifact import export_compact_octree_artifact, load_compact_octree_state_dict


class CompactV3Tests(unittest.TestCase):
    def test_anisotropic_packed_v3_round_trip(self):
        levels = [[2, 3, 4], [4, 6, 8], [8, 12, 16]]
        model = BernsteinOctree(levels=levels, max_degree=2, attenuation_shift=0.0)
        model.split_leaf(model.find_leaf((0, 0, 0, 0)))
        model.split_leaf(model.find_leaf((1, 0, 0, 0)))
        model.elevate_degree(model.find_leaf((2, 0, 0, 0)), 1)
        with torch.no_grad():
            values = 0.1 + 0.8 * torch.rand_like(model.coefficient_logits)
            model.coefficient_logits.copy_(model._coefficients_to_logits(values))
        points = torch.rand((512, 3)) * 1.998 - 0.999
        expected = model.forward_mu(points).detach()

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "anisotropic.npz"
            export_compact_octree_artifact(
                model,
                path,
                quantization="float16",
                topology="packed_hierarchy",
            )
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(payload["level_shapes"].tolist(), levels)
                self.assertNotIn("level_resolutions", payload.files)
            restored = BernsteinOctree(levels=levels, max_degree=2, attenuation_shift=0.0)
            state, metadata = load_compact_octree_state_dict(restored, path)
            restored.prepare_sparse_from_state_dict(state)
            restored.load_state_dict(state)
            actual = restored.forward_mu(points).detach()
        self.assertEqual(metadata["schema"], "compact_bernstein_octree_packed_v3")
        self.assertTrue(torch.allclose(actual, expected, atol=5e-4, rtol=5e-4))

    def test_legacy_v1_and_v2_remain_readable(self):
        model = BernsteinOctree(levels=[2, 4, 8], max_degree=2, attenuation_shift=0.0)
        model.split_leaf(model.find_leaf((0, 0, 0, 0)))
        model.split_leaf(model.find_leaf((1, 0, 0, 0)))
        with torch.no_grad():
            values = 0.1 + 0.8 * torch.rand_like(model.coefficient_logits)
            model.coefficient_logits.copy_(model._coefficients_to_logits(values))
        points = torch.rand((256, 3)) * 1.998 - 0.999
        expected = model.forward_mu(points).detach()
        cases = {
            "explicit": "compact_bernstein_octree_v1",
            "split_masks": "compact_bernstein_octree_splitmask_v2",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            for topology, expected_schema in cases.items():
                with self.subTest(topology=topology):
                    path = Path(temporary_directory) / f"{topology}.npz"
                    export_compact_octree_artifact(
                        model,
                        path,
                        quantization="float16",
                        topology=topology,
                    )
                    restored = BernsteinOctree(levels=[2, 4, 8], max_degree=2, attenuation_shift=0.0)
                    state, metadata = load_compact_octree_state_dict(restored, path)
                    restored.prepare_sparse_from_state_dict(state)
                    restored.load_state_dict(state)
                    actual = restored.forward_mu(points).detach()
                    self.assertEqual(metadata["schema"], expected_schema)
                    self.assertTrue(torch.allclose(actual, expected, atol=5e-4, rtol=5e-4))

    def test_arbitrary_depth_mixed_degree_round_trip(self):
        model = BernsteinOctree(levels=[2, 4, 8, 16], max_degree=2, attenuation_shift=0.0)
        model.split_leaf(model.find_leaf((0, 0, 0, 0)))
        model.split_leaf(model.find_leaf((1, 0, 0, 0)))
        model.split_leaf(model.find_leaf((2, 0, 0, 0)))
        model.elevate_degree(model.find_leaf((3, 0, 0, 0)), 0)
        with torch.no_grad():
            values = 0.1 + 0.8 * torch.rand_like(model.coefficient_logits)
            model.coefficient_logits.copy_(model._coefficients_to_logits(values))
        points = torch.rand((512, 3)) * 1.998 - 0.999
        expected = model.forward_mu(points).detach()

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.npz"
            export_compact_octree_artifact(model, path, quantization="float16", topology="auto")
            with np.load(path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"][0]))
                self.assertEqual(metadata["schema"], "compact_bernstein_octree_packed_v3")
                self.assertNotIn("leaf_coords", payload.files)
            restored = BernsteinOctree(levels=[2, 4, 8, 16], max_degree=2, attenuation_shift=0.0)
            state, _ = load_compact_octree_state_dict(restored, path)
            restored.prepare_sparse_from_state_dict(state)
            restored.load_state_dict(state)
            actual = restored.forward_mu(points).detach()
        self.assertTrue(torch.allclose(actual, expected, atol=5e-4, rtol=5e-4))
