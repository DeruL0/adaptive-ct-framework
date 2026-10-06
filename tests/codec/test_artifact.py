import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from adaptive_ct.codec.artifact import export_compact_octree_artifact, load_compact_octree_state_dict
from adaptive_ct.representation.octree import BernsteinOctree


class CompactFileTests(unittest.TestCase):
    def _round_trip(self, levels, build):
        model = BernsteinOctree(levels=levels, max_degree=1, attenuation_shift=0.0)
        build(model)
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(0.1 + 0.8 * torch.rand_like(model.coefficient_logits)))
        points = torch.rand((512, 3)) * 1.998 - 0.999
        expected = model.forward_mu(points).detach()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.npz"
            export_compact_octree_artifact(model, path, quantization="float16")
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual([list(row) for row in payload["level_shapes"].tolist()],
                                 [list(s) for s in model.level_shapes])
                self.assertNotIn("leaf_coords", payload.files)  # recovered from the packed hierarchy
            restored = BernsteinOctree(levels=levels, max_degree=1, attenuation_shift=0.0)
            state, metadata = load_compact_octree_state_dict(restored, path)
            restored.prepare_sparse_from_state_dict(state)
            restored.load_state_dict(state)
            actual = restored.forward_mu(points).detach()
        self.assertEqual(metadata["schema"], "compact_bernstein_octree_packed_v3")
        self.assertTrue(torch.allclose(actual, expected, atol=5e-4, rtol=5e-4))  # fp16 storage

    def test_anisotropic_mixed_round_trip(self):
        def build(model):
            model.split_leaves_batch([model.find_leaf((0, 0, 0, 0))])
            model.split_leaves_batch([model.find_leaf((1, 0, 0, 0))])
            model.to_trilinear([model.find_leaf((2, 0, 0, 0)), model.find_leaf((0, 1, 1, 1))])

        self._round_trip([[2, 3, 4], [4, 6, 8], [8, 12, 16]], build)

    def test_four_level_mixed_round_trip(self):
        def build(model):
            for level in range(3):
                model.split_leaves_batch([model.find_leaf((level, 0, 0, 0))])
            model.to_trilinear([model.find_leaf((3, 0, 0, 0))])

        self._round_trip([2, 4, 8, 16], build)


if __name__ == "__main__":
    unittest.main()
