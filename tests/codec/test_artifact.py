import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from adaptive_ct.codec.artifact import export_compact_octree_artifact, load_compact_octree_state_dict
from adaptive_ct.codec.quantize import quantize_roundtrip_
from adaptive_ct.representation.octree import BernsteinOctree
from tests.representation.test_octree import make_step


class CompactFileTests(unittest.TestCase):
    def _round_trip(self, levels, build):
        model = BernsteinOctree(levels=levels, leaf_types=("constant", "trilinear", "step"), attenuation_shift=0.0)
        build(model)
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(0.1 + 0.8 * torch.rand_like(model.coefficient_logits)))
        quantize_roundtrip_(model, "float16")  # the file must reproduce the deployed (fp16) model
        points = torch.rand((512, 3)) * 1.998 - 0.999
        expected = model.forward_mu(points).detach()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.npz"
            export_compact_octree_artifact(model, path, quantization="float16")
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual([list(row) for row in payload["level_shapes"].tolist()],
                                 [list(s) for s in model.level_shapes])
                self.assertNotIn("leaf_coords", payload.files)  # recovered from the packed hierarchy
            restored = BernsteinOctree(levels=levels, leaf_types=("constant", "trilinear", "step"), attenuation_shift=0.0)
            state, metadata = load_compact_octree_state_dict(restored, path)
            restored.prepare_sparse_from_state_dict(state)
            restored.load_state_dict(state)
            actual = restored.forward_mu(points).detach()
        self.assertEqual(metadata["schema"], "compact_octree_typed_v4")
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_anisotropic_three_type_round_trip(self):
        def build(model):
            model.split_leaves_batch([model.find_leaf((0, 0, 0, 0))])
            model.split_leaves_batch([model.find_leaf((1, 0, 0, 0))])
            model.to_trilinear([model.find_leaf((2, 0, 0, 0)), model.find_leaf((0, 1, 1, 1))])
            make_step(model, [model.find_leaf((1, 1, 1, 1)), model.find_leaf((2, 1, 1, 1))], seed=3)

        self._round_trip([[2, 3, 4], [4, 6, 8], [8, 12, 16]], build)

    def test_four_level_three_type_round_trip(self):
        def build(model):
            for level in range(3):
                model.split_leaves_batch([model.find_leaf((level, 0, 0, 0))])
            model.to_trilinear([model.find_leaf((3, 0, 0, 0))])
            make_step(model, [model.find_leaf((0, 1, 1, 1)), model.find_leaf((3, 1, 1, 1))], seed=4, width=0.0)

        self._round_trip([2, 4, 8, 16], build)


if __name__ == "__main__":
    unittest.main()
