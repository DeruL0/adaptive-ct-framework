import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from adaptive_ct.codec.artifact import SCHEMA, export_octree_artifact, load_octree_artifact
from adaptive_ct.codec.budget import model_bytes
from adaptive_ct.codec.quantize import quantize_roundtrip_
from adaptive_ct.representation.octree import BernsteinOctree
from tests.representation.test_octree import make_step


class StoredFileTests(unittest.TestCase):
    def _round_trip(self, levels, build):
        model = BernsteinOctree(levels=levels, leaf_types=("constant", "trilinear", "step"))
        build(model)
        with torch.no_grad():
            model.values.copy_(0.1 + 0.8 * torch.rand_like(model.values))
        quantize_roundtrip_(model, 0.001)  # the file must reproduce the deployed model exactly
        points = torch.rand((512, 3)) * 1.998 - 0.999
        expected = model.forward_mu(points).detach()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.npz"
            export_octree_artifact(model, path, 0.001)
            with np.load(path, allow_pickle=False) as payload:
                raw = sum(payload[k].nbytes for k in ("root_bits", "masks", "types", "step_geometry", "block_base", "block_width", "values"))
            restored = BernsteinOctree(levels=levels, leaf_types=("constant", "trilinear", "step"))
            header = load_octree_artifact(restored, path)
            actual = restored.forward_mu(points).detach()
        self.assertEqual(header["schema"], SCHEMA)
        self.assertTrue(torch.equal(actual, expected))
        self.assertLessEqual(raw, model_bytes(model, 0.001))  # the budget counts every stored byte

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
