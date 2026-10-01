import unittest

import torch

from adaptive_ct.model import DynamicVoxelGrid, build_model
from adaptive_ct.refinement import backproject_residual_score, gradient_magnitude, normalize_score


class RefinementTests(unittest.TestCase):
    def test_backproject_zero_residual_is_zero(self):
        residual = torch.zeros((2, 8, 8), dtype=torch.float32)
        angles = torch.tensor([0.0, 0.5], dtype=torch.float32)
        score = backproject_residual_score(residual, angles, resolution=4, voxel_chunk=16)
        self.assertEqual(tuple(score.shape), (4, 4, 4))
        self.assertTrue(torch.allclose(score, torch.zeros_like(score)))

    def test_gradient_and_normalization_are_finite(self):
        volume = torch.zeros((6, 6, 6), dtype=torch.float32)
        volume[3:, :, :] = 1.0
        grad = gradient_magnitude(volume)
        score = normalize_score(grad)
        self.assertTrue(torch.isfinite(score).all())
        self.assertGreater(float(score.max().item()), 0.0)

    def test_dynamic_voxel_split_replaces_parent_with_children(self):
        model = build_model(
            {
                "model": {
                    "representation": "dynamic_leaf_voxel",
                    "levels": [2, 4, 8],
                    "init_std": 0.0,
                }
            }
        )
        self.assertIsInstance(model, DynamicVoxelGrid)
        self.assertEqual(model.stats().leaf_cells, 8)

        score_l1 = torch.zeros((4, 4, 4), dtype=torch.float32)
        score_l1[0, 0, 0] = 1.0
        active_l1 = model.activate_level_from_score(1, score_l1, active_fraction=1.0 / score_l1.numel(), halo=0)
        stats = model.stats()
        self.assertEqual(active_l1, 8)
        self.assertEqual(stats.l0_active, 7)
        self.assertEqual(stats.l1_active, 8)
        self.assertEqual(stats.leaf_cells, 15)

        score_l2 = torch.zeros((8, 8, 8), dtype=torch.float32)
        score_l2[0, 0, 0] = 1.0
        active_l2 = model.activate_level_from_score(2, score_l2, active_fraction=1.0 / score_l2.numel(), halo=0)
        stats = model.stats()
        self.assertEqual(active_l2, 8)
        self.assertEqual(stats.l1_active, 7)
        self.assertEqual(stats.l2_active, 8)
        self.assertEqual(stats.leaf_cells, 22)

    def test_dynamic_voxel_split_uses_exact_top_k_without_zero_tie_expansion(self):
        model = build_model(
            {
                "model": {
                    "representation": "dynamic_leaf_voxel",
                    "levels": [2, 4],
                    "init_std": 0.0,
                }
            }
        )
        score = torch.zeros((4, 4, 4), dtype=torch.float32)
        score[0, 0, 0] = 1.0
        score[1, 1, 1] = 0.5
        model.activate_level_from_score(1, score, active_fraction=2.0 / score.numel(), halo=0)
        stats = model.stats()
        self.assertEqual(stats.l0_active, 7)
        self.assertEqual(stats.l1_active, 8)
        self.assertEqual(stats.leaf_cells, 15)


if __name__ == "__main__":
    unittest.main()
