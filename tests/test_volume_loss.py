import unittest

import torch

from adaptive_ct.volume_loss import VolumeSampler, volume_sample_loss


class VolumeLossTests(unittest.TestCase):
    def test_sampler_returns_normalized_points_and_targets(self):
        volume = torch.zeros((4, 4, 4), dtype=torch.float32)
        volume[1:3, 1:3, 1:3] = 0.5
        sampler = VolumeSampler(volume, material_threshold=0.1)
        batch = sampler.sample(32, material_fraction=0.5, boundary_fraction=0.25)
        self.assertEqual(tuple(batch.points.shape), (32, 3))
        self.assertEqual(tuple(batch.target.shape), (32,))
        self.assertGreaterEqual(float(batch.points.min()), -1.0)
        self.assertLessEqual(float(batch.points.max()), 1.0)
        self.assertTrue(torch.all(batch.target >= 0.0))

    def test_volume_sample_loss_modes_are_finite(self):
        pred = torch.tensor([0.0, 0.5, 1.0])
        target = torch.tensor([0.0, 0.25, 1.0])
        for loss_type in ("mse", "l1", "huber"):
            loss = volume_sample_loss(pred, target, loss_type)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreaterEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
