from pathlib import Path
import unittest

import torch

from adaptive_ct.data import ProjectionSplit
from adaptive_ct.geometry import random_projection_gradient_rays
from adaptive_ct.projection_domain import weighted_projection_gradient_loss


class ProjectionGradientLossTests(unittest.TestCase):
    def _split(self):
        rows = torch.arange(8, dtype=torch.float32)[:, None]
        cols = torch.arange(8, dtype=torch.float32)[None, :]
        projection = rows + 2.0 * cols
        return ProjectionSplit(
            angles=torch.tensor([0.0, 0.7]),
            projections=torch.stack([projection, projection + 1.0]),
            paths=[Path("a"), Path("b")],
        )

    def test_multiscale_pairs_have_derivative_units_and_exact_zero_loss(self):
        generator = torch.Generator().manual_seed(17)
        batch = random_projection_gradient_rays(
            self._split(),
            edge_rays=64,
            detector_h=8,
            detector_w=8,
            samples_per_ray=8,
            strides=[1, 2, 4],
            uniform_fraction=0.5,
            candidate_multiplier=4,
            generator=generator,
        )
        expected = torch.tensor([1.0, 2.0])
        self.assertTrue(torch.all(torch.isin(batch.target_derivative, expected)))
        loss = weighted_projection_gradient_loss(
            batch.ray_batch.target,
            batch,
            mode="uniform",
        )
        self.assertAlmostEqual(float(loss), 0.0, places=7)
        self.assertGreater(
            float(weighted_projection_gradient_loss(torch.zeros_like(batch.ray_batch.target), batch, mode="uniform")),
            0.0,
        )
        half_prediction = 0.5 * batch.ray_batch.target
        signed = weighted_projection_gradient_loss(half_prediction, batch, mode="uniform")
        calibrated = weighted_projection_gradient_loss(
            half_prediction,
            batch,
            mode="uniform",
            magnitude_weight=2.0,
            magnitude_quantile=0.75,
        )
        self.assertGreater(float(calibrated), float(signed))
        moment_calibrated = weighted_projection_gradient_loss(
            half_prediction,
            batch,
            mode="uniform",
            moment_weight=10.0,
            moment_quantile=0.75,
        )
        self.assertGreater(float(moment_calibrated), float(signed))
        endpoint_anchored = weighted_projection_gradient_loss(
            half_prediction,
            batch,
            mode="uniform",
            endpoint_weight=0.25,
            endpoint_quantile=0.75,
        )
        self.assertGreater(float(endpoint_anchored), float(signed))

    def test_sampling_is_reproducible_with_explicit_generator(self):
        kwargs = dict(
            edge_rays=32,
            detector_h=8,
            detector_w=8,
            samples_per_ray=8,
            strides=[1, 2],
            uniform_fraction=0.25,
            candidate_multiplier=3,
        )
        first = random_projection_gradient_rays(
            self._split(), generator=torch.Generator().manual_seed(5), **kwargs
        )
        second = random_projection_gradient_rays(
            self._split(), generator=torch.Generator().manual_seed(5), **kwargs
        )
        self.assertTrue(torch.equal(first.ray_batch.rows, second.ray_batch.rows))
        self.assertTrue(torch.equal(first.ray_batch.cols, second.ray_batch.cols))
        self.assertTrue(torch.equal(first.target_derivative, second.target_derivative))


if __name__ == "__main__":
    unittest.main()
