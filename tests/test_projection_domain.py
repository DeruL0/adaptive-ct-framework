from pathlib import Path
import unittest

import torch

from adaptive_ct.data import ProjectionSplit
from adaptive_ct.projection_domain import (
    projection_weights,
    split_projection_views,
    weighted_projection_loss,
)


class ProjectionDomainTests(unittest.TestCase):
    def test_view_split_is_disjoint_and_reproducible(self):
        split = ProjectionSplit(
            angles=torch.arange(10, dtype=torch.float32),
            projections=torch.zeros(10, 2, 2),
            paths=[Path(str(index)) for index in range(10)],
        )
        train_a, validation_a = split_projection_views(split, validation_fraction=0.2, seed=7)
        train_b, validation_b = split_projection_views(split, validation_fraction=0.2, seed=7)
        self.assertEqual(train_a.paths, train_b.paths)
        self.assertEqual(validation_a.paths, validation_b.paths)
        self.assertTrue(set(train_a.paths).isdisjoint(validation_a.paths))
        self.assertEqual(len(validation_a.paths), 2)

    def test_uniform_weights_reduce_to_mse(self):
        target = torch.tensor([0.1, 0.5, 2.0])
        prediction = torch.tensor([0.2, 0.4, 1.0])
        self.assertTrue(torch.allclose(projection_weights(target, mode="uniform"), torch.ones(3)))
        loss = weighted_projection_loss(prediction, target, mode="uniform")
        self.assertAlmostEqual(float(loss), float(torch.mean((prediction - target) ** 2)), places=6)


if __name__ == "__main__":
    unittest.main()
