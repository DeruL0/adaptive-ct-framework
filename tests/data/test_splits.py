from pathlib import Path
import unittest

import torch

from adaptive_ct.data.dataset import ProjectionSplit
from adaptive_ct.data.splits import split_projection_views


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


if __name__ == "__main__":
    unittest.main()
