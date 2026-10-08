import unittest

import torch

from adaptive_ct.evaluation.deblock import deblock


class DeblockTests(unittest.TestCase):
    def _block(self, jump, step=False):
        # two leaves side by side along x (4 voxels each), values 0.3 | 0.3 + jump
        v = torch.full((1, 1, 8), 0.3)
        v[..., 4:] += jump
        ids = torch.zeros((1, 1, 8), dtype=torch.long)
        ids[..., 4:] = 1
        return deblock(v, ids, torch.tensor([0.02, 0.02]), torch.tensor([step, False]), delta=1.5), v

    def test_small_seam_is_smoothed_symmetrically(self):
        out, v = self._block(0.01)  # below delta * h = 0.03
        self.assertLess(float(out[0, 0, 4] - out[0, 0, 3]), 0.01)
        self.assertAlmostEqual(float(out.sum()), float(v.sum()), places=5)  # what one side gains the other loses
        self.assertTrue(torch.equal(out[..., :3], v[..., :3]) and torch.equal(out[..., 5:], v[..., 5:]))

    def test_edges_and_steps_are_left_alone(self):
        out, v = self._block(0.1)  # an edge: above delta * h
        self.assertTrue(torch.equal(out, v))
        out, v = self._block(0.01, step=True)  # a step leaf's face
        self.assertTrue(torch.equal(out, v))


if __name__ == "__main__":
    unittest.main()
