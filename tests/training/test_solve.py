import unittest
from pathlib import Path

import torch

from adaptive_ct.data.dataset import ProjectionSplit
from adaptive_ct.evaluation.render import render_split
from adaptive_ct.representation.octree import BernsteinOctree
from adaptive_ct.training.solve import solve


def _tree():
    m = BernsteinOctree(levels=[2, 4], max_degree=1, attenuation_shift=0.0, cuda_integrator=False, init_std=0.0)
    m.elevate_leaves_isotropic_batch(torch.arange(m.leaf_levels.shape[0]), (1, 1, 1))
    m.split_leaves_batch(torch.tensor([0, 5]))
    return m


class FixedTreeSolverTests(unittest.TestCase):
    def test_recovers_noise_free_coefficients_and_stays_nonnegative(self):
        torch.manual_seed(0)
        truth = _tree()
        with torch.no_grad():
            c_true = 0.05 + 0.5 * torch.rand(truth.coefficient_logits.shape[0])
            c_true[::7] = 0.0  # some exact zeros: the constraint must be active
            truth.coefficient_logits.copy_(truth._coefficients_to_logits(c_true.clamp_min(1e-12)))
        angles = torch.linspace(0.0, 3.1, 24)
        split = ProjectionSplit(angles=angles, projections=torch.zeros(24, 8, 8), paths=[Path("x")] * 24)
        split = ProjectionSplit(angles=angles, projections=render_split(truth, split, detector_h=8, detector_w=8, samples_per_ray=8, ray_chunk=4096).detach(), paths=split.paths)
        model = _tree()
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(torch.full_like(c_true, 0.2)))
        c, hist, converged = solve(model, split, passes=200, tol=1e-4, groups=4, group_passes=5)
        self.assertGreaterEqual(float(c.min()), 0.0)
        self.assertTrue(converged)
        self.assertLess(hist[-1]["pass"], 200)  # stopped by the tolerance, not the pass cap
        self.assertLess(hist[-1]["objective_at_eval_point"], 1e-3 * hist[0]["objective_at_eval_point"])
        self.assertLessEqual(hist[-1]["optimality_residual"], 1e-4)


if __name__ == "__main__":
    unittest.main()
