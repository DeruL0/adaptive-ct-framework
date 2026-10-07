import unittest
from pathlib import Path

import torch

from adaptive_ct.data.dataset import ProjectionSplit
from adaptive_ct.evaluation.render import render_split
from adaptive_ct.representation.octree import BernsteinOctree
from adaptive_ct.training.solve import curvature, gradient, solve


def _tree():
    m = BernsteinOctree(levels=[2, 4], leaf_types=("constant", "trilinear", "step"), cuda_integrator=False)
    m.to_trilinear(torch.arange(m.leaf_levels.shape[0]))
    m.split_leaves_batch(torch.tensor([0, 5]))
    return m


class FixedTreeSolverTests(unittest.TestCase):
    def test_recovers_noise_free_coefficients_and_stays_nonnegative(self):
        torch.manual_seed(0)
        truth = _tree()
        with torch.no_grad():
            c_true = 0.05 + 0.5 * torch.rand(truth.values.shape[0])
            c_true[::7] = 0.0  # some exact zeros: the constraint must be active
            truth.values.copy_(c_true.clamp_min(0))
        angles = torch.linspace(0.0, 3.1, 24)
        split = ProjectionSplit(angles=angles, projections=torch.zeros(24, 8, 8), paths=[Path("x")] * 24)
        split = ProjectionSplit(angles=angles, projections=render_split(truth, split, detector_h=8, detector_w=8, ray_chunk=4096).detach(), paths=split.paths)
        model = _tree()
        with torch.no_grad():
            model.values.fill_(0.2)
        c, hist, converged = solve(model, split, passes=200, tol=1e-4, groups=4, group_passes=5)
        self.assertGreaterEqual(float(c.min()), 0.0)
        self.assertTrue(converged)
        self.assertLess(hist[-1]["pass"], 200)  # stopped by the tolerance, not the pass cap
        self.assertLess(hist[-1]["objective_at_eval_point"], 1e-3 * hist[0]["objective_at_eval_point"])
        self.assertLessEqual(hist[-1]["optimality_residual"], 1e-4)
        # convergence is judged at the returned coefficients: recompute the residual there independently
        model.values.requires_grad_(True)
        g, _ = gradient(model, c, split.angles, split.projections, list(range(24)))
        d = curvature(model, split.angles, split.projections, list(range(24)))
        rho = float((c - (c - g / d).clamp_min(0)).norm() / c.norm())
        self.assertLessEqual(rho, 1e-4)
        self.assertAlmostEqual(rho, hist[-1]["optimality_residual_at_returned"], delta=1e-6)

    def test_continuation_reaches_the_same_solution_without_new_warm_up(self):
        torch.manual_seed(1)
        truth = _tree()
        with torch.no_grad():
            truth.values.copy_(0.05 + 0.5 * torch.rand(truth.values.shape[0]))
        angles = torch.linspace(0.0, 3.1, 24)
        split = ProjectionSplit(angles=angles, projections=torch.zeros(24, 8, 8), paths=[Path("x")] * 24)
        split = ProjectionSplit(angles=angles, projections=render_split(truth, split, detector_h=8, detector_w=8, ray_chunk=4096).detach(), paths=split.paths)

        def fresh():
            m = _tree()
            with torch.no_grad():
                m.values.fill_(0.2)
            return m

        one, state = fresh(), {}
        _, h1, done1 = solve(one, split, passes=300, tol=1e-4, groups=4, group_passes=3)
        split_run, state = fresh(), {}
        _, ha, done_a = solve(split_run, split, passes=12, tol=1e-4, groups=4, group_passes=3, state=state)
        self.assertFalse(done_a)  # stopped by the pass cap
        total = 12
        while not done_a and total < 400:
            _, hb, done_a = solve(split_run, split, passes=12, tol=1e-4, groups=4, group_passes=3, state=state)
            self.assertEqual(hb[0]["phase"], "full")  # continuation: no grouped warm-up again
            total += 12
        self.assertTrue(done1 and done_a)
        self.assertLess(float((one.values - split_run.values).abs().max()), 5e-3)


if __name__ == "__main__":
    unittest.main()
