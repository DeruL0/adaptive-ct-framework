import unittest
from pathlib import Path

import torch

from adaptive_ct.data.dataset import ProjectionSplit
from adaptive_ct.evaluation.render import render_split
from adaptive_ct.representation.octree import BernsteinOctree
from adaptive_ct.training.regularizer import EdgePenalty
from adaptive_ct.training.solve import gradient, solve


def _tree():
    m = BernsteinOctree(levels=[2, 4], leaf_types=("constant", "trilinear", "step"), cuda_integrator=False)
    m.to_trilinear(torch.arange(m.leaf_levels.shape[0]))
    m.split_leaves_batch(torch.tensor([0, 5]))
    return m


def _problem(zeros=False, seed=0, views=96):
    torch.manual_seed(seed)
    truth = _tree()
    with torch.no_grad():
        c_true = 0.05 + 0.5 * torch.rand(truth.values.shape[0])
        if zeros:
            c_true[::7] = 0.0  # some exact zeros: the constraint must be active
        truth.values.copy_(c_true)
    angles = torch.linspace(0.0, 3.1, views)
    blank = ProjectionSplit(angles=angles, projections=torch.zeros(views, 8, 8), paths=[Path("x")] * views)
    split = ProjectionSplit(angles=angles, projections=render_split(truth, blank, detector_h=8, detector_w=8, ray_chunk=4096).detach(), paths=blank.paths)
    return split


def _start():
    m = _tree()
    with torch.no_grad():
        m.values.fill_(0.2)
    return m


def _data_term(model, split):
    model.values.requires_grad_(True)
    _, J = gradient(model, model.values.detach().clone(), split.angles, split.projections, list(range(int(split.angles.numel()))))
    model.values.requires_grad_(False)
    return float(J)


class FixedTreeSolverTests(unittest.TestCase):
    def test_recovers_noise_free_values_and_stays_nonnegative(self):
        split = _problem(zeros=True)
        model = _start()
        before = _data_term(model, split)
        c, hist, converged = solve(model, split, subsets=4, tol=1e-3, max_epochs=800, min_epochs=3)
        self.assertGreaterEqual(float(c.min()), 0.0)
        self.assertTrue(converged)
        self.assertLess(len(hist), 800)  # stopped by the tolerance, not the epoch cap
        self.assertLess(_data_term(model, split), 1e-3 * before)
        self.assertLessEqual(abs(hist[-1]["relative_decrease"]), 1e-3)
        self.assertTrue(torch.equal(c, model.values))  # the model holds the returned values

    def test_diverging_subsets_fall_back_to_fewer_ones(self):
        # 12 views per subset diverge on this problem (data term 150 after 60 epochs without the guard)
        split = _problem(zeros=True)
        model = _start()
        before = _data_term(model, split)
        c, hist, converged = solve(model, split, subsets=8, tol=1e-3, max_epochs=800, min_epochs=3)
        restarts = [h["restart_with_subsets"] for h in hist if "restart_with_subsets" in h]
        self.assertTrue(restarts)
        self.assertTrue(converged)
        self.assertLess(_data_term(model, split), 1e-3 * before)
        self.assertGreaterEqual(float(c.min()), 0.0)

    def test_objective_estimate_follows_the_true_objective(self):
        # the estimate (subset losses right after their updates + beta R) is what stops the solve: it must track J
        split = _problem()
        model = _start()
        penalty = EdgePenalty(model, 0.02)
        _, hist, _ = solve(model, split, subsets=4, tol=-1e9, max_epochs=8, min_epochs=1, penalty=penalty, penalty_weight=50.0)
        exact = _data_term(model, split) + 50.0 * penalty.value_and_gradient(model.values.detach())[0]
        self.assertAlmostEqual(hist[-1]["objective_estimate"], exact, delta=0.1 * exact + 1e-3)
        self.assertLess(hist[-1]["objective_estimate"], hist[0]["objective_estimate"])

    def test_continuation_keeps_counting_from_the_same_state(self):
        split = _problem(seed=1)
        part, state = _start(), {}
        _, first, done = solve(part, split, subsets=4, tol=1e-3, max_epochs=20, min_epochs=3, state=state)
        self.assertFalse(done)  # stopped by the epoch cap
        total = len(first)
        while not done and total < 800:
            _, more, done = solve(part, split, subsets=4, tol=1e-3, max_epochs=20, min_epochs=3, state=state)
            self.assertEqual(more[0]["epoch"], total + 1)  # epochs (and the decay of rho) go on, no new start
            total += len(more)
        self.assertTrue(done)
        whole = _start()
        _, hist, done_whole = solve(whole, split, subsets=4, tol=1e-3, max_epochs=800, min_epochs=3)
        self.assertTrue(done_whole)
        self.assertLess(float((whole.values - part.values).abs().max()), 1e-3)  # the same solution in the end


if __name__ == "__main__":
    unittest.main()
