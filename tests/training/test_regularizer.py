import unittest

import torch

from adaptive_ct.representation.octree import BernsteinOctree
from adaptive_ct.training.regularizer import EdgePenalty
from tests.representation.test_octree import make_step


def _tree():
    model = BernsteinOctree(levels=[2, 4, 8], leaf_types=("constant", "trilinear", "step"), attenuation_shift=0.0, cuda_integrator=False)
    model.split_leaves_batch([0, 5])
    model.to_trilinear([1, 2, 9, 18])
    make_step(model, [3, 15], seed=2)
    return model


class EdgePenaltyTests(unittest.TestCase):
    def test_constant_field_has_no_penalty_and_every_face_is_sampled(self):
        model = _tree()
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(torch.full_like(model.coefficient_logits, 0.4)))
        penalty = EdgePenalty(model, delta=0.05)
        value, grad = penalty.value_and_gradient(model.coefficients().detach())
        self.assertAlmostEqual(value, 0.0, places=9)
        self.assertLess(float(grad.abs().max()), 1e-6)
        # 2x2 samples per shared face patch, from the finer side: sampled area = interior face area
        # (2x2x2 grid of unit cells: 12; each split unit cell adds 3 internal unit faces)
        self.assertAlmostEqual(float(penalty.area.sum()), 12 + 2 * 3, places=5)

    def test_gradient_and_curvature_bound(self):
        torch.manual_seed(0)
        model = _tree()
        penalty = EdgePenalty(model, delta=0.05)
        c = 0.05 + 0.5 * torch.rand(model.coefficient_logits.numel(), dtype=torch.float32)
        value, grad = penalty.value_and_gradient(c)
        step = 1e-3 * torch.randn_like(c)
        numeric = penalty.value_and_gradient(c + step)[0] - penalty.value_and_gradient(c - step)[0]
        self.assertAlmostEqual(numeric, 2 * float((grad * step).sum()), delta=1e-4 * max(1.0, abs(numeric)))
        d = penalty.curvature()
        for _ in range(5):  # separable quadratic bound: R(c + s) <= R(c) + g.s + 1/2 s.diag(d).s
            s = 0.2 * torch.randn_like(c)
            bound = value + float((grad * s).sum()) + 0.5 * float((d * s * s).sum())
            self.assertLessEqual(penalty.value_and_gradient(c + s)[0], bound + 1e-6)

    def test_large_jumps_cost_linearly(self):
        model = BernsteinOctree(levels=[2], leaf_types=("constant",), attenuation_shift=0.0, cuda_integrator=False)
        values = torch.full((8,), 0.1)
        values[model.find_leaf((0, 1, 0, 0))] = 0.1 + 1.0
        small = values.clone()
        small[model.find_leaf((0, 1, 0, 0))] = 0.1 + 0.01
        penalty = EdgePenalty(model, delta=0.05)
        big, tiny = penalty.value_and_gradient(values)[0], penalty.value_and_gradient(small)[0]
        # jump 1.0 vs 0.01: quadratic would give 10^4 x, linear beyond delta gives far less
        self.assertLess(big / tiny, 1e3)


if __name__ == "__main__":
    unittest.main()
