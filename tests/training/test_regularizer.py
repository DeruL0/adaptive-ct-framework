import unittest

import torch

from adaptive_ct.representation.octree import BernsteinOctree
from adaptive_ct.training.regularizer import EdgePenalty
from tests.representation.test_octree import make_step


def _tree():
    model = BernsteinOctree(levels=[2, 4, 8], leaf_types=("constant", "trilinear", "step"), cuda_integrator=False)
    model.split_leaves_batch([0, 5])
    model.to_trilinear([1, 2, 9, 18])
    make_step(model, [3, 15], seed=2)
    return model


class EdgePenaltyTests(unittest.TestCase):
    def test_constant_field_has_no_penalty_and_every_face_is_sampled(self):
        model = _tree()
        with torch.no_grad():
            model.values.fill_(0.4)
        penalty = EdgePenalty(model, delta=0.05)
        value, grad = penalty.value_and_gradient(model.values.detach())
        self.assertAlmostEqual(value, 0.0, places=9)
        self.assertLess(float(grad.abs().max()), 1e-6)
        # 2x2 samples per shared face patch, from the finer side: sampled area = interior face area
        # (2x2x2 grid of unit cells: 12; each split unit cell adds 3 internal unit faces)
        self.assertAlmostEqual(float((penalty.weight * penalty.h).sum()), 12 + 2 * 3, places=5)

    def test_gradient_and_curvature_bound(self):
        torch.manual_seed(0)
        model = _tree()
        penalty = EdgePenalty(model, delta=0.05)
        c = 0.05 + 0.5 * torch.rand(model.values.numel(), dtype=torch.float32)
        value, grad = penalty.value_and_gradient(c)
        step = 1e-3 * torch.randn_like(c)
        numeric = penalty.value_and_gradient(c + step)[0] - penalty.value_and_gradient(c - step)[0]
        self.assertAlmostEqual(numeric, 2 * float((grad * step).sum()), delta=1e-4 * max(1.0, abs(numeric)))
        d = penalty.curvature()
        for _ in range(5):  # separable quadratic bound: R(c + s) <= R(c) + g.s + 1/2 s.diag(d).s
            s = 0.2 * torch.randn_like(c)
            bound = value + float((grad * s).sum()) + 0.5 * float((d * s * s).sum())
            self.assertLessEqual(penalty.value_and_gradient(c + s)[0], bound + 1e-6)

    def test_frozen_weights_give_the_same_value_gradient_and_curvature(self):
        torch.manual_seed(3)
        model = _tree()
        penalty = EdgePenalty(model, delta=0.05)
        penalty.CHUNK = 7  # several chunks, the last one short
        c = 0.05 + 0.5 * torch.rand(model.values.numel())
        value, grad, curv = (*penalty.value_and_gradient(c), penalty.curvature())
        penalty.freeze()
        fvalue, fgrad = penalty.value_and_gradient(c)
        self.assertAlmostEqual(fvalue, value, delta=1e-9 * max(1.0, abs(value)))
        self.assertLess(float((fgrad - grad).abs().max()), 1e-6)
        self.assertLess(float((penalty.curvature() - curv).abs().max()), 1e-6)
        penalty.release()
        self.assertEqual(penalty.value_and_gradient(c)[0], value)  # unfrozen again: computed from the model
        # a frozen penalty is the weights at freezing time: after a geometry change only a new freeze() agrees
        penalty.freeze()
        with torch.no_grad():
            model.step_geometry[:, 3] += 0.05
        fresh = penalty.value_and_gradient(c)[0]
        penalty.release()
        self.assertNotAlmostEqual(penalty.value_and_gradient(c)[0], fresh, places=6)
        self.assertAlmostEqual(penalty.freeze().value_and_gradient(c)[0], penalty.release() or penalty.value_and_gradient(c)[0], places=9)

    def test_decision_penalty_statistics_in_blocks_scaled_into_a_given_array(self):
        import adaptive_ct.refinement.decision as D
        torch.manual_seed(4)
        model = _tree()
        with torch.no_grad():
            model.values.copy_(0.05 + 0.5 * torch.rand(model.values.numel()))
        candidate = torch.tensor([[1.0, 0.0, 0.0, 0.1, 0.05]]).repeat(model.leaf_levels.numel(), 1)
        whole = D.penalty_statistics(model, 0.05, candidate)
        chunk, D.PENALTY_CHUNK = D.PENALTY_CHUNK, 5  # several blocks per face direction, the last one short
        try:
            out = torch.full_like(whole, 1.0)
            D.penalty_statistics(model, 0.05, candidate, scale=3.0, out=out)
            ids = torch.tensor([7, 2, 11])
            part = D.penalty_statistics(model, 0.05, candidate, ids=ids)
        finally:
            D.PENALTY_CHUNK = chunk
        self.assertLess(float((out - (1.0 + 3.0 * whole)).abs().max()), 1e-9 * (1 + float(whole.abs().max())))
        self.assertLess(float((part - whole[ids]).abs().max()), 1e-9 * (1 + float(whole.abs().max())))

    def test_decision_statistics_match_the_penalty(self):
        from adaptive_ct.refinement.decision import penalty_statistics
        torch.manual_seed(1)
        model = _tree()
        with torch.no_grad():
            model.values.copy_(0.05 + 0.5 * torch.rand(model.values.numel()))
        c = model.values.detach()
        candidate = torch.tensor([[1.0, 0.0, 0.0, 0.1, 0.05]]).repeat(model.leaf_levels.numel(), 1)
        delta = 10.0  # every jump in the quadratic range: the surrogate is the penalty itself
        acc = penalty_statistics(model, delta, candidate).double()
        R = EdgePenalty(model, delta).value_and_gradient(c)[0]
        self.assertAlmostEqual(float(acc[:, 96].sum()), 4 * R, delta=1e-5 * max(1.0, 4 * R))  # each sample on both leaves, phi = t^2/2
        iu, ju = torch.triu_indices(8, 8)
        for leaf in torch.nonzero(model.leaf_type() == 1).flatten().tolist():  # refit option = current values: t unchanged
            G = torch.zeros(8, 8, dtype=torch.float64)
            G[iu, ju] = acc[leaf, 2:38]
            G[ju, iu] = acc[leaf, 2:38]
            v = c[model.value_offsets()[leaf] + torch.arange(8)].double()
            self.assertAlmostEqual(float(v @ G @ v - 2 * acc[leaf, 38:46] @ v + acc[leaf, 95]), float(acc[leaf, 96]), places=5)
    def test_geometry_gradient_matches_finite_differences(self):
        torch.manual_seed(5)
        model = _tree()
        c = 0.05 + 0.5 * torch.rand(model.values.numel())
        with torch.no_grad():
            model.values.copy_(c)
        penalty = EdgePenalty(model, delta=0.05)
        stats = penalty.geometry_statistics(c)
        self.assertEqual(tuple(stats.shape), (model.n_step, 20))
        direction = torch.randn(model.n_step, 5)
        g0 = model.step_geometry.clone()
        values = []
        for e in (1e-3, -1e-3):
            with torch.no_grad():
                model.step_geometry[:, :5] = g0[:, :5] + e * direction
            values.append(penalty.value_and_gradient(c)[0])
        with torch.no_grad():
            model.step_geometry.copy_(g0)
        numeric = (values[0] - values[1]) / 2e-3
        self.assertAlmostEqual(numeric, float((stats[:, :5] * direction).sum()), delta=1e-3 * max(1.0, abs(numeric)))
        iu, ju = torch.triu_indices(5, 5)
        G = torch.zeros(model.n_step, 5, 5)
        G[:, iu, ju] = stats[:, 5:]
        G[:, ju, iu] = stats[:, 5:]
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(G).min()), -1e-6)  # positive semidefinite

    def test_face_jump_energy_sits_on_the_leaves_of_the_jump(self):
        from adaptive_ct.training.regularizer import face_jump_energy
        model = BernsteinOctree(levels=[2], leaf_types=("constant",), cuda_integrator=False)
        with torch.no_grad():
            model.values.fill_(0.1)
            model.values[model.find_leaf((0, 1, 0, 0))] = 0.15  # one leaf differs by 0.05
        e = face_jump_energy(model, delta=10.0)  # quadratic range: area t^2 / (2 h) per face, unit faces, h = 1
        k = model.find_leaf((0, 1, 0, 0))
        self.assertAlmostEqual(float(e[k]), 3 * 0.05 ** 2 / 2, places=7)  # its three interior faces
        self.assertAlmostEqual(float(e.sum()), 2 * 3 * 0.05 ** 2 / 2, places=7)  # each face on both leaves
        self.assertEqual(int((e > 0).sum()), 4)  # the leaf and its three neighbours

    def test_merged_constant_pairs_give_the_same_penalty(self):
        from adaptive_ct.training.regularizer import basis_rows, face_samples
        torch.manual_seed(7)
        model = _tree()  # constant, trilinear and step leaves
        c = 0.05 + 0.5 * torch.rand(model.values.numel())
        with torch.no_grad():
            model.values.copy_(c)
        d, R = 0.05, 0.0
        for i, j, ui, uj, a, h in face_samples(model, merge_constant_pairs=False):  # every Gauss point
            wi, ii = basis_rows(model, i, ui); wj, jj = basis_rows(model, j, uj)
            x = ((wi * c[ii]).sum(1) - (wj * c[jj]).sum(1)) / h
            R += float((a * h * torch.where(x.abs() <= d, 0.5 * x * x, d * (x.abs() - 0.5 * d))).sum())
        penalty = EdgePenalty(model, d)
        self.assertLess(penalty.samples, sum(int(i.numel()) for i, *_ in face_samples(model, merge_constant_pairs=False)))
        self.assertAlmostEqual(penalty.value_and_gradient(c)[0], R, delta=1e-4 * max(1.0, R))  # its points are stored in fp16
        self.assertAlmostEqual(penalty.freeze().value_and_gradient(c)[0], R, delta=1e-4 * max(1.0, R))

    def test_large_jumps_cost_linearly(self):
        model = BernsteinOctree(levels=[2], leaf_types=("constant",), cuda_integrator=False)
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
