import math
import unittest

import torch

from adaptive_ct.core.geometry import parallel_rays
from adaptive_ct.representation.octree import BernsteinOctree, bernstein_basis, type_ids


def _rays(angles, rows, cols, h, w):
    return parallel_rays(torch.as_tensor(angles, dtype=torch.float32), torch.as_tensor(rows), torch.as_tensor(cols),
                         detector_h=h, detector_w=w, targets=torch.zeros(len(rows)))


def _sampled_integral(model, angle, row, col, h, w, samples=20000):
    """Midpoint-rule line integral through forward_mu (same ray convention as the projector)."""
    a = -float(angle)
    direction = torch.tensor([math.cos(a), math.sin(a)])
    u = -1.0 + (col + 0.5) * 2.0 / w
    z = -1.0 + (row + 0.5) * 2.0 / h
    base = torch.tensor([-math.sin(a), math.cos(a)]) * u
    t = torch.linspace(-2.0, 2.0, samples + 1)
    t = 0.5 * (t[1:] + t[:-1])
    xy = base[None] + direction[None] * t[:, None]
    pts = torch.cat([xy, torch.full((samples, 1), z)], 1)
    inside = torch.all(pts.abs() <= 1.0, dim=1)
    return float((model.forward_mu(pts[inside]).sum() * (4.0 / samples)).detach())


def make_step(model, ids, seed, width=0.1):
    """Turn leaves into step leaves with random direction / position (values set by _random_values)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.as_tensor(ids)
    n = torch.randn(len(ids), 3, generator=g)
    n = n / n.norm(dim=1, keepdim=True)
    d = 0.3 * (torch.rand(len(ids), 1, generator=g) - 0.5)
    geometry = torch.cat([n, d, torch.full((len(ids), 1), float(width))], 1)
    model.set_types(ids, torch.full((len(ids),), 2), torch.full((2 * len(ids),), 0.3), geometry)


def _random_values(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        model.coefficient_logits.copy_(model._coefficients_to_logits(0.05 + 0.9 * torch.rand_like(model.coefficient_logits)))


class BernsteinOctreeTests(unittest.TestCase):
    def test_anisotropic_dyadic_levels_use_all_three_axes(self):
        model = BernsteinOctree(levels=[[2, 3, 4], [4, 6, 8], [8, 12, 16]], leaf_types=("constant", "trilinear", "step"), attenuation_shift=0.0)
        self.assertEqual(model.level_shapes, [(2, 3, 4), (4, 6, 8), (8, 12, 16)])
        self.assertEqual(model.stats().l0_cells, 24)
        model.split_leaves_batch([model.find_leaf((0, 0, 0, 0))])
        model.split_leaves_batch([model.find_leaf((1, 0, 0, 0))])
        leaf_ids = model.resolve_leaf_ids(torch.tensor([[-0.99, -0.99, -0.99], [0.99, 0.99, 0.99]]))
        self.assertEqual(int(model.leaf_levels[leaf_ids[0]]), 2)
        self.assertEqual(int(model.leaf_levels[leaf_ids[1]]), 0)

    def test_basis_is_nonnegative_partition_of_unity(self):
        x = torch.linspace(0.0, 1.0, 31)
        for degree in (0, 1):
            basis = bernstein_basis(degree, x)
            self.assertTrue(torch.all(basis >= 0.0))
            self.assertTrue(torch.allclose(basis.sum(dim=1), torch.ones_like(x), atol=1e-6))

    def test_to_trilinear_and_mixed_batch_split_preserve_the_field(self):
        model = BernsteinOctree(levels=[2, 4], leaf_types=("constant", "trilinear", "step"), attenuation_shift=0.0)
        _random_values(model, 40)
        points = torch.rand((4096, 3)) * 1.998 - 0.999
        before = model.forward_mu(points).detach()
        model.to_trilinear([1, 2, 5])  # mixed constant / trilinear tree, same function
        self.assertTrue(torch.allclose(before, model.forward_mu(points).detach(), atol=1e-6))
        _random_values(model, 41)
        before = model.forward_mu(points).detach()
        trilinear = torch.nonzero(model.leaf_degrees[:, 0] == 1).flatten()[:2]
        constant = torch.nonzero(model.leaf_degrees[:, 0] == 0).flatten()[:2]
        self.assertEqual(model.split_leaves_batch(torch.cat([trilinear, constant])), 4)
        self.assertEqual(int(model.leaf_levels.shape[0]), 8 + 4 * 7)
        self.assertTrue(torch.allclose(before, model.forward_mu(points).detach(), atol=2e-6, rtol=2e-6))

    def test_step_split_preserves_the_field(self):
        model = BernsteinOctree(levels=[2, 4], leaf_types=("constant", "trilinear", "step"), attenuation_shift=0.0)
        make_step(model, [0, 3], seed=1)
        make_step(model, [4], seed=2, width=0.0)
        _random_values(model, 3)
        points = torch.rand((4096, 3)) * 1.998 - 0.999
        before = model.forward_mu(points).detach()
        steps = torch.nonzero(type_ids(model.leaf_degrees) == 2).flatten()
        self.assertEqual(model.split_leaves_batch(steps), 3)
        self.assertEqual(model.stats().leaf_type_counts, (5, 0, 24))
        self.assertTrue(torch.allclose(before, model.forward_mu(points).detach(), atol=1e-5))

    def test_exact_projector_integrates_constant_field(self):
        model = BernsteinOctree(levels=[2, 4], leaf_types=("constant",), attenuation_shift=0.0)
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(torch.full_like(model.coefficient_logits, 0.35)))
        angles, rows, cols = [0.0, math.pi / 7.0, math.pi / 3.0], [1, 2, 3], [2, 1, 3]
        prediction = model.integrate_ray_batch(_rays(angles, rows, cols, 4, 4)).detach()
        for i in range(3):
            self.assertAlmostEqual(float(prediction[i]), _sampled_integral(model, angles[i], rows[i], cols[i], 4, 4), places=3)

    def test_exact_projector_integrates_linear_field(self):
        model = BernsteinOctree(levels=[1, 2], leaf_types=("constant", "trilinear", "step"), attenuation_shift=0.0)
        model.to_trilinear([0])
        corners = torch.tensor([0.1] * 4 + [0.9] * 4)  # value 0.1 at x = -1, 0.9 at x = +1
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(corners))
        # A ray along x: length 2 times the mean value 0.5.
        self.assertAlmostEqual(float(model.integrate_ray_batch(_rays([0.0], [1], [1], 2, 2))[0].detach()), 1.0, places=6)

    def test_exact_projector_matches_dense_quadrature_on_a_mixed_tree(self):
        model = BernsteinOctree(levels=[1, 2, 4], leaf_types=("constant", "trilinear", "step"), attenuation_shift=0.0)
        model.split_leaves_batch([0])
        model.to_trilinear([0, 3, 6])
        model.split_leaves_batch([model.find_leaf((1, 0, 0, 0))])
        make_step(model, [model.find_leaf((1, 1, 1, 1)), model.find_leaf((2, 0, 0, 1))], seed=5)
        make_step(model, [model.find_leaf((1, 1, 0, 1))], seed=6, width=0.0)  # hard step
        _random_values(model, 11)
        for angle, row, col in ((0.43, 2, 1), (1.9, 1, 2), (2.7, 3, 0), (0.8, 3, 3), (2.2, 2, 2)):
            exact = float(model.integrate_ray_batch(_rays([angle], [row], [col], 4, 4))[0].detach())
            self.assertAlmostEqual(exact, _sampled_integral(model, angle, row, col, 4, 4), places=3)

    def test_checkpoint_rebuilds_packed_topology(self):
        model = BernsteinOctree(levels=[2, 4], leaf_types=("constant", "trilinear", "step"), attenuation_shift=0.0)
        model.to_trilinear([0])
        model.split_leaves_batch([model.find_leaf((0, 0, 0, 1))])
        state = model.state_dict()
        restored = BernsteinOctree(levels=[2, 4], leaf_types=("constant", "trilinear", "step"), attenuation_shift=0.0)
        restored.prepare_sparse_from_state_dict(state)
        restored.load_state_dict(state)
        points = torch.rand((256, 3)) * 1.998 - 0.999
        self.assertTrue(torch.allclose(model.forward_mu(points), restored.forward_mu(points)))
        self.assertEqual(model.stats(), restored.stats())

    def test_four_level_topology_is_packed_and_never_allocates_dense_finest_index(self):
        model = BernsteinOctree(levels=[2, 4, 8, 16], leaf_types=("constant", "trilinear", "step"), attenuation_shift=0.0)
        for level in range(3):
            model.split_leaves_batch([model.find_leaf((level, 0, 0, 0))])
        self.assertFalse(any(name.endswith("_index") for name, _ in model.named_buffers()))
        self.assertLess(model.node_child_base.numel(), 16**3)
        leaf_ids = model.resolve_leaf_ids(torch.tensor([[-0.99, -0.99, -0.99], [0.75, 0.75, 0.75]]))
        self.assertEqual(int(model.leaf_levels[leaf_ids[0]]), 3)
        self.assertEqual(int(model.leaf_levels[leaf_ids[1]]), 0)
        self.assertEqual(model.stats().l3_active, 8)


if __name__ == "__main__":
    unittest.main()
