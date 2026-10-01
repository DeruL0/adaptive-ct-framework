import math
import unittest

import torch

from adaptive_ct.bernstein import BernsteinOctree, bernstein_basis
from adaptive_ct.geometry import make_parallel_ray_points


class BernsteinOctreeTests(unittest.TestCase):
    def test_anisotropic_dyadic_levels_use_all_three_axes(self):
        model = BernsteinOctree(
            levels=[[2, 3, 4], [4, 6, 8], [8, 12, 16]],
            max_degree=1,
            attenuation_shift=0.0,
        )
        self.assertEqual(model.level_shapes, [(2, 3, 4), (4, 6, 8), (8, 12, 16)])
        self.assertEqual(model.stats().l0_cells, 24)
        model.split_leaf(model.find_leaf((0, 0, 0, 0)))
        model.split_leaf(model.find_leaf((1, 0, 0, 0)))
        points = torch.tensor(
            [
                [-0.99, -0.99, -0.99],
                [0.99, 0.99, 0.99],
                [-0.20, 0.10, 0.30],
            ]
        )
        leaf_ids = model.resolve_leaf_ids(points)
        self.assertEqual(int(model.leaf_levels[leaf_ids[0]]), 2)
        self.assertEqual(int(model.leaf_levels[leaf_ids[1]]), 0)
        self.assertEqual(tuple(model.decoded_l0().shape), (2, 3, 4))

    def test_basis_is_nonnegative_partition_of_unity(self):
        x = torch.linspace(0.0, 1.0, 31)
        for degree in range(5):
            basis = bernstein_basis(degree, x)
            self.assertTrue(torch.all(basis >= 0.0))
            self.assertTrue(torch.allclose(basis.sum(dim=1), torch.ones_like(x), atol=1e-6))

    def test_degree_elevation_preserves_the_field(self):
        torch.manual_seed(3)
        model = BernsteinOctree(levels=[2, 4], max_degree=3, attenuation_shift=0.0)
        points = torch.rand((1024, 3)) * 1.998 - 0.999
        before = model.forward_mu(points).detach()
        model.elevate_degree(0, axis=0)
        after = model.forward_mu(points).detach()
        self.assertTrue(torch.allclose(before, after, atol=1e-6, rtol=1e-6))
        self.assertEqual(tuple(model.leaf_degrees[model.find_leaf((0, 0, 0, 0))].tolist()), (1, 0, 0))

    def test_de_casteljau_split_preserves_the_field(self):
        torch.manual_seed(4)
        model = BernsteinOctree(levels=[2, 4], max_degree=3, attenuation_shift=0.0)
        leaf_id = model.find_leaf((0, 0, 0, 0))
        model.elevate_degree(leaf_id, axis=0)
        leaf_id = model.find_leaf((0, 0, 0, 0))
        with torch.no_grad():
            start = int(model.coefficient_offsets[leaf_id])
            stop = int(model.coefficient_offsets[leaf_id + 1])
            model.coefficient_logits[start:stop].copy_(
                model._coefficients_to_logits(torch.tensor([0.1, 0.9]))
            )
        points = torch.rand((2048, 3)) * 1.998 - 0.999
        before = model.forward_mu(points).detach()
        leaf_id = model.find_leaf((0, 0, 0, 0))
        change = model.split_leaf(leaf_id)
        after = model.forward_mu(points).detach()
        self.assertEqual(len(change.new_leaf_keys), 8)
        self.assertTrue(torch.allclose(before, after, atol=2e-6, rtol=2e-6))

    def test_vectorized_trilinear_batch_split_preserves_the_field(self):
        torch.manual_seed(40)
        model = BernsteinOctree(levels=[2, 4], max_degree=1, attenuation_shift=0.0)
        model.elevate_leaves_isotropic_batch(
            torch.arange(model.leaf_levels.shape[0]), (1, 1, 1)
        )
        with torch.no_grad():
            values = 0.05 + 0.9 * torch.rand_like(model.coefficient_logits)
            model.coefficient_logits.copy_(model._coefficients_to_logits(values))
        points = torch.rand((4096, 3)) * 1.998 - 0.999
        before = model.forward_mu(points).detach()
        model.split_leaves_batch(torch.tensor([0, 3, 7]))
        after = model.forward_mu(points).detach()
        self.assertTrue(torch.all(model.leaf_degrees == 1))
        self.assertTrue(torch.allclose(before, after, atol=2e-6, rtol=2e-6))

    def test_exact_projector_integrates_constant_field(self):
        model = BernsteinOctree(
            levels=[2, 4],
            max_degree=2,
            attenuation_shift=0.0,
            integration_mode="exact",
        )
        coefficient = torch.tensor(0.35)
        with torch.no_grad():
            model.coefficient_logits.copy_(
                model._coefficients_to_logits(torch.full_like(model.coefficient_logits, coefficient))
            )
        angles = torch.tensor([0.0, math.pi / 7.0, math.pi / 3.0])
        rows = torch.tensor([1, 2, 3])
        cols = torch.tensor([2, 1, 3])
        ray_batch = make_parallel_ray_points(
            angles,
            rows,
            cols,
            detector_h=4,
            detector_w=4,
            samples_per_ray=8,
            targets=torch.zeros(3),
        )
        prediction = model.integrate_ray_batch(ray_batch)
        # Expected path length through the square is recovered from the batch.
        expected = coefficient * ray_batch.step * ray_batch.samples_per_ray
        self.assertTrue(torch.allclose(prediction, expected, atol=1e-5, rtol=1e-5))

    def test_exact_projector_integrates_linear_bernstein_field(self):
        model = BernsteinOctree(levels=[1, 2], max_degree=2, attenuation_shift=0.0)
        model.elevate_degree(0, axis=0)
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(torch.tensor([0.1, 0.9])))
        ray_batch = make_parallel_ray_points(
            torch.tensor([0.0]),
            torch.tensor([1]),
            torch.tensor([1]),
            detector_h=2,
            detector_w=2,
            samples_per_ray=3,
            targets=torch.zeros(1),
        )
        # Integral of the x-linear field is length 2 times mean coefficient 0.5.
        self.assertAlmostEqual(float(model.integrate_ray_batch(ray_batch)[0].detach()), 1.0, places=6)

    def test_exact_projector_matches_dense_quadrature_for_piecewise_polynomials(self):
        torch.manual_seed(11)
        model = BernsteinOctree(levels=[1, 2], max_degree=2, attenuation_shift=0.0)
        model.elevate_degree(0, axis=0)
        model.elevate_degree(0, axis=0)
        model.elevate_degree(0, axis=1)
        change = model.split_leaf(0)
        with torch.no_grad():
            physical = 0.05 + 0.9 * torch.rand_like(model.coefficient_logits)
            model.coefficient_logits.copy_(model._coefficients_to_logits(physical))
        angle = torch.tensor([0.43])
        rows = torch.tensor([2])
        cols = torch.tensor([1])
        exact_batch = make_parallel_ray_points(
            angle,
            rows,
            cols,
            detector_h=4,
            detector_w=4,
            samples_per_ray=16,
            targets=torch.zeros(1),
        )
        exact = model.integrate_ray_batch(exact_batch).detach()
        model.integration_mode = "sampled"
        sampled_batch = make_parallel_ray_points(
            angle,
            rows,
            cols,
            detector_h=4,
            detector_w=4,
            samples_per_ray=20000,
            targets=torch.zeros(1),
        )
        sampled = model.integrate_ray_batch(sampled_batch).detach()
        self.assertTrue(torch.allclose(exact, sampled, atol=2e-4, rtol=2e-4))

    def test_sibling_merge_recovers_an_unchanged_split_field(self):
        model = BernsteinOctree(levels=[1, 2], max_degree=2, attenuation_shift=0.0)
        model.elevate_degree(0, axis=0)
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(torch.tensor([0.1, 0.9])))
        points = torch.rand((512, 3)) * 1.998 - 0.999
        before = model.forward_mu(points).detach()
        change = model.split_leaf(0)
        child_ids = [model.find_leaf(key) for key in change.new_leaf_keys]
        model.merge_siblings(child_ids)
        after = model.forward_mu(points).detach()
        self.assertTrue(torch.allclose(before, after, atol=1e-5, rtol=1e-5))

    def test_checkpoint_rebuilds_packed_topology(self):
        model = BernsteinOctree(levels=[2, 4], max_degree=2, attenuation_shift=0.0)
        model.elevate_degree(0, 1)
        leaf_id = model.find_leaf((0, 0, 0, 1))
        model.split_leaf(leaf_id)
        state = model.state_dict()
        restored = BernsteinOctree(levels=[2, 4], max_degree=2, attenuation_shift=0.0)
        restored.prepare_sparse_from_state_dict(state)
        restored.load_state_dict(state)
        points = torch.rand((256, 3)) * 1.998 - 0.999
        self.assertTrue(torch.allclose(model.forward_mu(points), restored.forward_mu(points)))
        self.assertEqual(model.stats(), restored.stats())

    def test_four_level_topology_is_packed_and_never_allocates_dense_finest_index(self):
        model = BernsteinOctree(levels=[2, 4, 8, 16], max_degree=1, attenuation_shift=0.0)
        model.split_leaf(model.find_leaf((0, 0, 0, 0)))
        model.split_leaf(model.find_leaf((1, 0, 0, 0)))
        model.split_leaf(model.find_leaf((2, 0, 0, 0)))
        self.assertFalse(any(name.endswith("_index") for name, _ in model.named_buffers()))
        self.assertLess(model.node_child_base.numel(), 16**3)
        points = torch.tensor([[-0.99, -0.99, -0.99], [0.75, 0.75, 0.75]])
        leaf_ids = model.resolve_leaf_ids(points)
        self.assertEqual(int(model.leaf_levels[leaf_ids[0]]), 3)
        self.assertEqual(int(model.leaf_levels[leaf_ids[1]]), 0)
        self.assertEqual(model.stats().l3_active, 8)

    def test_balance_2to1_splits_face_adjacent_coarse_neighbor(self):
        model = BernsteinOctree(
            levels=[2, 4, 8],
            max_degree=1,
            attenuation_shift=0.0,
            balance_2to1=True,
            max_leaf_count=256,
        )
        model.split_leaf(model.find_leaf((0, 0, 0, 0)))
        model.split_leaf(model.find_leaf((1, 1, 0, 0)))
        self.assertIsNotNone(model.find_leaf((0, 1, 0, 0)))
        summary = model.balance_2to1()
        self.assertTrue(summary["complete"])
        self.assertGreater(summary["split_parents"], 0)
        self.assertIsNone(model.find_leaf((0, 1, 0, 0)))


if __name__ == "__main__":
    unittest.main()
