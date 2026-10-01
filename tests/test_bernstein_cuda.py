import unittest

import torch

from adaptive_ct.backend import has_bernstein_native
from adaptive_ct.bernstein import BernsteinOctree
from adaptive_ct.geometry import make_parallel_ray_points


@unittest.skipUnless(torch.cuda.is_available() and has_bernstein_native(), "Bernstein CUDA extension unavailable")
class BernsteinCudaTests(unittest.TestCase):
    def _model(self):
        torch.manual_seed(19)
        model = BernsteinOctree(levels=[2, 4], max_degree=3, attenuation_shift=0.0)
        model.elevate_degree(model.find_leaf((0, 0, 0, 0)), 0)
        model.elevate_degree(model.find_leaf((0, 0, 0, 0)), 0)
        model.elevate_degree(model.find_leaf((0, 0, 0, 0)), 1)
        model.split_leaf(model.find_leaf((0, 1, 1, 1)))
        with torch.no_grad():
            physical = 0.05 + 0.9 * torch.rand_like(model.coefficient_logits)
            model.coefficient_logits.copy_(model._coefficients_to_logits(physical))
        return model.cuda()

    def _ray_batch(self, ray_count=96):
        torch.manual_seed(23)
        return make_parallel_ray_points(
            torch.rand(ray_count, device="cuda") * 3.14159265,
            torch.randint(0, 8, (ray_count,), device="cuda"),
            torch.randint(0, 8, (ray_count,), device="cuda"),
            detector_h=8,
            detector_w=8,
            samples_per_ray=16,
            targets=torch.zeros(ray_count, device="cuda"),
            materialize_points=False,
        )

    def test_forward_backward_matches_torch_exact_path(self):
        model = self._model()
        ray_batch = self._ray_batch()
        upstream = torch.randn(ray_batch.num_rays, device="cuda")

        model.cuda_integrator = False
        reference = model.integrate_ray_batch(ray_batch)
        reference.backward(upstream)
        reference_gradient = model.coefficient_logits.grad.detach().clone()
        model.coefficient_logits.grad = None

        model.cuda_integrator = True
        actual = model.integrate_ray_batch(ray_batch)
        actual.backward(upstream)
        actual_gradient = model.coefficient_logits.grad.detach().clone()

        self.assertTrue(torch.allclose(actual, reference, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(actual_gradient, reference_gradient, atol=5e-5, rtol=5e-5))

    def test_segment_contributions_match_reference(self):
        model = self._model()
        ray_batch = self._ray_batch(ray_count=48)
        leaf_count = int(model.leaf_levels.shape[0])

        model.cuda_integrator = False
        ref_rays, ref_leaves, ref_values = model.ray_cell_contributions(ray_batch)
        reference = torch.zeros((ray_batch.num_rays, leaf_count), device="cuda")
        reference.index_put_((ref_rays, ref_leaves), ref_values, accumulate=True)

        model.cuda_integrator = True
        cuda_rays, cuda_leaves, cuda_values = model.ray_cell_contributions(ray_batch)
        actual = torch.zeros_like(reference)
        actual.index_put_((cuda_rays, cuda_leaves), cuda_values, accumulate=True)

        self.assertTrue(torch.allclose(actual, reference, atol=2e-5, rtol=2e-5))

    def test_large_grid_shallow_angle_leaf_assignment(self):
        torch.manual_seed(5)
        model = BernsteinOctree(levels=[24, 48, 96], max_degree=3, attenuation_shift=0.0).cuda()
        with torch.no_grad():
            model.coefficient_logits.copy_(torch.randn_like(model.coefficient_logits) * 0.3)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260628)
        ray_count = 128
        ray_batch = make_parallel_ray_points(
            torch.rand(ray_count, generator=generator, device="cuda") * 6.2831853,
            torch.randint(0, 96, (ray_count,), generator=generator, device="cuda"),
            torch.randint(0, 96, (ray_count,), generator=generator, device="cuda"),
            detector_h=96,
            detector_w=96,
            samples_per_ray=96,
            targets=torch.zeros(ray_count, device="cuda"),
            materialize_points=False,
        )
        model.cuda_integrator = False
        reference = model.integrate_ray_batch(ray_batch)
        reference.sum().backward()
        reference_gradient = model.coefficient_logits.grad.detach().clone()
        model.coefficient_logits.grad = None

        model.cuda_integrator = True
        actual = model.integrate_ray_batch(ray_batch)
        actual.sum().backward()
        actual_gradient = model.coefficient_logits.grad.detach()

        self.assertTrue(torch.allclose(actual, reference, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(actual_gradient, reference_gradient, atol=2e-6, rtol=2e-5))

    def test_four_level_forward_backward_and_segments_match_reference(self):
        model = BernsteinOctree(levels=[2, 4, 8, 16], max_degree=1, attenuation_shift=0.0)
        model.split_leaf(model.find_leaf((0, 0, 0, 0)))
        model.split_leaf(model.find_leaf((1, 0, 0, 0)))
        model.split_leaf(model.find_leaf((2, 0, 0, 0)))
        model.elevate_degree(model.find_leaf((3, 0, 0, 0)), 0)
        with torch.no_grad():
            model.coefficient_logits.copy_(torch.randn_like(model.coefficient_logits) * 0.2)
        model = model.cuda()
        ray_batch = self._ray_batch(ray_count=64)
        upstream = torch.randn(ray_batch.num_rays, device="cuda")

        model.cuda_integrator = False
        reference = model.integrate_ray_batch(ray_batch)
        reference.backward(upstream)
        reference_gradient = model.coefficient_logits.grad.detach().clone()
        model.coefficient_logits.grad = None
        ref_rays, ref_leaves, ref_values = model.ray_cell_contributions(ray_batch)

        model.cuda_integrator = True
        actual = model.integrate_ray_batch(ray_batch)
        actual.backward(upstream)
        actual_gradient = model.coefficient_logits.grad.detach()
        cuda_rays, cuda_leaves, cuda_values = model.ray_cell_contributions(ray_batch)
        reference_segments = torch.zeros((ray_batch.num_rays, model.leaf_levels.numel()), device="cuda")
        actual_segments = torch.zeros_like(reference_segments)
        reference_segments.index_put_((ref_rays, ref_leaves), ref_values, accumulate=True)
        actual_segments.index_put_((cuda_rays, cuda_leaves), cuda_values, accumulate=True)

        self.assertTrue(torch.allclose(actual, reference, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(actual_gradient, reference_gradient, atol=5e-5, rtol=5e-5))
        self.assertTrue(torch.allclose(actual_segments, reference_segments, atol=2e-5, rtol=2e-5))

    def test_anisotropic_forward_backward_and_segments_match_reference(self):
        model = BernsteinOctree(
            levels=[[2, 3, 4], [4, 6, 8], [8, 12, 16]],
            max_degree=1,
            attenuation_shift=0.0,
        )
        model.split_leaf(model.find_leaf((0, 0, 0, 0)))
        model.split_leaf(model.find_leaf((1, 0, 0, 0)))
        model.elevate_degree(model.find_leaf((2, 0, 0, 0)), 2)
        with torch.no_grad():
            model.coefficient_logits.copy_(torch.randn_like(model.coefficient_logits) * 0.2)
        model = model.cuda()
        ray_batch = self._ray_batch(ray_count=64)
        upstream = torch.randn(ray_batch.num_rays, device="cuda")

        model.cuda_integrator = False
        reference = model.integrate_ray_batch(ray_batch)
        reference.backward(upstream)
        reference_gradient = model.coefficient_logits.grad.detach().clone()
        model.coefficient_logits.grad = None
        ref_rays, ref_leaves, ref_values = model.ray_cell_contributions(ray_batch)

        model.cuda_integrator = True
        actual = model.integrate_ray_batch(ray_batch)
        actual.backward(upstream)
        actual_gradient = model.coefficient_logits.grad.detach()
        cuda_rays, cuda_leaves, cuda_values = model.ray_cell_contributions(ray_batch)
        reference_segments = torch.zeros((ray_batch.num_rays, model.leaf_levels.numel()), device="cuda")
        actual_segments = torch.zeros_like(reference_segments)
        reference_segments.index_put_((ref_rays, ref_leaves), ref_values, accumulate=True)
        actual_segments.index_put_((cuda_rays, cuda_leaves), cuda_values, accumulate=True)

        self.assertTrue(torch.allclose(actual, reference, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(actual_gradient, reference_gradient, atol=5e-5, rtol=5e-5))
        self.assertTrue(torch.allclose(actual_segments, reference_segments, atol=2e-5, rtol=2e-5))


if __name__ == "__main__":
    unittest.main()
