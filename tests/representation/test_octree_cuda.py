import unittest

import torch

from adaptive_ct.core.geometry import parallel_rays
from adaptive_ct.core.projector import has_bernstein_native
from adaptive_ct.representation.octree import BernsteinOctree
from tests.representation.test_octree import make_step


def _rays(ray_count, detector, seed, full_turn=False):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return parallel_rays(
        torch.rand(ray_count, generator=generator, device="cuda") * (6.2831853 if full_turn else 3.14159265),
        torch.randint(0, detector, (ray_count,), generator=generator, device="cuda"),
        torch.randint(0, detector, (ray_count,), generator=generator, device="cuda"),
        detector_h=detector,
        detector_w=detector,
        targets=torch.zeros(ray_count, device="cuda"),
    )


@unittest.skipUnless(torch.cuda.is_available() and has_bernstein_native(), "Bernstein CUDA extension unavailable")
class BernsteinCudaTests(unittest.TestCase):
    def _compare(self, model, rays, atol=5e-5):
        """CUDA traversal vs the PyTorch reference: line integrals and their gradients in the values."""
        torch.manual_seed(7)
        with torch.no_grad():
            model.values.copy_(0.05 + torch.rand_like(model.values))
        model = model.cuda()
        model.values.requires_grad_(True)
        upstream = torch.randn(rays.num_rays, device="cuda")
        out = {}
        for cuda in (False, True):
            model.cuda_integrator = cuda
            model.values.grad = None
            pred = model.integrate_ray_batch(rays)
            pred.backward(upstream)
            out[cuda] = (pred.detach(), model.values.grad.detach().clone())
        for ref, act in zip(out[False], out[True]):
            self.assertTrue(torch.allclose(act, ref, atol=atol, rtol=atol))

    def test_mixed_constant_trilinear_step_tree(self):
        model = BernsteinOctree(levels=[2, 4], leaf_types=("constant", "trilinear", "step"))
        model.to_trilinear([0, 3, 5])
        model.split_leaves_batch([model.find_leaf((0, 1, 1, 1)), model.find_leaf((0, 0, 0, 0))])
        make_step(model, [model.find_leaf((1, 2, 2, 3)), model.find_leaf((0, 1, 0, 0))], seed=4)
        make_step(model, [model.find_leaf((1, 3, 3, 2))], seed=8, width=0.0)
        self._compare(model, _rays(256, 8, 23))

    def test_large_grid_shallow_angle_leaf_assignment(self):
        model = BernsteinOctree(levels=[24, 48, 96], leaf_types=("constant",))
        self._compare(model, _rays(128, 96, 20260628, full_turn=True), atol=2e-5)

    def test_shallow_angles_next_to_finer_leaves(self):
        # At shallow angles the slow coordinate moves less than its float resolution per probe nudge along the
        # ray: the cell after a crossing must come from a probe pushed across the crossed face (0.47 deg failed).
        model = BernsteinOctree(levels=[16, 32, 64], leaf_types=("constant", "trilinear", "step"))
        torch.manual_seed(0)
        model.split_leaves_batch(torch.randperm(model.leaf_count)[:900])
        model.split_leaves_batch(torch.nonzero(model.leaf_levels == 1).flatten()[torch.randperm(7200)[:2000]])
        angles = torch.tensor([0.0082, 0.0123, 0.0614, 0.3068, 1.5790, 3.1293, 3.1416 - 0.0041], device="cuda")
        rows, cols = torch.meshgrid(torch.arange(0, 96, 3, device="cuda"), torch.arange(96, device="cuda"), indexing="ij")
        n = rows.numel()
        rays = parallel_rays(angles.repeat_interleave(n), rows.reshape(-1).repeat(7), cols.reshape(-1).repeat(7),
                             detector_h=96, detector_w=96, targets=torch.zeros(7 * n, device="cuda"))
        self._compare(model, rays, atol=1e-4)  # gradients: sums over many rays in a different order

    def test_four_level_tree(self):
        model = BernsteinOctree(levels=[2, 4, 8, 16], leaf_types=("constant", "trilinear", "step"))
        for level in range(3):
            model.split_leaves_batch([model.find_leaf((level, 0, 0, 0))])
        model.to_trilinear([model.find_leaf((3, 0, 0, 0))])
        self._compare(model, _rays(64, 8, 23))

    def test_anisotropic_levels(self):
        model = BernsteinOctree(levels=[[2, 3, 4], [4, 6, 8], [8, 12, 16]], leaf_types=("constant", "trilinear", "step"))
        model.split_leaves_batch([model.find_leaf((0, 0, 0, 0))])
        model.split_leaves_batch([model.find_leaf((1, 0, 0, 0))])
        model.to_trilinear([model.find_leaf((2, 0, 0, 0))])
        self._compare(model, _rays(64, 8, 23))


if __name__ == "__main__":
    unittest.main()
