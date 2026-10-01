import unittest

import torch

from adaptive_ct.backend import require_native
from adaptive_ct.geometry import make_parallel_ray_points
from adaptive_ct.model import DynamicVoxelGrid


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DynamicVoxelCudaTests(unittest.TestCase):
    def test_single_leaf_forward_and_backward(self):
        require_native()
        model = DynamicVoxelGrid(levels=[1], attenuation_shift=0.0, init_std=0.0).cuda()
        with torch.no_grad():
            model.leaf_logits.fill_(0.25)
        angles = torch.zeros(4, device="cuda")
        rows = torch.tensor([0, 0, 1, 1], dtype=torch.long, device="cuda")
        cols = torch.tensor([0, 1, 0, 1], dtype=torch.long, device="cuda")
        targets = torch.zeros(4, device="cuda")
        rays = make_parallel_ray_points(
            angles,
            rows,
            cols,
            detector_h=2,
            detector_w=2,
            samples_per_ray=8,
            targets=targets,
            materialize_points=False,
        )
        pred = model.integrate_ray_batch(rays)
        expected = torch.ones_like(pred) * (2.0 * torch.nn.functional.softplus(model.leaf_logits[0]))
        self.assertTrue(torch.allclose(pred, expected, atol=1e-6, rtol=1e-6))

        pred.sum().backward()
        expected_grad = 4.0 * 2.0 * torch.sigmoid(model.leaf_logits.detach()[0])
        self.assertTrue(torch.allclose(model.leaf_logits.grad[0], expected_grad, atol=1e-5, rtol=1e-5))

    def test_axis_aligned_compact_cuda_matches_materialized_sampling(self):
        require_native()
        torch.manual_seed(7)
        model = DynamicVoxelGrid(levels=[2, 4], attenuation_shift=-1.0, init_std=0.0).cuda()
        score = torch.zeros((4, 4, 4), dtype=torch.float32, device="cuda")
        score[0, 0, 0] = 1.0
        model.activate_level_from_score(1, score, active_fraction=1.0 / score.numel(), halo=0)
        with torch.no_grad():
            model.leaf_logits.normal_(mean=0.0, std=0.2)

        rows, cols = torch.meshgrid(
            torch.arange(4, dtype=torch.long, device="cuda"),
            torch.arange(4, dtype=torch.long, device="cuda"),
            indexing="ij",
        )
        rows = rows.reshape(-1)
        cols = cols.reshape(-1)
        angles = torch.zeros(rows.numel(), dtype=torch.float32, device="cuda")
        targets = torch.zeros(rows.numel(), dtype=torch.float32, device="cuda")
        compact = make_parallel_ray_points(
            angles,
            rows,
            cols,
            detector_h=4,
            detector_w=4,
            samples_per_ray=64,
            targets=targets,
            materialize_points=False,
        )
        materialized = make_parallel_ray_points(
            angles,
            rows,
            cols,
            detector_h=4,
            detector_w=4,
            samples_per_ray=64,
            targets=targets,
            materialize_points=True,
        )
        pred_cuda = model.integrate_ray_batch(compact)
        model.cuda_integrator = False
        pred_sampling = model.integrate_ray_batch(materialized)
        self.assertTrue(torch.allclose(pred_cuda, pred_sampling, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
