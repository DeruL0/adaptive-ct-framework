import unittest

import torch

from adaptive_ct.backend import has_native, project_dense_parallel
from adaptive_ct.ceiling_diagnostics import differentiable_dense_project, grid_sample_dense_project


@unittest.skipUnless(torch.cuda.is_available() and has_native(), "CUDA native extension required")
class DenseProjectorTests(unittest.TestCase):
    def test_differentiable_dense_projector_matches_native_forward(self):
        torch.manual_seed(7)
        volume = torch.rand((8, 8, 8), device="cuda", dtype=torch.float32)
        # Avoid angles whose midpoint samples land exactly on an outer voxel
        # centre; the native CUDA kernel and PyTorch may round that boundary in
        # opposite directions while agreeing everywhere in the interior.
        angles = torch.tensor([0.17, 0.43, 0.91], device="cuda")
        detector_h = 8
        detector_w = 8
        rows, cols = torch.meshgrid(
            torch.arange(detector_h, device="cuda"),
            torch.arange(detector_w, device="cuda"),
            indexing="ij",
        )
        rows = rows.reshape(-1)
        cols = cols.reshape(-1)
        repeated_angles = angles[:, None].expand(-1, rows.numel()).reshape(-1)
        repeated_rows = rows[None, :].expand(angles.numel(), -1).reshape(-1)
        repeated_cols = cols[None, :].expand(angles.numel(), -1).reshape(-1)
        actual = differentiable_dense_project(
            volume,
            repeated_angles,
            repeated_rows,
            repeated_cols,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=11,
        ).reshape(angles.numel(), detector_h, detector_w)
        expected = project_dense_parallel(
            volume,
            angles,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=11,
        )
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)

    def test_dense_projector_propagates_volume_gradient(self):
        volume = torch.zeros((4, 4, 4), device="cuda", requires_grad=True)
        prediction = differentiable_dense_project(
            volume,
            torch.tensor([0.2], device="cuda"),
            torch.tensor([2], device="cuda"),
            torch.tensor([2], device="cuda"),
            detector_h=4,
            detector_w=4,
            samples_per_ray=8,
        )
        prediction.sum().backward()
        self.assertIsNotNone(volume.grad)
        self.assertGreater(float(volume.grad.abs().sum().item()), 0.0)

    def test_fast_grid_projector_matches_reference_away_from_outer_centres(self):
        torch.manual_seed(11)
        volume = torch.rand((9, 9, 9), device="cuda", dtype=torch.float32)
        angles = torch.tensor([0.23, 0.71], device="cuda")
        rows = torch.tensor([2, 4, 6, 3], device="cuda")
        cols = torch.tensor([2, 4, 6, 5], device="cuda")
        ray_angles = angles[torch.tensor([0, 0, 1, 1], device="cuda")]
        expected = differentiable_dense_project(
            volume,
            ray_angles,
            rows,
            cols,
            detector_h=9,
            detector_w=9,
            samples_per_ray=13,
        )
        actual = grid_sample_dense_project(
            volume,
            ray_angles,
            rows,
            cols,
            detector_h=9,
            detector_w=9,
            samples_per_ray=13,
        )
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
