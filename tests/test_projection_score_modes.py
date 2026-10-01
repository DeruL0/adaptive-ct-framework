import unittest

import torch

from adaptive_ct.bernstein import BernsteinOctree
from adaptive_ct.data import ProjectionSplit
from adaptive_ct.geometry import detector_grid, make_parallel_ray_points
from adaptive_ct.train import _bernstein_projection_residual_leaf_score, _select_global_residual_h_candidates


class ProjectionScoreModeTests(unittest.TestCase):
    def test_global_candidates_are_ranked_by_accuracy_score(self):
        model = BernsteinOctree(levels=[2, 4], max_degree=1, attenuation_shift=0.0)
        leaf_ids = torch.tensor([0, 1, 2, 3])
        scores = torch.tensor([0.2, 0.8, -1.0, 0.5])

        selected, summary = _select_global_residual_h_candidates(
            model,
            leaf_ids,
            scores,
            max_candidates=2,
        )

        self.assertEqual(selected.tolist(), [1, 3])
        self.assertAlmostEqual(summary["proposal_accuracy_score_max"], 0.8, places=6)
        self.assertNotIn("proposal_score_per_byte_max", summary)

    def test_sqrt_exposure_is_distinct_from_both_endpoint_scores(self):
        model = BernsteinOctree(levels=[2, 4], max_degree=1, attenuation_shift=0.0)
        model.elevate_leaves_isotropic_batch(torch.arange(8), (1, 1, 1))
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(torch.linspace(0.1, 0.8, 64)))
        split = ProjectionSplit(
            angles=torch.tensor([0.37]),
            projections=torch.zeros((1, 5, 5)),
            paths=[],
        )
        common = dict(
            model=model,
            split=split,
            level=0,
            detector_h=5,
            detector_w=5,
            samples_per_ray=8,
            ray_chunk=25,
            max_views=1,
            max_rays_per_view=None,
            weight_kwargs={"mode": "uniform"},
            gradient_weight=0.0,
            max_buffer_mb=16.0,
        )
        _, exposure, _ = _bernstein_projection_residual_leaf_score(**common, score_mode="residual_exposure")
        _, sqrt_exposure, _ = _bernstein_projection_residual_leaf_score(**common, score_mode="residual_sqrt_exposure")
        _, density, _ = _bernstein_projection_residual_leaf_score(**common, score_mode="residual_density")
        leaf_ids, squared_density, _ = _bernstein_projection_residual_leaf_score(
            **common,
            score_mode="residual_squared_density",
        )
        self.assertTrue(torch.isfinite(sqrt_exposure).all())
        self.assertTrue(torch.isfinite(squared_density).all())
        self.assertFalse(torch.allclose(sqrt_exposure, exposure))
        self.assertFalse(torch.allclose(sqrt_exposure, density))

        rows, cols = detector_grid(5, 5, torch.device("cpu"))
        rays = make_parallel_ray_points(
            split.angles[0].expand(rows.shape[0]),
            rows,
            cols,
            detector_h=5,
            detector_w=5,
            samples_per_ray=8,
            targets=split.projections[0, rows, cols],
            materialize_points=False,
        )
        residual = model.integrate_ray_batch(rays) - rays.target
        ray_ids, segment_leaf_ids, intervals = model._parallel_ray_segments(rays)
        lengths = intervals[:, 1] - intervals[:, 0]
        numerator = torch.zeros(model.leaf_levels.numel())
        denominator = torch.zeros_like(numerator)
        numerator.scatter_add_(0, segment_leaf_ids, torch.square(residual)[ray_ids] * lengths)
        denominator.scatter_add_(0, segment_leaf_ids, lengths)
        expected = numerator[leaf_ids] / denominator[leaf_ids].clamp_min(1e-8)
        self.assertTrue(torch.allclose(squared_density, expected))


if __name__ == "__main__":
    unittest.main()
