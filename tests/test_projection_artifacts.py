from pathlib import Path
import unittest

import torch

from adaptive_ct.bernstein import BernsteinOctree
from adaptive_ct.compression import build_mact_artifact
from adaptive_ct.data import ProjectionSplit
from adaptive_ct.geometry import detector_grid, make_parallel_ray_points
from adaptive_ct.surface import (
    estimate_wls_coefficient_uncertainty,
    extract_coefficient_surface_points,
)


class ProjectionArtifactTests(unittest.TestCase):
    def _linear_model(self):
        model = BernsteinOctree(levels=[1, 2], max_degree=2, attenuation_shift=0.0)
        model.elevate_degree(0, axis=0)
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(torch.tensor([0.1, 0.9])))
        return model

    def test_mact_uses_coefficient_groups(self):
        model = self._linear_model()
        artifact, summary = build_mact_artifact(model, material_clusters=2)
        self.assertEqual(artifact["representation"], "mact_bernstein_templates")
        self.assertEqual(summary.leaf_count, 1)
        self.assertEqual(summary.template_groups, 1)
        self.assertEqual(summary.original_coefficient_bytes, 8)

    def test_surface_roots_and_wls_uncertainty_are_finite(self):
        model = self._linear_model()
        points = extract_coefficient_surface_points(model, threshold=0.5)
        self.assertGreaterEqual(points.shape[0], 4)
        self.assertTrue(torch.any(torch.abs(points[:, 0]) < 1e-3))

        angles = torch.tensor([0.3, 1.0])
        rows, cols = detector_grid(3, 3, torch.device("cpu"))
        projections = []
        for angle in angles:
            ray_batch = make_parallel_ray_points(
                angle.expand(rows.numel()),
                rows,
                cols,
                detector_h=3,
                detector_w=3,
                samples_per_ray=6,
                targets=torch.zeros(rows.numel()),
            )
            projections.append(model.integrate_ray_batch(ray_batch).detach().reshape(3, 3))
        split = ProjectionSplit(angles, torch.stack(projections), [Path("a"), Path("b")])
        uncertainty = estimate_wls_coefficient_uncertainty(
            model,
            split,
            detector_h=3,
            detector_w=3,
            samples_per_ray=6,
            max_rays=2,
        )
        self.assertEqual(tuple(uncertainty.shape), (2,))
        self.assertTrue(torch.all(torch.isfinite(uncertainty)))


if __name__ == "__main__":
    unittest.main()
