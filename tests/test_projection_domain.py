from pathlib import Path
import unittest

import torch

from adaptive_ct.bernstein import BernsteinOctree
from adaptive_ct.data import ProjectionSplit
from adaptive_ct.geometry import detector_grid, make_parallel_ray_points
from adaptive_ct.projection_domain import (
    CandidateAction,
    coefficient_diagnostics,
    evaluate_candidate,
    propose_forward_actions,
    split_projection_views,
)


def _render(model, angles, detector_h=4, detector_w=4):
    rows, cols = detector_grid(detector_h, detector_w, torch.device("cpu"))
    projections = []
    for angle in angles:
        ray_batch = make_parallel_ray_points(
            angle.expand(rows.numel()),
            rows,
            cols,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=8,
            targets=torch.zeros(rows.numel()),
        )
        projections.append(model.integrate_ray_batch(ray_batch).detach().reshape(detector_h, detector_w))
    return torch.stack(projections)


class ProjectionDomainAdaptationTests(unittest.TestCase):
    def test_diagnostics_find_interface_without_decoding_volume(self):
        model = BernsteinOctree(levels=[2, 4], max_degree=2, attenuation_shift=0.0)
        with torch.no_grad():
            values = torch.where(model.leaf_coords[:, 0] == 0, torch.tensor(0.1), torch.tensor(0.8))
            model.coefficient_logits.copy_(model._coefficients_to_logits(values))
        diagnostics = coefficient_diagnostics(model)
        self.assertGreater(float(diagnostics.interface_jump[:, 0].max()), 0.69)
        self.assertEqual(float(diagnostics.interface_jump[:, 1].max()), 0.0)
        actions = propose_forward_actions(
            model,
            diagnostics,
            variation_threshold=0.01,
            material_jump_threshold=0.2,
            p_ratio_threshold=0.5,
            max_candidates=2,
        )
        self.assertTrue(actions)
        self.assertTrue(all(action.action == "h_split" and action.axis == 0 for action in actions))

    def test_view_split_is_disjoint_and_reproducible(self):
        split = ProjectionSplit(
            angles=torch.arange(10, dtype=torch.float32),
            projections=torch.zeros(10, 2, 2),
            paths=[Path(str(index)) for index in range(10)],
        )
        train_a, validation_a = split_projection_views(split, validation_fraction=0.2, seed=7)
        train_b, validation_b = split_projection_views(split, validation_fraction=0.2, seed=7)
        self.assertEqual(train_a.paths, train_b.paths)
        self.assertEqual(validation_a.paths, validation_b.paths)
        self.assertTrue(set(train_a.paths).isdisjoint(validation_a.paths))
        self.assertEqual(len(validation_a.paths), 2)

    def test_held_out_gain_can_accept_true_high_order_structure(self):
        truth = BernsteinOctree(levels=[1, 2], max_degree=2, attenuation_shift=0.0)
        truth.elevate_degree(0, 0)
        with torch.no_grad():
            truth.coefficient_logits.copy_(truth._coefficients_to_logits(torch.tensor([0.1, 0.9])))
        angles = torch.tensor([0.2, 0.7, 1.1, 1.4])
        projections = _render(truth, angles)
        paths = [Path(str(index)) for index in range(4)]
        training = ProjectionSplit(angles[:3], projections[:3], paths[:3])
        validation = ProjectionSplit(angles[3:], projections[3:], paths[3:])

        model = BernsteinOctree(levels=[1, 2], max_degree=2, attenuation_shift=0.0)
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(torch.tensor([0.5])))
        candidate = CandidateAction("p_elevate", ((0, 0, 0, 0),), 0, 1.0)
        updated, evaluation = evaluate_candidate(
            model,
            candidate,
            training,
            validation,
            detector_h=4,
            detector_w=4,
            samples_per_ray=8,
            tune_steps=12,
            tune_batch_rays=16,
            tune_learning_rate=0.05,
            validation_ray_chunk=16,
            validation_views=None,
            validation_rays_per_view=None,
            rate_lambda=-1.0,
            merge_lambda=0.0,
            weight_epsilon=1e-4,
        )
        self.assertTrue(evaluation.accepted)
        self.assertGreater(evaluation.validation_gain, 0.0)
        self.assertEqual(updated.stats().coefficient_count, 2)


if __name__ == "__main__":
    unittest.main()

