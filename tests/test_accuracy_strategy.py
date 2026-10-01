import unittest
from pathlib import Path

import torch

from adaptive_ct.bernstein import BernsteinOctree
from adaptive_ct.data import ProjectionSplit
from adaptive_ct.geometry import detector_grid, make_parallel_ray_points
from adaptive_ct.train import (
    _apply_gradient_rate_distortion_p_refinement,
    _apply_h_gauss_newton_rate_distortion,
    _apply_h_jump_rate_distortion,
    _apply_hp_gauss_newton_rate_distortion,
    _bernstein_haar_predicted_gain_leaf_score,
    _projection_gradient_effective_weight,
    _projection_weight_kwargs,
    _select_coefficient_h_candidates,
    _select_p_residual_candidates,
    _select_residual_h_candidates,
    _virtual_haar_ray_features,
    _virtual_trilinear_ray_features,
)
from adaptive_ct.projection_domain import projection_weights


def _render(model, angles, detector_h=8, detector_w=8):
    rows, cols = detector_grid(detector_h, detector_w, torch.device("cpu"))
    values = []
    for angle in angles:
        batch = make_parallel_ray_points(
            angle.expand(rows.numel()),
            rows,
            cols,
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=16,
            targets=torch.zeros(rows.numel()),
        )
        values.append(model.integrate_ray_batch(batch).detach().reshape(detector_h, detector_w))
    return torch.stack(values)


class AccuracyStrategyTests(unittest.TestCase):
    def test_global_jump_selector_uses_only_rate_budget(self):
        model = BernsteinOctree(levels=[2, 4, 8], max_leaf_count=64, attenuation_shift=0.0)
        with torch.no_grad():
            values = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
            model.coefficient_logits.copy_(model._coefficients_to_logits(values))
        active, summary = _apply_h_jump_rate_distortion(
            model,
            max_added_bytes=99,
        )
        self.assertEqual(active, 1)
        self.assertEqual(summary["predicted_added_bytes"], 99)
        self.assertEqual(sum(summary["selected_by_parent_level"]), 1)
        self.assertEqual(model.leaf_levels.numel(), 15)

    def test_h_gauss_newton_selector_obeys_byte_budget(self):
        angles = torch.linspace(0.0, torch.pi, 8)[:-1]
        truth = BernsteinOctree(levels=[1, 2], max_degree=1, attenuation_shift=0.0)
        truth.split_leaf(0)
        with torch.no_grad():
            values = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
            truth.coefficient_logits.copy_(truth._coefficients_to_logits(values))
        split = ProjectionSplit(angles=angles, projections=_render(truth, angles), paths=[])
        model = BernsteinOctree(levels=[1, 2], max_degree=1, attenuation_shift=0.0)
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(torch.tensor([0.5])))

        active, summary = _apply_h_gauss_newton_rate_distortion(
            model,
            split,
            level=0,
            detector_h=8,
            detector_w=8,
            samples_per_ray=16,
            ray_chunk=128,
            max_views=None,
            max_rays_per_view=None,
            weight_kwargs={"mode": "uniform"},
            fisher_damping=1.0e-8,
            max_added_bytes=99,
        )
        self.assertEqual(active, 1)
        self.assertEqual(summary["predicted_added_bytes"], 99)
        self.assertEqual(model.leaf_levels.numel(), 8)

    def test_unified_hp_rate_prefers_p_for_smooth_trilinear_field(self):
        angles = torch.linspace(0.0, torch.pi, 8)[:-1]
        truth = BernsteinOctree(levels=[1, 2], max_degree=1, attenuation_shift=0.0)
        truth.elevate_leaves_isotropic_batch([0], (1, 1, 1))
        with torch.no_grad():
            local_x = torch.tensor([0.2, 0.8]).view(2, 1, 1).expand(2, 2, 2)
            truth.coefficient_logits.copy_(truth._coefficients_to_logits(local_x.reshape(-1)))
        split = ProjectionSplit(angles=angles, projections=_render(truth, angles), paths=[])
        model = BernsteinOctree(levels=[1, 2], max_degree=1, attenuation_shift=0.0)
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(torch.tensor([0.5])))

        active, summary = _apply_hp_gauss_newton_rate_distortion(
            model,
            split,
            levels=[0],
            detector_h=8,
            detector_w=8,
            samples_per_ray=16,
            ray_chunk=128,
            max_views=None,
            max_rays_per_view=None,
            weight_kwargs={"mode": "uniform"},
            fisher_damping=1.0e-8,
            max_added_bytes=14,
        )
        self.assertEqual(active, 1)
        self.assertEqual(summary["p_selected"], 1)
        self.assertEqual(summary["h_selected"], 0)
        self.assertEqual(tuple(model.leaf_degrees[0].tolist()), (1, 1, 1))

    def test_trilinear_virtual_features_match_finite_difference(self):
        model = BernsteinOctree(levels=[1, 2], max_degree=1, attenuation_shift=0.0)
        ray_batch = make_parallel_ray_points(
            torch.tensor([0.17, 0.73]),
            torch.tensor([1, 2]),
            torch.tensor([2, 1]),
            detector_h=4,
            detector_w=4,
            samples_per_ray=8,
            targets=torch.zeros(2),
        )
        ray_ids = torch.tensor([0, 1])
        leaf_ids = torch.tensor([0, 0])
        features = _virtual_trilinear_ray_features(model, ray_batch, ray_ids, leaf_ids)
        basis = torch.tensor(
            [
                [1, 1, 1, 1, -1, -1, -1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, 1, -1, 1, -1, 1, -1],
            ],
            dtype=torch.float32,
        ).T / (8.0**0.5)
        model.elevate_leaves_isotropic_batch([0], (1, 1, 1))
        epsilon = 1.0e-3
        for detail in range(3):
            predictions = []
            for sign in (-1.0, 1.0):
                physical = torch.full((8,), 0.5) + sign * epsilon * basis[:, detail]
                with torch.no_grad():
                    model.coefficient_logits.copy_(model._coefficients_to_logits(physical))
                predictions.append(model.integrate_ray_batch(ray_batch).detach())
            derivative = (predictions[1] - predictions[0]) / (2.0 * epsilon)
            self.assertTrue(torch.allclose(derivative, features[:, detail], atol=2e-4, rtol=2e-4))

    def test_virtual_haar_features_match_axis_aligned_child_lengths(self):
        model = BernsteinOctree(levels=[1, 2], attenuation_shift=0.0)
        batch = make_parallel_ray_points(
            torch.tensor([0.0]),
            torch.tensor([0]),
            torch.tensor([0]),
            detector_h=2,
            detector_w=2,
            samples_per_ray=4,
            targets=torch.zeros(1),
            materialize_points=False,
        )
        features = _virtual_haar_ray_features(model, batch, torch.tensor([0]), torch.tensor([0]))
        expected = torch.tensor([[0.0, 2**-0.5, 2**-0.5, 0.0, 0.0, 2**-0.5, 0.0]])
        self.assertTrue(torch.allclose(features, expected, atol=1.0e-6, rtol=1.0e-6))

    def test_haar_predicted_gain_detects_unmodelled_child_detail(self):
        truth = BernsteinOctree(levels=[1, 2], attenuation_shift=0.0)
        truth.split_leaf(0)
        with torch.no_grad():
            truth.coefficient_logits.copy_(
                truth._coefficients_to_logits(
                    torch.tensor([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
                )
            )
        angles = torch.tensor([0.1, 0.7, 1.3, 2.0])
        projections = _render(truth, angles)
        split = ProjectionSplit(angles, projections, [Path(str(index)) for index in range(4)])
        model = BernsteinOctree(levels=[1, 2], attenuation_shift=0.0)
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(torch.tensor([0.5])))
        leaf_ids, scores, summary = _bernstein_haar_predicted_gain_leaf_score(
            model,
            split,
            level=0,
            detector_h=8,
            detector_w=8,
            samples_per_ray=16,
            ray_chunk=16,
            max_views=4,
            max_rays_per_view=64,
            weight_kwargs={
                "mode": "uniform",
                "epsilon": 1.0e-4,
                "target_power": 1.0,
                "min_weight": None,
                "max_weight": None,
            },
        )
        self.assertEqual(leaf_ids.tolist(), [0])
        self.assertGreater(float(scores[0]), 0.0)
        self.assertEqual(summary["haar_supported_leaves"], 1)

    def test_p_candidate_quantile_band_uses_positive_mid_residuals(self):
        leaf_ids = torch.arange(10)
        scores = torch.tensor([0.0, float("nan"), 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        selected, summary = _select_p_residual_candidates(
            leaf_ids,
            scores,
            candidate_fraction=0.08,
            residual_quantile_range=[0.5, 0.85],
        )
        low, high = summary["p_candidate_score_range"]
        self.assertEqual(summary["p_candidate_selection"], "residual_quantile_band")
        self.assertEqual(summary["p_positive_score_count"], 8)
        self.assertTrue(torch.all(scores[selected] >= low))
        self.assertTrue(torch.all(scores[selected] <= high))
        self.assertTrue(torch.all(scores[selected] > 0))

    def test_projection_gradient_weight_ramps_and_stops(self):
        config = {"weight": 0.4, "start_iteration": 11, "ramp_steps": 4, "stop_iteration": 20}
        self.assertEqual(_projection_gradient_effective_weight(config, 10), 0.0)
        self.assertAlmostEqual(_projection_gradient_effective_weight(config, 11), 0.1)
        self.assertAlmostEqual(_projection_gradient_effective_weight(config, 14), 0.4)
        self.assertEqual(_projection_gradient_effective_weight(config, 21), 0.0)

    def test_projection_weight_schedule_is_iteration_aware(self):
        config = {
            "projection_weighting_schedule": {
                1: {"mode": "uniform"},
                6001: {"mode": "target_power", "target_power": 0.25, "min_weight": 0.3, "max_weight": 2.0},
            }
        }
        self.assertEqual(_projection_weight_kwargs(config, {}, iteration=6000)["mode"], "uniform")
        late = _projection_weight_kwargs(config, {}, iteration=6001)
        self.assertEqual(late["mode"], "target_power")
        self.assertEqual(late["target_power"], 0.25)

    def test_inverse_target_blend_is_normalized_and_bounded(self):
        target = torch.linspace(0.0, 1.0, 100)
        weights = projection_weights(
            target,
            mode="inverse_target_blend",
            target_power=0.25,
            blend_alpha=0.15,
            min_weight=0.3,
            max_weight=2.0,
        )
        self.assertTrue(torch.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)

    def test_dynamic_range_balanced_is_the_public_name_for_legacy_blend(self):
        target = torch.tensor([0.0, 0.01, 0.1, 0.5], dtype=torch.float32)
        kwargs = {
            "target_power": 0.25,
            "min_weight": 0.3,
            "max_weight": 2.0,
            "blend_alpha": 0.5,
        }
        public = projection_weights(target, mode="dynamic_range_balanced", **kwargs)
        legacy = projection_weights(target, mode="inverse_target_blend", **kwargs)
        torch.testing.assert_close(public, legacy)

    def test_h_selector_unions_threshold_and_respects_leaf_budget(self):
        model = BernsteinOctree(levels=[2, 4], max_leaf_count=15)
        leaf_ids = torch.arange(8)
        scores = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
        selected, summary = _select_residual_h_candidates(
            model,
            leaf_ids,
            scores,
            active_fraction=0.125,
            threshold_statistic="q95",
            threshold_multiplier=0.5,
            max_fraction=0.5,
            reserve_leaf_count=0,
        )
        self.assertEqual(selected.numel(), 1)
        self.assertEqual(summary["selection_cap"], 1)
        self.assertEqual(int(selected[0]), 0)

    def test_h_gain_selector_preserves_material_threshold_under_budget(self):
        model = BernsteinOctree(levels=[2, 4], max_leaf_count=29, attenuation_shift=0.0)
        with torch.no_grad():
            model.coefficient_logits.copy_(
                model._coefficients_to_logits(
                    torch.tensor([0.01, 0.20, 0.30, 0.01, 0.01, 0.01, 0.01, 0.01])
                )
            )
        selected, summary = _select_residual_h_candidates(
            model,
            torch.arange(8),
            torch.tensor([10.0, 1.0, 2.0, 9.0, 8.0, 7.0, 6.0, 5.0]),
            active_fraction=0.125,
            threshold_statistic=None,
            threshold_multiplier=0.5,
            max_fraction=0.5,
            min_mu_threshold=0.1,
        )
        self.assertEqual(summary["selection_cap"], 3)
        self.assertEqual(summary["material_threshold_count"], 2)
        self.assertEqual(set(selected.tolist()), {0, 1, 2})

    def test_coefficient_h_selector_prioritizes_material_threshold_then_top_score(self):
        model = BernsteinOctree(levels=[2, 4], max_leaf_count=29, attenuation_shift=0.0)
        with torch.no_grad():
            model.coefficient_logits.copy_(
                model._coefficients_to_logits(
                    torch.tensor([0.01, 0.20, 0.30, 0.01, 0.01, 0.01, 0.01, 0.01])
                )
            )
        score = torch.tensor(
            [
                [[10.0, 9.0], [8.0, 7.0]],
                [[6.0, 5.0], [4.0, 3.0]],
            ]
        )
        selected, summary = _select_coefficient_h_candidates(
            model,
            level=1,
            score=score,
            active_fraction=0.125,
            min_mu_threshold=0.1,
            max_fraction=0.5,
        )
        self.assertEqual(summary["selection_cap"], 3)
        self.assertEqual(summary["material_threshold_count"], 2)
        self.assertEqual(set(selected.tolist()), {0, 1, 2})

    def test_gradient_rd_uses_new_function_subspace_and_applies_selected_elevation(self):
        truth = BernsteinOctree(levels=[1, 2], max_degree=1, attenuation_shift=0.0)
        truth.elevate_leaves_isotropic_batch(torch.tensor([0]), (1, 1, 1))
        with torch.no_grad():
            truth.coefficient_logits.copy_(
                truth._coefficients_to_logits(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]))
            )
        angles = torch.tensor([0.1, 0.7, 1.3, 2.0])
        projections = _render(truth, angles)
        split = ProjectionSplit(angles, projections, [Path(str(index)) for index in range(4)])
        model = BernsteinOctree(levels=[1, 2], max_degree=1, attenuation_shift=0.0)
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(torch.tensor([0.5])))
        active, summary = _apply_gradient_rate_distortion_p_refinement(
            model,
            torch.tensor([0]),
            (1, 1, 1),
            training_split=split,
            detector_h=8,
            detector_w=8,
            samples_per_ray=16,
            gradient_batches=2,
            batch_rays=64,
            active_fraction=1.0,
            eligible_leaf_count=1,
            max_added_coefficients=7,
            learning_rate=0.01,
            weight_kwargs={"mode": "uniform", "epsilon": 1.0e-4, "target_power": 1.0, "min_weight": None, "max_weight": None},
        )
        self.assertEqual(active, 1)
        self.assertTrue(summary["accepted"])
        self.assertGreater(summary["score_max"], 0.0)
        self.assertEqual(tuple(model.leaf_degrees[0].tolist()), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
