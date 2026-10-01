import unittest
from pathlib import Path

import torch

from adaptive_ct.representation.octree import BernsteinOctree
from adaptive_ct.data.dataset import ProjectionSplit
from adaptive_ct.data.rays import fixed_rss
from adaptive_ct.refinement.controller import DorflerController
from adaptive_ct.refinement.estimator import attributed_excess_residual, resolvable_mask
from adaptive_ct.codec.budget import codec_bytes, split_delta_bytes
from adaptive_ct.refinement.marking import dorfler_mark
from adaptive_ct.data.rays import sample_random_rays
from adaptive_ct.core.geometry import detector_grid, make_parallel_ray_points
from adaptive_ct.representation.latent import LatentP1Octree
from adaptive_ct.representation.snapshot import clone_sparse_state, restore_sparse_state


def _compact_raw_payload_bytes(model, quantization: str = "float16") -> int:
    """Reference byte count of the compact packed-hierarchy export (no metadata)."""
    bytes_per_float = 4 if quantization == "float32" else 2
    total = int(model.node_child_base.numel()) * 8
    total += int(model.leaf_levels.shape[0]) * 3 + 4 + len(model.level_shapes) * 6 + 3 + 4
    if hasattr(model, "leaf_latents"):
        total += int(model.leaf_latents.numel() + model.decoder.weight.numel() + model.decoder.bias.numel()) * bytes_per_float
    else:
        total += int(model.coefficients().numel()) * bytes_per_float
    return total


DETECTOR = 8
WEIGHT = {"mode": "uniform"}


def _model(levels=(2, 4, 8), degree=1):
    model = BernsteinOctree(
        levels=list(levels),
        max_degree=degree,
        attenuation_shift=0.0,
        cuda_integrator=False,
        integration_mode="exact",
        init_std=0.0,
    )
    if degree == 1:
        model.elevate_leaves_isotropic_batch(torch.arange(model.leaf_levels.shape[0]), (1, 1, 1))
    return model


def _render(model, angles):
    rows, cols = detector_grid(DETECTOR, DETECTOR, torch.device("cpu"))
    images = []
    for angle in angles:
        batch = make_parallel_ray_points(
            angle.expand(rows.numel()),
            rows,
            cols,
            detector_h=DETECTOR,
            detector_w=DETECTOR,
            samples_per_ray=8,
            targets=torch.zeros(rows.numel()),
            materialize_points=False,
        )
        images.append(model.integrate_ray_batch(batch).detach().reshape(DETECTOR, DETECTOR))
    return torch.stack(images)


def _split_from(truth, views=6):
    angles = torch.linspace(0.0, 3.0, views)
    return ProjectionSplit(angles=angles, projections=_render(truth, angles), paths=[Path(str(i)) for i in range(views)])


def _truth(levels=(2, 4, 8), degree=1):
    truth = _model(levels, degree)
    torch.manual_seed(3)
    with torch.no_grad():
        values = 0.05 + 0.5 * torch.rand(truth.coefficient_logits.shape[0])
        truth.coefficient_logits.copy_(truth._coefficients_to_logits(values))
    return truth


class DorflerMarkTests(unittest.TestCase):
    def test_mark_is_minimal_prefix_reaching_theta(self):
        eta = torch.tensor([5.0, 1.0, 3.0, 1.0])
        ids = torch.tensor([10, 11, 12, 13])
        marked, info = dorfler_mark(eta, ids, delta_bytes=10, theta=0.5, remaining_bytes=float("inf"))
        self.assertEqual(marked.tolist(), [10])
        self.assertAlmostEqual(info["eta_fraction"], 0.5)
        marked, _ = dorfler_mark(eta, ids, delta_bytes=10, theta=0.51, remaining_bytes=float("inf"))
        self.assertEqual(marked.tolist(), [10, 12])

    def test_mark_respects_byte_and_parent_caps(self):
        eta = torch.tensor([4.0, 3.0, 2.0, 1.0])
        ids = torch.arange(4)
        marked, info = dorfler_mark(eta, ids, delta_bytes=10, theta=1.0, remaining_bytes=25.0)
        self.assertEqual(marked.tolist(), [0, 1])
        self.assertEqual(info["k_bytes"], 2)
        marked, _ = dorfler_mark(eta, ids, delta_bytes=10, theta=1.0, remaining_bytes=float("inf"), max_parents=1)
        self.assertEqual(marked.tolist(), [0])


class ByteModelTests(unittest.TestCase):
    def test_p0_codec_matches_compact_payload_and_split_delta(self):
        model = _model(degree=0)
        codec = {"type": "p0", "quantization": "float16"}
        self.assertEqual(codec_bytes(model, codec), _compact_raw_payload_bytes(model))
        before = codec_bytes(model, codec)
        model.split_leaves_batch(torch.tensor([0, 3]))
        self.assertEqual(codec_bytes(model, codec) - before, 2 * split_delta_bytes(codec))
        self.assertEqual(split_delta_bytes(codec), 99)

    def test_latent_codec_matches_latent_model_payload(self):
        bern = _truth()
        bern.split_leaves_batch(torch.tensor([1, 5]))
        latent = LatentP1Octree.from_bernstein(bern, 4)
        codec = {"type": "latent_p1", "latent_dim": 4, "quantization": "float16"}
        self.assertEqual(codec_bytes(bern, codec), _compact_raw_payload_bytes(latent))


class EstimatorTests(unittest.TestCase):
    def test_excess_residual_sums_to_sampled_energy(self):
        truth = _truth()
        split = _split_from(truth)
        model = _model()
        rays = sample_random_rays(
            split,
            ray_count=512,
            detector_h=DETECTOR,
            detector_w=DETECTOR,
            samples_per_ray=8,
            seed=1,
            chunk_rays=128,
            materialize_points=False,
        )
        eta, exposure, summary = attributed_excess_residual(model, rays, noise_variance=0.0, weight_kwargs=WEIGHT)
        self.assertAlmostEqual(float(eta.sum()), summary["estimator_weighted_rss"], places=3)
        self.assertGreater(float((exposure > 0).float().mean()), 0.0)
        shifted, _, shifted_summary = attributed_excess_residual(model, rays, noise_variance=1e-3, weight_kwargs=WEIGHT)
        self.assertAlmostEqual(
            float(shifted.sum()),
            summary["estimator_weighted_rss"] - 1e-3 * rays.observation_count,
            places=2,
        )

    def test_resolvable_mask_follows_pixel_footprint(self):
        model = _model(levels=(2, 4, 8), degree=0)
        model.split_leaves_batch(torch.tensor([0]))
        fine = resolvable_mask(model, detector_h=16, detector_w=16, min_footprint_px=1.0)
        self.assertTrue(bool(fine[model.leaf_levels == 0].all()))
        self.assertTrue(bool(fine[model.leaf_levels == 1].all()))
        coarse = resolvable_mask(model, detector_h=4, detector_w=4, min_footprint_px=1.0)
        self.assertFalse(bool(coarse[model.leaf_levels == 1].any()))
        self.assertFalse(bool(resolvable_mask(model, detector_h=16, detector_w=16, min_footprint_px=1.0)[model.leaf_levels == 2].any()))


class SplitAndCodecTests(unittest.TestCase):
    def test_p1_split_preserves_projections(self):
        truth = _truth()
        split = _split_from(truth)
        model = _model()
        with torch.no_grad():
            model.coefficient_logits.copy_(model._coefficients_to_logits(0.05 + 0.5 * torch.rand(model.coefficient_logits.shape[0])))
        rays = sample_random_rays(
            split,
            ray_count=256,
            detector_h=DETECTOR,
            detector_w=DETECTOR,
            samples_per_ray=8,
            seed=2,
            chunk_rays=128,
            materialize_points=False,
        )
        before = fixed_rss(model, rays)["rss"]
        model.split_leaves_batch(torch.tensor([0, 2, 5]))
        after = fixed_rss(model, rays)["rss"]
        self.assertAlmostEqual(before, after, delta=1e-5 * max(before, 1.0))

    def test_latent_codec_is_lossless_at_full_rank(self):
        bern = _truth()
        bern.split_leaves_batch(torch.tensor([1, 4]))
        latent = LatentP1Octree.from_bernstein(bern, 8)
        points = torch.rand(200, 3) * 2.0 - 1.0
        self.assertTrue(torch.allclose(bern.forward_mu(points), latent.forward_mu(points), atol=1e-5))


class ControllerTests(unittest.TestCase):
    def _controller(self, split, dev, **spec):
        base = {
            "theta": 0.5,
            "estimator_rays": 512,
            "dev_ray_count": 256,
            "diagnostic_rays": 128,
            "min_footprint_px": 1.0,
            "max_halvings": 2,
            "codec": {"type": "p1", "quantization": "float16"},
        }
        base.update(spec)
        return DorflerController(
            spec=base,
            estimator_split=split,
            dev_split=dev,
            detector_h=DETECTOR,
            detector_w=DETECTOR,
            samples_per_ray=8,
            materialize_points=False,
            chunk_rays=128,
            seed=5,
            byte_budget=None,
        )

    def test_round_marks_splits_and_is_function_preserving(self):
        truth = _truth()
        split = _split_from(truth)
        dev = _split_from(truth, views=3)
        model = _model()
        controller = self._controller(split, dev)
        leaves_before = int(model.leaf_levels.shape[0])
        active, event = controller.on_trigger(model, iteration=1, noise_variance=0.0, weight_kwargs=WEIGHT)
        self.assertGreater(active, 0)
        self.assertEqual(int(model.leaf_levels.shape[0]), leaves_before + 7 * active)
        self.assertLess(event["split_function_error"], 1e-5)
        self.assertFalse(event["stop"])

    def test_no_excess_residual_stops_at_noise_floor(self):
        truth = _truth()
        split = _split_from(truth)
        controller = self._controller(split, None, rollback=False)
        _, event = controller.on_trigger(truth, iteration=1, noise_variance=1.0, weight_kwargs=WEIGHT)
        self.assertTrue(event["stop"])
        self.assertIn("noise_floor_reached", event["stop_reasons"])

    def test_noise_scale_zero_disables_noise_floor_stop(self):
        truth = _truth()
        split = _split_from(truth)
        controller = self._controller(split, None, rollback=False, noise_variance_scale=0.0)
        active, event = controller.on_trigger(_model(), iteration=1, noise_variance=1.0, weight_kwargs=WEIGHT)
        self.assertEqual(event["noise_variance_used"], 0.0)
        self.assertFalse(event["stop"])
        self.assertGreater(active, 0)

    def test_harmful_batch_is_rolled_back_with_halved_prefix_then_stops(self):
        truth = _truth()
        split = _split_from(truth)
        dev = _split_from(truth, views=3)
        model = _model()
        controller = self._controller(split, dev)
        pre_leaves = int(model.leaf_levels.shape[0])
        marked, _ = controller.on_trigger(model, iteration=1, noise_variance=0.0, weight_kwargs=WEIGHT)
        self.assertGreater(marked, 3)
        # No training happens between triggers, so the dev error cannot improve.
        active, event = controller.on_trigger(model, iteration=2, noise_variance=0.0, weight_kwargs=WEIGHT)
        self.assertEqual(event["verdict"], "rolled_back")
        self.assertEqual(active, marked // 2)
        self.assertEqual(int(model.leaf_levels.shape[0]), pre_leaves + 7 * (marked // 2))
        _, event = controller.on_trigger(model, iteration=3, noise_variance=0.0, weight_kwargs=WEIGHT)
        self.assertEqual(event["verdict"], "rolled_back")
        _, event = controller.on_trigger(model, iteration=4, noise_variance=0.0, weight_kwargs=WEIGHT)
        self.assertTrue(event["stop"])
        self.assertIn("rollback_exhausted", event["stop_reasons"])
        self.assertEqual(int(model.leaf_levels.shape[0]), pre_leaves)


if __name__ == "__main__":
    unittest.main()
