import unittest
from pathlib import Path

import torch

from adaptive_ct.representation.octree import BernsteinOctree
from adaptive_ct.data.dataset import ProjectionSplit
from adaptive_ct.refinement.controller import RefinementController
from adaptive_ct.refinement.estimator import attributed_residual_energy, resolvable_mask
from adaptive_ct.codec.budget import codec_bytes, split_delta_bytes
from adaptive_ct.refinement.marking import mark_by_gain_share
from adaptive_ct.data.rays import sample_random_rays
from adaptive_ct.core.geometry import detector_grid, make_parallel_ray_points
from adaptive_ct.representation.latent import LatentP1Octree


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


def _batch(rays):
    from adaptive_ct.data.rays import ray_batch

    return ray_batch(rays, 0, rays.observation_count)
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
        marked, info = mark_by_gain_share(eta, ids, delta_bytes=10, theta=0.5, remaining_bytes=float("inf"))
        self.assertEqual(marked.tolist(), [10])
        self.assertAlmostEqual(info["eta_fraction"], 0.5)
        marked, _ = mark_by_gain_share(eta, ids, delta_bytes=10, theta=0.51, remaining_bytes=float("inf"))
        self.assertEqual(marked.tolist(), [10, 12])

    def test_mark_respects_byte_and_parent_caps(self):
        eta = torch.tensor([4.0, 3.0, 2.0, 1.0])
        ids = torch.arange(4)
        marked, info = mark_by_gain_share(eta, ids, delta_bytes=10, theta=1.0, remaining_bytes=25.0)
        self.assertEqual(marked.tolist(), [0, 1])
        self.assertEqual(info["k_bytes"], 2)
        marked, _ = mark_by_gain_share(eta, ids, delta_bytes=10, theta=1.0, remaining_bytes=float("inf"), max_parents=1)
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
    def test_residual_energy_sums_to_sampled_energy(self):
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
        eta, exposure, summary = attributed_residual_energy(model, rays, weight_kwargs=WEIGHT)
        self.assertAlmostEqual(float(eta.sum()), summary["estimator_weighted_rss"], places=3)
        self.assertGreater(float((exposure > 0).float().mean()), 0.0)


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
        before = float(((model.integrate_ray_batch(_batch(rays)) - _batch(rays).target) ** 2).sum())
        model.split_leaves_batch(torch.tensor([0, 2, 5]))
        after = float(((model.integrate_ray_batch(_batch(rays)) - _batch(rays).target) ** 2).sum())
        self.assertAlmostEqual(before, after, delta=1e-5 * max(before, 1.0))

    def test_latent_codec_is_lossless_at_full_rank(self):
        bern = _truth()
        bern.split_leaves_batch(torch.tensor([1, 4]))
        latent = LatentP1Octree.from_bernstein(bern, 8)
        points = torch.rand(200, 3) * 2.0 - 1.0
        self.assertTrue(torch.allclose(bern.forward_mu(points), latent.forward_mu(points), atol=1e-5))


class ControllerTests(unittest.TestCase):
    def _controller(self, split, **spec):
        base = {"gain_share": 0.5, "estimator_rays": 512, "min_footprint_px": 1.0, "codec": {"type": "p1", "quantization": "float16"}}
        base.update(spec)
        return RefinementController(spec=base, estimator_split=split, detector_h=DETECTOR, detector_w=DETECTOR, samples_per_ray=8,
                                    materialize_points=False, chunk_rays=128, seed=5, byte_budget=spec.get("byte_budget"))

    def test_round_splits_a_batch_without_changing_projections(self):
        truth = _truth()
        split = _split_from(truth)
        model = _model()
        rays = sample_random_rays(split, ray_count=256, detector_h=DETECTOR, detector_w=DETECTOR, samples_per_ray=8, seed=3, chunk_rays=128, materialize_points=False)
        before = float(model.integrate_ray_batch(_batch(rays)).sum())
        leaves = int(model.leaf_levels.shape[0])
        active, event = self._controller(split).refine(model, weight_kwargs=WEIGHT)
        self.assertGreater(active, 0)
        self.assertEqual(int(model.leaf_levels.shape[0]), leaves + 7 * active)
        self.assertAlmostEqual(float(model.integrate_ray_batch(_batch(rays)).sum()), before, delta=1e-4 * abs(before))
        self.assertFalse(event["stop"])

    def test_exact_model_has_nothing_to_split(self):
        truth = _truth()
        _, event = self._controller(_split_from(truth)).refine(truth, weight_kwargs=WEIGHT)
        self.assertTrue(event["stop"])
        self.assertIn("no_splittable_leaf", event["stop_reasons"])

    def test_byte_budget_stops_growth(self):
        truth = _truth()
        model = _model()
        _, event = self._controller(_split_from(truth), byte_budget=1).refine(model, weight_kwargs=WEIGHT)
        self.assertIn("byte_budget_exhausted", event["stop_reasons"])


if __name__ == "__main__":
    unittest.main()
