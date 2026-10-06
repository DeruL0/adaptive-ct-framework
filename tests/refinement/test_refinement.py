import unittest
from pathlib import Path

import torch

from adaptive_ct.core.projector import has_bernstein_native
from adaptive_ct.codec.budget import leaf_bytes, model_bytes, split_bytes
from adaptive_ct.core.geometry import detector_grid, parallel_rays
from adaptive_ct.data.dataset import ProjectionSplit
from adaptive_ct.refinement.decision import candidate_step_geometry, decide_and_apply, decision_views, local_statistics, resolvable_mask
from adaptive_ct.representation.octree import BernsteinOctree, type_ids
from adaptive_ct.training.solve import solve

DETECTOR = 8
ALL = ("constant", "trilinear", "step")


def _model(levels=(2, 4, 8), types=ALL, trilinear=True):
    model = BernsteinOctree(levels=list(levels), leaf_types=types, attenuation_shift=0.0, cuda_integrator=False)
    if trilinear:
        model.to_trilinear(torch.arange(model.leaf_levels.shape[0]))
    return model


def _render(model, angles):
    rows, cols = detector_grid(DETECTOR, DETECTOR, torch.device("cpu"))
    return torch.stack([model.integrate_ray_batch(parallel_rays(a.expand(rows.numel()), rows, cols, detector_h=DETECTOR, detector_w=DETECTOR,
                                                                targets=torch.zeros(rows.numel()))).detach().reshape(DETECTOR, DETECTOR)
                        for a in angles])


def _split_from(truth, views=16):
    angles = torch.linspace(0.0, 3.0, views)
    return ProjectionSplit(angles=angles, projections=_render(truth, angles), paths=[Path(str(i)) for i in range(views)])


def _decide(model, split, **kw):
    report, _ = decide_and_apply(model, split, decision_views(split, 16, 0), detector_h=DETECTOR, detector_w=DETECTOR, **kw)
    return report


def _random_values(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        model.coefficient_logits.copy_(model._coefficients_to_logits(0.05 + 0.5 * torch.rand(model.coefficient_logits.shape[0])))


class ByteModelTests(unittest.TestCase):
    def test_bytes_per_leaf_type_and_per_split(self):
        self.assertEqual(leaf_bytes("float16").tolist(), [5, 19, 15])
        model = _model(trilinear=False)
        before = model_bytes(model)
        model.to_trilinear([0, 1])
        self.assertEqual(model_bytes(model) - before, 2 * 14)
        before = model_bytes(model)
        kind = type_ids(model.leaf_degrees)
        ids = torch.tensor([int(torch.nonzero(kind == 1)[0]), int(torch.nonzero(kind == 0)[0])])
        expected = int(split_bytes(kind[ids]).sum())
        model.split_leaves_batch(ids)
        self.assertEqual(model_bytes(model) - before, expected)
        self.assertEqual(expected, (64 + 7 * 19) + (64 + 7 * 5))

    def test_resolvable_mask_follows_pixel_footprint(self):
        model = _model(trilinear=False)
        model.split_leaves_batch(torch.tensor([0]))
        fine = resolvable_mask(model, detector_h=16, detector_w=16, min_footprint_px=1.0)
        self.assertTrue(bool(fine[model.leaf_levels == 0].all()))
        self.assertTrue(bool(fine[model.leaf_levels == 1].all()))
        self.assertFalse(bool(resolvable_mask(model, detector_h=4, detector_w=4, min_footprint_px=1.0)[model.leaf_levels == 1].any()))
        self.assertFalse(bool(fine[model.leaf_levels == 2].any()))  # deepest level


class DecisionTests(unittest.TestCase):
    def test_exact_model_has_no_beneficial_action(self):
        truth = _model()
        _random_values(truth, 3)
        report = _decide(truth, _split_from(truth), byte_budget=10**9)
        self.assertEqual(sum(report["actions"].values()), 0)

    def test_planar_step_inside_a_leaf_becomes_a_step_leaf(self):
        truth = _model(levels=(1, 2, 4), trilinear=False)
        truth.set_types([0], [2], torch.tensor([0.1, 0.5]), torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]))  # hard step at x = 0
        split = _split_from(truth)
        model = _model(levels=(1, 2, 4))
        solve(model, split, passes=60, tol=1e-6)  # trilinear fit: a ramp along x
        report = _decide(model, split, byte_budget=10**9, action_share=1.0)
        self.assertEqual(report["actions"]["step"], 1)
        self.assertEqual(type_ids(model.leaf_degrees).tolist(), [2])
        self.assertGreater(abs(float(model.step_geometry[0, 0])), 0.9)  # direction along x
        err = (_render(model, split.angles) - split.projections).square().mean()
        self.assertLess(float(err), 0.05 * float(split.projections.square().mean()))

    def test_actions_respect_the_byte_budget(self):
        truth = _model(levels=(2, 4, 8))
        truth.split_leaves_batch(torch.arange(8))
        _random_values(truth, 5)
        split = _split_from(truth)
        model = _model(levels=(2, 4, 8))
        solve(model, split, passes=30, tol=1e-4)
        budget = model_bytes(model) + 400
        report = _decide(model, split, byte_budget=budget, action_share=1.0)
        self.assertGreater(sum(report["actions"].values()), 0)
        self.assertLessEqual(report["bytes_after"], budget)
        self.assertEqual(report["bytes_after"], model_bytes(model))

    def test_no_split_gain_that_needs_negative_attenuation(self):
        truth = _model(levels=(2, 4, 8), trilinear=False)
        with torch.no_grad():
            truth.coefficient_logits.copy_(truth._coefficients_to_logits(torch.full_like(truth.coefficient_logits, 1e-12)))
        split = _split_from(truth)  # empty object: all projections ~0
        model = _model(levels=(2, 4, 8), trilinear=False)
        with torch.no_grad():
            values = torch.full_like(model.coefficient_logits, 1e-12)
            values[0] = 0.3  # one wrong leaf; every other leaf already sits at the bound 0
            model.coefficient_logits.copy_(model._coefficients_to_logits(values))
        report = _decide(model, split, byte_budget=10**9, action_share=1.0)
        # only the wrong leaf can improve; its zero-valued neighbours would need negative values
        self.assertLessEqual(sum(report["actions"].values()), 1)

    def test_frozen_leaves_cannot_change_type(self):
        truth = _model(levels=(1, 2, 4), trilinear=False)
        truth.set_types([0], [2], torch.tensor([0.1, 0.5]), torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]))
        split = _split_from(truth)
        model = _model(levels=(1, 2, 4))
        solve(model, split, passes=60, tol=1e-6)
        report = _decide(model, split, byte_budget=10**9, action_share=1.0, frozen=torch.tensor([True]))
        self.assertEqual(report["actions"]["constant"] + report["actions"]["trilinear"] + report["actions"]["step"], 0)


@unittest.skipUnless(torch.cuda.is_available() and has_bernstein_native(), "Bernstein CUDA extension unavailable")
class DecisionCudaTests(unittest.TestCase):
    def test_cuda_statistics_match_reference(self):
        from tests.representation.test_octree import make_step
        model = BernsteinOctree(levels=[2, 4, 8], leaf_types=ALL, attenuation_shift=0.0)
        model.to_trilinear([0, 3, 5, 6])
        model.split_leaves_batch([model.find_leaf((0, 0, 0, 0)), model.find_leaf((0, 1, 1, 0))])
        make_step(model, [model.find_leaf((1, 1, 1, 1)), model.find_leaf((0, 1, 0, 1))], seed=2)
        make_step(model, [model.find_leaf((1, 2, 2, 0))], seed=3, width=0.0)
        _random_values(model, 9)
        model = model.cuda()
        truth = _model(levels=(2, 4, 8))
        _random_values(truth, 1)
        split = _split_from(truth)
        split = ProjectionSplit(angles=split.angles.cuda(), projections=split.projections.cuda(), paths=split.paths)
        candidate = candidate_step_geometry(model, detector_w=DETECTOR)
        views = decision_views(split, 16, 0)
        cuda = local_statistics(model, split, views, candidate)
        model.cuda_integrator = False
        reference = local_statistics(model, split, views, candidate)
        for key in reference:
            scale = float(reference[key].abs().max()) + 1e-12
            self.assertLess(float((cuda[key] - reference[key]).abs().max()) / scale, 2e-4, key)


if __name__ == "__main__":
    unittest.main()
