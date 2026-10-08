import unittest
from pathlib import Path

import torch

from adaptive_ct.core.projector import has_bernstein_native
from adaptive_ct.codec.budget import leaf_bytes, model_bytes, split_bytes
from adaptive_ct.core.geometry import detector_grid, parallel_rays
from adaptive_ct.data.dataset import ProjectionSplit
from adaptive_ct.refinement.decision import (candidate_step_geometry, child_minimum, choose, decide_and_apply, decision_views,
                                             evaluate_options, local_statistics, penalty_statistics, resolvable_mask)
from adaptive_ct.representation.octree import BernsteinOctree
from adaptive_ct.training.solve import solve

DETECTOR = 8
ALL = ("constant", "trilinear", "step")


def _model(levels=(2, 4, 8), types=ALL, trilinear=True):
    model = BernsteinOctree(levels=list(levels), leaf_types=types, cuda_integrator=False)
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
    kw = {"action_share": 0.5, "min_footprint_px": 1.0, "penalty_weight": 0.0, "penalty_delta": 0.02, "lookahead_leaves": 64, **kw}
    report, _ = decide_and_apply(model, split, decision_views(split, 16, 0), detector_h=DETECTOR, detector_w=DETECTOR, **kw)
    return report


def _random_values(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        model.values.copy_(0.05 + 0.5 * torch.rand(model.values.shape[0]))


class ByteModelTests(unittest.TestCase):
    def test_bytes_per_leaf_type_and_per_split(self):
        self.assertEqual(leaf_bytes().tolist(), [2.25, 16.25, 8.25])
        model = _model(trilinear=False)
        before = model_bytes(model)
        model.to_trilinear([0, 1])
        self.assertEqual(model_bytes(model) - before, 2 * 14)
        before = model_bytes(model)
        kind = model.leaf_type()
        ids = torch.tensor([int(torch.nonzero(kind == 1)[0]), int(torch.nonzero(kind == 0)[0])])
        expected = float(split_bytes(kind[ids]).sum())
        model.split_leaves_batch(ids)
        self.assertLessEqual(abs(model_bytes(model) - before - expected), 1)  # whole bytes of the 2-bit types
        self.assertEqual(expected, (1 + 7 * 16.25) + (1 + 7 * 2.25))

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

    def test_value_refit_alone_earns_no_structure_change(self):
        # Data from a trilinear tree; the model has that tree with the values of one leaf perturbed.  Refitting the
        # values of that leaf (same type) removes all its error, so no type change may be credited for it.
        # (Neighbouring leaves may still gain: with the perturbed leaf fixed, finer cells compensate for its error.
        # A split adds corrections to the current function and is measured against it, see evaluate_options.)
        truth = _model(levels=(2, 4, 8))
        _random_values(truth, 21)
        split = _split_from(truth)
        model = _model(levels=(2, 4, 8))
        with torch.no_grad():
            model.values.copy_(truth.values)
            model.values[8 * 3 : 8 * 3 + 8] += 0.25 * torch.rand(8)
        candidate = candidate_step_geometry(model, detector_w=DETECTOR)
        acc = local_statistics(model, split, decision_views(split, 16, 0), candidate)
        self.assertGreater(float(acc[:, 3, 96].sum()), 1e-3)  # the leaf's rays do show an error
        gains = evaluate_options(acc, -child_minimum(model), model.leaf_type())
        self.assertLessEqual(float(gains[3, :3].max()), 0.0, gains[3].tolist())

    def test_candidate_equal_to_the_current_type_has_zero_gain(self):
        model = _model(levels=(2, 4, 8), trilinear=False)
        _random_values(model, 5)
        model.set_types([0, 3], [2, 2], torch.tensor([0.1, 0.6, 0.2, 0.7]),
                        torch.tensor([[0.6, 0.8, 0.0, 0.1, 0.05], [0.0, 0.6, 0.8, -0.1, 0.0]]))
        truth = _model(levels=(2, 4, 8))
        _random_values(truth, 8)
        split = _split_from(truth)
        candidate = model.geometry(torch.arange(model.leaf_count))  # candidate geometry = the current one
        acc = local_statistics(model, split, decision_views(split, 16, 0), candidate)
        gains = evaluate_options(acc, -child_minimum(model), model.leaf_type())
        steps = torch.nonzero(model.leaf_type() == 2).flatten()
        self.assertTrue(torch.allclose(gains[steps, 2], torch.zeros(steps.numel(), dtype=gains.dtype), atol=1e-9))

    def test_planar_step_inside_a_leaf_becomes_a_step_leaf(self):
        truth = _model(levels=(1, 2, 4), trilinear=False)
        truth.set_types([0], [2], torch.tensor([0.1, 0.5]), torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]))  # hard step at x = 0
        split = _split_from(truth)
        model = _model(levels=(1, 2, 4))
        solve(model, split, subsets=4, tol=1e-3, max_epochs=200, min_epochs=3)  # trilinear fit: a ramp along x
        report = _decide(model, split, byte_budget=10**9, action_share=1.0)
        self.assertEqual(report["actions"]["step"], 1)
        self.assertEqual(model.leaf_type().tolist(), [2])
        self.assertGreater(abs(float(model.step_geometry[0, 0])), 0.9)  # direction along x
        err = (_render(model, split.angles) - split.projections).square().mean()
        self.assertLess(float(err), 0.05 * float(split.projections.square().mean()))

    def test_actions_respect_the_byte_budget(self):
        truth = _model(levels=(2, 4, 8))
        truth.split_leaves_batch(torch.arange(8))
        _random_values(truth, 5)
        split = _split_from(truth)
        model = _model(levels=(2, 4, 8))
        solve(model, split, subsets=4, tol=1e-3, max_epochs=100, min_epochs=3)
        budget = model_bytes(model) + 400
        report = _decide(model, split, byte_budget=budget, action_share=1.0)
        self.assertGreater(sum(report["actions"].values()), 0)
        self.assertLessEqual(report["bytes_after"], budget)
        self.assertEqual(report["bytes_after"], model_bytes(model))

    def test_no_split_gain_that_needs_negative_attenuation(self):
        truth = _model(levels=(2, 4, 8), trilinear=False)
        with torch.no_grad():
            truth.values.zero_()
        split = _split_from(truth)  # empty object: all projections ~0
        model = _model(levels=(2, 4, 8), trilinear=False)
        with torch.no_grad():
            values = torch.zeros_like(model.values)
            values[0] = 0.3  # one wrong leaf; every other leaf already sits at the bound 0
            model.values.copy_(values)
        report = _decide(model, split, byte_budget=10**9, action_share=1.0)
        # only the wrong leaf can improve; its zero-valued neighbours would need negative values
        self.assertLessEqual(sum(report["actions"].values()), 1)

    def test_frozen_leaves_cannot_change_type(self):
        truth = _model(levels=(1, 2, 4), trilinear=False)
        truth.set_types([0], [2], torch.tensor([0.1, 0.5]), torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]))
        split = _split_from(truth)
        model = _model(levels=(1, 2, 4))
        solve(model, split, subsets=4, tol=1e-3, max_epochs=200, min_epochs=3)
        report = _decide(model, split, byte_budget=10**9, action_share=1.0, frozen=torch.tensor([True]))
        self.assertEqual(report["actions"]["constant"] + report["actions"]["trilinear"] + report["actions"]["step"], 0)


@unittest.skipUnless(torch.cuda.is_available() and has_bernstein_native(), "Bernstein CUDA extension unavailable")
class DecisionCudaTests(unittest.TestCase):
    def test_two_level_split_pays_only_for_children_worth_their_bytes(self):
        gains = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
        extra = torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0]], dtype=torch.float64)
        allowed = torch.tensor([[False, False, False, True, True]])
        child = (torch.tensor([[30.0, 30.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
                 torch.full((1, 8), 10.0, dtype=torch.float64), torch.ones((1, 8), dtype=torch.bool))
        ids, option, lam, second = choose(gains, extra, allowed, 1e9, child)
        self.assertEqual(option.tolist(), [4])
        self.assertEqual(second[0].tolist(), [True, True, True] + [False] * 5)  # price 0: every positive child
        ids, option, lam, second = choose(gains, extra, allowed, 30.0, child)  # room for the split and 2 children
        self.assertEqual(option.tolist(), [4])
        self.assertEqual(second[0].tolist(), [True, True] + [False] * 6)  # the 0.5 child does not pay for 10 bytes
        self.assertGreater(lam, 0.05)

    def test_slab_thinner_than_the_children_is_reached_in_one_decision(self):
        # a slab one finest cell wide inside root leaves: a one-level split only half-resolves it, two levels do
        truth = _model(trilinear=False)
        truth.split_leaves_batch(torch.arange(truth.leaf_count))
        truth.split_leaves_batch(torch.arange(truth.leaf_count))
        with torch.no_grad():
            truth.values.fill_(0.2)
            truth.values[truth.leaf_coords[:, 0] == 1] = 0.6  # constant leaves: one value each, in leaf order
        split = _split_from(truth)
        model = _model()
        solve(model, split, subsets=4, tol=1e-3, max_epochs=200, min_epochs=3)
        report = _decide(model, split, byte_budget=10 ** 6)
        self.assertGreater(report["actions"]["split2"], 0)
        self.assertGreater(report["second_level_splits"], 0)
        self.assertEqual(int(model.leaf_levels.max()), 2)

    def test_cuda_statistics_match_reference(self):
        from tests.representation.test_octree import make_step
        model = BernsteinOctree(levels=[2, 4, 8], leaf_types=ALL)
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
        self.assertEqual(cuda.dtype, torch.float64)
        for column in range(cuda.shape[-1]):  # every statistic: the kernel (float32 per view) against the reference
            scale = float(reference[..., column].abs().max()) + 1e-12
            self.assertLess(float((cuda[..., column] - reference[..., column]).abs().max()) / scale, 2e-4, column)
        # statistics of chosen leaves only: the same rows as the full statistics (kernel and reference), penalty too
        ids = torch.arange(0, model.leaf_count, 3, device="cuda").flip(0)
        part_ref = local_statistics(model, split, views, candidate, ids=ids)
        model.cuda_integrator = True
        part = local_statistics(model, split, views, candidate, ids=ids)
        for full, sub in ((reference, part_ref), (cuda, part)):
            self.assertLess(float((sub - full[:, ids]).abs().max()), 1e-9 + 1e-6 * float(full.abs().max()))
        pen = penalty_statistics(model, 0.02, candidate)
        self.assertLess(float((penalty_statistics(model, 0.02, candidate, ids) - pen[ids]).abs().max()), 1e-9 * (1 + float(pen.abs().max())))

if __name__ == "__main__":
    unittest.main()
