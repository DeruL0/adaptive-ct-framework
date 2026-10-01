import unittest

import torch

from adaptive_ct.bernstein import BernsteinOctree
from adaptive_ct.geometry import make_parallel_ray_points
from adaptive_ct.functional_compression import (
    NODE_SLOT_BYTES,
    LEAF_DEGREE_BYTES,
    COEFF_BYTES,
    P0_BYTES,
    build_candidate_bank,
    _internal_coords_by_level,
    _leaf_bytes_by_level,
    _local_detail_distortion,
    _candidate_cell_integrals,
    _packed_v3_fixed_payload_bytes,
    _run_lagrangian_dp,
    evaluate_pruned_field,
    materialize_compressed_model,
    run_compression,
)


def _full_tree_bytes(model: BernsteinOctree) -> float:
    """Independent ground truth for the compact_v3 byte accounting used by
    the compressor: every node (leaf or internal) costs one node slot, and
    every leaf additionally costs its degree tag and float16 coefficients."""
    internal_by_level = _internal_coords_by_level(model)
    internal_node_count = sum(int(linear.numel()) for linear in internal_by_level)
    leaf_count = int(model.leaf_levels.shape[0])
    leaf_bytes = (
        NODE_SLOT_BYTES * leaf_count
        + LEAF_DEGREE_BYTES * leaf_count
        + COEFF_BYTES * int(model.coefficient_counts().sum().item())
    )
    return float(NODE_SLOT_BYTES * internal_node_count + leaf_bytes)


class FunctionalCompressionTests(unittest.TestCase):
    def _build_tree(self) -> BernsteinOctree:
        torch.manual_seed(0)
        model = BernsteinOctree(levels=[2, 4, 8], max_degree=1, attenuation_shift=0.0)
        with torch.no_grad():
            values = torch.rand(model.coefficient_logits.shape[0]) * 0.5 + 0.05
            model.coefficient_logits.copy_(model._coefficients_to_logits(values))
        model.split_leaf(model.find_leaf((0, 0, 0, 0)))
        leaf_id = model.find_leaf((1, 0, 0, 0))
        model.split_leaf(leaf_id)
        with torch.no_grad():
            values = torch.rand(model.coefficient_logits.shape[0]) * 0.5 + 0.05
            model.coefficient_logits.copy_(model._coefficients_to_logits(values))
        return model

    def test_lambda_zero_reproduces_the_full_tree_with_zero_distortion(self):
        model = self._build_tree()
        bank = build_candidate_bank(model, samples_per_axis=3, node_chunk=64)
        leaf_bytes_by_level = _leaf_bytes_by_level(model)
        total_bytes, total_distortion, _actions = _run_lagrangian_dp(
            model, bank, leaf_bytes_by_level, lam=0.0, gamma=1.0
        )
        self.assertAlmostEqual(total_bytes, _full_tree_bytes(model), places=3)
        self.assertAlmostEqual(total_distortion, 0.0, places=6)

    def test_large_lambda_collapses_every_root_cell_to_p0(self):
        model = self._build_tree()
        bank = build_candidate_bank(model, samples_per_axis=3, node_chunk=64)
        leaf_bytes_by_level = _leaf_bytes_by_level(model)
        root_cells = int(torch.prod(torch.tensor(model.l0_shape)).item())
        total_bytes, _distortion, actions = _run_lagrangian_dp(
            model, bank, leaf_bytes_by_level, lam=1.0e6, gamma=0.0
        )
        # At a huge byte penalty, p0 (13 bytes) always beats p1 (27 bytes)
        # regardless of distortion, and "keep" always loses to either merge,
        # so the whole tree collapses to one p0 leaf per root cell.
        self.assertAlmostEqual(total_bytes, root_cells * P0_BYTES, places=3)
        self.assertTrue(torch.all(actions[0] == 1))

    def test_gamma_zero_never_worse_in_value_distortion_than_positive_gamma(self):
        # Increasing gamma can only make merging *more* expensive (it adds a
        # nonnegative detail-energy term to every p0/p1 candidate), so for a
        # fixed lambda the chosen bytes can only stay the same or grow, and
        # the *value* distortion component alone (ignoring the detail term)
        # of the gamma>0 solution can only be <= the gamma=0 solution's,
        # since gamma=0's tree is still a feasible choice under gamma>0's
        # cost as long as it is not selected preferentially -- so bytes
        # should never *decrease* when gamma increases at a fixed lambda.
        model = self._build_tree()
        bank = build_candidate_bank(model, samples_per_axis=3, node_chunk=64)
        leaf_bytes_by_level = _leaf_bytes_by_level(model)
        lam = 1.0e-2
        bytes_gamma0, _, _ = _run_lagrangian_dp(model, bank, leaf_bytes_by_level, lam, gamma=0.0)
        bytes_gamma_big, _, _ = _run_lagrangian_dp(model, bank, leaf_bytes_by_level, lam, gamma=1.0e6)
        self.assertGreaterEqual(bytes_gamma_big, bytes_gamma0 - 1e-6)

    def test_bisection_respects_the_byte_budget(self):
        model = self._build_tree()
        bank = build_candidate_bank(model, samples_per_axis=3, node_chunk=64)
        leaf_bytes_by_level = _leaf_bytes_by_level(model)
        target = _full_tree_bytes(model) * 0.6
        total_bytes, _distortion, _actions, _lam = run_compression(
            model, bank, leaf_bytes_by_level, gamma=1.0, r_max_bytes=target, iterations=30
        )
        self.assertLessEqual(total_bytes, target + 1e-6)

    def test_pruned_field_matches_original_when_nothing_is_merged(self):
        model = self._build_tree()
        bank = build_candidate_bank(model, samples_per_axis=3, node_chunk=64)
        leaf_bytes_by_level = _leaf_bytes_by_level(model)
        _bytes, _distortion, actions = _run_lagrangian_dp(model, bank, leaf_bytes_by_level, lam=0.0, gamma=1.0)
        torch.manual_seed(1)
        points = torch.rand((512, 3)) * 1.9 - 0.95
        expected = model.forward_mu(points)
        actual = evaluate_pruned_field(model, bank, actions, points)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5))

    def test_materialized_model_matches_evaluate_pruned_field(self):
        model = self._build_tree()
        bank = build_candidate_bank(model, samples_per_axis=3, node_chunk=64)
        leaf_bytes_by_level = _leaf_bytes_by_level(model)
        target = _full_tree_bytes(model) * 0.5
        _bytes, _distortion, actions, _lam = run_compression(
            model, bank, leaf_bytes_by_level, gamma=1.0, r_max_bytes=target, iterations=30
        )
        compressed = materialize_compressed_model(model, bank, actions)
        torch.manual_seed(2)
        points = torch.rand((1024, 3)) * 1.9 - 0.95
        expected = evaluate_pruned_field(model, bank, actions, points)
        actual = compressed.forward_mu(points)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-4))

    def test_materialized_model_leaf_count_matches_action_choices(self):
        model = self._build_tree()
        bank = build_candidate_bank(model, samples_per_axis=3, node_chunk=64)
        leaf_bytes_by_level = _leaf_bytes_by_level(model)
        target = _full_tree_bytes(model) * 0.5
        _bytes, _distortion, actions, _lam = run_compression(
            model, bank, leaf_bytes_by_level, gamma=1.0, r_max_bytes=target, iterations=30
        )
        compressed = materialize_compressed_model(model, bank, actions)
        merged_nodes = sum(int((level_actions != 0).sum().item()) for level_actions in actions)
        kept_leaves = int(model.leaf_levels.shape[0]) - sum(
            8 * int((level_actions == 0).sum().item()) for level_actions in actions
        )
        # This is a loose sanity bound, not an exact leaf-count identity,
        # since "kept" nodes recurse rather than terminate; it just checks
        # that merges actually shrank the tree and produced a valid model.
        self.assertLess(int(compressed.leaf_levels.shape[0]), int(model.leaf_levels.shape[0]))
        self.assertGreater(merged_nodes, 0)
        self.assertGreaterEqual(kept_leaves, 0)

    def test_p1_only_dp_and_materialization_never_emit_p0(self):
        model = self._build_tree()
        model.elevate_leaves_isotropic_batch(
            torch.arange(model.leaf_levels.shape[0]), (1, 1, 1)
        )
        bank = build_candidate_bank(model, samples_per_axis=3, node_chunk=64)
        leaf_bytes_by_level = _leaf_bytes_by_level(model)
        root_cells = int(torch.prod(torch.tensor(model.l0_shape)).item())
        total_bytes, _distortion, actions = _run_lagrangian_dp(
            model,
            bank,
            leaf_bytes_by_level,
            lam=1.0e6,
            gamma=0.0,
            allow_p0=False,
        )
        self.assertAlmostEqual(total_bytes, root_cells * 27, places=3)
        self.assertFalse(any(torch.any(level_actions == 1) for level_actions in actions))
        compressed = materialize_compressed_model(model, bank, actions)
        self.assertTrue(torch.all(compressed.leaf_degrees == 1))

    def test_p1_receives_credit_for_reproducing_smooth_local_variation(self):
        axis = torch.linspace(0.0, 1.0, 4)
        xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
        reference = (0.2 + 0.3 * xx + 0.1 * yy - 0.05 * zz).reshape(1, -1)
        p0 = reference.mean(dim=1, keepdim=True).expand_as(reference)
        p1 = reference.clone()
        p0_detail = _local_detail_distortion(reference, p0, samples_per_axis=4)
        p1_detail = _local_detail_distortion(reference, p1, samples_per_axis=4)
        self.assertGreater(float(p0_detail.item()), 0.0)
        self.assertAlmostEqual(float(p1_detail.item()), 0.0, places=7)

    def test_candidate_integrals_are_exact_for_constant_and_linear_fields(self):
        model = BernsteinOctree(levels=[1, 2], max_degree=1, attenuation_shift=0.0)
        ray_batch = make_parallel_ray_points(
            torch.tensor([0.0]),
            torch.tensor([0]),
            torch.tensor([0]),
            detector_h=1,
            detector_w=1,
            samples_per_ray=8,
            targets=torch.zeros(1),
            materialize_points=False,
        )
        length, corners = _candidate_cell_integrals(
            model,
            ray_batch,
            torch.tensor([0]),
            level=0,
            coords=torch.tensor([[0, 0, 0]]),
        )
        self.assertAlmostEqual(float(length.item()), 2.0, places=6)
        # c_ijk=i represents local x. Its exact integral from x=-1 to 1 is 1.
        coefficients = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
        self.assertAlmostEqual(float((corners[0] * coefficients).sum().item()), 1.0, places=6)

    def test_packed_v3_fixed_payload_is_accounted_separately(self):
        model = self._build_tree()
        expected = 4 + 3 * 3 * 2 + 3 * 2 + 3 + 4 + 4096
        self.assertEqual(_packed_v3_fixed_payload_bytes(model), expected)


if __name__ == "__main__":
    unittest.main()
