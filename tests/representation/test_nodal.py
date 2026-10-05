import unittest
from pathlib import Path

import torch

from adaptive_ct.data.dataset import ProjectionSplit
from adaptive_ct.evaluation.render import render_split
from adaptive_ct.representation.nodal import build_nodal_map
from adaptive_ct.representation.octree import BernsteinOctree
from adaptive_ct.training.solve import solve


def _mixed_tree():
    """Leaves at three levels, including coarse leaves next to fine ones."""
    m = BernsteinOctree(levels=[2, 4, 8], max_degree=1, attenuation_shift=0.0, cuda_integrator=False, init_std=0.0)
    m.elevate_leaves_isotropic_batch(torch.arange(m.leaf_levels.shape[0]), (1, 1, 1))
    m.split_leaves_batch(torch.tensor([0]))
    m.split_leaves_batch(torch.tensor([int(torch.nonzero(m.leaf_levels == 1)[3])]))
    return m


def _set(m, c):
    with torch.no_grad():
        m.coefficient_logits.copy_(m._coefficients_to_logits(c.clamp_min(1e-12)))


def _max_face_jump(m, seed=0):
    """Largest |f(p+) - f(p-)| over random points on the grid planes of the finest level."""
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(4000, 3, generator=g) * 2 - 1
    axis = torch.randint(0, 3, (4000,), generator=g)
    k = torch.randint(1, 8, (4000,), generator=g).float()
    p[torch.arange(4000), axis] = -1 + k * 2 / 8
    e = torch.zeros(4000, 3)
    e[torch.arange(4000), axis] = 1e-6
    with torch.no_grad():
        return float((m.forward_mu(p + e) - m.forward_mu(p - e)).abs().max())


class SharedCornerTests(unittest.TestCase):
    def test_uniform_grid_has_one_node_per_lattice_corner(self):
        m = BernsteinOctree(levels=[2, 4], max_degree=1, cuda_integrator=False, init_std=0.0)
        m.elevate_leaves_isotropic_batch(torch.arange(8), (1, 1, 1))
        self.assertEqual(build_nodal_map(m).free_nodes, 27)

    def test_weights_are_convex_and_the_field_is_continuous_across_coarse_fine_faces(self):
        m = _mixed_tree()
        nm = build_nodal_map(m)
        dense = nm.P.to_dense()
        self.assertTrue(bool((dense >= 0).all()))
        self.assertTrue(torch.allclose(dense.sum(1), torch.ones(dense.shape[0]), atol=1e-6))
        torch.manual_seed(1)
        _set(m, nm.to_corners(torch.rand(nm.free_nodes)))
        self.assertLess(_max_face_jump(m), 1e-4)
        _set(m, torch.rand(m.coefficient_logits.shape[0]))  # independent corners do jump
        self.assertGreater(_max_face_jump(m), 1e-2)

    def test_solver_in_node_space_fits_a_continuous_truth(self):
        torch.manual_seed(2)
        truth = _mixed_tree()
        nm = build_nodal_map(truth)
        _set(truth, nm.to_corners(0.05 + 0.4 * torch.rand(nm.free_nodes)))
        angles = torch.linspace(0.0, 3.1, 20)
        empty = ProjectionSplit(angles=angles, projections=torch.zeros(20, 8, 8), paths=[Path("x")] * 20)
        split = ProjectionSplit(angles=angles, projections=render_split(truth, empty, detector_h=8, detector_w=8, samples_per_ray=8, ray_chunk=4096).detach(), paths=empty.paths)
        model = _mixed_tree()
        _set(model, torch.full((model.coefficient_logits.shape[0],), 0.2))
        c, hist = solve(model, split, detector_h=8, detector_w=8, passes=60, groups=4, group_passes=3, nodal=nm)
        self.assertLess(hist[-1]["objective_at_eval_point"], 1e-2 * hist[0]["objective_at_eval_point"])
        self.assertGreaterEqual(float(c.min()), 0.0)
        self.assertLess(_max_face_jump(model), 1e-4)


if __name__ == "__main__":
    unittest.main()
