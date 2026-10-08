import unittest
from pathlib import Path

import torch

from adaptive_ct.core.geometry import detector_grid, parallel_rays
from adaptive_ct.core.projector import has_bernstein_native
from adaptive_ct.data.dataset import ProjectionSplit
from adaptive_ct.representation.octree import BernsteinOctree
from adaptive_ct.training.geometry import geometry_step, statistics

DETECTOR = 12


def _step_model(geometry, values=(0.1, 0.6), cuda_integrator=False):
    model = BernsteinOctree(levels=[1, 2], cuda_integrator=cuda_integrator)
    model.set_types([0], [2], torch.tensor(values), torch.tensor([geometry]))
    return model


def _two_level_model(cuda_integrator=False):
    model = BernsteinOctree(levels=[2, 4], cuda_integrator=cuda_integrator)
    model.to_trilinear([1, 6])
    g = torch.Generator().manual_seed(3)
    n = torch.randn(3, 3, generator=g)
    n = n / n.norm(dim=1, keepdim=True)
    geometry = torch.cat([n, 0.2 * (torch.rand(3, 1, generator=g) - 0.5), torch.tensor([[0.08], [0.15], [0.3]])], 1)
    model.set_types([0, 3, 5], [2, 2, 2], torch.rand(6, generator=g) + 0.05, geometry)
    with torch.no_grad():
        model.values.copy_(0.05 + torch.rand(model.values.shape, generator=g))
    return model


def _split(truth, views=18):
    angles = torch.linspace(0.0, 3.0, views)
    rows, cols = detector_grid(DETECTOR, DETECTOR, torch.device("cpu"))
    proj = torch.stack([truth.integrate_ray_batch(parallel_rays(a.expand(rows.numel()), rows, cols, detector_h=DETECTOR, detector_w=DETECTOR,
                                                                targets=torch.zeros(rows.numel()))).detach().reshape(DETECTOR, DETECTOR)
                        for a in angles])
    return ProjectionSplit(angles=angles, projections=proj, paths=[Path(str(i)) for i in range(views)])


class GeometryStatisticsTests(unittest.TestCase):
    def test_gradient_is_the_derivative_of_the_data_term(self):
        model = _two_level_model()
        split = _split(_two_level_model(), views=6)
        with torch.no_grad():
            split.projections.mul_(1.3)  # data the model does not fit
        accum, J = statistics(model, split.angles, split.projections, list(range(6)))
        model.step_geometry.requires_grad_(True)
        rows, cols = detector_grid(DETECTOR, DETECTOR, torch.device("cpu"))
        loss = 0.0
        for v in range(6):
            b = parallel_rays(split.angles[v].expand(rows.numel()), rows, cols, detector_h=DETECTOR, detector_w=DETECTOR,
                              targets=split.projections[v].reshape(-1))
            loss = loss + 0.5 * ((model.integrate_ray_batch(b) - b.target) ** 2).sum()
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), J, places=4)
        self.assertTrue(torch.allclose(accum[:, :5], model.step_geometry.grad[:, :5], rtol=1e-3, atol=1e-6))
        # Gauss-Newton matrix: positive semidefinite, diagonal = sum of squared derivatives
        self.assertTrue(bool((accum[:, [5, 10, 14, 17, 19]] >= 0).all()))

    @unittest.skipUnless(torch.cuda.is_available() and has_bernstein_native(), "Bernstein CUDA extension unavailable")
    def test_cuda_statistics_match_reference(self):
        model = _two_level_model()
        split = _split(_two_level_model(), views=6)
        split.projections.mul_(1.3)
        reference, J_ref = statistics(model, split.angles, split.projections, list(range(6)))
        model = model.cuda()
        model.cuda_integrator = True
        cuda_split = ProjectionSplit(angles=split.angles.cuda(), projections=split.projections.cuda(), paths=split.paths)
        actual, J = statistics(model, cuda_split.angles, cuda_split.projections, list(range(6)))
        scale = reference.abs().amax(0) + 1e-12
        self.assertLess(float(((actual.cpu() - reference).abs() / scale).max()), 2e-3)
        self.assertAlmostEqual(J, J_ref, delta=1e-4 * J_ref)


class GeometryStepTests(unittest.TestCase):
    def test_steps_move_a_misplaced_edge_towards_the_data(self):
        truth = _step_model([0.8, 0.6, 0.0, 0.1, 0.03])
        split = _split(truth)
        model = _step_model([0.6, 0.8, 0.0, -0.05, 0.06])
        error = lambda: float((model.step_geometry[0, :4] - truth.step_geometry[0, :4]).norm())
        before = error()
        reports = [geometry_step(model, split, detector_bin=1, penalty_weight=0.0, penalty_delta=0.02) for _ in range(8)]
        self.assertTrue(all(r["objective_after"] <= r["objective_before"] for r in reports))
        self.assertGreater(reports[0]["accepted_step"], 0.0)
        self.assertLess(error(), 0.3 * before)
        self.assertAlmostEqual(float(model.step_geometry[0, :3].norm()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
