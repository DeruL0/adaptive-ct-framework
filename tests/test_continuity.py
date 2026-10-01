import unittest

import torch

from adaptive_ct.bernstein import BernsteinOctree
from adaptive_ct.losses import coefficient_face_continuity_loss


class ContinuityTests(unittest.TestCase):
    def test_cross_level_face_polynomial_continuity(self):
        model = BernsteinOctree(levels=[2, 4], max_degree=1, attenuation_shift=0.0)
        model.split_leaf(model.find_leaf((0, 0, 0, 0)))
        with torch.no_grad():
            model.coefficient_logits.copy_(
                model._coefficients_to_logits(torch.full_like(model.coefficient_logits, 0.5))
            )
        zero = coefficient_face_continuity_loss(
            model, max_pairs=4096, include_cross_level=True, face_quadrature_order=2
        )
        self.assertAlmostEqual(float(zero), 0.0, places=7)
        child = model.find_leaf((1, 1, 0, 0))
        with torch.no_grad():
            coefficient_id = int(model.coefficient_offsets[child])
            model.coefficient_logits[coefficient_id] = model._coefficients_to_logits(torch.tensor(0.9))
        nonzero = coefficient_face_continuity_loss(
            model, max_pairs=4096, include_cross_level=True, face_quadrature_order=2
        )
        self.assertGreater(float(nonzero), 0.0)
