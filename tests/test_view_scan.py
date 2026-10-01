import unittest

import numpy as np

from adaptive_ct.view_scan import common_test_angles, interleaved_holdout_angles, uniform_train_angles


class ViewScanAngleTests(unittest.TestCase):
    def test_common_test_angles_are_disjoint_from_48_and_128_grids(self):
        test = common_test_angles(4)
        for count in (48, 128):
            train = uniform_train_angles(count)
            wrapped = np.abs(np.angle(np.exp(1j * (train[:, None] - test[None, :]))))
            self.assertGreater(float(wrapped.min()), 1.0e-6)

    def test_uniform_train_angles_cover_independent_parallel_half_turn(self):
        angles = uniform_train_angles(48)
        self.assertEqual(angles.shape, (48,))
        self.assertEqual(float(angles[0]), 0.0)
        self.assertLess(float(angles[-1]), np.pi)
        np.testing.assert_allclose(np.diff(angles), np.diff(angles)[0])
        self.assertAlmostEqual(float(np.diff(angles)[0]), np.pi / 48.0)

    def test_common_test_angles_are_unique_modulo_parallel_symmetry(self):
        angles = common_test_angles(4)
        self.assertTrue(np.all(angles >= 0.0))
        self.assertTrue(np.all(angles < np.pi))
        wrapped = np.mod(angles, np.pi)
        self.assertEqual(np.unique(np.round(wrapped, decimals=12)).size, angles.size)

    def test_validation_and_final_test_grids_are_mutually_disjoint(self):
        train = uniform_train_angles(128)
        validation = interleaved_holdout_angles(16, train_views=128, phase=1.0 / 3.0)
        test = interleaved_holdout_angles(16, train_views=128, phase=2.0 / 3.0)
        for left, right in ((train, validation), (train, test), (validation, test)):
            wrapped = np.abs(np.angle(np.exp(1j * (left[:, None] - right[None, :]))))
            self.assertGreater(float(wrapped.min()), 1.0e-6)
        self.assertTrue(np.all(validation >= 0.0))
        self.assertTrue(np.all(validation < np.pi))
        self.assertTrue(np.all(test >= 0.0))
        self.assertTrue(np.all(test < np.pi))


if __name__ == "__main__":
    unittest.main()
