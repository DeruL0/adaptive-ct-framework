import unittest

from adaptive_ct.train import _convergence_window_summary


class ConvergenceRefinementTests(unittest.TestCase):
    def test_plateau_is_ready(self):
        summary = _convergence_window_summary(
            [1.0] * 200,
            window=200,
            relative_improvement_threshold=0.002,
        )
        self.assertTrue(summary["ready"])
        self.assertAlmostEqual(summary["relative_improvement"], 0.0)

    def test_improving_window_is_not_ready(self):
        summary = _convergence_window_summary(
            [1.0 - 0.1 * index / 199.0 for index in range(200)],
            window=200,
            relative_improvement_threshold=0.002,
        )
        self.assertFalse(summary["ready"])
        self.assertGreater(summary["relative_improvement"], 0.002)

    def test_short_window_is_not_ready(self):
        summary = _convergence_window_summary(
            [1.0] * 50,
            window=200,
            relative_improvement_threshold=0.002,
        )
        self.assertFalse(summary["ready"])
        self.assertIsNone(summary["relative_improvement"])


if __name__ == "__main__":
    unittest.main()
