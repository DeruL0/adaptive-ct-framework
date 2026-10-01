import unittest

from adaptive_ct.train import _convergence_window_summary, _dynamic_refinement_status


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

    def test_dynamic_refinement_stops_after_low_acceptance(self):
        running = _dynamic_refinement_status(
            tested=100,
            accepted_gains=[2.0] * 20,
            round_index=3,
            min_rounds=4,
            max_rounds=24,
            stop_acceptance_ratio=0.03,
            current_bytes=1024,
            byte_budget=8192,
            min_added_bytes=128,
        )
        stopped = _dynamic_refinement_status(
            tested=100,
            accepted_gains=[2.0, 4.0],
            round_index=4,
            min_rounds=4,
            max_rounds=24,
            stop_acceptance_ratio=0.03,
            current_bytes=1024,
            byte_budget=8192,
            min_added_bytes=128,
        )

        self.assertFalse(running["stop"])
        self.assertTrue(stopped["stop"])
        self.assertAlmostEqual(stopped["acceptance_ratio"], 0.02)
        self.assertAlmostEqual(stopped["mean_accepted_validation_gain"], 3.0)


if __name__ == "__main__":
    unittest.main()
