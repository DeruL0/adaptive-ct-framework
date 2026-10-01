import unittest

from adaptive_ct.accuracy_acceptance import build_report


def _quantization(psnr, high_psnr, distance):
    metrics = {
        "full": {"psnr": psnr, "mae": 0.01},
        "high_gradient_top1": {
            "psnr": high_psnr,
            "mae": 0.02,
            "gradient_ratio_distance": distance,
        },
    }
    return {
        "results": {
            "float16": {"metrics": metrics},
            "uint8": {
                "metrics": {
                    "full": {"psnr": psnr - 0.1, "mae": 0.01},
                    "high_gradient_top1": {
                        "psnr": high_psnr,
                        "mae": 0.02,
                        "gradient_ratio_distance": distance,
                    },
                }
            },
        }
    }


def _training(psnr, leaves, coefficients):
    return {
        "projection_test": {"psnr": psnr, "mae": 0.01},
        "model": {
            "leaf_cells": leaves,
            "coefficient_count": coefficients,
            "active_by_level": [1, 1, 1, max(0, leaves - 3)],
        },
        "elapsed_sec": 10.0,
        "total_elapsed_sec": 12.0,
        "peak_allocated_bytes": 1024**3,
        "peak_reserved_bytes": 2 * 1024**3,
    }


class AccuracyAcceptanceTests(unittest.TestCase):
    def test_rejects_l3_when_only_full_psnr_improves(self):
        old_benchmark = {"cases": [{"cuda_native": {"step_ms": 1.0}}]}
        packed_benchmark = {"cases": [{"cuda_native": {"step_ms": 1.05}}]}
        report = build_report(
            _quantization(32.0, 22.0, 0.6),
            _training(32.5, 100, 100),
            _quantization(32.5, 22.6, 0.5),
            _training(32.6, 100, 120),
            _quantization(32.6, 22.7, 0.49),
            _training(32.6001, 180, 220),
            _quantization(32.6001, 22.69, 0.50),
            old_benchmark,
            packed_benchmark,
        )
        self.assertEqual(report["selected_variant"], "WP")
        self.assertFalse(report["l3_retention"]["retained"])
        self.assertGreater(report["l3_retention"]["full_psnr_delta_db"], 0.0)
        self.assertLess(report["l3_retention"]["high_gradient_psnr_delta_db"], 0.0)


if __name__ == "__main__":
    unittest.main()
