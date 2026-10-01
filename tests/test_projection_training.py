import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from adaptive_ct.train import run_training


class ProjectionDomainTrainingTests(unittest.TestCase):
    def test_tiny_training_run_stays_projection_only_and_records_adaptation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            output_root = root / "output"
            dataset_root.mkdir()
            np.save(dataset_root / "volume.npy", np.ones((2, 2, 2), dtype=np.float32) * 0.5)
            frames_train = []
            for index, angle in enumerate((0.0, 0.8)):
                name = f"train_{index}.npy"
                np.save(dataset_root / name, np.ones((2, 2), dtype=np.float32))
                frames_train.append({"file_path": name, "angle": angle})
            np.save(dataset_root / "test_0.npy", np.ones((2, 2), dtype=np.float32))
            metadata = {
                "scanner": {"mode": "parallel", "nDetector": [2, 2], "nVoxel": [2, 2, 2]},
                "vol": "volume.npy",
                "proj_train": frames_train,
                "proj_test": [{"file_path": "test_0.npy", "angle": 0.4}],
            }
            (dataset_root / "meta_data.json").write_text(json.dumps(metadata), encoding="utf-8")
            config = {
                "device": "cpu",
                "dataset": {"root": str(dataset_root)},
                "geometry": {"samples_per_ray": 4},
                "model": {
                    "representation": "bernstein_octree",
                    "levels": [1, 2],
                    "max_degree": [1, 1, 1],
                    "integration_mode": "exact",
                    "initialization": "projection_mean",
                    "attenuation_shift": 0.0,
                },
                "training": {
                    "iterations": 2,
                    "batch_rays": 4,
                    "lr": 0.01,
                    "tv_weight": 0.0,
                    "log_every": 1,
                    "compute_eval": False,
                    "projection_gradient_loss": {
                        "weight": 0.1,
                        "edge_rays": 2,
                        "strides": [1],
                        "uniform_fraction": 0.5,
                        "candidate_multiplier": 2,
                    },
                    "adaptive": {
                        "enabled": True,
                        "rounds": 1,
                        "start_iteration": 2,
                        "max_candidates": 0,
                        "max_reverse_candidates": 0,
                    },
                },
                "metrics": {
                    "material_threshold": 0.1,
                    "compute_projection_metrics": False,
                    "compute_volume_metrics": False,
                    "compute_boundary_metrics": False,
                },
                "export": {"mact": {"enabled": False}, "surface": {"enabled": False}},
                "output": {"dir": str(output_root)},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            report = run_training(config_path)

            self.assertEqual(report["model"]["representation"], "bernstein_octree")
            self.assertEqual(report["training_objective"]["projection_loss"], "weighted_least_squares")
            self.assertFalse(report["training_objective"]["volume_loss"]["enabled"])
            self.assertTrue(report["training_objective"]["projection_gradient_loss"]["enabled"])
            self.assertIn("projection_gradient_loss", report["history"][0])
            self.assertFalse(report["training_objective"]["dense_volume_loaded_during_training"])
            self.assertEqual(len(report["adaptive_events"]), 1)
            self.assertTrue((output_root / "checkpoint.pt").exists())
            self.assertTrue((output_root / "training_report.json").exists())


if __name__ == "__main__":
    unittest.main()
