import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from adaptive_ct.representation.octree import BernsteinOctree
from adaptive_ct.core.geometry import detector_grid, make_parallel_ray_points
from adaptive_ct.training.loop import run_training

DETECTOR = 8


def _render(model, angle):
    rows, cols = detector_grid(DETECTOR, DETECTOR, torch.device("cpu"))
    batch = make_parallel_ray_points(
        torch.tensor(angle).expand(rows.numel()),
        rows,
        cols,
        detector_h=DETECTOR,
        detector_w=DETECTOR,
        samples_per_ray=8,
        targets=torch.zeros(rows.numel()),
        materialize_points=False,
    )
    return model.integrate_ray_batch(batch).detach().reshape(DETECTOR, DETECTOR).numpy()


def _write_dataset(root: Path, train_views: int = 12) -> Path:
    torch.manual_seed(11)
    truth = BernsteinOctree(
        levels=[2, 4, 8],
        max_degree=1,
        attenuation_shift=0.0,
        cuda_integrator=False,
        init_std=0.0,
    )
    truth.elevate_leaves_isotropic_batch(torch.arange(truth.leaf_levels.shape[0]), (1, 1, 1))
    with torch.no_grad():
        values = 0.02 + 0.4 * torch.rand(truth.coefficient_logits.shape[0])
        truth.coefficient_logits.copy_(truth._coefficients_to_logits(values))
    dataset_root = root / "dataset"
    dataset_root.mkdir()
    np.save(dataset_root / "volume.npy", np.ones((8, 8, 8), dtype=np.float32) * 0.1)
    train, test = [], []
    for index in range(train_views):
        angle = 3.0 * index / train_views
        name = f"train_{index}.npy"
        np.save(dataset_root / name, _render(truth, angle))
        train.append({"file_path": name, "angle": angle})
    for index, angle in enumerate((0.11, 1.7)):
        name = f"test_{index}.npy"
        np.save(dataset_root / name, _render(truth, angle))
        test.append({"file_path": name, "angle": angle})
    metadata = {
        "scanner": {"mode": "parallel", "nDetector": [DETECTOR, DETECTOR], "nVoxel": [8, 8, 8]},
        "vol": "volume.npy",
        "proj_train": train,
        "proj_test": test,
    }
    (dataset_root / "meta_data.json").write_text(json.dumps(metadata), encoding="utf-8")
    return dataset_root


def _config(dataset_root: Path, output_root: Path, *, codec: dict, degree) -> dict:
    return {
        "device": "cpu",
        "dataset": {"root": str(dataset_root), "dev_views": 3, "dev_seed": 1},
        "geometry": {"samples_per_ray": 8},
        "model": {
            "representation": "bernstein_octree",
            "levels": [2, 4, 8],
            "max_degree": degree,
            **({"initial_leaf_degree": [1, 1, 1]} if degree == [1, 1, 1] else {}),
            "integration_mode": "exact",
            "cuda_integrator": False,
            "initialization": "projection_mean",
            "attenuation_shift": 0.0,
        },
        "training": {
            "seed": 3,
            "eval_ray_chunk": 4096,
            "allow_test_during_training": False,
            "solver": {"groups": 3, "first_group_passes": 2, "group_passes": 1, "round_tol": 1e-3, "round_passes": 6, "final_tol": 1e-4, "final_passes": 8},
            "refinement": {"gain_share": 0.5, "estimator_rays": 256, "min_footprint_px": 1.0, "byte_budget": 12000, "max_rounds": 3, "codec": codec},
        },
        "metrics": {
            "compute_projection_metrics": True,
            "compute_volume_metrics": False,
            "compute_boundary_metrics": False,
        },
        "export": {
            "codec": {**codec, "finetune_iterations": 5, "finetune_lr": 0.01},
            "compact_octree": {"enabled": True, "artifact_name": "ANVO.npz", "quantization": "float16", "topology": "packed_hierarchy"},
        },
        "output": {"dir": str(output_root)},
    }


class DorflerTrainingTests(unittest.TestCase):
    def _run(self, *, codec: dict, degree):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = _write_dataset(root)
            config = _config(dataset_root, root / "output", codec=codec, degree=degree)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            report = run_training(config_path)
            artifacts = {
                name: (root / "output" / name).exists()
                for name in ("checkpoint.pt", "checkpoint_codec.pt", "ANVO.npz", "training_report.json")
            }
        return report, artifacts

    def test_latent_p1_run_refines_from_projections_and_exports_codec(self):
        report, artifacts = self._run(
            codec={"type": "latent_p1", "latent_dim": 4, "quantization": "float16"},
            degree=[1, 1, 1],
        )
        events = [event for event in report["growth_events"] if event.get("active_splits")]
        self.assertGreaterEqual(len(events), 1)
        self.assertGreater(events[0]["active_splits"], 0)
        self.assertEqual(report["evaluation_protocol"]["training_views"], 9)
        self.assertEqual(report["evaluation_protocol"]["validation_views"], 3)
        self.assertEqual(report["model"]["representation"], "latent_p1_octree")
        self.assertIn("projection_test_deployed", report["codec"])
        self.assertIn("projection_test_after_pca", report["codec"])
        self.assertTrue(report["codec"]["quantization_applied_in_eval"])
        self.assertGreaterEqual(len(report["rounds"]), 2)
        self.assertTrue(report["rounds"][-1]["final"])
        self.assertTrue(all(artifacts.values()), artifacts)

    def test_p0_control_runs(self):
        report, artifacts = self._run(
            codec={"type": "p0", "quantization": "float16"},
            degree=[0, 0, 0],
        )
        events = [event for event in report["growth_events"] if event.get("active_splits")]
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(report["model"]["representation"], "bernstein_octree")
        self.assertFalse(artifacts["checkpoint_codec.pt"])
        self.assertTrue(artifacts["ANVO.npz"])


if __name__ == "__main__":
    unittest.main()
