import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from adaptive_ct.core.geometry import detector_grid, parallel_rays
from adaptive_ct.representation.octree import BernsteinOctree
from adaptive_ct.training.loop import run_training

DETECTOR = 8


def _render(model, angle):
    rows, cols = detector_grid(DETECTOR, DETECTOR, torch.device("cpu"))
    batch = parallel_rays(torch.tensor(angle).expand(rows.numel()), rows, cols, detector_h=DETECTOR, detector_w=DETECTOR,
                          targets=torch.zeros(rows.numel()))
    return model.integrate_ray_batch(batch).detach().reshape(DETECTOR, DETECTOR).numpy()


def _write_dataset(root: Path, train_views: int = 12) -> Path:
    torch.manual_seed(11)
    truth = BernsteinOctree(levels=[2, 4, 8], leaf_types=("constant", "trilinear", "step"), cuda_integrator=False)
    truth.to_trilinear(torch.arange(truth.leaf_levels.shape[0]))
    with torch.no_grad():
        truth.values.copy_(0.02 + 0.4 * torch.rand(truth.values.shape[0]))
    dataset_root = root / "dataset"
    dataset_root.mkdir()
    np.save(dataset_root / "volume.npy", np.ones((8, 8, 8), dtype=np.float32) * 0.1)
    train, test = [], []
    for index in range(train_views):
        angle = 3.0 * index / train_views
        np.save(dataset_root / f"train_{index}.npy", _render(truth, angle))
        train.append({"file_path": f"train_{index}.npy", "angle": angle})
    for index, angle in enumerate((0.11, 1.7)):
        np.save(dataset_root / f"test_{index}.npy", _render(truth, angle))
        test.append({"file_path": f"test_{index}.npy", "angle": angle})
    metadata = {
        "scanner": {"mode": "parallel", "nDetector": [DETECTOR, DETECTOR], "nVoxel": [8, 8, 8]},
        "vol": "volume.npy",
        "proj_train": train,
        "proj_test": test,
    }
    (dataset_root / "meta_data.json").write_text(json.dumps(metadata), encoding="utf-8")
    return dataset_root


def _config(dataset_root: Path, output_root: Path, *, types: list) -> dict:
    return {
        "device": "cpu",
        "dataset": {"root": str(dataset_root), "dev_views": 3, "dev_seed": 1},
        "model": {"levels": [2, 4, 8], "leaf_types": types, "cuda_integrator": False},
        "training": {
            "seed": 3,
            "max_seconds": 600,
            "eval_ray_chunk": 4096,
            "solver": {"subsets": 3, "round_tol": 1e-2, "round_epochs": 6, "final_tol": 1e-3, "final_epochs": 8, "min_epochs": 2,
                       "min_leaf_px": 2.5, "edge_penalty": {"weight": 2300, "delta": 0.02}, "geometry_steps": 1},
            "refinement": {"action_share": 0.5, "stop_below_remaining": 0.01, "decision_views": 6, "min_footprint_px": 1.0, "byte_budget": 12000, "max_rounds": 3,
                           "continuation_share": 0.2, "error_share": 0.2, "spend_all_below": 0.0},
        },
        "metrics": {"compute_projection_metrics": True, "compute_volume_metrics": False, "compute_boundary_metrics": False},
        "output": {"dir": str(output_root)},
    }


class PipelineTests(unittest.TestCase):
    def _run(self, *, types: list, patch=None):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(_write_dataset(root), root / "output", types=types)
            if patch:
                patch(config)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            report = run_training(config_path)
            artifacts = {name: (root / "output" / name).exists() for name in
                         ("checkpoint.pt", "model.npz", "training_report.json", "states/pre_final_solve.pt", "states/pre_decision_last.pt")}
        return report, artifacts

    def test_three_type_run_refines_from_projections_and_stores_fp16(self):
        report, artifacts = self._run(types=["constant", "trilinear", "step"])
        events = [event for event in report["growth_events"] if sum(event["actions"].values())]
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(report["evaluation_protocol"]["training_views"], 9)
        self.assertEqual(report["evaluation_protocol"]["validation_views"], 3)
        self.assertIn("projection_test_deployed", report["codec"])
        self.assertGreaterEqual(len(report["rounds"]), 2)
        self.assertTrue(report["rounds"][-1]["final"])
        self.assertTrue(all(artifacts.values()), artifacts)
        self.assertEqual(report["status"], "complete")
        self.assertTrue(all(r["values_converged"] for r in report["rounds"]))  # no structure change on unconverged values
        self.assertTrue(all(isinstance(a, bool) for r in report["rounds"] for a in r["geometry_accepted"]))

    def test_tail_spends_the_rest_of_the_budget_in_one_decision(self):
        def patch(config):
            config["training"]["refinement"].update(spend_all_below=0.9, max_rounds=6)
        report, _ = self._run(types=["constant", "trilinear", "step"], patch=patch)
        shares = [event["action_share"] for event in report["growth_events"] if "action_share" in event]
        self.assertIn(1.0, shares)  # remaining fell below 90 %: the next decision may spend all of it
        self.assertTrue(all(s in (0.5, 1.0) for s in shares))

    def test_unconverged_values_stop_the_run_without_a_structure_change(self):
        def patch(config):
            config["training"]["max_seconds"] = 1e-6
            config["training"]["solver"].update(round_epochs=2, final_epochs=2, round_tol=-1.0, final_tol=-1.0, min_epochs=1)

        report, _ = self._run(types=["constant", "trilinear", "step"], patch=patch)
        self.assertEqual(report["status"], "time_limit_not_converged")
        self.assertEqual(len(report["rounds"]), 1)
        self.assertFalse(report["rounds"][0]["values_converged"])
        self.assertEqual(report["growth_events"], [])  # the tree was not refined

    def test_unconverged_values_are_solved_on_before_the_structure_changes(self):
        def patch(config):
            config["training"]["solver"].update(round_epochs=1, round_tol=0.5, min_epochs=2)

        report, _ = self._run(types=["constant", "trilinear", "step"], patch=patch)
        self.assertEqual(report["status"], "complete")
        self.assertGreater(max(r["continuation_solves"] for r in report["rounds"]), 0)
        self.assertTrue(all(r["values_converged"] for r in report["rounds"]))

    def test_constant_control_runs(self):
        report, artifacts = self._run(types=["constant"])
        self.assertGreaterEqual(len([e for e in report["growth_events"] if sum(e["actions"].values())]), 1)
        self.assertEqual(list(report["model"]["leaf_type_counts"][1:]), [0, 0])
        self.assertTrue(artifacts["model.npz"])


if __name__ == "__main__":
    unittest.main()
