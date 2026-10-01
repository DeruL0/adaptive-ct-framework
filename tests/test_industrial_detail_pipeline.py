import tempfile
import unittest
from pathlib import Path

import yaml

from adaptive_ct.industrial_detail_pipeline import PIPELINE_SCHEMA, build_commands


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _reference_config(output: Path) -> dict:
    return {
        "device": "cuda",
        "dataset": {"root": "projections"},
        "geometry": {"samples_per_ray": 128},
        "model": {
            "representation": "bernstein_octree",
            "levels": [48, 96, 192, 384],
            "topology": "packed_hierarchy",
            "initialization": "projection_mean",
        },
        "training": {
            "tv_weight": 0.0,
            "projection_weighting_schedule": {1: {"mode": "uniform"}},
            "coefficient_continuity": {"weight": 0.0},
            "progressive": {
                "convergence_milestones": [{"operation": "h_jump_rd", "levels": [0, 1, 2]}]
            },
        },
        "metrics": {"compute_volume_metrics": False, "compute_boundary_metrics": False},
        "output": {"dir": str(output)},
    }


class IndustrialDetailPipelineTests(unittest.TestCase):
    def test_derives_internal_checkpoint_and_never_accepts_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.yaml"
            _write(reference_path, _reference_config(root / "reference_out"))
            pipeline_path = root / "pipeline.yaml"
            _write(
                pipeline_path,
                {
                    "schema": PIPELINE_SCHEMA,
                    "device": "cuda",
                    "reference": {"config": reference_path.name},
                    "compression": {"raw_packed_budget_bytes": 20_000_000, "quantization": "float16"},
                    "output": {"dir": str(root / "final")},
                },
            )
            resolved, train_command, compress_command = build_commands(pipeline_path)
            self.assertEqual(Path(resolved["reference_checkpoint"]), root / "reference_out" / "checkpoint.pt")
            self.assertIn("adaptive_ct.train", train_command)
            self.assertIn("adaptive_ct.functional_compression", compress_command)
            self.assertIn(resolved["reference_checkpoint"], compress_command)

            payload = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
            payload["checkpoint"] = "external.pt"
            _write(pipeline_path, payload)
            with self.assertRaisesRegex(ValueError, "does not accept an external"):
                build_commands(pipeline_path)

    def test_p1_only_switch_is_forwarded_to_compressor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.yaml"
            reference = _reference_config(root / "reference_out")
            reference["model"]["initial_leaf_degree"] = [1, 1, 1]
            _write(reference_path, reference)
            pipeline_path = root / "pipeline.yaml"
            _write(
                pipeline_path,
                {
                    "schema": PIPELINE_SCHEMA,
                    "reference": {"config": reference_path.name},
                    "compression": {
                        "raw_packed_budget_bytes": 20_000_000,
                        "p1_only": True,
                    },
                    "output": {"dir": str(root / "final")},
                },
            )
            _resolved, _train, compress = build_commands(pipeline_path)
            self.assertIn("--p1-only", compress)

    def test_rejects_non_v5_reference_rules(self):
        mutations = [
            (lambda c: c["training"].update({"tv_weight": 0.1}), "tv_weight"),
            (
                lambda c: c["training"]["progressive"]["convergence_milestones"][0].update(
                    {"operation": "p_elevate"}
                ),
                "not h-only",
            ),
            (
                lambda c: c["training"]["progressive"].update({"support_mask": "mask.npy"}),
                "forbidden object-specific",
            ),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                reference = _reference_config(root / "reference_out")
                mutate(reference)
                _write(root / "reference.yaml", reference)
                _write(
                    root / "pipeline.yaml",
                    {
                        "schema": PIPELINE_SCHEMA,
                        "reference": {"config": "reference.yaml"},
                        "compression": {"raw_packed_budget_bytes": 20_000_000},
                        "output": {"dir": str(root / "final")},
                    },
                )
                with self.assertRaisesRegex(ValueError, message):
                    build_commands(root / "pipeline.yaml")


if __name__ == "__main__":
    unittest.main()
