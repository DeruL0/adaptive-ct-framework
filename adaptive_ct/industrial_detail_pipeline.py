"""Industrial CT detail-preserving adaptive functional-field pipeline v5.

This is the single, auditable entry point for the method. It deliberately
does not accept a reference checkpoint: the h-only reference field must be
reconstructed from sparse-view projections by this invocation before the
functional subtree compressor is allowed to run.

Data flow:
projections -> h-only adaptive reconstruction -> frozen mu_ref
            -> keep/p0/p1 subtree R-D selection
            -> fixed-topology projection-only coefficient refinement
            -> compact packed hierarchy
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import load_config


PIPELINE_SCHEMA = "industrial_ct_detail_functional_field_v5"
EXPECTED_LEVELS = [48, 96, 192, 384]
H_ONLY_OPERATIONS = {
    "h",
    "h_split",
    "split",
    "h_jump_rd",
    "h_jump_rate_distortion",
}


def _resolve_from_config(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _zero(value: Any) -> bool:
    return value is None or float(value) == 0.0


def _validate_reference_config(config: dict[str, Any]) -> None:
    """Enforce the exact projection-only h-reference contract."""
    errors: list[str] = []
    model = config.get("model", {}) or {}
    training = config.get("training", {}) or {}
    progressive = training.get("progressive", {}) or {}
    metrics = config.get("metrics", {}) or {}

    if str(model.get("representation", "")).lower() != "bernstein_octree":
        errors.append("model.representation must be bernstein_octree")
    if list(model.get("levels", [])) != EXPECTED_LEVELS:
        errors.append(f"model.levels must be {EXPECTED_LEVELS}")
    if str(model.get("topology", "")).lower() != "packed_hierarchy":
        errors.append("model.topology must be packed_hierarchy")
    if str(model.get("initialization", "projection_mean")).lower() not in {
        "projection_mean",
        "random",
    }:
        errors.append("reference initialization must not use a volume")
    if not _zero(training.get("tv_weight", 0.0)):
        errors.append("training.tv_weight must be 0")
    if not _zero((training.get("coefficient_continuity", {}) or {}).get("weight", 0.0)):
        errors.append("coefficient_continuity.weight must be 0")
    if not _zero((training.get("volume_loss", {}) or {}).get("weight", 0.0)):
        errors.append("volume_loss.weight must be 0")
    if not _zero((training.get("projection_gradient_loss", {}) or {}).get("weight", 0.0)):
        errors.append("projection_gradient_loss.weight must be 0")

    schedule = training.get("projection_weighting_schedule", {}) or {}
    if not schedule:
        errors.append("projection_weighting_schedule must explicitly select uniform WLS")
    for iteration, value in schedule.items():
        if str((value or {}).get("mode", "")).lower() not in {"uniform", "none", "mse"}:
            errors.append(f"projection weighting at iteration {iteration} must be uniform")

    milestones: list[dict[str, Any]] = list(progressive.get("convergence_milestones", []) or [])
    milestones.extend((progressive.get("milestones", {}) or {}).values())
    if not milestones:
        errors.append("the h-reference needs at least one adaptive h-refinement milestone")
    for index, milestone in enumerate(milestones):
        operation = str((milestone or {}).get("operation", "h_split")).lower()
        if operation not in H_ONLY_OPERATIONS:
            errors.append(f"refinement milestone {index} is not h-only: {operation}")

    forbidden_keys = {
        "support_mask",
        "support_threshold",
        "material_threshold",
        "force_material",
        "force_split",
        "minimum_resolution",
        "min_active_level",
        "level_quota",
    }

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if str(key).lower() in forbidden_keys:
                    errors.append(f"forbidden object-specific rule: {child_path}")
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(progressive, "training.progressive")
    if bool(metrics.get("compute_volume_metrics", False)):
        errors.append("compute_volume_metrics must be false for the reference stage")
    if bool(metrics.get("compute_boundary_metrics", False)):
        errors.append("compute_boundary_metrics must be false for the reference stage")
    if errors:
        raise ValueError("Invalid industrial-detail v5 h-reference config:\n- " + "\n- ".join(errors))


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def build_commands(pipeline_config_path: str | Path) -> tuple[dict[str, Any], list[str], list[str]]:
    pipeline_path = Path(pipeline_config_path).resolve()
    pipeline = load_config(pipeline_path)
    if str(pipeline.get("schema", "")) != PIPELINE_SCHEMA:
        raise ValueError(f"schema must be {PIPELINE_SCHEMA!r}")
    if "checkpoint" in pipeline or "reference_checkpoint" in pipeline:
        raise ValueError("This pipeline does not accept an external reference checkpoint.")

    reference = pipeline.get("reference", {}) or {}
    compression = pipeline.get("compression", {}) or {}
    output = pipeline.get("output", {}) or {}
    reference_config_value = reference.get("config")
    if not reference_config_value:
        raise ValueError("reference.config is required")
    reference_config_path = _resolve_from_config(pipeline_path, reference_config_value)
    reference_config = load_config(reference_config_path)
    _validate_reference_config(reference_config)

    reference_output = Path(reference_config["output"]["dir"])
    if not reference_output.is_absolute():
        reference_output = (Path.cwd() / reference_output).resolve()
    reference_checkpoint = reference_output / "checkpoint.pt"

    final_output_value = output.get("dir")
    if not final_output_value:
        raise ValueError("output.dir is required")
    final_output = Path(final_output_value)
    if not final_output.is_absolute():
        final_output = (Path.cwd() / final_output).resolve()

    raw_budget = int(compression.get("raw_packed_budget_bytes", 0))
    if raw_budget <= 0:
        raise ValueError("compression.raw_packed_budget_bytes must be positive")
    if str(compression.get("quantization", "float16")).lower() != "float16":
        raise ValueError("v5 deployment coefficients must use float16")

    train_command = [sys.executable, "-m", "adaptive_ct.train", "--config", str(reference_config_path)]
    compress_command = [
        sys.executable,
        "-m",
        "adaptive_ct.functional_compression",
        "--config",
        str(reference_config_path),
        "--checkpoint",
        str(reference_checkpoint),
        "--r-max-bytes",
        str(raw_budget),
        "--samples-per-axis",
        str(int(compression.get("samples_per_axis", 4))),
        "--max-workspace-mb",
        str(float(compression.get("max_workspace_mb", 2048.0))),
        "--finetune-iterations",
        str(int(compression.get("finetune_iterations", 15000))),
        "--finetune-batch-rays",
        str(int(compression.get("finetune_batch_rays", 65536))),
        "--finetune-lr",
        str(float(compression.get("finetune_lr", 0.015))),
        "--eval-ray-chunk",
        str(int(compression.get("eval_ray_chunk", 65536))),
        "--score-ray-chunk",
        str(int(compression.get("score_ray_chunk", 8192))),
        "--score-views",
        str(int(compression.get("score_views", 0))),
        "--score-rays-per-view",
        str(int(compression.get("score_rays_per_view", 8192))),
        "--device",
        str(pipeline.get("device", "cuda")),
        "--out-checkpoint",
        str(final_output / "checkpoint.pt"),
        "--out-compact",
        str(final_output / "compact_octree.npz"),
        "--compact-quantization",
        "float16",
        "--out-report",
        str(final_output / "compression_report.json"),
    ]
    if bool(compression.get("p1_only", False)):
        compress_command.append("--p1-only")
    resolved = {
        "pipeline_path": str(pipeline_path),
        "reference_config": str(reference_config_path),
        "reference_output": str(reference_output),
        "reference_checkpoint": str(reference_checkpoint),
        "final_output": str(final_output),
        "raw_packed_budget_bytes": raw_budget,
        "device": str(pipeline.get("device", "cuda")),
    }
    return resolved, train_command, compress_command


def run_pipeline(pipeline_config_path: str | Path, *, validate_only: bool = False) -> dict[str, Any]:
    resolved, train_command, compress_command = build_commands(pipeline_config_path)
    final_output = Path(resolved["final_output"])
    final_output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": PIPELINE_SCHEMA,
        "status": "validated" if validate_only else "running",
        "started_at_unix": time.time(),
        **resolved,
        "stages": {
            "h_reference": _command_text(train_command),
            "functional_compression": _command_text(compress_command),
        },
        "provenance": {
            "reference_source": "sparse_view_projections_only",
            "external_reference_checkpoint_allowed": False,
            "ground_truth_used_for_training_or_selection": False,
        },
    }
    manifest_path = final_output / "pipeline_report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if validate_only:
        return manifest

    started = time.perf_counter()
    subprocess.run(train_command, check=True, cwd=Path.cwd())
    reference_checkpoint = Path(resolved["reference_checkpoint"])
    if not reference_checkpoint.is_file():
        raise RuntimeError(f"h-reference stage did not create {reference_checkpoint}")
    subprocess.run(compress_command, check=True, cwd=Path.cwd())

    compact_path = final_output / "compact_octree.npz"
    compression_report_path = final_output / "compression_report.json"
    if not compact_path.is_file() or not compression_report_path.is_file():
        raise RuntimeError("functional compression stage did not produce its required outputs")
    compression_report = json.loads(compression_report_path.read_text(encoding="utf-8"))
    functional_leaf_counts = compression_report.get("functional_leaf_counts", {}) or {}
    p1_leaf_count = int(functional_leaf_counts.get("p1", 0))
    manifest.update(
        {
            # A zero-p1 result is valid R-D output, but it is not a functional
            # p1 field and must never be presented as one in reports/viewers.
            "status": "complete" if p1_leaf_count > 0 else "complete_without_p1",
            "elapsed_seconds": time.perf_counter() - started,
            "final_compact_file_bytes": compact_path.stat().st_size,
            "final_raw_packed_bytes": compression_report.get("raw_packed_bytes"),
            "final_leaf_count": compression_report.get("compressed_leaf_count"),
            "final_action_counts": compression_report.get("action_counts"),
            "final_functional_leaf_counts": functional_leaf_counts,
            "projection_evaluation": compression_report.get("evaluation"),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Industrial CT detail-preserving adaptive function field pipeline v5")
    parser.add_argument("--config", required=True, help="pipeline YAML; external checkpoints are not accepted")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run_pipeline(args.config, validate_only=args.validate_only), indent=2), flush=True)


if __name__ == "__main__":
    main()
