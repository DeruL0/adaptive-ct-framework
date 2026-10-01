from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "reference_hp_l3_continuity_scheduled": (
        "output/research_figure473_128v_reference/training_report.json",
        "output/research_audit/science_reference_256/original_checkpoint_diagnostics.json",
    ),
    "h_only_l3_continuity_scheduled": (
        "output/research_figure473_128v_h_only/training_report.json",
        "output/research_audit/science_h_only_256/original_checkpoint_diagnostics.json",
    ),
    "hp_no_l3_continuity_scheduled": (
        "output/research_figure473_128v_no_l3/training_report.json",
        "output/research_audit/science_no_l3_256/original_checkpoint_diagnostics.json",
    ),
    "h_only_no_l3_continuity_scheduled": (
        "output/research_figure473_128v_h_only_no_l3/training_report.json",
        "output/research_audit/science_h_only_no_l3_256/original_checkpoint_diagnostics.json",
    ),
    "h_only_no_l3_no_continuity_scheduled": (
        "output/research_figure473_128v_no_continuity/training_report.json",
        "output/research_audit/science_no_continuity_256/original_checkpoint_diagnostics.json",
    ),
    "h_only_no_l3_continuity_uniform": (
        "output/research_figure473_128v_uniform_fidelity/training_report.json",
        "output/research_audit/science_uniform_256/original_checkpoint_diagnostics.json",
    ),
    "simple_static_balanced_15k": (
        "output/research_figure473_128v_static_balanced/training_report.json",
        "output/research_audit/science_static_balanced_256/original_checkpoint_diagnostics.json",
    ),
    "simple_static_balanced_25k": (
        "output/research_figure473_128v_simple_25k/training_report.json",
        "output/research_audit/science_simple_25k_256/original_checkpoint_diagnostics.json",
    ),
    "anisotropic_60x128x128_25k": (
        "output/research_figure473_128v_simple_25k_anisotropic/training_report.json",
        "output/research_audit/science_simple_25k_anisotropic_256/original_checkpoint_diagnostics.json",
    ),
    "greedy_unified_hp_48root_25k": (
        "output/research_figure473_48root_128v_unified_hp_25k/training_report.json",
        "output/research_audit/science_48root_unified_hp_25k_256/original_checkpoint_diagnostics.json",
    ),
    "haar_gn_h_only_48root_25k": (
        "output/research_figure473_48root_128v_h_gn_25k/training_report.json",
        "output/research_audit/science_48root_h_gn_25k_256/original_checkpoint_diagnostics.json",
    ),
    "haar_gn_h_only_96root_25k": (
        "output/research_figure473_96root_128v_h_gn_25k/training_report.json",
        "output/research_audit/science_96root_h_gn_25k_256/original_checkpoint_diagnostics.json",
    ),
    "support_floor_budget_25k_rejected_trick": (
        "output/research_figure473_128v_support_budget_25k/training_report.json",
        "output/research_audit/science_support_budget_25k_256/original_checkpoint_diagnostics.json",
    ),
    "pure_global_jump_48root_seed20260702_25k": (
        "output/research_figure473_pure_global_jump_48root_128v_25k/training_report.json",
        "output/research_audit/science_pure_global_jump_48root_25k_256/original_checkpoint_diagnostics.json",
    ),
    "pure_global_jump_48root_seed20260703_25k": (
        "output/research_figure473_pure_global_jump_48root_128v_seed20260703_25k/training_report.json",
        "output/research_audit/science_pure_global_jump_seed20260703_256/original_checkpoint_diagnostics.json",
    ),
    "canonical_pure_global_jump_uniform_48root_seed20260702_25k": (
        "output/research_figure473_pure_global_jump_uniform_48root_128v_25k/training_report.json",
        "output/research_audit/science_pure_global_jump_uniform_48root_25k_256/original_checkpoint_diagnostics.json",
    ),
    "canonical_pure_global_jump_uniform_48root_seed20260703_25k": (
        "output/research_figure473_pure_global_jump_uniform_48root_128v_seed20260703_25k/training_report.json",
        "output/research_audit/science_pure_global_jump_uniform_seed20260703_256/original_checkpoint_diagnostics.json",
    ),
}


def main() -> None:
    rows = []
    for name, (training_path, volume_path) in VARIANTS.items():
        training = json.loads((ROOT / training_path).read_text(encoding="utf-8"))
        volume = json.loads((ROOT / volume_path).read_text(encoding="utf-8"))
        volume_metrics = volume["resampled_volume_metrics"]
        rows.append(
            {
                "variant": name,
                "test_projection_psnr": training["projection_test"]["psnr"],
                "test_projection_mae": training["projection_test"]["mae"],
                "validation_projection_psnr": training["evaluation_history"][-1]["psnr"],
                "volume_256_psnr": volume_metrics["full"]["psnr"],
                "volume_256_mae": volume_metrics["full"]["mae"],
                "material_256_psnr": volume_metrics["material"]["psnr"],
                "material_256_mae": volume_metrics["material"]["mae"],
                "leaves": training["model"]["leaf_cells"],
                "coefficients": training["model"]["coefficient_count"],
                "compact_bytes": training["exports"]["compact_octree"]["file_bytes"],
                "elapsed_seconds": training["total_elapsed_sec"],
                "peak_reserved_bytes": training["peak_reserved_bytes"],
                "test_used_during_training": training["evaluation_protocol"]["test_used_during_training"],
                "seed": training.get("training_seed"),
            }
        )
    fandisk_training = json.loads(
        (ROOT / "output/research_fandisk_pure_global_jump_48v_500k/training_report.json").read_text(
            encoding="utf-8"
        )
    )
    legacy_fandisk = json.loads(
        (ROOT / "output/fandisk_48v/training_report.json").read_text(encoding="utf-8")
    )
    output = {
        "schema": "ct_research_audit_ablation_v2",
        "protocol": {
            "train_views": 128,
            "validation_views": 16,
            "lockbox_test_views": 16,
            "detector": [1024, 1024],
            "volume_protocol": "model and GT sampled at 256^3 voxel centres",
            "seed": 20260702,
        },
        "variants": rows,
        "second_object_fandisk": {
            "protocol_note": "48 train / 12 test views at 96^3; old dynamic baseline has a different objective and is contextual, not a controlled ablation.",
            "pure_global_jump": {
                "projection_test": fandisk_training["projection_test"],
                "material_volume": fandisk_training["material_volume"],
                "boundary_sharpness": fandisk_training["boundary_sharpness"],
                "parameters": fandisk_training["model"]["parameter_count"],
                "compact_bytes": fandisk_training["exports"]["compact_octree"]["file_bytes"],
            },
            "legacy_dynamic_context": {
                "projection_test": legacy_fandisk["projection_test"],
                "material_volume": legacy_fandisk["material_volume"],
                "boundary_sharpness": legacy_fandisk["boundary_sharpness"],
                "parameters": legacy_fandisk["model"]["parameter_count"],
            },
        },
    }
    output_path = ROOT / "output/research_audit/ablation_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
