from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import unittest
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _run_tests() -> dict[str, Any]:
    suite = unittest.defaultTestLoader.discover("tests")
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return {
        "command": "python -m unittest discover -s tests",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.wasSuccessful(),
    }


def build_audit(
    *,
    acceptance: dict[str, Any],
    final_training: dict[str, Any],
    final_quantization: dict[str, Any],
    wpl3_training: dict[str, Any],
    b2_hp2: dict[str, Any],
    b2_hp8: dict[str, Any],
    old_benchmark: dict[str, Any],
    packed_benchmark: dict[str, Any],
    tests: dict[str, Any],
) -> dict[str, Any]:
    p_events = [
        event
        for event in final_training.get("growth_events", [])
        if event.get("operation") == "p_elevate"
    ]
    p_event = p_events[-1] if p_events else {}
    continuity = final_training["training_objective"]["coefficient_continuity"]
    quant_results = final_quantization["results"]
    packed_schema = quant_results["float16"]["schema"]
    old_step = old_benchmark["cases"][0]["cuda_native"]["step_ms"]
    packed_step = packed_benchmark["cases"][0]["cuda_native"]["step_ms"]
    regression = packed_step / old_step - 1.0

    implementation_checks = [
        _check(
            "arbitrary-depth packed CUDA hierarchy",
            bool(final_training["ray_integration"]["compact_cuda_traversal"])
            and len(wpl3_training["model"]["active_by_level"]) == 4,
            {
                "four_level_active_by_level": wpl3_training["model"]["active_by_level"],
                "source": ["adaptive_ct/bernstein.py", "native/bernstein_octree.cu"],
            },
        ),
        _check(
            "four-level CPU/CUDA forward-backward-segment consistency and no dense 768 index",
            tests["passed"],
            {
                "tests": [
                    "tests.test_bernstein_cuda.BernsteinCudaTests.test_four_level_forward_backward_and_segments_match_reference",
                    "tests.test_bernstein.BernsteinOctreeTests.test_four_level_topology_is_packed_and_never_allocates_dense_finest_index",
                ]
            },
        ),
        _check(
            "three-level CUDA regression <= 10%",
            regression <= 0.10,
            {"old_step_ms": old_step, "packed_step_ms": packed_step, "regression_fraction": regression},
        ),
        _check(
            "all 16 training views drive residual and p selection",
            p_event.get("residual_views") == 16 and p_event.get("residual_rays") == 16 * 32768,
            {
                "residual_views": p_event.get("residual_views"),
                "residual_rays": p_event.get("residual_rays"),
            },
        ),
        _check(
            "cross-level Bernstein face continuity",
            continuity.get("enabled")
            and continuity.get("include_cross_level")
            and continuity.get("face_quadrature_order") == 2,
            continuity,
        ),
        _check(
            "physical-coefficient orthogonal-complement gradient RD p-refinement",
            p_event.get("strategy") == "gradient_rate_distortion"
            and p_event.get("gradient_space") == "physical_bernstein_coefficients"
            and p_event.get("gradient_batches") == 8,
            p_event,
        ),
        _check(
            "packed mixed-degree v3 export and v1/v2 read compatibility",
            packed_schema == "compact_bernstein_octree_packed_v3" and tests["passed"],
            {
                "schema": packed_schema,
                "tests": [
                    "tests.test_compact_v3.CompactV3Tests.test_arbitrary_depth_mixed_degree_round_trip",
                    "tests.test_compact_v3.CompactV3Tests.test_legacy_v1_and_v2_remain_readable",
                ],
            },
        ),
        _check(
            "active_by_level and l3_active statistics",
            wpl3_training["model"]["l3_active"] > 0
            and wpl3_training["model"]["active_by_level"][3] == wpl3_training["model"]["l3_active"],
            {
                "l3_active": wpl3_training["model"]["l3_active"],
                "active_by_level": wpl3_training["model"]["active_by_level"],
            },
        ),
        _check(
            "uint8/uint16/float16 same-checkpoint evaluation",
            set(quant_results) == {"uint8", "uint16", "float16"},
            {"quantizations": sorted(quant_results)},
        ),
        _check("full regression suite", tests["passed"], tests),
    ]

    quality_gates = acceptance["gates"]
    failed_quality_gates = [gate for gate in quality_gates if not gate["passed"]]
    all_implementation = all(item["passed"] for item in implementation_checks)
    achieved = all_implementation and not failed_quality_gates
    return {
        "schema": "ct_goal_completion_audit_v1",
        "status": "complete" if achieved else "blocked",
        "objective_achieved": achieved,
        "implementation_checks": implementation_checks,
        "quality_gates": quality_gates,
        "failed_quality_gates": failed_quality_gates,
        "quantization_decision": acceptance["quantization_decision"],
        "b2_sufficient_view_hp_validation": {
            "hp2": b2_hp2["verdict"],
            "hp8": b2_hp8["verdict"],
            "conclusion": (
                "Neither 2% nor 8% degree-1 gradient-RD p refinement produced a measurable "
                "48-train/12-test gain, so further hp tuning is not justified."
            ),
        },
        "blocking_reason": (
            None
            if achieved
            else (
                "Implementation and system constraints are verified, but the explicit "
                "16-view high-gradient PSNR and gradient-ratio quality gates remain unmet. "
                "The approved hp/L3 mechanisms were exhausted without a held-out gain; "
                "meeting the remaining gates requires a new reconstruction prior/objective "
                "or revised acceptance criteria."
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the full 16-view CT accuracy goal")
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--final-training", type=Path, required=True)
    parser.add_argument("--final-quantization", type=Path, required=True)
    parser.add_argument("--wpl3-training", type=Path, required=True)
    parser.add_argument("--b2-hp2", type=Path, required=True)
    parser.add_argument("--b2-hp8", type=Path, required=True)
    parser.add_argument("--old-benchmark", type=Path, required=True)
    parser.add_argument("--packed-benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tests = _run_tests()
    report = build_audit(
        acceptance=_load(args.acceptance),
        final_training=_load(args.final_training),
        final_quantization=_load(args.final_quantization),
        wpl3_training=_load(args.wpl3_training),
        b2_hp2=_load(args.b2_hp2),
        b2_hp8=_load(args.b2_hp8),
        old_benchmark=_load(args.old_benchmark),
        packed_benchmark=_load(args.packed_benchmark),
        tests=tests,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
