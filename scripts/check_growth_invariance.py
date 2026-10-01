from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from adaptive_ct.ambrosio_tortorelli import (
    NodalPhaseField,
    phase_leaf_face_scores,
    phase_leaf_interface_scores,
)
from adaptive_ct.at_adaptive import (
    apply_phase_topology_actions,
    model_private_face_mask,
)
from adaptive_ct.data import load_r2_dataset
from adaptive_ct.decision_rays import FixedDecisionRaySet, paired_fixed_rss
from scripts.run_at_adaptive import load_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that a pure split/untie growth batch preserves projection RSS."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase-field", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-splits", type=int, default=64)
    parser.add_argument("--max-unties", type=int, default=128)
    parser.add_argument("--split-threshold", type=float, default=0.5)
    parser.add_argument("--face-threshold", type=float, default=0.5)
    parser.add_argument("--isolation-rings", type=int, default=2)
    parser.add_argument("--max-leaf-count", type=int, default=7_500_000)
    parser.add_argument("--ray-chunk", type=int, default=65536)
    parser.add_argument("--numerical-atol", type=float, default=1.0e-8)
    parser.add_argument("--numerical-rtol", type=float, default=1.0e-7)
    args = parser.parse_args()

    started = time.perf_counter()
    device = torch.device(args.device)
    config, baseline = load_model(args.config, args.checkpoint, device)
    if not baseline.constrained_nodal:
        raise ValueError("growth invariance requires a constrained nodal model")
    dataset = load_r2_dataset(config["dataset"]["root"], device=device, load_volume=False)
    detector_h, detector_w = dataset.detector_shape
    samples_per_ray = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))

    payload = torch.load(args.phase_field, map_location=device, weights_only=True)
    phase_values = payload.get("master_values")
    if not isinstance(phase_values, torch.Tensor):
        raise ValueError("phase field does not contain master_values")
    if phase_values.shape != baseline.coefficient_logits.shape:
        raise ValueError("phase field does not match the baseline nodal graph")
    phase = NodalPhaseField(baseline)
    with torch.no_grad():
        phase.values.copy_(phase_values.to(device=device, dtype=phase.values.dtype))
    leaf_scores = phase_leaf_interface_scores(baseline, phase)
    measured_face_scores = phase_leaf_face_scores(baseline, phase)
    old_faces = model_private_face_mask(baseline)
    # Pure growth: existing private faces are pinned on. Only false->true face
    # changes are eligible; merge and retie are disabled.
    growth_face_scores = torch.where(
        old_faces,
        torch.ones_like(measured_face_scores),
        measured_face_scores,
    )
    candidate, topology = apply_phase_topology_actions(
        baseline,
        leaf_scores,
        growth_face_scores,
        split_threshold=float(args.split_threshold),
        face_threshold=float(args.face_threshold),
        max_leaf_count=int(args.max_leaf_count),
        max_splits=int(args.max_splits),
        max_merges=0,
        max_face_changes=int(args.max_unties),
        isolation_rings=int(args.isolation_rings),
    )
    decision = FixedDecisionRaySet.full(
        dataset.train,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        chunk_rays=int(args.ray_chunk),
        materialize_points=not (
            baseline.prefer_compact_ray_batch()
            and candidate.prefer_compact_ray_batch()
        ),
    )
    paired = paired_fixed_rss(
        baseline,
        candidate,
        decision,
        numerical_atol=float(args.numerical_atol),
        numerical_rtol=float(args.numerical_rtol),
    )
    report = {
        "schema": "fixed_full_ray_growth_invariance_v1",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "phase_field": str(Path(args.phase_field).resolve()),
        "topology": topology.__dict__,
        "paired_full_train_rss": paired,
        "passed": bool(paired["nonincreasing_within_numerical_tolerance"]),
        "elapsed_sec": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
