from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from adaptive_ct.compression import export_compact_octree_artifact
from adaptive_ct.data import load_r2_dataset
from adaptive_ct.functional_compression import (
    _evaluate_projection_split,
    finetune_fixed_topology,
    load_reference_model,
)
from adaptive_ct.nodal import convert_to_conforming_nodal_p1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a fixed p5 octree to a shared-vertex Q1 field and run projection finetuning."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-rays", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--eval-ray-chunk", type=int, default=65536)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--already-nodal",
        action="store_true",
        help="Checkpoint already contains the constrained nodal map; skip conversion.",
    )
    parser.add_argument(
        "--private-actions",
        default="",
        help="private_actions.pt produced by score_nodal_private; enables projection-selected interface leaves.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def timed(name, function):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings[name] = time.perf_counter() - start
        print(json.dumps({"phase": name, "seconds": timings[name]}), flush=True)
        return result

    config, model = timed(
        "load_checkpoint",
        lambda: load_reference_model(args.config, args.checkpoint, device),
    )
    dataset = timed(
        "load_dataset",
        lambda: load_r2_dataset(config["dataset"]["root"], device=device, load_volume=False),
    )
    detector_h, detector_w = dataset.detector_shape
    samples_per_ray = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))

    evaluation = {}
    if not args.skip_eval:
        evaluation["before"] = timed(
            "evaluate_before",
            lambda: _evaluate_projection_split(
                model,
                dataset.test,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                ray_chunk=args.eval_ray_chunk,
            ),
        )

    if args.already_nodal:
        if not model.constrained_nodal:
            raise ValueError("--already-nodal was set but the checkpoint is not nodal.")
        conversion = None
    else:
        private_mask = None
        enforce_balance = True
        if args.private_actions:
            model.balance_2to1()
            action_payload = torch.load(args.private_actions, map_location=device, weights_only=False)
            if not torch.equal(model.leaf_levels.cpu(), action_payload["leaf_levels"].cpu()) or not torch.equal(
                model.leaf_coords.cpu(), action_payload["leaf_coords"].cpu()
            ):
                raise ValueError("Private-action topology does not match the balanced source checkpoint.")
            private_mask = action_payload["private_leaf_mask"].to(device=device, dtype=torch.bool)
            enforce_balance = False
        conversion = timed(
            "convert_nodal_p1",
            lambda: convert_to_conforming_nodal_p1(
                model,
                discontinuous_leaf_mask=private_mask,
                enforce_balance=enforce_balance,
            ),
        )
        evaluation["conversion"] = conversion.__dict__

    if not args.skip_eval:
        evaluation["after_conversion"] = timed(
            "evaluate_after_conversion",
            lambda: _evaluate_projection_split(
                model,
                dataset.test,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                ray_chunk=args.eval_ray_chunk,
            ),
        )

    finetune = None
    if args.iterations > 0:
        finetune = timed(
            "finetune",
            lambda: finetune_fixed_topology(
                model,
                dataset,
                iterations=args.iterations,
                batch_rays=args.batch_rays,
                lr=args.lr,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                log_every=max(1, min(20, args.iterations)),
            ),
        )
    if not args.skip_eval:
        evaluation["after_finetune"] = timed(
            "evaluate_after_finetune",
            lambda: _evaluate_projection_split(
                model,
                dataset.test,
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                ray_chunk=args.eval_ray_chunk,
            ),
        )

    checkpoint = output / "checkpoint.pt"
    torch.save({"config": config, "model": model.state_dict()}, checkpoint)
    compact = output / "compact_nodal_v4.npz"
    compact_summary = export_compact_octree_artifact(
        model,
        compact,
        quantization="float16",
        topology="packed_hierarchy",
        checkpoint_path=checkpoint,
    )
    report = {
        "schema": "conforming_nodal_p1_experiment_v1",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "timings": timings,
        "evaluation": evaluation,
        "finetune": finetune,
        "model_stats": model.stats().__dict__,
        "compact": compact_summary.__dict__,
    }
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
