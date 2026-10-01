from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from adaptive_ct.compression import export_compact_octree_artifact
from adaptive_ct.data import ProjectionSplit, load_r2_dataset
from adaptive_ct.functional_compression import (
    _evaluate_projection_split,
    finetune_fixed_topology,
    load_reference_model,
)
from adaptive_ct.geometry import random_training_rays
from adaptive_ct.nodal import (
    convert_to_conforming_nodal_p1,
    materialize_leaf_local_model,
    score_private_leaf_actions,
    update_private_mask_with_swaps,
)


def downsample_projection_split(split: ProjectionSplit, factor: int) -> ProjectionSplit:
    """Average detector bins while preserving the measured angular grid."""

    factor = int(factor)
    if factor < 1:
        raise ValueError("projection scale factors must be positive")
    if factor == 1:
        return split
    height, width = (int(value) for value in split.projections.shape[-2:])
    if height % factor or width % factor:
        raise ValueError(
            f"detector shape {(height, width)} is not divisible by scale factor {factor}"
        )
    projections = F.avg_pool2d(
        split.projections[:, None],
        kernel_size=factor,
        stride=factor,
    )[:, 0]
    return ProjectionSplit(
        angles=split.angles,
        projections=projections.contiguous(),
        paths=split.paths,
    )


@torch.no_grad()
def ray_batch_loss(model, ray_batch) -> float:
    prediction = model.integrate_ray_batch(ray_batch)
    return float((0.5 * torch.mean(torch.square(prediction - ray_batch.target))).item())


def save_checkpoint(path: Path, config: dict, model) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config": config, "model": model.state_dict()}, temporary)
    temporary.replace(path)


def release_cuda(*objects) -> None:
    for value in objects:
        del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconsider nodal private leaves in a reversible coarse-to-fine loop. "
            "The octree, loss, anisotropic levels, and compact schema remain unchanged."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scales", default="4,2,1")
    parser.add_argument("--rounds-per-scale", type=int, default=1)
    parser.add_argument("--score-rays-per-view", type=int, default=32768)
    parser.add_argument("--score-batch-rays", type=int, default=4096)
    parser.add_argument("--max-swaps-per-round", type=int, default=8192)
    parser.add_argument("--raw-budget-bytes", type=int, default=30_000_000)
    parser.add_argument("--base-raw-bytes", type=int, default=27_174_983)
    parser.add_argument("--bytes-per-private-leaf", type=int, default=16)
    parser.add_argument("--finetune-iterations", type=int, default=100)
    parser.add_argument("--finetune-batch-rays", type=int, default=8192)
    parser.add_argument("--finetune-lr", type=float, default=5.0e-4)
    parser.add_argument("--accept-rays", type=int, default=65536)
    parser.add_argument("--eval-ray-chunk", type=int, default=65536)
    args = parser.parse_args()

    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scales = tuple(int(value.strip()) for value in args.scales.split(",") if value.strip())
    if not scales or any(value < 1 for value in scales):
        raise ValueError("--scales must contain positive comma-separated integers")
    if tuple(sorted(scales, reverse=True)) != scales:
        raise ValueError("--scales must be ordered coarse-to-fine, for example 4,2,1")

    config, initial = load_reference_model(args.config, args.checkpoint, device)
    if not initial.constrained_nodal:
        convert_to_conforming_nodal_p1(initial, enforce_balance=True)
    seed = int(config.get("training", {}).get("seed", 0))
    dataset = load_r2_dataset(config["dataset"]["root"], device=device, load_volume=False)
    full_h, full_w = dataset.detector_shape
    samples_per_ray = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))
    accepted_checkpoint = output / "checkpoint.pt"
    save_checkpoint(accepted_checkpoint, config, initial)
    del initial
    release_cuda()

    added_budget = max(0, int(args.raw_budget_bytes) - int(args.base_raw_bytes))
    private_leaf_budget = added_budget // int(args.bytes_per_private_leaf)
    history: list[dict] = []
    round_id = 0
    started = time.perf_counter()

    for scale in scales:
        stage_train = downsample_projection_split(dataset.train, scale)
        stage_h, stage_w = (int(value) for value in stage_train.projections.shape[-2:])
        stage_dataset = replace(dataset, train=stage_train)
        total_score_rays = int(stage_train.projections.shape[0]) * int(args.score_rays_per_view)
        score_batches = max(1, math.ceil(total_score_rays / int(args.score_batch_rays)))

        for local_round in range(int(args.rounds_per_scale)):
            round_id += 1
            round_start = time.perf_counter()
            config, current = load_reference_model(args.config, accepted_checkpoint, device)
            old_mask = current.nodal_private_leaf_mask.detach().clone()

            torch.manual_seed(seed + 10_000)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + 10_000)
            accept_batch = random_training_rays(
                dataset.train,
                batch_rays=int(args.accept_rays),
                detector_h=full_h,
                detector_w=full_w,
                samples_per_ray=samples_per_ray,
                materialize_points=not current.prefer_compact_ray_batch(),
            )
            baseline_loss = ray_batch_loss(current, accept_batch)

            reference = materialize_leaf_local_model(current)
            del current
            release_cuda()
            tied = materialize_leaf_local_model(reference)
            convert_to_conforming_nodal_p1(tied, enforce_balance=False)

            torch.manual_seed(seed + round_id)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + round_id)
            scores, score_report = score_private_leaf_actions(
                reference,
                tied,
                stage_train,
                batches=score_batches,
                batch_rays=int(args.score_batch_rays),
                detector_h=stage_h,
                detector_w=stage_w,
                samples_per_ray=samples_per_ray,
            )
            new_mask = update_private_mask_with_swaps(
                old_mask,
                scores,
                private_leaf_budget=private_leaf_budget,
                max_swaps=int(args.max_swaps_per_round),
            )
            added = int(torch.sum(new_mask & ~old_mask).item())
            removed = int(torch.sum(old_mask & ~new_mask).item())
            del tied, scores, old_mask
            release_cuda()

            candidate = materialize_leaf_local_model(reference)
            convert_to_conforming_nodal_p1(
                candidate,
                discontinuous_leaf_mask=new_mask,
                enforce_balance=False,
            )
            del reference, new_mask
            release_cuda()

            finetune = finetune_fixed_topology(
                candidate,
                stage_dataset,
                iterations=int(args.finetune_iterations),
                batch_rays=int(args.finetune_batch_rays),
                lr=float(args.finetune_lr),
                detector_h=stage_h,
                detector_w=stage_w,
                samples_per_ray=samples_per_ray,
                log_every=max(1, int(args.finetune_iterations) // 5),
            )
            candidate_loss = ray_batch_loss(candidate, accept_batch)
            accepted = candidate_loss < baseline_loss
            if accepted:
                save_checkpoint(accepted_checkpoint, config, candidate)

            row = {
                "round": round_id,
                "scale": scale,
                "scale_round": local_round,
                "detector_shape": [stage_h, stage_w],
                "score_report": score_report.__dict__,
                "private_before": int(candidate.nodal_private_leaf_mask.sum().item()) - added + removed,
                "private_after": int(candidate.nodal_private_leaf_mask.sum().item()),
                "added": added,
                "retied": removed,
                "baseline_full_resolution_loss": baseline_loss,
                "candidate_full_resolution_loss": candidate_loss,
                "accepted": accepted,
                "finetune": finetune,
                "seconds": time.perf_counter() - round_start,
            }
            history.append(row)
            print(json.dumps({"phase": "structure_round", **row}), flush=True)
            del candidate, accept_batch
            release_cuda()

    config, final_model = load_reference_model(args.config, accepted_checkpoint, device)
    final_test = _evaluate_projection_split(
        final_model,
        dataset.test,
        detector_h=full_h,
        detector_w=full_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=int(args.eval_ray_chunk),
    )
    compact = export_compact_octree_artifact(
        final_model,
        output / "compact_nodal_v4.npz",
        quantization="float16",
        topology="packed_hierarchy",
        checkpoint_path=accepted_checkpoint,
    )
    report = {
        "schema": "reversible_nodal_coarse_to_fine_v1",
        "config": str(Path(args.config).resolve()),
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "scales": list(scales),
        "rounds_per_scale": int(args.rounds_per_scale),
        "private_leaf_budget": int(private_leaf_budget),
        "max_swaps_per_round": int(args.max_swaps_per_round),
        "history": history,
        "final_test": final_test,
        "model_stats": final_model.stats().__dict__,
        "compact": compact.__dict__,
        "total_seconds": time.perf_counter() - started,
    }
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
