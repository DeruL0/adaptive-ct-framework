from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from adaptive_ct.data import load_r2_dataset
from adaptive_ct.functional_compression import load_reference_model
from adaptive_ct.nodal import score_private_leaf_actions, select_private_leaves_under_budget
from adaptive_ct.projection_domain import coefficient_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score projection-domain private-leaf actions for a tied nodal field."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--tied-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--batch-rays", type=int, default=4096)
    parser.add_argument("--raw-budget-bytes", type=int, required=True)
    parser.add_argument("--base-raw-bytes", type=int, required=True)
    parser.add_argument("--bytes-per-private-leaf", type=int, default=16)
    parser.add_argument(
        "--rank-mode",
        choices=("projection", "projection_jump"),
        default="projection",
        help="projection_jump ranks positive projection gains by gain times reference interface jump.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config, reference = load_reference_model(args.config, args.reference_checkpoint, device)
    balance = reference.balance_2to1()
    _, tied = load_reference_model(args.config, args.tied_checkpoint, device)
    dataset = load_r2_dataset(config["dataset"]["root"], device=device, load_volume=False)
    detector_h, detector_w = dataset.detector_shape
    samples_per_ray = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))

    torch.manual_seed(int(config.get("training", {}).get("seed", 0)))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.get("training", {}).get("seed", 0)))
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    scores, score_report = score_private_leaf_actions(
        reference,
        tied,
        dataset.train,
        batches=args.batches,
        batch_rays=args.batch_rays,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    added_budget = max(0, int(args.raw_budget_bytes) - int(args.base_raw_bytes))
    jump = torch.amax(coefficient_diagnostics(reference).interface_jump, dim=1)
    selection_scores = scores
    if args.rank_mode == "projection_jump":
        gain = (-scores).clamp_min(0.0)
        positive_jump = jump[jump > 0.0]
        jump_scale = (
            torch.median(positive_jump).clamp_min(1e-12)
            if positive_jump.numel()
            else torch.tensor(1.0, device=jump.device)
        )
        selection_scores = -(gain * (jump / jump_scale))
    selected = select_private_leaves_under_budget(
        selection_scores,
        added_byte_budget=added_budget,
        bytes_per_private_leaf=args.bytes_per_private_leaf,
    )
    level_counts = [
        int(torch.sum(selected & (tied.leaf_levels == level)).item())
        for level in range(len(tied.level_shapes))
    ]
    positive = (-scores).clamp_min(0.0)
    positive_values = positive[positive > 0.0]
    quantiles = {}
    if positive_values.numel():
        for name, value in (("q50", 0.5), ("q90", 0.9), ("q99", 0.99)):
            quantiles[name] = float(torch.quantile(positive_values, value).item())
    payload = {
        "scores": scores.cpu(),
        "selection_scores": selection_scores.cpu(),
        "interface_jump": jump.cpu(),
        "private_leaf_mask": selected.cpu(),
        "leaf_levels": tied.leaf_levels.cpu(),
        "leaf_coords": tied.leaf_coords.cpu(),
    }
    torch.save(payload, output / "private_actions.pt")
    report = {
        "schema": "nodal_private_leaf_scores_v1",
        "elapsed_sec": elapsed,
        "score_report": score_report.__dict__,
        "raw_budget_bytes": int(args.raw_budget_bytes),
        "base_raw_bytes": int(args.base_raw_bytes),
        "added_byte_budget": added_budget,
        "bytes_per_private_leaf": int(args.bytes_per_private_leaf),
        "rank_mode": args.rank_mode,
        "selected_leaf_count": int(selected.sum().item()),
        "selected_by_level": level_counts,
        "selected_interface_jump_mean": float(jump[selected].mean().item()) if torch.any(selected) else 0.0,
        "positive_gain_quantiles": quantiles,
        "balance": balance,
    }
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
