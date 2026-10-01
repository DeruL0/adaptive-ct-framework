from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_ct.ambrosio_tortorelli import (
    NodalPhaseField,
    ambrosio_tortorelli_terms,
    phase_leaf_interface_scores,
    select_phase_interface_leaves,
    solve_phase_field_cg,
    volume_weighted_leaf_sample,
)
from adaptive_ct.compression import export_compact_octree_artifact
from adaptive_ct.data import load_r2_dataset
from adaptive_ct.functional_compression import (
    _evaluate_projection_split,
    finetune_fixed_topology,
    load_reference_model,
)
from adaptive_ct.geometry import random_training_rays
from adaptive_ct.metrics import mae, psnr
from adaptive_ct.nodal import (
    convert_to_conforming_nodal_p1,
    materialize_leaf_local_model,
    update_private_mask_with_swaps,
)


def slice_points(shape: tuple[int, int, int], axis: int, index: int, device: torch.device) -> torch.Tensor:
    free = [value for value in range(3) if value != axis]
    aa, bb = torch.meshgrid(
        torch.arange(shape[free[0]], device=device),
        torch.arange(shape[free[1]], device=device),
        indexing="ij",
    )
    coords = torch.empty((aa.numel(), 3), dtype=torch.float32, device=device)
    coords[:, axis] = float(index)
    coords[:, free[0]] = aa.reshape(-1)
    coords[:, free[1]] = bb.reshape(-1)
    scale = torch.tensor(shape, dtype=torch.float32, device=device)
    return -1.0 + (coords + 0.5) * 2.0 / scale


def image_gradient(image: torch.Tensor) -> torch.Tensor:
    gx = image[1:, :-1] - image[:-1, :-1]
    gy = image[:-1, 1:] - image[:-1, :-1]
    return torch.sqrt(gx.square() + gy.square() + 1.0e-20)


@torch.no_grad()
def evaluate_y_slice(model, target: torch.Tensor, *, index: int, chunk: int = 524288) -> dict[str, float]:
    shape = tuple(int(value) for value in target.shape)
    points = slice_points(shape, 1, int(index), model.coefficient_logits.device)
    values = []
    for start in range(0, points.shape[0], int(chunk)):
        values.append(model.forward_mu(points[start : start + int(chunk)]))
    prediction = torch.cat(values).reshape(shape[0], shape[2])
    reference = target[:, int(index), :].to(device=prediction.device)
    target_grad = image_gradient(reference)
    pred_grad = image_gradient(prediction)
    cutoff = torch.quantile(target_grad, 0.99)
    high = target_grad >= cutoff
    grad_ratio = pred_grad[high].mean() / target_grad[high].mean().clamp_min(1.0e-12)

    material = reference > 0.1
    near_material = F.max_pool2d(
        material[None, None].to(dtype=torch.float32),
        kernel_size=31,
        stride=1,
        padding=15,
    )[0, 0] > 0.0
    near_air = near_material & (~material) & (reference < 0.02)
    halo_mae = torch.mean(torch.abs(prediction[near_air] - reference[near_air])) if torch.any(near_air) else prediction.new_zeros(())
    return {
        "psnr": psnr(prediction, reference),
        "mae": mae(prediction, reference),
        "high_gradient_psnr": psnr(prediction[:-1, :-1][high], reference[:-1, :-1][high]),
        "high_gradient_mae": mae(prediction[:-1, :-1][high], reference[:-1, :-1][high]),
        "gradient_ratio": float(grad_ratio.item()),
        "gradient_ratio_distance": float(torch.abs(grad_ratio - 1.0).item()),
        "near_air_halo_mae": float(halo_mae.item()),
    }


def load_volume_only(dataset_root: str | Path) -> torch.Tensor:
    root = Path(dataset_root)
    with (root / "meta_data.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    volume = np.load(root / metadata["vol"], mmap_mode="r")
    return torch.from_numpy(np.asarray(volume, dtype=np.float32).copy())


def terms_row(terms) -> dict[str, float]:
    return {
        "at_total": float(terms.total.detach().item()),
        "at_interior": float(terms.interior.detach().item()),
        "at_phase_gradient": float(terms.phase_gradient.detach().item()),
        "at_phase_well": float(terms.phase_well.detach().item()),
        "phase_mean": float(terms.phase_mean.detach().item()),
        "edge_fraction": float(terms.edge_fraction.detach().item()),
        "underresolved_fraction": float(terms.underresolved_fraction.detach().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-topology Ambrosio--Tortorelli nodal CT proof of concept.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=1.0e-8)
    parser.add_argument("--beta", type=float, default=1.0e-7)
    parser.add_argument("--epsilon-multipliers", default="8,4")
    parser.add_argument("--outer-iterations", type=int, default=4)
    parser.add_argument("--mu-steps", type=int, default=20)
    parser.add_argument("--v-steps", type=int, default=10)
    parser.add_argument("--ray-batch", type=int, default=8192)
    parser.add_argument("--leaf-batch", type=int, default=65536)
    parser.add_argument("--mu-lr", type=float, default=2.0e-4)
    parser.add_argument("--v-lr", type=float, default=2.0e-2)
    parser.add_argument("--phase-solver", choices=("cg", "adam"), default="cg")
    parser.add_argument("--phase-cg-iterations", type=int, default=24)
    parser.add_argument("--phase-cg-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--phase-leaf-chunk", type=int, default=65536)
    parser.add_argument("--phase-threshold", type=float, default=0.5)
    parser.add_argument(
        "--topology-mode",
        choices=("frozen", "phase-swaps"),
        default="frozen",
        help="Freeze the existing private topology for the controlled AT test, or enable bounded phase-driven swaps.",
    )
    parser.add_argument(
        "--max-private-leaves",
        type=int,
        default=-1,
        help="Global deployment-rate cap. Negative reuses the source model's private-leaf count.",
    )
    parser.add_argument(
        "--max-private-swaps",
        type=int,
        default=-1,
        help="Structural trust-region size. Negative permits at most 10%% of the private-leaf budget per run.",
    )
    parser.add_argument("--post-untie-iterations", type=int, default=120)
    parser.add_argument("--post-untie-lr", type=float, default=5.0e-4)
    parser.add_argument("--residual-stiffness", type=float, default=1.0e-6)
    parser.add_argument("--slice-index", type=int, default=512)
    parser.add_argument("--eval-ray-chunk", type=int, default=65536)
    args = parser.parse_args()

    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    multipliers = tuple(float(value.strip()) for value in args.epsilon_multipliers.split(",") if value.strip())
    if not multipliers or any(value <= 0.0 for value in multipliers):
        raise ValueError("epsilon multipliers must be positive")

    config, source = load_reference_model(args.config, args.checkpoint, device)
    volume = load_volume_only(config["dataset"]["root"])
    source_slice = evaluate_y_slice(source, volume, index=args.slice_index)
    source_private_count = (
        int(source.nodal_private_leaf_mask.sum().item())
        if source.constrained_nodal and source.nodal_private_leaf_mask.numel()
        else 0
    )
    source_private_mask = (
        source.nodal_private_leaf_mask.detach().clone()
        if source.constrained_nodal and source.nodal_private_leaf_mask.numel()
        else torch.zeros(source.leaf_levels.shape[0], dtype=torch.bool, device=device)
    )
    reference = None
    if args.topology_mode == "frozen":
        if not source.constrained_nodal:
            raise ValueError("--topology-mode=frozen requires an existing constrained nodal checkpoint")
        model = source
    else:
        reference = materialize_leaf_local_model(source)
        local = materialize_leaf_local_model(reference)
        del source
        if device.type == "cuda":
            torch.cuda.empty_cache()
        convert_to_conforming_nodal_p1(local, enforce_balance=False)
        model = local
    tied_slice = evaluate_y_slice(model, volume, index=args.slice_index)
    dataset = load_r2_dataset(config["dataset"]["root"], device=device, load_volume=False)
    detector_h, detector_w = dataset.detector_shape
    samples_per_ray = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))
    before_projection = _evaluate_projection_split(
        model,
        dataset.test,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=int(args.eval_ray_chunk),
    )

    phase = NodalPhaseField(model, initial_value=1.0).to(device=device)
    mu_optimizer = torch.optim.Adam(model.parameters(), lr=float(args.mu_lr))
    v_optimizer = torch.optim.Adam(phase.parameters(), lr=float(args.v_lr)) if args.phase_solver == "adam" else None
    finest_sizes = 2.0 / torch.tensor(model.level_shapes[-1], dtype=torch.float32)
    finest_h = float(torch.amax(finest_sizes).item())
    epsilons = tuple(multiplier * finest_h for multiplier in multipliers)
    materialize_points = not model.prefer_compact_ray_batch()
    seed = int(config.get("training", {}).get("seed", 0))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    history: list[dict] = []
    started = time.perf_counter()

    for epsilon_id, epsilon in enumerate(epsilons):
        for outer in range(int(args.outer_iterations)):
            phase.values.requires_grad_(False)
            model.coefficient_logits.requires_grad_(True)
            last_data_loss = 0.0
            last_mu_terms = None
            for _ in range(int(args.mu_steps)):
                rays = random_training_rays(
                    dataset.train,
                    batch_rays=int(args.ray_batch),
                    detector_h=detector_h,
                    detector_w=detector_w,
                    samples_per_ray=samples_per_ray,
                    materialize_points=materialize_points,
                )
                prediction = model.integrate_ray_batch(rays)
                data_loss = 0.5 * torch.mean(torch.square(prediction - rays.target))
                leaves = volume_weighted_leaf_sample(model, int(args.leaf_batch))
                terms = ambrosio_tortorelli_terms(
                    model,
                    phase,
                    leaves,
                    alpha=args.alpha,
                    beta=args.beta,
                    epsilon=epsilon,
                    residual_stiffness=args.residual_stiffness,
                )
                loss = data_loss + terms.total
                mu_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                mu_optimizer.step()
                last_data_loss = float(data_loss.detach().item())
                last_mu_terms = terms

            model.coefficient_logits.requires_grad_(False)
            phase.values.requires_grad_(True)
            phase_solve = None
            if args.phase_solver == "cg":
                phase_solve = solve_phase_field_cg(
                    model,
                    phase,
                    alpha=args.alpha,
                    beta=args.beta,
                    epsilon=epsilon,
                    max_iterations=int(args.phase_cg_iterations),
                    tolerance=float(args.phase_cg_tolerance),
                    leaf_chunk=int(args.phase_leaf_chunk),
                )
                leaves = volume_weighted_leaf_sample(model, int(args.leaf_batch))
                last_v_terms = ambrosio_tortorelli_terms(
                    model,
                    phase,
                    leaves,
                    alpha=args.alpha,
                    beta=args.beta,
                    epsilon=epsilon,
                    residual_stiffness=args.residual_stiffness,
                )
            else:
                last_v_terms = None
                for _ in range(int(args.v_steps)):
                    leaves = volume_weighted_leaf_sample(model, int(args.leaf_batch))
                    terms = ambrosio_tortorelli_terms(
                        model,
                        phase,
                        leaves,
                        alpha=args.alpha,
                        beta=args.beta,
                        epsilon=epsilon,
                        residual_stiffness=args.residual_stiffness,
                    )
                    v_optimizer.zero_grad(set_to_none=True)
                    terms.total.backward()
                    v_optimizer.step()
                    phase.project_()
                    last_v_terms = terms

            row = {
                "epsilon_stage": epsilon_id,
                "epsilon": epsilon,
                "epsilon_over_finest_h": epsilon / finest_h,
                "outer_iteration": outer,
                "projection_loss": last_data_loss,
                "mu_step": terms_row(last_mu_terms),
                "v_step": terms_row(last_v_terms),
                "phase_solve": phase_solve.__dict__ if phase_solve is not None else None,
                "elapsed_sec": time.perf_counter() - started,
            }
            history.append(row)
            print(json.dumps({"phase": "at_alternating", **row}), flush=True)

    model.coefficient_logits.requires_grad_(True)
    phase.values.requires_grad_(True)
    tied_after_projection = _evaluate_projection_split(
        model,
        dataset.test,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=int(args.eval_ray_chunk),
    )
    tied_after_slice = evaluate_y_slice(model, volume, index=args.slice_index)
    phase_values = phase.master_values().detach()
    phase_quantile_levels = torch.tensor(
        [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 1.0],
        dtype=phase_values.dtype,
        device=phase_values.device,
    )
    phase_summary = {
        "count": int(phase_values.numel()),
        "mean": float(phase_values.mean().item()),
        "min": float(phase_values.min().item()),
        "quantile_levels": phase_quantile_levels.tolist(),
        "quantiles": torch.quantile(phase_values, phase_quantile_levels).tolist(),
        "below_0_99": int(torch.sum(phase_values < 0.99).item()),
        "below_0_9": int(torch.sum(phase_values < 0.9).item()),
        "below_0_5": int(torch.sum(phase_values < 0.5).item()),
    }

    interface_scores = phase_leaf_interface_scores(
        model,
        phase,
        leaf_chunk=int(args.phase_leaf_chunk),
    )
    max_private_leaves = int(args.max_private_leaves)
    if max_private_leaves < 0:
        max_private_leaves = source_private_count if source_private_count > 0 else None
    phase_target_mask = select_phase_interface_leaves(
        interface_scores,
        phase_threshold=float(args.phase_threshold),
        max_private_leaves=max_private_leaves,
    )
    max_private_swaps = int(args.max_private_swaps)
    if max_private_swaps < 0:
        max_private_swaps = max(1, int(max_private_leaves or 0) // 10)
    if args.topology_mode == "frozen":
        private_mask = source_private_mask
    elif source_private_count > 0 and max_private_leaves is not None:
        cutoff = 1.0 - float(args.phase_threshold)
        phase_action_scores = torch.full_like(interface_scores, float("inf"))
        eligible = interface_scores > cutoff
        phase_action_scores[eligible] = -interface_scores[eligible]
        private_mask = update_private_mask_with_swaps(
            source_private_mask,
            phase_action_scores,
            private_leaf_budget=int(max_private_leaves),
            max_swaps=max_private_swaps,
        )
    else:
        private_mask = phase_target_mask

    if args.topology_mode == "frozen":
        candidate = model
        conversion = None
    else:
        # Start smooth regions from the AT-optimised tied field, but restore
        # one-sided Q1 functions on phase-selected interface leaves.
        candidate = materialize_leaf_local_model(model)
        candidate_values = candidate.coefficients().detach().clone()
        reference_values = reference.coefficients().detach()
        private_coefficients = private_mask[candidate.coefficient_leaf_ids]
        candidate_values[private_coefficients] = reference_values[private_coefficients]
        with torch.no_grad():
            candidate.coefficient_logits.copy_(candidate._coefficients_to_logits(candidate_values))
        conversion = convert_to_conforming_nodal_p1(
            candidate,
            discontinuous_leaf_mask=private_mask,
            enforce_balance=False,
        )
        del reference
        if device.type == "cuda":
            torch.cuda.empty_cache()
    untied_initial_slice = evaluate_y_slice(candidate, volume, index=args.slice_index)
    untied_initial_projection = _evaluate_projection_split(
        candidate,
        dataset.test,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=int(args.eval_ray_chunk),
    )
    finetune = None
    if int(args.post_untie_iterations) > 0:
        finetune = finetune_fixed_topology(
            candidate,
            dataset,
            iterations=int(args.post_untie_iterations),
            batch_rays=int(args.ray_batch),
            lr=float(args.post_untie_lr),
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            log_every=max(1, min(20, int(args.post_untie_iterations))),
        )
    model = candidate
    after_projection = _evaluate_projection_split(
        model,
        dataset.test,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=int(args.eval_ray_chunk),
    )
    after_slice = evaluate_y_slice(model, volume, index=args.slice_index)
    checkpoint = output / "checkpoint.pt"
    torch.save(
        {
            "config": config,
            "model": model.state_dict(),
            "phase_field": phase.state_dict(),
            "at": {
                "alpha": float(args.alpha),
                "beta": float(args.beta),
                "epsilons": epsilons,
                "residual_stiffness": float(args.residual_stiffness),
                "phase_threshold": float(args.phase_threshold),
            },
        },
        checkpoint,
    )
    torch.save(
        {
            "master_values": phase.master_values().detach().cpu(),
            "epsilons": epsilons,
        },
        output / "phase_field.pt",
    )
    compact = export_compact_octree_artifact(
        model,
        output / "compact_nodal_v4.npz",
        quantization="float16",
        topology="packed_hierarchy",
        checkpoint_path=checkpoint,
    )
    report = {
        "schema": "ambrosio_tortorelli_nodal_ct_stage_b_v2",
        "config": str(Path(args.config).resolve()),
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "source_slice": source_slice,
        "fully_tied_initial_slice": tied_slice,
        "tied_after_slice": tied_after_slice,
        "untied_initial_slice": untied_initial_slice,
        "after_slice": after_slice,
        "projection_before": before_projection,
        "tied_projection_after": tied_after_projection,
        "untied_initial_projection": untied_initial_projection,
        "projection_after": after_projection,
        "phase_summary": phase_summary,
        "interface_selection": {
            "phase_threshold": float(args.phase_threshold),
            "source_private_leaves": source_private_count,
            "max_private_leaves": max_private_leaves,
            "max_private_swaps": max_private_swaps,
            "phase_target_leaves": int(phase_target_mask.sum().item()),
            "selected_private_leaves": int(private_mask.sum().item()),
            "retained_source_private_leaves": int((private_mask & source_private_mask).sum().item()),
            "added_private_leaves": int((private_mask & ~source_private_mask).sum().item()),
            "removed_private_leaves": int((source_private_mask & ~private_mask).sum().item()),
            "score_min": float(interface_scores.min().item()),
            "score_mean": float(interface_scores.mean().item()),
            "score_max": float(interface_scores.max().item()),
        },
        "topology_mode": args.topology_mode,
        "conversion": conversion.__dict__ if conversion is not None else None,
        "post_untie_finetune": finetune,
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "finest_h": finest_h,
        "epsilons": epsilons,
        "history": history,
        "compact": compact.__dict__,
        "elapsed_sec": time.perf_counter() - started,
    }
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
