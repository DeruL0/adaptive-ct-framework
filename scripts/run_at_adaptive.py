from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time

import torch

from adaptive_ct.ambrosio_tortorelli import (
    NodalPhaseField,
    ambrosio_tortorelli_terms,
    phase_leaf_face_scores,
    phase_leaf_interface_scores,
    phase_weighted_face_jump_terms,
    solve_phase_field_cg,
    volume_weighted_leaf_sample,
)
from adaptive_ct.at_adaptive import (
    affected_master_mask,
    alpha_for_epsilon,
    apply_phase_topology_actions,
    attenuation_gradient_tail_statistics,
    calibrate_at_weights,
    estimate_air_noise_variance,
    estimate_projection_rss,
    evaluate_direct_decision_energy,
    evaluate_total_energy,
    paired_rss_bootstrap_guard,
    projection_gradient_structure_scores,
    select_global_action_budget,
)
from adaptive_ct.compression import export_compact_octree_artifact, load_compact_octree_state_dict
from adaptive_ct.config import load_config
from adaptive_ct.data import load_r2_dataset
from adaptive_ct.functional_compression import (
    _evaluate_projection_split,
    load_reference_model,
)
from adaptive_ct.geometry import random_training_rays
from adaptive_ct.decision_rays import FixedDecisionRaySet, fixed_rss, paired_fixed_rss
from adaptive_ct.model import build_model
from scripts.run_at_nodal import evaluate_y_slice, load_volume_only


def save_checkpoint(path: Path, config: dict, model, *, metadata: dict | None = None) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config": config, "model": model.state_dict(), "adaptive_at": metadata or {}}, temporary)
    temporary.replace(path)


def load_model(config_path: str | Path, model_path: str | Path, device: torch.device):
    path = Path(model_path)
    if path.suffix.lower() != ".npz":
        return load_reference_model(config_path, path, device)
    config = load_config(Path(config_path))
    model = build_model(config).to(device=device)
    state, _metadata = load_compact_octree_state_dict(model, path, device=device)
    model.prepare_sparse_from_state_dict(state)
    model.load_state_dict(state, strict=False)
    model.eval()
    return config, model


def release_cuda(*objects) -> None:
    for value in objects:
        del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def energy_delta(candidate, baseline) -> dict[str, float]:
    return {
        "total": float(candidate.total - baseline.total),
        "rss_over_sigma_squared": float(candidate.data - baseline.data),
        "alpha_e_smooth": float(candidate.smooth - baseline.smooth),
        "beta_effective_e_edge": float(candidate.edge - baseline.edge),
        "k_nodal_log_n": float(candidate.complexity - baseline.complexity),
        "rss": float(candidate.rss - baseline.rss),
    }


def finetune_after_actions(
    model,
    dataset,
    *,
    affected_leaf_keys,
    warmup_iterations: int,
    min_iterations: int,
    max_iterations: int,
    decision_eval_every: int,
    plateau_window: int,
    plateau_relative_improvement: float,
    decision_rays: FixedDecisionRaySet,
    batch_rays: int,
    lr: float,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    neighborhood_rings: int,
) -> dict:
    setup_started = time.perf_counter()
    master_mask, affected_leaves = affected_master_mask(
        model,
        affected_leaf_keys,
        rings=int(neighborhood_rings),
    )
    print(
        json.dumps(
            {
                "phase": "finetune_mask_ready",
                "affected_leaves": affected_leaves,
                "affected_master_values": int(master_mask.sum().item()),
                "seconds": time.perf_counter() - setup_started,
            }
        ),
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    materialize_points = not model.prefer_compact_ray_batch()
    history = []
    log_every = max(1, int(decision_eval_every))
    initial_decision = fixed_rss(model, decision_rays)
    print(
        json.dumps(
            {
                "phase": "finetune_fixed_baseline_ready",
                "mse": float(initial_decision["mse"]),
                "seconds": time.perf_counter() - setup_started,
            }
        ),
        flush=True,
    )
    best_mse = float(initial_decision["mse"])
    best_iteration = -1
    best_values = model.coefficient_logits.detach().clone()
    decision_values = [best_mse]
    stopped_for_plateau = False
    completed_iterations = 0
    for iteration in range(int(max_iterations)):
        rays = random_training_rays(
            dataset.train,
            batch_rays=int(batch_rays),
            detector_h=int(detector_h),
            detector_w=int(detector_w),
            samples_per_ray=int(samples_per_ray),
            materialize_points=materialize_points,
        )
        prediction = model.integrate_ray_batch(rays)
        loss = 0.5 * torch.mean(torch.square(prediction - rays.target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if iteration < int(warmup_iterations) and model.coefficient_logits.grad is not None:
            model.coefficient_logits.grad[~master_mask] = 0
        optimizer.step()
        completed_iterations = iteration + 1
        if iteration % log_every == 0 or iteration == int(max_iterations) - 1:
            decision = fixed_rss(model, decision_rays)
            decision_mse = float(decision["mse"])
            decision_values.append(decision_mse)
            if decision_mse < best_mse:
                best_mse = decision_mse
                best_iteration = iteration
                best_values = model.coefficient_logits.detach().clone()
            row = {
                "iteration": iteration,
                "stage": "new_dof_warmup" if iteration < int(warmup_iterations) else "global",
                "projection_loss": float(loss.detach().item()),
                "fixed_decision_mse": decision_mse,
                "best_fixed_decision_mse": best_mse,
            }
            history.append(row)
            print(json.dumps({"phase": "finetune", **row}), flush=True)
            if (
                completed_iterations >= int(min_iterations)
                and len(decision_values) >= int(plateau_window)
            ):
                recent = decision_values[-int(plateau_window) :]
                start = float(recent[0])
                end = float(min(recent))
                relative_improvement = (start - end) / max(abs(start), 1.0e-12)
                if relative_improvement < float(plateau_relative_improvement):
                    stopped_for_plateau = True
                    break
    with torch.no_grad():
        model.coefficient_logits.copy_(best_values)
    return {
        "history": history,
        "iterations": completed_iterations,
        "warmup_iterations": int(warmup_iterations),
        "min_iterations": int(min_iterations),
        "max_iterations": int(max_iterations),
        "stopped_for_plateau": stopped_for_plateau,
        "best_iteration": best_iteration,
        "initial_fixed_decision_mse": float(initial_decision["mse"]),
        "best_fixed_decision_mse": best_mse,
        "affected_leaves": affected_leaves,
        "affected_master_values": int(master_mask.sum().item()),
    }


def at_optimize(
    model,
    dataset,
    *,
    alpha: float,
    beta: float,
    epsilon: float,
    outer_iterations: int,
    mu_steps: int,
    ray_batch: int,
    leaf_batch: int,
    mu_lr: float,
    cg_iterations: int,
    cg_tolerance: float,
    phase_leaf_chunk: int,
    face_jump_weight: float,
    max_face_pairs: int,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    sigma_squared: float,
    full_observation_count: int,
) -> tuple[NodalPhaseField, list[dict]]:
    phase = NodalPhaseField(model, initial_value=1.0).to(device=model.coefficient_logits.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(mu_lr))
    history = []
    materialize_points = not model.prefer_compact_ray_batch()
    for outer in range(int(outer_iterations)):
        outer_started = time.perf_counter()
        phase.values.requires_grad_(False)
        model.coefficient_logits.requires_grad_(True)
        last = None
        for _ in range(int(mu_steps)):
            rays = random_training_rays(
                dataset.train,
                batch_rays=int(ray_batch),
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                materialize_points=materialize_points,
            )
            prediction = model.integrate_ray_batch(rays)
            projection_mse = torch.mean((prediction - rays.target) ** 2)
            data_loss = projection_mse / float(sigma_squared)
            leaves = volume_weighted_leaf_sample(model, int(leaf_batch))
            volume_terms = ambrosio_tortorelli_terms(
                model,
                phase,
                leaves,
                alpha=alpha,
                beta=beta,
                epsilon=epsilon,
            )
            face_terms = phase_weighted_face_jump_terms(
                model,
                phase,
                weight=float(face_jump_weight),
                max_pairs=int(max_face_pairs),
            )
            regularizer = volume_terms.total + float(alpha) * face_terms.total
            loss = data_loss + regularizer / float(full_observation_count)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last = {
                "data_loss": float(data_loss.detach().item()),
                "projection_mse": float(projection_mse.detach().item()),
                "at_volume": float(volume_terms.total.detach().item()),
                "face_jump": float(face_terms.total.detach().item()),
                "raw_face_jump": float(face_terms.raw_jump.detach().item()),
            }
            print(json.dumps({"phase": "at_mu_step", "outer": outer, **last, "seconds": time.perf_counter() - outer_started}), flush=True)
        model.coefficient_logits.requires_grad_(False)
        phase.values.requires_grad_(True)
        solve = solve_phase_field_cg(
            model,
            phase,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            max_iterations=int(cg_iterations),
            tolerance=float(cg_tolerance),
            absolute_tolerance=0.0,
            leaf_chunk=int(phase_leaf_chunk),
            face_jump_weight=float(face_jump_weight),
            max_face_pairs=int(max_face_pairs),
        )
        print(json.dumps({"phase": "at_phase_solve", "outer": outer, **solve.__dict__, "seconds": time.perf_counter() - outer_started}), flush=True)
        history.append({"outer": outer, **(last or {}), "phase_solve": solve.__dict__})
    model.coefficient_logits.requires_grad_(True)
    phase.values.requires_grad_(True)
    return phase, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive free-discontinuity nodal CT with AT-driven h/face actions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--air-quantile", type=float, default=0.05)
    parser.add_argument("--air-noise-samples", type=int, default=1_000_000)
    parser.add_argument("--gradient-tail-quantile", type=float, default=0.995)
    parser.add_argument("--gradient-sampled-leaves", type=int, default=262144)
    parser.add_argument("--phase-kappa", type=float, default=2.0)
    parser.add_argument("--parameters-per-face", type=float, default=4.0)
    parser.add_argument("--phase-gate-minimum", type=float, default=0.1)
    parser.add_argument(
        "--phase-gate-only",
        action="store_true",
        help="Solve and export the calibrated phase detector, but never execute structural actions.",
    )
    parser.add_argument("--epsilon-multipliers", default="8,4,2")
    parser.add_argument("--outer-iterations", type=int, default=1)
    parser.add_argument("--mu-steps", type=int, default=10)
    parser.add_argument("--mu-lr", type=float, default=2e-4)
    parser.add_argument("--ray-batch", type=int, default=8192)
    parser.add_argument("--leaf-batch", type=int, default=65536)
    parser.add_argument("--cg-iterations", type=int, default=200)
    parser.add_argument("--cg-tolerance", type=float, default=1e-4)
    parser.add_argument("--phase-leaf-chunk", type=int, default=65536)
    parser.add_argument("--face-jump-weight", type=float, default=1.0)
    parser.add_argument("--max-face-pairs", type=int, default=65536)
    parser.add_argument("--split-threshold", type=float, default=0.5)
    parser.add_argument("--merge-threshold", type=float, default=0.1)
    parser.add_argument("--face-threshold", type=float, default=0.5)
    parser.add_argument("--compatibility-threshold", type=float, default=1e-5)
    parser.add_argument("--round-added-byte-budget", type=int, default=4_000_000)
    parser.add_argument("--isolation-rings", type=int, default=0)
    parser.add_argument("--max-leaf-count", type=int, default=7_500_000)
    parser.add_argument("--decision-rays", type=int, default=262144)
    parser.add_argument("--finetune-warmup-iterations", type=int, default=200)
    parser.add_argument("--finetune-min-iterations", type=int, default=500)
    parser.add_argument("--finetune-max-iterations", type=int, default=2000)
    parser.add_argument("--decision-eval-every", type=int, default=50)
    parser.add_argument("--plateau-window", type=int, default=5)
    parser.add_argument("--plateau-relative-improvement", type=float, default=2.0e-4)
    parser.add_argument(
        "--prefix-bisection-steps",
        type=int,
        default=0,
        help="Maximum prefix trials; 0 searches to a one-action interval.",
    )
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--energy-rays", "--bic-rays", dest="energy_rays", type=int, default=65536)
    parser.add_argument("--rss-cv-blocks", type=int, default=16)
    parser.add_argument("--rss-bootstrap-blocks", type=int, default=32)
    parser.add_argument("--rss-bootstrap-samples", type=int, default=4096)
    parser.add_argument("--direct-jump-weight", type=float, default=1.0)
    parser.add_argument("--slice-index", type=int, default=512)
    parser.add_argument("--eval-ray-chunk", type=int, default=65536)
    parser.add_argument("--action-score-leaf-chunk", type=int, default=8192)
    args = parser.parse_args()

    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    epsilon_multipliers = tuple(float(item) for item in args.epsilon_multipliers.split(",") if item.strip())
    if not epsilon_multipliers or any(value <= 0.0 for value in epsilon_multipliers):
        raise ValueError("epsilon multipliers must be positive")

    config, initial = load_model(args.config, args.checkpoint, device)
    if not initial.constrained_nodal:
        raise ValueError("Adaptive AT requires a constrained nodal p1 checkpoint")
    dataset = load_r2_dataset(config["dataset"]["root"], device=device, load_volume=False)
    detector_h, detector_w = dataset.detector_shape
    samples_per_ray = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))
    full_observation_count = int(dataset.train.projections.numel())
    seed = int(config.get("training", {}).get("seed", 0))
    torch.manual_seed(seed + 771)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 771)
    full_decision_rays = FixedDecisionRaySet.full(
        dataset.train,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        chunk_rays=int(args.energy_rays),
        materialize_points=not initial.prefer_compact_ray_batch(),
    )
    optimization_decision_rays = full_decision_rays.evenly_spaced_subset(
        min(int(args.decision_rays), full_decision_rays.observation_count)
    )
    accepted_compact = output / "accepted_state.npz"
    export_compact_octree_artifact(
        initial,
        accepted_compact,
        quantization="float16",
        topology="packed_hierarchy",
    )
    noise = estimate_air_noise_variance(
        dataset.train.projections,
        quantile=float(args.air_quantile),
        max_samples=int(args.air_noise_samples),
    )
    sigma_squared = float(noise["sigma_squared"])
    initial_rss = fixed_rss(initial, full_decision_rays)
    finest_h = float(torch.max(2.0 / torch.tensor(initial.level_shapes[-1], dtype=torch.float32)).item())
    started = time.perf_counter()
    gradient_statistics = attenuation_gradient_tail_statistics(
        initial,
        quantile=float(args.gradient_tail_quantile),
        sampled_leaves=int(args.gradient_sampled_leaves),
        seed=seed + 991,
    )
    epsilon_final = min(epsilon_multipliers) * finest_h
    calibration = calibrate_at_weights(
        initial,
        observations=full_observation_count,
        epsilon_final=epsilon_final,
        g_edge=float(gradient_statistics["g_edge"]),
        kappa=float(args.phase_kappa),
        parameters_per_face=float(args.parameters_per_face),
    )
    beta_effective = float(calibration["beta_effective"])
    alpha_final = float(calibration["alpha_final"])
    print(
        json.dumps(
            {
                "phase": "closed_form_calibration",
                "gradient_statistics": gradient_statistics,
                "calibration": calibration,
            }
        ),
        flush=True,
    )
    detector_phase, detector_history = at_optimize(
        initial,
        dataset,
        alpha=alpha_final,
        beta=beta_effective,
        epsilon=epsilon_final,
        outer_iterations=1,
        mu_steps=0,
        ray_batch=int(args.ray_batch),
        leaf_batch=int(args.leaf_batch),
        mu_lr=float(args.mu_lr),
        cg_iterations=int(args.cg_iterations),
        cg_tolerance=float(args.cg_tolerance),
        phase_leaf_chunk=int(args.phase_leaf_chunk),
        face_jump_weight=float(args.face_jump_weight),
        max_face_pairs=int(args.max_face_pairs),
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        sigma_squared=sigma_squared,
        full_observation_count=full_observation_count,
    )
    detector_face_scores = phase_leaf_face_scores(
        initial,
        detector_phase,
        leaf_chunk=int(args.phase_leaf_chunk),
    )
    active_detector_faces = int((detector_face_scores > float(args.face_threshold)).sum().item())
    detector_values = detector_phase.master_values().detach()
    detector_gate = {
        "phase_min": float(detector_values.min().item()),
        "phase_mean": float(detector_values.mean().item()),
        "phase_max": float(detector_values.max().item()),
        "active_faces": active_detector_faces,
        "required_phase_min_below": float(args.phase_gate_minimum),
        "pass": bool(
            float(detector_values.min().item()) < float(args.phase_gate_minimum)
            and active_detector_faces > 0
        ),
        "history": detector_history,
    }
    torch.save(
        {
            "master_values": detector_values.cpu(),
            "metadata": {
                "model": str(accepted_compact.resolve()),
                "epsilon": epsilon_final,
                "alpha": alpha_final,
                "beta_effective": beta_effective,
                "gradient_statistics": gradient_statistics,
                "calibration": calibration,
                "detector_gate": detector_gate,
            },
        },
        output / "phase_field.pt",
    )
    print(json.dumps({"phase": "phase_detector_gate", **detector_gate}), flush=True)
    del initial, detector_phase, detector_face_scores
    release_cuda()

    rounds = []
    accepted_epsilons = []
    scheduled_multipliers = (
        epsilon_multipliers
        if not bool(args.phase_gate_only)
        else ()
    )
    for round_id, multiplier in enumerate(scheduled_multipliers, start=1):
        _cfg, current = load_model(args.config, accepted_compact, device)
        round_baseline = output / f"round_{round_id:02d}_baseline.npz"
        export_compact_octree_artifact(
            current,
            round_baseline,
            quantization="float16",
            topology="packed_hierarchy",
        )
        baseline = evaluate_direct_decision_energy(
            current,
            full_decision_rays,
            sigma_squared=sigma_squared,
            face_parameters=float(args.parameters_per_face),
            direct_jump_weight=float(args.direct_jump_weight),
            max_face_pairs=int(args.max_face_pairs),
        )
        leaf_scores, face_scores = projection_gradient_structure_scores(
            current,
            full_decision_rays,
        )
        action_plan = select_global_action_budget(
            current,
            leaf_scores,
            face_scores,
            added_byte_budget=int(args.round_added_byte_budget),
            split_threshold=float(args.split_threshold),
            merge_threshold=float(args.merge_threshold),
            face_threshold=float(args.face_threshold),
            compatibility_threshold=float(args.compatibility_threshold),
        )
        ordered_kinds = action_plan.ordered_action_kinds
        prefix_low = 0
        prefix_high = len(ordered_kinds) + 1
        prefix = len(ordered_kinds)
        attempts = []
        accepted_prefix = 0
        accepted_energy = None
        accepted_topology = None
        accepted_finetune = None
        prefix_attempt_limit = int(args.prefix_bisection_steps)
        if prefix_attempt_limit <= 0:
            prefix_attempt_limit = max(1, int(math.ceil(math.log2(max(len(ordered_kinds), 1)))) + 2)
        for attempt_id in range(prefix_attempt_limit):
            if prefix <= 0:
                break
            prefix_kinds = ordered_kinds[:prefix]
            split_count = prefix_kinds.count("split")
            merge_count = prefix_kinds.count("merge")
            face_count = prefix_kinds.count("face")
            attempt_started = time.perf_counter()
            candidate, topology = apply_phase_topology_actions(
                current,
                leaf_scores,
                face_scores,
                split_threshold=float(args.split_threshold),
                merge_threshold=float(args.merge_threshold),
                face_threshold=float(args.face_threshold),
                compatibility_threshold=float(args.compatibility_threshold),
                max_leaf_count=int(args.max_leaf_count),
                max_splits=split_count,
                max_merges=merge_count,
                max_face_changes=face_count,
                isolation_rings=int(args.isolation_rings),
            )
            print(
                json.dumps(
                    {
                        "phase": "topology_candidate_ready",
                        "attempt": attempt_id,
                        "prefix": prefix,
                        "seconds": time.perf_counter() - attempt_started,
                        "split_parents": topology.split_parents,
                        "merged_parents": topology.merged_parents,
                        "face_changes": topology.face_changes,
                        "leaves_after": topology.leaves_after,
                        "private_faces_after": topology.private_faces_after,
                    }
                ),
                flush=True,
            )
            finetune = finetune_after_actions(
                candidate,
                dataset,
                affected_leaf_keys=topology.affected_leaf_keys,
                warmup_iterations=int(args.finetune_warmup_iterations),
                min_iterations=int(args.finetune_min_iterations),
                max_iterations=int(args.finetune_max_iterations),
                decision_eval_every=int(args.decision_eval_every),
                plateau_window=int(args.plateau_window),
                plateau_relative_improvement=float(args.plateau_relative_improvement),
                decision_rays=optimization_decision_rays,
                batch_rays=int(args.ray_batch),
                lr=float(args.finetune_lr),
                detector_h=detector_h,
                detector_w=detector_w,
                samples_per_ray=samples_per_ray,
                neighborhood_rings=int(args.isolation_rings),
            )
            criterion = evaluate_direct_decision_energy(
                candidate,
                full_decision_rays,
                sigma_squared=sigma_squared,
                face_parameters=float(args.parameters_per_face),
                direct_jump_weight=float(args.direct_jump_weight),
                max_face_pairs=int(args.max_face_pairs),
            )
            rss_relative_delta = (
                float(criterion.rss - baseline.rss)
                / max(abs(float(baseline.rss)), 1.0e-12)
            )
            rss_safety_pass = rss_relative_delta <= 1.0e-7
            accepted = criterion.total < baseline.total and rss_safety_pass
            topology_summary = {
                key: value
                for key, value in topology.__dict__.items()
                if key != "affected_leaf_keys"
            }
            topology_summary["affected_leaf_key_count"] = len(topology.affected_leaf_keys)
            attempt = {
                "attempt": attempt_id,
                "prefix": prefix,
                "requested_counts": {
                    "split": split_count,
                    "merge": merge_count,
                    "face": face_count,
                },
                "topology": topology_summary,
                "finetune": finetune,
                "candidate_energy": criterion.__dict__,
                "energy_delta": {
                    "total": float(criterion.total - baseline.total),
                    "data": float(criterion.data - baseline.data),
                    "complexity": float(criterion.complexity - baseline.complexity),
                    "direct_jump": float(criterion.direct_jump - baseline.direct_jump),
                    "rss": float(criterion.rss - baseline.rss),
                },
                "rss_relative_delta": rss_relative_delta,
                "rss_safety_pass": rss_safety_pass,
                "accepted": accepted,
            }
            attempts.append(attempt)
            print(json.dumps({"phase": "direct_prefix_attempt", **attempt}), flush=True)
            if accepted:
                prefix_low = prefix
                if prefix >= accepted_prefix:
                    accepted_prefix = prefix
                    accepted_energy = criterion
                    accepted_topology = topology
                    accepted_finetune = finetune
                    export_compact_octree_artifact(
                        candidate,
                        accepted_compact,
                        quantization="float16",
                        topology="packed_hierarchy",
                    )
                if prefix_high > len(ordered_kinds):
                    break
                next_prefix = (prefix + prefix_high) // 2
            else:
                prefix_high = prefix
                next_prefix = (prefix_low + prefix) // 2
            del candidate
            release_cuda()
            if next_prefix == prefix:
                break
            prefix = next_prefix
        accepted = accepted_prefix > 0
        if accepted:
            accepted_epsilons.append(multiplier)
        action_plan_summary = {
            key: value
            for key, value in action_plan.__dict__.items()
            if key != "ordered_action_kinds"
        }
        action_plan_summary["ordered_action_count"] = len(action_plan.ordered_action_kinds)
        accepted_topology_summary = None
        if accepted_topology is not None:
            accepted_topology_summary = {
                key: value
                for key, value in accepted_topology.__dict__.items()
                if key != "affected_leaf_keys"
            }
            accepted_topology_summary["affected_leaf_key_count"] = len(
                accepted_topology.affected_leaf_keys
            )
        row = {
            "round": round_id,
            "epsilon_multiplier": multiplier,
            "decision_mode": "fixed_rays_direct_jump_mdl",
            "baseline_energy": baseline.__dict__,
            "action_plan": action_plan_summary,
            "prefix_attempts": attempts,
            "accepted_prefix": accepted_prefix,
            "accepted_energy": accepted_energy.__dict__ if accepted_energy is not None else None,
            "accepted_topology": accepted_topology_summary,
            "accepted_finetune": accepted_finetune,
            "accepted": accepted,
        }
        rounds.append(row)
        print(json.dumps({"phase": "adaptive_direct_round", **row}), flush=True)
        del current, leaf_scores, face_scores
        release_cuda()

    _cfg, final_model = load_model(args.config, accepted_compact, device)
    final_projection = _evaluate_projection_split(
        final_model,
        dataset.test,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        ray_chunk=int(args.eval_ray_chunk),
    )
    volume = load_volume_only(config["dataset"]["root"])
    final_slice = evaluate_y_slice(final_model, volume, index=int(args.slice_index))
    compact = export_compact_octree_artifact(
        final_model,
        output / "compact_nodal_v5.npz",
        quantization="float16",
        topology="packed_hierarchy",
    )
    report = {
        "schema": "adaptive_nodal_ct_v4_fixed_decision_direct_mdl",
        "config": str(Path(args.config).resolve()),
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "gradient_statistics": gradient_statistics,
        "calibration": calibration,
        "alpha_final": alpha_final,
        "alpha_by_epsilon_multiplier": {
            str(multiplier): alpha_for_epsilon(
                calibration,
                float(multiplier) * finest_h,
            )
            for multiplier in epsilon_multipliers
        },
        "beta_effective": beta_effective,
        "noise_estimate": noise,
        "initial_rss_estimate": initial_rss,
        "phase_detector_gate": detector_gate,
        "phase_role": "visualization_and_ablation_only",
        "decision_protocol": {
            "training_views": int(dataset.train.projections.shape[0]),
            "dev_views": 0,
            "full_fixed_decision_rays": int(full_decision_rays.observation_count),
            "optimization_decision_rays": int(optimization_decision_rays.observation_count),
            "judge": "RSS/sigma^2 + nodal MDL + face MDL + direct attenuation jump",
            "selection": "single Lagrangian split/merge/untie/retie queue",
            "acceptance": "prefix bisection with fixed-ray best iterate",
        },
        "phase_gate_only": bool(args.phase_gate_only),
        "structure_rounds_skipped": not bool(scheduled_multipliers),
        "structure_rounds_skipped_reason": (
            "phase_gate_only"
            if bool(args.phase_gate_only)
            else None
        ),
        "epsilon_multipliers": list(epsilon_multipliers),
        "accepted_epsilon_multipliers": accepted_epsilons,
        "rounds": rounds,
        "final_projection": final_projection,
        "final_slice": final_slice,
        "model_stats": final_model.stats().__dict__,
        "private_faces": int(final_model.nodal_private_face_mask.sum().item()),
        "compact": compact.__dict__,
        "elapsed_sec": time.perf_counter() - started,
    }
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
