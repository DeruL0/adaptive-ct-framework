"""Reproducible timing of the pipeline's parts from states a training run saved (output/<run>/states/).

  ops       single-view micro-timings of the model at the final solve: ray batch, forward, forward + backward,
            edge penalty (build, value + gradient, curvature), one step-geometry update
  solve     the final strict solve from pre_final_solve.pt (same settings as the run): seconds, passes, continuations,
            objective and the reference optimality residual at the returned values (fixed denominator D_ref = the
            solver's curvature at the start, so solvers with other preconditioners are judged by the same number)
  decision  one structure decision from pre_decision_last.pt: seconds of each part, actions, memory peak

usage: python benchmark_pipeline.py <run> [--stages ops,solve,decision] [--repeats N] [--max-seconds S]
Results: output/_bench/<run>_<commit>.json (one entry per repeat, plus median and spread).  No ground truth is read.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adaptive_ct.data.dataset import load_r2_dataset
from adaptive_ct.refinement.decision import decide_and_apply, decision_views
from adaptive_ct.representation.factory import build_model
from adaptive_ct.training.geometry import geometry_step
from adaptive_ct.training.loop import _split_dev_views
from adaptive_ct.training.regularizer import EdgePenalty
from adaptive_ct.training.solve import curvature, gradient, ray_batches, solve

dev = torch.device("cuda")


def sync_time(fn, repeats: int) -> list[float]:
    fn()  # warm-up
    out = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t)
    return out


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "runs": values}


def load(run_dir: Path, name: str):
    state = torch.load(run_dir / "states" / name, map_location="cpu", weights_only=False)
    config = state["config"] if "config" in state else torch.load(run_dir / "states" / "pre_final_solve.pt", map_location="cpu", weights_only=False)["config"]
    return state, config


def build(state, config):
    model = build_model(config)
    model.prepare_sparse_from_state_dict(state["model"])
    model.load_state_dict(state["model"], strict=True)
    return model.to(dev)


def dataset(config, state):
    ds = load_r2_dataset(config["dataset"]["root"], device=dev, load_volume=False)
    ds = _split_dev_views(ds, config.get("dataset", {}) or {}, int(config["training"]["seed"]))
    if state.get("train_angles") is not None:
        assert torch.equal(ds.train.angles.cpu(), state["train_angles"]), "view split differs from the run's"
    return ds


def stage_ops(run_dir, repeats):
    state, config = load(run_dir, "pre_final_solve.pt")
    model, ds = build(state, config), dataset(config, state)
    edge = config["training"]["solver"]["edge_penalty"]
    batch = next(ray_batches(ds.train.angles, ds.train.projections, [100]))
    out = {"leaves": model.leaf_count, "rays_per_view": batch.num_rays}
    out["ray_batch_prep_s"] = spread(sync_time(lambda: next(ray_batches(ds.train.angles, ds.train.projections, [100])), repeats))
    with torch.no_grad():
        out["forward_s"] = spread(sync_time(lambda: model.integrate_ray_batch(batch), repeats))
    model.values.requires_grad_(True)

    def forward_backward():
        model.values.grad = None
        e = model.integrate_ray_batch(batch) - batch.target
        (0.5 * (e * e).sum()).backward()

    out["forward_plus_backward_s"] = spread(sync_time(forward_backward, repeats))
    model.values.requires_grad_(False)
    penalty = EdgePenalty(model, float(edge["delta"]))
    c = model.values.detach()
    out["penalty_samples"] = penalty.samples
    out["penalty_build_s"] = spread(sync_time(lambda: EdgePenalty(model, float(edge["delta"])), max(1, repeats // 2)))
    out["penalty_value_gradient_s"] = spread(sync_time(lambda: penalty.value_and_gradient(c), repeats))
    out["penalty_curvature_s"] = spread(sync_time(penalty.curvature, repeats))
    del penalty
    if model.n_step:
        saved = model.step_geometry.clone()

        def one_geometry_step():
            model.step_geometry.copy_(saved)
            geometry_step(model, ds.train, detector_bin=1, penalty_weight=float(edge["weight"]), penalty_delta=float(edge["delta"]), alpha=0.5)

        out["geometry_step_s"] = spread(sync_time(one_geometry_step, max(1, repeats // 2)))
    return out


def stage_solve(run_dir, repeats, max_seconds):
    state, config = load(run_dir, "pre_final_solve.pt")
    ds = dataset(config, state)
    cfg, edge = config["training"]["solver"], config["training"]["solver"]["edge_penalty"]
    weight = float(edge["weight"])
    results = []
    for _ in range(repeats):
        model = build(state, config)
        penalty = EdgePenalty(model, float(edge["delta"]))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        sstate, calls, passes, converged = {}, 0, 0, False
        while not converged and time.perf_counter() - t0 < max_seconds:
            _, more, converged = solve(model, ds.train, passes=int(cfg["final_passes"]), tol=float(cfg["final_tol"]), groups=int(cfg["groups"]),
                                       group_passes=int(cfg["group_passes"]), extrapolate=bool(cfg["extrapolate"]), detector_bin=1,
                                       penalty=penalty, penalty_weight=weight, state=sstate)
            passes += int(more[-1]["pass"])  # full-data-equivalent passes of the call, curvature and checks included
            calls += 1
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        # objective and reference residual at the returned values, with the start's curvature as denominator
        model.values.requires_grad_(True)
        views = list(range(int(ds.train.angles.numel())))
        c = model.values.detach().clone()
        g, J = gradient(model, c, ds.train.angles, ds.train.projections, views)
        J = float(J)
        R, gR = penalty.value_and_gradient(c)
        d_ref = sstate["d"]
        rho = float((c - (c - (g + weight * gR) / d_ref).clamp_min(0)).norm() / c.norm())
        results.append({"seconds": seconds, "passes": passes, "solve_calls": calls, "converged": converged, "objective": J + weight * R, "data_term": J, "penalty": R,
                        "reference_residual": rho, "peak_gb": torch.cuda.max_memory_allocated() / 2**30})
        del penalty, model
    return {"runs": results, "seconds": spread([r["seconds"] for r in results])}


def stage_decision(run_dir, repeats):
    state, config = load(run_dir, "pre_decision_last.pt")
    final_state, _ = load(run_dir, "pre_final_solve.pt")
    ds = dataset(config, final_state)
    ref, edge, solver = config["training"]["refinement"], config["training"]["solver"]["edge_penalty"], config["training"]["solver"]
    h, w = ds.train.projections.shape[1:]
    results = []
    for _ in range(repeats):
        model = build(state, config)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        report, _ = decide_and_apply(model, ds.train, decision_views(ds.train, int(ref["decision_views"]), int(state["round"])), detector_h=h,
                                     detector_w=w, byte_budget=int(ref["byte_budget"]), action_share=float(ref["action_share"]),
                                     min_footprint_px=float(ref["min_footprint_px"]), penalty_weight=float(edge["weight"]),
                                     penalty_delta=float(edge["delta"]))
        torch.cuda.synchronize()
        results.append({"seconds": time.perf_counter() - t0, "parts": report["seconds"], "actions": report["actions"],
                        "candidates_with_positive_gain": report["candidates_with_positive_gain"], "bytes_after": report["bytes_after"],
                        "split_step_children": report.get("split_step_children"), "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                        "leaves": int(state["model"]["cells"].shape[0])})
        del model
    return {"runs": results, "seconds": spread([r["seconds"] for r in results])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--stages", default="ops,solve,decision")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-seconds", type=float, default=1800.0)
    args = ap.parse_args()
    run_dir = ROOT / "output" / args.run
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    result = {"run": args.run, "commit": commit + ("+uncommitted" if dirty else ""), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "repeats": args.repeats, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    for stage in args.stages.split(","):
        result[stage] = {"ops": lambda: stage_ops(run_dir, args.repeats), "solve": lambda: stage_solve(run_dir, args.repeats, args.max_seconds),
                         "decision": lambda: stage_decision(run_dir, args.repeats)}[stage]()
        print(stage, json.dumps(result[stage], default=float)[:1500], flush=True)
    out = ROOT / "output" / "_bench"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.run}_{commit}.json"
    path.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
