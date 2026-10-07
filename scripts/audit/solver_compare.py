"""Fixed-tree comparison of value solvers from the saved state before the final solve (states/pre_final_solve.pt).

  baseline  the pipeline's solver (solve.py: grouped warm-up, then accelerated full-data steps)
  lalm      relaxed OS-LALM (Nien & Fessler 2016, arXiv:1512.04564, Algorithm 1), W = I:
              s = rho (D_L x - h) + (1 - rho) g ;  x+ = [x - (rho D_L + D_R)^-1 (s + grad R(x))]_{>=0}
              zeta = M grad L_m(x+) ;  g+ = rho/(rho+1) (alpha zeta + (1-alpha) g) + g/(rho+1) ;  h+ = alpha (D_L x+ - zeta) + (1-alpha) h
            rho_0 = 1, rho_k = pi / (alpha (k+1)) sqrt(1 - (pi / (2 alpha (k+1)))^2), k = subiteration (--rho-per epoch:
            the epoch; the paper decays per subiteration); D_L = diag(A^T A 1) and
            D_R = beta * (the penalty's separable curvature bound) are the solver's own d; grad R at the start of each epoch
            (--penalty-every subiter, the paper's choice: before every subset; it costs more than a subset gradient)

Same projector, penalty and nonnegativity; the same objective J(c) = 1/2 |Ac - y|^2 + beta R(c) at bin 1.  Judged by one number
for all methods: rho_ref(c) = |c - max(0, c - grad J(c) / D_ref)| / |c| with D_ref = D_L + D_R at the start, measured at the
returned values; the time is the algorithm's own (the measuring pass is not counted).  No ground truth is used for any choice.

usage: python solver_compare.py <run> [--methods baseline,lalm] [--subsets 24] [--alpha 1.999] [--epochs 40]
                                [--penalty-every outer|subiter] [--measure-every 2] [--target 1e-4]
Results: output/_bench/solver_compare_<run>_<commit>.json
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adaptive_ct.data.dataset import load_r2_dataset
from adaptive_ct.representation.factory import build_model
from adaptive_ct.training.loop import _split_dev_views
from adaptive_ct.training.regularizer import EdgePenalty
from adaptive_ct.training.solve import curvature, gradient, solve

dev = torch.device("cuda")


class Clock:
    """Seconds of the algorithm: wall time minus the time spent measuring."""

    def __init__(self):
        self.t0, self.paused = time.perf_counter(), 0.0

    def now(self):
        torch.cuda.synchronize()
        return time.perf_counter() - self.t0 - self.paused

    def measuring(self, fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        self.paused += time.perf_counter() - t
        return out


def setup(run):
    run_dir = ROOT / "output" / run
    state = torch.load(run_dir / "states" / "pre_final_solve.pt", map_location="cpu", weights_only=False)
    config = state["config"]
    ds = _split_dev_views(load_r2_dataset(config["dataset"]["root"], device=dev, load_volume=False), config.get("dataset", {}) or {}, int(config["training"]["seed"]))

    def fresh():
        model = build_model(config)
        model.prepare_sparse_from_state_dict(state["model"])
        model.load_state_dict(state["model"], strict=True)
        return model.to(dev)

    return config, ds, fresh


def make_measure(model, ds, penalty, beta, d_ref):
    views = list(range(int(ds.train.angles.numel())))
    angles, projections = ds.train.angles, ds.train.projections

    def measure(c):
        model.values.requires_grad_(True)
        g, J = gradient(model, c, angles, projections, views)
        R, gR = penalty.value_and_gradient(c)
        rho = float((c - (c - (g + beta * gR) / d_ref).clamp_min(0)).norm() / c.norm().clamp_min(1e-30))
        return {"objective": float(J) + beta * R, "reference_residual": rho}

    return measure


def run_baseline(model, ds, penalty, beta, cfg, measure, every, clock):
    traj, state = [], {}

    def on_pass(c):
        row = {}
        if len(traj_passes) % every == 0:
            row = clock.measuring(lambda: measure(c))
            traj.append({"passes": len(traj_passes), "seconds": clock.now(), **row})
        traj_passes.append(1)
        return {}

    traj_passes = []
    solve(model, ds.train, passes=int(cfg["final_passes"]), tol=float(cfg["final_tol"]), groups=int(cfg["groups"]), group_passes=int(cfg["group_passes"]),
          extrapolate=bool(cfg["extrapolate"]), detector_bin=1, penalty=penalty, penalty_weight=beta, on_pass=on_pass, state=state)
    return traj


def run_lalm(model, ds, penalty, beta, d_l, d_r, subsets, alpha, epochs, penalty_every, rho_per, measure, every, clock):
    angles, projections = ds.train.angles, ds.train.projections
    order = sorted(range(int(angles.numel())), key=lambda v: float(angles[v]))
    groups = [order[g::subsets] for g in range(subsets)]
    stride = next(s for s in range(max(1, subsets // 3), subsets + 1) if math.gcd(s, subsets) == 1)  # consecutive subsets far apart in angle
    visit = [(i * stride) % subsets for i in range(subsets)]
    M = float(subsets)
    model.values.requires_grad_(True)
    x = model.values.detach().clone()
    zeta = M * gradient(model, x, angles, projections, groups[visit[-1]])[0]
    g, h, traj, sub = zeta.clone(), d_l * x - zeta, [], 0
    schedule = lambda k: 1.0 if k == 0 else (math.pi / (alpha * (k + 1))) * math.sqrt(1 - (math.pi / (2 * alpha * (k + 1))) ** 2)
    for k in range(epochs):
        rho = schedule(k)
        gR = beta * penalty.value_and_gradient(x)[1]
        for i, m in enumerate(visit):
            if rho_per == "subiter":
                rho = schedule(sub)
            sub += 1
            if penalty_every == "subiter" and i:
                gR = beta * penalty.value_and_gradient(x)[1]
            s = rho * (d_l * x - h) + (1 - rho) * g
            x = (x - (s + gR) / (rho * d_l + d_r)).clamp_min(0)
            zeta = M * gradient(model, x, angles, projections, groups[m])[0]
            g = (rho / (rho + 1)) * (alpha * zeta + (1 - alpha) * g) + g / (rho + 1)
            h = alpha * (d_l * x - zeta) + (1 - alpha) * h
        if (k + 1) % every == 0:
            traj.append({"passes": k + 1, "seconds": clock.now(), "rho": rho, **clock.measuring(lambda: measure(x))})
    return traj


def time_to(traj, target):
    for row in traj:
        if row["reference_residual"] <= target:
            return {"seconds": row["seconds"], "passes": row["passes"]}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--methods", default="baseline,lalm")
    ap.add_argument("--subsets", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=1.999)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--penalty-every", default="subiter", choices=["outer", "subiter"])
    ap.add_argument("--rho-per", default="subiter", choices=["epoch", "subiter"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--measure-every", type=int, default=2)
    ap.add_argument("--target", type=float, default=1e-4)
    args = ap.parse_args()
    config, ds, fresh = setup(args.run)
    cfg, edge = config["training"]["solver"], config["training"]["solver"]["edge_penalty"]
    beta = float(edge["weight"])
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    result = {"run": args.run, "commit": commit, "args": vars(args), "device": torch.cuda.get_device_name(0)}
    for method in args.methods.split(","):
        model = fresh()
        penalty = EdgePenalty(model, float(edge["delta"]))
        model.values.requires_grad_(True)
        views = list(range(int(ds.train.angles.numel())))
        d_l = curvature(model, ds.train.angles, ds.train.projections, views)
        d_r = beta * penalty.curvature()
        d_ref = d_l + d_r
        measure = make_measure(model, ds, penalty, beta, d_ref)
        start = measure(model.values.detach().clone())
        torch.cuda.synchronize()
        clock = Clock()
        if method == "baseline":
            traj = run_baseline(model, ds, penalty, beta, cfg, measure, args.measure_every, clock)
        else:
            traj = run_lalm(model, ds, penalty, beta, d_l, d_r, args.subsets, args.alpha, args.epochs, args.penalty_every, args.rho_per, measure, args.measure_every, clock)
        result[method] = {"start": start, "trajectory": traj, "time_to_target": time_to(traj, args.target), "best_residual": min(r["reference_residual"] for r in traj),
                          "final_objective": traj[-1]["objective"], "final_seconds": traj[-1]["seconds"]}
        print(method, "start", start, "to target", result[method]["time_to_target"], "best residual", result[method]["best_residual"], "final objective", traj[-1]["objective"], flush=True)
        for r in traj[:: max(1, len(traj) // 12)]:
            print("   ", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        del model, penalty
        torch.cuda.empty_cache()
    out = ROOT / "output" / "_bench"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"solver_compare_{args.run}_{commit}{args.tag}.json"
    path.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
