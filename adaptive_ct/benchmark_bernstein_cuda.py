from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .backend import has_bernstein_native
from .config import load_config
from .data import load_r2_dataset
from .geometry import make_parallel_ray_points
from .model import build_model


@torch.no_grad()
def _set_uniform_degree(model, degree: int) -> None:
    degree = int(degree)
    if degree < 0 or degree > 3:
        raise ValueError("CUDA benchmark degree must be in 0..3.")
    device = model.coefficient_logits.device
    leaf_count = int(model.leaf_levels.shape[0])
    coefficients_per_leaf = (degree + 1) ** 3
    leaf_mean = model._leaf_mu().detach()
    physical = leaf_mean.repeat_interleave(coefficients_per_leaf)
    model.coefficient_logits = torch.nn.Parameter(model._coefficients_to_logits(physical).contiguous())
    model.leaf_degrees = torch.full((leaf_count, 3), degree, dtype=torch.long, device=device)
    counts = torch.full((leaf_count,), coefficients_per_leaf, dtype=torch.long, device=device)
    model.coefficient_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(counts, dim=0)]
    )
    model.coefficient_leaf_ids = torch.repeat_interleave(torch.arange(leaf_count, device=device), counts)


def _make_ray_batch(dataset, ray_count: int, detector_h: int, detector_w: int, samples_per_ray: int):
    generator = torch.Generator(device=dataset.train.angles.device)
    generator.manual_seed(20260628)
    view_ids = torch.randint(
        0,
        dataset.train.angles.shape[0],
        (ray_count,),
        generator=generator,
        device=dataset.train.angles.device,
    )
    rows = torch.randint(0, detector_h, (ray_count,), generator=generator, device=dataset.train.angles.device)
    cols = torch.randint(0, detector_w, (ray_count,), generator=generator, device=dataset.train.angles.device)
    targets = dataset.train.projections[view_ids, rows, cols]
    return make_parallel_ray_points(
        dataset.train.angles[view_ids],
        rows,
        cols,
        detector_h=detector_h,
        detector_w=detector_w,
        samples_per_ray=samples_per_ray,
        targets=targets,
        materialize_points=False,
    )


def _measure(model, ray_batch, *, cuda_integrator: bool, warmup: int, iterations: int) -> dict:
    model.cuda_integrator = bool(cuda_integrator)
    for _ in range(int(warmup)):
        model.coefficient_logits.grad = None
        prediction = model.integrate_ray_batch(ray_batch)
        torch.mean(torch.square(prediction - ray_batch.target)).backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    forward_seconds = 0.0
    backward_seconds = 0.0
    for _ in range(int(iterations)):
        model.coefficient_logits.grad = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = model.integrate_ray_batch(ray_batch)
        torch.cuda.synchronize()
        middle = time.perf_counter()
        torch.mean(torch.square(prediction - ray_batch.target)).backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        forward_seconds += middle - start
        backward_seconds += end - middle
    forward_seconds /= float(iterations)
    backward_seconds /= float(iterations)
    total_seconds = forward_seconds + backward_seconds
    return {
        "forward_ms": 1000.0 * forward_seconds,
        "backward_ms": 1000.0 * backward_seconds,
        "step_ms": 1000.0 * total_seconds,
        "rays_per_second": float(ray_batch.num_rays) / max(total_seconds, 1e-12),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _correctness(model, ray_batch) -> dict:
    original_logits = model.coefficient_logits.detach().clone()
    with torch.no_grad():
        coefficient_ids = torch.arange(model.coefficient_logits.numel(), device=model.coefficient_logits.device)
        model.coefficient_logits.add_(0.05 * torch.sin(coefficient_ids.to(dtype=torch.float32)))
    model.cuda_integrator = False
    model.coefficient_logits.grad = None
    reference = model.integrate_ray_batch(ray_batch)
    reference.sum().backward()
    reference_gradient = model.coefficient_logits.grad.detach().clone()

    model.cuda_integrator = True
    model.coefficient_logits.grad = None
    actual = model.integrate_ray_batch(ray_batch)
    actual.sum().backward()
    actual_gradient = model.coefficient_logits.grad.detach()
    gradient_difference = actual_gradient - reference_gradient
    significant = torch.abs(reference_gradient) > 1e-6
    result = {
        "forward_max_abs": float(torch.max(torch.abs(actual - reference)).item()),
        "forward_max_rel": float(
            torch.max(torch.abs(actual - reference) / torch.abs(reference).clamp_min(1e-8)).item()
        ),
        "gradient_max_abs": float(torch.max(torch.abs(gradient_difference)).item()),
        "gradient_l2_rel": float(
            torch.linalg.vector_norm(gradient_difference).div(
                torch.linalg.vector_norm(reference_gradient).clamp_min(1e-12)
            ).item()
        ),
        "gradient_max_rel_significant": float(
            torch.max(torch.abs(gradient_difference[significant]) / torch.abs(reference_gradient[significant])).item()
            if torch.any(significant)
            else 0.0
        ),
    }
    with torch.no_grad():
        model.coefficient_logits.copy_(original_logits)
    model.coefficient_logits.grad = None
    return result


def run_benchmark(
    config_path: str | Path,
    *,
    batch_rays: int = 2048,
    warmup: int = 5,
    iterations: int = 20,
    degrees: tuple[int, ...] = (0, 3),
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    if not has_bernstein_native():
        raise RuntimeError("Bernstein CUDA extension is unavailable; rebuild setup.py first.")
    config = load_config(config_path)
    device = torch.device("cuda")
    dataset = load_r2_dataset(config["dataset"]["root"], device=device, load_volume=False)
    detector_h, detector_w = dataset.detector_shape
    samples_per_ray = int(config["geometry"].get("samples_per_ray", dataset.volume_shape[0]))
    cases = []
    for degree in degrees:
        model = build_model(config).to(device)
        initial_mu = (dataset.train.projections.mean() * 0.5).clamp_min(1e-8)
        with torch.no_grad():
            model.coefficient_logits.copy_(
                model._coefficients_to_logits(torch.full_like(model.coefficient_logits, float(initial_mu.item())))
            )
        _set_uniform_degree(model, degree)
        ray_batch = _make_ray_batch(dataset, batch_rays, detector_h, detector_w, samples_per_ray)
        correctness_batch = _make_ray_batch(
            dataset,
            min(128, batch_rays),
            detector_h,
            detector_w,
            samples_per_ray,
        )
        correctness = _correctness(model, correctness_batch)
        torch_reference = _measure(
            model,
            ray_batch,
            cuda_integrator=False,
            warmup=max(1, min(warmup, 2)),
            iterations=max(1, min(iterations, 5)),
        )
        cuda_native = _measure(
            model,
            ray_batch,
            cuda_integrator=True,
            warmup=warmup,
            iterations=iterations,
        )
        cases.append(
            {
                "uniform_degree": degree,
                "leaf_cells": int(model.leaf_levels.shape[0]),
                "coefficient_count": int(model.coefficient_logits.numel()),
                "correctness": correctness,
                "torch_exact": torch_reference,
                "cuda_native": cuda_native,
                "speedup_step": torch_reference["step_ms"] / max(cuda_native["step_ms"], 1e-12),
                "speedup_forward": torch_reference["forward_ms"] / max(cuda_native["forward_ms"], 1e-12),
                "speedup_backward": torch_reference["backward_ms"] / max(cuda_native["backward_ms"], 1e-12),
            }
        )
    return {
        "schema": "bernstein_cuda_benchmark_v1",
        "config": str(Path(config_path).resolve()),
        "dataset": str(dataset.root),
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_rays": int(batch_rays),
        "warmup": int(warmup),
        "iterations": int(iterations),
        "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark native CUDA Bernstein octree ray integration.")
    parser.add_argument("--config", default="configs/figure473_48v.yaml")
    parser.add_argument("--batch-rays", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--degrees", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--out", default="output/benchmarks/bernstein_cuda.json")
    args = parser.parse_args(argv)
    report = run_benchmark(
        args.config,
        batch_rays=args.batch_rays,
        warmup=args.warmup,
        iterations=args.iterations,
        degrees=tuple(args.degrees),
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
