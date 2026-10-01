"""Projection-driven Dörfler h-refinement.

Estimate -> mark -> split loop for packed Bernstein octrees:

* the leaf error indicator is the noise-corrected projection residual energy,
  attributed to leaves by ray length (additive, unbiased over random rays);
* Dörfler marking picks the smallest rate-ranked set carrying a fraction
  ``theta`` of the total indicator, truncated by the byte budget;
* splitting is function preserving (exact de Casteljau), so a batch can never
  change the projections by construction;
* optionally, each batch is verified once on held-out dev rays after digestion
  and rolled back with a halved prefix when it did not help.

Nothing here uses a reference volume or an analytic reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .backend import bernstein_segment_ray_capacity
from .decision_rays import FixedDecisionRaySet, fixed_rss
from .geometry import make_parallel_ray_points, random_training_rays
from .projection_domain import projection_weights, weighted_projection_loss

_STRUCTURE_BYTES_PER_SPLIT = 8 * 8  # eight new packed-hierarchy nodes, two int32 each


def sample_random_rays(
    split,
    *,
    ray_count: int,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    seed: int,
    chunk_rays: int,
    materialize_points: bool,
) -> FixedDecisionRaySet:
    """Uniform random rays over every view of ``split`` (with replacement)."""
    device = split.projections.device
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    count = max(1, int(ray_count))
    views = int(split.angles.numel())
    return FixedDecisionRaySet(
        split=split,
        view_ids=torch.randint(0, views, (count,), generator=generator).to(device),
        rows=torch.randint(0, int(detector_h), (count,), generator=generator).to(device),
        cols=torch.randint(0, int(detector_w), (count,), generator=generator).to(device),
        detector_h=int(detector_h),
        detector_w=int(detector_w),
        samples_per_ray=int(samples_per_ray),
        chunk_rays=max(1, int(chunk_rays)),
        materialize_points=bool(materialize_points),
    )


def _ray_batch(rays: FixedDecisionRaySet, start: int, stop: int):
    view_ids = rays.view_ids[start:stop]
    rows = rays.rows[start:stop]
    cols = rays.cols[start:stop]
    return make_parallel_ray_points(
        rays.split.angles[view_ids],
        rows,
        cols,
        detector_h=rays.detector_h,
        detector_w=rays.detector_w,
        samples_per_ray=rays.samples_per_ray,
        targets=rays.split.projections[view_ids, rows, cols],
        materialize_points=rays.materialize_points,
    )


def head_rays(rays: FixedDecisionRaySet, count: int) -> FixedDecisionRaySet:
    count = max(1, min(int(count), rays.observation_count))
    return FixedDecisionRaySet(
        split=rays.split,
        view_ids=rays.view_ids[:count],
        rows=rays.rows[:count],
        cols=rays.cols[:count],
        detector_h=rays.detector_h,
        detector_w=rays.detector_w,
        samples_per_ray=rays.samples_per_ray,
        chunk_rays=rays.chunk_rays,
        materialize_points=rays.materialize_points,
    )


@torch.no_grad()
def attributed_excess_residual(
    model,
    rays: FixedDecisionRaySet,
    *,
    noise_variance: float,
    weight_kwargs: dict,
    max_buffer_mb: float = 512.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Noise-corrected residual energy per leaf.

    ``eta_K = sum_r w_r (res_r^2 - sigma^2) * l_rK / L_r`` with ``l_rK`` the exact
    chord length of ray r inside leaf K and ``L_r`` its total length.  The sum
    over leaves of ``eta`` equals the sampled excess residual energy.  Also
    returns per-leaf exposure (total chord length) and a summary.
    """
    leaf_count = int(model.leaf_levels.shape[0])
    device = model.leaf_levels.device
    eta = torch.zeros((leaf_count,), dtype=torch.float32, device=device)
    exposure = torch.zeros_like(eta)
    chunk = min(
        rays.chunk_rays,
        bernstein_segment_ray_capacity(model.level_shapes, float(max_buffer_mb)),
    )
    rss = 0.0
    excess_total = 0.0
    for start in range(0, rays.observation_count, chunk):
        stop = min(start + chunk, rays.observation_count)
        batch = _ray_batch(rays, start, stop)
        residual = model.integrate_ray_batch(batch) - batch.target
        weights = projection_weights(batch.target, **weight_kwargs)
        energy = weights * torch.square(residual)
        excess = energy - weights * float(noise_variance)
        ray_ids, leaf_ids, intervals = model._parallel_ray_segments(batch)
        segment = intervals[:, 1] - intervals[:, 0]
        path = torch.zeros((batch.num_rays,), dtype=segment.dtype, device=device)
        path.scatter_add_(0, ray_ids, segment)
        share = segment / path[ray_ids].clamp_min(1.0e-12)
        eta.scatter_add_(0, leaf_ids, excess[ray_ids] * share)
        exposure.scatter_add_(0, leaf_ids, segment)
        rss += float(energy.sum().item())
        excess_total += float(excess.sum().item())
    return eta, exposure, {
        "estimator_rays": int(rays.observation_count),
        "estimator_weighted_rss": rss,
        "estimator_excess_total": excess_total,
        "estimator_excess_fraction": excess_total / max(rss, 1.0e-30),
        "covered_fraction": float((exposure > 0).float().mean().item()) if leaf_count else 0.0,
    }


def resolvable_mask(model, *, detector_h: int, detector_w: int, min_footprint_px: float) -> torch.Tensor:
    """Leaves whose children stay resolvable on the detector.

    Axes 0/1 project onto detector columns, axis 2 onto detector rows (same
    convention as ``make_parallel_ray_points``).  A child cell must span at
    least ``min_footprint_px`` detector pixels along every axis.
    """
    shapes = torch.tensor(model.level_shapes, dtype=torch.float32, device=model.leaf_levels.device)
    child_width = 1.0 / shapes[model.leaf_levels]  # 2 / (2 * shape)
    pixel = torch.tensor(
        [2.0 / float(detector_w), 2.0 / float(detector_w), 2.0 / float(detector_h)],
        dtype=torch.float32,
        device=model.leaf_levels.device,
    )
    fine_enough = torch.all(child_width >= float(min_footprint_px) * pixel[None, :] - 1.0e-9, dim=1)
    return fine_enough & (model.leaf_levels + 1 < len(model.level_shapes))


def _bytes_per_float(codec: dict) -> int:
    return 4 if str(codec.get("quantization", "float16")).lower() == "float32" else 2


def _per_leaf_bytes(codec: dict) -> int:
    kind = str(codec.get("type", "p0")).lower()
    if kind == "p0":
        values = 1
    elif kind == "p1":
        values = 8
    elif kind == "latent_p1":
        values = int(codec.get("latent_dim", 4))
    else:
        raise ValueError(f"Unknown codec type {kind!r}.")
    return values * _bytes_per_float(codec) + 3  # values + degree tag


def split_delta_bytes(codec: dict) -> int:
    """Raw packed bytes added by one octree split (parent leaf -> 8 children)."""
    return _STRUCTURE_BYTES_PER_SPLIT + 7 * _per_leaf_bytes(codec)


def codec_bytes(model, codec: dict) -> int:
    """Deterministic raw payload bytes of ``model`` stored with ``codec``."""
    kind = str(codec.get("type", "p0")).lower()
    degrees = model.leaf_degrees
    if kind == "p0" and bool(torch.any(degrees != 0)):
        raise ValueError("codec 'p0' requires every leaf to have degree 0.")
    if kind in {"p1", "latent_p1"} and bool(torch.any(degrees != 1)):
        raise ValueError(f"codec {kind!r} requires every leaf to be trilinear (degree 1).")
    leaf_count = int(model.leaf_levels.shape[0])
    nodes = int(model.node_child_base.numel())
    total = nodes * 8 + leaf_count * _per_leaf_bytes(codec)
    total += 4 + len(model.level_shapes) * 3 * 2 + 3 + 4
    if kind == "latent_p1":
        total += (8 * int(codec.get("latent_dim", 4)) + 8) * _bytes_per_float(codec)
    return int(total)


def dorfler_mark(
    eta: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    delta_bytes: int,
    theta: float,
    remaining_bytes: float,
    max_parents: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Smallest rate-ranked prefix carrying ``theta`` of the indicator mass.

    ``eta`` holds the (positive) indicator of ``candidate_ids``.  The prefix is
    then truncated by the byte budget and by ``max_parents``.  Every split costs
    the same ``delta_bytes`` (uniform degree), so the rate ranking is the eta
    ranking.
    """
    count = int(candidate_ids.numel())
    if count == 0:
        return candidate_ids, {"k_theta": 0, "k_bytes": 0, "eta_fraction": 0.0}
    order = torch.argsort(eta, descending=True)
    cumulative = torch.cumsum(eta[order].double(), dim=0)
    total = float(cumulative[-1].item())
    target = float(theta) * total
    k_theta = int(torch.searchsorted(cumulative, cumulative.new_tensor(target)).item()) + 1
    k_theta = min(k_theta, count)
    k_bytes = count if remaining_bytes == float("inf") else max(0, int(remaining_bytes // max(1, int(delta_bytes))))
    k = min(k_theta, k_bytes, count)
    if max_parents is not None:
        k = min(k, max(0, int(max_parents)))
    fraction = float(cumulative[k - 1].item()) / max(total, 1.0e-30) if k > 0 else 0.0
    return candidate_ids[order[:k]], {"k_theta": k_theta, "k_bytes": k_bytes, "eta_fraction": fraction}


def _level_histogram(levels: torch.Tensor, level_count: int) -> list[int]:
    return torch.bincount(levels, minlength=level_count).tolist() if levels.numel() else [0] * level_count


class DorflerController:
    """Stateful estimate/mark/split loop driven by plateau triggers."""

    def __init__(
        self,
        *,
        spec: dict,
        estimator_split,
        dev_split,
        detector_h: int,
        detector_w: int,
        samples_per_ray: int,
        materialize_points: bool,
        chunk_rays: int,
        seed: int,
        byte_budget: int | None,
        clone_state: Callable,
        restore_state: Callable,
    ):
        self.spec = dict(spec)
        self.codec = dict(spec.get("codec") or {"type": "p0", "quantization": "float16"})
        self.theta = float(spec.get("theta", 0.5))
        if not 0.0 < self.theta <= 1.0:
            raise ValueError("dorfler theta must be in (0, 1].")
        self.estimator_split = estimator_split
        self.detector_h = int(detector_h)
        self.detector_w = int(detector_w)
        self.samples_per_ray = int(samples_per_ray)
        self.materialize_points = bool(materialize_points)
        self.chunk_rays = int(chunk_rays)
        self.seed = int(seed)
        self.byte_budget = int(byte_budget) if byte_budget is not None else None
        self.clone_state = clone_state
        self.restore_state = restore_state
        # The air-quantile variance estimate is a heuristic upper bound that can exceed
        # the whole residual MSE; the scale lets a run disable the noise-floor stop.
        self.noise_variance_scale = float(spec.get("noise_variance_scale", 1.0))
        self.rollback_enabled = bool(spec.get("rollback", True)) and dev_split is not None
        self.dev_rays = None
        self.diagnostic_rays = None
        if self.rollback_enabled:
            self.dev_rays = sample_random_rays(
                dev_split,
                ray_count=int(spec.get("dev_ray_count", 1 << 20)),
                detector_h=self.detector_h,
                detector_w=self.detector_w,
                samples_per_ray=self.samples_per_ray,
                seed=self.seed + 17,
                chunk_rays=self.chunk_rays,
                materialize_points=self.materialize_points,
            )
            self.diagnostic_rays = head_rays(self.dev_rays, int(spec.get("diagnostic_rays", 1 << 18)))
        self.round_index = 0
        self.pending: dict | None = None
        self.stopped = False
        self.stop_reasons: list[str] = []

    def window_overrides(self) -> dict:
        """Plateau window settings for the current (possibly early) round."""
        if self.round_index < int(self.spec.get("early_rounds", 0)):
            return {
                "window": self.spec.get("early_window"),
                "min_iterations_after_previous": self.spec.get("early_min_after_previous"),
            }
        return {}

    def _stop(self, event: dict, reason: str) -> tuple[int, dict]:
        self.stopped = True
        self.stop_reasons.append(reason)
        event["stop"] = True
        event["stop_reasons"] = list(self.stop_reasons)
        return 0, event

    def on_trigger(self, model, *, iteration: int, noise_variance: float, weight_kwargs: dict) -> tuple[int, dict]:
        event: dict = {
            "dorfler_round": self.round_index + 1,
            "theta": self.theta,
            "rollback_enabled": self.rollback_enabled,
            "codec": self.codec,
            "stop": False,
        }
        level_count = len(model.level_shapes)
        if self.pending is not None:
            pending = self.pending
            rss_now = float(fixed_rss(model, self.dev_rays)["rss"])
            gain = (pending["rss_before"] - rss_now) / max(pending["rss_before"], 1.0e-30)
            event.update(
                verdict_rss_before=pending["rss_before"],
                verdict_rss_after=rss_now,
                verdict_relative_gain=gain,
                verdict_prefix=pending["prefix"],
            )
            min_gain = float(self.spec.get("rollback_min_relative_gain", 0.0))
            if gain > min_gain:
                event["verdict"] = "confirmed"
                self.pending = None
            else:
                self.restore_state(model, pending["pre_state"])
                halvings = pending["halvings"] + 1
                prefix = pending["prefix"] // 2
                event.update(verdict="rolled_back", halvings=halvings, next_prefix=prefix)
                if halvings > int(self.spec.get("max_halvings", 3)) or prefix < 1:
                    self.pending = None
                    return self._stop(event, "rollback_exhausted")
                model.split_leaves_batch(pending["marked_ids"][:prefix])
                pending.update(prefix=prefix, halvings=halvings)
                event.update(active_splits=prefix, stop=False)
                return prefix, event

        self.round_index += 1
        noise_variance = float(noise_variance) * self.noise_variance_scale
        event["noise_variance_used"] = noise_variance
        rays = sample_random_rays(
            self.estimator_split,
            ray_count=int(self.spec.get("estimator_rays", 1 << 21)),
            detector_h=self.detector_h,
            detector_w=self.detector_w,
            samples_per_ray=self.samples_per_ray,
            seed=self.seed + 1000 * self.round_index,
            chunk_rays=self.chunk_rays,
            materialize_points=self.materialize_points,
        )
        eta, exposure, summary = attributed_excess_residual(
            model,
            rays,
            noise_variance=noise_variance,
            weight_kwargs=weight_kwargs,
            max_buffer_mb=float(self.spec.get("estimator_max_buffer_mb", 512.0)),
        )
        event.update(summary)
        resolvable = resolvable_mask(
            model,
            detector_h=self.detector_h,
            detector_w=self.detector_w,
            min_footprint_px=float(self.spec.get("min_footprint_px", 1.0)),
        )
        supported = (exposure > 0) & (eta > 0)
        eligible = supported & resolvable
        event["filtered_unresolvable"] = int((supported & ~resolvable).sum().item())
        candidate_ids = torch.nonzero(eligible, as_tuple=False).reshape(-1)
        event["eligible_count"] = int(candidate_ids.numel())
        event["eligible_by_level"] = _level_histogram(model.leaf_levels[candidate_ids], level_count)
        if candidate_ids.numel() == 0:
            return self._stop(event, "noise_floor_reached")

        bytes_before = codec_bytes(model, self.codec)
        delta = split_delta_bytes(self.codec)
        remaining = float("inf") if self.byte_budget is None else float(self.byte_budget - bytes_before)
        if remaining < delta:
            event["codec_bytes"] = bytes_before
            return self._stop(event, "byte_budget_exhausted")
        max_parents = None
        if model.max_leaf_count is not None:
            max_parents = (int(model.max_leaf_count) - int(model.leaf_levels.shape[0])) // 7
        marked_ids, mark = dorfler_mark(
            eta[candidate_ids],
            candidate_ids,
            delta_bytes=delta,
            theta=self.theta,
            remaining_bytes=remaining,
            max_parents=max_parents,
        )
        event.update(mark)
        event["eta_total_eligible"] = float(eta[candidate_ids].sum().item())
        if marked_ids.numel() == 0:
            return self._stop(event, "no_affordable_split")

        rss_before = None
        diagnostic_before = None
        pre_state = None
        if self.rollback_enabled:
            pre_state = self.clone_state(model)
            rss_before = float(fixed_rss(model, self.dev_rays)["rss"])
            diagnostic_before = float(fixed_rss(model, self.diagnostic_rays)["rss"])
        marked_by_level = _level_histogram(model.leaf_levels[marked_ids], level_count)
        model.split_leaves_batch(marked_ids)
        if diagnostic_before is not None:
            diagnostic_after = float(fixed_rss(model, self.diagnostic_rays)["rss"])
            event["split_function_error"] = abs(diagnostic_after - diagnostic_before) / max(diagnostic_before, 1.0e-30)
        if self.rollback_enabled:
            self.pending = {
                "pre_state": pre_state,
                "marked_ids": marked_ids,
                "prefix": int(marked_ids.numel()),
                "halvings": 0,
                "rss_before": rss_before,
            }
        event.update(
            active_splits=int(marked_ids.numel()),
            marked_by_level=marked_by_level,
            codec_bytes_before=bytes_before,
            codec_bytes=codec_bytes(model, self.codec),
            leaf_count=int(model.leaf_levels.shape[0]),
        )
        return int(marked_ids.numel()), event


def quantize_roundtrip_(model, codec: dict) -> None:
    """Round the deployed parameters through the codec's storage precision."""
    dtype = torch.float32 if str(codec.get("quantization", "float16")).lower() == "float32" else torch.float16
    if dtype == torch.float32:
        return
    with torch.no_grad():
        if hasattr(model, "leaf_latents"):
            model.leaf_latents.copy_(model.leaf_latents.to(dtype).float())
            model.decoder.weight.copy_(model.decoder.weight.to(dtype).float())
            model.decoder.bias.copy_(model.decoder.bias.to(dtype).float())
        else:
            physical = model.coefficients().to(dtype).float()
            model.coefficient_logits.copy_(model._coefficients_to_logits(physical))


def finetune_model(
    model,
    split,
    *,
    steps: int,
    lr: float,
    batch_rays: int,
    detector_h: int,
    detector_w: int,
    samples_per_ray: int,
    weight_kwargs: dict,
    materialize_points: bool,
) -> dict:
    """Plain projection-MSE fine-tuning of every trainable parameter."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=float(lr))
    last = float("nan")
    for _ in range(int(steps)):
        batch = random_training_rays(
            split,
            batch_rays=int(batch_rays),
            detector_h=detector_h,
            detector_w=detector_w,
            samples_per_ray=samples_per_ray,
            materialize_points=materialize_points,
        )
        loss = weighted_projection_loss(model.integrate_ray_batch(batch), batch.target, **weight_kwargs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last = float(loss.detach().item())
    return {"steps": int(steps), "lr": float(lr), "last_loss": last}
