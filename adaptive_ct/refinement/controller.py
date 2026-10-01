"""Plateau-triggered Dorfler h-refinement controller (estimate, mark, split, verify)."""

from __future__ import annotations

import torch

from adaptive_ct.codec.budget import codec_bytes, split_delta_bytes
from adaptive_ct.data.rays import fixed_rss, head_rays, sample_random_rays
from adaptive_ct.refinement.estimator import attributed_excess_residual, resolvable_mask
from adaptive_ct.refinement.marking import dorfler_mark
from adaptive_ct.representation.snapshot import clone_sparse_state, restore_sparse_state


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
                restore_sparse_state(model, pending["pre_state"])
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
            pre_state = clone_sparse_state(model)
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
