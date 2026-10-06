"""Structure update on a solved tree: score leaves, select a batch, split it (function preserving)."""

from __future__ import annotations

import torch

from adaptive_ct.codec.budget import codec_bytes, split_delta_bytes
from adaptive_ct.data.rays import sample_random_rays
from adaptive_ct.refinement.estimator import attributed_residual_energy, resolvable_mask
from adaptive_ct.refinement.marking import mark_by_gain_share


def _level_histogram(levels: torch.Tensor, level_count: int) -> list[int]:
    return torch.bincount(levels, minlength=level_count).tolist() if levels.numel() else [0] * level_count


class RefinementController:
    """Called once per round, after the fixed-tree solve has converged."""

    def __init__(self, *, spec, estimator_split, detector_h, detector_w, chunk_rays, seed, byte_budget):
        self.spec = dict(spec)
        self.codec = dict(spec["codec"])
        self.share = float(spec.get("gain_share", 0.5))
        if not 0.0 < self.share <= 1.0:
            raise ValueError("refinement.gain_share must be in (0, 1].")
        self.estimator_split = estimator_split
        self.detector_h, self.detector_w = int(detector_h), int(detector_w)
        self.chunk_rays = int(chunk_rays)
        self.seed = int(seed)
        self.byte_budget = int(byte_budget) if byte_budget is not None else None
        self.round_index = 0
        self.stopped = False
        self.stop_reasons: list[str] = []

    def _stop(self, event, reason):
        self.stopped = True
        self.stop_reasons.append(reason)
        event.update(stop=True, stop_reasons=list(self.stop_reasons))
        return 0, event

    def refine(self, model) -> tuple[int, dict]:
        self.round_index += 1
        event: dict = {"round": self.round_index, "gain_share": self.share, "codec": self.codec, "stop": False}
        level_count = len(model.level_shapes)
        rays = sample_random_rays(
            self.estimator_split,
            ray_count=int(self.spec.get("estimator_rays", 1 << 21)),
            detector_h=self.detector_h,
            detector_w=self.detector_w,
            seed=self.seed + 1000 * self.round_index,
            chunk_rays=self.chunk_rays,
        )
        eta, exposure, summary = attributed_residual_energy(
            model, rays, max_buffer_mb=float(self.spec.get("estimator_max_buffer_mb", 512.0))
        )
        event.update(summary)
        resolvable = resolvable_mask(model, detector_h=self.detector_h, detector_w=self.detector_w,
                                     min_footprint_px=float(self.spec.get("min_footprint_px", 1.0)))
        supported = (exposure > 0) & (eta > 0)
        candidate_ids = torch.nonzero(supported & resolvable, as_tuple=False).reshape(-1)
        event["filtered_unresolvable"] = int((supported & ~resolvable).sum().item())
        event["eligible_count"] = int(candidate_ids.numel())
        event["eligible_by_level"] = _level_histogram(model.leaf_levels[candidate_ids], level_count)
        if candidate_ids.numel() == 0:
            return self._stop(event, "no_splittable_leaf")
        bytes_before = codec_bytes(model, self.codec)
        delta = split_delta_bytes(self.codec)
        remaining = float("inf") if self.byte_budget is None else float(self.byte_budget - bytes_before)
        if remaining < delta:
            event["codec_bytes"] = bytes_before
            return self._stop(event, "byte_budget_exhausted")
        max_parents = None if model.max_leaf_count is None else (int(model.max_leaf_count) - int(model.leaf_levels.shape[0])) // 7
        marked_ids, mark = mark_by_gain_share(eta[candidate_ids], candidate_ids, delta_bytes=delta, theta=self.share,
                                              remaining_bytes=remaining, max_parents=max_parents)
        event.update(mark)
        if marked_ids.numel() == 0:
            return self._stop(event, "no_affordable_split")
        event["marked_by_level"] = _level_histogram(model.leaf_levels[marked_ids], level_count)
        model.split_leaves_batch(marked_ids)
        event.update(active_splits=int(marked_ids.numel()), codec_bytes_before=bytes_before,
                     codec_bytes=codec_bytes(model, self.codec), leaf_count=int(model.leaf_levels.shape[0]))
        return int(marked_ids.numel()), event
