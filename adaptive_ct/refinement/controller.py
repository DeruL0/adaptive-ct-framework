"""One structure / type decision per round, after the fixed-tree solve has converged."""

from __future__ import annotations

import time

import torch

from adaptive_ct.codec.budget import model_bytes
from adaptive_ct.refinement.decision import decide_and_apply, decision_views


def _cell_keys(levels: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    return ((levels * 4096 + coords[:, 0]) * 4096 + coords[:, 1]) * 4096 + coords[:, 2]


class RefinementController:
    def __init__(self, *, spec, estimator_split, detector_h, detector_w, seed, byte_budget):
        self.spec = dict(spec)
        self.quantization = str(spec.get("quantization", "float16"))
        self.estimator_split = estimator_split
        self.detector_h, self.detector_w = int(detector_h), int(detector_w)
        self.seed = int(seed)
        self.byte_budget = int(byte_budget)
        self.round_index = 0
        self.stopped = False
        self.stop_reasons: list[str] = []
        self.changed_cells: torch.Tensor | None = None  # cells whose type changed in the previous round

    def _stop(self, event, reason):
        self.stopped = True
        self.stop_reasons.append(reason)
        event["stop"] = True
        return event

    def refine(self, model) -> dict:
        self.round_index += 1
        event = {"round": self.round_index}
        remaining = (self.byte_budget - model_bytes(model, self.quantization)) / self.byte_budget
        if remaining < float(self.spec.get("stop_below_remaining", 0.01)):
            return self._stop(event, "byte_budget_nearly_spent")
        frozen = None
        if self.changed_cells is not None and self.changed_cells.numel():
            frozen = torch.isin(_cell_keys(model.leaf_levels, model.leaf_coords),
                                _cell_keys(self.changed_cells[:, 0], self.changed_cells[:, 1:]))
        views = decision_views(self.estimator_split, int(self.spec.get("decision_views", 16)), self.round_index)
        t0 = time.perf_counter()
        report, self.changed_cells = decide_and_apply(
            model, self.estimator_split, views, detector_h=self.detector_h, detector_w=self.detector_w,
            byte_budget=self.byte_budget, quantization=self.quantization, frozen=frozen,
            action_share=float(self.spec.get("action_share", 0.5)), min_footprint_px=float(self.spec.get("min_footprint_px", 1.0)))
        event.update(report)
        event["decision_seconds"] = time.perf_counter() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # hand the decision's temporary memory back before the next solve
        event["type_change_blocked"] = int(frozen.sum()) if frozen is not None else 0
        if sum(event["actions"].values()) == 0:
            return self._stop(event, "byte_budget_spent" if event["byte_price"] > 0 else "no_beneficial_action")
        event["stop"] = False
        return event
