"""One structure / type decision per round, after the fixed-tree solve has converged."""

from __future__ import annotations

import math
import time

import torch

from adaptive_ct.codec.budget import model_bytes
from adaptive_ct.refinement.decision import decide_and_apply, decision_views


def _cell_keys(levels: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    return ((levels * 4096 + coords[:, 0]) * 4096 + coords[:, 1]) * 4096 + coords[:, 2]


class RefinementController:
    def __init__(self, *, spec, edge_penalty, estimator_split, detector_h, detector_w, seed, byte_budget, state_dir):
        self.spec = dict(spec)
        self.edge_penalty = dict(edge_penalty)
        self.estimator_split = estimator_split
        self.detector_h, self.detector_w = int(detector_h), int(detector_w)
        self.seed = int(seed)
        self.byte_budget = int(byte_budget)
        self.state_dir = state_dir
        self.round_index = 0
        self.stopped = False
        self.stop_reasons: list[str] = []
        self.changed_cells: torch.Tensor | None = None  # cells whose type changed in the previous round
        self.productive_keys: torch.Tensor | None = None  # keys of every cell split by gain so far
        self.split_keys = self.split_errors = None  # sorted keys of every cell split so far and its modified error then

    def _stop(self, event, reason):
        self.stopped = True
        self.stop_reasons.append(reason)
        event["stop"] = True
        return event

    def refine(self, model) -> dict:
        self.round_index += 1
        event = {"round": self.round_index}
        remaining = (self.byte_budget - model_bytes(model)) / self.byte_budget
        if remaining < float(self.spec["stop_below_remaining"]):
            return self._stop(event, "byte_budget_nearly_spent")
        frozen = None
        if self.changed_cells is not None and self.changed_cells.numel():
            frozen = torch.isin(_cell_keys(model.leaf_levels, model.leaf_coords),
                                _cell_keys(self.changed_cells[:, 0], self.changed_cells[:, 1:]))
        parent_error = torch.full((model.leaf_count,), math.inf, device=model.leaf_levels.device)
        if self.split_keys is not None:  # parent cell (level - 1, coords // 2) of each leaf, if split by a decision
            keys = _cell_keys(model.leaf_levels - 1, model.leaf_coords // 2)
            at = torch.searchsorted(self.split_keys, keys).clamp_max(self.split_keys.numel() - 1)
            hit = (self.split_keys[at] == keys) & (model.leaf_levels > 0)
            parent_error[hit] = self.split_errors[at[hit]]
        productive = None
        if self.productive_keys is not None:  # parent cell among the cells split by gain
            productive = torch.isin(_cell_keys(model.leaf_levels - 1, model.leaf_coords // 2), self.productive_keys) & (model.leaf_levels > 0)
        views = decision_views(self.estimator_split, int(self.spec["decision_views"]), self.round_index)
        self.state_dir.mkdir(parents=True, exist_ok=True)  # the last decision's input (audit/benchmark_pipeline.py)
        torch.save({"model": model.state_dict(), "round": self.round_index}, self.state_dir / "pre_decision_last.pt")
        t0 = time.perf_counter()
        # the tail: once less than spend_all_below of the budget is left, one decision spends all of it (instead of a
        # share of the rest per round, which adds rounds of a few percent of the bytes each)
        share = 1.0 if remaining < float(self.spec["spend_all_below"]) else float(self.spec["action_share"])
        event["action_share"] = share
        report, self.changed_cells, (cells, errors, by_gain) = decide_and_apply(
            model, self.estimator_split, views, detector_h=self.detector_h, detector_w=self.detector_w,
            byte_budget=self.byte_budget, frozen=frozen,
            action_share=share, min_footprint_px=float(self.spec["min_footprint_px"]),
            continuation_share=float(self.spec["continuation_share"]), error_share=float(self.spec["error_share"]),
            productive_parent=productive, parent_error=parent_error,
            penalty_weight=float(self.edge_penalty["weight"]), penalty_delta=float(self.edge_penalty["delta"]))
        event.update(report)
        if by_gain.any():
            keys = _cell_keys(cells[by_gain, 0], cells[by_gain, 1:])
            self.productive_keys = keys if self.productive_keys is None else torch.cat([self.productive_keys, keys])
        if cells.numel():
            keys = torch.cat([_cell_keys(cells[:, 0], cells[:, 1:])] + ([self.split_keys] if self.split_keys is not None else []))
            errors = torch.cat([errors] + ([self.split_errors] if self.split_errors is not None else []))
            self.split_keys, order = torch.sort(keys)
            self.split_errors = errors[order]
        event["decision_seconds"] = time.perf_counter() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # hand the decision's temporary memory back before the next solve
        event["type_change_blocked"] = int(frozen.sum()) if frozen is not None else 0
        if sum(event["actions"].values()) == 0:
            return self._stop(event, "byte_budget_spent" if event["byte_price"] > 0 else "no_beneficial_action")
        event["stop"] = False
        return event
