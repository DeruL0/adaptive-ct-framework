"""One structure / type decision per round, after the fixed-tree solve has converged."""

from __future__ import annotations

import time

from adaptive_ct.data.rays import sample_random_rays
from adaptive_ct.refinement.decision import decide_and_apply


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

    def refine(self, model) -> dict:
        self.round_index += 1
        rays = sample_random_rays(self.estimator_split, ray_count=int(self.spec.get("estimator_rays", 1 << 22)),
                                  detector_h=self.detector_h, detector_w=self.detector_w,
                                  seed=self.seed + 1000 * self.round_index, chunk_rays=8192)
        event = {"round": self.round_index}
        t0 = time.perf_counter()
        event.update(decide_and_apply(model, rays, detector_h=self.detector_h, detector_w=self.detector_w,
                                      byte_budget=self.byte_budget, quantization=self.quantization,
                                      action_share=float(self.spec.get("action_share", 0.5)),
                                      min_footprint_px=float(self.spec.get("min_footprint_px", 1.0))))
        event["decision_seconds"] = time.perf_counter() - t0
        if sum(event["actions"].values()) == 0:
            self.stopped = True
            self.stop_reasons.append("byte_budget_spent" if event["byte_price"] > 0 else "no_beneficial_action")
        event["stop"] = self.stopped
        return event
