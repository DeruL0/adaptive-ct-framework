"""Convergence (plateau) detection that triggers refinement."""

from __future__ import annotations


def convergence_window_summary(
    values,
    *,
    window: int,
    relative_improvement_threshold: float,
    max_relative_regression: float = 0.005,
) -> dict:
    """Summarize whether a smoothed stochastic loss window has plateaued."""
    window = max(4, int(window))
    if len(values) < window:
        return {
            "ready": False,
            "samples": len(values),
            "window": window,
            "relative_improvement": None,
        }
    recent = list(values)[-window:]
    edge = max(1, window // 4)
    start = sum(recent[:edge]) / float(edge)
    end = sum(recent[-edge:]) / float(edge)
    relative_improvement = (start - end) / max(abs(start), 1.0e-12)
    return {
        "ready": (
            relative_improvement >= -abs(float(max_relative_regression))
            and relative_improvement < float(relative_improvement_threshold)
        ),
        "samples": len(recent),
        "window": window,
        "start_ema": float(start),
        "end_ema": float(end),
        "relative_improvement": float(relative_improvement),
    }


def plateau_ready(spec: dict, controller, values, iteration: int, last_trigger: int):
    """Return (ready, trigger_info) for the convergence-triggered refinement."""
    overrides = {k: v for k, v in controller.window_overrides().items() if v is not None}
    min_after_previous = int(
        overrides.get("min_iterations_after_previous", spec.get("min_iterations_after_previous", 0))
    )
    window = int(overrides.get("window", spec.get("window", 200)))
    summary = convergence_window_summary(
        values,
        window=window,
        relative_improvement_threshold=float(spec.get("relative_improvement", 0.002)),
        max_relative_regression=float(spec.get("max_relative_regression", 0.005)),
    )
    max_iteration = spec.get("max_iteration")
    time_ready = (
        iteration >= int(spec.get("min_iteration", 1))
        and iteration - last_trigger >= min_after_previous
    )
    forced = max_iteration is not None and iteration >= int(max_iteration)
    if not (time_ready and (summary["ready"] or forced)):
        return False, {}
    return True, {
        "trigger": "convergence" if summary["ready"] else "max_iteration",
        "convergence": summary,
        "convergence_min_iteration": int(spec.get("min_iteration", 1)),
        "convergence_max_iteration": int(max_iteration) if max_iteration is not None else None,
    }
