"""
Load and downsample scalar metrics from TensorBoard event files.
"""

import os
from collections import defaultdict


def load_tfevents(logdir: str) -> dict[str, list[tuple[int, float]]]:
    """
    Recursively walk logdir, load all EventAccumulators,
    return { tag: [(step, value), ...] } sorted by step.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    merged: dict[str, list[tuple[int, float]]] = defaultdict(list)

    for root, dirs, files in os.walk(logdir):
        for fname in files:
            if not fname.startswith("events.out.tfevents"):
                continue
            path = os.path.join(root, fname)
            try:
                ea = EventAccumulator(path, size_guidance={"scalars": 0})
                ea.Reload()
                for tag in ea.Tags().get("scalars", []):
                    for e in ea.Scalars(tag):
                        merged[tag].append((e.step, e.value))
            except Exception:
                continue  # skip corrupt / unreadable files

    return {
        tag: sorted(set(pts), key=lambda x: x[0])
        for tag, pts in merged.items()
    }


def downsample(
    series: list[tuple[int, float]],
    max_pts: int = 200,
) -> list[tuple[int, float]]:
    if len(series) <= max_pts:
        return series
    step = len(series) / max_pts
    return [series[int(i * step)] for i in range(max_pts)]


def summarise(series: list[tuple[int, float]], pts: int = 100) -> dict:
    """Return a compact stat summary used in prompt construction."""
    ds = downsample(series, pts)
    vals = [v for _, v in ds]
    recent = vals[int(len(vals) * 0.9):]
    return {
        "step_start": ds[0][0],
        "step_end": ds[-1][0],
        "min": min(vals),
        "max": max(vals),
        "first": vals[0],
        "last": vals[-1],
        "recent_delta": recent[-1] - recent[0] if len(recent) > 1 else 0.0,
    }
