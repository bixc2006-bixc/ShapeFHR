from __future__ import annotations

from math import ceil, floor

import numpy as np


SAMPLE_RATE_HZ = 4.0
FHR_MIN = 50.0
FHR_MAX = 220.0

ARTIFACT_PROBS = {
    "dropout": 0.30,
    "halving": 0.22,
    "doubling": 0.22,
    "maternal_hr": 0.18,
    "spike": 0.08,
}

MAX_SPIKE_EVENTS = 3
MAX_RAMP_SAMPLES = 8
MATERNAL_BASE_RANGE = (75.0, 105.0)
MATERNAL_DRIFT_AMP_RANGE = (1.0, 4.0)
MATERNAL_DRIFT_PERIOD_SEC = (20.0, 60.0)
MATERNAL_NOISE_STD_RANGE = (0.25, 0.80)
SPIKE_MAGNITUDE_RANGE = (15.0, 35.0)
SPIKE_WIDTH_SAMPLES = (1, 3)
DROPOUT_LEN_RANGE = (8, 120)
OTHER_EVENT_LEN_RANGE = (24, 240)


def _clip_modified(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32), FHR_MIN, FHR_MAX)


def _free_intervals(occupied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boundaries = np.flatnonzero(np.diff(np.r_[False, ~occupied, False]))
    return boundaries[::2], boundaries[1::2]


def _blend_profile(length: int, max_ramp: int = MAX_RAMP_SAMPLES) -> np.ndarray:
    length = int(length)
    if length <= 2:
        return np.ones(length, dtype=np.float32)

    ramp = min(int(max_ramp), max(1, length // 4))
    alpha = np.ones(length, dtype=np.float32)
    u = np.linspace(1.0 / (ramp + 1.0), 1.0, ramp, dtype=np.float32)
    edge = (0.5 - 0.5 * np.cos(np.pi * u)).astype(np.float32)
    alpha[:ramp] = edge
    alpha[-ramp:] = edge[::-1]
    return alpha


def _linear_gap_reconstruction(original: np.ndarray, start: int, end: int) -> np.ndarray:
    n = int(original.size)
    length = int(end - start)
    if start > 0:
        left = float(original[start - 1])
    else:
        left = float(original[end] if end < n else original[0])
    if end < n:
        right = float(original[end])
    else:
        right = float(original[start - 1] if start > 0 else original[-1])
    return np.linspace(left, right, length + 2, dtype=np.float32)[1:-1]


def _apply_interval_artifact(
    *,
    original: np.ndarray,
    noisy: np.ndarray,
    kind: str,
    start: int,
    end: int,
    rng: np.random.Generator,
) -> None:
    source = np.asarray(original[start:end], dtype=np.float32)
    length = int(end - start)
    alpha = _blend_profile(length)

    if kind == "dropout":
        target = _linear_gap_reconstruction(original, start, end)
    elif kind == "halving":
        target = source * np.float32(0.5)
    elif kind == "doubling":
        target = source * np.float32(2.0)
    elif kind == "maternal_hr":
        t = np.arange(length, dtype=np.float32) / np.float32(SAMPLE_RATE_HZ)
        base = float(rng.uniform(*MATERNAL_BASE_RANGE))
        amp = float(rng.uniform(*MATERNAL_DRIFT_AMP_RANGE))
        period = float(rng.uniform(*MATERNAL_DRIFT_PERIOD_SEC))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        noise_std = float(rng.uniform(*MATERNAL_NOISE_STD_RANGE))
        target = (
            base
            + amp * np.sin(2.0 * np.pi * t / period + phase)
            + rng.normal(0.0, noise_std, size=length)
        ).astype(np.float32)
    else:
        raise ValueError(f"unknown interval artifact: {kind}")

    mixed = source + alpha * (np.asarray(target, dtype=np.float32) - source)
    noisy[start:end] = _clip_modified(mixed)


def _apply_spike(
    *,
    original: np.ndarray,
    noisy: np.ndarray,
    start: int,
    end: int,
    rng: np.random.Generator,
) -> dict:
    width = end - start
    magnitude = float(rng.uniform(*SPIKE_MAGNITUDE_RANGE))
    sign = float(rng.choice([-1.0, 1.0]))

    if width == 1:
        shape = np.ones(1, dtype=np.float32)
    else:
        center = (width - 1) / 2.0
        idx = np.arange(width, dtype=np.float32)
        shape = (1.0 - 0.65 * np.abs(idx - center) / max(center, 1.0)).astype(np.float32)

    source = np.asarray(original[start:end], dtype=np.float32)
    target = source + np.float32(sign * magnitude) * shape
    noisy[start:end] = _clip_modified(target)
    return {
        "type": "spike",
        "start": int(start),
        "end": int(end),
        "magnitude_bpm": float(sign * magnitude),
    }


def inject_artifact_noise(
    x: np.ndarray,
    *,
    coverage: tuple[float, float],
    rng: np.random.Generator,
    return_meta: bool = False,
):
    original = np.asarray(x, dtype=np.float32).reshape(-1).copy()
    noisy = original.copy()
    lo, hi = coverage
    lower, upper = ceil(lo * original.size), floor(hi * original.size)
    min_interval = DROPOUT_LEN_RANGE[0]
    if not 0 < lo <= hi < 1 or lower < 2 * min_interval or lower > upper:
        raise ValueError("Coverage must allow at least two non-overlapping intervals")

    target_n = int(rng.integers(lower, upper + 1))
    occupied = np.zeros(original.size, dtype=bool)
    events: list[dict] = []
    spike_count = 0
    remaining = target_n

    while remaining:
        starts, ends = _free_intervals(occupied)
        max_length = min(remaining, int((ends - starts).max(initial=0)))
        candidates = {}
        for kind in ARTIFACT_PROBS:
            if kind == "spike":
                if spike_count >= MAX_SPIKE_EVENTS:
                    continue
                minimum, maximum = SPIKE_WIDTH_SAMPLES
            elif kind == "dropout":
                minimum, maximum = DROPOUT_LEN_RANGE
            else:
                minimum, maximum = OTHER_EVENT_LEN_RANGE
            lengths = np.arange(minimum, min(maximum, max_length) + 1)
            # Leave room for another interval, or finish at the sampled budget.
            fits = remaining - lengths >= min_interval
            if events:
                fits |= lengths == remaining
            if fits.any():
                candidates[kind] = lengths[fits]
        if not candidates:
            raise ValueError("The sampled coverage cannot fit the available intervals")

        probabilities = np.array([ARTIFACT_PROBS[kind] for kind in candidates])
        kind = str(rng.choice(tuple(candidates), p=probabilities / probabilities.sum()))
        length = int(rng.choice(candidates[kind]))
        start_counts = np.maximum(ends - starts - length + 1, 0)
        cumulative = start_counts.cumsum()
        draw = int(rng.integers(cumulative[-1]))
        interval = int(np.searchsorted(cumulative, draw, side="right"))
        start = int(starts[interval] + draw - (cumulative[interval - 1] if interval else 0))
        end = start + length
        if kind == "spike":
            event = _apply_spike(original=original, noisy=noisy, start=start, end=end, rng=rng)
            spike_count += 1
        else:
            _apply_interval_artifact(original=original, noisy=noisy, kind=kind,
                                     start=start, end=end, rng=rng)
            event = {"type": kind, "start": start, "end": end}

        occupied[start:end] = True
        remaining -= length
        events.append(event)

    changed = ~np.isclose(noisy, original, rtol=0.0, atol=1e-6)
    diag = {
        "target_n": int(target_n),
        "mask_n": int(occupied.sum()),
        "changed_n": int(changed.sum()),
        "mask_coverage": float(occupied.mean()),
        "actual_changed_ratio": float(changed.mean()),
        "mean_abs_change_bpm": float(np.mean(np.abs(noisy - original))),
        "changed_abs_change_bpm": float(np.mean(np.abs(noisy[changed] - original[changed]))) if changed.any() else 0.0,
        "num_events": int(len(events)),
        "num_spikes": int(sum(e.get("type") == "spike" for e in events)),
    }
    if return_meta:
        return noisy.astype(np.float32), occupied, events, diag
    return noisy.astype(np.float32)
