from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEFT_LEGEND_LOC = "lower left"
RIGHT_LEGEND_LOC = "lower right"
LEGEND_FRAME_ALPHA = 0.92
import numpy as np
import torch

from data.shapelets import deserialize_shapelets
from models.shapefhr import ShapeFHR


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    std = float(x.std())
    if std < 1e-8:
        return x - float(x.mean())
    return (x - float(x.mean())) / std


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    window = max(1, int(window))
    if window <= 1 or x.size <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    max_window = x.size if x.size % 2 == 1 else max(1, x.size - 1)
    window = min(window, max_window)
    if window <= 1:
        return x.copy()
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _context_bounds(center: int, length: int, total: int) -> tuple[int, int]:
    length = min(max(1, int(length)), int(total))
    start = int(center) - length // 2
    start = max(0, min(start, total - length))
    return start, start + length


def _linear_slope(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size < 3:
        return 0.0
    x = np.arange(y.size, dtype=np.float64)
    x -= x.mean()
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * (y - y.mean())) / denom)


def _signal_is_clean_enough(
    x_bpm: np.ndarray,
    *,
    min_valid_bpm: float,
    max_valid_bpm: float,
    max_raw_step_bpm: float,
) -> bool:
    x = np.asarray(x_bpm, dtype=np.float32).reshape(-1)
    if x.size < 32 or not np.all(np.isfinite(x)):
        return False
    if float(np.min(x)) < float(min_valid_bpm):
        return False
    if float(np.max(x)) > float(max_valid_bpm):
        return False
    if x.size > 1 and float(np.max(np.abs(np.diff(x)))) > float(max_raw_step_bpm):
        return False
    return True


def _candidate_event_from_nadir(
    x: np.ndarray,
    smooth: np.ndarray,
    nadir: int,
    anchor: int,
    *,
    sample_rate_hz: float,
    min_drop_bpm: float,
    max_drop_bpm: float,
    min_duration_sec: float,
    max_duration_sec: float,
    min_side_sec: float,
    baseline_diff_max_bpm: float,
    recovery_margin_bpm: float,
    max_smooth_step_bpm: float,
    require_anchor_inside: bool,
) -> dict | None:
    n = int(x.size)
    fs = float(sample_rate_hz)
    nadir = int(nadir)

    ref_gap = max(4, int(round(10.0 * fs)))
    ref_far = max(ref_gap + 8, int(round(90.0 * fs)))

    left_ref = smooth[max(0, nadir - ref_far) : max(0, nadir - ref_gap)]
    right_ref = smooth[min(n, nadir + ref_gap) : min(n, nadir + ref_far)]
    if left_ref.size < max(8, int(5 * fs)) or right_ref.size < max(8, int(5 * fs)):
        return None

    left_baseline = float(np.median(left_ref))
    right_baseline = float(np.median(right_ref))
    if abs(left_baseline - right_baseline) > float(baseline_diff_max_bpm):
        return None

    baseline = 0.5 * (left_baseline + right_baseline)
    nadir_bpm = float(smooth[nadir])
    drop = baseline - nadir_bpm
    if not np.isfinite(drop):
        return None
    if drop < float(min_drop_bpm) or drop > float(max_drop_bpm):
        return None

    return_level = baseline - float(recovery_margin_bpm)
    max_side = max(8, int(round(float(max_duration_sec) * fs)))

    left_limit = max(0, nadir - max_side)
    before = smooth[left_limit:nadir]
    left_candidates = np.flatnonzero(before >= return_level)
    if left_candidates.size == 0:
        return None
    onset = left_limit + int(left_candidates[-1])

    right_limit = min(n, nadir + max_side + 1)
    after = smooth[nadir + 1 : right_limit]
    right_candidates = np.flatnonzero(after >= return_level)
    if right_candidates.size == 0:
        return None
    end = nadir + 1 + int(right_candidates[0])

    if not (0 <= onset < nadir < end < n):
        return None

    duration_sec = (end - onset) / fs
    if duration_sec < float(min_duration_sec) or duration_sec > float(max_duration_sec):
        return None

    left_sec = (nadir - onset) / fs
    right_sec = (end - nadir) / fs
    if left_sec < float(min_side_sec) or right_sec < float(min_side_sec):
        return None

    if require_anchor_inside and not (onset <= int(anchor) <= end):
        return None

    event_raw = x[onset : end + 1]
    event_smooth = smooth[onset : end + 1]
    if event_raw.size < 16:
        return None

    if float(np.max(np.abs(np.diff(event_smooth)))) > float(max_smooth_step_bpm):
        return None

    left_half = smooth[onset : nadir + 1]
    right_half = smooth[nadir : end + 1]
    left_slope = _linear_slope(left_half)
    right_slope = _linear_slope(right_half)
    if left_slope >= -0.01 or right_slope <= 0.01:
        return None

    nadir_fraction = (nadir - onset) / max(1, end - onset)
    if nadir_fraction < 0.15 or nadir_fraction > 0.85:
        return None

    edge_window = max(4, int(round(5.0 * fs)))
    left_edge = smooth[max(0, onset - edge_window) : onset + 1]
    right_edge = smooth[end : min(n, end + edge_window + 1)]
    if left_edge.size < 2 or right_edge.size < 2:
        return None
    left_recovery = float(np.mean(left_edge))
    right_recovery = float(np.mean(right_edge))
    if left_recovery < baseline - 7.0 or right_recovery < baseline - 7.0:
        return None

    pre_window = smooth[max(0, onset - int(round(25.0 * fs))) : onset]
    post_window = smooth[end + 1 : min(n, end + 1 + int(round(25.0 * fs)))]
    pre_var = float(np.std(pre_window)) if pre_window.size > 2 else 0.0
    post_var = float(np.std(post_window)) if post_window.size > 2 else 0.0
    event_var = float(np.std(np.diff(event_smooth))) if event_smooth.size > 2 else 0.0

    if pre_var > 8.0 or post_var > 8.0:
        return None

    symmetry_penalty = abs(nadir_fraction - 0.5)
    duration_target = 60.0
    duration_penalty = abs(duration_sec - duration_target) / duration_target
    quality = (
        2.2 * float(drop)
        + 0.010 * float(end - onset)
        - 1.8 * abs(left_baseline - right_baseline)
        - 2.0 * event_var
        - 0.4 * abs(pre_var - post_var)
        - 10.0 * symmetry_penalty
        - 3.0 * duration_penalty
    )

    return {
        "start": int(onset),
        "end": int(end + 1),
        "nadir": int(nadir),
        "baseline_bpm": float(baseline),
        "left_baseline_bpm": float(left_baseline),
        "right_baseline_bpm": float(right_baseline),
        "nadir_bpm": float(nadir_bpm),
        "drop_bpm": float(drop),
        "duration_sec": float(duration_sec),
        "left_recovery_bpm": float(left_recovery),
        "right_recovery_bpm": float(right_recovery),
        "left_slope": float(left_slope),
        "right_slope": float(right_slope),
        "pre_variability": float(pre_var),
        "post_variability": float(post_var),
        "event_variability": float(event_var),
        "quality_score": float(quality),
    }


def _candidate_nadirs(
    smooth: np.ndarray,
    *,
    sample_rate_hz: float,
    max_candidates: int,
    min_separation_sec: float,
    region: tuple[int, int] | None = None,
) -> list[int]:
    n = int(smooth.size)
    lo, hi = (0, n) if region is None else (max(0, int(region[0])), min(n, int(region[1])))
    if hi - lo < 16:
        return []
    local = smooth[lo:hi]
    order = np.argsort(local)
    min_separation = max(4, int(round(float(min_separation_sec) * float(sample_rate_hz))))
    nadirs: list[int] = []
    for rel in order:
        idx = lo + int(rel)
        if all(abs(idx - prev) >= min_separation for prev in nadirs):
            nadirs.append(idx)
        if len(nadirs) >= int(max_candidates):
            break
    return nadirs


def detect_complete_deceleration_near_anchor(
    x_bpm: np.ndarray,
    anchor: int,
    *,
    sample_rate_hz: float = 4.0,
    min_drop_bpm: float = 15.0,
    max_drop_bpm: float = 60.0,
    min_duration_sec: float = 20.0,
    max_duration_sec: float = 180.0,
    min_side_sec: float = 5.0,
    baseline_diff_max_bpm: float = 12.0,
    recovery_margin_bpm: float = 5.0,
    search_radius_sec: float = 120.0,
    smooth_sec: float = 2.0,
    min_valid_bpm: float = 50.0,
    max_valid_bpm: float = 220.0,
    max_raw_step_bpm: float = 35.0,
    max_smooth_step_bpm: float = 8.0,
    require_anchor_inside: bool = True,
) -> dict | None:
    x = np.asarray(x_bpm, dtype=np.float32).reshape(-1)
    n = int(x.size)
    if n < 32:
        return None

    if not _signal_is_clean_enough(
        x,
        min_valid_bpm=min_valid_bpm,
        max_valid_bpm=max_valid_bpm,
        max_raw_step_bpm=max_raw_step_bpm,
    ):
        return None

    anchor = int(np.clip(anchor, 0, n - 1))
    smooth_window = max(3, int(round(float(sample_rate_hz) * float(smooth_sec))))
    smooth = _moving_average(x, smooth_window)

    radius = max(8, int(round(float(sample_rate_hz) * float(search_radius_sec))))
    lo = max(0, anchor - radius)
    hi = min(n, anchor + radius + 1)
    candidate_nadirs = _candidate_nadirs(
        smooth,
        sample_rate_hz=sample_rate_hz,
        max_candidates=24,
        min_separation_sec=5.0,
        region=(lo, hi),
    )

    events: list[dict] = []
    for nadir in candidate_nadirs:
        event = _candidate_event_from_nadir(
            x,
            smooth,
            nadir,
            anchor,
            sample_rate_hz=sample_rate_hz,
            min_drop_bpm=min_drop_bpm,
            max_drop_bpm=max_drop_bpm,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            min_side_sec=min_side_sec,
            baseline_diff_max_bpm=baseline_diff_max_bpm,
            recovery_margin_bpm=recovery_margin_bpm,
            max_smooth_step_bpm=max_smooth_step_bpm,
            require_anchor_inside=require_anchor_inside,
        )
        if event is not None:
            events.append(event)

    if not events:
        return None
    return max(events, key=lambda item: float(item["quality_score"]))


def detect_complete_decelerations_whole_signal(
    x_bpm: np.ndarray,
    *,
    sample_rate_hz: float = 4.0,
    min_drop_bpm: float = 15.0,
    max_drop_bpm: float = 60.0,
    min_duration_sec: float = 20.0,
    max_duration_sec: float = 180.0,
    min_side_sec: float = 5.0,
    baseline_diff_max_bpm: float = 12.0,
    recovery_margin_bpm: float = 5.0,
    smooth_sec: float = 2.0,
    min_valid_bpm: float = 50.0,
    max_valid_bpm: float = 220.0,
    max_raw_step_bpm: float = 35.0,
    max_smooth_step_bpm: float = 8.0,
) -> list[dict]:
    x = np.asarray(x_bpm, dtype=np.float32).reshape(-1)
    if x.size < 32:
        return []
    if not _signal_is_clean_enough(
        x,
        min_valid_bpm=min_valid_bpm,
        max_valid_bpm=max_valid_bpm,
        max_raw_step_bpm=max_raw_step_bpm,
    ):
        return []

    smooth_window = max(3, int(round(float(sample_rate_hz) * float(smooth_sec))))
    smooth = _moving_average(x, smooth_window)
    candidate_nadirs = _candidate_nadirs(
        smooth,
        sample_rate_hz=sample_rate_hz,
        max_candidates=48,
        min_separation_sec=8.0,
    )

    events: list[dict] = []
    used: list[int] = []
    for nadir in candidate_nadirs:
        if any(abs(nadir - prev) < int(round(12.0 * float(sample_rate_hz))) for prev in used):
            continue
        event = _candidate_event_from_nadir(
            x,
            smooth,
            nadir,
            nadir,
            sample_rate_hz=sample_rate_hz,
            min_drop_bpm=min_drop_bpm,
            max_drop_bpm=max_drop_bpm,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            min_side_sec=min_side_sec,
            baseline_diff_max_bpm=baseline_diff_max_bpm,
            recovery_margin_bpm=recovery_margin_bpm,
            max_smooth_step_bpm=max_smooth_step_bpm,
            require_anchor_inside=False,
        )
        if event is None:
            continue
        used.append(int(event["nadir"]))
        events.append(event)

    events.sort(key=lambda item: float(item["quality_score"]), reverse=True)
    return events


def _candidate_event_from_peak(
    x: np.ndarray,
    smooth: np.ndarray,
    peak: int,
    anchor: int,
    *,
    sample_rate_hz: float,
    min_rise_bpm: float,
    max_rise_bpm: float,
    min_duration_sec: float,
    max_duration_sec: float,
    min_side_sec: float,
    baseline_diff_max_bpm: float,
    recovery_margin_bpm: float,
    max_smooth_step_bpm: float,
    require_anchor_inside: bool,
) -> dict | None:
    """Return a complete acceleration candidate with rise, peak and recovery."""
    n = int(x.size)
    fs = float(sample_rate_hz)
    peak = int(peak)

    ref_gap = max(4, int(round(8.0 * fs)))
    ref_far = max(ref_gap + 8, int(round(75.0 * fs)))

    left_ref = smooth[max(0, peak - ref_far): max(0, peak - ref_gap)]
    right_ref = smooth[min(n, peak + ref_gap): min(n, peak + ref_far)]
    if left_ref.size < max(8, int(5 * fs)) or right_ref.size < max(8, int(5 * fs)):
        return None

    left_baseline = float(np.median(left_ref))
    right_baseline = float(np.median(right_ref))
    if abs(left_baseline - right_baseline) > float(baseline_diff_max_bpm):
        return None

    baseline = 0.5 * (left_baseline + right_baseline)
    peak_bpm = float(smooth[peak])
    rise = peak_bpm - baseline
    if not np.isfinite(rise):
        return None
    if rise < float(min_rise_bpm) or rise > float(max_rise_bpm):
        return None

    return_level = baseline + float(recovery_margin_bpm)
    max_side = max(8, int(round(float(max_duration_sec) * fs)))

    left_limit = max(0, peak - max_side)
    before = smooth[left_limit:peak]
    left_candidates = np.flatnonzero(before <= return_level)
    if left_candidates.size == 0:
        return None
    onset = left_limit + int(left_candidates[-1])

    right_limit = min(n, peak + max_side + 1)
    after = smooth[peak + 1:right_limit]
    right_candidates = np.flatnonzero(after <= return_level)
    if right_candidates.size == 0:
        return None
    end = peak + 1 + int(right_candidates[0])

    if not (0 <= onset < peak < end < n):
        return None

    duration_sec = (end - onset) / fs
    if duration_sec < float(min_duration_sec) or duration_sec > float(max_duration_sec):
        return None

    left_sec = (peak - onset) / fs
    right_sec = (end - peak) / fs
    if left_sec < float(min_side_sec) or right_sec < float(min_side_sec):
        return None

    if require_anchor_inside and not (onset <= int(anchor) <= end):
        return None

    event_raw = x[onset:end + 1]
    event_smooth = smooth[onset:end + 1]
    if event_raw.size < 24:
        return None
    if float(np.max(np.abs(np.diff(event_smooth)))) > float(max_smooth_step_bpm):
        return None

    left_half = smooth[onset:peak + 1]
    right_half = smooth[peak:end + 1]
    left_slope = _linear_slope(left_half)
    right_slope = _linear_slope(right_half)
    if left_slope <= 0.01 or right_slope >= -0.01:
        return None

    peak_fraction = (peak - onset) / max(1, end - onset)
    if peak_fraction < 0.20 or peak_fraction > 0.80:
        return None

    edge_window = max(4, int(round(5.0 * fs)))
    left_edge = smooth[max(0, onset - edge_window): onset + 1]
    right_edge = smooth[end: min(n, end + edge_window + 1)]
    if left_edge.size < 2 or right_edge.size < 2:
        return None

    left_recovery = float(np.mean(left_edge))
    right_recovery = float(np.mean(right_edge))
    if left_recovery > baseline + 7.0 or right_recovery > baseline + 7.0:
        return None

    pre_window = smooth[max(0, onset - int(round(25.0 * fs))): onset]
    post_window = smooth[end + 1: min(n, end + 1 + int(round(25.0 * fs)))]
    pre_var = float(np.std(pre_window)) if pre_window.size > 2 else 0.0
    post_var = float(np.std(post_window)) if post_window.size > 2 else 0.0
    event_var = float(np.std(np.diff(event_smooth))) if event_smooth.size > 2 else 0.0
    if pre_var > 8.0 or post_var > 8.0:
        return None

    symmetry_penalty = abs(peak_fraction - 0.5)
    duration_target = 45.0
    duration_penalty = abs(duration_sec - duration_target) / duration_target
    quality = (
        2.3 * float(rise)
        + 0.010 * float(end - onset)
        - 1.8 * abs(left_baseline - right_baseline)
        - 2.0 * event_var
        - 0.4 * abs(pre_var - post_var)
        - 10.0 * symmetry_penalty
        - 3.0 * duration_penalty
    )

    return {
        "start": int(onset),
        "end": int(end + 1),
        "peak": int(peak),
        "baseline_bpm": float(baseline),
        "left_baseline_bpm": float(left_baseline),
        "right_baseline_bpm": float(right_baseline),
        "peak_bpm": float(peak_bpm),
        "rise_bpm": float(rise),
        "duration_sec": float(duration_sec),
        "left_recovery_bpm": float(left_recovery),
        "right_recovery_bpm": float(right_recovery),
        "left_slope": float(left_slope),
        "right_slope": float(right_slope),
        "pre_variability": float(pre_var),
        "post_variability": float(post_var),
        "event_variability": float(event_var),
        "peak_fraction": float(peak_fraction),
        "quality_score": float(quality),
    }


def detect_complete_accelerations_whole_signal(
    x_bpm: np.ndarray,
    *,
    sample_rate_hz: float = 4.0,
    min_rise_bpm: float = 15.0,
    max_rise_bpm: float = 45.0,
    min_duration_sec: float = 20.0,
    max_duration_sec: float = 120.0,
    min_side_sec: float = 5.0,
    baseline_diff_max_bpm: float = 12.0,
    recovery_margin_bpm: float = 5.0,
    smooth_sec: float = 2.0,
    min_valid_bpm: float = 50.0,
    max_valid_bpm: float = 220.0,
    max_raw_step_bpm: float = 35.0,
    max_smooth_step_bpm: float = 8.0,
) -> list[dict]:
    x = np.asarray(x_bpm, dtype=np.float32).reshape(-1)
    if x.size < 32:
        return []
    if not _signal_is_clean_enough(
        x,
        min_valid_bpm=min_valid_bpm,
        max_valid_bpm=max_valid_bpm,
        max_raw_step_bpm=max_raw_step_bpm,
    ):
        return []

    smooth_window = max(3, int(round(float(sample_rate_hz) * float(smooth_sec))))
    smooth = _moving_average(x, smooth_window)
    order = np.argsort(smooth)[::-1]
    min_separation = max(4, int(round(8.0 * float(sample_rate_hz))))

    peaks: list[int] = []
    for idx in order:
        idx = int(idx)
        if all(abs(idx - prev) >= min_separation for prev in peaks):
            peaks.append(idx)
        if len(peaks) >= 64:
            break

    events: list[dict] = []
    used: list[int] = []
    for peak in peaks:
        if any(abs(peak - prev) < int(round(12.0 * float(sample_rate_hz))) for prev in used):
            continue
        event = _candidate_event_from_peak(
            x,
            smooth,
            peak,
            peak,
            sample_rate_hz=sample_rate_hz,
            min_rise_bpm=min_rise_bpm,
            max_rise_bpm=max_rise_bpm,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            min_side_sec=min_side_sec,
            baseline_diff_max_bpm=baseline_diff_max_bpm,
            recovery_margin_bpm=recovery_margin_bpm,
            max_smooth_step_bpm=max_smooth_step_bpm,
            require_anchor_inside=False,
        )
        if event is None:
            continue
        used.append(int(event["peak"]))
        events.append(event)

    events.sort(key=lambda item: float(item["quality_score"]), reverse=True)
    return events


def extract_match_example(
    *,
    model: ShapeFHR,
    checkpoint_payload: dict,
    x_model: np.ndarray,
    x_bpm: np.ndarray,
    sample_name: str,
    label_name: str,
    shapelet_class: int,
    device: torch.device,
) -> tuple[dict, dict]:
    tensor = torch.from_numpy(np.asarray(x_model, dtype=np.float32)[None, None, :]).to(device)
    info = model.shapelet_match_info(tensor)

    scores = info["best_scores"][0].detach().cpu().numpy()
    indices = info["best_indices"][0].detach().cpu().numpy()
    classes = info["shapelet_classes"].detach().cpu().numpy()

    candidates = np.flatnonzero(classes == int(shapelet_class))
    if candidates.size == 0:
        candidates = np.arange(len(scores))

    shapelet_idx = int(candidates[np.argmax(scores[candidates])])
    hidden_index = int(indices[shapelet_idx])
    learned_shapelet = (
        model.specific.shapelet_params[shapelet_idx]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    center = hidden_index + model.specific.kernel_size // 2
    start, end = _context_bounds(center, learned_shapelet.size, len(x_bpm))
    context = np.asarray(x_model[start:end], dtype=np.float32)

    initial_shapelets = deserialize_shapelets(checkpoint_payload["initial_shapelets"])
    record = {
        "sample_name": sample_name,
        "label_name": label_name,
        "selection_mode": "global_match",
        "shapelet_index": shapelet_idx,
        "shapelet_class": int(classes[shapelet_idx]),
        "best_hidden_index": hidden_index,
        "best_hidden_center": int(center),
        "best_similarity": float(scores[shapelet_idx]),
        "context_start": int(start),
        "context_end": int(end),
        "shapelet_length": int(learned_shapelet.size),
        "initial_shapelet_ig": float(initial_shapelets[shapelet_idx].ig),
    }
    plot_data = {
        "mode": "global_match",
        "title": f"{label_name}: {sample_name}",
        "x_bpm": np.asarray(x_bpm, dtype=np.float32),
        "context": context,
        "shapelet": learned_shapelet,
        "center": int(center),
        "context_start": int(start),
        "context_end": int(end),
    }
    return record, plot_data


def extract_real_complete_deceleration_example(
    *,
    model: ShapeFHR,
    checkpoint_payload: dict,
    x_model: np.ndarray,
    x_bpm: np.ndarray,
    sample_name: str,
    label_name: str,
    device: torch.device,
    sample_rate_hz: float,
    min_drop_bpm: float,
    max_drop_bpm: float,
    min_duration_sec: float,
    max_duration_sec: float,
    min_side_sec: float,
    baseline_diff_max_bpm: float,
    recovery_margin_bpm: float,
    context_margin_sec: float,
    smooth_sec: float,
    min_valid_bpm: float,
    max_valid_bpm: float,
    max_raw_step_bpm: float,
    max_smooth_step_bpm: float,
    anchor_tolerance_sec: float,
    require_anchor_relation: bool = True,
) -> tuple[dict, dict] | None:
    x_bpm = np.asarray(x_bpm, dtype=np.float32).reshape(-1)
    if not _signal_is_clean_enough(
        x_bpm,
        min_valid_bpm=min_valid_bpm,
        max_valid_bpm=max_valid_bpm,
        max_raw_step_bpm=max_raw_step_bpm,
    ):
        return None

    tensor = torch.from_numpy(np.asarray(x_model, dtype=np.float32)[None, None, :]).to(device)
    info = model.shapelet_match_info(tensor)

    scores = info["best_scores"][0].detach().cpu().numpy()
    indices = info["best_indices"][0].detach().cpu().numpy()
    classes = info["shapelet_classes"].detach().cpu().numpy()
    abnormal_shapelets = np.flatnonzero(classes == 1)
    if abnormal_shapelets.size == 0:
        return None

    events = detect_complete_decelerations_whole_signal(
        x_bpm,
        sample_rate_hz=sample_rate_hz,
        min_drop_bpm=min_drop_bpm,
        max_drop_bpm=max_drop_bpm,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        min_side_sec=min_side_sec,
        baseline_diff_max_bpm=baseline_diff_max_bpm,
        recovery_margin_bpm=recovery_margin_bpm,
        smooth_sec=smooth_sec,
        min_valid_bpm=min_valid_bpm,
        max_valid_bpm=max_valid_bpm,
        max_raw_step_bpm=max_raw_step_bpm,
        max_smooth_step_bpm=max_smooth_step_bpm,
    )
    if not events:
        return None

    tolerance = int(round(float(anchor_tolerance_sec) * float(sample_rate_hz)))
    event_candidates: list[tuple[float, dict, int, int, int, str, int, float]] = []

    for event in events:
        start = int(event["start"])
        end = int(event["end"])
        nadir = int(event["nadir"])
        event_center = int(round(0.5 * (start + end)))

        best_inside = None
        best_near = None
        best_nearest = None

        for shapelet_idx in abnormal_shapelets:
            hidden_index = int(indices[shapelet_idx])
            center = hidden_index + model.specific.kernel_size // 2
            similarity = float(scores[shapelet_idx])

            if start <= center <= end:
                distance = abs(center - nadir)
                key = (similarity - 0.005 * distance, similarity, -distance, int(shapelet_idx), hidden_index, center, distance)
                if best_inside is None or key > best_inside[0]:
                    best_inside = (key, int(shapelet_idx), hidden_index, center, distance, similarity)
            else:
                gap = min(abs(center - start), abs(center - end))
                if gap <= tolerance:
                    key = (similarity - 0.01 * gap, similarity, -gap, int(shapelet_idx), hidden_index, center, gap)
                    if best_near is None or key > best_near[0]:
                        best_near = (key, int(shapelet_idx), hidden_index, center, gap, similarity)
                nearest_dist = abs(center - event_center)
                key = (similarity - 0.002 * nearest_dist, similarity, -nearest_dist, int(shapelet_idx), hidden_index, center, nearest_dist)
                if best_nearest is None or key > best_nearest[0]:
                    best_nearest = (key, int(shapelet_idx), hidden_index, center, nearest_dist, similarity)

        if best_inside is not None:
            _, shapelet_idx, hidden_index, center, distance, similarity = best_inside
            selection_kind = "anchor_inside_event"
            penalty = 0.00 * distance
        elif best_near is not None:
            _, shapelet_idx, hidden_index, center, distance, similarity = best_near
            selection_kind = "anchor_near_event"
            penalty = 0.01 * distance
        elif (not require_anchor_relation) and best_nearest is not None:
            _, shapelet_idx, hidden_index, center, distance, similarity = best_nearest
            selection_kind = "real_event_nearest_abnormal_anchor"
            penalty = 0.004 * distance
        else:
            continue

        score = (
            1.25 * float(event["quality_score"])
            + 0.15 * float(event["drop_bpm"])
            + 0.10 * float(event["duration_sec"])
            + 0.08 * float(similarity)
            - float(penalty)
        )
        event_candidates.append(
            (
                float(score),
                event,
                int(shapelet_idx),
                int(hidden_index),
                int(center),
                selection_kind,
                int(distance),
                float(similarity),
            )
        )

    if not event_candidates:
        return None

    event_candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, event, shapelet_idx, hidden_index, center, selection_kind, distance, best_similarity = event_candidates[0]

    learned_shapelet = (
        model.specific.shapelet_params[shapelet_idx]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    initial_shapelets = deserialize_shapelets(checkpoint_payload["initial_shapelets"])

    start = int(event["start"])
    end = int(event["end"])
    margin = int(round(float(context_margin_sec) * float(sample_rate_hz)))
    context_start = max(0, start - margin)
    context_end = min(len(x_bpm), end + margin)

    # ------------------------------------------------------------------
    # Right-panel display window
    # ------------------------------------------------------------------
    # The normal examples use a roughly 380-400 point context.  To make the
    # abnormal examples visually comparable, do not plot only the short core
    # deceleration event.  Instead, keep 400 REAL FHR samples around the whole
    # deceleration, so the right panel also contains pre-deceleration baseline,
    # the complete descent/nadir/recovery, and post-deceleration baseline.
    plot_start, plot_end = _context_bounds(center, learned_shapelet.size, len(x_bpm))
    display_bpm = np.asarray(x_bpm[plot_start:plot_end], dtype=np.float32)

    record = {
        "sample_name": sample_name,
        "label_name": label_name,
        "selection_mode": "real_complete_deceleration",
        "selection_detail": selection_kind,
        "shapelet_index": int(shapelet_idx),
        "shapelet_class": 1,
        "best_hidden_index": int(hidden_index),
        "best_hidden_center": int(center),
        "best_similarity": float(best_similarity),
        "anchor_event_distance": int(distance),
        "shapelet_length": int(learned_shapelet.size),
        "initial_shapelet_ig": float(initial_shapelets[shapelet_idx].ig),
        "complete_deceleration": True,
        "synthetic": False,
        "display_context_start": int(context_start),
        "display_context_end": int(context_end),
        "right_plot_start": int(plot_start),
        "right_plot_end": int(plot_end),
        "right_plot_length": int(plot_end - plot_start),
        "event_quality_score": float(best_score),
        **{f"deceleration_{key}": value for key, value in event.items()},
    }

    plot_data = {
        "mode": "real_complete_deceleration",
        "title": f"{label_name}: {sample_name}",
        "x_bpm": x_bpm,
        "display_bpm": display_bpm,
        "shapelet": learned_shapelet,
        "center": int(center),
        "event_start": start,
        "event_end": end,
        "nadir": int(event["nadir"]),
        "context_start": int(context_start),
        "context_end": int(context_end),
    }
    return record, plot_data


def extract_real_complete_acceleration_example(
    *,
    model: ShapeFHR,
    checkpoint_payload: dict,
    x_model: np.ndarray,
    x_bpm: np.ndarray,
    sample_name: str,
    label_name: str,
    device: torch.device,
    sample_rate_hz: float,
    min_rise_bpm: float = 15.0,
    max_rise_bpm: float = 45.0,
    min_duration_sec: float = 20.0,
    max_duration_sec: float = 120.0,
    min_side_sec: float = 5.0,
    baseline_diff_max_bpm: float = 12.0,
    recovery_margin_bpm: float = 5.0,
    context_margin_sec: float = 18.0,
    smooth_sec: float = 2.0,
    min_valid_bpm: float = 50.0,
    max_valid_bpm: float = 220.0,
    max_raw_step_bpm: float = 35.0,
    max_smooth_step_bpm: float = 8.0,
    anchor_tolerance_sec: float = 20.0,
    require_anchor_relation: bool = True,
) -> tuple[dict, dict] | None:
    """Select a complete real acceleration associated with a normal shapelet."""
    x_bpm = np.asarray(x_bpm, dtype=np.float32).reshape(-1)
    if not _signal_is_clean_enough(
        x_bpm,
        min_valid_bpm=min_valid_bpm,
        max_valid_bpm=max_valid_bpm,
        max_raw_step_bpm=max_raw_step_bpm,
    ):
        return None

    events = detect_complete_accelerations_whole_signal(
        x_bpm,
        sample_rate_hz=sample_rate_hz,
        min_rise_bpm=min_rise_bpm,
        max_rise_bpm=max_rise_bpm,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        min_side_sec=min_side_sec,
        baseline_diff_max_bpm=baseline_diff_max_bpm,
        recovery_margin_bpm=recovery_margin_bpm,
        smooth_sec=smooth_sec,
        min_valid_bpm=min_valid_bpm,
        max_valid_bpm=max_valid_bpm,
        max_raw_step_bpm=max_raw_step_bpm,
        max_smooth_step_bpm=max_smooth_step_bpm,
    )
    if not events:
        return None

    tensor = torch.from_numpy(np.asarray(x_model, dtype=np.float32)[None, None, :]).to(device)
    info = model.shapelet_match_info(tensor)
    scores = info["best_scores"][0].detach().cpu().numpy()
    indices = info["best_indices"][0].detach().cpu().numpy()
    classes = info["shapelet_classes"].detach().cpu().numpy()
    normal_shapelets = np.flatnonzero(classes == 0)
    if normal_shapelets.size == 0:
        return None

    tolerance = int(round(float(anchor_tolerance_sec) * float(sample_rate_hz)))
    event_candidates: list[tuple[float, dict, int, int, int, str, int, float]] = []

    for event in events:
        start = int(event["start"])
        end = int(event["end"])
        peak = int(event["peak"])
        event_center = int(round(0.5 * (start + end)))
        best_inside = None
        best_near = None
        best_nearest = None

        for shapelet_idx in normal_shapelets:
            hidden_index = int(indices[shapelet_idx])
            center = hidden_index + model.specific.kernel_size // 2
            similarity = float(scores[shapelet_idx])
            if start <= center <= end:
                distance = abs(center - peak)
                key = (similarity - 0.005 * distance, similarity, -distance, int(shapelet_idx))
                if best_inside is None or key > best_inside[0]:
                    best_inside = (key, int(shapelet_idx), hidden_index, center, distance, similarity)
            else:
                gap = min(abs(center - start), abs(center - end))
                if gap <= tolerance:
                    key = (similarity - 0.01 * gap, similarity, -gap, int(shapelet_idx))
                    if best_near is None or key > best_near[0]:
                        best_near = (key, int(shapelet_idx), hidden_index, center, gap, similarity)
                nearest_dist = abs(center - event_center)
                key = (similarity - 0.002 * nearest_dist, similarity, -nearest_dist, int(shapelet_idx))
                if best_nearest is None or key > best_nearest[0]:
                    best_nearest = (key, int(shapelet_idx), hidden_index, center, nearest_dist, similarity)

        if best_inside is not None:
            _, shapelet_idx, hidden_index, center, distance, similarity = best_inside
            selection_kind = "anchor_inside_event"
            penalty = 0.0
        elif best_near is not None:
            _, shapelet_idx, hidden_index, center, distance, similarity = best_near
            selection_kind = "anchor_near_event"
            penalty = 0.01 * distance
        elif (not require_anchor_relation) and best_nearest is not None:
            _, shapelet_idx, hidden_index, center, distance, similarity = best_nearest
            selection_kind = "real_event_nearest_normal_anchor"
            penalty = 0.004 * distance
        else:
            continue

        score = (
            1.40 * float(event["quality_score"])
            + 0.18 * float(event["rise_bpm"])
            + 0.10 * float(event["duration_sec"])
            + 0.08 * float(similarity)
            - float(penalty)
        )
        event_candidates.append(
            (float(score), event, int(shapelet_idx), int(hidden_index), int(center),
             selection_kind, int(distance), float(similarity))
        )

    if not event_candidates:
        return None

    event_candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, event, shapelet_idx, hidden_index, center, selection_kind, distance, best_similarity = event_candidates[0]

    learned_shapelet = (
        model.specific.shapelet_params[shapelet_idx]
        .detach().cpu().numpy().astype(np.float32, copy=False)
    )
    initial_shapelets = deserialize_shapelets(checkpoint_payload["initial_shapelets"])

    start = int(event["start"])
    end = int(event["end"])
    margin = int(round(float(context_margin_sec) * float(sample_rate_hz)))
    context_start = max(0, start - margin)
    context_end = min(len(x_bpm), end + margin)

    plot_start, plot_end = _context_bounds(center, learned_shapelet.size, len(x_bpm))
    display_bpm = np.asarray(x_bpm[plot_start:plot_end], dtype=np.float32)

    record = {
        "sample_name": sample_name,
        "label_name": label_name,
        "selection_mode": "real_complete_acceleration",
        "selection_detail": selection_kind,
        "shapelet_index": int(shapelet_idx),
        "shapelet_class": 0,
        "best_hidden_index": int(hidden_index),
        "best_hidden_center": int(center),
        "best_similarity": float(best_similarity),
        "anchor_event_distance": int(distance),
        "shapelet_length": int(learned_shapelet.size),
        "initial_shapelet_ig": float(initial_shapelets[shapelet_idx].ig),
        "complete_acceleration": True,
        "synthetic": False,
        "display_context_start": int(context_start),
        "display_context_end": int(context_end),
        "right_plot_start": int(plot_start),
        "right_plot_end": int(plot_end),
        "right_plot_length": int(plot_end - plot_start),
        "event_quality_score": float(best_score),
        **{f"acceleration_{key}": value for key, value in event.items()},
    }

    plot_data = {
        "mode": "real_complete_acceleration",
        "title": f"{label_name}: {sample_name}",
        "x_bpm": x_bpm,
        "display_bpm": display_bpm,
        "shapelet": learned_shapelet,
        "center": int(center),
        "event_start": start,
        "event_end": end,
        "peak": int(event["peak"]),
        "context_start": int(context_start),
        "context_end": int(context_end),
    }
    return record, plot_data


def _plot_example_row(left_ax, right_ax, plot_data: dict) -> None:
    mode = str(plot_data.get("mode", "global_match"))
    x_bpm = np.asarray(plot_data["x_bpm"], dtype=np.float32)
    center = int(plot_data["center"])

    left_ax.plot(x_bpm, linewidth=0.9)
    left_ax.set_ylabel("FHR (bpm)")
    left_ax.set_title(plot_data["title"])
    left_ax.set_xlabel("Time index")

    if mode == "global_match":
        context = np.asarray(plot_data["context"], dtype=np.float32)
        shapelet = np.asarray(plot_data["shapelet"], dtype=np.float32)
        context_start = int(plot_data["context_start"])
        context_end = int(plot_data["context_end"])
        left_ax.axvspan(
            context_start,
            context_end,
            alpha=0.13,
            label="Matched region",
        )
        left_ax.axvline(center, linestyle="--", linewidth=1.0, label="Matched anchor")
        left_ax.legend(loc=LEFT_LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)

        right_ax.plot(_zscore(context), label="FHR context", linewidth=1.2)
        right_ax.plot(_zscore(shapelet), label="Learned shapelet", linewidth=1.2)
        right_ax.set_xlabel("Relative sample")
        right_ax.set_ylabel("Normalized amplitude")
        right_ax.legend(loc=RIGHT_LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
        return

    event_start = int(plot_data["event_start"])
    event_end = int(plot_data["event_end"])

    if mode == "real_complete_acceleration":
        peak = int(plot_data["peak"])
        left_ax.axvspan(event_start, event_end, alpha=0.13, label="Complete acceleration")
        left_ax.axvline(peak, linestyle=":", linewidth=1.0, label="Peak")
        left_ax.axvline(center, linestyle="--", linewidth=1.0, label="Matched anchor")
    else:
        nadir = int(plot_data["nadir"])
        left_ax.axvspan(event_start, event_end, alpha=0.13, label="Complete deceleration")
        left_ax.axvline(nadir, linestyle=":", linewidth=1.0, label="Nadir")
        left_ax.axvline(center, linestyle="--", linewidth=1.0, label="Matched anchor")
    left_ax.legend(loc=LEFT_LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)

    right_ax.plot(_zscore(plot_data["display_bpm"]), label="Matched FHR context", linewidth=1.35)
    right_ax.plot(_zscore(plot_data["shapelet"]), label="Learned shapelet", linewidth=1.25)
    right_ax.set_ylabel("Normalized amplitude")

    right_ax.set_xlabel("Relative sample")
    right_ax.legend(loc=RIGHT_LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)


def save_match_figure(
    plot_data: dict,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), constrained_layout=True)
    _plot_example_row(axes[0], axes[1], plot_data)
    fig.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def save_combined_match_figure(
    examples: list[dict],
    output_path: str | Path,
) -> None:
    if not examples:
        raise ValueError("examples is empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(examples),
        2,
        figsize=(12, 3.0 * len(examples)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, item in enumerate(examples):
        _plot_example_row(axes[row, 0], axes[row, 1], item)

    fig.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


__all__ = [
    "detect_complete_deceleration_near_anchor",
    "detect_complete_decelerations_whole_signal",
    "detect_complete_accelerations_whole_signal",
    "extract_match_example",
    "extract_real_complete_deceleration_example",
    "extract_real_complete_acceleration_example",
    "save_combined_match_figure",
    "save_match_figure",
]
