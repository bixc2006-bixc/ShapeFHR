from pathlib import Path

import numpy as np
from datetime import datetime

from config import (
    MODEL,
    OUTPUT_ROOT,
    PRIMARY_SEED,
    PRIVATE_I,
    TRAIN_CONFIGS,
    VIS_ABNORMAL_SEARCH_ALL_POSITIVES,
    VIS_DECEL_ANCHOR_TOLERANCE_SEC,
    VIS_DECEL_BASELINE_DIFF_MAX_BPM,
    VIS_DECEL_CONTEXT_MARGIN_SEC,
    VIS_DECEL_MAX_DROP_BPM,
    VIS_DECEL_MAX_DURATION_SEC,
    VIS_DECEL_MAX_RAW_STEP_BPM,
    VIS_DECEL_MAX_SMOOTH_STEP_BPM,
    VIS_DECEL_MAX_VALID_BPM,
    VIS_DECEL_MIN_DROP_BPM,
    VIS_DECEL_MIN_DURATION_SEC,
    VIS_DECEL_MIN_SIDE_SEC,
    VIS_DECEL_MIN_VALID_BPM,
    VIS_DECEL_RECOVERY_MARGIN_BPM,
    VIS_DECEL_SMOOTH_SEC,
    VIS_NORMAL_SAMPLE_NAMES,
    VIS_OUTPUT_FOLDER,
    VIS_SAMPLE_RATE_HZ,
    VIS_TN_COUNT,
    VIS_TP_COUNT,
)
from data.dataset import normalize_bpm
from data.fixed_splits import load_private_test
from experiments.common import device, load_full_source, main_checkpoint
from training.io import save_json
from training.metrics import metrics_from_scores
from training.trainer import predict_scores_array
from visualization.matching import (
    detect_complete_accelerations_whole_signal,
    detect_complete_decelerations_whole_signal,
    extract_real_complete_acceleration_example,
    extract_real_complete_deceleration_example,
    save_combined_match_figure,
    save_match_figure,
)


# -----------------------------------------------------------------------------
# Visualization-only acceleration criteria.
# These are kept here so only matching.py and run_visualization.py need changes.
# The strict pass prefers a clear complete acceleration rather than a short spike.
# -----------------------------------------------------------------------------
VIS_ACCEL_MIN_RISE_BPM = 15.0
VIS_ACCEL_MAX_RISE_BPM = 45.0
VIS_ACCEL_MIN_DURATION_SEC = 25.0
VIS_ACCEL_MAX_DURATION_SEC = 120.0
VIS_ACCEL_MIN_SIDE_SEC = 5.0
VIS_ACCEL_BASELINE_DIFF_MAX_BPM = 12.0
VIS_ACCEL_RECOVERY_MARGIN_BPM = 5.0
VIS_ACCEL_CONTEXT_MARGIN_SEC = 18.0
VIS_ACCEL_SMOOTH_SEC = 2.0
VIS_ACCEL_MAX_SMOOTH_STEP_BPM = 8.0
VIS_ACCEL_ANCHOR_TOLERANCE_SEC = 20.0

# Limit the expensive model-level matching to the most standard-looking raw
# events after a cheap morphology pre-screen. This keeps the rerun practical.
VIS_MAX_NORMAL_EVENT_CANDIDATES = 120
VIS_MAX_ABNORMAL_EVENT_CANDIDATES = 120


def _stem(path_str: str) -> str:
    return Path(path_str).stem


def _preferred_normal_set() -> set[str]:
    return {str(name) for name in VIS_NORMAL_SAMPLE_NAMES}


def _rank_normal_candidates_by_event_quality(
    *,
    file_names: list[str],
    x_bpm: np.ndarray,
    y_true: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray,
) -> list[int]:
    """Pre-screen TN samples for complete acceleration morphology.

    The normal examples are no longer selected only by classification confidence.
    We first search for a complete acceleration with a clear rise, peak, recovery,
    and stable baseline. This makes the displayed normal examples much more
    standard and avoids tiny partial events.
    """
    tn = np.flatnonzero((y_true == 0) & (pred == 0))
    preferred = _preferred_normal_set()
    ranked: list[tuple[float, int]] = []

    for idx in tn:
        idx = int(idx)
        events = detect_complete_accelerations_whole_signal(
            x_bpm[idx],
            sample_rate_hz=VIS_SAMPLE_RATE_HZ,
            min_rise_bpm=VIS_ACCEL_MIN_RISE_BPM,
            max_rise_bpm=VIS_ACCEL_MAX_RISE_BPM,
            min_duration_sec=VIS_ACCEL_MIN_DURATION_SEC,
            max_duration_sec=VIS_ACCEL_MAX_DURATION_SEC,
            min_side_sec=VIS_ACCEL_MIN_SIDE_SEC,
            baseline_diff_max_bpm=VIS_ACCEL_BASELINE_DIFF_MAX_BPM,
            recovery_margin_bpm=VIS_ACCEL_RECOVERY_MARGIN_BPM,
            smooth_sec=VIS_ACCEL_SMOOTH_SEC,
            min_valid_bpm=VIS_DECEL_MIN_VALID_BPM,
            max_valid_bpm=VIS_DECEL_MAX_VALID_BPM,
            max_raw_step_bpm=VIS_DECEL_MAX_RAW_STEP_BPM,
            max_smooth_step_bpm=VIS_ACCEL_MAX_SMOOTH_STEP_BPM,
        )
        if not events:
            continue

        best_quality = float(events[0]["quality_score"])
        sample_name = _stem(file_names[idx])

        # Keep a small preference for the two previously approved TN examples,
        # but only when they also contain a complete acceleration.
        preferred_bonus = 4.0 if sample_name in preferred else 0.0

        # A small confidence term breaks ties without dominating morphology.
        confidence_bonus = 0.5 * float(1.0 - scores[idx])
        ranked.append((best_quality + preferred_bonus + confidence_bonus, idx))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in ranked[:VIS_MAX_NORMAL_EVENT_CANDIDATES]]


def _abnormal_label_name(pred_value: int) -> str:
    return "Abnormal TP" if int(pred_value) == 1 else "Abnormal sample"


def _rank_abnormal_candidates_by_event_quality(
    *,
    file_names: list[str],
    x_bpm: np.ndarray,
    y_true: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray,
) -> list[int]:
    """Pre-screen abnormal samples for complete deceleration morphology."""
    positives = np.flatnonzero(y_true == 1)
    if not VIS_ABNORMAL_SEARCH_ALL_POSITIVES:
        positives = positives[pred[positives] == 1]

    ranked: list[tuple[float, int]] = []
    for idx in positives:
        idx = int(idx)
        events = detect_complete_decelerations_whole_signal(
            x_bpm[idx],
            sample_rate_hz=VIS_SAMPLE_RATE_HZ,
            min_drop_bpm=VIS_DECEL_MIN_DROP_BPM,
            max_drop_bpm=VIS_DECEL_MAX_DROP_BPM,
            # Prefer longer, visually complete events during the pre-screen.
            min_duration_sec=max(30.0, VIS_DECEL_MIN_DURATION_SEC),
            max_duration_sec=VIS_DECEL_MAX_DURATION_SEC,
            min_side_sec=VIS_DECEL_MIN_SIDE_SEC,
            baseline_diff_max_bpm=VIS_DECEL_BASELINE_DIFF_MAX_BPM,
            recovery_margin_bpm=VIS_DECEL_RECOVERY_MARGIN_BPM,
            smooth_sec=VIS_DECEL_SMOOTH_SEC,
            min_valid_bpm=VIS_DECEL_MIN_VALID_BPM,
            max_valid_bpm=VIS_DECEL_MAX_VALID_BPM,
            max_raw_step_bpm=VIS_DECEL_MAX_RAW_STEP_BPM,
            max_smooth_step_bpm=VIS_DECEL_MAX_SMOOTH_STEP_BPM,
        )
        if not events:
            continue

        best_quality = float(events[0]["quality_score"])
        tp_bonus = 3.0 if int(pred[idx]) == 1 else 0.0
        confidence_bonus = 0.5 * float(scores[idx])
        ranked.append((best_quality + tp_bonus + confidence_bonus, idx))

    ranked.sort(key=lambda item: item[0], reverse=True)

    # If the strict morphology pre-screen finds too few samples, append the
    # remaining positives in model-score order for the relaxed passes below.
    chosen = [idx for _, idx in ranked[:VIS_MAX_ABNORMAL_EVENT_CANDIDATES]]
    used = set(chosen)
    tp = positives[pred[positives] == 1]
    fn = positives[pred[positives] == 0]
    tp_order = tp[np.argsort(scores[tp])[::-1]] if tp.size else np.empty(0, dtype=int)
    fn_order = fn[np.argsort(scores[fn])[::-1]] if fn.size else np.empty(0, dtype=int)
    fallback = np.concatenate([tp_order, fn_order])
    for idx in fallback:
        idx = int(idx)
        if idx in used:
            continue
        chosen.append(idx)
        used.add(idx)
        if len(chosen) >= VIS_MAX_ABNORMAL_EVENT_CANDIDATES:
            break
    return chosen


def _take_best_results(
    candidates: list[tuple[dict, dict]],
    count: int,
) -> list[tuple[dict, dict]]:
    candidates.sort(
        key=lambda pair: (
            float(pair[0].get("event_quality_score", -np.inf)),
            float(pair[0].get("best_similarity", -np.inf)),
        ),
        reverse=True,
    )
    return candidates[:count]


def _select_abnormal_examples(
    *,
    model,
    payload: dict,
    x_model: np.ndarray,
    x_bpm: np.ndarray,
    y_true: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray,
    file_names: list[str],
    dev,
) -> list[tuple[dict, dict]]:
    selected: list[tuple[dict, dict]] = []
    used_samples: set[str] = set()
    candidate_order = _rank_abnormal_candidates_by_event_quality(
        file_names=file_names,
        x_bpm=x_bpm,
        y_true=y_true,
        pred=pred,
        scores=scores,
    )

    def try_pass(
        *,
        pass_name: str,
        anchor_tolerance_sec: float,
        require_anchor_relation: bool,
        min_drop_bpm: float,
        min_duration_sec: float,
        max_smooth_step_bpm: float,
    ) -> None:
        nonlocal selected
        need = VIS_TP_COUNT - len(selected)
        if need <= 0:
            return

        valid: list[tuple[dict, dict]] = []
        for idx in candidate_order:
            idx = int(idx)
            sample_name = _stem(file_names[idx])
            if sample_name in used_samples:
                continue

            result = extract_real_complete_deceleration_example(
                model=model,
                checkpoint_payload=payload,
                x_model=x_model[idx, 0],
                x_bpm=x_bpm[idx],
                sample_name=sample_name,
                label_name=_abnormal_label_name(pred[idx]),
                device=dev,
                sample_rate_hz=VIS_SAMPLE_RATE_HZ,
                min_drop_bpm=min_drop_bpm,
                max_drop_bpm=VIS_DECEL_MAX_DROP_BPM,
                min_duration_sec=min_duration_sec,
                max_duration_sec=VIS_DECEL_MAX_DURATION_SEC,
                min_side_sec=VIS_DECEL_MIN_SIDE_SEC,
                baseline_diff_max_bpm=VIS_DECEL_BASELINE_DIFF_MAX_BPM,
                recovery_margin_bpm=VIS_DECEL_RECOVERY_MARGIN_BPM,
                context_margin_sec=VIS_DECEL_CONTEXT_MARGIN_SEC,
                smooth_sec=VIS_DECEL_SMOOTH_SEC,
                min_valid_bpm=VIS_DECEL_MIN_VALID_BPM,
                max_valid_bpm=VIS_DECEL_MAX_VALID_BPM,
                max_raw_step_bpm=VIS_DECEL_MAX_RAW_STEP_BPM,
                max_smooth_step_bpm=max_smooth_step_bpm,
                anchor_tolerance_sec=anchor_tolerance_sec,
                require_anchor_relation=require_anchor_relation,
            )
            if result is None:
                continue

            record, _ = result
            record["search_pass"] = pass_name
            valid.append(result)

        for result in _take_best_results(valid, need):
            record, _ = result
            used_samples.add(str(record["sample_name"]))
            selected.append(result)
            print(
                "[visualization] selected complete deceleration: "
                f"{record['sample_name']} "
                f"drop={record['deceleration_drop_bpm']:.1f} bpm "
                f"duration={record['deceleration_duration_sec']:.1f}s "
                f"similarity={record['best_similarity']:.4f} "
                f"mode={record['selection_detail']} "
                f"pass={pass_name}"
            )

    # Pass 1: prefer a clearly complete and longer deceleration with the model
    # anchor inside or near the event.
    try_pass(
        pass_name="standard_complete_deceleration",
        anchor_tolerance_sec=VIS_DECEL_ANCHOR_TOLERANCE_SEC,
        require_anchor_relation=True,
        min_drop_bpm=VIS_DECEL_MIN_DROP_BPM,
        min_duration_sec=max(30.0, VIS_DECEL_MIN_DURATION_SEC),
        max_smooth_step_bpm=VIS_DECEL_MAX_SMOOTH_STEP_BPM,
    )
    if len(selected) >= VIS_TP_COUNT:
        return selected

    # Pass 2: keep the same standard morphology but allow a wider model-anchor
    # neighborhood.
    try_pass(
        pass_name="standard_wider_anchor",
        anchor_tolerance_sec=max(60.0, VIS_DECEL_ANCHOR_TOLERANCE_SEC),
        require_anchor_relation=True,
        min_drop_bpm=VIS_DECEL_MIN_DROP_BPM,
        min_duration_sec=max(30.0, VIS_DECEL_MIN_DURATION_SEC),
        max_smooth_step_bpm=VIS_DECEL_MAX_SMOOTH_STEP_BPM,
    )
    if len(selected) >= VIS_TP_COUNT:
        return selected

    # Pass 3: retain a standard real complete deceleration even if the nearest
    # abnormal-shapelet anchor is not inside the event.
    try_pass(
        pass_name="standard_real_event_nearest_anchor",
        anchor_tolerance_sec=max(120.0, VIS_DECEL_ANCHOR_TOLERANCE_SEC),
        require_anchor_relation=False,
        min_drop_bpm=VIS_DECEL_MIN_DROP_BPM,
        min_duration_sec=max(30.0, VIS_DECEL_MIN_DURATION_SEC),
        max_smooth_step_bpm=VIS_DECEL_MAX_SMOOTH_STEP_BPM,
    )
    if len(selected) >= VIS_TP_COUNT:
        return selected

    # Pass 4: relax duration slightly but still require a complete real event.
    try_pass(
        pass_name="complete_deceleration_relaxed_duration",
        anchor_tolerance_sec=max(120.0, VIS_DECEL_ANCHOR_TOLERANCE_SEC),
        require_anchor_relation=False,
        min_drop_bpm=max(12.0, VIS_DECEL_MIN_DROP_BPM - 3.0),
        min_duration_sec=max(22.0, VIS_DECEL_MIN_DURATION_SEC),
        max_smooth_step_bpm=VIS_DECEL_MAX_SMOOTH_STEP_BPM + 2.0,
    )
    if len(selected) >= VIS_TP_COUNT:
        return selected

    raise RuntimeError(
        f"Only {len(selected)} real complete deceleration example(s) were found "
        f"after all real-data passes, but VIS_TP_COUNT={VIS_TP_COUNT}."
    )


def _select_normal_examples(
    *,
    model,
    payload: dict,
    x_model: np.ndarray,
    x_bpm: np.ndarray,
    y_true: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray,
    file_names: list[str],
    dev,
) -> list[tuple[dict, dict]]:
    """Select complete, standard-looking real accelerations from TN samples."""
    selected: list[tuple[dict, dict]] = []
    used_samples: set[str] = set()
    candidate_order = _rank_normal_candidates_by_event_quality(
        file_names=file_names,
        x_bpm=x_bpm,
        y_true=y_true,
        pred=pred,
        scores=scores,
    )

    if not candidate_order:
        raise RuntimeError(
            "No TN sample containing a complete acceleration was found under the "
            "strict morphology pre-screen."
        )

    def try_pass(
        *,
        pass_name: str,
        anchor_tolerance_sec: float,
        require_anchor_relation: bool,
        min_rise_bpm: float,
        min_duration_sec: float,
        max_smooth_step_bpm: float,
    ) -> None:
        nonlocal selected
        need = VIS_TN_COUNT - len(selected)
        if need <= 0:
            return

        valid: list[tuple[dict, dict]] = []
        for idx in candidate_order:
            idx = int(idx)
            sample_name = _stem(file_names[idx])
            if sample_name in used_samples:
                continue

            result = extract_real_complete_acceleration_example(
                model=model,
                checkpoint_payload=payload,
                x_model=x_model[idx, 0],
                x_bpm=x_bpm[idx],
                sample_name=sample_name,
                label_name="Normal TN",
                device=dev,
                sample_rate_hz=VIS_SAMPLE_RATE_HZ,
                min_rise_bpm=min_rise_bpm,
                max_rise_bpm=VIS_ACCEL_MAX_RISE_BPM,
                min_duration_sec=min_duration_sec,
                max_duration_sec=VIS_ACCEL_MAX_DURATION_SEC,
                min_side_sec=VIS_ACCEL_MIN_SIDE_SEC,
                baseline_diff_max_bpm=VIS_ACCEL_BASELINE_DIFF_MAX_BPM,
                recovery_margin_bpm=VIS_ACCEL_RECOVERY_MARGIN_BPM,
                context_margin_sec=VIS_ACCEL_CONTEXT_MARGIN_SEC,
                smooth_sec=VIS_ACCEL_SMOOTH_SEC,
                min_valid_bpm=VIS_DECEL_MIN_VALID_BPM,
                max_valid_bpm=VIS_DECEL_MAX_VALID_BPM,
                max_raw_step_bpm=VIS_DECEL_MAX_RAW_STEP_BPM,
                max_smooth_step_bpm=max_smooth_step_bpm,
                anchor_tolerance_sec=anchor_tolerance_sec,
                require_anchor_relation=require_anchor_relation,
            )
            if result is None:
                continue

            record, _ = result
            record["search_pass"] = pass_name
            valid.append(result)

        for result in _take_best_results(valid, need):
            record, _ = result
            used_samples.add(str(record["sample_name"]))
            selected.append(result)
            print(
                "[visualization] selected complete acceleration: "
                f"{record['sample_name']} "
                f"rise={record['acceleration_rise_bpm']:.1f} bpm "
                f"duration={record['acceleration_duration_sec']:.1f}s "
                f"similarity={record['best_similarity']:.4f} "
                f"mode={record['selection_detail']} "
                f"pass={pass_name}"
            )

    try_pass(
        pass_name="standard_complete_acceleration",
        anchor_tolerance_sec=VIS_ACCEL_ANCHOR_TOLERANCE_SEC,
        require_anchor_relation=True,
        min_rise_bpm=VIS_ACCEL_MIN_RISE_BPM,
        min_duration_sec=VIS_ACCEL_MIN_DURATION_SEC,
        max_smooth_step_bpm=VIS_ACCEL_MAX_SMOOTH_STEP_BPM,
    )
    if len(selected) >= VIS_TN_COUNT:
        return selected

    try_pass(
        pass_name="standard_acceleration_wider_anchor",
        anchor_tolerance_sec=max(60.0, VIS_ACCEL_ANCHOR_TOLERANCE_SEC),
        require_anchor_relation=True,
        min_rise_bpm=VIS_ACCEL_MIN_RISE_BPM,
        min_duration_sec=VIS_ACCEL_MIN_DURATION_SEC,
        max_smooth_step_bpm=VIS_ACCEL_MAX_SMOOTH_STEP_BPM,
    )
    if len(selected) >= VIS_TN_COUNT:
        return selected

    try_pass(
        pass_name="standard_acceleration_nearest_anchor",
        anchor_tolerance_sec=max(120.0, VIS_ACCEL_ANCHOR_TOLERANCE_SEC),
        require_anchor_relation=False,
        min_rise_bpm=VIS_ACCEL_MIN_RISE_BPM,
        min_duration_sec=VIS_ACCEL_MIN_DURATION_SEC,
        max_smooth_step_bpm=VIS_ACCEL_MAX_SMOOTH_STEP_BPM,
    )
    if len(selected) >= VIS_TN_COUNT:
        return selected

    # Final real-data fallback: slightly relax the rise and duration thresholds,
    # but still require an actual complete acceleration with recovery.
    try_pass(
        pass_name="complete_acceleration_relaxed_morphology",
        anchor_tolerance_sec=max(120.0, VIS_ACCEL_ANCHOR_TOLERANCE_SEC),
        require_anchor_relation=False,
        min_rise_bpm=12.0,
        min_duration_sec=18.0,
        max_smooth_step_bpm=VIS_ACCEL_MAX_SMOOTH_STEP_BPM + 2.0,
    )
    if len(selected) >= VIS_TN_COUNT:
        return selected

    raise RuntimeError(
        f"Only {len(selected)} complete real acceleration example(s) were found, "
        f"but VIS_TN_COUNT={VIS_TN_COUNT}."
    )


def _save_both(plot_data: dict, base_path: Path) -> None:
    save_match_figure(plot_data, base_path.with_suffix(".pdf"))
    save_match_figure(plot_data, base_path.with_suffix(".png"))


def main() -> None:
    dev = device()
    source_run = OUTPUT_ROOT
    checkpoint = main_checkpoint(PRIVATE_I.key, PRIMARY_SEED)

    print(f"[visualization] device     : {dev}")
    print(f"[visualization] source run : {source_run}")
    print(f"[visualization] checkpoint : {checkpoint}")

    model, payload = load_full_source(PRIVATE_I, PRIMARY_SEED, dev)
    test = load_private_test(PRIVATE_I, MODEL.seq_len)

    norm = payload["normalization"]
    x_model = normalize_bpm(test.x, norm["mean"], norm["std"])
    scores = predict_scores_array(
        model,
        x_model,
        dev,
        batch_size=TRAIN_CONFIGS[PRIVATE_I.key].batch_size,
    )
    threshold = float(payload["threshold"])
    pred = (scores >= threshold).astype(np.int64)

    rerun_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = source_run / VIS_OUTPUT_FOLDER / rerun_stamp
    root.mkdir(parents=True, exist_ok=True)
    print(f"[visualization] output     : {root}")

    records: list[dict] = []
    plot_examples: list[dict] = []

    abnormal_examples = _select_abnormal_examples(
        model=model,
        payload=payload,
        x_model=x_model,
        x_bpm=test.x,
        y_true=test.y,
        pred=pred,
        scores=scores,
        file_names=test.file_names,
        dev=dev,
    )
    for order, (record, plot_data) in enumerate(abnormal_examples, 1):
        _save_both(plot_data, root / f"abnormal_{order}")
        records.append(record)
        plot_examples.append(plot_data)

    normal_examples = _select_normal_examples(
        model=model,
        payload=payload,
        x_model=x_model,
        x_bpm=test.x,
        y_true=test.y,
        pred=pred,
        scores=scores,
        file_names=test.file_names,
        dev=dev,
    )
    for order, (record, plot_data) in enumerate(normal_examples, 1):
        _save_both(plot_data, root / f"normal_{order}")
        records.append(record)
        plot_examples.append(plot_data)

    save_combined_match_figure(plot_examples, root / "Shapelet_Matching.pdf")
    save_combined_match_figure(plot_examples, root / "Shapelet_Matching.png")

    save_json(
        {
            "source_run": str(source_run),
            "checkpoint": str(checkpoint),
            "threshold": threshold,
            "test_metrics": metrics_from_scores(test.y, scores, threshold),
            "selection_goal": (
                "Prefer complete, standard-looking real decelerations for abnormal "
                "examples and complete, standard-looking real accelerations for "
                "normal examples. Each right panel shows the learned shapelet and a "
                "real FHR context of equal length around the selected hidden feature."
            ),
            "normal_selection": {
                "preferred_samples": list(VIS_NORMAL_SAMPLE_NAMES),
                "rule": (
                    "Previously approved TN samples receive only a small preference; "
                    "they are used only when a complete acceleration is detected."
                ),
            },
            "deceleration_criteria": {
                "sample_rate_hz": VIS_SAMPLE_RATE_HZ,
                "min_drop_bpm": VIS_DECEL_MIN_DROP_BPM,
                "max_drop_bpm": VIS_DECEL_MAX_DROP_BPM,
                "preferred_min_duration_sec": max(30.0, VIS_DECEL_MIN_DURATION_SEC),
                "max_duration_sec": VIS_DECEL_MAX_DURATION_SEC,
                "min_side_sec": VIS_DECEL_MIN_SIDE_SEC,
                "baseline_diff_max_bpm": VIS_DECEL_BASELINE_DIFF_MAX_BPM,
                "recovery_margin_bpm": VIS_DECEL_RECOVERY_MARGIN_BPM,
                "anchor_tolerance_sec": VIS_DECEL_ANCHOR_TOLERANCE_SEC,
            },
            "acceleration_criteria": {
                "sample_rate_hz": VIS_SAMPLE_RATE_HZ,
                "min_rise_bpm": VIS_ACCEL_MIN_RISE_BPM,
                "max_rise_bpm": VIS_ACCEL_MAX_RISE_BPM,
                "preferred_min_duration_sec": VIS_ACCEL_MIN_DURATION_SEC,
                "max_duration_sec": VIS_ACCEL_MAX_DURATION_SEC,
                "min_side_sec": VIS_ACCEL_MIN_SIDE_SEC,
                "baseline_diff_max_bpm": VIS_ACCEL_BASELINE_DIFF_MAX_BPM,
                "recovery_margin_bpm": VIS_ACCEL_RECOVERY_MARGIN_BPM,
                "anchor_tolerance_sec": VIS_ACCEL_ANCHOR_TOLERANCE_SEC,
            },
            "examples": records,
        },
        root / "visualization_metadata.json",
    )

    print("[done] visualization-only rerun completed")
    print(f"[done] figures: {root}")


if __name__ == "__main__":
    main()
