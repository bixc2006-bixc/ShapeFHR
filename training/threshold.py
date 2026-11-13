from __future__ import annotations

import numpy as np

from config import THRESHOLD_GRID_START, THRESHOLD_GRID_STEP, THRESHOLD_GRID_STOP
from training.metrics import metrics_from_scores


def threshold_grid() -> np.ndarray:
    return np.arange(
        THRESHOLD_GRID_START,
        THRESHOLD_GRID_STOP,
        THRESHOLD_GRID_STEP,
        dtype=np.float64,
    )


def best_gmean_operating_point(y_true, y_score) -> tuple[float, float]:
    labels = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if not labels.size or labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Validation labels and scores must be nonempty, aligned and finite")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Validation labels must be binary")

    grid = threshold_grid()
    positive = np.sort(scores[labels == 1])
    negative = np.sort(scores[labels == 0])
    tp = len(positive) - np.searchsorted(positive, grid)
    tn = np.searchsorted(negative, grid)
    fp, fn = len(negative) - tn, len(positive) - tp
    sensitivity = tp / max(len(positive), 1)
    specificity = tn / max(len(negative), 1)
    precision = tp / np.maximum(tp + fp, 1)
    denominator = 4.0 * precision + sensitivity
    f2 = np.divide(5.0 * precision * sensitivity, denominator,
                   out=np.zeros_like(precision), where=denominator > 1e-12)
    keys = zip(np.sqrt(sensitivity * specificity), (sensitivity + specificity) / 2,
               f2, (tp + tn) / (tp + tn + fp + fn), -np.abs(grid - 0.5))
    best_index = max(enumerate(keys), key=lambda entry: entry[1])[0]
    return float(grid[best_index]), float(np.sqrt(sensitivity[best_index] * specificity[best_index]))


def find_best_gmean_threshold(y_true, y_score) -> tuple[float, dict]:
    threshold, _ = best_gmean_operating_point(y_true, y_score)
    return threshold, metrics_from_scores(y_true, y_score, threshold)
