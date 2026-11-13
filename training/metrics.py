from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    beta2 = float(beta) ** 2
    denom = beta2 * precision + recall
    if denom <= 1e-12:
        return 0.0
    return float((1.0 + beta2) * precision * recall / denom)


def metrics_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.size != y_score.size:
        raise ValueError("y_true/y_score length mismatch")

    y_pred = (y_score >= float(threshold)).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    balanced_accuracy = float((sensitivity + specificity) / 2.0)
    gmean = float(np.sqrt(max(0.0, sensitivity * specificity)))
    f2 = fbeta(precision, recall, beta=2.0)
    npv = float(tn / max(tn + fn, 1))

    try:
        auc_roc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc_roc = None
    try:
        auc_pr = float(average_precision_score(y_true, y_score))
    except ValueError:
        auc_pr = None

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "gmean": gmean,
        "qi": gmean,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "npv": npv,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.astype(int).tolist(),
    }


PAPER_METRICS = (
    "accuracy", "balanced_accuracy", "sensitivity", "specificity",
    "f2", "gmean", "auc_roc", "auc_pr",
)
