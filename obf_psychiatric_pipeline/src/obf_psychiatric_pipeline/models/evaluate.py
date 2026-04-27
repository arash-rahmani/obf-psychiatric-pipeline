"""
Evaluation module — all metrics for the OBF classification experiments.

Design contract:
    - One function per metric.
    - evaluate_predictions() runs all metrics and returns a structured dict.
    - bootstrap_ci() resamples predictions 1000x to produce 95% CI on macro-F1.
    - No I/O here — caller handles saving.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI on macro-F1.

    Resamples (y_true, y_pred) pairs with replacement 1000 times,
    recomputes macro-F1 each time, returns (2.5th, 97.5th) percentiles.
    """
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        score = f1_score(y_true[idx], y_pred[idx],
                         average="macro", zero_division=0)
        scores.append(score)
    scores = np.array(scores)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        label: {
            "precision": float(precision[i]),
            "recall":    float(recall[i]),
            "f1":        float(f1[i]),
        }
        for i, label in enumerate(labels)
    }


def confusion_matrix_normalized(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict:
    """Row-normalized confusion matrix (each row = recall per class)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    cm_norm = cm / row_sums
    return {
        "labels": labels,
        "matrix": cm_norm.tolist(),
    }


def roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str],
) -> float:
    """ROC AUC — one-vs-rest for multiclass, standard for binary."""
    try:
        if len(labels) == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(roc_auc_score(
            y_true, y_proba, multi_class="ovr",
            average="macro", labels=labels,
        ))
    except ValueError:
        return float("nan")


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Run all metrics and return a structured result dict.

    Parameters
    ----------
    y_true : array of true class labels (strings)
    y_pred : array of predicted class labels (strings)
    y_proba : 2D array of class probabilities, shape (n_samples, n_classes)
    labels : ordered list of class names matching y_proba columns
    """
    f1 = macro_f1(y_true, y_pred)
    ci_low, ci_high = bootstrap_ci(y_true, y_pred,
                                    n_resamples=n_bootstrap, seed=seed)
    return {
        "macro_f1":        f1,
        "ci_low":          ci_low,
        "ci_high":         ci_high,
        "per_class":       per_class_metrics(y_true, y_pred, labels),
        "confusion_matrix": confusion_matrix_normalized(y_true, y_pred, labels),
        "roc_auc":         roc_auc(y_true, y_proba, labels),
    }