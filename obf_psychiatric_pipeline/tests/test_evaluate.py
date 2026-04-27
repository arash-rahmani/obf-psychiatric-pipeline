"""Tests for the evaluate module."""

import numpy as np
import pytest

from obf_psychiatric_pipeline.models.evaluate import (
    macro_f1,
    bootstrap_ci,
    per_class_metrics,
    confusion_matrix_normalized,
    evaluate_predictions,
)

LABELS = ["control", "depression", "schizophrenia"]


def _perfect_predictions():
    y_true = np.array(["control"] * 10 + ["depression"] * 10 + ["schizophrenia"] * 10)
    y_pred = y_true.copy()
    y_proba = np.zeros((30, 3))
    y_proba[:10, 0] = 1.0
    y_proba[10:20, 1] = 1.0
    y_proba[20:, 2] = 1.0
    return y_true, y_pred, y_proba


def test_macro_f1_perfect():
    y_true, y_pred, _ = _perfect_predictions()
    assert macro_f1(y_true, y_pred) == pytest.approx(1.0)


def test_bootstrap_ci_perfect_predictions():
    y_true, y_pred, _ = _perfect_predictions()
    low, high = bootstrap_ci(y_true, y_pred, n_resamples=100, seed=0)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_bootstrap_ci_bounds():
    rng = np.random.default_rng(0)
    y_true = rng.choice(LABELS, size=60)
    y_pred = rng.choice(LABELS, size=60)
    low, high = bootstrap_ci(y_true, y_pred, n_resamples=200, seed=0)
    assert 0.0 <= low <= high <= 1.0


def test_confusion_matrix_rows_sum_to_one():
    y_true, y_pred, _ = _perfect_predictions()
    cm = confusion_matrix_normalized(y_true, y_pred, LABELS)
    for row in cm["matrix"]:
        assert sum(row) == pytest.approx(1.0)


def test_evaluate_predictions_keys():
    y_true, y_pred, y_proba = _perfect_predictions()
    result = evaluate_predictions(y_true, y_pred, y_proba,
                                  LABELS, n_bootstrap=50)
    for key in ("macro_f1", "ci_low", "ci_high",
                "per_class", "confusion_matrix", "roc_auc"):
        assert key in result