"""
Tests for obf_psychiatric_pipeline.cv.runner.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from obf_psychiatric_pipeline.cv.folds import generate_repeated_folds
from obf_psychiatric_pipeline.cv.runner import (
    build_results_tables,
    evaluate_repeated,
    paired_summary,
    summarize_reps,
)

N_PARTICIPANTS = 30
N_FEATURES = 6
N_REPS = 4
N_SPLITS = 5


def _make_synthetic(n=N_PARTICIPANTS, n_features=N_FEATURES, n_classes=3, signal=2.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([i % n_classes for i in range(n)])
    X = rng.standard_normal((n, n_features))
    for c in range(n_classes):
        X[y == c, 0] += signal * c
    ids = np.array([f"p{i:03d}" for i in range(n)])
    return X, y, ids


@pytest.fixture()
def synthetic():
    return _make_synthetic()


@pytest.fixture()
def folds_fixture(synthetic):
    _, _, ids = synthetic
    return generate_repeated_folds(
        participant_ids=ids.tolist(),
        n_splits=N_SPLITS,
        n_reps=N_REPS,
        seeds=list(range(N_REPS)),
    )


@pytest.fixture()
def simple_estimator():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=500, random_state=42)),
    ])


class TestEvaluateRepeated:
    def test_output_shape(self, synthetic, folds_fixture, simple_estimator):
        X, y, ids = synthetic
        scores = evaluate_repeated(X, y, ids, folds_fixture, simple_estimator)
        assert scores.shape == (N_REPS, N_SPLITS)

    def test_scores_in_unit_interval(self, synthetic, folds_fixture, simple_estimator):
        X, y, ids = synthetic
        scores = evaluate_repeated(X, y, ids, folds_fixture, simple_estimator)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_no_nans(self, synthetic, folds_fixture, simple_estimator):
        X, y, ids = synthetic
        scores = evaluate_repeated(X, y, ids, folds_fixture, simple_estimator)
        assert not np.any(np.isnan(scores))

    def test_determinism(self, synthetic, folds_fixture, simple_estimator):
        X, y, ids = synthetic
        s1 = evaluate_repeated(X, y, ids, folds_fixture, simple_estimator)
        s2 = evaluate_repeated(X, y, ids, folds_fixture, simple_estimator)
        np.testing.assert_array_equal(s1, s2)

    def test_informative_features_beat_noise(self, folds_fixture, simple_estimator):
        X_signal, y, ids = _make_synthetic(signal=3.0, seed=0)
        X_noise = np.random.default_rng(99).standard_normal(X_signal.shape)
        s_signal = evaluate_repeated(X_signal, y, ids, folds_fixture, simple_estimator)
        s_noise = evaluate_repeated(X_noise, y, ids, folds_fixture, simple_estimator)
        assert s_signal.mean() > s_noise.mean()

    def test_raises_on_length_mismatch(self, folds_fixture, simple_estimator):
        X, y, ids = _make_synthetic()
        with pytest.raises(ValueError, match="Inconsistent lengths"):
            evaluate_repeated(X, y[:-1], ids, folds_fixture, simple_estimator)

    def test_2class_binary_labels(self, folds_fixture, simple_estimator):
        X, y3, ids = _make_synthetic()
        y2 = (y3 > 0).astype(int)
        scores = evaluate_repeated(X, y2, ids, folds_fixture, simple_estimator)
        assert scores.shape == (N_REPS, N_SPLITS)
        assert np.all((scores >= 0.0) & (scores <= 1.0))


class TestSummarizeReps:
    def test_keys_present(self):
        scores = np.array([0.7, 0.72, 0.68, 0.71, 0.69])
        s = summarize_reps(scores, "test/condition")
        for key in ("label", "n_reps", "mean_f1", "sd_f1", "ci_lo", "ci_hi"):
            assert key in s

    def test_mean_correct(self):
        scores = np.array([0.60, 0.70, 0.80])
        s = summarize_reps(scores, "lbl")
        assert abs(s["mean_f1"] - 0.70) < 1e-4

    def test_ci_contains_mean(self):
        scores = np.random.default_rng(42).uniform(0.5, 0.9, size=20)
        s = summarize_reps(scores, "lbl")
        assert s["ci_lo"] < s["mean_f1"] < s["ci_hi"]

    def test_ci_width_decreases_with_more_reps(self):
        scores_20 = np.random.default_rng(1).uniform(0.6, 0.8, 20)
        s20 = summarize_reps(scores_20, "lbl")
        s5 = summarize_reps(scores_20[:5], "lbl")
        assert (s5["ci_hi"] - s5["ci_lo"]) > (s20["ci_hi"] - s20["ci_lo"])

    def test_private_rep_scores_accessible(self):
        scores = np.linspace(0.6, 0.8, 10)
        s = summarize_reps(scores, "lbl")
        assert "_rep_scores" in s


class TestPairedSummary:
    def _make_pair(self, diff=0.05, n=20, seed=0):
        rng = np.random.default_rng(seed)
        scores_a = rng.uniform(0.65, 0.75, n)
        scores_b = scores_a + diff + rng.normal(0, 0.01, n)
        return summarize_reps(scores_a, "a"), summarize_reps(scores_b, "b")

    def test_keys_present(self):
        sa, sb = self._make_pair()
        ps = paired_summary(sa, sb, "test")
        for key in ("label", "n_reps", "mean_diff", "sd_diff",
                    "ci_lo_diff", "ci_hi_diff", "frac_positive"):
            assert key in ps

    def test_positive_diff_when_b_better(self):
        sa, sb = self._make_pair(diff=0.05)
        assert paired_summary(sa, sb, "lbl")["mean_diff"] > 0

    def test_negative_diff_when_a_better(self):
        sa, sb = self._make_pair(diff=-0.05)
        assert paired_summary(sa, sb, "lbl")["mean_diff"] < 0

    def test_frac_positive_all_ones_when_always_better(self):
        sa = summarize_reps(np.full(20, 0.65), "a")
        sb = summarize_reps(np.full(20, 0.75), "b")
        assert paired_summary(sa, sb, "lbl")["frac_positive"] == 1.0

    def test_raises_on_mismatched_lengths(self):
        sa = summarize_reps(np.ones(10), "a")
        sb = summarize_reps(np.ones(20), "b")
        with pytest.raises(ValueError, match="Cannot pair"):
            paired_summary(sa, sb, "lbl")


class TestBuildResultsTables:
    def _make_summaries(self):
        rng = np.random.default_rng(0)
        summaries = {}
        for task in ("3class", "2class"):
            for feat in ("distributional", "temporal", "combined"):
                base = 0.65 if task == "3class" else 0.78
                summaries[f"{task}/{feat}"] = summarize_reps(
                    rng.uniform(base, base + 0.1, 20), f"{task}/{feat}"
                )
        return summaries

    def test_summary_df_shape(self):
        summaries = self._make_summaries()
        paired_specs = [
            ("3class/distributional", "3class/combined", "3c"),
            ("2class/distributional", "2class/combined", "2c"),
        ]
        summary_df, paired_df = build_results_tables(summaries, paired_specs)
        assert len(summary_df) == 6
        assert len(paired_df) == 2

    def test_summary_df_columns(self):
        summary_df, _ = build_results_tables(self._make_summaries(), [])
        for col in ("task", "feature_set", "n_reps", "mean_f1", "sd_f1",
                    "ci_lo_mean", "ci_hi_mean"):
            assert col in summary_df.columns

    def test_ci_columns_named_for_distinction(self):
        summary_df, _ = build_results_tables(self._make_summaries(), [])
        assert "ci_lo_mean" in summary_df.columns
        assert "ci_lo" not in summary_df.columns
