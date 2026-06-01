"""
Repeated GroupKFold evaluation and summary statistics.

The two CIs produced here are conceptually distinct and must NOT be conflated:

  fold_ci  (from evaluate_repeated)
      95% CI on mean macro-F1 across 20 fold assignments.
      Answers: how stable is performance across random splits of the same data?
      Width driven by fold-assignment variance, not sample-size uncertainty.

  bootstrap_ci  (from models.evaluate, existing pipeline)
      95% CI on the macro-F1 point estimate given n=76 participants.
      Answers: how uncertain is this estimate due to finite sample size?
      Width driven by the small n; wider is more honest.

Both belong in the paper. The fold_ci replaces the single-split result as the
headline number; the bootstrap_ci still characterises sample-size uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score

from obf_psychiatric_pipeline.cv.folds import iter_folds


def evaluate_repeated(
    X: np.ndarray,
    y: np.ndarray,
    participant_ids: np.ndarray,
    folds: dict,
    estimator: BaseEstimator,
) -> np.ndarray:
    """
    Run repeated CV and return fold-level macro-F1, shape (n_reps, n_splits).

    The estimator must include any preprocessing (e.g. StandardScaler) as a
    Pipeline so that scaling is fitted on train and applied to test within
    each fold.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    participant_ids = np.asarray(participant_ids)

    if not (X.shape[0] == len(y) == len(participant_ids)):
        raise ValueError(
            f"Inconsistent lengths: X={X.shape[0]}, y={len(y)}, "
            f"participant_ids={len(participant_ids)}"
        )

    n_reps = folds["metadata"]["n_reps"]
    n_splits = folds["metadata"]["n_splits"]
    fold_scores = np.full((n_reps, n_splits), np.nan)

    for rep_idx, fold_idx, train_mask, test_mask in iter_folds(folds, participant_ids):
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        clf = clone(estimator)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        fold_scores[rep_idx, fold_idx] = f1_score(
            y_te, y_pred, average="macro", zero_division=0
        )

    if np.any(np.isnan(fold_scores)):
        n_nan = np.sum(np.isnan(fold_scores))
        raise RuntimeError(f"{n_nan} fold(s) did not complete; check fold fixtures.")

    return fold_scores


def summarize_reps(
    rep_scores: np.ndarray,
    label: str,
) -> dict:
    """
    Compute mean, SD, and t-based 95% CI on the mean over repetitions.

    rep_scores: 1-D array of per-repetition macro-F1, shape (n_reps,).
    CI is a t-interval (df = n_reps - 1), NOT a bootstrap CI.
    """
    n = len(rep_scores)
    mean_ = float(np.mean(rep_scores))
    sd_ = float(np.std(rep_scores, ddof=1))
    se = sd_ / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))

    return {
        "label": label,
        "n_reps": n,
        "mean_f1": round(mean_, 4),
        "sd_f1": round(sd_, 4),
        "ci_lo": round(mean_ - t_crit * se, 4),
        "ci_hi": round(mean_ + t_crit * se, 4),
        "_rep_scores": rep_scores,
    }


def paired_summary(
    summary_a: dict,
    summary_b: dict,
    label: str,
) -> dict:
    """
    Paired comparison of two conditions evaluated on the same fold fixtures.

    frac_positive has 1/n_reps resolution. Descriptive only; not a p-value.
    """
    scores_a = summary_a["_rep_scores"]
    scores_b = summary_b["_rep_scores"]

    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Cannot pair: len(a)={len(scores_a)}, len(b)={len(scores_b)}"
        )

    diffs = scores_b - scores_a
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    sd_d = float(np.std(diffs, ddof=1))
    se_d = sd_d / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))

    return {
        "label": label,
        "n_reps": n,
        "mean_diff": round(mean_d, 4),
        "sd_diff": round(sd_d, 4),
        "ci_lo_diff": round(mean_d - t_crit * se_d, 4),
        "ci_hi_diff": round(mean_d + t_crit * se_d, 4),
        "frac_positive": round(float(np.mean(diffs > 0)), 3),
    }


def build_results_tables(
    all_summaries: dict[str, dict],
    paired_specs: list[tuple[str, str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build summary and paired DataFrames for CSV export.

    summary_df columns: task, feature_set, n_reps, mean_f1, sd_f1,
        ci_lo_mean, ci_hi_mean.
    paired_df columns: comparison, n_reps, mean_diff, sd_diff,
        ci_lo_diff, ci_hi_diff, frac_positive.
    """
    summary_rows = []
    for label, s in all_summaries.items():
        task, feature_set = label.split("/", 1)
        summary_rows.append({
            "task": task,
            "feature_set": feature_set,
            "n_reps": s["n_reps"],
            "mean_f1": s["mean_f1"],
            "sd_f1": s["sd_f1"],
            "ci_lo_mean": s["ci_lo"],
            "ci_hi_mean": s["ci_hi"],
        })

    paired_rows = []
    for label_a, label_b, comp_label in paired_specs:
        ps = paired_summary(all_summaries[label_a], all_summaries[label_b], comp_label)
        paired_rows.append({k: v for k, v in ps.items() if k != "_rep_scores"})

    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)
