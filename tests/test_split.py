"""Tests for the split module."""

import numpy as np
import pandas as pd

from obf_psychiatric_pipeline.data.split import make_splits


def _make_features() -> pd.DataFrame:
    """Synthetic feature matrix: 4 participants, multiple days each."""
    rows = []
    for user_id, cohort, n_days in [(1, "control", 5), (2, "control", 4),
                                     (3, "depression", 6), (4, "schizophrenia", 5)]:
        for _ in range(n_days):
            rows.append({
                "user": user_id, "class": cohort,
                "mean": 200, "sd": 300, "pctZeros": 30,
                "median": 100, "q75": 400,
            })
    return pd.DataFrame(rows).reset_index(drop=True)


def test_no_participant_leaks_across_folds():
    """No participant may appear in both train and test of any fold."""
    features = _make_features()
    splits = make_splits(features, n_folds=4, seed=42)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_users = set(features.iloc[train_idx]["user"].unique())
        test_users = set(features.iloc[test_idx]["user"].unique())
        overlap = train_users & test_users
        assert overlap == set(), (
            f"Fold {fold_idx}: participants {overlap} appear in both train and test"
        )


def test_all_rows_covered_exactly_once():
    """Every row should appear in exactly one test fold."""
    features = _make_features()
    splits = make_splits(features, n_folds=4, seed=42)

    all_test_indices = np.concatenate([test for _, test in splits])
    assert len(all_test_indices) == len(features)
    assert len(set(all_test_indices)) == len(features)