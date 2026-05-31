"""
Split module — participant-level cross-validation splits.

Design contract:
    - One pure function: make_splits(features, n_folds, seed) -> list of (train_idx, test_idx)
    - Splits are at the PARTICIPANT level using GroupKFold.
    - No participant ever appears in both train and test of any fold.
    - Indices are positional (iloc-compatible) into the features DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_splits(
    features: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate participant-level GroupKFold splits.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with a 'user' column identifying each participant.
    n_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of (train_indices, test_indices)
        Each element is a tuple of integer arrays indexing into features.
        Safe to use with features.iloc[train_indices].
    """
    groups = features["user"].values
    X = np.zeros(len(features))  # placeholder — GroupKFold only needs groups

    rng = np.random.default_rng(seed)
    unique_users = np.array(sorted(features["user"].unique()))
    shuffled_users = rng.permutation(unique_users)
    user_rank = {user: rank for rank, user in enumerate(shuffled_users)}
    shuffled_idx = np.argsort([user_rank[u] for u in groups], kind="stable")

    gkf = GroupKFold(n_splits=n_folds)
    splits = []
    for train_pos, test_pos in gkf.split(X[shuffled_idx], groups=groups[shuffled_idx]):
        train_idx = shuffled_idx[train_pos]
        test_idx = shuffled_idx[test_pos]
        splits.append((train_idx, test_idx))

    return splits

import json as _json


def load_splits_from_fixture(
    features: pd.DataFrame,
    fixture_path,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load pre-computed fold assignments from a JSON fixture.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with a 'user' column.
    fixture_path : Path or str
        Path to the JSON fixture file (folds_76.json or folds_77.json).

    Returns
    -------
    list of (train_indices, test_indices)
        Same format as make_splits output. Participants in the fixture
        but absent from features are silently skipped.
    """
    with open(fixture_path) as f:
        fold_data = _json.load(f)
    user_to_idx = {user: i for i, user in enumerate(features["user"].values)}
    splits = []
    for fold in fold_data:
        train_idx = np.array([user_to_idx[u] for u in fold["train"] if u in user_to_idx])
        test_idx = np.array([user_to_idx[u] for u in fold["test"] if u in user_to_idx])
        splits.append((train_idx, test_idx))
    return splits
