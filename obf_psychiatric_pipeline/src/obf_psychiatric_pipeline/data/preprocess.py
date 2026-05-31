"""
Preprocessing module — filters and cleans loaded OBF data.

Design contract:
    - One pure function: preprocess(metadata, features, cfg) -> (metadata, features)
    - Applies min-days filter at participant level across both metadata and features.
    - Drops excluded features from the feature matrix.
    - No side effects. No I/O. Data in, data out.
"""

from __future__ import annotations

import pandas as pd

from obf_psychiatric_pipeline.config import Config

COHORTS_WITH_FEATURES = ("control", "depression", "schizophrenia")


def preprocess(
    metadata: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    cfg: Config,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Apply preprocessing filters to metadata and feature matrix.

    Steps:
        1. Drop participants with fewer than min_days_per_participant days.
        2. Remove excluded features from the feature matrix.

    Parameters
    ----------
    metadata : dict[str, pd.DataFrame]
        Cohort metadata keyed by cohort name.
    features : pd.DataFrame
        Per-day feature matrix with 'user' and 'class' columns.
    cfg : Config
        Loaded pipeline config.

    Returns
    -------
    (metadata, features) : filtered copies. Originals are not modified.
    """
    min_days = cfg.preprocessing.min_days_per_participant
    excluded = cfg.preprocessing.excluded_features

    # Step 1: build set of participant numbers that meet the days threshold.
    filtered_metadata = {}
    for cohort, df in metadata.items():
        kept = df[df["days"] >= min_days].copy()
        filtered_metadata[cohort] = kept

    # Build the set of valid user IDs from cohorts that have features.
    valid_users: set[int] = set()
    for cohort in COHORTS_WITH_FEATURES:
        if cohort in filtered_metadata:
            valid_users.update(filtered_metadata[cohort]["number"].unique())

    # Step 2: filter feature matrix to valid users only.
    filtered_features = features[features["user"].isin(valid_users)].copy()

    # Step 3: drop excluded feature columns.
    cols_to_drop = [c for c in excluded if c in filtered_features.columns]
    filtered_features = filtered_features.drop(columns=cols_to_drop)

    return filtered_metadata, filtered_features