"""Per-day → per-participant feature aggregation."""

from __future__ import annotations

import pandas as pd


FEATURE_COLS = ("mean", "sd", "pctZeros", "median", "q75")


def to_participant_level(features: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-day rows to one row per participant.

    Numeric features are averaged across days.
    Class label is taken as the mode (all days share the same label).

    Parameters
    ----------
    features : pd.DataFrame
        Per-day feature matrix with 'user' and 'class' columns.

    Returns
    -------
    pd.DataFrame
        One row per participant, same columns as input.
    """
    available_cols = [c for c in FEATURE_COLS if c in features.columns]

    agg_dict = {col: "mean" for col in available_cols}
    agg_dict["class"] = lambda x: x.mode()[0]

    aggregated = (
        features.groupby("user")
        .agg(agg_dict)
        .reset_index()
    )
    return aggregated