"""Binary label transformation — 3-class → 2-class."""

from __future__ import annotations

import pandas as pd


def to_binary(features: pd.DataFrame) -> pd.DataFrame:
    """
    Merge depression and schizophrenia into a single 'patient' class.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with a 'class' column containing
        'control', 'depression', 'schizophrenia'.

    Returns
    -------
    pd.DataFrame
        Copy with 'class' mapped to 'control' or 'patient'.
    """
    df = features.copy()
    df["class"] = df["class"].map(
        lambda c: "control" if c == "control" else "patient"
    )
    return df