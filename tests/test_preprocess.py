"""Tests for the preprocessing module."""

import pandas as pd
import pytest

from obf_psychiatric_pipeline.data.preprocess import preprocess
from obf_psychiatric_pipeline.config import Config, DataConfig, PreprocessingConfig, SplitConfig, OutputConfig
from pathlib import Path


def _make_cfg(min_days: int = 7, excluded: list[str] = None) -> Config:
    return Config(
        data=DataConfig(root=Path("data/raw")),
        preprocessing=PreprocessingConfig(
            min_days_per_participant=min_days,
            excluded_features=excluded or ["q25"],
        ),
        split=SplitConfig(seed=42, n_folds=5),
        output=OutputConfig(
            results_dir=Path("results"),
            eda_dir=Path("results/eda"),
        ),
    )


def _make_metadata() -> dict:
    return {
        "control": pd.DataFrame({
            "number": [1, 2, 3],
            "days":   [13, 5, 10],  # user 2 below threshold
            "gender": [1, 1, 2],
            "age":    [30, 25, 35],
            "acc_time": [0.9, 0.8, 0.9],
            "cohort": ["control"] * 3,
        }),
        "depression": pd.DataFrame({
            "number": [4, 5],
            "days":   [12, 8],
            "gender": [2, 1],
            "age":    [28, 33],
            "madrs1": [20, 15],
            "madrs2": [18, 12],
            "afftype": [1, 2],
            "cohort": ["depression"] * 2,
        }),
        "schizophrenia": pd.DataFrame({
            "number": [6],
            "days":   [13],
            "gender": [1],
            "age":    [40],
            "bprs":   [45],
            "schtype": [1],
            "cohort": ["schizophrenia"],
        }),
        "adhd": pd.DataFrame({
            "number": [7], "days": [9], "gender": [1],
            "age": [22], "adhd": [1], "asrs": [55], "cohort": ["adhd"],
        }),
        "clinical": pd.DataFrame({
            "number": [8], "days": [8], "gender": [2],
            "age": [29], "adhd": [0], "cohort": ["clinical"],
        }),
    }


def _make_features() -> pd.DataFrame:
    return pd.DataFrame({
        "user":     [1, 1, 2, 3, 4, 5, 6],
        "mean":     [300, 310, 150, 280, 180, 190, 130],
        "sd":       [400, 390, 200, 350, 250, 260, 180],
        "pctZeros": [30, 28, 55, 32, 42, 40, 48],
        "median":   [100, 110, 40, 90, 50, 55, 35],
        "q25":      [0, 0, 0, 0, 0, 0, 0],
        "q75":      [500, 490, 250, 450, 300, 310, 220],
        "class":    ["control", "control", "control", "control",
                     "depression", "depression", "schizophrenia"],
    })


def test_min_days_filter_drops_correct_participants():
    """Participants below min_days threshold must be removed from metadata and features."""
    metadata = _make_metadata()
    features = _make_features()
    cfg = _make_cfg(min_days=7)

    filtered_meta, filtered_feat = preprocess(metadata, features, cfg)

    # User 2 has 5 days — must be gone from control metadata
    assert 2 not in filtered_meta["control"]["number"].values
    # User 2's feature rows must be gone
    assert 2 not in filtered_feat["user"].values
    # Users above threshold must remain
    assert 1 in filtered_feat["user"].values
    assert 3 in filtered_feat["user"].values


def test_excluded_features_are_dropped():
    """Columns listed in excluded_features must not appear in filtered features."""
    metadata = _make_metadata()
    features = _make_features()
    cfg = _make_cfg(excluded=["q25"])

    _, filtered_feat = preprocess(metadata, features, cfg)

    assert "q25" not in filtered_feat.columns
    # Other columns must survive
    assert "mean" in filtered_feat.columns
    assert "sd" in filtered_feat.columns