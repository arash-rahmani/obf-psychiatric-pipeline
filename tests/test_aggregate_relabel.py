"""Tests for aggregate and relabel modules."""

import pandas as pd
import pytest

from obf_psychiatric_pipeline.models.aggregate import to_participant_level
from obf_psychiatric_pipeline.models.relabel import to_binary


def _make_features() -> pd.DataFrame:
    return pd.DataFrame({
        "user":     [1, 1, 2, 2, 3],
        "mean":     [100, 200, 300, 400, 500],
        "sd":       [10, 20, 30, 40, 50],
        "pctZeros": [5, 15, 25, 35, 45],
        "median":   [80, 180, 280, 380, 480],
        "q75":      [150, 250, 350, 450, 550],
        "class":    ["control", "control",
                     "depression", "depression",
                     "schizophrenia"],
    })


def test_to_participant_level_one_row_per_user():
    features = _make_features()
    agg = to_participant_level(features)
    assert len(agg) == 3
    assert set(agg["user"]) == {1, 2, 3}


def test_to_participant_level_averages_correctly():
    features = _make_features()
    agg = to_participant_level(features)
    user1 = agg[agg["user"] == 1].iloc[0]
    assert user1["mean"] == pytest.approx(150.0)


def test_to_participant_level_preserves_class():
    features = _make_features()
    agg = to_participant_level(features)
    assert set(agg["class"]) == {"control", "depression", "schizophrenia"}


def test_to_binary_maps_patients():
    features = _make_features()
    binary = to_binary(features)
    assert set(binary["class"]) == {"control", "patient"}


def test_to_binary_preserves_control():
    features = _make_features()
    binary = to_binary(features)
    control_rows = binary[binary["class"] == "control"]
    assert len(control_rows) == 2