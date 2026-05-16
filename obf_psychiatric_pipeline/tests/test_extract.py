"""
Tests for features/derived.py and features/extract.py.

Synthetic per-minute activity data is constructed using the same
hourly-pattern helper used in test_temporal_features.py.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obf_psychiatric_pipeline.data.raw_loader import ParticipantRecord
from obf_psychiatric_pipeline.features.derived import amplitude, relative_amplitude
from obf_psychiatric_pipeline.features.extract import (
    FEATURE_NAMES,
    extract_all_features,
    extract_participant_features,
)
from obf_psychiatric_pipeline.features.temporal import WindowResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_activity(n_days: int, pattern_hourly: np.ndarray) -> pd.Series:
    """n_days of per-minute activity from a 24-element hourly pattern."""
    assert len(pattern_hourly) == 24
    per_minute = np.repeat(pattern_hourly, 60).astype(float)
    start = pd.Timestamp("2020-01-01")
    idx = pd.date_range(start, periods=n_days * 1440, freq="1min")
    return pd.Series(np.tile(per_minute, n_days), index=idx)


def _make_record(
    n_days: int = 10,
    participant_id: str = "depression_1",
    label: str = "depression",
    pattern: np.ndarray | None = None,
) -> ParticipantRecord:
    if pattern is None:
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
    activity = _make_activity(n_days, pattern)
    return ParticipantRecord(
        participant_id=participant_id,
        label=label,
        activity=activity,
    )


# ---------------------------------------------------------------------------
# Tests: derived metrics
# ---------------------------------------------------------------------------


class TestDerivedMetrics:

    def test_amplitude_known_values(self):
        """amplitude = M10.value - L5.value."""
        l5 = WindowResult(value=100.0, onset=None)
        m10 = WindowResult(value=600.0, onset=None)
        assert amplitude(l5, m10) == pytest.approx(500.0)

    def test_amplitude_zero_for_equal_windows(self):
        """Flat recording: M10 == L5 → amplitude = 0."""
        w = WindowResult(value=300.0, onset=None)
        assert amplitude(w, w) == pytest.approx(0.0)

    def test_amplitude_nan_if_l5_nan(self):
        l5 = WindowResult(value=float("nan"), onset=None)
        m10 = WindowResult(value=600.0, onset=None)
        assert math.isnan(amplitude(l5, m10))

    def test_amplitude_nan_if_m10_nan(self):
        l5 = WindowResult(value=100.0, onset=None)
        m10 = WindowResult(value=float("nan"), onset=None)
        assert math.isnan(amplitude(l5, m10))

    def test_relative_amplitude_known_values(self):
        """RA = (600 - 100) / (600 + 100) = 500/700 ≈ 0.714."""
        l5 = WindowResult(value=100.0, onset=None)
        m10 = WindowResult(value=600.0, onset=None)
        assert relative_amplitude(l5, m10) == pytest.approx(500.0 / 700.0, abs=1e-9)

    def test_relative_amplitude_zero_for_equal_windows(self):
        """RA = 0 when M10 == L5."""
        w = WindowResult(value=300.0, onset=None)
        assert relative_amplitude(w, w) == pytest.approx(0.0)

    def test_relative_amplitude_nan_if_sum_zero(self):
        """M10 + L5 = 0 → RA undefined → NaN."""
        w = WindowResult(value=0.0, onset=None)
        assert math.isnan(relative_amplitude(w, w))

    def test_relative_amplitude_nan_propagation(self):
        l5 = WindowResult(value=float("nan"), onset=None)
        m10 = WindowResult(value=600.0, onset=None)
        assert math.isnan(relative_amplitude(l5, m10))


# ---------------------------------------------------------------------------
# Tests: extract_participant_features
# ---------------------------------------------------------------------------


class TestExtractParticipantFeatures:

    def test_returns_all_expected_keys(self):
        """Output dict contains participant_id, label, and all 17 feature keys."""
        record = _make_record(n_days=10)
        result = extract_participant_features(record)

        assert "participant_id" in result
        assert "label" in result
        for name in FEATURE_NAMES:
            assert name in result, f"missing feature: {name}"

    def test_participant_id_and_label_preserved(self):
        """participant_id and label pass through unchanged."""
        record = _make_record(participant_id="control_5", label="control")
        result = extract_participant_features(record)
        assert result["participant_id"] == "control_5"
        assert result["label"] == "control"

    def test_all_feature_values_are_float(self):
        """Every feature value is a Python float (including NaN)."""
        record = _make_record(n_days=10)
        result = extract_participant_features(record)
        for name in FEATURE_NAMES:
            assert isinstance(result[name], float), (
                f"feature '{name}' is {type(result[name])}, expected float"
            )

    def test_is_nan_for_short_recording(self):
        """IS requires min_days=7; 3-day recording → IS = NaN."""
        record = _make_record(n_days=3)
        result = extract_participant_features(record)
        assert math.isnan(result["is"])

    def test_finite_values_for_adequate_recording(self):
        """10-day structured recording: most features are finite."""
        record = _make_record(n_days=10)
        result = extract_participant_features(record)
        # IS needs 7 days: should be finite.
        assert not math.isnan(result["is"])
        # IV needs 1 day: finite.
        assert not math.isnan(result["iv"])
        # Cosinor needs 1 day: finite.
        assert not math.isnan(result["cosinor_mesor"])
        # L5/M10 need 1 day: finite.
        assert not math.isnan(result["l5_value"])
        assert not math.isnan(result["m10_value"])


# ---------------------------------------------------------------------------
# Tests: extract_all_features
# ---------------------------------------------------------------------------


class TestExtractAllFeatures:

    def test_returns_dataframe(self):
        """Output is a pd.DataFrame."""
        records = [_make_record(n_days=10, participant_id=f"control_{i}") for i in range(3)]
        df = extract_all_features(records)
        assert isinstance(df, pd.DataFrame)

    def test_index_is_participant_id(self):
        """DataFrame index is participant_id."""
        records = [
            _make_record(n_days=10, participant_id="control_1", label="control"),
            _make_record(n_days=10, participant_id="depression_1", label="depression"),
        ]
        df = extract_all_features(records)
        assert "control_1" in df.index
        assert "depression_1" in df.index

    def test_one_row_per_participant(self):
        """One row per record, no duplication."""
        records = [_make_record(n_days=10, participant_id=f"p_{i}") for i in range(5)]
        df = extract_all_features(records)
        assert len(df) == 5

    def test_label_column_present(self):
        """'label' is the first column."""
        records = [_make_record(n_days=10)]
        df = extract_all_features(records)
        assert "label" in df.columns
        assert df.columns[0] == "label"

    def test_all_feature_columns_present(self):
        """All 17 feature columns are in the DataFrame."""
        records = [_make_record(n_days=10)]
        df = extract_all_features(records)
        for name in FEATURE_NAMES:
            assert name in df.columns, f"missing column: {name}"

    def test_failed_participant_appears_as_nan_row(self):
        """A record that causes an exception produces a NaN row, not a missing row."""
        good = _make_record(n_days=10, participant_id="control_1", label="control")
        # A record with an empty series will cause feature functions to raise.
        bad_activity = pd.Series(
            [], dtype=float, index=pd.DatetimeIndex([], dtype="datetime64[ns]")
        )
        bad = ParticipantRecord(
            participant_id="broken_1", label="control", activity=bad_activity
        )
        df = extract_all_features([good, bad])
        assert "broken_1" in df.index
        assert all(math.isnan(v) for v in df.loc["broken_1", FEATURE_NAMES])