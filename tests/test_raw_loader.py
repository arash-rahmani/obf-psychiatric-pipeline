"""
Tests for data/raw_loader.py.

All tests use pytest's tmp_path fixture to create synthetic CSVs that
match the observed OBF format:

    timestamp,date,activity
    2003-05-07 12:00:00,2003-05-07,0
    2003-05-07 12:01:00,2003-05-07,143

No real data files are required.
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obf_psychiatric_pipeline.data.raw_loader import (
    ParticipantRecord,
    RawSchemaError,
    load_all_actigraphy,
    load_cohort_actigraphy,
    load_participant_activity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_activity_csv(
    path: Path,
    n_minutes: int = 10,
    start: str = "2003-05-07 12:00:00",
    include_date_column: bool = True,
    activity_values: list[int] | None = None,
) -> Path:
    """Write a minimal valid OBF-format CSV to *path* and return it."""
    timestamps = pd.date_range(start, periods=n_minutes, freq="1min")
    if activity_values is None:
        rng = np.random.default_rng(0)
        activity = rng.integers(0, 500, size=n_minutes)
    else:
        activity = activity_values

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ts, act in zip(timestamps, activity):
        date_str = ts.strftime("%Y-%m-%d")
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        if include_date_column:
            rows.append(f"{ts_str},{date_str},{act}")
        else:
            rows.append(f"{ts_str},{act}")

    header = "timestamp,date,activity" if include_date_column else "timestamp,activity"
    path.write_text("\n".join([header] + rows))
    return path


def _make_cohort_dir(
    parent: Path,
    cohort_name: str,
    n_participants: int = 3,
    n_minutes: int = 10,
) -> Path:
    """Create a cohort subdirectory with n_participants CSV files."""
    cohort_dir = parent / cohort_name
    cohort_dir.mkdir(parents=True)
    for i in range(1, n_participants + 1):
        _write_activity_csv(
            cohort_dir / f"{cohort_name}_{i}.csv",
            n_minutes=n_minutes,
        )
    return cohort_dir


# ---------------------------------------------------------------------------
# Tests: load_participant_activity
# ---------------------------------------------------------------------------


class TestLoadParticipantActivity:

    def test_happy_path_returns_participant_record(self, tmp_path):
        """Valid CSV produces a ParticipantRecord with correct fields."""
        cohort_dir = tmp_path / "depression"
        cohort_dir.mkdir()
        csv = _write_activity_csv(cohort_dir / "depression_1.csv", n_minutes=5)

        record = load_participant_activity(csv)

        assert isinstance(record, ParticipantRecord)
        assert isinstance(record.activity, pd.Series)
        assert isinstance(record.activity.index, pd.DatetimeIndex)
        assert len(record.activity) == 5

    def test_participant_id_matches_filename_stem(self, tmp_path):
        """participant_id is the filename stem, not the full path."""
        cohort_dir = tmp_path / "control"
        cohort_dir.mkdir()
        csv = _write_activity_csv(cohort_dir / "control_7.csv", n_minutes=5)

        record = load_participant_activity(csv)
        assert record.participant_id == "control_7"

    def test_label_from_parent_directory_name(self, tmp_path):
        """label is derived from the parent directory name."""
        for cohort in ("control", "depression", "schizophrenia"):
            d = tmp_path / cohort
            d.mkdir()
            csv = _write_activity_csv(d / f"{cohort}_1.csv", n_minutes=3)
            record = load_participant_activity(csv)
            assert record.label == cohort

    def test_output_is_sorted_by_timestamp(self, tmp_path):
        """Even if the CSV rows are out of order, output index is monotonic."""
        csv_path = tmp_path / "control" / "control_1.csv"
        csv_path.parent.mkdir()
        # Write rows in reverse order
        csv_path.write_text(
            "timestamp,date,activity\n"
            "2003-05-07 12:03:00,2003-05-07,30\n"
            "2003-05-07 12:01:00,2003-05-07,10\n"
            "2003-05-07 12:02:00,2003-05-07,20\n"
        )
        record = load_participant_activity(csv_path)
        assert record.activity.index.is_monotonic_increasing

    def test_activity_is_float_dtype(self, tmp_path):
        """Activity values are cast to float64 for pipeline compatibility."""
        d = tmp_path / "control"
        d.mkdir()
        csv = _write_activity_csv(d / "control_1.csv", n_minutes=5)
        record = load_participant_activity(csv)
        assert record.activity.dtype == float

    def test_date_column_is_optional(self, tmp_path):
        """CSV without the redundant date column still loads correctly."""
        d = tmp_path / "control"
        d.mkdir()
        csv = _write_activity_csv(
            d / "control_1.csv", n_minutes=5, include_date_column=False
        )
        record = load_participant_activity(csv)
        assert len(record.activity) == 5

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_participant_activity(tmp_path / "nonexistent.csv")

    def test_raises_on_missing_activity_column(self, tmp_path):
        """CSV missing the activity column raises RawSchemaError."""
        csv_path = tmp_path / "control_1.csv"
        csv_path.write_text(
            "timestamp,date\n"
            "2003-05-07 12:00:00,2003-05-07\n"
        )
        with pytest.raises(RawSchemaError, match="activity"):
            load_participant_activity(csv_path)

    def test_raises_on_missing_timestamp_column(self, tmp_path):
        """CSV missing the timestamp column raises RawSchemaError."""
        csv_path = tmp_path / "control_1.csv"
        csv_path.write_text(
            "date,activity\n"
            "2003-05-07,100\n"
        )
        with pytest.raises(RawSchemaError, match="timestamp"):
            load_participant_activity(csv_path)

    def test_raises_on_negative_activity(self, tmp_path):
        """CSV with negative activity values raises RawSchemaError."""
        csv_path = tmp_path / "control_1.csv"
        csv_path.write_text(
            "timestamp,date,activity\n"
            "2003-05-07 12:00:00,2003-05-07,-5\n"
        )
        with pytest.raises(RawSchemaError, match="negative"):
            load_participant_activity(csv_path)


# ---------------------------------------------------------------------------
# Tests: load_cohort_actigraphy
# ---------------------------------------------------------------------------


class TestLoadCohortActigraphy:

    def test_loads_all_csvs_in_directory(self, tmp_path):
        """All CSV files in the cohort directory are loaded."""
        cohort_dir = _make_cohort_dir(tmp_path, "depression", n_participants=4)
        records = load_cohort_actigraphy(cohort_dir)
        assert len(records) == 4

    def test_participant_ids_match_filenames(self, tmp_path):
        """participant_ids match the filename stems."""
        cohort_dir = _make_cohort_dir(tmp_path, "control", n_participants=3)
        records = load_cohort_actigraphy(cohort_dir)
        ids = {r.participant_id for r in records}
        assert ids == {"control_1", "control_2", "control_3"}

    def test_label_override(self, tmp_path):
        """Passing label overrides the directory-derived label."""
        cohort_dir = _make_cohort_dir(tmp_path, "condition", n_participants=2)
        records = load_cohort_actigraphy(cohort_dir, label="depression")
        assert all(r.label == "depression" for r in records)

    def test_raises_if_directory_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_cohort_actigraphy(tmp_path / "nonexistent_cohort")

    def test_raises_if_no_csv_files(self, tmp_path):
        """Empty directory raises FileNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No CSV files"):
            load_cohort_actigraphy(empty_dir)


# ---------------------------------------------------------------------------
# Tests: load_all_actigraphy
# ---------------------------------------------------------------------------


class TestLoadAllActigraphy:

    def test_loads_three_cohorts(self, tmp_path):
        """load_all_actigraphy loads all participants from all three cohorts."""
        root = tmp_path / "actigraphy"
        _make_cohort_dir(root, "control", n_participants=3)
        _make_cohort_dir(root, "depression", n_participants=2)
        _make_cohort_dir(root, "schizophrenia", n_participants=2)

        records = load_all_actigraphy(root)
        assert len(records) == 7

    def test_labels_match_cohort_dirs(self, tmp_path):
        """Labels are derived from cohort directory names."""
        root = tmp_path / "actigraphy"
        _make_cohort_dir(root, "control", n_participants=2)
        _make_cohort_dir(root, "depression", n_participants=2)
        _make_cohort_dir(root, "schizophrenia", n_participants=2)

        records = load_all_actigraphy(root)
        label_counts = {}
        for r in records:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1

        assert label_counts == {"control": 2, "depression": 2, "schizophrenia": 2}

    def test_raises_if_root_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_all_actigraphy(tmp_path / "nonexistent_root")

    def test_custom_cohort_subset(self, tmp_path):
        """Only specified cohorts are loaded when cohorts parameter is set."""
        root = tmp_path / "actigraphy"
        _make_cohort_dir(root, "control", n_participants=2)
        _make_cohort_dir(root, "depression", n_participants=2)
        _make_cohort_dir(root, "schizophrenia", n_participants=2)

        records = load_all_actigraphy(root, cohorts=("control",))
        assert len(records) == 2
        assert all(r.label == "control" for r in records)