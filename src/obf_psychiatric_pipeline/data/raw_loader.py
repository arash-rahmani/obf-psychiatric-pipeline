"""
Raw per-minute actigraphy loader for OBF psychiatric pipeline.

File layout expected under actigraphy_root:

    data/raw/actigraphy/
        control/
            control_1.csv, control_2.csv, ...
        depression/
            depression_1.csv, depression_2.csv, ...
        schizophrenia/
            schizophrenia_1.csv, schizophrenia_2.csv, ...

Each CSV has three columns:
    timestamp  — "YYYY-MM-DD HH:MM:SS", one row per minute
    date       — "YYYY-MM-DD" (redundant; ignored by the loader)
    activity   — non-negative integer activity counts

Participant mapping:
    participant_id = filename stem  (e.g. "depression_1")
    label          = parent dir name (e.g. "depression")

Both match the corresponding columns in features.csv (user, class),
making downstream joins straightforward.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: frozenset[str] = frozenset({"timestamp", "activity"})
_DEFAULT_COHORTS: tuple[str, ...] = ("control", "depression", "schizophrenia")


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class ParticipantRecord(NamedTuple):
    """Raw actigraphy for one participant.

    Attributes
    ----------
    participant_id : str
        Derived from the CSV filename stem (e.g. "depression_1").
        Matches the ``user`` column in features.csv.
    label : str
        Cohort label derived from the parent directory name
        (e.g. "control", "depression", "schizophrenia").
        Matches the ``class`` column in features.csv.
    activity : pd.Series
        Per-minute activity counts with a monotonic DatetimeIndex
        (dtype float64).  Recordings do not always start at midnight;
        partial days at the start and end of the recording are handled
        downstream by ``_keep_full_days``.  No NaN filling is applied;
        gaps in the recording remain as missing index entries.
    """

    participant_id: str
    label: str
    activity: pd.Series


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class RawSchemaError(ValueError):
    """Raised when a raw actigraphy CSV fails schema validation."""


def _validate_raw_csv(df: pd.DataFrame, source: Path) -> None:
    """Raise ``RawSchemaError`` if *df* does not conform to the expected schema."""
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise RawSchemaError(
            f"{source.name}: missing required column(s) {sorted(missing)}. "
            f"Found columns: {sorted(df.columns)}."
        )
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise RawSchemaError(
            f"{source.name}: 'timestamp' column could not be parsed as datetime. "
            f"Dtype: {df['timestamp'].dtype}."
        )
    if (df["activity"] < 0).any():
        raise RawSchemaError(
            f"{source.name}: 'activity' contains negative values "
            f"(min = {df['activity'].min()})."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_participant_activity(filepath: Path | str) -> ParticipantRecord:
    """Load one participant's raw actigraphy CSV.

    Parameters
    ----------
    filepath : Path or str
        Path to the participant CSV file.  The participant_id is derived
        from the filename stem; the label is derived from the parent
        directory name.

    Returns
    -------
    ParticipantRecord
        participant_id, label, and a float Series with a monotonic
        DatetimeIndex.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RawSchemaError
        If the CSV is missing required columns, has an un-parseable
        timestamp column, or contains negative activity values.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw actigraphy file not found: {filepath}")

    try:
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
    except ValueError as exc:
        raise RawSchemaError(
            f"{filepath.name}: could not read CSV — {exc}"
        ) from exc
    _validate_raw_csv(df, filepath)

    activity = (
        df.set_index("timestamp")["activity"]
        .astype(float)
        .sort_index()
    )
    activity.index.name = "timestamp"

    return ParticipantRecord(
        participant_id=filepath.stem,
        label=filepath.parent.name,
        activity=activity,
    )


def load_cohort_actigraphy(
    cohort_dir: Path | str,
    label: str | None = None,
) -> list[ParticipantRecord]:
    """Load all participant CSVs from one cohort directory.

    Parameters
    ----------
    cohort_dir : Path or str
        Directory containing per-participant CSV files.
    label : str, optional
        Override the cohort label.  If ``None``, the label is derived
        from the directory name.

    Returns
    -------
    list of ParticipantRecord
        Sorted by participant_id for deterministic ordering.

    Raises
    ------
    FileNotFoundError
        If *cohort_dir* does not exist or contains no CSV files.
    RawSchemaError
        If any CSV fails schema validation.
    """
    cohort_dir = Path(cohort_dir)
    if not cohort_dir.exists():
        raise FileNotFoundError(f"Cohort directory not found: {cohort_dir}")

    csv_files = sorted(cohort_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cohort_dir}.")

    records = []
    for f in csv_files:
        record = load_participant_activity(f)
        if label is not None:
            record = ParticipantRecord(
                participant_id=record.participant_id,
                label=label,
                activity=record.activity,
            )
        records.append(record)

    return records


def load_all_actigraphy(
    actigraphy_root: Path | str,
    cohorts: tuple[str, ...] = _DEFAULT_COHORTS,
) -> list[ParticipantRecord]:
    """Load all cohort directories under *actigraphy_root*.

    Parameters
    ----------
    actigraphy_root : Path or str
        Directory containing cohort subdirectories (e.g. ``data/raw/actigraphy``).
    cohorts : tuple of str, default ('control', 'depression', 'schizophrenia')
        Cohort subdirectory names to load.

    Returns
    -------
    list of ParticipantRecord
        All participants across all requested cohorts, in cohort order.

    Raises
    ------
    FileNotFoundError
        If *actigraphy_root* or any cohort subdirectory does not exist,
        or if any cohort directory contains no CSV files.
    RawSchemaError
        If any CSV fails schema validation.
    """
    actigraphy_root = Path(actigraphy_root)
    if not actigraphy_root.exists():
        raise FileNotFoundError(f"Actigraphy root not found: {actigraphy_root}")

    records: list[ParticipantRecord] = []
    for cohort in cohorts:
        records.extend(load_cohort_actigraphy(actigraphy_root / cohort))

    return records