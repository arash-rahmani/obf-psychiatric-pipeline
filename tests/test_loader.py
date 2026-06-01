# tests/test_loader.py

import pytest
import pandas as pd
from pathlib import Path

from obf_psychiatric_pipeline.data.loader import (
    load_all,
    SchemaError,
    COHORTS_WITH_FEATURES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_cohort_csvs(root: Path) -> None:
    """Write minimal but schema-valid metadata CSVs for all five cohorts."""
    files = {
        "controlinfo.csv": {
            "number": [1], "days": [10], "gender": [1], "age": [30], "acc_time": [0.9],
        },
        "depressioninfo.csv": {
            "number": [2], "days": [14], "gender": [2], "age": [25],
            "madrs1": [18], "madrs2": [12], "afftype": [1],
        },
        "schizophreniainfo.csv": {
            "number": [3], "days": [12], "gender": [1], "age": [35],
            "bprs": [40], "schtype": [1],
        },
        "adhdinfo.csv": {
            "number": [4], "days": [8], "gender": [2], "age": [22],
            "adhd": [1], "asrs": [55],
        },
        "clinicalinfo.csv": {
            "number": [5], "days": [9], "gender": [1], "age": [28], "adhd": [0],
        },
    }
    for filename, data in files.items():
        pd.DataFrame(data).to_csv(root / filename, index=False)


def _write_features_csv(root: Path, classes: list[str] | None = None) -> None:
    """Write a minimal features.csv. Defaults to the three valid cohort classes."""
    if classes is None:
        classes = list(COHORTS_WITH_FEATURES)
    pd.DataFrame({
        "user":     range(1, len(classes) + 1),
        "mean":     [0.5] * len(classes),
        "sd":       [0.1] * len(classes),
        "pctZeros": [0.0] * len(classes),
        "median":   [0.5] * len(classes),
        "q25":      [0.3] * len(classes),
        "q75":      [0.7] * len(classes),
        "class":    classes,
    }).to_csv(root / "features.csv", index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_all_happy_path(tmp_path):
    """All six CSVs are valid — load_all returns well-formed metadata and features."""
    _write_cohort_csvs(tmp_path)
    _write_features_csv(tmp_path)

    metadata, features = load_all(tmp_path)

    # Every cohort key is present and carries its cohort tag.
    assert set(metadata.keys()) == {
        "control", "depression", "schizophrenia", "adhd", "clinical"}
    for cohort, df in metadata.items():
        assert "cohort" in df.columns, f"'cohort' column missing from {cohort}"
        assert (df["cohort"] == cohort).all()

    # Features DataFrame has the expected shape and only known classes.
    assert "class" in features.columns
    assert set(features["class"].unique()).issubset(set(COHORTS_WITH_FEATURES))


def test_load_all_missing_madrs1_raises_schema_error(tmp_path):
    """depressioninfo.csv without 'madrs1' must raise SchemaError, not silently load."""
    _write_cohort_csvs(tmp_path)
    _write_features_csv(tmp_path)

    # Overwrite depressioninfo.csv with madrs1 stripped out.
    pd.DataFrame({
        "number": [2], "days": [14], "gender": [2], "age": [25],
        # madrs1 intentionally omitted
        "madrs2": [12], "afftype": [1],
    }).to_csv(tmp_path / "depressioninfo.csv", index=False)

    with pytest.raises(SchemaError, match="madrs1"):
        load_all(tmp_path)


def test_load_all_rogue_class_in_features_raises_schema_error(tmp_path):
    """A class value not in COHORTS_WITH_FEATURES must raise SchemaError."""
    _write_cohort_csvs(tmp_path)
    _write_features_csv(tmp_path, classes=[
                        "control", "depression", "alien_cohort"])

    with pytest.raises(SchemaError, match="alien_cohort"):
        load_all(tmp_path)
