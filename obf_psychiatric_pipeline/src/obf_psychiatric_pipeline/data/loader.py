"""
OBF-Psychiatric data loader.

Loads the five participant-metadata CSVs and the pre-computed feature matrix
from a config-specified root directory, validates expected columns, and
returns typed DataFrames ready for downstream feature engineering and
classification.

Design contract:
    - No hardcoded paths. All locations come from `Config`.
    - Each loader returns a single DataFrame with a known schema.
    - `class` / cohort labels are assigned at load time, not downstream.
    - ADHD and clinical cohorts have metadata but no features in v1 of
      the dataset - this is handled explicitly, not silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Cohort names used consistently throughout the pipeline.
COHORTS = ("control", "depression", "schizophrenia", "adhd", "clinical")

# Cohorts that have rows in features.csv (v1 of the dataset).
COHORTS_WITH_FEATURES = ("control", "depression", "schizophrenia")

# Expected columns per metadata CSV. Minimal, not exhaustive — the loader
# checks these exist, additional columns pass through untouched.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "control":       ("number", "days", "gender", "age", "acc_time"),
    "depression":    ("number", "days", "gender", "age", "madrs1", "madrs2", "afftype"),
    "schizophrenia": ("number", "days", "gender", "age", "bprs", "schtype"),
    "adhd":          ("number", "days", "gender", "age", "adhd", "asrs"),
    "clinical":      ("number", "days", "gender", "age", "adhd"),
    "features":      ("user", "mean", "sd", "pctZeros", "median", "q25", "q75", "class"),
}


@dataclass(frozen=True)
class LoaderPaths:
    """Resolved absolute paths to every file the loader reads."""
    control: Path
    depression: Path
    schizophrenia: Path
    adhd: Path
    clinical: Path
    features: Path

    @classmethod
    def from_root(cls, root: Path | str) -> "LoaderPaths":
        root = Path(root)
        return cls(
            control=root / "controlinfo.csv",
            depression=root / "depressioninfo.csv",
            schizophrenia=root / "schizophreniainfo.csv",
            adhd=root / "adhdinfo.csv",
            clinical=root / "clinicalinfo.csv",
            features=root / "features.csv",
        )


class SchemaError(ValueError):
    """Raised when a loaded CSV is missing required columns."""


def _check_schema(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{name}: missing required columns {missing}. "
            f"Got columns: {list(df.columns)}"
        )


def _load_cohort(path: Path, cohort: str) -> pd.DataFrame:
    """Load one metadata CSV, validate its schema, tag the cohort."""
    if not path.exists():
        raise FileNotFoundError(f"{cohort} metadata not found at {path}")
    df = pd.read_csv(path)
    _check_schema(df, REQUIRED_COLUMNS[cohort], cohort)
    df = df.copy()
    df["cohort"] = cohort
    return df


def load_metadata(paths: LoaderPaths) -> dict[str, pd.DataFrame]:
    """
    Load all five participant-level metadata CSVs.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: 'control', 'depression', 'schizophrenia', 'adhd', 'clinical'.
        Each DataFrame carries a `cohort` column matching its key.
    """
    return {
        "control":       _load_cohort(paths.control,       "control"),
        "depression":    _load_cohort(paths.depression,    "depression"),
        "schizophrenia": _load_cohort(paths.schizophrenia, "schizophrenia"),
        "adhd":          _load_cohort(paths.adhd,          "adhd"),
        "clinical":      _load_cohort(paths.clinical,      "clinical"),
    }


def load_features(paths: LoaderPaths) -> pd.DataFrame:
    """
    Load the pre-computed per-day feature matrix.

    Note: as of dataset v1, only control / depression / schizophrenia rows
    exist here. ADHD and clinical cohorts are metadata-only until raw
    actigraphy is processed.
    """
    if not paths.features.exists():
        raise FileNotFoundError(f"features.csv not found at {paths.features}")
    df = pd.read_csv(paths.features)
    _check_schema(df, REQUIRED_COLUMNS["features"], "features")

    unexpected = set(df["class"].unique()) - set(COHORTS_WITH_FEATURES)
    if unexpected:
        raise SchemaError(f"Unexpected classes in features.csv: {unexpected}")
    return df


def load_all(root: Path | str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Top-level convenience loader.

    Parameters
    ----------
    root : Path | str
        Directory containing the six OBF CSV files.

    Returns
    -------
    (metadata, features)
        `metadata` is a dict keyed by cohort name; `features` is the
        per-day feature matrix for the three cohorts that have it.
    """
    paths = LoaderPaths.from_root(root)
    metadata = load_metadata(paths)
    features = load_features(paths)
    return metadata, features