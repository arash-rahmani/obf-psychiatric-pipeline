from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


class ConfigError(ValueError):
    """Raised when the config file is missing required keys."""


@dataclass(frozen=True)
class DataConfig:
    root: Path


@dataclass(frozen=True)
class PreprocessingConfig:
    min_days_per_participant: int
    excluded_features: List[str]


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    n_folds: int


@dataclass(frozen=True)
class OutputConfig:
    results_dir: Path
    eda_dir: Path


@dataclass(frozen=True)
class Config:
    data: DataConfig
    preprocessing: PreprocessingConfig
    split: SplitConfig
    output: OutputConfig


def load_config(path: Path | str) -> Config:
    """Load and validate config.yaml, return a frozen Config dataclass."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    with path.open() as f:
        raw = yaml.safe_load(f)

    for key in ("data", "preprocessing", "split", "output"):
        if key not in raw:
            raise ConfigError(f"Config missing required section: '{key}'")

    return Config(
        data=DataConfig(root=Path(raw["data"]["root"])),
        preprocessing=PreprocessingConfig(
            min_days_per_participant=raw["preprocessing"]["min_days_per_participant"],
            excluded_features=raw["preprocessing"]["excluded_features"],
        ),
        split=SplitConfig(
            seed=raw["split"]["seed"],
            n_folds=raw["split"]["n_folds"],
        ),
        output=OutputConfig(
            results_dir=Path(raw["output"]["results_dir"]),
            eda_dir=Path(raw["output"]["eda_dir"]),
        ),
    )