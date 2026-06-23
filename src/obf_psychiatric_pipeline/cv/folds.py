"""
Committed fold fixtures for repeated KFold cross-validation.

Fixtures are generated once (on Linux, via generate_fold_fixtures.py) and
committed to config/folds_repeated/. At experiment time the fixture is loaded
verbatim, so fold assignments are identical across platforms and runtimes.

Fixture format (JSON):
    {
      "metadata": {
        "n_splits": 5,
        "n_reps": 20,
        "seeds": [0, ..., 19],
        "n_participants": 76,
        "participant_ids": [...],
        "generated_at": "2026-05-31",
        "platform": "Linux"
      },
      "repetitions": [
        {
          "seed": 0,
          "folds": [
            {"test": ["user_A", "user_B", ...]},
            ...
          ]
        },
        ...
      ]
    }

Each test list contains participant IDs; train = complement. No redundant
storage of train sets.
"""

from __future__ import annotations

import json
import platform
from datetime import date
from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.model_selection import KFold


def generate_repeated_folds(
    participant_ids: list[str],
    n_splits: int = 5,
    n_reps: int = 20,
    seeds: list[int] | None = None,
) -> dict:
    """
    Generate repeated k-fold assignments over participants.

    At participant level (one row per participant) KFold is the correct
    splitter: each participant IS its own group, so there is no within-subject
    leakage risk that GroupKFold is designed to prevent. GroupKFold is the
    right tool for per-day data where multiple rows share a participant;
    at participant level it sorts unique groups internally and ignores
    permutation order, making seeded repetitions impossible.

    Each repetition shuffles the participant list with a distinct seed before
    assigning consecutive chunks to folds. This gives genuinely independent
    fold assignments across repetitions.
    """
    if seeds is None:
        seeds = list(range(n_reps))
    if len(seeds) != n_reps:
        raise ValueError(f"len(seeds)={len(seeds)} must equal n_reps={n_reps}")
    if len(participant_ids) < n_splits:
        raise ValueError(
            f"n_participants={len(participant_ids)} < n_splits={n_splits}; "
            "not enough participants to form folds"
        )
    if len(participant_ids) != len(set(participant_ids)):
        raise ValueError("participant_ids contains duplicates")

    ids = np.array(sorted(participant_ids), dtype=str)
    n = len(ids)

    repetitions: list[dict] = []
    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        folds: list[dict] = []
        for _, test_idx in kf.split(ids):
            test_ids = sorted(ids[test_idx].tolist())
            folds.append({"test": test_ids})
        repetitions.append({"seed": int(seed), "folds": folds})

    return {
        "metadata": {
            "n_splits": n_splits,
            "n_reps": n_reps,
            "seeds": [int(s) for s in seeds],
            "n_participants": n,
            "participant_ids": sorted(participant_ids),
            "generated_at": str(date.today()),
            "platform": platform.system(),
        },
        "repetitions": repetitions,
    }


def save_folds(folds: dict, path: Path | str) -> None:
    """Serialise fixture dict to JSON. Creates parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2)
    reloaded = load_folds(path)
    if reloaded["metadata"]["n_reps"] != folds["metadata"]["n_reps"]:
        raise RuntimeError(f"Round-trip verification failed for {path}")


def load_folds(path: Path | str) -> dict:
    """Deserialise fixture JSON. Raises FileNotFoundError if absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fold fixture not found: {path}\n"
            "Run scripts/generate_fold_fixtures.py first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_folds(folds: dict, participant_ids: list[str]) -> None:
    """
    Hard-fail if the fixture's participant set does not exactly match the
    current dataset.
    """
    stored = set(folds["metadata"]["participant_ids"])
    current = set(participant_ids)
    if stored == current:
        return

    extra = current - stored
    missing = stored - current
    parts: list[str] = []
    if extra:
        parts.append(f"IDs in data but not in fixture ({len(extra)}): {sorted(extra)[:5]}...")
    if missing:
        parts.append(f"IDs in fixture but not in data ({len(missing)}): {sorted(missing)[:5]}...")
    raise ValueError(
        "Fold fixture participant set does not match current data. "
        "Regenerate the fixture with generate_fold_fixtures.py.\n"
        + "\n".join(parts)
    )


def iter_folds(
    folds: dict,
    participant_ids: np.ndarray,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Yield (rep_idx, fold_idx, train_mask, test_mask) boolean arrays over
    participant_ids for each fold in each repetition.
    """
    ids = np.asarray(participant_ids)
    for rep_idx, rep in enumerate(folds["repetitions"]):
        for fold_idx, fold in enumerate(rep["folds"]):
            test_set = set(fold["test"])
            test_mask = np.isin(ids, list(test_set))
            train_mask = ~test_mask
            yield rep_idx, fold_idx, train_mask, test_mask
