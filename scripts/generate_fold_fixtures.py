"""
Generate and commit fold fixtures for repeated KFold CV.

Run ONCE on Linux before the experiment. The resulting JSON is committed to
version control so fold assignments are identical on all platforms.

Usage:
    python scripts/generate_fold_fixtures.py [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from obf_psychiatric_pipeline.config import load_config
from obf_psychiatric_pipeline.cv.folds import generate_repeated_folds, save_folds
from obf_psychiatric_pipeline.data.loader import load_all
from obf_psychiatric_pipeline.data.preprocess import preprocess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate committed fold fixtures.")
    p.add_argument("--config", default=str(Path(__file__).parent.parent / "config" / "config.yaml"))
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing fixture. Use deliberately.")
    p.add_argument("--n-splits", type=int, default=None)
    p.add_argument("--n-reps", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    n_splits = args.n_splits or getattr(cfg.cv, "n_splits", 5)
    n_reps   = args.n_reps   or getattr(cfg.cv, "n_reps",   20)
    seeds    = list(range(n_reps))
    fixture_dir  = Path(getattr(cfg.cv, "fixture_dir", "config/folds_repeated"))
    fixture_path = fixture_dir / f"folds_n{n_splits}_r{n_reps}.json"

    if fixture_path.exists() and not args.force:
        print(
            f"[ABORT] Fixture already exists: {fixture_path}\n"
            "Pass --force to regenerate. Only do this if the participant set changed."
        )
        sys.exit(1)

    print("[1/3] Loading and preprocessing data to get participant set...")
    metadata, features = load_all(cfg.data.root)
    _, participants  = preprocess(metadata, features, cfg)

    participant_ids = sorted(participants["user"].unique().tolist())
    print(f"      {len(participant_ids)} participants after preprocessing.")

    print(f"[2/3] Generating {n_reps} x {n_splits}-fold fixtures (seeds 0-{n_reps-1})...")
    folds = generate_repeated_folds(
        participant_ids=participant_ids,
        n_splits=n_splits,
        n_reps=n_reps,
        seeds=seeds,
    )

    test_counts: Counter = Counter()
    for rep in folds["repetitions"]:
        for fold in rep["folds"]:
            test_counts.update(fold["test"])
    unexpected = {uid: cnt for uid, cnt in test_counts.items() if cnt != n_reps}
    if unexpected:
        raise RuntimeError(f"Fold generation error: unexpected test counts: {unexpected}")

    print(f"[3/3] Writing fixture to {fixture_path}...")
    save_folds(folds, fixture_path)
    print(f"[OK]  Fixture saved. Commit {fixture_path} to version control.")
    print(f"      Platform : {folds['metadata']['platform']}")
    print(f"      Generated: {folds['metadata']['generated_at']}")


if __name__ == "__main__":
    main()
