"""
Repeated GroupKFold CV experiment: distributional vs temporal vs combined.

20-rep x 5-fold CV, logistic regression only, two tasks (3-class, 2-class).

Prerequisites:
  1. python scripts/generate_fold_fixtures.py
  2. Temporal features pre-computed at cfg.data.temporal_features_csv

Usage:
    python scripts/run_repeated_cv.py [--config config/config.yaml]

Outputs in results/repeated_cv/:
  summary.csv   -- mean_f1, sd_f1, ci_lo_mean, ci_hi_mean per task/feature_set
  paired.csv    -- combined-vs-dist paired comparison per task
  raw_scores.csv -- fold-level F1 audit trail
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from obf_psychiatric_pipeline.config import load_config
from obf_psychiatric_pipeline.cv.folds import load_folds, validate_folds
from obf_psychiatric_pipeline.cv.runner import (
    build_results_tables,
    evaluate_repeated,
    summarize_reps,
)
from obf_psychiatric_pipeline.data.loader import load_all
from obf_psychiatric_pipeline.data.raw_loader import load_all_actigraphy
from obf_psychiatric_pipeline.features.extract import extract_all_features
from obf_psychiatric_pipeline.models.aggregate import to_participant_level
from obf_psychiatric_pipeline.models.classifiers import make_dummy
from obf_psychiatric_pipeline.data.preprocess import preprocess

DIST_FEATURES = ["mean", "sd", "pctZeros", "median", "q75"]
LABEL_COL = "class"


def _make_logreg() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )),
    ])


def _binary_labels(y_3class: np.ndarray, le: LabelEncoder) -> np.ndarray:
    control_idx = list(le.classes_).index("control")
    return (y_3class != control_idx).astype(int)


def _load_distributional(cfg) -> pd.DataFrame:
    metadata, features = load_all(cfg.data.root)
    _, features = preprocess(metadata, features, cfg)
    df = to_participant_level(features)
    missing = [c for c in DIST_FEATURES + [LABEL_COL, "user"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns after preprocessing: {missing}")
    return df[["user"] + DIST_FEATURES + [LABEL_COL]].copy()


def _load_temporal(cfg) -> pd.DataFrame:
    records = load_all_actigraphy(cfg.data.actigraphy_root)
    df = extract_all_features(records)
    df = df.rename_axis("user").reset_index()
    df = df.rename(columns={"label": "class"})
    return df


def _join(dist_df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
    before = len(dist_df)
    # Drop class from temp_df — class label comes from dist_df only.
    # Without this, merge produces "class_temp" which contaminates temp_cols.
    temp_df = temp_df.drop(columns=[LABEL_COL], errors="ignore")
    merged = dist_df.merge(temp_df, on="user", how="inner", suffixes=("", "_temp"))
    after = len(merged)
    if after < before:
        warnings.warn(f"{before - after} participant(s) dropped by inner join.", stacklevel=2)
    return merged


def _temporal_cols(combined_df: pd.DataFrame) -> list[str]:
    exclude = {"user", LABEL_COL}
    return [c for c in combined_df.columns if c not in exclude and c not in DIST_FEATURES]


def _feature_matrix(df, feature_cols):
    df = df.sort_values("user").reset_index(drop=True)
    X = df[feature_cols].to_numpy(dtype=float)
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL].to_numpy())
    return X, y, df["user"].to_numpy(), le


def run_experiment(cfg, folds: dict, output_dir: Path) -> None:
    print("[1/5] Loading features...")
    dist_df     = _load_distributional(cfg)
    temp_df     = _load_temporal(cfg)
    combined_df = _join(dist_df, temp_df)
    temp_cols   = _temporal_cols(combined_df)

    print(f"      {len(combined_df)} participants after join.")
    print(f"      {len(temp_cols)} temporal feature columns.")

    print("[2/5] Validating fold fixtures against participant set...")
    validate_folds(folds, combined_df["user"].unique().tolist())

    X_dist, y3, ids, le = _feature_matrix(combined_df, DIST_FEATURES)
    X_temp, _,  _,  _  = _feature_matrix(combined_df, temp_cols)
    X_comb, _,  _,  _  = _feature_matrix(combined_df, DIST_FEATURES + temp_cols)
    y2 = _binary_labels(y3, le)

    estimator = _make_logreg()
    dummy     = make_dummy(seed=0)
    n_reps   = folds["metadata"]["n_reps"]
    n_splits = folds["metadata"]["n_splits"]
    print(f"[3/5] Running {n_reps} x {n_splits}-fold CV (8 conditions)...")

    conditions = [
        ("3class", "distributional", X_dist, y3, estimator),
        ("3class", "temporal",       X_temp, y3, estimator),
        ("3class", "combined",       X_comb, y3, estimator),
        ("2class", "distributional", X_dist, y2, estimator),
        ("2class", "temporal",       X_temp, y2, estimator),
        ("2class", "combined",       X_comb, y2, estimator),
        ("3class", "dummy",          X_dist, y3, dummy),
        ("2class", "dummy",          X_dist, y2, dummy),
    ]

    all_summaries: dict[str, dict] = {}
    raw_records: list[dict] = []

    for task, feat_set, X, y, est in conditions:
        label = f"{task}/{feat_set}"
        print(f"      {label}...", end=" ", flush=True)
        fold_scores = evaluate_repeated(X, y, ids, folds, est)
        rep_scores  = fold_scores.mean(axis=1)
        all_summaries[label] = summarize_reps(rep_scores, label)
        all_summaries[label]["_fold_scores"] = fold_scores
        print(f"mean={rep_scores.mean():.3f}  sd={rep_scores.std(ddof=1):.3f}")

        for rep_idx in range(n_reps):
            for fold_idx in range(n_splits):
                raw_records.append({
                    "rep": rep_idx, "fold": fold_idx,
                    "task": task, "feature_set": feat_set,
                    "macro_f1": round(float(fold_scores[rep_idx, fold_idx]), 5),
                })

    print("[4/5] Computing summary statistics...")
    paired_specs = [
        ("3class/distributional", "3class/combined", "3class: combined vs dist"),
        ("2class/distributional", "2class/combined", "2class: combined vs dist"),
        ("3class/distributional", "3class/temporal", "3class: temporal vs dist"),
        ("3class/dummy",          "3class/combined", "3class: combined vs dummy"),
        ("2class/dummy",          "2class/combined", "2class: combined vs dummy"),
    ]
    summary_df, paired_df = build_results_tables(all_summaries, paired_specs)

    print(f"[5/5] Writing results to {output_dir}/...")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    paired_df.to_csv(output_dir / "paired.csv", index=False)
    pd.DataFrame(raw_records).to_csv(output_dir / "raw_scores.csv", index=False)

    print("\n" + "=" * 60)
    print("REPEATED CV SUMMARY")
    print(f"  CI: 95% t-interval on mean across {n_reps} reps")
    print(f"  (NOT bootstrap CI -- see runner.py docstring)")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("\nPAIRED COMPARISONS")
    print(paired_df.to_string(index=False))
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).parent.parent / "config" / "config.yaml"))
    p.add_argument("--fixture",    default=None)
    p.add_argument("--output-dir", default="results/repeated_cv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    if args.fixture:
        fixture_path = Path(args.fixture)
    else:
        n_splits    = getattr(cfg.cv, "n_splits", 5)
        n_reps      = getattr(cfg.cv, "n_reps",   20)
        fixture_dir = Path(__file__).parent.parent / getattr(cfg.cv, "fixture_dir", "config/folds_repeated")
        fixture_path = fixture_dir / f"folds_n{n_splits}_r{n_reps}.json"

    print(f"Loading fold fixture: {fixture_path}")
    folds = load_folds(fixture_path)
    meta  = folds["metadata"]
    print(f"  {meta['n_reps']} reps x {meta['n_splits']} folds, "
          f"{meta['n_participants']} participants, "
          f"generated on {meta['platform']} ({meta['generated_at']})")

    run_experiment(cfg, folds, Path(args.output_dir))


if __name__ == "__main__":
    main()
