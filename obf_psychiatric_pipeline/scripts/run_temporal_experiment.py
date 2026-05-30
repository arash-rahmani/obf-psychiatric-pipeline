#!/usr/bin/env python
"""
scripts/run_temporal_experiment.py

Phase 3: compare distributional-only, temporal-only, and combined features
on the OBF psychiatric classification tasks.

Three feature sets × two tasks (3-class, 2-class) × three classifiers
(dummy, logreg, xgb) = 18 experiments total, all at participant level
with 5-fold GroupKFold and 1 000-resample bootstrap CIs.

Results are printed as a comparison table and saved to
results/temporal_experiment/.

Run from repo root:
    python scripts/run_temporal_experiment.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from obf_psychiatric_pipeline.config import load_config
from obf_psychiatric_pipeline.data.loader import load_all
from obf_psychiatric_pipeline.data.preprocess import preprocess
from obf_psychiatric_pipeline.data.raw_loader import load_all_actigraphy
from obf_psychiatric_pipeline.data.split import make_splits
from obf_psychiatric_pipeline.features.extract import FEATURE_NAMES, extract_all_features
from obf_psychiatric_pipeline.models.aggregate import to_participant_level
from obf_psychiatric_pipeline.models.classifiers import make_dummy, make_logreg, make_xgb
from obf_psychiatric_pipeline.models.evaluate import evaluate_predictions
from obf_psychiatric_pipeline.models.relabel import to_binary

# ---------------------------------------------------------------------------
# Feature column lists
# ---------------------------------------------------------------------------

# Existing distributional features (q25 excluded by preprocessing).
DIST_FEATURES = ["mean", "sd", "pctZeros", "median", "q75"]

# 17 temporal features from Phase 1–2.
TEMPORAL_FEATURES = list(FEATURE_NAMES)

# Combined: distributional + temporal.
COMBINED_FEATURES = DIST_FEATURES + TEMPORAL_FEATURES

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
OUT_DIR = Path("results/temporal_experiment")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def _run_one(
    df: pd.DataFrame,
    feature_cols: list[str],
    splits: list,
    clf_name: str,
    clf,
    feature_set: str,
    task: str,
    out_dir: Path,
) -> dict:
    """Run one classifier fold-by-fold with per-fold median imputation.

    Imputation is fit on training data only and applied to test data,
    preventing any leakage of test-set statistics.
    """
    labels = sorted(df["class"].unique().tolist())
    le = LabelEncoder().fit(labels)

    all_true, all_pred, all_proba = [], [], []

    for train_idx, test_idx in splits:
        X_train = df.iloc[train_idx][feature_cols].values.astype(float)
        y_train_str = df.iloc[train_idx]["class"].values
        X_test = df.iloc[test_idx][feature_cols].values.astype(float)
        y_test_str = df.iloc[test_idx]["class"].values

        # Per-fold imputation: fit on train, transform both sides.
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        if clf_name == "xgb":
            clf.fit(X_train, le.transform(y_train_str))
            y_pred_str = le.inverse_transform(clf.predict(X_test).astype(int))
            y_proba = clf.predict_proba(X_test)
        else:
            clf.fit(X_train, y_train_str)
            y_pred_str = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                raw = clf.predict_proba(X_test)
                clf_classes = list(clf.classes_)
                y_proba = np.zeros((len(X_test), len(labels)))
                for i, lbl in enumerate(labels):
                    if lbl in clf_classes:
                        y_proba[:, i] = raw[:, clf_classes.index(lbl)]
            else:
                y_proba = np.ones((len(X_test), len(labels))) / len(labels)

        all_true.extend(y_test_str)
        all_pred.extend(y_pred_str)
        all_proba.append(y_proba)

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_proba = np.vstack(all_proba)

    metrics = evaluate_predictions(y_true, y_pred, y_proba, labels)

    key = f"{feature_set}__{task}__{clf_name}"
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"{key}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "feature_set": feature_set,
        "task": task,
        "classifier": clf_name,
        "n_features": len(feature_cols),
        "macro_f1": metrics["macro_f1"],
        "ci_low": metrics["ci_low"],
        "ci_high": metrics["ci_high"],
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_distributional(cfg) -> pd.DataFrame:
    """Load features.csv, preprocess, aggregate to participant level."""
    metadata, features = load_all(cfg.data.root)
    _, features = preprocess(metadata, features, cfg)
    return to_participant_level(features)  # user, class, feature cols


def _load_temporal(cfg) -> pd.DataFrame:
    """Load raw actigraphy, extract 17 temporal features, return tidy DataFrame."""
    records = load_all_actigraphy(cfg.data.actigraphy_root)
    df = extract_all_features(records)
    df = df.rename_axis("user").reset_index()
    df = df.rename(columns={"label": "class"})
    return df  # user, class, 17 feature cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(CONFIG_PATH)

    # ---- Load feature sets ----
    print("Loading distributional features (features.csv)...")
    dist_df = _load_distributional(cfg)
    print(f"  {len(dist_df)} participants, {len(DIST_FEATURES)} features")

    print("\nLoading raw actigraphy and extracting temporal features...")
    t0 = time.time()
    temporal_df = _load_temporal(cfg)
    print(f"  {len(temporal_df)} participants, {len(TEMPORAL_FEATURES)} features "
          f"({time.time() - t0:.1f}s)")

    nan_counts = temporal_df[TEMPORAL_FEATURES].isna().sum()
    if nan_counts.any():
        print("  NaN counts per feature (will be imputed per fold):")
        for col, n in nan_counts[nan_counts > 0].items():
            print(f"    {col}: {n}")

    print("\nJoining distributional + temporal on user...")
    combined_df = dist_df.merge(
        temporal_df.drop(columns=["class"]),
        on="user",
        how="inner",
    )
    print(f"  {len(combined_df)} participants after inner join "
          f"(distributional: {len(dist_df)}, temporal: {len(temporal_df)})")

    # ---- Run all 18 experiments ----
    feature_sets = {
        "distributional": (dist_df,    DIST_FEATURES),
        "temporal":       (temporal_df, TEMPORAL_FEATURES),
        "combined":       (combined_df, COMBINED_FEATURES),
    }

    summary_rows = []
    print()

    for fs_name, (fs_df, fs_cols) in feature_sets.items():
        # Keep only columns that exist in the DataFrame.
        cols = [c for c in fs_cols if c in fs_df.columns]

        for task_name, task_df in [
            ("3class", fs_df),
            ("2class", to_binary(fs_df.copy())),
        ]:
            splits = make_splits(
                task_df, n_folds=cfg.split.n_folds, seed=cfg.split.seed
            )
            n_classes = len(task_df["class"].unique())
            classifiers = {
                "dummy":  make_dummy(cfg.split.seed),
                "logreg": make_logreg(cfg.split.seed),
                "xgb":    make_xgb(cfg.split.seed, n_classes=n_classes),
            }

            for clf_name, clf in classifiers.items():
                label = f"{fs_name}/{task_name}/{clf_name}"
                print(f"  {label:<40s}", end=" ", flush=True)
                t0 = time.time()
                row = _run_one(
                    df=task_df,
                    feature_cols=cols,
                    splits=splits,
                    clf_name=clf_name,
                    clf=clf,
                    feature_set=fs_name,
                    task=task_name,
                    out_dir=OUT_DIR,
                )
                print(
                    f"F1={row['macro_f1']:.3f} "
                    f"[{row['ci_low']:.3f}-{row['ci_high']:.3f}]  "
                    f"({time.time() - t0:.1f}s)"
                )
                summary_rows.append(row)

    # ---- Summary table ----
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    print("\n" + "=" * 72)
    print("SUMMARY -- per-participant, macro-F1 (95% bootstrap CI)")
    print("=" * 72)

    for task in ["2class", "3class"]:
        baseline = summary[
            (summary["task"] == task)
            & (summary["classifier"] == "dummy")
            & (summary["feature_set"] == "distributional")
        ]["macro_f1"].values[0]

        print(f"\n{'-'*72}")
        print(f"  Task: {task}   (dummy baseline: {baseline:.3f})")
        print(f"{'-'*72}")
        print(f"  {'Feature set':15s}  {'Clf':6s}  {'Macro-F1':8s}  {'95% CI':18s}  {'Lift':>6s}")
        print(f"  {'-'*15}  {'-'*6}  {'-'*8}  {'-'*18}  {'-'*6}")

        for _, row in summary[summary["task"] == task].iterrows():
            lift = row["macro_f1"] - baseline
            print(
                f"  {row['feature_set']:15s}  {row['classifier']:6s}  "
                f"{row['macro_f1']:.3f}     "
                f"[{row['ci_low']:.3f}-{row['ci_high']:.3f}]      "
                f"{lift:+.3f}"
            )

    print(f"\nDetailed metrics saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()