from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from obf_psychiatric_pipeline.config import Config
from obf_psychiatric_pipeline.data.loader import load_all
from obf_psychiatric_pipeline.data.preprocess import preprocess
from obf_psychiatric_pipeline.data.split import make_splits
from obf_psychiatric_pipeline.models.aggregate import to_participant_level
from obf_psychiatric_pipeline.models.relabel import to_binary
from obf_psychiatric_pipeline.models.classifiers import make_dummy, make_logreg, make_xgb
from obf_psychiatric_pipeline.models.evaluate import evaluate_predictions

FEATURE_COLS_ALL = ["mean", "sd", "pctZeros", "median", "q75"]


def _get_feature_cols(features: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS_ALL if c in features.columns]


def _run_one_experiment(
    features: pd.DataFrame,
    track: str,
    aggregation: str,
    clf_name: str,
    clf,
    splits: list,
    out_dir: Path,
) -> dict:
    feature_cols = _get_feature_cols(features)
    labels = sorted(features["class"].unique().tolist())
    le = LabelEncoder().fit(labels)

    all_true, all_pred, all_proba, all_users = [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = features.iloc[train_idx][feature_cols].values
        y_train_str = features.iloc[train_idx]["class"].values
        X_test  = features.iloc[test_idx][feature_cols].values
        y_test_str  = features.iloc[test_idx]["class"].values
        users_test = features.iloc[test_idx]["user"].values

        # XGBoost needs numeric labels
        y_train = le.transform(y_train_str)
        y_test  = le.transform(y_test_str)

        if clf_name == "xgb":
            clf.fit(X_train, y_train)
            y_pred_enc = clf.predict(X_test)
            y_pred_str = le.inverse_transform(y_pred_enc.astype(int))
            y_proba = clf.predict_proba(X_test)
        else:
            clf.fit(X_train, y_train_str)
            y_pred_str = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                y_proba_raw = clf.predict_proba(X_test)
                clf_classes = list(clf.classes_)
                y_proba = np.zeros((len(X_test), len(labels)))
                for i, label in enumerate(labels):
                    if label in clf_classes:
                        y_proba[:, i] = y_proba_raw[:, clf_classes.index(label)]
            else:
                y_proba = np.ones((len(X_test), len(labels))) / len(labels)

        all_true.extend(y_test_str)
        all_pred.extend(y_pred_str)
        all_proba.append(y_proba)
        all_users.extend(users_test)

    y_true  = np.array(all_true)
    y_pred  = np.array(all_pred)
    y_proba = np.vstack(all_proba)

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({        "user":   all_users,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    for i, label in enumerate(labels):
        pred_df[f"proba_{label}"] = y_proba[:, i]

    pred_key = f"{track}__{aggregation}__{clf_name}"
    pred_df.to_csv(pred_dir / f"{pred_key}.csv", index=False)

    metrics = evaluate_predictions(y_true, y_pred, y_proba, labels)

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"{pred_key}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "track":       track,
        "aggregation": aggregation,
        "classifier":  clf_name,
        "macro_f1":    metrics["macro_f1"],
        "ci_low":      metrics["ci_low"],
        "ci_high":     metrics["ci_high"],
    }


def run_experiments(
    features: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    tracks = {
        "3class": features,
        "2class": to_binary(features),
    }

    for track_name, track_features in tracks.items():
        aggregations = {
            "per_day":         track_features,
            "per_participant": to_participant_level(track_features),
        }

        for agg_name, agg_features in aggregations.items():
            splits = make_splits(
                agg_features,
                n_folds=cfg.split.n_folds,
                seed=cfg.split.seed,
            )

            n_classes = len(agg_features["class"].unique())
            classifiers = {
                "dummy":  make_dummy(cfg.split.seed),
                "logreg": make_logreg(cfg.split.seed),
                "xgb":    make_xgb(cfg.split.seed, n_classes=n_classes),
            }

            for clf_name, clf in classifiers.items():
                print(f"  Running {track_name} / {agg_name} / {clf_name} ...")
                row = _run_one_experiment(
                    features=agg_features,
                    track=track_name,
                    aggregation=agg_name,
                    clf_name=clf_name,
                    clf=clf,
                    splits=splits,
                    out_dir=out_dir,
                )
                summary_rows.append(row)
                print(f"    macro-F1: {row['macro_f1']:.3f} "
                      f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSummary written to {out_dir / 'summary.csv'}")
    return summary
