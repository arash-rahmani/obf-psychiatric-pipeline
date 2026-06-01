#!/usr/bin/env python
"""
Regenerate 3-class confusion matrix from a single disclosed fixture fold.

Uses repetition 0, fold 0 of config/folds_repeated/folds_n5_r20.json.
This fold is the disclosed representative; its identity is stated in the
README caption so the figure is reproducible from a named fold, not an
unspecified single split.

Output:
    results/figures/confusion_3class_combined_logreg.png
"""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from obf_psychiatric_pipeline.config import load_config
from obf_psychiatric_pipeline.cv.folds import load_folds, validate_folds
from obf_psychiatric_pipeline.data.loader import load_all
from obf_psychiatric_pipeline.data.preprocess import preprocess
from obf_psychiatric_pipeline.data.raw_loader import load_all_actigraphy
from obf_psychiatric_pipeline.features.extract import extract_all_features
from obf_psychiatric_pipeline.models.aggregate import to_participant_level

REP   = 0   # disclosed repetition index
FOLD  = 0   # disclosed fold index
DIST_FEATURES = ["mean", "sd", "pctZeros", "median", "q75"]
CONFIG_PATH   = Path(__file__).parent.parent / "config" / "config.yaml"
FIXTURE_PATH  = Path(__file__).parent.parent / "config" / "folds_repeated" / "folds_n5_r20.json"
OUT_PATH      = Path("results/figures/confusion_3class_combined_logreg.png")


def main() -> None:
    cfg = load_config(CONFIG_PATH)

    # --- Load features ---
    metadata, features = load_all(cfg.data.root)
    _, features = preprocess(metadata, features, cfg)
    dist_df = to_participant_level(features)

    records  = load_all_actigraphy(cfg.data.actigraphy_root)
    temp_raw = extract_all_features(records)
    temp_df  = temp_raw.rename_axis("user").reset_index()
    temp_df  = temp_df.drop(columns=["label"], errors="ignore")

    combined = dist_df.merge(temp_df, on="user", how="inner")
    combined = combined.sort_values("user").reset_index(drop=True)

    temp_cols    = [c for c in combined.columns
                    if c not in {"user", "class"} and c not in DIST_FEATURES]
    feature_cols = DIST_FEATURES + temp_cols

    le = LabelEncoder()
    y  = le.fit_transform(combined["class"].to_numpy())
    X  = combined[feature_cols].to_numpy(dtype=float)
    ids = combined["user"].to_numpy()

    # --- Load fixture and extract the disclosed fold ---
    folds = load_folds(FIXTURE_PATH)
    validate_folds(folds, ids.tolist())

    test_ids  = set(folds["repetitions"][REP]["folds"][FOLD]["test"])
    test_mask = np.isin(ids, list(test_ids))
    train_mask = ~test_mask

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, random_state=42, solver="lbfgs",
        )),
    ])
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    # --- Normalised confusion matrix ---
    class_names = le.classes_
    n = len(class_names)
    cm = np.zeros((n, n), dtype=float)
    for true, pred in zip(y_te, y_pred):
        cm[true, pred] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm, row_sums, where=row_sums > 0)

    # --- Plot (match existing style) ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(
        "3-class: combined features, logistic regression\n"
        f"(rep {REP}, fold {FOLD} of folds_n5_r20.json)",
        fontsize=10,
    )

    for i in range(n):
        for j in range(n):
            val   = cm_norm[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")
    print(f"Test set: {test_mask.sum()} participants")
    print(f"Classes:  {list(class_names)}")


if __name__ == "__main__":
    main()
