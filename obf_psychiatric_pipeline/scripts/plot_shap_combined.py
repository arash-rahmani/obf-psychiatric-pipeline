#!/usr/bin/env python
"""
scripts/plot_shap_combined.py

Figure 4: SHAP feature attribution on the combined-feature XGBoost model.

Trains XGBoost on all participants (no CV; this is for interpretability,
not evaluation), computes SHAP values, and produces two summary plots:

    1. Binary task   (control vs patient) — comparable to original Phase 1 SHAP
    2. 3-class task  (control / depression / schizophrenia) — disorder-specific drivers

Output:
    results/figures/shap_summary_combined_binary.png/.pdf
    results/figures/shap_summary_combined_3class.png/.pdf
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from obf_psychiatric_pipeline.config import load_config
from obf_psychiatric_pipeline.data.loader import load_all
from obf_psychiatric_pipeline.data.preprocess import preprocess
from obf_psychiatric_pipeline.data.raw_loader import load_all_actigraphy
from obf_psychiatric_pipeline.features.extract import FEATURE_NAMES, extract_all_features
from obf_psychiatric_pipeline.models.aggregate import to_participant_level
from obf_psychiatric_pipeline.models.classifiers import make_xgb
from obf_psychiatric_pipeline.models.relabel import to_binary

CONFIG_PATH = Path("config/config.yaml")
OUT_DIR = Path("results/figures")

DIST_FEATURES = ["mean", "sd", "pctZeros", "median", "q75"]
TEMPORAL_FEATURES = list(FEATURE_NAMES)
COMBINED_FEATURES = DIST_FEATURES + TEMPORAL_FEATURES

# Pretty names for figure axes
FEATURE_DISPLAY = {
    "mean": "Mean activity",
    "sd": "SD activity",
    "pctZeros": "% zeros (sedentary)",
    "median": "Median activity",
    "q75": "75th percentile",
    "is": "Interdaily stability (IS)",
    "iv": "Intradaily variability (IV)",
    "l5_value": "L5 activity",
    "l5_onset_hour": "L5 onset (h)",
    "m10_value": "M10 activity",
    "m10_onset_hour": "M10 onset (h)",
    "amplitude": "Amplitude (M10-L5)",
    "relative_amplitude": "Relative amplitude",
    "cosinor_mesor": "Cosinor mesor",
    "cosinor_amplitude": "Cosinor amplitude",
    "cosinor_acrophase_hours": "Cosinor acrophase (h)",
    "cosinor_r_squared": "Cosinor R²",
    "tst_hours": "Total sleep time (h)",
    "tst_24h_hours": "TST 24h (h)",
    "waso_minutes": "WASO (min)",
    "sleep_efficiency": "Sleep efficiency",
    "sol_minutes": "Sleep onset latency (min)",
}


def build_combined_matrix(cfg) -> tuple[pd.DataFrame, list[str]]:
    """Load distributional + temporal, return combined matrix and feature list."""
    metadata, dist_features = load_all(cfg.data.root)
    _, dist_features = preprocess(metadata, dist_features, cfg)
    dist_part = to_participant_level(dist_features)

    records = load_all_actigraphy(cfg.data.actigraphy_root)
    temporal_df = extract_all_features(records).rename_axis("user").reset_index()
    temporal_df = temporal_df.rename(columns={"label": "class"})

    combined = dist_part.merge(
        temporal_df.drop(columns=["class"]),
        on="user", how="inner",
    )
    cols = [c for c in COMBINED_FEATURES if c in combined.columns]
    return combined, cols


def _custom_summary_plot(
    shap_values: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    title: str,
    out_stem: Path,
    max_display: int = 15,
) -> None:
    """Render and save SHAP summary plot with our own styling."""
    pretty_names = [FEATURE_DISPLAY.get(f, f) for f in feature_names]

    plt.figure(figsize=(8.5, 5.5))
    shap.summary_plot(
        shap_values,
        feature_matrix,
        feature_names=pretty_names,
        max_display=max_display,
        show=False,
        plot_size=None,
        color_bar_label="Feature value",
    )

    fig = plt.gcf()
    fig.patch.set_facecolor("white")
    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10, color="#444")
    ax.tick_params(axis="y", labelsize=9.5)
    ax.tick_params(axis="x", labelsize=9.5, labelcolor="#555")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.suptitle(title, fontsize=10, color="#444", y=1.00, fontweight="normal")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    for fmt in ("png", "pdf"):
        path = out_stem.with_suffix(f".{fmt}")
        fig.savefig(path,
                    dpi=180 if fmt == "png" else None,
                    bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
    plt.close(fig)


def run_shap_for_task(
    df: pd.DataFrame,
    feature_cols: list[str],
    task_name: str,
    title: str,
    out_stem: Path,
) -> None:
    """Train XGBoost on all data, compute SHAP, save plot."""
    print(f"\n--- {task_name} ---")
    X = df[feature_cols].values.astype(float)

    # Impute NaN (some temporal features may be NaN for short recordings)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    labels = sorted(df["class"].unique().tolist())
    le = LabelEncoder().fit(labels)
    y = le.transform(df["class"].values)
    n_classes = len(labels)

    cfg = load_config(CONFIG_PATH)
    clf = make_xgb(seed=cfg.split.seed, n_classes=n_classes)
    print(f"  Training XGBoost on {X.shape[0]} participants, {X.shape[1]} features...")
    clf.fit(X, y)
    print(f"  Training accuracy: {clf.score(X, y):.3f}  (interpretability only — NOT a generalisation estimate)")

    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # For multiclass, shap returns (n_samples, n_features, n_classes) or list of arrays.
    # We want per-class plots for multiclass, single plot for binary.
    if n_classes == 2:
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        _custom_summary_plot(
            shap_values, X, feature_cols,
            title=title,
            out_stem=out_stem,
        )
    else:
        # Multiclass: render one panel per class with shared layout
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv_list = [shap_values[:, :, k] for k in range(n_classes)]
        else:
            sv_list = list(shap_values)

        for k, label in enumerate(labels):
            class_title = f"{title}\nClass: {label}"
            class_stem = out_stem.with_name(f"{out_stem.name}_{label}")
            _custom_summary_plot(
                sv_list[k], X, feature_cols,
                title=class_title,
                out_stem=class_stem,
            )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(CONFIG_PATH)

    print("Building combined feature matrix...")
    combined, cols = build_combined_matrix(cfg)
    print(f"  {len(combined)} participants, {len(cols)} features")

    # Binary task
    binary_df = to_binary(combined.copy())
    run_shap_for_task(
        binary_df, cols,
        task_name="Binary (control vs patient)",
        title="SHAP feature importance — combined features, binary classifier",
        out_stem=OUT_DIR / "shap_summary_combined_binary",
    )

    # 3-class task
    run_shap_for_task(
        combined, cols,
        task_name="3-class (control / depression / schizophrenia)",
        title="SHAP feature importance — combined features, 3-class classifier",
        out_stem=OUT_DIR / "shap_summary_combined_3class",
    )


if __name__ == "__main__":
    main()