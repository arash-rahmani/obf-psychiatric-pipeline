"""SHAP analysis on best model: binary per-participant XGBoost."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder

from obf_psychiatric_pipeline.config import load_config
from obf_psychiatric_pipeline.data.loader import load_all
from obf_psychiatric_pipeline.data.preprocess import preprocess
from obf_psychiatric_pipeline.models.aggregate import to_participant_level
from obf_psychiatric_pipeline.models.relabel import to_binary
from obf_psychiatric_pipeline.models.classifiers import make_xgb

FEATURE_COLS = ["mean", "sd", "pctZeros", "median", "q75"]


def main() -> None:
    cfg = load_config(Path("config/config.yaml"))
    metadata, features = load_all(cfg.data.root)
    _, features = preprocess(metadata, features, cfg)

    # Binary per-participant — the headline model
    features_bin = to_binary(features)
    features_agg = to_participant_level(features_bin)

    feat_cols = [c for c in FEATURE_COLS if c in features_agg.columns]
    X = features_agg[feat_cols].values
    y_str = features_agg["class"].values

    le = LabelEncoder().fit(y_str)
    y = le.transform(y_str)

    clf = make_xgb(seed=cfg.split.seed, n_classes=2)
    clf.fit(X, y)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # For binary XGBoost, shap_values is 2D (n_samples, n_features)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # class 1 = patient
    else:
        shap_vals = shap_values

    out_dir = Path(cfg.output.results_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_vals, X,
        feature_names=feat_cols,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP feature importance — binary classifier (control vs patient)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot saved to {out_dir / 'shap_summary.png'}")


if __name__ == "__main__":
    main()