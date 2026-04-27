"""ROC curve plots — binary (standard) and ternary (one-vs-rest)."""
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Best models only: logreg for both tasks
BINARY_KEY  = "2class__per_participant__logreg"
TERNARY_KEY = "3class__per_participant__logreg"
TERNARY_LABELS = ["control", "depression", "schizophrenia"]
BINARY_LABELS  = ["control", "patient"]

PALETTE = {
    "control":       "#4C72B0",
    "depression":    "#C44E52",
    "schizophrenia": "#8172B2",
    "patient":       "#C44E52",
}


def _load_predictions(pred_dir: Path, key: str) -> pd.DataFrame:
    return pd.read_csv(pred_dir / f"{key}.csv")


def plot_binary_roc(df: pd.DataFrame, out_path: Path) -> None:
    y_true = (df["y_true"] == "patient").astype(int).values
    y_score = df["proba_patient"].values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE["patient"], lw=2,
            label=f"control vs patient (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — binary classifier (logreg, per-participant)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_ternary_roc(df: pd.DataFrame, out_path: Path) -> None:
    y_true = df["y_true"].values
    y_bin = label_binarize(y_true, classes=TERNARY_LABELS)

    fig, ax = plt.subplots(figsize=(6, 5))

    for i, label in enumerate(TERNARY_LABELS):
        col = f"proba_{label}"
        if col not in df.columns:
            continue
        y_score = df[col].values
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[label], lw=2,
                label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — 3-class one-vs-rest (logreg, per-participant)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    pred_dir = Path("results/models/predictions")
    out_dir  = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_binary  = _load_predictions(pred_dir, BINARY_KEY)
    df_ternary = _load_predictions(pred_dir, TERNARY_KEY)

    plot_binary_roc(df_binary,   out_dir / "roc_binary.png")
    plot_ternary_roc(df_ternary, out_dir / "roc_ternary.png")


if __name__ == "__main__":
    main()