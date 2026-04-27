"""Confusion matrix plots for the four headline experiments."""
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

EXPERIMENTS = [
    ("2class", "per_participant", "logreg"),
    ("2class", "per_participant", "xgb"),
    ("3class", "per_participant", "logreg"),
    ("3class", "per_participant", "xgb"),
]


def plot_confusion_matrix(cm_data: dict, title: str, out_path: Path) -> None:
    labels = cm_data["labels"]
    matrix = np.array(cm_data["matrix"])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=11)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    metrics_dir = Path("results/models/metrics")
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    for track, agg, clf in EXPERIMENTS:
        key = f"{track}__{agg}__{clf}"
        with open(metrics_dir / f"{key}.json") as f:
            metrics = json.load(f)

        title = f"{track} | {agg} | {clf}\nmacro-F1: {metrics['macro_f1']:.3f}"
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            title=title,
            out_path=out_dir / f"cm_{key}.png",
        )


if __name__ == "__main__":
    main()