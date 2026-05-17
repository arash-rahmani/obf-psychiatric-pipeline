#!/usr/bin/env python
"""
scripts/plot_temporal_results.py

Figure 1: Feature set performance comparison.

Two-panel grouped bar chart (binary | 3-class), best classifier per
feature set per task, with 95% bootstrap CI error bars.

Output:
    results/figures/temporal_performance_comparison.png
    results/figures/temporal_performance_comparison.pdf
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SUMMARY_CSV = Path("results/temporal_experiment/summary.csv")
OUT_DIR = Path("results/figures")

COLORS = {
    "distributional": "#8D9DB6",
    "temporal":       "#4A7FB5",
    "combined":       "#1B3A5C",
}

LABELS = {
    "distributional": "Distributional",
    "temporal":       "Temporal",
    "combined":       "Combined",
}

TASK_TITLES = {
    "2class": "Binary classification\n(Control vs Patient)",
    "3class": "3-class classification\n(Control / Depression / Schizophrenia)",
}

CLF_DISPLAY = {"logreg": "LogReg", "xgb": "XGB", "dummy": "Dummy"}


def best_per_feature_task(df: pd.DataFrame) -> pd.DataFrame:
    """Select highest macro-F1 non-dummy classifier per feature_set x task."""
    return (
        df[df["classifier"] != "dummy"]
        .sort_values("macro_f1", ascending=False)
        .groupby(["feature_set", "task"], sort=False)
        .first()
        .reset_index()
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_CSV)
    best = best_per_feature_task(df)

    feature_sets = ["distributional", "temporal", "combined"]
    tasks = ["2class", "3class"]
    x = np.arange(len(feature_sets))
    bar_w = 0.52

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.8),
        sharey=False,
        gridspec_kw={"wspace": 0.32},
    )
    fig.patch.set_facecolor("white")

    for ax, task in zip(axes, tasks):
        task_best = best[best["task"] == task].set_index("feature_set")

        for i, fs in enumerate(feature_sets):
            row = task_best.loc[fs]
            f1, ci_lo, ci_hi = row["macro_f1"], row["ci_low"], row["ci_high"]

            ax.bar(
                i, f1, width=bar_w,
                color=COLORS[fs], alpha=0.90, zorder=3,
                linewidth=0,
            )
            ax.errorbar(
                i, f1,
                yerr=[[f1 - ci_lo], [ci_hi - f1]],
                fmt="none", color="#1a1a1a",
                capsize=4, capthick=1.1, linewidth=1.1, zorder=4,
            )
            clf_label = CLF_DISPLAY.get(row["classifier"], row["classifier"])
            ax.text(
                i, ci_hi + 0.014, clf_label,
                ha="center", va="bottom",
                fontsize=7.5, color="#555555", style="italic",
            )
            ax.text(
                i, f1 / 2, f"{f1:.3f}",
                ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold",
            )

        # Dummy baseline
        dummy_vals = df[
            (df["task"] == task)
            & (df["classifier"] == "dummy")
            & (df["feature_set"] == "distributional")
        ]["macro_f1"]
        if not dummy_vals.empty:
            bl = dummy_vals.values[0]
            ax.axhline(bl, color="#BBBBBB", linewidth=0.85,
                       linestyle="--", zorder=2)
            ax.text(
                len(feature_sets) - 0.05, bl + 0.010,
                f"chance ({bl:.2f})",
                ha="right", va="bottom",
                fontsize=7, color="#999999",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [LABELS[fs] for fs in feature_sets],
            fontsize=10,
        )
        ax.set_title(TASK_TITLES[task], fontsize=10, pad=10,
                     fontweight="normal", color="#222222")
        ax.set_ylabel("Macro-F1" if task == "2class" else "",
                      fontsize=10, color="#444444")
        ax.set_ylim(0.30, 1.02)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.10))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax.grid(axis="y", which="major", color="#EBEBEB",
                linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#CCCCCC")
        ax.tick_params(axis="both", length=0)
        ax.tick_params(axis="y", labelsize=9, labelcolor="#555555")

    # Shared legend
    patches = [
        mpatches.Patch(color=COLORS[fs], alpha=0.90, label=LABELS[fs])
        for fs in feature_sets
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=3,
        frameon=False, fontsize=9.5,
        bbox_to_anchor=(0.5, -0.05),
        handlelength=1.4, handleheight=0.9,
    )

    fig.suptitle(
        "Psychiatric classification performance by feature set\n"
        "OBF-Psychiatric actigraphy dataset  |  n = 76  |  "
        "5-fold GroupKFold  |  95% bootstrap CI",
        fontsize=9.5, color="#444444",
        y=1.03, fontweight="normal",
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    for fmt in ("png", "pdf"):
        path = OUT_DIR / f"temporal_performance_comparison.{fmt}"
        fig.savefig(
            path,
            dpi=180 if fmt == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"Saved: {path}")

    plt.close()


if __name__ == "__main__":
    main()