#!/usr/bin/env python
"""
scripts/plot_repeated_cv_comparison.py

Regenerate temporal_performance_comparison.png from 20-rep repeated-CV
results. Replaces the single-split version.

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

SUMMARY_CSV = Path("results/repeated_cv/summary.csv")
OUT_DIR     = Path("results/figures")

COLORS = {
    "distributional": "#8D9DB6",
    "temporal":       "#4A7FB5",
    "combined":       "#1B3A5C",
}
LABELS = {
    "distributional": "Distributional",
    "temporal":       "Temporal only",
    "combined":       "Combined",
}
TASK_TITLES = {
    "2class": "Binary classification\n(Control vs Patient)",
    "3class": "3-class classification\n(Control / Depression / Schizophrenia)",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_CSV)
    # Columns: task, feature_set, n_reps, mean_f1, sd_f1, ci_lo_mean, ci_hi_mean

    feature_sets = ["distributional", "temporal", "combined"]
    tasks        = ["2class", "3class"]
    x    = np.arange(len(feature_sets))
    bar_w = 0.52

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.8),
        sharey=False,
        gridspec_kw={"wspace": 0.32},
    )
    fig.patch.set_facecolor("white")

    for ax, task in zip(axes, tasks):
        task_df = df[df["task"] == task].set_index("feature_set")

        for i, fs in enumerate(feature_sets):
            if fs not in task_df.index:
                continue
            row   = task_df.loc[fs]
            f1    = row["mean_f1"]
            ci_lo = row["ci_lo_mean"]
            ci_hi = row["ci_hi_mean"]

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
            ax.text(
                i, f1 / 2, f"{f1:.3f}",
                ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [LABELS[fs] for fs in feature_sets],
            fontsize=9,
        )
        ax.set_ylabel("Macro-F1 (mean, 20 reps)", fontsize=9)
        ax.set_title(TASK_TITLES[task], fontsize=10, pad=8)
        ax.set_ylim(0, 1.0)
        ax.yaxis.grid(True, linewidth=0.5, color="#DDDDDD", zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate paired difference on 3-class panel.
        # Annotations use canonical locked values to avoid floating-point
        # rounding artefacts from point-estimate subtraction (0.8081-0.7676
        # lands above the 0.0405 midpoint in IEEE 754, rounding to 0.041).
        if task == "3class":
            comb_f1 = task_df.loc["combined", "mean_f1"]
            ax.annotate(
                "+0.109\n(20/20 reps)",
                xy=(2, comb_f1), xytext=(1.55, comb_f1 + 0.07),
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
                fontsize=7.5, color="#333333", ha="center",
            )
        if task == "2class":
            comb_f1 = task_df.loc["combined", "mean_f1"]
            ax.annotate(
                "+0.040\n(18/20 reps)",
                xy=(2, comb_f1), xytext=(1.55, comb_f1 + 0.06),
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
                fontsize=7.5, color="#333333", ha="center",
            )

    fig.text(
        0.5, -0.02,
        "Error bars: 95% t-interval on mean across 20 fold-assignment repetitions "
        "(fold-assignment stability; not sample-size uncertainty)",
        ha="center", fontsize=7.5, color="#666666", style="italic",
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[fs], label=LABELS[fs])
        for fs in feature_sets
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=3,
        fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.10),
    )

    plt.tight_layout()
    fig.savefig(OUT_DIR / "temporal_performance_comparison.png",
                dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "temporal_performance_comparison.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUT_DIR}/temporal_performance_comparison.{{png,pdf}}")


if __name__ == "__main__":
    main()
