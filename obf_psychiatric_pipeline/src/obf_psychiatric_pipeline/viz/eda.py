"""
EDA module — five core exploratory plots for the OBF psychiatric pipeline.

Design contract:
    - One function per plot. Pure: data in, out_path out, returns None.
    - Style constants imported from viz/__init__.py.
    - No path resolution logic here — caller passes resolved paths.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from obf_psychiatric_pipeline.viz import CLASS_PALETTE, FIGSIZE_DEFAULT, save_fig

FEATURE_COLS = ("mean", "sd", "pctZeros", "median", "q25", "q75")
COHORTS_WITH_FEATURES = ("control", "depression", "schizophrenia")


def plot_class_balance(
    metadata: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot 1: participants vs feature-rows per cohort."""
    cohorts = list(COHORTS_WITH_FEATURES)
    n_participants = [metadata[c]["number"].nunique() for c in cohorts]
    n_rows = [int((features["class"] == c).sum()) for c in cohorts]

    x = np.arange(len(cohorts))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    bars1 = ax.bar(x - width / 2, n_participants, width, label="Participants",
                   color=[CLASS_PALETTE[c] for c in cohorts], alpha=0.9)
    bars2 = ax.bar(x + width / 2, n_rows, width, label="Feature rows (days)",
                   color=[CLASS_PALETTE[c] for c in cohorts], alpha=0.4)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Count")
    ax.set_title("Class balance: participants vs recorded days")
    ax.legend()

    save_fig(fig, out_path)


def plot_days_per_participant(
    metadata: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """Plot 2: recording days distribution per cohort."""
    cohorts = list(COHORTS_WITH_FEATURES)
    data = [metadata[c]["days"].dropna().values for c in cohorts]

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

    bp = ax.boxplot(data, positions=range(len(cohorts)), patch_artist=True,
                    widths=0.4, showfliers=False)

    for patch, cohort in zip(bp["boxes"], cohorts):
        patch.set_facecolor(CLASS_PALETTE[cohort])
        patch.set_alpha(0.6)

    rng = np.random.default_rng(42)
    for i, (vals, cohort) in enumerate(zip(data, cohorts)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(i + jitter, vals, color=CLASS_PALETTE[cohort],
                   alpha=0.6, s=30, zorder=3)

    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Recording days")
    ax.set_title("Days per participant by cohort")

    save_fig(fig, out_path)


def plot_feature_distributions(
    features: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot 3: 2x3 violin grid, one panel per feature, colored by class."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    cohorts = list(COHORTS_WITH_FEATURES)

    for ax, feat in zip(axes, FEATURE_COLS):
        groups = [features.loc[features["class"] == c, feat].dropna().values
                  for c in cohorts]

        parts = ax.violinplot(groups, positions=range(len(cohorts)),
                              showmedians=True, showextrema=True)

        for pc, cohort in zip(parts["bodies"], cohorts):
            pc.set_facecolor(CLASS_PALETTE[cohort])
            pc.set_alpha(0.7)

        for partname in ("cmedians", "cbars", "cmins", "cmaxes"):
            if partname in parts:
                parts[partname].set_color("black")
                parts[partname].set_linewidth(0.8)

        ax.set_xticks(range(len(cohorts)))
        ax.set_xticklabels(cohorts, fontsize=8)
        ax.set_title(feat, fontsize=10)

    patches = [mpatches.Patch(color=CLASS_PALETTE[c], label=c) for c in cohorts]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Feature distributions by class", fontsize=13)

    save_fig(fig, out_path)


def plot_feature_correlation(
    features: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot 4: 6x6 correlation heatmap of the six features."""
    corr = features[list(FEATURE_COLS)].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(FEATURE_COLS, fontsize=9)

    for i in range(len(FEATURE_COLS)):
        for j in range(len(FEATURE_COLS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.6 else "black")

    ax.set_title("Feature correlation matrix")
    save_fig(fig, out_path)


def plot_pca_projection(
    features: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot 5: PC1 vs PC2 scatter after StandardScaler, colored by class."""
    X = features[list(FEATURE_COLS)].dropna()
    labels = features.loc[X.index, "class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

    for cohort in COHORTS_WITH_FEATURES:
        mask = labels == cohort
        ax.scatter(components[mask, 0], components[mask, 1],
                   color=CLASS_PALETTE[cohort], label=cohort,
                   alpha=0.5, s=20)

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title("PCA projection of standardized features")
    ax.legend()

    save_fig(fig, out_path)


def run_eda(
    metadata: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Run all five EDA plots, saving each PNG to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_class_balance(metadata, features, out_dir / "class_balance.png")
    plot_days_per_participant(metadata, out_dir / "days_per_participant.png")
    plot_feature_distributions(features, out_dir / "feature_distributions.png")
    plot_feature_correlation(features, out_dir / "feature_correlation.png")
    plot_pca_projection(features, out_dir / "pca_projection.png")