#!/usr/bin/env python3
"""
Cosinor acrophase distribution by diagnostic cohort.

Mirrors the loading chain of run_repeated_cv.py exactly:
  - distributional path applies the min_days filter (excludes depression_8)
  - temporal path extracts acrophase from raw actigraphy
  - inner join produces the canonical n=76 participant set

Output: results/figures/acrophase_by_cohort.png (300 DPI)

Run from pipeline root:
    python3 plot_acrophase.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from obf_psychiatric_pipeline.config import load_config
from obf_psychiatric_pipeline.data.loader import load_all
from obf_psychiatric_pipeline.data.preprocess import preprocess
from obf_psychiatric_pipeline.data.raw_loader import load_all_actigraphy
from obf_psychiatric_pipeline.features.extract import extract_all_features
from obf_psychiatric_pipeline.models.aggregate import to_participant_level

# ── constants ────────────────────────────────────────────────────────────────
CONFIG        = Path("config/config.yaml")
OUTPUT        = Path("results/figures/acrophase_by_cohort.png")
COHORT_ORDER  = ["control", "depression", "schizophrenia"]
COHORT_LABELS = ["Control", "Depression", "Schizophrenia"]
COLORS        = ["#7a8fa6", "#4e7db0", "#c0614b"]   # neutral / blue / brick
DIST_FEATURES = ["mean", "sd", "pctZeros", "median", "q75"]
LABEL_COL     = "class"


# ── data loading ─────────────────────────────────────────────────────────────
def _load_data(cfg) -> pd.DataFrame:
    """Return n=76 DataFrame with 'user', 'class', 'cosinor_acrophase_hours'."""

    # Distributional path: preprocess applies the min_days filter
    # (depression_8, 5 recording days, is dropped here).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metadata, features = load_all(cfg.data.root)
        _, features = preprocess(metadata, features, cfg)
    dist_df = to_participant_level(features)           # user, class, dist cols

    # Temporal path: extract all 17 features from raw actigraphy.
    records = load_all_actigraphy(cfg.data.actigraphy_root)
    temp_df = extract_all_features(records)
    temp_df = temp_df.rename_axis("user").reset_index()
    temp_df = temp_df.rename(columns={"label": LABEL_COL})
    temp_df = temp_df.drop(columns=[LABEL_COL], errors="ignore")

    # Inner join: temporal df inherits the min_days exclusion from dist_df.
    merged = dist_df.merge(temp_df, on="user", how="inner")

    # R² filter: remove participants with unreliable cosinor fits
    R2_THRESHOLD = 0.5
    n_before = len(merged)
    merged = merged[merged["cosinor_r_squared"] >= R2_THRESHOLD].copy()
    print(f"R² >= {R2_THRESHOLD} filter: {len(merged)} of {n_before} retained")
    

    return merged


# ── plot ──────────────────────────────────────────────────────────────────────
def plot(df: pd.DataFrame, out: Path) -> None:
    """Violin + jitter plot, acrophase in clock hours, three cohorts."""

    data  = [df.loc[df[LABEL_COL] == c, "cosinor_acrophase_hours"].dropna().values
             for c in COHORT_ORDER]
    ns    = [len(d) for d in data]
    rng   = np.random.default_rng(42)
    pos   = list(range(len(COHORT_ORDER)))

    fig, ax = plt.subplots(figsize=(6, 5))

    # ── violins ──────────────────────────────────────────────────────────────
    parts = ax.violinplot(
        data,
        positions=pos,
        showmedians=True,
        showextrema=False,
        widths=0.55,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[i])
        pc.set_edgecolor("white")
        pc.set_alpha(0.50)
        pc.set_linewidth(0.8)

    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2.2)
    parts["cmedians"].set_zorder(4)

    # ── jittered participant points ──────────────────────────────────────────
    for i, (d, col) in enumerate(zip(data, COLORS)):
        jitter = rng.uniform(-0.10, 0.10, len(d))
        ax.scatter(
            np.full(len(d), i) + jitter,
            d,
            color=col,
            s=24,
            alpha=0.85,
            linewidths=0.5,
            edgecolors="white",
            zorder=3,
        )

    # ── axes and labels ──────────────────────────────────────────────────────
    ax.set_yticks([6, 12, 18])
    ax.set_yticklabels(["06:00", "12:00", "18:00"], fontsize=10)
    ax.set_ylabel("Clock hour of activity peak (acrophase)", fontsize=11)

    ax.set_xticks(pos)
    ax.set_xticklabels(
        [f"{lbl}\n(n = {n})" for lbl, n in zip(COHORT_LABELS, ns)],
        fontsize=11,
    )
    ax.set_xlim(-0.6, len(COHORT_ORDER) - 0.4)
    ax.set_title(
    "Cosinor acrophase by diagnostic cohort\n"
    r"(participants with cosinor $R^2 \geq 0.5$; n = 72 of 76)",
    fontsize=11,
    pad=10,
    )

    # ── grid and spine cleanup ───────────────────────────────────────────────
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.55, color="#aaaaaa")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    ax.set_ylim(6, 20)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_config(CONFIG)
    df  = _load_data(cfg)

    counts = df[LABEL_COL].value_counts().sort_index().to_dict()
    print(f"Cohort counts: {counts}")
    assert len(df) >= 70, f"Unexpectedly few participants after R² filter: {len(df)}"
    print(f"Final sample: {len(df)} participants (after R² >= 0.5 filter)")
    

    n_nan = df["cosinor_acrophase_hours"].isna().sum()
    if n_nan:
        print(f"Warning: {n_nan} NaN acrophase value(s) excluded from violin.")

    for c, lbl in zip(COHORT_ORDER, COHORT_LABELS):
        vals = df.loc[df[LABEL_COL] == c, "cosinor_acrophase_hours"].dropna()
        print(f"  {lbl:15s}  median acrophase = {vals.median():.2f} h  "
              f"({vals.min():.2f} – {vals.max():.2f})")

    plot(df, OUTPUT)


if __name__ == "__main__":
    main()