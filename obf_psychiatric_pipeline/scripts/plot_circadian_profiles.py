#!/usr/bin/env python
"""
scripts/plot_circadian_profiles.py

Figure 2: Mean 24-hour activity profile per diagnostic cohort.

Aggregates raw per-minute actigraphy across all participants and days,
plots smooth group-mean curves with SEM ribbons for controls,
depression, and schizophrenia.

Output:
    results/figures/circadian_activity_profiles.png
    results/figures/circadian_activity_profiles.pdf
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from obf_psychiatric_pipeline.config import load_config
from obf_psychiatric_pipeline.data.raw_loader import load_all_actigraphy

CONFIG_PATH = Path("config/config.yaml")
OUT_DIR = Path("results/figures")

COHORT_STYLE = {
    "control":      {"color": "#4A7FB5", "label": "Control",      "zorder": 4},
    "depression":   {"color": "#E07B3A", "label": "Depression",   "zorder": 3},
    "schizophrenia":{"color": "#6B4C8A", "label": "Schizophrenia","zorder": 2},
}

SMOOTH_WINDOW = 21   # minutes — smooths over ~20 min, preserves circadian shape


def build_minute_profiles(records) -> dict[str, pd.DataFrame]:
    """
    For each cohort, build a (1440 x n_participants) DataFrame of
    mean activity per minute of day, averaged across all days per participant.

    record.activity is a named Series with DatetimeIndex (confirmed).
    """
    cohort_profiles: dict[str, list[pd.Series]] = {
        "control": [], "depression": [], "schizophrenia": [],
    }

    for record in records:
        if record.label not in cohort_profiles:
            continue

        series = record.activity          # Series, index = DatetimeIndex
        minutes = series.index.hour * 60 + series.index.minute

        tmp = pd.DataFrame({
            "minute_of_day": minutes,
            "activity":      series.values,
        })

        profile = (
            tmp.groupby("minute_of_day")["activity"]
            .mean()
            .reindex(range(1440), fill_value=0.0)
        )
        cohort_profiles[record.label].append(profile)

    return {
        label: pd.DataFrame(profiles)
        for label, profiles in cohort_profiles.items()
        if profiles
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(CONFIG_PATH)
    print("Loading raw actigraphy...")
    records = load_all_actigraphy(cfg.data.actigraphy_root)
    print(f"  {len(records)} participants loaded")

    print("Building minute-of-day profiles per cohort...")
    profiles = build_minute_profiles(records)

    hours = np.arange(1440) / 60.0   # x-axis in hours

    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("white")

    for cohort, style in COHORT_STYLE.items():
        if cohort not in profiles:
            continue

        mat = profiles[cohort].values          # shape: (n_participants, 1440)
        group_mean = mat.mean(axis=0)
        group_sem  = mat.std(axis=0) / np.sqrt(mat.shape[0])

        # Smooth mean and bounds
        mean_smooth = uniform_filter1d(group_mean, size=SMOOTH_WINDOW, mode="wrap")
        lo_smooth   = uniform_filter1d(group_mean - group_sem, size=SMOOTH_WINDOW, mode="wrap")
        hi_smooth   = uniform_filter1d(group_mean + group_sem, size=SMOOTH_WINDOW, mode="wrap")

        n = mat.shape[0]   # participants, not minutes
        ax.fill_between(
            hours, lo_smooth, hi_smooth,
            color=style["color"], alpha=0.14,
            zorder=style["zorder"],
        )
        ax.plot(
            hours, mean_smooth,
            color=style["color"],
            linewidth=1.8,
            label=f"{style['label']} (n={n})",
            zorder=style["zorder"],
        )

    # Clock ticks
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels(
        [f"{h:02d}:00" for h in range(0, 25, 3)],
        fontsize=9, color="#555555",
    )
    ax.set_xlim(0, 24)

    ax.set_ylabel("Mean activity (counts/min)", fontsize=10, color="#444444")
    ax.set_xlabel("Time of day", fontsize=10, color="#444444")
    ax.set_title(
        "24-hour motor activity profiles by diagnostic cohort\n"
        "OBF-Psychiatric actigraphy dataset  |  group mean ± SEM  |  "
        "smoothed over 20-minute window",
        fontsize=9.5, color="#444444", pad=10, fontweight="normal",
    )

    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(axis="y", color="#EBEBEB", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CCCCCC")
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="y", labelsize=9, labelcolor="#555555")

    ax.legend(
        frameon=False, fontsize=9.5,
        loc="upper right",
    )

    plt.tight_layout()

    for fmt in ("png", "pdf"):
        path = OUT_DIR / f"circadian_activity_profiles.{fmt}"
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