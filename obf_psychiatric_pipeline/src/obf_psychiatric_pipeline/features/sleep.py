"""
Sleep-wake scoring and derived sleep metrics for OBF psychiatric pipeline.

Cole-Kripke (1992) is the primary scorer, applied with Webster (1982)
rescue rules (most implementations omit these; they are included here).
Sadeh (1994) runs in parallel as a sensitivity check.

Nightly rest window
-------------------
Sleep metrics (TST, WASO, SE, SOL) require a defined "time in bed" period.
A fixed clock window (e.g. 21:00–09:00) is cohort-confounded for
psychiatric inpatients: depression carries delayed phase and daytime
bed-rest; schizophrenia carries severe fragmentation and shifted timing.
A fixed window would bake the classification target into the feature.

Instead, the rest window is detected per recording night from the
per-day L5 onset: the start of the least-active 5-hour block that day.
Default window: [L5_onset − 30 min, L5_onset + 6 h]
(asymmetric padding — sleep onset is sharper than morning wake).

tst_24h_hours is computed over the full 24-hour day and does not depend
on window detection; it serves as a sanity check and protects against
window-detection failure on the most fragmented nights.

Device calibration note
-----------------------
Cole-Kripke weights were validated on AMI Motionlogger (PIM mode) counts.
OBF uses Actiwatch.  Absolute sleep-minute accuracy is reduced, but
group separation survives because miscalibration affects all cohorts
uniformly.  This is standard in published psychiatric actigraphy studies.
Sadeh parallel scoring provides a sensitivity check on cohort-level
features.

Phase 2 note
------------
Private helpers imported from temporal.py should move to
features/_helpers.py.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from obf_psychiatric_pipeline.features._helpers import (
    _bin_activity,
    _circular_window_search,
    _count_full_recording_days,
    _full_day_dates,
    _keep_full_days,
    _validate_activity_series,
)


# ---------------------------------------------------------------------------
# Scorer constants
# ---------------------------------------------------------------------------

# Cole-Kripke (1992) weights for lags t−4 through t+3 (8 lags).
# Source: Cole et al., Sleep 15(5):461-469, Table 1.
_CK_WEIGHTS = np.array([404, 598, 326, 441, 1408, 508, 350, 188], dtype=float)
_CK_SCALE = 0.00001
_CK_THRESHOLD = 1.0

# Sadeh (1994) regression coefficients.
# Source: Sadeh et al., Sleep 17(3):201-207.
_SADEH_INTERCEPT = 7.601
_SADEH_MEAN_COEF = -0.065
_SADEH_NAT_COEF = -1.08
_SADEH_SD_COEF = -0.056
_SADEH_LG_COEF = -0.703


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class SleepResult(NamedTuple):
    """Per-day averaged sleep metrics from Cole-Kripke scoring.

    All fields are float.  ``float('nan')`` when recording duration or
    window-detection requirements are not met.

    Attributes
    ----------
    tst_hours : float
        Mean total sleep time within the data-driven nightly rest
        window (hours/day).
    tst_24h_hours : float
        Mean total sleep time over the full 24-hour recording day
        (hours/day).  Independent of window detection; use to
        sanity-check tst_hours.
    waso_minutes : float
        Mean wake after sleep onset: wake minutes between first and last
        sleep epoch within the rest window (minutes/day).
    sleep_efficiency : float
        Mean TST / rest-window duration.  In [0, 1].
    sol_minutes : float
        Mean sleep onset latency: minutes from rest-window start to
        the first sleep epoch (minutes/day).
    """

    tst_hours: float
    tst_24h_hours: float
    waso_minutes: float
    sleep_efficiency: float
    sol_minutes: float


_NAN_SLEEP_RESULT = SleepResult(
    tst_hours=float("nan"),
    tst_24h_hours=float("nan"),
    waso_minutes=float("nan"),
    sleep_efficiency=float("nan"),
    sol_minutes=float("nan"),
)


# ---------------------------------------------------------------------------
# Private: Cole-Kripke scorer
# ---------------------------------------------------------------------------


def _score_cole_kripke(activity: np.ndarray) -> np.ndarray:
    """Apply Cole-Kripke weights to per-minute activity counts.

    Score(t) = 0.00001 × Σ_{i=0}^{7} w[i] × A[t − 4 + i]

    Epochs with Score >= 1 are wake; Score < 1 are sleep.
    Zero-padded at boundaries (equivalent to no activity outside recording).

    Returns np.ndarray of int8: 0 = sleep, 1 = wake.
    """
    padded = np.pad(activity.astype(float), (4, 3), constant_values=0.0)
    windows = sliding_window_view(padded, 8)  # shape (n, 8)
    scores = (windows @ _CK_WEIGHTS) * _CK_SCALE
    return (scores >= _CK_THRESHOLD).astype(np.int8)


def _apply_webster_rules(labels: np.ndarray) -> np.ndarray:
    """Apply Webster (1982) rescue rules to Cole-Kripke output.

    Rescores brief isolated sleep runs within prolonged wake as wake.
    Most Cole-Kripke implementations omit these rules; inclusion here
    matches the original Cole et al. (1992) specification.

    Rules:
        1.  Sleep run of ≤ 1 epoch surrounded by ≥ 4 wake → rescore as wake.
        2.  Sleep run of ≤ 2 epochs surrounded by ≥ 10 wake → rescore as wake.
        3.  Sleep run of ≤ 3 epochs surrounded by ≥ 20 wake → rescore as wake.
    """
    labels = labels.copy()
    if len(labels) == 0:
        return labels

    # Run-length encoding.
    changes = np.where(
        np.concatenate([[True], labels[1:] != labels[:-1], [True]])
    )[0]
    run_starts = changes[:-1]
    run_lengths = np.diff(changes)
    run_values = labels[run_starts]
    n_runs = len(run_lengths)

    for i in range(n_runs):
        if run_values[i] != 0:
            continue  # not a sleep run

        length = int(run_lengths[i])
        wake_before = int(run_lengths[i - 1]) if i > 0 and run_values[i - 1] == 1 else 0
        wake_after = (
            int(run_lengths[i + 1]) if i < n_runs - 1 and run_values[i + 1] == 1 else 0
        )

        if (
            (length <= 1 and wake_before >= 4 and wake_after >= 4)
            or (length <= 2 and wake_before >= 10 and wake_after >= 10)
            or (length <= 3 and wake_before >= 20 and wake_after >= 20)
        ):
            labels[run_starts[i] : run_starts[i] + length] = 1

    return labels


# ---------------------------------------------------------------------------
# Private: Sadeh scorer
# ---------------------------------------------------------------------------


def _score_sadeh(activity: np.ndarray) -> np.ndarray:
    """Apply Sadeh (1994) algorithm to per-minute activity counts.

    PS(t) = 7.601 − 0.065·M − 1.08·Nat − 0.056·SD − 0.703·ln(1 + A_t)

    PS >= 0 → sleep (0); PS < 0 → wake (1).
    M, Nat, SD computed over the 11-epoch window [t−5, t+5].
    Zero-padded at boundaries.
    """
    a = activity.astype(float)
    padded = np.pad(a, (5, 5), constant_values=0.0)
    windows = sliding_window_view(padded, 11)  # shape (n, 11)

    mean_w = windows.mean(axis=1)
    nat_w = ((windows > 50) & (windows <= 100)).sum(axis=1).astype(float)
    sd_w = windows.std(axis=1)
    lg = np.log1p(a)

    ps = (
        _SADEH_INTERCEPT
        + _SADEH_MEAN_COEF * mean_w
        + _SADEH_NAT_COEF * nat_w
        + _SADEH_SD_COEF * sd_w
        + _SADEH_LG_COEF * lg
    )
    return (ps < 0.0).astype(np.int8)


# ---------------------------------------------------------------------------
# Private: rest window detection
# ---------------------------------------------------------------------------


def _detect_rest_onset_minute(
    day_binned: pd.Series,
    bin_minutes: int,
    l5_hours: int,
) -> int:
    """Return the L5 onset for one day as minutes from midnight.

    Parameters
    ----------
    day_binned : pd.Series
        Binned activity for exactly one calendar day (p bins).
    bin_minutes, l5_hours : int
        Bin width in minutes; L5 window size in hours.
    """
    window_bins = (l5_hours * 60) // bin_minutes
    profile = day_binned.to_numpy(dtype=float)
    _, onset_bin = _circular_window_search(profile, window_bins, kind="min")
    return int(onset_bin * bin_minutes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_sleep(
    activity: pd.Series,
    *,
    scorer: str = "cole_kripke",
) -> pd.Series:
    """Score per-minute activity as sleep (0) or wake (1).

    Parameters
    ----------
    activity : pd.Series
        Per-minute activity with a monotonic DatetimeIndex.
        ``NaN`` values are treated as zero activity (wake-conservative).
    scorer : {'cole_kripke', 'sadeh'}, default 'cole_kripke'
        'cole_kripke' : Cole et al. (1992) with Webster rescue rules.
        'sadeh'       : Sadeh et al. (1994) — use for sensitivity checks.

    Returns
    -------
    pd.Series
        0 = sleep, 1 = wake, same index as *activity*.

    Raises
    ------
    ValueError
        If the index is not a DatetimeIndex, or *scorer* is invalid.
    """
    if not isinstance(activity.index, pd.DatetimeIndex):
        raise ValueError(
            f"activity must be indexed by a DatetimeIndex; "
            f"got {type(activity.index).__name__}."
        )
    if scorer not in {"cole_kripke", "sadeh"}:
        raise ValueError(
            f"scorer must be 'cole_kripke' or 'sadeh'; got {scorer!r}."
        )

    a = np.maximum(activity.fillna(0.0).to_numpy(dtype=float), 0.0)

    if scorer == "cole_kripke":
        labels = _apply_webster_rules(_score_cole_kripke(a))
    else:
        labels = _score_sadeh(a)

    return pd.Series(labels.astype(int), index=activity.index, name="sleep_wake")


def sleep_metrics(
    activity: pd.Series,
    *,
    bin_minutes: int = 60,
    min_days: int = 3,
    scorer: str = "cole_kripke",
    l5_hours: int = 5,
    padding_before_min: int = 30,
    padding_after_min: int = 60,
) -> SleepResult:
    """Compute per-day averaged sleep metrics from wrist actigraphy.

    Nightly rest windows are anchored on the per-day L5 onset rather
    than fixed clock times.  See module docstring for rationale.

    Parameters
    ----------
    activity : pd.Series
        Per-minute activity with a monotonic, regularly-spaced DatetimeIndex.
    bin_minutes : int, default 60
        Bin width for per-day L5 detection.  Must evenly divide 1440.
    min_days : int, default 3
        Minimum full recording days required.  Three days yield at least
        one complete cross-midnight window.
    scorer : {'cole_kripke', 'sadeh'}, default 'cole_kripke'
        Scorer for sleep/wake labelling.
    l5_hours : int, default 5
        Window size for least-active period detection.
    padding_before_min : int, default 30
        Minutes of padding before L5 onset (window start).
    padding_after_min : int, default 60
        Minutes of padding after L5 offset (window end).

    Returns
    -------
    SleepResult
        Per-day averages of tst_hours, tst_24h_hours, waso_minutes,
        sleep_efficiency, sol_minutes.  All NaN if min_days is unmet
        or no valid nights are found.

    Raises
    ------
    ValueError
        If the index is not a DatetimeIndex, bin_minutes is invalid,
        or l5_hours × 60 is not divisible by bin_minutes.

    Notes
    -----
    Nights where the extracted window contains fewer than 30 minutes of
    scored labels (recording boundary truncation) are excluded from
    the nightly averages.

    References
    ----------
    Cole, R. J., et al. (1992). Automatic sleep/wake identification from
    wrist activity. *Sleep*, 15(5), 461–469.

    Webster, J. B., Kripke, D. F., Messin, S., Mullaney, D. J., & Wyborney,
    G. (1982). An activity-based sleep monitor system for ambulatory use.
    *Sleep*, 5(4), 389–399.  (Rescue rules.)

    Van Hees, V. T., et al. (2018). Estimating sleep parameters using an
    accelerometer without sleep diary. *Scientific Reports*, 8(1), 12975.
    (Background for data-driven rest-window detection.)
    """
    _validate_activity_series(activity, bin_minutes)

    if (l5_hours * 60) % bin_minutes != 0:
        raise ValueError(
            f"l5_hours={l5_hours} and bin_minutes={bin_minutes} do not "
            f"produce an integer window width."
        )

    s = activity.dropna()
    if s.empty:
        return _NAN_SLEEP_RESULT

    binned = _bin_activity(s, bin_minutes)

    if _count_full_recording_days(binned, bin_minutes) < min_days:
        return _NAN_SLEEP_RESULT

    binned_full = _keep_full_days(binned, bin_minutes)
    full_dates = sorted(_full_day_dates(binned_full, bin_minutes))

    # Score the full per-minute series.
    scored = score_sleep(activity, scorer=scorer)

    # 24-hour TST: total sleep across the full recording, averaged per day.
    tst_24h_hours = float((scored == 0).sum() / len(full_dates) / 60)

    # Per-night windowed metrics.
    date_arr = np.array(binned_full.index.date)
    night_records: list[dict] = []

    for d in full_dates:
        day_binned = binned_full[date_arr == d]
        onset_min = _detect_rest_onset_minute(day_binned, bin_minutes, l5_hours)

        day_midnight = pd.Timestamp(str(d))
        window_start = day_midnight + pd.Timedelta(minutes=onset_min - padding_before_min)
        window_end = day_midnight + pd.Timedelta(
            minutes=onset_min + l5_hours * 60 + padding_after_min
        )

        window_labels = scored.loc[window_start:window_end].to_numpy(dtype=int)

        if len(window_labels) < 30:
            continue  # truncated window; skip this night

        sleep_indices = np.where(window_labels == 0)[0]

        if len(sleep_indices) == 0:
            night_records.append(dict(tst=0, waso=0, se=0.0, sol=len(window_labels)))
            continue

        first_sleep = int(sleep_indices[0])
        last_sleep = int(sleep_indices[-1])
        in_period = window_labels[first_sleep : last_sleep + 1]

        night_records.append(
            dict(
                tst=int((in_period == 0).sum()),
                waso=int((in_period == 1).sum()),
                se=float((in_period == 0).sum() / len(window_labels)),
                sol=first_sleep,
            )
        )

    if not night_records:
        return _NAN_SLEEP_RESULT

    return SleepResult(
        tst_hours=float(np.mean([r["tst"] for r in night_records]) / 60),
        tst_24h_hours=tst_24h_hours,
        waso_minutes=float(np.mean([r["waso"] for r in night_records])),
        sleep_efficiency=float(np.mean([r["se"] for r in night_records])),
        sol_minutes=float(np.mean([r["sol"] for r in night_records])),
    )