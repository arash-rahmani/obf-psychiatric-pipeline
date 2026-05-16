"""
Temporal and circadian features for OBF psychiatric pipeline.

Functions here are stateless and single-participant.  The caller is
responsible for splitting multi-participant data upstream and
collecting results into the feature matrix downstream.

Feature families
----------------
Nonparametric rhythm metrics (van Someren 1999):
    interdaily_stability       — coupling to the 24-hour cycle
    intradaily_variability     — within-day fragmentation        [not yet implemented]

Nonparametric amplitude metrics (Witting 1990):
    least_active_period        — L-window (L5 by default)        [not yet implemented]
    most_active_period         — M-window (M10 by default)       [not yet implemented]

Cosinor model parameters live in ``cosinor.py``;
sleep-wake estimation lives in ``sleep.py`` (both forthcoming).

Input convention
----------------
All public functions accept a ``pd.Series`` indexed by a monotonic,
regularly-spaced ``DatetimeIndex``.  Values are non-negative activity
counts.  For OBF data this is per-minute wrist-worn actigraphy.
Multi-participant data must be split before calling these functions.

NaN policy
----------
Insufficient-data conditions → return ``float('nan')``.
Structural errors            → raise ``ValueError``.
"""

from __future__ import annotations

import datetime
from typing import NamedTuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


class WindowResult(NamedTuple):
    """Result of an L5/M10 consecutive-window search.

    Attributes
    ----------
    value : float
        Mean activity in the selected window.
        ``float('nan')`` when recording-duration requirements are unmet.
    onset : datetime.time | None
        Clock time at which the selected window begins.
        ``None`` when recording-duration requirements are unmet.
    """

    value: float
    onset: datetime.time | None


from obf_psychiatric_pipeline.features._helpers import (
    _bin_activity,
    _circular_window_search,
    _count_full_recording_days,
    _full_day_dates,
    _keep_full_days,
    _validate_activity_series,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def interdaily_stability(
    activity: pd.Series,
    *,
    bin_minutes: int = 60,
    min_days: int = 7,
) -> float:
    """Compute interdaily stability (IS) of an activity time series.

    IS quantifies the coupling of the rest-activity rhythm to the
    24-hour environmental cycle.  It is the ratio of the variance of
    the averaged 24-hour profile to the variance of the full signal.

    Formula (van Someren 1999):

        IS = [ N · Σ_h (x̄_h − x̄)² ] / [ p · Σ_i (x_i − x̄)² ]

    where:
        N    = total binned samples across all full recording days
        p    = bins per 24-hour day  (= 1440 // bin_minutes)
        x_i  = individual binned activity values,  i = 1 .. N
        x̄_h  = mean of bin position h averaged across days,  h = 0 .. p−1
        x̄    = grand mean over all x_i

    A perfectly stable rhythm (same pattern every day) gives IS = 1.
    White noise gives IS → 0; the expected value for iid data is
    approximately (p − 1) / (n_days · p).  Healthy young adults
    typically score 0.5–0.8.  Lower IS is associated with dementia,
    shift work, and severe psychiatric illness.

    Parameters
    ----------
    activity : pd.Series
        Activity time series with a monotonic, regularly-spaced
        DatetimeIndex.  Values are non-negative activity counts (for
        OBF data: per-minute wrist-worn actigraphy).  ``NaN`` values
        are dropped before binning; upstream code is responsible for
        within-day imputation policy.
    bin_minutes : int, default 60
        Width of within-day bins in minutes.  Must evenly divide 1440.
        Hourly (60) is the van Someren default.  IS is sensitive to
        bin choice; report ``bin_minutes`` alongside any published value.
    min_days : int, default 7
        Minimum number of full recording days required.  Returns NaN
        if fewer full days are present.  The default of 7 rather than
        the mathematical minimum of 2 reflects the canonical actigraphy
        literature, where IS estimates computed from fewer than one
        week of data are considered unstable.  Override with explicit
        intent: ``min_days=2`` is the mathematical floor but produces
        estimates too uncertain to be clinically useful.

    Returns
    -------
    float
        IS in [0, 1], or ``float('nan')`` if the recording does not
        satisfy *min_days* or if total activity variance is zero
        (flat signal, IS undefined).

    Raises
    ------
    ValueError
        If the index is not a DatetimeIndex, if 1440 % bin_minutes
        != 0, if bin_minutes < 1, or if the series is empty.

    Notes
    -----
    Population-style sum-of-squares (divisor N, not N − 1) are used
    throughout, matching both the original van Someren formulation and
    the pyActigraphy reference implementation.

    Computing IS on a single recording day always returns 1.0 (hourly
    means equal the actual values; between-bin variance equals total
    variance).  This is expected and is precisely why the *min_days*
    floor matters: any meaningful IS estimate requires at least two
    days of data.

    References
    ----------
    Van Someren, E. J. W., et al. (1999). Bright light therapy:
    improved sensitivity to its effects on rest-activity rhythms in
    Alzheimer patients by application of nonparametric methods.
    *Chronobiology International*, 16(4), 505–518.
    """
    _validate_activity_series(activity, bin_minutes)

    s = activity.dropna()
    if s.empty:
        return float("nan")

    binned = _bin_activity(s, bin_minutes)

    if _count_full_recording_days(binned, bin_minutes) < min_days:
        return float("nan")

    binned = _keep_full_days(binned, bin_minutes)

    grand_mean = float(binned.mean())
    total_ss = float(((binned - grand_mean) ** 2).sum())

    if total_ss == 0.0:
        # Flat signal: all bins equal the grand mean; IS is undefined.
        return float("nan")

    N = len(binned)
    p = 1440 // bin_minutes

    # Within-day bin position for each sample: integer in [0, p).
    # Derived from clock time, not from positional index, so that
    # gaps in the recording cannot corrupt bin assignments.
    bin_pos = (binned.index.hour * 60 + binned.index.minute) // bin_minutes
    hourly_means = binned.groupby(bin_pos).mean()  # shape (p,), index 0..p−1

    between_ss = float(((hourly_means - grand_mean) ** 2).sum())

    return float((N * between_ss) / (p * total_ss))


def intradaily_variability(
    activity: pd.Series,
    *,
    bin_minutes: int = 60,
    min_days: int = 1,
) -> float:
    """Compute intradaily variability (IV) of an activity time series.

    IV quantifies the fragmentation of the rest-activity rhythm within
    days: how frequently and steeply the signal transitions between rest
    and activity.  It is the ratio of the mean squared successive
    difference to the total variance.

    Formula (van Someren 1999):

        IV = [ N · Σ_{i=2}^{N} (x_i − x_{i−1})² ]
             / [ (N − 1) · Σ_{i=1}^{N} (x_i − x̄)² ]

    where x_i are sequential binned samples and x̄ is the grand mean.

    Lower IV indicates consolidated, well-structured rest and activity
    periods.  Higher IV indicates frequent transitions (fragmentation).
    White noise has IV ≈ 2; a perfectly alternating signal reaches IV = 4.
    Healthy young adults score roughly 0.3–0.7.  Elevated IV is reported
    in dementia, depression, and schizophrenia, though effect sizes are
    typically modest.

    Parameters
    ----------
    activity : pd.Series
        Activity time series with a monotonic, regularly-spaced
        DatetimeIndex.  Values are non-negative activity counts (for
        OBF data: per-minute wrist-worn actigraphy).  ``NaN`` values
        are dropped before binning.
    bin_minutes : int, default 60
        Width of within-day bins in minutes.  Must evenly divide 1440.
        Hourly (60) is the van Someren default.  IV is more sensitive
        to bin choice than IS; smaller bins produce systematically
        higher IV.  Report ``bin_minutes`` alongside any published value.
    min_days : int, default 1
        Minimum number of full recording days required.  IV is defined
        on a single day, unlike IS which requires cross-day averaging.
        Multi-day recordings stabilize the estimate; using at least 3
        days is advisable in practice.

    Returns
    -------
    float
        IV (typically in [0, 2] for empirical data; can exceed 2 for
        highly fragmented or artificial signals), or ``float('nan')``
        if the recording does not satisfy *min_days* or if total
        activity variance is zero (flat signal, IV undefined).

    Raises
    ------
    ValueError
        If the index is not a DatetimeIndex, if 1440 % bin_minutes
        != 0, if bin_minutes < 1, or if the series is empty.

    Notes
    -----
    Successive differences are computed across the entire concatenated
    series, including transitions across midnight.  This treats the
    recording as one continuous signal and follows the canonical van
    Someren formulation.

    Where full-day filtering produces a non-contiguous series (e.g. an
    isolated partial day is dropped, leaving a gap), transitions across
    that gap are excluded from the numerator sum — they are artefactual,
    not biological.  N in the denominator is not adjusted; the
    approximation is negligible when gaps are rare (as in OBF data).

    References
    ----------
    Van Someren, E. J. W., et al. (1999). Bright light therapy:
    improved sensitivity to its effects on rest-activity rhythms in
    Alzheimer patients by application of nonparametric methods.
    *Chronobiology International*, 16(4), 505–518.
    """
    _validate_activity_series(activity, bin_minutes)

    s = activity.dropna()
    if s.empty:
        return float("nan")

    binned = _bin_activity(s, bin_minutes)

    if _count_full_recording_days(binned, bin_minutes) < min_days:
        return float("nan")

    binned = _keep_full_days(binned, bin_minutes)

    grand_mean = float(binned.mean())
    total_ss = float(((binned - grand_mean) ** 2).sum())

    if total_ss == 0.0:
        # Flat signal: zero variance; IV is undefined.
        return float("nan")

    N = len(binned)

    # Successive differences across the full concatenated series.
    # Cross-midnight transitions are included (canonical van Someren).
    # Transitions across recording gaps are masked out: where the time
    # between consecutive bins exceeds one bin width, that difference
    # is artefactual and set to NaN before summing.
    expected_step = pd.Timedelta(minutes=bin_minutes)
    time_gaps = binned.index.to_series().diff()
    raw_diffs = binned.diff()
    raw_diffs[time_gaps > expected_step] = float("nan")
    valid_diffs = raw_diffs.dropna()

    if len(valid_diffs) == 0:
        return float("nan")

    mssd_num = float((valid_diffs ** 2).sum())

    return float((N * mssd_num) / ((N - 1) * total_ss))


def least_active_period(
    activity: pd.Series,
    *,
    hours: int = 5,
    bin_minutes: int = 60,
    min_days: int = 1,
) -> WindowResult:
    """Find the least-active consecutive window in the 24-hour profile.

    The canonical "L5" metric (when hours=5) is the mean activity over
    the five consecutive clock hours with the lowest activity in the
    participant's averaged 24-hour profile.  It is a nonparametric
    proxy for sleep-period intensity that does not require sleep
    scoring.

    Procedure:
        1. Bin activity into *bin_minutes* intervals.
        2. Average each bin position across all full recording days,
           producing a 24-hour mean profile of length p = 1440 // bin_minutes.
        3. Slide a window of *hours* hours across the profile circularly
           (wrapping across midnight) and select the window with the
           minimum mean.
        4. Return the mean and the clock time at which the window begins.

    Parameters
    ----------
    activity : pd.Series
        Activity time series with a monotonic, regularly-spaced
        DatetimeIndex.
    hours : int, default 5
        Window width in hours.  The canonical L5 uses 5; some literature
        also uses L6.  Must be in [1, 23].
    bin_minutes : int, default 60
        Bin width in minutes.  Must evenly divide 1440, and
        ``hours * 60`` must be divisible by *bin_minutes* so that the
        window width is an integer number of bins.
    min_days : int, default 1
        Minimum number of full recording days.  Single-day L5 is defined
        but unstable; the canonical use case averages across ≥ 7 days.

    Returns
    -------
    WindowResult
        Named tuple with fields:
            value : float — mean activity in the least-active window.
            onset : datetime.time — clock time at window start.
        Both fields are ``float('nan')`` / ``None`` if min_days is unmet.

    Raises
    ------
    ValueError
        If the index is not a DatetimeIndex, bin_minutes is invalid,
        hours is outside [1, 23], or hours * 60 is not divisible by
        bin_minutes.

    Notes
    -----
    The averaged 24-hour profile is treated circularly: the window
    search includes wrap-around windows (e.g. 22:00–02:59).  This is
    the standard approach and is particularly important for participants
    whose least-active period spans midnight.

    When multiple windows tie for the minimum, the earliest onset is
    returned.

    References
    ----------
    Witting, W., et al. (1990). Alterations in the circadian
    rest-activity rhythm in aging and Alzheimer's disease.
    *Biological Psychiatry*, 27(6), 563–572.
    """
    _validate_activity_series(activity, bin_minutes)

    if not 1 <= hours <= 23:
        raise ValueError(f"hours must be in [1, 23]; got {hours}.")
    if (hours * 60) % bin_minutes != 0:
        raise ValueError(
            f"hours={hours} and bin_minutes={bin_minutes} do not produce "
            f"an integer window width "
            f"({hours * 60} / {bin_minutes} = {hours * 60 / bin_minutes:.3g}). "
            f"hours * 60 must be divisible by bin_minutes."
        )

    _nan = WindowResult(value=float("nan"), onset=None)

    s = activity.dropna()
    if s.empty:
        return _nan

    binned = _bin_activity(s, bin_minutes)

    if _count_full_recording_days(binned, bin_minutes) < min_days:
        return _nan

    binned = _keep_full_days(binned, bin_minutes)

    p = 1440 // bin_minutes
    window_bins = (hours * 60) // bin_minutes

    # Averaged 24-hour profile: mean across all full recording days.
    bin_pos = (binned.index.hour * 60 + binned.index.minute) // bin_minutes
    profile = binned.groupby(bin_pos).mean().to_numpy(dtype=float)  # shape (p,)

    value, onset_bin = _circular_window_search(profile, window_bins, kind="min")

    total_minutes = onset_bin * bin_minutes
    onset_time = datetime.time(total_minutes // 60, total_minutes % 60)

    return WindowResult(value=value, onset=onset_time)


def most_active_period(
    activity: pd.Series,
    *,
    hours: int = 10,
    bin_minutes: int = 60,
    min_days: int = 1,
) -> WindowResult:
    """Find the most-active consecutive window in the 24-hour profile.

    The canonical "M10" metric (when hours=10) is the mean activity over
    the ten consecutive clock hours with the highest activity in the
    participant's averaged 24-hour profile.  It captures wake-period
    intensity without requiring sleep-wake scoring.

    Together with L5, M10 yields:
        amplitude          = M10.value − L5.value
        relative_amplitude = (M10.value − L5.value) / (M10.value + L5.value)
    Both are computed downstream in ``features/derived.py``.

    Parameters
    ----------
    activity : pd.Series
        Activity time series with a monotonic, regularly-spaced
        DatetimeIndex.
    hours : int, default 10
        Window width in hours.  The canonical M10 uses 10.
        Must be in [1, 23].
    bin_minutes : int, default 60
        Bin width in minutes.  Must evenly divide 1440, and
        ``hours * 60`` must be divisible by *bin_minutes*.
    min_days : int, default 1
        Minimum number of full recording days.

    Returns
    -------
    WindowResult
        Named tuple with fields:
            value : float — mean activity in the most-active window.
            onset : datetime.time — clock time at window start.
        The onset is a candidate phase marker; a pathologically late
        M10 onset is itself a phenotype worth modeling.
        Both fields are ``float('nan')`` / ``None`` if min_days is unmet.

    Raises
    ------
    ValueError
        If the index is not a DatetimeIndex, bin_minutes is invalid,
        hours is outside [1, 23], or hours * 60 is not divisible by
        bin_minutes.

    Notes
    -----
    Circular window search, as in ``least_active_period``.  For diurnal
    participants the M10 onset typically falls in mid-morning.  When
    multiple windows tie, the earliest onset is returned.

    References
    ----------
    Witting, W., et al. (1990). Alterations in the circadian
    rest-activity rhythm in aging and Alzheimer's disease.
    *Biological Psychiatry*, 27(6), 563–572.
    """
    _validate_activity_series(activity, bin_minutes)

    if not 1 <= hours <= 23:
        raise ValueError(f"hours must be in [1, 23]; got {hours}.")
    if (hours * 60) % bin_minutes != 0:
        raise ValueError(
            f"hours={hours} and bin_minutes={bin_minutes} do not produce "
            f"an integer window width "
            f"({hours * 60} / {bin_minutes} = {hours * 60 / bin_minutes:.3g}). "
            f"hours * 60 must be divisible by bin_minutes."
        )

    _nan = WindowResult(value=float("nan"), onset=None)

    s = activity.dropna()
    if s.empty:
        return _nan

    binned = _bin_activity(s, bin_minutes)

    if _count_full_recording_days(binned, bin_minutes) < min_days:
        return _nan

    binned = _keep_full_days(binned, bin_minutes)

    p = 1440 // bin_minutes
    window_bins = (hours * 60) // bin_minutes

    bin_pos = (binned.index.hour * 60 + binned.index.minute) // bin_minutes
    profile = binned.groupby(bin_pos).mean().to_numpy(dtype=float)

    value, onset_bin = _circular_window_search(profile, window_bins, kind="max")

    total_minutes = onset_bin * bin_minutes
    onset_time = datetime.time(total_minutes // 60, total_minutes % 60)

    return WindowResult(value=value, onset=onset_time)