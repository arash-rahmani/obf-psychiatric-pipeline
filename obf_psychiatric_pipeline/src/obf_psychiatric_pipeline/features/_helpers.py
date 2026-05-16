"""
Shared private helpers for OBF psychiatric pipeline feature modules.

This module exists solely to eliminate cross-module underscore imports.
Prior to this refactor, temporal.py, cosinor.py, and sleep.py all
imported private symbols directly from temporal.py.  Moving the shared
helpers here gives each module a single clean import source.

Nothing in this module is part of the public API.  All names are
underscore-prefixed and should not be imported outside this package.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd


def _validate_activity_series(s: pd.Series, bin_minutes: int) -> None:
    """Raise ``ValueError`` if *s* or *bin_minutes* are structurally invalid."""
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError(
            f"activity must be indexed by a DatetimeIndex; "
            f"got {type(s.index).__name__}."
        )
    if s.empty:
        raise ValueError("activity series is empty.")
    if not s.index.is_monotonic_increasing:
        raise ValueError(
            "activity index must be monotonically increasing; "
            "sort the series before passing it to this function."
        )
    if bin_minutes < 1:
        raise ValueError(f"bin_minutes must be >= 1; got {bin_minutes}.")
    if 1440 % bin_minutes != 0:
        valid = [1, 2, 5, 10, 15, 20, 30, 60, 120, 240, 480, 720, 1440]
        raise ValueError(
            f"bin_minutes={bin_minutes} does not evenly divide 1440 "
            f"(minutes per day).  Valid values include: {valid}."
        )


def _bin_activity(s: pd.Series, bin_minutes: int) -> pd.Series:
    """Resample *s* to non-overlapping *bin_minutes* windows, summing counts.

    ``origin='start_day'`` anchors bins to midnight regardless of when
    the recording starts, keeping bin-position labels clock-aligned.
    Bins with no original observations return ``NaN`` (``min_count=1``).
    """
    return s.resample(f"{bin_minutes}min", origin="start_day").sum(min_count=1)


def _full_day_dates(binned: pd.Series, bin_minutes: int) -> set[datetime.date]:
    """Return calendar dates whose *p* bins are all present and non-NaN.

    *p* = 1440 // bin_minutes.  A date is 'full' when every one of its
    *p* bins in the resampled series is non-NaN.  Partial days at the
    recording start or end are excluded because their bin count is less
    than *p*.

    Note: a bin counts as present if it contains any observation in the
    original series (``min_count=1`` semantics from ``_bin_activity``).
    Bins sourced from fewer than *bin_minutes* original observations are
    not separately excluded.  This is acceptable for OBF data, where
    recordings are midnight-aligned; revisit for datasets with intra-day
    start/stop boundaries.
    """
    p = 1440 // bin_minutes
    date_arr = np.array(binned.index.date)  # object array of datetime.date
    return {
        d
        for d in set(date_arr)
        if int(binned[date_arr == d].notna().sum()) == p
    }


def _count_full_recording_days(binned: pd.Series, bin_minutes: int) -> int:
    """Count calendar dates with all *p* bins present and non-NaN."""
    return len(_full_day_dates(binned, bin_minutes))


def _keep_full_days(binned: pd.Series, bin_minutes: int) -> pd.Series:
    """Return the subset of *binned* belonging to full recording days only."""
    full_dates = _full_day_dates(binned, bin_minutes)
    date_s = pd.Series(np.array(binned.index.date), index=binned.index)
    return binned[date_s.isin(full_dates)]


def _circular_window_search(
    profile: np.ndarray, window_bins: int, kind: str
) -> tuple[float, int]:
    """Sliding-window search over a circular 24-hour profile.

    Parameters
    ----------
    profile : np.ndarray, shape (p,)
        Mean activity for each within-day bin position, averaged across
        all recording days.
    window_bins : int
        Number of consecutive bins in the search window.
    kind : {'min', 'max'}
        'min' selects the least-active window (L5);
        'max' selects the most-active window (M10).

    Returns
    -------
    (mean_value, onset_bin)
        mean_value : mean activity in the selected window.
        onset_bin  : 0-based bin index at which the window starts.

    Notes
    -----
    The profile is treated circularly by doubling it before convolution,
    so windows that wrap across midnight are considered on equal footing
    with windows that fall entirely within one calendar day.

    When multiple windows tie, the earliest onset bin is returned
    (``np.argmin`` / ``np.argmax`` return the first occurrence).
    """
    p = len(profile)
    doubled = np.concatenate([profile, profile])
    rolling_sums = np.convolve(
        doubled, np.ones(window_bins, dtype=float), mode="valid"
    )
    candidates = rolling_sums[:p]
    onset_bin = int(
        np.argmin(candidates) if kind == "min" else np.argmax(candidates)
    )
    mean_value = float(candidates[onset_bin] / window_bins)
    return mean_value, onset_bin