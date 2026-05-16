"""
Cosinor model parameters for OBF psychiatric pipeline.

The cosinor model fits a 24-hour sinusoid to the averaged activity
profile, yielding four rhythm parameters: mesor (rhythm-adjusted mean),
amplitude (peak-to-trough half-range), acrophase (time of peak activity),
and R² (goodness-of-fit of the sinusoidal assumption).

These parameters complement the nonparametric IS, IV, L5, and M10 metrics
in temporal.py: where those describe rhythm strength and fragmentation
without assuming a specific waveform, the cosinor parameterizes the
rhythm's shape under a sinusoidal assumption.  R² signals when that
assumption breaks down.

Input convention and NaN policy match temporal.py.

Phase 2 note
------------
The private helpers imported from temporal.py (_validate_activity_series,
_bin_activity, _count_full_recording_days, _keep_full_days) should be
moved to features/_helpers.py in Phase 2 to eliminate cross-module
private imports.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

# Shared private helpers.  Phase 2: moved to features/_helpers.py.
from obf_psychiatric_pipeline.features._helpers import (
    _bin_activity,
    _count_full_recording_days,
    _keep_full_days,
    _validate_activity_series,
)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class CosinorResult(NamedTuple):
    """Parameters of a fitted 24-hour cosinor model.

    Model: x(t) = mesor + amplitude · cos(2π/24 · t − φ)

    All fields are ``float('nan')`` when the recording does not satisfy
    min_days or when the averaged profile has zero variance (R² undefined).

    Attributes
    ----------
    mesor : float
        Rhythm-adjusted mean.  In the same units as the binned activity
        sums (total counts per bin_minutes window).  To convert to
        per-minute counts, divide by bin_minutes.
    amplitude : float
        Half the fitted peak-to-trough range.  Same units as mesor.
        A larger amplitude indicates a more pronounced circadian swing.
    acrophase_hours : float
        Time of peak activity in clock hours, in [0, 24).
        Derived from the fitted phase angle: atan2(γ, β) converted to
        hours.  Numerically undefined when amplitude ≈ 0 (flat rhythm);
        treat as unreliable when R² is low.
    r_squared : float
        Coefficient of determination of the cosinor fit on the averaged
        24-hour profile.  Values near 1 indicate a sinusoidal rhythm;
        lower values indicate non-sinusoidal or irregular rhythms that
        may need additional harmonics.
    """

    mesor: float
    amplitude: float
    acrophase_hours: float
    r_squared: float


# ---------------------------------------------------------------------------
# Module-level NaN sentinel — avoids repeating CosinorResult(nan, nan, ...)
# ---------------------------------------------------------------------------

_NAN_RESULT = CosinorResult(
    mesor=float("nan"),
    amplitude=float("nan"),
    acrophase_hours=float("nan"),
    r_squared=float("nan"),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cosinor_parameters(
    activity: pd.Series,
    *,
    bin_minutes: int = 60,
    min_days: int = 1,
) -> CosinorResult:
    """Fit a 24-hour cosinor model to an activity time series.

    The cosinor model expresses the activity rhythm as a sinusoid of
    fixed 24-hour period:

        x(t) = M + A · cos(2π/24 · t − φ)

    It is linearised for OLS fitting by substituting
    β = A cos φ  and  γ = A sin φ:

        x(t) = M + β · cos(2π/24 · t) + γ · sin(2π/24 · t)

    so that [M, β, γ] are estimated by ordinary least squares on the
    averaged 24-hour activity profile.  Derived parameters:

        amplitude       = √(β² + γ²)
        acrophase_hours = atan2(γ, β) mod 2π  ×  24/2π   (in [0, 24))

    Parameters
    ----------
    activity : pd.Series
        Activity time series with a monotonic, regularly-spaced
        DatetimeIndex.  Values are non-negative activity counts.
    bin_minutes : int, default 60
        Width of within-day bins.  Must evenly divide 1440.  The cosinor
        is fit to the averaged profile at this resolution; hourly bins
        (60) give 24 data points per cycle, which is the standard.
    min_days : int, default 1
        Minimum number of full recording days required.  More days
        produce a more stable averaged profile; ≥ 7 days is advisable.

    Returns
    -------
    CosinorResult
        Named tuple (mesor, amplitude, acrophase_hours, r_squared).
        All NaN if min_days is unmet or the profile has zero variance.

    Raises
    ------
    ValueError
        If the index is not a DatetimeIndex, if 1440 % bin_minutes != 0,
        or if the series is empty.

    Notes
    -----
    The fit is performed on the full p-point averaged 24-hour profile via
    ``numpy.linalg.lstsq``.  All p points are weighted equally.

    Only the fundamental (24-hour) harmonic is fitted.  Non-sinusoidal
    rhythms — e.g. the biphasic activity pattern sometimes seen in
    schizophrenia — will yield low R² without the fit failing.  R² is
    therefore informative in its own right as a feature: low R² for a
    participant with reasonable IS may indicate a biphasic or irregular
    rhythm that a single sinusoid cannot capture.

    Mesor and amplitude are in the units of the binned activity sums.
    Dividing both by bin_minutes converts them to per-minute counts, but
    the conversion is not applied here so that the ratio amplitude/mesor
    (relative amplitude) remains dimensionless and unchanged.

    References
    ----------
    Halberg, F., et al. (1967). Circadian system phase — an aspect of
    temporal morphology; procedures and illustrative examples.
    *The Cellular Aspects of Biorhythms*, 20–48.
    """
    _validate_activity_series(activity, bin_minutes)

    s = activity.dropna()
    if s.empty:
        return _NAN_RESULT

    binned = _bin_activity(s, bin_minutes)

    if _count_full_recording_days(binned, bin_minutes) < min_days:
        return _NAN_RESULT

    binned = _keep_full_days(binned, bin_minutes)

    p = 1440 // bin_minutes

    # Averaged 24-hour profile: mean across all full recording days.
    bin_pos = (binned.index.hour * 60 + binned.index.minute) // bin_minutes
    profile = binned.groupby(bin_pos).mean().to_numpy(dtype=float)  # shape (p,)

    # Zero-variance profile: R² undefined; return all NaN.
    if profile.var() == 0.0:
        return _NAN_RESULT

    # Time axis in hours: 0, bin_minutes/60, 2·bin_minutes/60, ...
    # For hourly bins: [0, 1, 2, ..., 23].
    t = np.arange(p) * bin_minutes / 60.0
    omega = 2.0 * np.pi / 24.0

    # OLS design matrix: [1, cos(ωt), sin(ωt)]
    X = np.column_stack([np.ones(p), np.cos(omega * t), np.sin(omega * t)])

    # Fit: profile ≈ M + β·cos(ωt) + γ·sin(ωt)
    coeffs, _, _, _ = np.linalg.lstsq(X, profile, rcond=None)
    M = float(coeffs[0])
    beta = float(coeffs[1])
    gamma = float(coeffs[2])

    amplitude = float(np.sqrt(beta ** 2 + gamma ** 2))

    # Acrophase: the peak time of the fitted cosine in clock hours.
    # atan2 returns (-π, π]; Python % maps to [0, 2π); then scale to [0, 24).
    acrophase_rad = float(np.arctan2(gamma, beta))
    acrophase_hours = float((acrophase_rad % (2.0 * np.pi)) / (2.0 * np.pi) * 24.0)

    # R²: fraction of profile variance explained by the sinusoidal fit.
    predicted = X @ coeffs
    ss_res = float(np.sum((profile - predicted) ** 2))
    ss_tot = float(np.sum((profile - profile.mean()) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot)

    return CosinorResult(
        mesor=M,
        amplitude=amplitude,
        acrophase_hours=acrophase_hours,
        r_squared=r_squared,
    )