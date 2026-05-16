"""
Derived circadian metrics for OBF psychiatric pipeline.

These are downstream calculations on pre-computed L5 and M10 results,
not primary feature functions.  They live here rather than in temporal.py
because they are computed after the windowed search, not during it.
"""

from __future__ import annotations

import math

from obf_psychiatric_pipeline.features.temporal import WindowResult


def amplitude(l5: WindowResult, m10: WindowResult) -> float:
    """Rest-activity amplitude: M10.value − L5.value.

    Represents the peak-to-trough swing in activity level between the
    most-active and least-active windows.  In the same units as the
    binned activity sums (total counts per bin_minutes window).

    Returns NaN if either input value is NaN.
    """
    if math.isnan(l5.value) or math.isnan(m10.value):
        return float("nan")
    return float(m10.value - l5.value)


def relative_amplitude(l5: WindowResult, m10: WindowResult) -> float:
    """Relative amplitude: (M10 − L5) / (M10 + L5).

    Dimensionless, in [0, 1].  A higher RA indicates a more pronounced
    contrast between rest and activity periods.  RA is scale-invariant
    with respect to the activity count units, making it more comparable
    across participants and devices than raw amplitude.

    Returns NaN if either input value is NaN, or if M10 + L5 = 0
    (flat recording; ratio undefined).
    """
    if math.isnan(l5.value) or math.isnan(m10.value):
        return float("nan")
    denom = m10.value + l5.value
    if denom == 0.0:
        return float("nan")
    return float((m10.value - l5.value) / denom)