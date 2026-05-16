"""
Per-participant feature extraction orchestrator for OBF psychiatric pipeline.

``extract_participant_features`` calls all nine feature functions on one
participant's raw activity series and returns a flat dict of
feature-name → float.  ``extract_all_features`` applies this to a list of
ParticipantRecord objects and returns a DataFrame ready to join with
features.csv on the ``participant_id`` / ``user`` key.

Feature inventory (17 features)
---------------------------------
IS / IV (2):
    is                   — interdaily stability
    iv                   — intradaily variability

L5 / M10 (4):
    l5_value             — mean activity in least-active 5-hour window
    l5_onset_hour        — clock hour at L5 window start  [0, 24)
    m10_value            — mean activity in most-active 10-hour window
    m10_onset_hour       — clock hour at M10 window start [0, 24)

Derived (2):
    amplitude            — M10 − L5  (same units as binned activity)
    relative_amplitude   — (M10 − L5) / (M10 + L5)  [0, 1]

Cosinor (4):
    cosinor_mesor        — rhythm-adjusted mean
    cosinor_amplitude    — sinusoidal amplitude
    cosinor_acrophase_hours — time of peak activity  [0, 24)
    cosinor_r_squared    — goodness-of-fit of sinusoidal model

Sleep (5):
    tst_hours            — total sleep time in nightly rest window (h/day)
    tst_24h_hours        — total sleep time over full 24 h (h/day)
    waso_minutes         — wake after sleep onset (min/day)
    sleep_efficiency     — TST / rest-window duration  [0, 1]
    sol_minutes          — sleep onset latency (min/day)
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import pandas as pd

from obf_psychiatric_pipeline.data.raw_loader import ParticipantRecord
from obf_psychiatric_pipeline.features.cosinor import cosinor_parameters
from obf_psychiatric_pipeline.features.derived import amplitude, relative_amplitude
from obf_psychiatric_pipeline.features.sleep import sleep_metrics
from obf_psychiatric_pipeline.features.temporal import (
    interdaily_stability,
    intradaily_variability,
    least_active_period,
    most_active_period,
)

logger = logging.getLogger(__name__)

# Canonical feature names — defines column order in the output DataFrame.
FEATURE_NAMES: tuple[str, ...] = (
    "is",
    "iv",
    "l5_value",
    "l5_onset_hour",
    "m10_value",
    "m10_onset_hour",
    "amplitude",
    "relative_amplitude",
    "cosinor_mesor",
    "cosinor_amplitude",
    "cosinor_acrophase_hours",
    "cosinor_r_squared",
    "tst_hours",
    "tst_24h_hours",
    "waso_minutes",
    "sleep_efficiency",
    "sol_minutes",
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _time_to_hours(t: datetime.time | None) -> float:
    """Convert a ``datetime.time`` to decimal hours in [0, 24).

    Returns ``float('nan')`` when *t* is ``None`` (insufficient data).
    """
    if t is None:
        return float("nan")
    return float(t.hour + t.minute / 60.0 + t.second / 3600.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_participant_features(
    record: ParticipantRecord,
    *,
    bin_minutes: int = 60,
) -> dict[str, Any]:
    """Extract all temporal features for one participant.

    Parameters
    ----------
    record : ParticipantRecord
        Raw actigraphy from ``load_participant_activity`` or
        ``load_all_actigraphy``.
    bin_minutes : int, default 60
        Bin width forwarded to all feature functions.

    Returns
    -------
    dict
        Keys: ``"participant_id"``, ``"label"``, and all 17 feature names
        from ``FEATURE_NAMES``.  Feature values are always float; NaN when
        the recording does not satisfy a function's minimum-days requirement.

    Raises
    ------
    Does not raise.  Any unexpected exception from a feature function is
    caught, logged, and converted to a full-NaN row via ``extract_all_features``.
    Call this function directly if you want exceptions to propagate.
    """
    activity = record.activity

    is_val = interdaily_stability(activity, bin_minutes=bin_minutes, min_days=7)
    iv_val = intradaily_variability(activity, bin_minutes=bin_minutes, min_days=1)

    l5 = least_active_period(activity, hours=5, bin_minutes=bin_minutes, min_days=1)
    m10 = most_active_period(activity, hours=10, bin_minutes=bin_minutes, min_days=1)

    amp = amplitude(l5, m10)
    ra = relative_amplitude(l5, m10)

    cos = cosinor_parameters(activity, bin_minutes=bin_minutes, min_days=1)
    slp = sleep_metrics(activity, bin_minutes=bin_minutes, min_days=3)

    return {
        "participant_id": record.participant_id,
        "label": record.label,
        "is": is_val,
        "iv": iv_val,
        "l5_value": l5.value,
        "l5_onset_hour": _time_to_hours(l5.onset),
        "m10_value": m10.value,
        "m10_onset_hour": _time_to_hours(m10.onset),
        "amplitude": amp,
        "relative_amplitude": ra,
        "cosinor_mesor": cos.mesor,
        "cosinor_amplitude": cos.amplitude,
        "cosinor_acrophase_hours": cos.acrophase_hours,
        "cosinor_r_squared": cos.r_squared,
        "tst_hours": slp.tst_hours,
        "tst_24h_hours": slp.tst_24h_hours,
        "waso_minutes": slp.waso_minutes,
        "sleep_efficiency": slp.sleep_efficiency,
        "sol_minutes": slp.sol_minutes,
    }


def extract_all_features(
    records: list[ParticipantRecord],
    *,
    bin_minutes: int = 60,
) -> pd.DataFrame:
    """Extract features for all participants and return a tidy DataFrame.

    Parameters
    ----------
    records : list of ParticipantRecord
        Typically the output of ``load_all_actigraphy``.
    bin_minutes : int, default 60
        Bin width forwarded to all feature functions.

    Returns
    -------
    pd.DataFrame
        One row per participant.  Index is ``participant_id``.
        First column is ``label``; remaining columns are the 17 features
        in the order defined by ``FEATURE_NAMES``.
        Participants whose extraction fails entirely appear as all-NaN rows
        and are logged at WARNING level rather than silently dropped.
    """
    rows = []
    for record in records:
        try:
            row = extract_participant_features(record, bin_minutes=bin_minutes)
        except Exception as exc:
            logger.warning(
                "Feature extraction failed for %s (%s): %s",
                record.participant_id,
                record.label,
                exc,
            )
            row = {
                "participant_id": record.participant_id,
                "label": record.label,
                **{name: float("nan") for name in FEATURE_NAMES},
            }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("participant_id")
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    return df[["label"] + feature_cols]