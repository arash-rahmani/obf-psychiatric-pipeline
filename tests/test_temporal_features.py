"""
Tests for temporal circadian features.

Coverage: interdaily_stability (IS).
Tests for IV, L5, M10 are added when those functions are implemented.

Test categories
---------------
Known-result cases
    test_perfect_rhythm_returns_one
        — identical pattern each day → IS = 1.0 (formula verification)
    test_iid_noise_is_low
        — random daily activity without structure → IS < 0.20
        — expected E[IS] ≈ (p−1)/(n_days·p) ≈ 0.046 for p=24, n=21

Reference cross-validation
    test_perfect_rhythm_matches_reference
    test_iid_noise_matches_reference
    test_single_spike_matches_reference
        — compare main implementation against _reference_is, which uses
          explicit reshape rather than groupby so that bugs in shared
          helpers do not cancel between the two paths

Edge cases
    test_too_few_days_returns_nan
    test_boundary_six_days_nan_seven_days_not_nan
    test_min_days_one_single_day_returns_one
    test_all_zeros_returns_nan
    test_single_spike_in_bounds

Validation errors
    test_raises_non_datetime_index
    test_raises_empty_series
    test_raises_non_monotonic_index
    test_raises_invalid_bin_minutes

OBF dataset cross-validation
    test_obf_participant_cross_validation
        — placeholder; skipped unless raw OBF data is present
        — fill in IS_EXPECTED by running _reference_is on the same
          participant, then pin the result as a non-regression guard
"""

import math
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obf_psychiatric_pipeline.features.temporal import (
    interdaily_stability,
    intradaily_variability,
    least_active_period,
    most_active_period,
)
from obf_psychiatric_pipeline.features.cosinor import (
    cosinor_parameters,
    CosinorResult,
)
from obf_psychiatric_pipeline.features.sleep import (
    score_sleep,
    sleep_metrics,
    SleepResult,
)


# ---------------------------------------------------------------------------
# Test-only helpers
# ---------------------------------------------------------------------------


def _hourly_pattern_to_per_minute(
    pattern_hourly: np.ndarray, n_days: int
) -> pd.Series:
    """Expand a 24-element hourly pattern to per-minute activity over n_days.

    Each hour value is repeated for all 60 minutes of that hour.
    The resulting series starts at 2020-01-01 00:00 with 1-minute frequency.

    Parameters
    ----------
    pattern_hourly : np.ndarray, shape (24,)
        Activity value for each hour 0..23.
    n_days : int
        Number of full days to tile the pattern across.
    """
    assert len(pattern_hourly) == 24, "pattern_hourly must have 24 values"
    per_minute = np.repeat(pattern_hourly, 60).astype(float)  # shape (1440,)
    start = pd.Timestamp("2020-01-01")
    idx = pd.date_range(start, periods=n_days * 1440, freq="1min")
    return pd.Series(np.tile(per_minute, n_days), index=idx)


def _reference_is(activity: pd.Series, bin_minutes: int = 60) -> float:
    """Independent reference implementation of IS for cross-validation.

    Uses explicit numpy array reshaping rather than the ``groupby``-based
    approach in the main implementation.  Identical results indicate that
    bugs in the shared helper functions are not cancelling between the two
    code paths.

    Assumptions (valid for all synthetic test data constructed here;
    NOT a production function):
        - Series starts exactly at midnight.
        - No NaN values in the input.
        - Length is exactly n_days * 1440 for integer n_days.

    Do NOT call on real OBF data without verifying these assumptions.
    """
    p = 1440 // bin_minutes
    binned = activity.resample(
        f"{bin_minutes}min", origin="start_day"
    ).sum(min_count=1)
    assert binned.notna().all(), (
        "_reference_is: unexpected NaN after resampling; "
        "call only on clean midnight-aligned test data."
    )
    values = binned.to_numpy(dtype=float)
    N = len(values)
    assert N % p == 0, (
        f"_reference_is: expected N divisible by p={p}, got N={N}; "
        "ensure input spans exact whole days."
    )
    n_days = N // p
    x_bar = values.mean()
    total_ss = np.sum((values - x_bar) ** 2)
    if total_ss == 0.0:
        return float("nan")
    # Reshape to (n_days, p): row d holds day d's p bin values.
    # Valid because data starts at midnight and bins are contiguous.
    daily_matrix = values.reshape(n_days, p)
    hourly_means = daily_matrix.mean(axis=0)  # shape (p,)
    between_ss = np.sum((hourly_means - x_bar) ** 2)
    return float(N * between_ss / (p * total_ss))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInterdailyStability:

    # ---- Known-result cases ------------------------------------------------

    def test_perfect_rhythm_returns_one(self):
        """Identical 24-hour pattern across all days gives IS = 1.0.

        Derivation: when every day is identical, x̄_h = x_h (the actual
        bin value for that position).  Substituting into the IS formula:
            between_ss = Σ_h (x_h − x̄)²   (same as total_ss / n_days)
            IS = N * (total_ss / n_days) / (p * total_ss) = N / (n_days * p) = 1.
        """
        # Sleep 0–7 h (0 counts), wake 8–19 h (100 counts/min),
        # wind-down 20–23 h (20 counts/min).
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        result = interdaily_stability(activity, bin_minutes=60, min_days=7)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_iid_noise_is_low(self):
        """Exponential IID noise without diurnal structure gives low IS.

        Theoretical expectation for iid data:
            E[IS] ≈ (p − 1) / (n_days · p)
                  = 23 / (21 · 24) ≈ 0.046   (p=24, n_days=21)
        Test threshold 0.20 provides 4× margin above E[IS] while remaining
        well below the range for any real diurnal signal (>0.40 in practice).
        """
        rng = np.random.default_rng(42)
        n_days = 21
        values = rng.exponential(scale=50.0, size=n_days * 1440).astype(float)
        start = pd.Timestamp("2020-01-01")
        idx = pd.date_range(start, periods=n_days * 1440, freq="1min")
        activity = pd.Series(values, index=idx)
        result = interdaily_stability(activity, bin_minutes=60, min_days=7)
        assert 0.0 <= result < 0.20

    # ---- Reference cross-validation ----------------------------------------

    def test_perfect_rhythm_matches_reference(self):
        """Stable-rhythm result matches the independent reshape-based reference."""
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        main = interdaily_stability(activity, bin_minutes=60, min_days=7)
        ref = _reference_is(activity, bin_minutes=60)
        assert main == pytest.approx(ref, abs=1e-9)

    def test_iid_noise_matches_reference(self):
        """Noise IS matches reference implementation on a different random seed."""
        rng = np.random.default_rng(99)
        n_days = 14
        values = rng.exponential(scale=50.0, size=n_days * 1440).astype(float)
        start = pd.Timestamp("2020-01-01")
        idx = pd.date_range(start, periods=n_days * 1440, freq="1min")
        activity = pd.Series(values, index=idx)
        main = interdaily_stability(activity, bin_minutes=60, min_days=7)
        ref = _reference_is(activity, bin_minutes=60)
        assert main == pytest.approx(ref, abs=1e-9)

    def test_single_spike_matches_reference(self):
        """Single-spike result matches the reshape reference.

        After binning (60 min): 168 bins, 167 with value 0, one with
        value 1000 (spike at minute 100 = 01:40 on day 1 → hour-1 bin).
        Hand-computed IS ≈ 0.138; the test only asserts agreement between
        the two implementations, not the specific value.
        """
        start = pd.Timestamp("2020-01-01")
        idx = pd.date_range(start, periods=7 * 1440, freq="1min")
        values = np.zeros(7 * 1440)
        values[100] = 1000.0
        activity = pd.Series(values, index=idx)
        main = interdaily_stability(activity, bin_minutes=60, min_days=7)
        ref = _reference_is(activity, bin_minutes=60)
        assert main == pytest.approx(ref, abs=1e-9)

    # ---- Edge cases --------------------------------------------------------

    def test_too_few_days_returns_nan(self):
        """Recording with 3 full days but min_days=7 returns NaN."""
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=3)
        result = interdaily_stability(activity, bin_minutes=60, min_days=7)
        assert math.isnan(result)

    def test_boundary_six_days_nan_seven_days_not_nan(self):
        """min_days=7: exactly 6 full days → NaN; 7 full days → finite value."""
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)

        result_6 = interdaily_stability(
            _hourly_pattern_to_per_minute(pattern, n_days=6),
            bin_minutes=60,
            min_days=7,
        )
        assert math.isnan(result_6)

        result_7 = interdaily_stability(
            _hourly_pattern_to_per_minute(pattern, n_days=7),
            bin_minutes=60,
            min_days=7,
        )
        assert not math.isnan(result_7)

    def test_min_days_one_single_day_returns_one(self):
        """With min_days=1, a single full day always gives IS = 1.0.

        With one day, x̄_h = x_h for every bin (no averaging across days).
        between_ss therefore equals total_ss, and IS = N / p = 1.0.
        This confirms the formula and illustrates why min_days=7 is the
        correct production default: any single-day IS is trivially 1.
        """
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=1)
        result = interdaily_stability(activity, bin_minutes=60, min_days=1)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_all_zeros_returns_nan(self):
        """All-zero activity gives zero total variance; IS is undefined → NaN."""
        start = pd.Timestamp("2020-01-01")
        idx = pd.date_range(start, periods=7 * 1440, freq="1min")
        activity = pd.Series(np.zeros(7 * 1440), index=idx)
        result = interdaily_stability(activity, bin_minutes=60, min_days=7)
        assert math.isnan(result)

    def test_single_spike_in_bounds(self):
        """Single non-zero spike across 7 days: IS is finite and in [0, 1]."""
        start = pd.Timestamp("2020-01-01")
        idx = pd.date_range(start, periods=7 * 1440, freq="1min")
        values = np.zeros(7 * 1440)
        values[100] = 1000.0  # 01:40 on day 1, falls in hour-1 bin
        activity = pd.Series(values, index=idx)
        result = interdaily_stability(activity, bin_minutes=60, min_days=7)
        assert not math.isnan(result)
        assert 0.0 <= result <= 1.0

    # ---- Validation errors -------------------------------------------------

    def test_raises_on_non_datetime_index(self):
        activity = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            interdaily_stability(activity)

    def test_raises_on_empty_series(self):
        idx = pd.DatetimeIndex([], dtype="datetime64[ns]")
        activity = pd.Series([], index=idx, dtype=float)
        with pytest.raises(ValueError, match="empty"):
            interdaily_stability(activity)

    def test_raises_on_non_monotonic_index(self):
        start = pd.Timestamp("2020-01-01")
        idx = pd.date_range(start, periods=20, freq="1min")[::-1]
        activity = pd.Series(np.ones(20), index=idx)
        with pytest.raises(ValueError, match="monotonic"):
            interdaily_stability(activity)

    def test_raises_on_bin_minutes_that_does_not_divide_1440(self):
        start = pd.Timestamp("2020-01-01")
        idx = pd.date_range(start, periods=100, freq="1min")
        activity = pd.Series(np.ones(100), index=idx)
        with pytest.raises(ValueError, match="bin_minutes"):
            interdaily_stability(activity, bin_minutes=7)  # 1440 % 7 != 0

    # ---- OBF dataset cross-validation (placeholder) ------------------------

    @pytest.mark.skipif(
        not (Path("data") / "raw").exists(),
        reason="OBF raw actigraphy data not present at data/raw/",
    )
    def test_obf_participant_cross_validation(self):
        """
        Cross-validate IS against a hand-verified value for one OBF participant.

        HOW TO FILL THIS IN
        -------------------
        1.  Load per-minute activity for one participant (skeleton below).
        2.  Run ``_reference_is(activity, bin_minutes=60)`` in a REPL or
            notebook to get the expected value.
        3.  Optionally cross-check with pyActigraphy if the file is
            loadable via its reader:
                import pyActigraphy
                raw = pyActigraphy.io.read_raw_awd(filepath)  # or equivalent
                pyact_is = raw.IS(binarize=False, freq="60min")
            Confirm ``_reference_is`` and pyActigraphy agree to within 1e-3
            before pinning IS_EXPECTED.
        4.  Replace IS_EXPECTED with the pinned float.
        5.  Replace the activity loading block with the actual loader.
        6.  Remove this docstring once the test is fully live.

        Once pinned, this test is a non-regression guard: any refactor that
        silently changes IS for a real participant breaks here.
        """
        # ------------------------------------------------------------------ #
        IS_EXPECTED: float | None = None  # FILL IN from step 2/3 above

        # Replace with actual participant loading.
        # The series must be per-minute activity with a DatetimeIndex.
        # Example — adjust column names to match your OBF CSV schema:
        #
        #   df = pd.read_csv(
        #       "data/raw/condition_participant_001.csv",
        #       parse_dates=["timestamp"],
        #   )
        #   activity = df.set_index("timestamp")["activity"].sort_index()
        #
        activity: pd.Series | None = None  # FILL IN
        # ------------------------------------------------------------------ #

        if IS_EXPECTED is None or activity is None:
            pytest.skip(
                "OBF cross-validation not yet configured. "
                "Follow the instructions in this test's docstring."
            )

        result = interdaily_stability(activity, bin_minutes=60, min_days=7)
        assert result == pytest.approx(IS_EXPECTED, abs=1e-3)


# ---------------------------------------------------------------------------
# IV reference helper
# ---------------------------------------------------------------------------


def _reference_iv(activity: pd.Series, bin_minutes: int = 60) -> float:
    """Independent reference implementation of IV for cross-validation.

    Uses explicit numpy array operations (np.diff) rather than
    pandas .diff() + gap masking used in the main implementation.
    Identical results indicate no latent bugs hiding in the two paths.

    Assumptions (valid for all synthetic test data; NOT a production
    function):
        - Series starts exactly at midnight.
        - No NaN values, no recording gaps.
        - Length is exactly n_days * 1440 for integer n_days.
    """
    binned = activity.resample(
        f"{bin_minutes}min", origin="start_day"
    ).sum(min_count=1)
    assert binned.notna().all(), (
        "_reference_iv: unexpected NaN; call only on clean test data."
    )
    values = binned.to_numpy(dtype=float)
    N = len(values)
    x_bar = values.mean()
    total_ss = np.sum((values - x_bar) ** 2)
    if total_ss == 0.0:
        return float("nan")
    diffs_sq = np.diff(values) ** 2  # shape (N-1,)
    return float(N * diffs_sq.sum() / ((N - 1) * total_ss))


# ---------------------------------------------------------------------------
# Tests: intradaily_variability
# ---------------------------------------------------------------------------


class TestIntradailyVariability:

    # ---- Known-result cases ------------------------------------------------

    def test_flat_signal_returns_nan(self):
        """All-zero activity gives zero total variance; IV undefined → NaN."""
        start = pd.Timestamp("2020-01-01")
        idx = pd.date_range(start, periods=7 * 1440, freq="1min")
        activity = pd.Series(np.zeros(7 * 1440), index=idx)
        result = intradaily_variability(activity, bin_minutes=60, min_days=1)
        assert math.isnan(result)

    def test_stable_rhythm_iv_in_expected_range(self):
        """Structured daily rhythm gives IV in the expected human-data range.

        Pattern: sleep 0–7 h (0), wake 8–19 h (100/min), wind-down 20–23 h (20/min).
        Transitions occur only twice per day (0→6000 and 6000→1200) plus
        once at each midnight boundary.  IV is expected around 0.3.
        """
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        result = intradaily_variability(activity, bin_minutes=60, min_days=1)
        assert 0.20 < result < 0.50

    def test_alternating_signal_iv_equals_four(self):
        """Perfectly alternating hourly signal gives IV = 4.0 exactly.

        Derivation: alternating 0 / 6000 bins (x̄ = 3000).
            Σ(x_i − x̄)² = N * 3000²   →   total_ss = N * 9 000 000
            Σ(x_i − x_{i−1})² = (N−1) * 6000²   →   mssd_num = (N−1) * 36 000 000
            IV = N * (N−1) * 36M / ((N−1) * N * 9M) = 36M / 9M = 4.0
        """
        pattern = np.array(
            [0 if h % 2 == 0 else 100 for h in range(24)], dtype=float
        )
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        result = intradaily_variability(activity, bin_minutes=60, min_days=1)
        assert result == pytest.approx(4.0, abs=1e-9)

    # ---- Reference cross-validation ----------------------------------------

    def test_stable_rhythm_matches_reference(self):
        """Stable-rhythm IV matches the independent numpy-diff reference."""
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        main = intradaily_variability(activity, bin_minutes=60, min_days=1)
        ref = _reference_iv(activity, bin_minutes=60)
        assert main == pytest.approx(ref, abs=1e-9)

    def test_alternating_signal_matches_reference(self):
        """Alternating-signal IV matches reference."""
        pattern = np.array(
            [0 if h % 2 == 0 else 100 for h in range(24)], dtype=float
        )
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        main = intradaily_variability(activity, bin_minutes=60, min_days=1)
        ref = _reference_iv(activity, bin_minutes=60)
        assert main == pytest.approx(ref, abs=1e-9)

    # ---- Edge cases --------------------------------------------------------

    def test_too_few_days_returns_nan(self):
        """Recording with 2 full days but min_days=3 returns NaN."""
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=2)
        result = intradaily_variability(activity, bin_minutes=60, min_days=3)
        assert math.isnan(result)

    def test_single_day_accepted_with_min_days_one(self):
        """min_days=1 (the default): a single full day returns a finite value."""
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=1)
        result = intradaily_variability(activity, bin_minutes=60, min_days=1)
        assert not math.isnan(result)
        assert result > 0.0

    # ---- Validation errors -------------------------------------------------

    def test_raises_on_non_datetime_index(self):
        activity = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            intradaily_variability(activity)

    def test_raises_on_empty_series(self):
        idx = pd.DatetimeIndex([], dtype="datetime64[ns]")
        activity = pd.Series([], index=idx, dtype=float)
        with pytest.raises(ValueError, match="empty"):
            intradaily_variability(activity)

    def test_raises_on_invalid_bin_minutes(self):
        start = pd.Timestamp("2020-01-01")
        idx = pd.date_range(start, periods=100, freq="1min")
        activity = pd.Series(np.ones(100), index=idx)
        with pytest.raises(ValueError, match="bin_minutes"):
            intradaily_variability(activity, bin_minutes=7)

    # ---- OBF dataset cross-validation (placeholder) ------------------------

    @pytest.mark.skipif(
        not (Path("data") / "raw").exists(),
        reason="OBF raw actigraphy data not present at data/raw/",
    )
    def test_obf_participant_cross_validation(self):
        """
        Cross-validate IV against a hand-verified value for one OBF participant.

        HOW TO FILL THIS IN
        -------------------
        Same process as the IS cross-validation test above:
        1.  Load per-minute activity for one participant.
        2.  Run ``_reference_iv(activity, bin_minutes=60)`` in a REPL.
        3.  Pin the result as IV_EXPECTED.
        4.  Fill in the activity loading block.
        5.  Remove this docstring once the test is live.
        """
        IV_EXPECTED: float | None = None  # FILL IN

        activity: pd.Series | None = None  # FILL IN

        if IV_EXPECTED is None or activity is None:
            pytest.skip(
                "OBF IV cross-validation not yet configured. "
                "Follow the instructions in this test's docstring."
            )

        result = intradaily_variability(activity, bin_minutes=60, min_days=1)
        assert result == pytest.approx(IV_EXPECTED, abs=1e-3)


# ---------------------------------------------------------------------------
# L5 / M10 reference helpers
# ---------------------------------------------------------------------------


def _reference_window(
    activity: pd.Series, hours: int, bin_minutes: int, kind: str
) -> "WindowResult":
    """Brute-force circular window search for cross-validation.

    Uses explicit Python loops over a reshaped numpy profile — completely
    different from the convolution approach in the main implementation.

    Assumptions: midnight-aligned, gap-free, NaN-free data.
    """
    from obf_psychiatric_pipeline.features.temporal import WindowResult

    p = 1440 // bin_minutes
    window_bins = hours * 60 // bin_minutes
    binned = activity.resample(
        f"{bin_minutes}min", origin="start_day"
    ).sum(min_count=1)
    assert binned.notna().all(), "_reference_window: unexpected NaN"
    values = binned.to_numpy(dtype=float)
    n_days = len(values) // p
    profile = values.reshape(n_days, p).mean(axis=0)

    best_sum = float("inf") if kind == "min" else float("-inf")
    best_onset = 0
    for start in range(p):
        s = sum(profile[(start + i) % p] for i in range(window_bins))
        if (kind == "min" and s < best_sum) or (kind == "max" and s > best_sum):
            best_sum = s
            best_onset = start

    value = best_sum / window_bins
    total_minutes = best_onset * bin_minutes
    onset = datetime.time(total_minutes // 60, total_minutes % 60)
    return WindowResult(value=value, onset=onset)


# ---------------------------------------------------------------------------
# Tests: least_active_period (L5)
# ---------------------------------------------------------------------------


class TestLeastActivePeriod:

    # ---- Known-result cases ------------------------------------------------

    def test_known_onset_midnight_and_zero_value(self):
        """Structured rhythm: L5 onset at 00:00, value = 0.0.

        Pattern: sleep 0–7 h (0), wake 8–19 h (100/min), wind-down 20–23 h (20/min).
        The five lowest consecutive hours are 00:00–04:59 (all zero).
        Multiple windows tie (00:00–04:59 through 03:00–07:59); earliest
        onset is returned.
        """
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        result = least_active_period(activity, hours=5, bin_minutes=60, min_days=1)
        assert result.value == pytest.approx(0.0, abs=1e-9)
        assert result.onset == datetime.time(0, 0)

    def test_wrap_around_onset_at_2100(self):
        """L5 window spanning midnight is found correctly.

        Pattern: hours 0, 1, 21, 22, 23 = zero; rest = 100/min.
        The 5-hour minimum window is [21:00–01:59], wrapping midnight.
        Onset = 21:00, value = 0.0.
        """
        pattern = np.array(
            [0, 0] + [100] * 19 + [0, 0, 0], dtype=float  # h0,h1=0; h21,22,23=0
        )
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        result = least_active_period(activity, hours=5, bin_minutes=60, min_days=1)
        assert result.value == pytest.approx(0.0, abs=1e-9)
        assert result.onset == datetime.time(21, 0)

    # ---- Reference cross-validation ----------------------------------------

    def test_stable_rhythm_matches_reference(self):
        """Stable-rhythm L5 matches the brute-force reference."""
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        main = least_active_period(activity, hours=5, bin_minutes=60, min_days=1)
        ref = _reference_window(activity, hours=5, bin_minutes=60, kind="min")
        assert main.value == pytest.approx(ref.value, abs=1e-9)
        assert main.onset == ref.onset

    def test_wrap_around_matches_reference(self):
        """Wrap-around L5 matches reference."""
        pattern = np.array([0, 0] + [100] * 19 + [0, 0, 0], dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        main = least_active_period(activity, hours=5, bin_minutes=60, min_days=1)
        ref = _reference_window(activity, hours=5, bin_minutes=60, kind="min")
        assert main.value == pytest.approx(ref.value, abs=1e-9)
        assert main.onset == ref.onset

    # ---- Edge cases --------------------------------------------------------

    def test_insufficient_days_returns_nan_onset_none(self):
        """Recording with 2 days but min_days=3 returns WindowResult(nan, None)."""
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=2)
        result = least_active_period(activity, hours=5, bin_minutes=60, min_days=3)
        assert math.isnan(result.value)
        assert result.onset is None

    # ---- Validation errors -------------------------------------------------

    def test_raises_on_hours_out_of_range(self):
        start = pd.Timestamp("2020-01-01")
        idx = pd.date_range(start, periods=1440, freq="1min")
        activity = pd.Series(np.ones(1440), index=idx)
        with pytest.raises(ValueError, match="hours"):
            least_active_period(activity, hours=0)

    def test_raises_on_non_integer_window(self):
        """hours=5, bin_minutes=9 gives 300/9 = 33.3 — not integer → ValueError."""
        start = pd.Timestamp("2020-01-01")
        idx = pd.date_range(start, periods=7 * 1440, freq="1min")
        activity = pd.Series(np.ones(7 * 1440), index=idx)
        with pytest.raises(ValueError, match="window"):
            least_active_period(activity, hours=5, bin_minutes=9)

    # ---- OBF dataset cross-validation (placeholder) ------------------------

    @pytest.mark.skipif(
        not (Path("data") / "raw").exists(),
        reason="OBF raw actigraphy data not present at data/raw/",
    )
    def test_obf_participant_cross_validation(self):
        """
        Cross-validate L5 against a hand-verified value for one OBF participant.

        HOW TO FILL THIS IN
        -------------------
        1.  Load per-minute activity for one participant.
        2.  Run ``_reference_window(activity, hours=5, bin_minutes=60, kind='min')``
            in a REPL to get L5_VALUE_EXPECTED and L5_ONSET_EXPECTED.
        3.  Pin both and fill in the activity loading block.
        """
        L5_VALUE_EXPECTED: float | None = None   # FILL IN
        L5_ONSET_EXPECTED: datetime.time | None = None  # FILL IN
        activity: pd.Series | None = None  # FILL IN

        if any(x is None for x in [L5_VALUE_EXPECTED, L5_ONSET_EXPECTED, activity]):
            pytest.skip("OBF L5 cross-validation not yet configured.")

        result = least_active_period(activity, hours=5, bin_minutes=60, min_days=7)
        assert result.value == pytest.approx(L5_VALUE_EXPECTED, abs=1e-3)
        assert result.onset == L5_ONSET_EXPECTED


# ---------------------------------------------------------------------------
# Tests: most_active_period (M10)
# ---------------------------------------------------------------------------


class TestMostActivePeriod:

    # ---- Known-result cases ------------------------------------------------

    def test_known_onset_at_0800(self):
        """Structured rhythm: M10 onset at 08:00, value = 6000.0.

        Pattern: sleep 0–7 h (0), wake 8–19 h (100/min), wind-down 20–23 h (20/min).
        After binning, wake bins = 6000.  The 10-hour maximum window is
        08:00–17:59; multiple windows starting 08:00 through 09:59 all tie
        at 6000; earliest onset returned.
        """
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        result = most_active_period(activity, hours=10, bin_minutes=60, min_days=1)
        assert result.value == pytest.approx(6000.0, abs=1e-9)
        assert result.onset == datetime.time(8, 0)

    def test_wrap_around_onset_at_1900(self):
        """M10 window spanning midnight is found correctly.

        Pattern: hours 0–4 = 100/min, hours 5–18 = 0, hours 19–23 = 100/min.
        After binning, active bins = 6000 at positions 0–4 and 19–23.
        The 10-hour maximum window [19:00–04:59] wraps midnight.
        Onset = 19:00, value = 6000.0.
        """
        pattern = np.array(
            [100] * 5 + [0] * 14 + [100] * 5, dtype=float  # h0-4=100; h19-23=100
        )
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        result = most_active_period(activity, hours=10, bin_minutes=60, min_days=1)
        assert result.value == pytest.approx(6000.0, abs=1e-9)
        assert result.onset == datetime.time(19, 0)

    # ---- Reference cross-validation ----------------------------------------

    def test_stable_rhythm_matches_reference(self):
        """Stable-rhythm M10 matches the brute-force reference."""
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        main = most_active_period(activity, hours=10, bin_minutes=60, min_days=1)
        ref = _reference_window(activity, hours=10, bin_minutes=60, kind="max")
        assert main.value == pytest.approx(ref.value, abs=1e-9)
        assert main.onset == ref.onset

    def test_wrap_around_matches_reference(self):
        """Wrap-around M10 matches reference."""
        pattern = np.array([100] * 5 + [0] * 14 + [100] * 5, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        main = most_active_period(activity, hours=10, bin_minutes=60, min_days=1)
        ref = _reference_window(activity, hours=10, bin_minutes=60, kind="max")
        assert main.value == pytest.approx(ref.value, abs=1e-9)
        assert main.onset == ref.onset

    # ---- Edge cases --------------------------------------------------------

    def test_insufficient_days_returns_nan_onset_none(self):
        """Recording with 2 days but min_days=3 returns WindowResult(nan, None)."""
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=2)
        result = most_active_period(activity, hours=10, bin_minutes=60, min_days=3)
        assert math.isnan(result.value)
        assert result.onset is None

    # ---- OBF dataset cross-validation (placeholder) ------------------------

    @pytest.mark.skipif(
        not (Path("data") / "raw").exists(),
        reason="OBF raw actigraphy data not present at data/raw/",
    )
    def test_obf_participant_cross_validation(self):
        """
        Cross-validate M10 against a hand-verified value for one OBF participant.

        HOW TO FILL THIS IN
        -------------------
        Run ``_reference_window(activity, hours=10, bin_minutes=60, kind='max')``
        and pin M10_VALUE_EXPECTED and M10_ONSET_EXPECTED.
        """
        M10_VALUE_EXPECTED: float | None = None  # FILL IN
        M10_ONSET_EXPECTED: datetime.time | None = None  # FILL IN
        activity: pd.Series | None = None  # FILL IN

        if any(x is None for x in [M10_VALUE_EXPECTED, M10_ONSET_EXPECTED, activity]):
            pytest.skip("OBF M10 cross-validation not yet configured.")

        result = most_active_period(activity, hours=10, bin_minutes=60, min_days=7)
        assert result.value == pytest.approx(M10_VALUE_EXPECTED, abs=1e-3)
        assert result.onset == M10_ONSET_EXPECTED


# ---------------------------------------------------------------------------
# Cosinor helpers
# ---------------------------------------------------------------------------


def _make_cosine_activity(
    n_days: int,
    peak_hour: float,
    mesor_per_min: float = 50.0,
    amplitude_per_min: float = 30.0,
) -> pd.Series:
    """Per-minute activity where all 60 minutes in an hour share the same value.

    This ensures that after 60-min binning (sum), the profile is exactly:
        profile[h] = 60 * (mesor + amplitude * cos(2π/24 * (h − peak_hour)))
    allowing the cosinor fit to achieve R² ≈ 1.0 on this input.
    """
    omega = 2.0 * np.pi / 24.0
    phi = omega * peak_hour
    hourly_values = np.array(
        [mesor_per_min + amplitude_per_min * np.cos(omega * h - phi) for h in range(24)]
    )
    per_minute = np.repeat(hourly_values, 60)  # shape (1440,)
    start = pd.Timestamp("2020-01-01")
    idx = pd.date_range(start, periods=n_days * 1440, freq="1min")
    return pd.Series(np.tile(per_minute, n_days), index=idx)


def _reference_cosinor(activity: pd.Series, bin_minutes: int = 60) -> CosinorResult:
    """Reference cosinor using normal equations (X^T X)^{-1} X^T y.

    Uses a different linear algebra code path (solve vs lstsq) so that
    shared bugs in the design matrix or derivations do not cancel.

    Assumptions: midnight-aligned, gap-free, NaN-free data.
    """
    p = 1440 // bin_minutes
    binned = activity.resample(
        f"{bin_minutes}min", origin="start_day"
    ).sum(min_count=1)
    assert binned.notna().all(), "_reference_cosinor: unexpected NaN"
    values = binned.to_numpy(dtype=float)
    n_days = len(values) // p
    profile = values.reshape(n_days, p).mean(axis=0)

    t = np.arange(p) * bin_minutes / 60.0
    omega = 2.0 * np.pi / 24.0
    X = np.column_stack([np.ones(p), np.cos(omega * t), np.sin(omega * t)])

    # Normal equations: different code path from lstsq in main implementation.
    coeffs = np.linalg.solve(X.T @ X, X.T @ profile)
    M, beta, gamma = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    amplitude = float(np.sqrt(beta ** 2 + gamma ** 2))
    acrophase_rad = float(np.arctan2(gamma, beta))
    acrophase_hours = float((acrophase_rad % (2.0 * np.pi)) / (2.0 * np.pi) * 24.0)

    predicted = X @ coeffs
    ss_res = float(np.sum((profile - predicted) ** 2))
    ss_tot = float(np.sum((profile - profile.mean()) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return CosinorResult(
        mesor=M,
        amplitude=amplitude,
        acrophase_hours=acrophase_hours,
        r_squared=r_squared,
    )


# ---------------------------------------------------------------------------
# Tests: cosinor_parameters
# ---------------------------------------------------------------------------


class TestCosinorParameters:

    # ---- Known-result cases ------------------------------------------------

    def test_perfect_cosine_recovers_all_parameters(self):
        """A perfect cosine input recovers mesor, amplitude, acrophase, and R².

        Synthetic signal: peak at 14:00, mesor=50/min, amplitude=30/min.
        After 60-min binning:
            profile[h] = 60*(50 + 30*cos(2π/24*(h−14)))
                       = 3000 + 1800*cos(2π/24*(h−14))
        The cosinor fit is exact for a single sinusoid, so:
            mesor ≈ 3000, amplitude ≈ 1800,
            acrophase_hours ≈ 14.0,  R² ≈ 1.0.
        """
        activity = _make_cosine_activity(n_days=7, peak_hour=14.0)
        result = cosinor_parameters(activity, bin_minutes=60, min_days=1)

        assert result.mesor == pytest.approx(3000.0, abs=0.1)
        assert result.amplitude == pytest.approx(1800.0, abs=0.1)
        assert result.acrophase_hours == pytest.approx(14.0, abs=0.01)
        assert result.r_squared == pytest.approx(1.0, abs=1e-6)

    def test_flat_signal_returns_all_nan(self):
        """All-zero profile has zero variance; all cosinor parameters → NaN."""
        start = pd.Timestamp("2020-01-01")
        idx = pd.date_range(start, periods=7 * 1440, freq="1min")
        activity = pd.Series(np.zeros(7 * 1440), index=idx)
        result = cosinor_parameters(activity, bin_minutes=60, min_days=1)
        assert all(math.isnan(v) for v in result)

    def test_realistic_rhythm_r_squared_in_bounds(self):
        """Structured step-function rhythm gives R² in (0.5, 1.0).

        The sleep/wake step pattern is not a perfect sinusoid, so R² < 1.
        It does have strong diurnal structure, so R² > 0.5.
        """
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        result = cosinor_parameters(activity, bin_minutes=60, min_days=1)
        assert 0.5 < result.r_squared < 1.0

    # ---- Reference cross-validation ----------------------------------------

    def test_perfect_cosine_matches_reference(self):
        """Perfect-cosine cosinor matches the normal-equations reference."""
        activity = _make_cosine_activity(n_days=7, peak_hour=14.0)
        main = cosinor_parameters(activity, bin_minutes=60, min_days=1)
        ref = _reference_cosinor(activity, bin_minutes=60)
        assert main.mesor == pytest.approx(ref.mesor, abs=1e-6)
        assert main.amplitude == pytest.approx(ref.amplitude, abs=1e-6)
        assert main.acrophase_hours == pytest.approx(ref.acrophase_hours, abs=1e-6)
        assert main.r_squared == pytest.approx(ref.r_squared, abs=1e-6)

    def test_realistic_rhythm_matches_reference(self):
        """Step-function rhythm cosinor matches reference."""
        pattern = np.array([0] * 8 + [100] * 12 + [20] * 4, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        main = cosinor_parameters(activity, bin_minutes=60, min_days=1)
        ref = _reference_cosinor(activity, bin_minutes=60)
        assert main.mesor == pytest.approx(ref.mesor, abs=1e-6)
        assert main.amplitude == pytest.approx(ref.amplitude, abs=1e-6)
        assert main.acrophase_hours == pytest.approx(ref.acrophase_hours, abs=1e-6)
        assert main.r_squared == pytest.approx(ref.r_squared, abs=1e-6)

    # ---- Edge cases --------------------------------------------------------

    def test_too_few_days_returns_all_nan(self):
        """Recording with 2 days but min_days=7 returns CosinorResult of all NaN."""
        activity = _make_cosine_activity(n_days=2, peak_hour=14.0)
        result = cosinor_parameters(activity, bin_minutes=60, min_days=7)
        assert all(math.isnan(v) for v in result)

    # ---- Validation errors -------------------------------------------------

    def test_raises_on_non_datetime_index(self):
        activity = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            cosinor_parameters(activity)

    # ---- OBF dataset cross-validation (placeholder) ------------------------

    @pytest.mark.skipif(
        not (Path("data") / "raw").exists(),
        reason="OBF raw actigraphy data not present at data/raw/",
    )
    def test_obf_participant_cross_validation(self):
        """
        Cross-validate cosinor against a hand-verified value for one OBF participant.

        HOW TO FILL THIS IN
        -------------------
        1.  Load per-minute activity for one participant.
        2.  Run ``_reference_cosinor(activity, bin_minutes=60)`` in a REPL.
        3.  Pin all four fields and fill in the activity loading block.
        """
        MESOR_EXPECTED: float | None = None        # FILL IN
        AMPLITUDE_EXPECTED: float | None = None    # FILL IN
        ACROPHASE_EXPECTED: float | None = None    # FILL IN
        R2_EXPECTED: float | None = None           # FILL IN
        activity: pd.Series | None = None          # FILL IN

        if any(x is None for x in
               [MESOR_EXPECTED, AMPLITUDE_EXPECTED, ACROPHASE_EXPECTED,
                R2_EXPECTED, activity]):
            pytest.skip("OBF cosinor cross-validation not yet configured.")

        result = cosinor_parameters(activity, bin_minutes=60, min_days=7)
        assert result.mesor == pytest.approx(MESOR_EXPECTED, abs=1e-3)
        assert result.amplitude == pytest.approx(AMPLITUDE_EXPECTED, abs=1e-3)
        assert result.acrophase_hours == pytest.approx(ACROPHASE_EXPECTED, abs=0.1)
        assert result.r_squared == pytest.approx(R2_EXPECTED, abs=1e-3)


# ---------------------------------------------------------------------------
# Tests: score_sleep (Cole-Kripke + Webster + Sadeh)
# ---------------------------------------------------------------------------


def _make_flat_activity(n_days: int, value: float) -> pd.Series:
    """n_days of per-minute activity at a constant value."""
    start = pd.Timestamp("2020-01-01")
    idx = pd.date_range(start, periods=n_days * 1440, freq="1min")
    return pd.Series(np.full(n_days * 1440, value), index=idx)


class TestSleepScoring:

    def test_zero_activity_scored_as_sleep_ck(self):
        """All-zero activity scores as sleep under Cole-Kripke.

        Score(t) = 0.00001 × (all weights × 0) = 0 < 1 → sleep at every epoch.
        """
        activity = _make_flat_activity(n_days=3, value=0.0)
        labels = score_sleep(activity, scorer="cole_kripke")
        assert (labels == 0).all()

    def test_high_activity_scored_as_wake_ck(self):
        """Sustained high activity scores as wake under Cole-Kripke.

        Score(t) = 0.00001 × (sum of all weights) × 10000
                 = 0.00001 × 4014 × 10000 = 401.4 >> 1 → wake.
        """
        activity = _make_flat_activity(n_days=3, value=10000.0)
        labels = score_sleep(activity, scorer="cole_kripke")
        # Interior epochs (where all 8 lags are in the high-activity region)
        # should be wake.  Allow boundary epochs to differ.
        assert (labels[4:-3] == 1).all()

    def test_zero_activity_scored_as_sleep_sadeh(self):
        """All-zero activity: PS = 7.601 > 0 → sleep for all Sadeh epochs."""
        activity = _make_flat_activity(n_days=3, value=0.0)
        labels = score_sleep(activity, scorer="sadeh")
        assert (labels == 0).all()

    def test_high_activity_scored_as_wake_sadeh(self):
        """Sustained high activity: PS << 0 → wake for interior Sadeh epochs."""
        activity = _make_flat_activity(n_days=3, value=10000.0)
        labels = score_sleep(activity, scorer="sadeh")
        assert (labels[5:-5] == 1).all()

    def test_webster_rescores_isolated_sleep(self):
        """_apply_webster_rules rescores a brief sleep run inside prolonged wake.

        Rule 1: ≤ 1 sleep epoch surrounded by ≥ 4 wake on each side → wake.
        Rule 2: ≤ 2 sleep epochs surrounded by ≥ 10 wake on each side → wake.

        The test feeds handcrafted label arrays directly to _apply_webster_rules,
        bypassing Cole-Kripke.  This is correct: Webster rules operate on label
        sequences, not on raw activity counts.  (In real data the labels come from
        CK; here we construct them by hand to isolate the rescue logic.)
        """
        from obf_psychiatric_pipeline.features.sleep import _apply_webster_rules

        # Rule 1: single sleep epoch, 5 wake on each side → rescore as wake.
        labels_1 = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.int8)
        out_1 = _apply_webster_rules(labels_1)
        assert out_1[5] == 1, "isolated sleep (rule 1) should be rescored as wake"

        # Rule 2: two consecutive sleep epochs, 12 wake on each side → rescore.
        labels_2 = np.array([1] * 12 + [0, 0] + [1] * 12, dtype=np.int8)
        out_2 = _apply_webster_rules(labels_2)
        assert out_2[12] == 1, "2-epoch sleep run (rule 2) should be rescored as wake"
        assert out_2[13] == 1, "2-epoch sleep run (rule 2) should be rescored as wake"

        # Sanity check: a long sleep run is left alone.
        labels_long = np.array([1] * 5 + [0] * 10 + [1] * 5, dtype=np.int8)
        out_long = _apply_webster_rules(labels_long)
        assert (out_long[5:15] == 0).all(), "long sleep run should not be rescored"

    def test_score_sleep_preserves_datetime_index(self):
        """Output index matches input DatetimeIndex exactly."""
        activity = _make_flat_activity(n_days=2, value=0.0)
        labels = score_sleep(activity)
        pd.testing.assert_index_equal(labels.index, activity.index)

    def test_raises_on_invalid_scorer(self):
        activity = _make_flat_activity(n_days=1, value=0.0)
        with pytest.raises(ValueError, match="scorer"):
            score_sleep(activity, scorer="unknown_algo")

    def test_raises_on_non_datetime_index(self):
        activity = pd.Series([0.0, 1.0, 0.0], index=[0, 1, 2])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            score_sleep(activity)


# ---------------------------------------------------------------------------
# Tests: sleep_metrics
# ---------------------------------------------------------------------------


class TestSleepMetrics:

    def test_structured_rhythm_tst_in_valid_range(self):
        """Sleep pattern with 7h of quiet yields TST in (4, 8) hours.

        Pattern: hours 0–6 = 0 (sleep), hours 7–23 = 1000/min (wake).
        Cole-Kripke scores hours 0–6 as sleep, hours 7–23 as wake.
        Windowed TST should capture most of the quiet period.
        """
        pattern = np.array([0] * 7 + [1000] * 17, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        result = sleep_metrics(activity, bin_minutes=60, min_days=3)
        assert not math.isnan(result.tst_hours)
        assert 4.0 < result.tst_hours < 8.0

    def test_sleep_efficiency_in_unit_interval(self):
        """Sleep efficiency is always in [0, 1] for any valid recording."""
        pattern = np.array([0] * 7 + [1000] * 17, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        result = sleep_metrics(activity, bin_minutes=60, min_days=3)
        assert not math.isnan(result.sleep_efficiency)
        assert 0.0 <= result.sleep_efficiency <= 1.0

    def test_tst_24h_leq_24_hours(self):
        """24-hour TST is at most 24 hours regardless of signal."""
        pattern = np.array([0] * 7 + [1000] * 17, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        result = sleep_metrics(activity, bin_minutes=60, min_days=3)
        assert not math.isnan(result.tst_24h_hours)
        assert result.tst_24h_hours <= 24.0

    def test_tst_24h_geq_tst_windowed(self):
        """24-hour TST >= windowed TST because the window is a subset of the day."""
        pattern = np.array([0] * 7 + [1000] * 17, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=7)
        result = sleep_metrics(activity, bin_minutes=60, min_days=3)
        # 24h TST should be at least as large as windowed TST.
        assert result.tst_24h_hours >= result.tst_hours - 0.5  # 0.5h tolerance for boundary effects

    def test_too_few_days_returns_all_nan(self):
        """Recording with 2 days but min_days=3 returns SleepResult of all NaN."""
        pattern = np.array([0] * 7 + [1000] * 17, dtype=float)
        activity = _hourly_pattern_to_per_minute(pattern, n_days=2)
        result = sleep_metrics(activity, bin_minutes=60, min_days=3)
        assert all(math.isnan(v) for v in result)

    def test_raises_on_non_datetime_index(self):
        activity = pd.Series([0.0, 1.0], index=[0, 1])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            sleep_metrics(activity)

    # ---- OBF dataset cross-validation (placeholder) ------------------------

    @pytest.mark.skipif(
        not (Path("data") / "raw").exists(),
        reason="OBF raw actigraphy data not present at data/raw/",
    )
    def test_obf_participant_cross_validation(self):
        """
        Cross-validate sleep metrics for one OBF participant.

        HOW TO FILL THIS IN
        -------------------
        1.  Load per-minute activity for one participant.
        2.  Run sleep_metrics(activity) in a REPL and inspect all five fields.
        3.  If pyActigraphy is available, run its Cole-Kripke scorer on the
            same data and verify TST_24h_hours is broadly consistent.
        4.  Pin the five expected values.
        """
        TST_EXPECTED: float | None = None
        TST_24H_EXPECTED: float | None = None
        WASO_EXPECTED: float | None = None
        SE_EXPECTED: float | None = None
        SOL_EXPECTED: float | None = None
        activity: pd.Series | None = None  # FILL IN

        if activity is None or any(
            x is None
            for x in [TST_EXPECTED, TST_24H_EXPECTED, WASO_EXPECTED, SE_EXPECTED, SOL_EXPECTED]
        ):
            pytest.skip("OBF sleep cross-validation not yet configured.")

        result = sleep_metrics(activity, bin_minutes=60, min_days=3)
        assert result.tst_hours == pytest.approx(TST_EXPECTED, abs=0.5)
        assert result.tst_24h_hours == pytest.approx(TST_24H_EXPECTED, abs=0.5)
        assert result.waso_minutes == pytest.approx(WASO_EXPECTED, abs=10.0)
        assert result.sleep_efficiency == pytest.approx(SE_EXPECTED, abs=0.05)
        assert result.sol_minutes == pytest.approx(SOL_EXPECTED, abs=10.0)