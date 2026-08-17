"""Tests for S7. All offline.

The coverage tests are the point. An interval that claims 90% and delivers
60% is worse than no interval at all, because it converts uncertainty into
false confidence.
"""

import numpy as np
import pandas as pd
import pytest

from honest_forecast import intervals


def hours(n: int, start: str = "2025-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", tz="UTC", name="period_utc")


# --------------------------------------------------------------------------
# Calibration split
# --------------------------------------------------------------------------

def test_calibration_comes_from_the_end_of_training():
    """Recent residuals describe the current grid. 2021 residuals describe a
    grid whose baseline load was 11 GW lower."""
    df = pd.DataFrame({"demand": np.arange(4000.0)}, index=hours(4000))
    fit, calib = intervals.split_for_calibration(df, fraction=0.25)

    assert fit.index.max() < calib.index.min()
    assert len(calib) == 1000
    assert calib["demand"].iloc[0] == 3000.0


def test_split_sizes_add_up():
    df = pd.DataFrame({"demand": np.arange(4000.0)}, index=hours(4000))
    fit, calib = intervals.split_for_calibration(df, fraction=0.3)

    assert len(fit) + len(calib) == len(df)


def test_absurd_fraction_is_refused():
    df = pd.DataFrame({"demand": np.arange(4000.0)}, index=hours(4000))

    with pytest.raises(ValueError, match="sensible range"):
        intervals.split_for_calibration(df, fraction=0.9)


def test_too_few_calibration_hours_is_refused():
    df = pd.DataFrame({"demand": np.arange(1000.0)}, index=hours(1000))

    with pytest.raises(ValueError, match="too few"):
        intervals.split_for_calibration(df, fraction=0.05)


# --------------------------------------------------------------------------
# THE COVERAGE GUARANTEE
# --------------------------------------------------------------------------

def test_calibrated_interval_achieves_its_nominal_coverage():
    """The core property. If residuals at test time look like residuals at
    calibration time, a 90% interval must cover about 90%."""
    rng = np.random.default_rng(0)
    n = 8000

    calib_resid = pd.Series(rng.normal(0, 1000, n), index=hours(n))
    calib_temps = pd.Series(rng.uniform(-5, 40, n), index=hours(n))

    cal = intervals.calibrate(calib_resid, calib_temps, level=0.90)

    test_idx = hours(n, "2026-01-01")
    point = pd.Series(np.full(n, 50000.0), index=test_idx)
    temps = pd.Series(rng.uniform(-5, 40, n), index=test_idx)
    actual = point + rng.normal(0, 1000, n)

    iv = intervals.apply_intervals(point, temps, cal)
    coverage = iv.covers(actual).mean()

    assert 0.87 < coverage < 0.93, f"expected ~90%, got {coverage:.1%}"


def test_wider_nominal_level_gives_wider_intervals():
    rng = np.random.default_rng(1)
    n = 5000
    resid = pd.Series(rng.normal(0, 1000, n), index=hours(n))
    temps = pd.Series(rng.uniform(0, 30, n), index=hours(n))

    point = pd.Series(np.full(100, 50000.0), index=hours(100, "2026-01-01"))
    t = pd.Series(np.full(100, 20.0), index=point.index)

    narrow = intervals.apply_intervals(point, t, intervals.calibrate(resid, temps, 0.50))
    wide = intervals.apply_intervals(point, t, intervals.calibrate(resid, temps, 0.95))

    assert wide.width().mean() > narrow.width().mean()


def test_understated_uncertainty_produces_under_coverage():
    """THE S7 HEADLINE, in miniature.

    Calibrate on quiet residuals, deploy where errors are twice as large, and
    coverage collapses. This is what happens when intervals are calibrated on
    observed weather and then used with forecast weather.
    """
    rng = np.random.default_rng(2)
    n = 6000

    quiet = pd.Series(rng.normal(0, 500, n), index=hours(n))
    temps = pd.Series(rng.uniform(0, 30, n), index=hours(n))
    cal = intervals.calibrate(quiet, temps, level=0.90)

    test_idx = hours(n, "2026-01-01")
    point = pd.Series(np.full(n, 50000.0), index=test_idx)
    t = pd.Series(rng.uniform(0, 30, n), index=test_idx)
    actual = point + rng.normal(0, 1200, n)   # much noisier than calibration

    iv = intervals.apply_intervals(point, t, cal)

    assert iv.covers(actual).mean() < 0.75, "under-coverage must be detected"


def test_asymmetric_residuals_give_asymmetric_intervals():
    """The model over-predicts in the cold. Forcing a symmetric interval
    would waste width on the side where it is already accurate."""
    rng = np.random.default_rng(3)
    n = 5000
    # Residuals skewed positive.
    resid = pd.Series(rng.gamma(2, 500, n) - 500, index=hours(n))
    temps = pd.Series(np.full(n, 20.0), index=hours(n))

    cal = intervals.calibrate(resid, temps, level=0.90)
    lo, hi = cal["pooled"]

    assert abs(hi) > abs(lo), "a right-skewed residual distribution needs a longer upper arm"


# --------------------------------------------------------------------------
# CONDITIONAL CALIBRATION
# --------------------------------------------------------------------------

def test_bands_with_larger_errors_get_wider_intervals():
    """Uncertainty is not constant. Cold hours have much larger residuals, so
    their intervals must be wider -- that is the whole point of calibrating
    per band rather than pooling."""
    rng = np.random.default_rng(4)
    n = 8000

    temps = pd.Series(rng.uniform(-10, 40, n), index=hours(n))
    # Error grows sharply below 5C.
    scale = np.where(temps < 5, 3000, 800)
    resid = pd.Series(rng.normal(0, scale), index=hours(n))

    cal = intervals.calibrate(resid, temps, level=0.90)

    cold = [v for k, v in cal["per_band"].items() if k.startswith("(-30")]
    mild = [v for k, v in cal["per_band"].items() if k.startswith("(15")]

    assert cold and mild
    cold_width = cold[0][1] - cold[0][0]
    mild_width = mild[0][1] - mild[0][0]
    assert cold_width > 2 * mild_width


def test_sparse_bands_fall_back_to_pooled_residuals():
    """A band with 12 hours cannot support its own quantile estimate."""
    rng = np.random.default_rng(5)
    n = 5000

    temps = pd.Series(np.concatenate([rng.uniform(15, 25, n - 12), np.full(12, -20.0)]), index=hours(n))
    resid = pd.Series(rng.normal(0, 800, n), index=hours(n))

    cal = intervals.calibrate(resid, temps, level=0.90, min_hours=100)

    assert not any(k.startswith("(-30") for k in cal["per_band"]), "sparse band must be excluded"

    point = pd.Series([50000.0], index=hours(1, "2026-01-01"))
    t = pd.Series([-20.0], index=point.index)
    iv = intervals.apply_intervals(point, t, cal)

    expected = cal["pooled"][1] - cal["pooled"][0]
    assert iv.width().iloc[0] == pytest.approx(expected)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_coverage_report_has_an_overall_row_and_band_rows():
    rng = np.random.default_rng(6)
    n = 3000
    idx = hours(n)

    point = pd.Series(np.full(n, 50000.0), index=idx)
    iv = intervals.Interval(point, point - 2000, point + 2000, 0.90)
    actual = point + rng.normal(0, 800, n)
    temps = pd.Series(rng.uniform(-5, 40, n), index=idx)

    report = intervals.coverage_report(iv, actual, temps)

    assert report.iloc[0]["band"] == "ALL"
    assert report.iloc[0]["hours"] == n
    assert len(report) > 3
    assert report.iloc[1:]["hours"].sum() == n, "bands must partition the hours"


def test_report_flags_over_and_under_coverage_by_sign():
    n = 2000
    idx = hours(n)
    point = pd.Series(np.full(n, 50000.0), index=idx)
    temps = pd.Series(np.full(n, 20.0), index=idx)

    # Interval far too wide -> over-coverage, positive gap.
    huge = intervals.Interval(point, point - 50000, point + 50000, 0.90)
    actual = point + np.random.default_rng(7).normal(0, 500, n)
    assert intervals.coverage_report(huge, actual, temps).iloc[0]["gap"] > 0

    # Interval far too narrow -> under-coverage, negative gap.
    tiny = intervals.Interval(point, point - 10, point + 10, 0.90)
    assert intervals.coverage_report(tiny, actual, temps).iloc[0]["gap"] < 0


def test_width_is_reported_so_coverage_cannot_be_gamed():
    """An interval of [0, infinity] covers everything. Width is what stops
    that being a good result."""
    n = 1000
    idx = hours(n)
    point = pd.Series(np.full(n, 50000.0), index=idx)
    temps = pd.Series(np.full(n, 20.0), index=idx)
    actual = point.copy()

    absurd = intervals.Interval(point, point - 1e6, point + 1e6, 0.90)
    report = intervals.coverage_report(absurd, actual, temps)

    assert report.iloc[0]["coverage"] == 1.0
    assert report.iloc[0]["width_pct_of_load"] > 1000, "the width column exposes it"