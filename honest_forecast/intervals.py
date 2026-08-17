"""S7 -- from a number to a range, and then checking the range is honest.

A point forecast cannot express what S6 established: this model is ~3% wrong
on average, 10+ GW wrong on hours nobody could predict, and roughly 47 MW per
degree of weather error in the flat middle of the temperature curve against
767 MW per degree in the cold. One number carries none of that.

METHOD -- SPLIT CONFORMAL PREDICTION

Rather than fitting a separate model per quantile (a linear program that
scales badly, and gradient boosting would abandon the interpretable model
class we chose deliberately), we calibrate the model we already have:

    1. split the training data by time into a FIT set and a CALIBRATION set
    2. fit the point model on the fit set only
    3. predict on the calibration set and collect the residuals
    4. the interval is the prediction plus empirical quantiles of those
       residuals

If 90% of calibration residuals lie within +/- 2,000 MW, then prediction
+/- 2,000 MW is a 90% interval. Under exchangeability this carries a coverage
guarantee rather than an approximation.

CONDITIONAL CALIBRATION -- THE PART THAT MATTERS

A constant-width interval is wrong for this problem. The model's error depends
enormously on temperature, so we calibrate separately within temperature
bands: narrow intervals on mild days, wide ones at the extremes.

Then we check coverage WITHIN each band. An interval that covers 95% of mild
hours and 60% of extreme hours averages to 90% and is worthless exactly where
the stakes are. Aggregate calibration hides conditional failure, which is the
same reason a single MAPE hid the model's real behaviour at S5.

SHARPNESS

Coverage alone is trivially gamed -- an interval of [0, infinity] covers
everything. So width is reported alongside it. The goal is to be as sharp as
possible SUBJECT TO being calibrated (Gneiting, Balabdaoui & Raftery 2007).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import evaluate, features, models

# Fraction of the training window held back for calibration. Taken from the
# END of training, so it is the most recent data and closest in distribution
# to the test period.
CALIBRATION_FRACTION = 0.25

# Temperature bands for conditional calibration. Coarser than the reporting
# bands because each needs enough calibration hours to estimate a quantile.
CALIBRATION_BANDS = (-30.0, 5.0, 15.0, 25.0, 32.0, 60.0)

# Below this many hours, a band cannot support its own quantile estimate and
# falls back to the pooled residuals.
MIN_BAND_HOURS = 100


@dataclass
class Interval:
    """A prediction interval over a set of hours."""

    point: pd.Series
    lower: pd.Series
    upper: pd.Series
    level: float

    def width(self) -> pd.Series:
        return self.upper - self.lower

    def covers(self, actual: pd.Series) -> pd.Series:
        return (actual >= self.lower) & (actual <= self.upper)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def split_for_calibration(
    train: pd.DataFrame,
    fraction: float = CALIBRATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into (fit, calibration), by time.

    Calibration comes from the END of the window. Residuals from 2021 would
    describe a grid whose baseline load was 11 GW lower; recent residuals are
    a better guide to what the model does now.
    """
    if not 0.05 <= fraction <= 0.5:
        raise ValueError(f"calibration fraction {fraction} is outside a sensible range")

    cut = int(len(train) * (1 - fraction))
    fit, calib = train.iloc[:cut], train.iloc[cut:]

    if len(calib) < 500:
        raise ValueError(f"only {len(calib)} calibration hours -- too few to estimate quantiles")

    return fit, calib


def _band_labels(temps: pd.Series, bands: tuple[float, ...] = CALIBRATION_BANDS) -> pd.Series:
    return pd.cut(temps, bins=list(bands))


def calibrate(
    residuals: pd.Series,
    temps: pd.Series,
    level: float = 0.90,
    bands: tuple[float, ...] = CALIBRATION_BANDS,
    min_hours: int = MIN_BAND_HOURS,
) -> dict:
    """Empirical residual quantiles, per temperature band.

    Returns lower and upper offsets for each band, plus a pooled fallback for
    bands with too few hours to estimate reliably.

    Asymmetric on purpose: the model's errors are not symmetric (S5 showed
    systematic over-prediction in the cold), so forcing a symmetric interval
    would waste width on the side where the model is already accurate.
    """
    alpha = 1 - level
    lo_q, hi_q = alpha / 2, 1 - alpha / 2

    pooled = (float(residuals.quantile(lo_q)), float(residuals.quantile(hi_q)))

    frame = pd.DataFrame({"resid": residuals, "band": _band_labels(temps, bands)})

    per_band = {}
    for band, group in frame.groupby("band", observed=True):
        if len(group) >= min_hours:
            per_band[str(band)] = (
                float(group["resid"].quantile(lo_q)),
                float(group["resid"].quantile(hi_q)),
            )

    return {"level": level, "pooled": pooled, "per_band": per_band, "bands": bands}


def apply_intervals(
    point: pd.Series,
    temps: pd.Series,
    calibration: dict,
) -> Interval:
    """Turn point predictions into intervals using calibrated offsets."""
    labels = _band_labels(temps, calibration["bands"])

    lo_offsets, hi_offsets = [], []
    for label in labels:
        lo, hi = calibration["per_band"].get(str(label), calibration["pooled"])
        lo_offsets.append(lo)
        hi_offsets.append(hi)

    lower = point + pd.Series(lo_offsets, index=point.index)
    upper = point + pd.Series(hi_offsets, index=point.index)

    return Interval(point=point, lower=lower, upper=upper, level=calibration["level"])


# --------------------------------------------------------------------------
# Assessment
# --------------------------------------------------------------------------

def coverage_report(
    interval: Interval,
    actual: pd.Series,
    temps: pd.Series,
    bands: tuple[float, ...] = (-20, 0, 10, 20, 30, 35, 50),
) -> pd.DataFrame:
    """Coverage and sharpness, overall and by temperature band.

    `coverage` is the fraction of hours the interval actually contained. It
    should match the nominal level. `width` is the average interval size --
    the price paid for that coverage.
    """
    covered = interval.covers(actual)
    width = interval.width()

    frame = pd.DataFrame(
        {"covered": covered, "width": width, "temp": temps, "actual": actual}
    ).dropna()
    frame["band"] = pd.cut(frame["temp"], bins=list(bands))

    rows = [
        {
            "band": "ALL",
            "hours": len(frame),
            "nominal": interval.level,
            "coverage": float(frame["covered"].mean()),
            "gap": float(frame["covered"].mean() - interval.level),
            "mean_width_mw": float(frame["width"].mean()),
            "width_pct_of_load": float(100 * frame["width"].mean() / frame["actual"].mean()),
        }
    ]

    for band, group in frame.groupby("band", observed=True):
        rows.append(
            {
                "band": str(band),
                "hours": len(group),
                "nominal": interval.level,
                "coverage": float(group["covered"].mean()),
                "gap": float(group["covered"].mean() - interval.level),
                "mean_width_mw": float(group["width"].mean()),
                "width_pct_of_load": float(100 * group["width"].mean() / group["actual"].mean()),
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------

def run(
    feats: pd.DataFrame,
    test_start: str = models.DEFAULT_TEST_START,
    level: float = 0.90,
    *,
    verbose: bool = True,
) -> dict:
    """Build intervals two ways and compare their honesty.

    NAIVE   calibrated on residuals computed with OBSERVED weather
    HONEST  calibrated on residuals computed with FORECAST weather

    Both are then deployed on forecast-weather predictions, which is what
    would actually happen. The naive intervals never saw weather uncertainty
    during calibration, so they should be too narrow -- meaning the
    conventional approach overstates not just accuracy but CONFIDENCE.
    """
    train, test = models.time_split(feats, test_start)
    fit_set, calib_set = split_for_calibration(train)

    act_cols = features.feature_columns(fit_set, "act")
    fc1_cols = features.feature_columns(fit_set, "fc1")
    evaluate._assert_aligned(act_cols, fc1_cols, "act", "fc1")

    # The point model, config B: trained on observed weather (S6 found this
    # is the best honest configuration -- training on forecasts damps the
    # temperature response through regression dilution).
    X_fit = evaluate._neutralise(fit_set, act_cols, "act")
    fit_frame = X_fit.copy()
    fit_frame["demand"] = fit_set["demand"]
    model = models.fit(fit_frame, list(X_fit.columns))

    def predict(frame: pd.DataFrame, source: str) -> pd.Series:
        cols = features.feature_columns(frame, source)
        X = evaluate._neutralise(frame, cols, source)
        return pd.Series(model.predict(X), index=frame.index)

    # Residuals on the calibration set, computed both ways.
    resid_observed = calib_set["demand"] - predict(calib_set, "act")
    resid_forecast = calib_set["demand"] - predict(calib_set, "fc1")

    cal_naive = calibrate(resid_observed, calib_set["act_h_base"], level)
    cal_honest = calibrate(resid_forecast, calib_set["act_h_base"], level)

    # Deploy both on forecast-weather predictions -- the realistic setting.
    point_test = predict(test, "fc1")

    iv_naive = apply_intervals(point_test, test["act_h_base"], cal_naive)
    iv_honest = apply_intervals(point_test, test["act_h_base"], cal_honest)

    rep_naive = coverage_report(iv_naive, test["demand"], test["act_h_base"])
    rep_honest = coverage_report(iv_honest, test["demand"], test["act_h_base"])

    if verbose:
        print(f"  fit          {fit_set.index.min():%Y-%m-%d} .. {fit_set.index.max():%Y-%m-%d}  ({len(fit_set):,} hours)")
        print(f"  calibration  {calib_set.index.min():%Y-%m-%d} .. {calib_set.index.max():%Y-%m-%d}  ({len(calib_set):,} hours)")
        print(f"  test         {test.index.min():%Y-%m-%d} .. {test.index.max():%Y-%m-%d}  ({len(test):,} hours)")

        n_all = rep_naive.iloc[0]
        h_all = rep_honest.iloc[0]

        print(f"\n  Both interval sets deployed on forecast-weather predictions.")
        print(f"  Nominal level: {level:.0%}\n")
        print("  " + "-" * 66)
        print(f"  {'CALIBRATED ON':<32} {'COVERAGE':>10} {'GAP':>9} {'WIDTH':>12}")
        print("  " + "-" * 66)
        print(f"  {'observed weather (naive)':<32} {n_all['coverage']:>9.1%} {n_all['gap']:>+9.1%} {n_all['mean_width_mw']:>9,.0f} MW")
        print(f"  {'forecast weather (honest)':<32} {h_all['coverage']:>9.1%} {h_all['gap']:>+9.1%} {h_all['mean_width_mw']:>9,.0f} MW")
        print("  " + "-" * 66)

        shortfall = level - n_all["coverage"]
        if shortfall > 0.01:
            print(f"\n  The naive intervals are too narrow by {shortfall:.1%} of coverage.")
            print("  Calibrating on observed weather does not just overstate accuracy --")
            print("  it overstates CONFIDENCE, which is the more dangerous error, because")
            print("  it is the one that persuades you to skip the review.")

    return {
        "model": model,
        "calibration": {"naive": cal_naive, "honest": cal_honest},
        "intervals": {"naive": iv_naive, "honest": iv_honest},
        "reports": {"naive": rep_naive, "honest": rep_honest},
        "test": test,
        "point": point_test,
    }