"""Tests for S4. All offline.

Two groups matter most:

  * the leakage guard -- a load lag shorter than the horizon must be refused
  * build symmetry -- the actual and forecast temperature features must be
    constructed identically, or S6 compares apples to oranges
"""

import numpy as np
import pandas as pd
import pytest

from honest_forecast import features
from honest_forecast.config import get_region


def hours(n: int, start: str = "2023-06-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", tz="UTC", name="period_utc")


def aligned_frame(n: int = 400) -> pd.DataFrame:
    idx = hours(n)
    rng = np.random.default_rng(0)
    temp = 25 + 10 * np.sin(np.arange(n) * 2 * np.pi / 24)
    return pd.DataFrame(
        {
            "demand": 50000 + 500 * temp + rng.normal(0, 100, n),
            "day_ahead_forecast": 50000 + 500 * temp,
            "temp_actual": temp,
            "temp_fcst_day1": temp + rng.normal(0, 1, n),
        },
        index=idx,
    )


# --------------------------------------------------------------------------
# THE LEAKAGE GUARD
# --------------------------------------------------------------------------

def test_load_lag_shorter_than_horizon_is_refused():
    """The single most important test in this module.

    Forecasting 24h ahead, a 1-hour load lag does not exist at decision time.
    Including it would make the model look extraordinary and be unusable --
    the same failure as `recoveries` in a credit default model.
    """
    with pytest.raises(ValueError, match="leakage"):
        features._check_horizon_safety((1, 24), horizon=24)


def test_lag_equal_to_horizon_is_allowed():
    """Exactly 24h ahead: yesterday's load at this hour IS known."""
    features._check_horizon_safety((24, 168), horizon=24)


def test_default_load_lags_are_safe():
    features._check_horizon_safety(features.LOAD_LAGS, features.FORECAST_HORIZON_HOURS)


def test_load_features_shift_in_the_right_direction():
    """A lag must reach into the PAST. If the sign were flipped we would be
    feeding the model the future and nothing would raise."""
    idx = hours(50)
    demand = pd.Series(np.arange(50, dtype=float), index=idx)

    out = features.load_features(demand, lags=(24,), horizon=24)

    assert pd.isna(out["load_lag24"].iloc[0]), "first rows have no history"
    assert out["load_lag24"].iloc[24] == 0.0, "row 24 sees row 0"
    assert out["load_lag24"].iloc[30] == 6.0


def test_temperature_lags_reach_backwards():
    idx = hours(50)
    temps = pd.Series(np.arange(50, dtype=float), index=idx)
    cal = features.calendar_features(idx, get_region("ercot"))

    out = features.temperature_features(temps, cal, "act", lags=(3,), windows=())

    assert out["act_t_lag3"].iloc[10] == 7.0


def test_rolling_mean_excludes_the_current_hour():
    """The rolling window is shifted by one. Including the current hour would
    leak the present into a feature meant to summarise the past."""
    idx = hours(60)
    temps = pd.Series(np.arange(60, dtype=float), index=idx)
    cal = features.calendar_features(idx, get_region("ercot"))

    out = features.temperature_features(temps, cal, "act", lags=(), windows=(24,))

    # At row 30, the mean of rows 6..29 is 17.5. Including row 30 would be 18.0.
    assert out["act_t_mean24"].iloc[30] == pytest.approx(17.5)


# --------------------------------------------------------------------------
# BUILD SYMMETRY
# --------------------------------------------------------------------------

def test_the_two_temperature_builds_have_identical_structure():
    """If these drift apart, S6's comparison silently measures the wrong thing."""
    out = features.build(aligned_frame(), verbose=False)

    act = sorted(c.replace("act_", "") for c in out.columns if c.startswith("act_"))
    fc1 = sorted(c.replace("fc1_", "") for c in out.columns if c.startswith("fc1_"))

    assert act == fc1
    assert len(act) > 15, "expected the full vanilla + recency block"


def test_identical_inputs_produce_identical_features():
    """Feed both sources the same temperatures; every pair must match exactly.
    Any difference would be a bug in the builder, not in the weather."""
    df = aligned_frame()
    df["temp_fcst_day1"] = df["temp_actual"]

    out = features.build(df, verbose=False)

    for col in [c for c in out.columns if c.startswith("act_")]:
        twin = col.replace("act_", "fc1_")
        pd.testing.assert_series_equal(
            out[col], out[twin], check_names=False,
            obj=f"{col} vs {twin}",
        )


def test_feature_columns_selects_one_source_only():
    out = features.build(aligned_frame(), verbose=False)

    act_cols = features.feature_columns(out, "act")
    fc1_cols = features.feature_columns(out, "fc1")

    assert not any(c.startswith("fc1_") for c in act_cols)
    assert not any(c.startswith("act_") for c in fc1_cols)
    assert len(act_cols) == len(fc1_cols), "same count, so models are comparable"
    assert "demand" not in act_cols, "the target must never be a feature"
    assert "day_ahead_forecast" not in act_cols, "ERCOT's forecast is a benchmark, not an input"


def test_feature_columns_rejects_a_bad_source():
    out = features.build(aligned_frame(), verbose=False)
    with pytest.raises(ValueError, match="must be 'act' or 'fc1'"):
        features.feature_columns(out, "actual")


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------

def test_calendar_uses_local_time_not_utc():
    """06:00 UTC is midnight in Texas. If we used UTC the whole daily profile
    would be shifted six hours and every interaction would be wrong."""
    idx = pd.DatetimeIndex(["2023-06-01 06:00"], tz="UTC")
    cal = features.calendar_features(idx, get_region("ercot"))

    assert cal["hour"].iloc[0] == 1, "06:00 UTC = 01:00 CDT"


def test_cyclical_encoding_makes_hour_23_adjacent_to_hour_0():
    idx = hours(24, "2023-06-01")
    cal = features.calendar_features(idx, get_region("ercot"))

    def point(h):
        row = cal[cal["hour"] == h].iloc[0]
        return np.array([row["hour_sin"], row["hour_cos"]])

    gap_23_to_0 = np.linalg.norm(point(23) - point(0))
    gap_0_to_1 = np.linalg.norm(point(0) - point(1))

    assert gap_23_to_0 == pytest.approx(gap_0_to_1, rel=1e-6)


def test_weekend_flag():
    # 2023-06-03 is a Saturday.
    idx = pd.DatetimeIndex(["2023-06-03 18:00", "2023-06-05 18:00"], tz="UTC")
    cal = features.calendar_features(idx, get_region("ercot"))

    assert cal["is_weekend"].tolist() == [1, 0]


def test_trend_increases_monotonically():
    cal = features.calendar_features(hours(100), get_region("ercot"))
    assert cal["trend"].is_monotonic_increasing


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def test_polynomial_terms_are_actual_powers():
    df = aligned_frame(100)
    out = features.build(df, verbose=False)

    assert out["act_t2"].iloc[50] == pytest.approx(out["act_t1"].iloc[50] ** 2)
    assert out["act_t3"].iloc[50] == pytest.approx(out["act_t1"].iloc[50] ** 3)


def test_unsorted_index_is_refused():
    df = aligned_frame(100).sort_index(ascending=False)
    with pytest.raises(ValueError, match="sorted"):
        features.build(df, verbose=False)


def test_warmup_rows_are_dropped():
    out = features.build(aligned_frame(400), verbose=False)
    clean = features.drop_warmup(out, verbose=False)

    assert clean.notna().all().all()
    # The longest requirement is load_lag168.
    assert len(clean) == 400 - 168