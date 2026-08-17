"""Tests for S6. All offline.

The alignment tests matter most. If the two feature lists were not aligned
term by term, coefficient i would be applied to a different variable at
prediction time -- and the result would still be a plausible-looking number.
Silent, and fatal to the argument the project is making.
"""

import numpy as np
import pandas as pd
import pytest

from honest_forecast import evaluate, features, models


def hours(n: int, start: str = "2025-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", tz="UTC", name="period_utc")


def aligned_frame(n: int = 3000, temp_error_scale: float = 1.0) -> pd.DataFrame:
    """Synthetic data where demand genuinely depends on temperature."""
    idx = hours(n)
    rng = np.random.default_rng(7)

    daily = 10 * np.sin(np.arange(n) * 2 * np.pi / 24)
    seasonal = 12 * np.sin(np.arange(n) * 2 * np.pi / (24 * 365))
    temp = 20 + daily + seasonal

    # Non-linear response: flat in the middle, steep above 25.
    demand = 50000 + 300 * temp + 800 * np.clip(temp - 25, 0, None) + rng.normal(0, 300, n)

    return pd.DataFrame(
        {
            "demand": demand,
            "day_ahead_forecast": demand + rng.normal(0, 800, n),
            "temp_actual": temp,
            "temp_fcst_day1": temp + rng.normal(0, temp_error_scale, n),
        },
        index=idx,
    )


def built(n: int = 3000, temp_error_scale: float = 1.0) -> pd.DataFrame:
    raw = aligned_frame(n, temp_error_scale)
    return features.drop_warmup(features.build(raw, verbose=False), verbose=False)


# --------------------------------------------------------------------------
# THE ALIGNMENT GUARD
# --------------------------------------------------------------------------

def test_aligned_feature_lists_pass():
    a = ["act_h_base", "act_h20", "hour_sin", "trend"]
    b = ["fc1_h_base", "fc1_h20", "hour_sin", "trend"]

    evaluate._assert_aligned(a, b, "act", "fc1")


def test_different_lengths_are_refused():
    a = ["act_h_base", "act_h20", "hour_sin"]
    b = ["fc1_h_base", "hour_sin"]

    with pytest.raises(RuntimeError, match="feature counts differ"):
        evaluate._assert_aligned(a, b, "act", "fc1")


def test_reordered_lists_are_refused():
    """Same features, wrong order. Coefficient i would multiply the wrong
    variable, and the output would still look like a demand forecast."""
    a = ["act_h_base", "act_h20", "hour_sin"]
    b = ["fc1_h20", "fc1_h_base", "hour_sin"]

    with pytest.raises(RuntimeError, match="not term-by-term aligned"):
        evaluate._assert_aligned(a, b, "act", "fc1")


def test_the_real_feature_lists_are_aligned():
    df = built()

    act = features.feature_columns(df, "act")
    fc1 = features.feature_columns(df, "fc1")

    evaluate._assert_aligned(act, fc1, "act", "fc1")


# --------------------------------------------------------------------------
# The three configurations
# --------------------------------------------------------------------------

def test_perfect_forecast_makes_all_three_configs_identical():
    """THE CONTROL EXPERIMENT.

    If the archived weather forecast were perfect, A, B and C would have to
    produce exactly the same numbers -- because the only thing distinguishing
    them is the weather data. Any difference here would mean our own code is
    introducing an asymmetry, and every result in S6 would be an artifact.
    """
    raw = aligned_frame(2000)
    raw["temp_fcst_day1"] = raw["temp_actual"]          # a flawless forecast
    df = features.drop_warmup(features.build(raw, verbose=False), verbose=False)

    train, test = models.time_split(df, str(df.index[1400]))

    _, a = evaluate.run_config(train, test, "act", "act")
    _, b = evaluate.run_config(train, test, "act", "fc1")
    _, c = evaluate.run_config(train, test, "fc1", "fc1")

    assert a.mape == pytest.approx(b.mape, rel=1e-9)
    assert a.mape == pytest.approx(c.mape, rel=1e-9)


def test_forecast_error_degrades_performance():
    """With a genuinely noisy forecast, B must be worse than A."""
    df = built(3000, temp_error_scale=2.5)
    train, test = models.time_split(df, str(df.index[2200]))

    _, a = evaluate.run_config(train, test, "act", "act")
    _, b = evaluate.run_config(train, test, "act", "fc1")

    assert b.mape > a.mape


def test_larger_weather_error_causes_larger_degradation():
    """The dose-response check. Doubling the weather error should widen the
    A-to-B gap, not leave it unchanged."""
    gaps = []
    for scale in (0.5, 3.0):
        df = built(3000, temp_error_scale=scale)
        train, test = models.time_split(df, str(df.index[2200]))
        _, a = evaluate.run_config(train, test, "act", "act")
        _, b = evaluate.run_config(train, test, "act", "fc1")
        gaps.append(b.mape - a.mape)

    assert gaps[1] > gaps[0], "more weather error must mean more degradation"


def test_all_configs_run_end_to_end():
    df = built(3000)
    out = evaluate.run(df, str(df.index[2200]), verbose=False)

    assert set(evaluate.CONFIGS).issubset(set(out["results"]))
    for res in out["results"].values():
        assert res["scores"].n > 0


# --------------------------------------------------------------------------
# The mechanism table
# --------------------------------------------------------------------------

def test_mechanism_table_covers_every_hour():
    df = built(3000)
    train, test = models.time_split(df, str(df.index[2200]))

    pred_a, _ = evaluate.run_config(train, test, "act", "act")
    pred_b, _ = evaluate.run_config(train, test, "act", "fc1")

    table = evaluate.weather_error_vs_demand_error(test, pred_b, pred_a)

    assert table["hours"].sum() == len(test)
    assert (table["mae_observed_mw"] >= 0).all()


def test_degradation_is_positive_when_the_forecast_is_noisy():
    df = built(3000, temp_error_scale=3.0)
    train, test = models.time_split(df, str(df.index[2200]))

    pred_a, _ = evaluate.run_config(train, test, "act", "act")
    pred_b, _ = evaluate.run_config(train, test, "act", "fc1")

    table = evaluate.weather_error_vs_demand_error(test, pred_b, pred_a)

    assert table["degradation_mw"].mean() > 0


def test_worst_weather_misses_returns_the_largest_errors():
    df = built(3000, temp_error_scale=2.0)
    train, test = models.time_split(df, str(df.index[2200]))

    pred_a, _ = evaluate.run_config(train, test, "act", "act")
    pred_b, _ = evaluate.run_config(train, test, "act", "fc1")

    worst = evaluate.worst_weather_misses(test, pred_b, pred_a, n=5)

    assert len(worst) == 5
    all_errors = (test["fc1_h_base"] - test["act_h_base"]).abs()
    assert worst["temp_err_C"].abs().min() >= all_errors.quantile(0.99) - 1e-9


# --------------------------------------------------------------------------
# CONFIG D -- the operational split
#
# The point of D is that at T-24 you already KNOW some of the temperature
# history. Using a forecast for it discards information you actually have.
# These tests pin down which features land on which side.
# --------------------------------------------------------------------------

def test_only_genuinely_known_features_come_from_observed():
    df = built(3000)
    forecast_side, observed_side = evaluate.operational_columns(df)

    stripped = sorted(c.replace("act_", "") for c in observed_side)
    assert stripped == ["t_lag24", "t_mean168", "t_mean72"]


def test_short_lags_are_forecast_sourced_despite_their_names():
    """t_lag1 covers T-1, which is 23 hours into the FUTURE from T-24. The
    name is misleading; the feature is not known."""
    df = built(3000)
    forecast_side, observed_side = evaluate.operational_columns(df)

    for lag in ("t_lag1", "t_lag2", "t_lag3"):
        assert any(c.endswith(lag) for c in forecast_side), f"{lag} must be forecast-sourced"
        assert not any(c.endswith(lag) for c in observed_side)


def test_current_hour_terms_are_forecast_sourced():
    df = built(3000)
    forecast_side, _ = evaluate.operational_columns(df)

    assert any(c == "fc1_h_base" for c in forecast_side)
    assert any("h_base_x_hour" in c for c in forecast_side)


def test_split_covers_every_feature_exactly_once():
    df = built(3000)
    forecast_side, observed_side = evaluate.operational_columns(df)

    all_fc1 = evaluate.features.feature_columns(df, "fc1")
    assert len(forecast_side) + len(observed_side) == len(all_fc1)

    stems = [c.replace("fc1_", "").replace("act_", "") for c in forecast_side + observed_side]
    assert len(stems) == len(set(stems)), "no feature may appear on both sides"


def test_operational_config_runs_and_scores():
    df = built(3000, temp_error_scale=2.0)
    train, test = models.time_split(df, str(df.index[2200]))

    pred, scores = evaluate.run_operational(train, test)

    assert len(pred) == len(test)
    assert scores.n == len(test)
    assert scores.mape > 0


def test_perfect_forecast_makes_d_match_a():
    """Control: with a flawless weather forecast, the operational split has
    nothing to gain and must equal the all-observed configuration."""
    raw = aligned_frame(2000)
    raw["temp_fcst_day1"] = raw["temp_actual"]
    df = features.drop_warmup(features.build(raw, verbose=False), verbose=False)

    train, test = models.time_split(df, str(df.index[1400]))

    _, a = evaluate.run_config(train, test, "act", "act")
    _, d = evaluate.run_operational(train, test)

    assert d.mape == pytest.approx(a.mape, rel=1e-9)


def test_run_includes_all_four_configurations():
    df = built(3000)
    out = evaluate.run(df, str(df.index[2200]), verbose=False)

    assert "D_operational" in out["results"]
    assert len(out["results"]) == 4