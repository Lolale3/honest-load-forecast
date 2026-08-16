"""Tests for S5. All offline.

The split tests are the important ones. A broken time split produces a model
that scores brilliantly and cannot exist.
"""

import numpy as np
import pandas as pd
import pytest

from honest_forecast import models


def frame(n: int = 1000, start: str = "2025-06-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    temp = 25 + 10 * np.sin(np.arange(n) * 2 * np.pi / 24)
    return pd.DataFrame(
        {
            "demand": 50000 + 500 * temp + rng.normal(0, 200, n),
            "day_ahead_forecast": 50000 + 500 * temp,
            "act_t1": temp,
        },
        index=idx,
    )


# --------------------------------------------------------------------------
# THE SPLIT
# --------------------------------------------------------------------------

def test_test_set_is_strictly_after_train_set():
    """The property that makes the evaluation meaningful. If any test hour
    preceded any train hour, the model would have seen the future."""
    df = frame(2000, "2025-11-01")
    train, test = models.time_split(df, "2026-01-01")

    assert train.index.max() < test.index.min()


def test_split_boundary_is_inclusive_on_the_test_side():
    df = frame(2000, "2025-11-01")
    train, test = models.time_split(df, "2026-01-01")

    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    assert (train.index < cutoff).all()
    assert (test.index >= cutoff).all()


def test_no_rows_are_lost_or_duplicated():
    df = frame(2000, "2025-11-01")
    train, test = models.time_split(df, "2026-01-01")

    assert len(train) + len(test) == len(df)
    assert not train.index.intersection(test.index).size


def test_split_that_empties_either_side_is_refused():
    df = frame(500, "2025-06-01")

    with pytest.raises(ValueError, match="train and"):
        models.time_split(df, "2030-01-01")   # nothing in test

    with pytest.raises(ValueError, match="train and"):
        models.time_split(df, "2020-01-01")   # nothing in train


def test_split_preserves_chronological_order_within_each_side():
    df = frame(2000, "2025-11-01")
    train, test = models.time_split(df, "2026-01-01")

    assert train.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing


# --------------------------------------------------------------------------
# Vanilla feature selection
# --------------------------------------------------------------------------

def test_vanilla_excludes_history_features():
    cols = [
        "act_t1", "act_t2", "act_t3", "act_t1_x_hour_sin",
        "act_t_lag24", "act_t_mean24", "load_lag24", "load_lag168",
        "hour_sin", "trend",
    ]
    van = models.vanilla_columns(cols)

    assert "act_t1" in van
    assert "act_t1_x_hour_sin" in van
    assert "trend" in van, "Hong's vanilla model includes a chronological trend"
    assert "act_t_lag24" not in van
    assert "act_t_mean24" not in van
    assert "load_lag24" not in van


def test_vanilla_is_a_strict_subset_of_recency():
    cols = ["act_t1", "act_t_lag24", "load_lag24", "hour_sin"]
    assert set(models.vanilla_columns(cols)).issubset(set(cols))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_perfect_prediction_scores_zero():
    actual = pd.Series([100.0, 200.0, 300.0])
    s = models.score(actual, actual.copy())

    assert s.mape == pytest.approx(0.0)
    assert s.mae == pytest.approx(0.0)
    assert s.rmse == pytest.approx(0.0)


def test_mape_is_computed_correctly():
    actual = pd.Series([100.0, 100.0])
    pred = pd.Series([110.0, 90.0])     # 10% off each way

    assert models.score(actual, pred).mape == pytest.approx(10.0)


def test_mae_is_absolute_so_errors_do_not_cancel():
    actual = pd.Series([100.0, 100.0])
    pred = pd.Series([110.0, 90.0])     # +10 and -10

    s = models.score(actual, pred)
    assert s.mae == pytest.approx(10.0), "must not cancel to zero"


def test_rmse_punishes_large_errors_more_than_mae():
    actual = pd.Series([100.0, 100.0, 100.0, 100.0])
    even = pd.Series([105.0, 105.0, 105.0, 105.0])     # four small misses
    spiky = pd.Series([100.0, 100.0, 100.0, 120.0])    # one big miss, same MAE

    assert models.score(actual, even).mae == pytest.approx(models.score(actual, spiky).mae)
    assert models.score(actual, spiky).rmse > models.score(actual, even).rmse


def test_score_ignores_missing_rows():
    actual = pd.Series([100.0, 200.0, None])
    pred = pd.Series([100.0, 200.0, 300.0])

    assert models.score(actual, pred).n == 2


def test_score_by_temperature_splits_into_bands():
    n = 300
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    temps = pd.Series(np.linspace(-10, 45, n), index=idx)
    actual = pd.Series(np.full(n, 50000.0), index=idx)
    pred = actual + 1000

    out = models.score_by_temperature(actual, pred, temps)

    assert len(out) > 3
    assert out["hours"].sum() == n
    assert (out["bias_mw"] > 0).all(), "a uniform over-prediction must show positive bias"


def test_bias_distinguishes_direction_from_magnitude():
    """MAE cannot tell over-prediction from under-prediction. Bias can, and
    systematic bias in one temperature band is a real finding."""
    idx = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    temps = pd.Series(np.full(100, 25.0), index=idx)
    actual = pd.Series(np.full(100, 50000.0), index=idx)
    under = actual - 500

    out = models.score_by_temperature(actual, under, temps)
    assert (out["bias_mw"] < 0).all()


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def test_model_recovers_a_known_linear_relationship():
    """Sanity: if demand really is 500 * temperature + constant, the fitted
    model should reproduce it almost exactly."""
    df = frame(1000)
    model = models.fit(df, ["act_t1"])

    pred = pd.Series(model.predict(df[["act_t1"]]), index=df.index)
    assert models.score(df["demand"], pred).mape < 1.0


def test_model_trained_on_train_does_not_see_test():
    """Belt and braces: fit on train only, then confirm the fitted object has
    no knowledge of test rows by checking it was never given them."""
    df = frame(2000, "2025-11-01")
    train, test = models.time_split(df, "2026-01-01")

    model = models.fit(train, ["act_t1"])
    n_seen = model.named_steps["scale"].n_samples_seen_

    assert n_seen == len(train)
    assert n_seen < len(df)