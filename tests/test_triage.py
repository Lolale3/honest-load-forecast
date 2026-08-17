"""Tests for S8. All offline.

Two things matter here. First, the score must not use information unavailable
at decision time -- a score built from actual error would rank perfectly and
be unusable. Second, the risk-coverage curve must behave sensibly: a good
score removes error faster than random selection.
"""

import numpy as np
import pandas as pd
import pytest

from honest_forecast import triage


def hours(n: int, start: str = "2026-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", tz="UTC", name="period_utc")


def make_frames(n: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    idx = hours(n)
    temps = rng.uniform(-10, 45, n)

    train = pd.DataFrame(
        {"act_h_base": rng.normal(20, 8, 5000)},
        index=hours(5000, "2021-01-01"),
    )
    test = pd.DataFrame(
        {
            "fc1_h_base": temps,
            "temp_fcst_day1": temps,
            "temp_fcst_day2": temps + rng.normal(0, 1, n),
            "temp_fcst_day3": temps + rng.normal(0, 2, n),
        },
        index=idx,
    )
    return train, test


# --------------------------------------------------------------------------
# SIGNALS MUST BE AVAILABLE AT DECISION TIME
# --------------------------------------------------------------------------

def test_extremity_uses_the_forecast_not_the_observed_temperature():
    """The observed temperature is not known 24 hours ahead. Using it would
    be leakage dressed up as a confidence signal."""
    idx = hours(3)
    test = pd.DataFrame(
        {"fc1_h_base": [17.5, 30.0, 5.0], "act_h_base": [40.0, 40.0, 40.0]},
        index=idx,
    )

    out = triage.temp_extremity(test)

    assert out.iloc[0] == pytest.approx(0.0), "comfort centre scores zero"
    assert out.iloc[1] == pytest.approx(12.5)
    # If it had used act_h_base every value would be 22.5.
    assert out.iloc[2] != pytest.approx(22.5)


def test_lead_disagreement_is_zero_when_forecasts_agree():
    idx = hours(2)
    test = pd.DataFrame(
        {
            "temp_fcst_day1": [20.0, 20.0],
            "temp_fcst_day2": [20.0, 25.0],
            "temp_fcst_day3": [20.0, 15.0],
        },
        index=idx,
    )

    out = triage.lead_disagreement(test)

    assert out.iloc[0] == pytest.approx(0.0), "perfect agreement means no uncertainty"
    assert out.iloc[1] > 0


def test_training_density_flags_rare_temperatures():
    """The model saw thousands of mild hours and a handful of freezing ones.
    Scarcity is returned as a positive score."""
    train, _ = make_frames()
    test = pd.DataFrame({"fc1_h_base": [20.0, -12.0]}, index=hours(2))

    out = triage.training_density(train, test)

    assert out.iloc[1] > out.iloc[0], "the rare temperature must score higher"


def test_signals_frame_has_all_four():
    train, test = make_frames()
    width = pd.Series(np.random.default_rng(1).uniform(3000, 12000, len(test)), index=test.index)

    signals = triage.build_signals(train, test, width)

    assert list(signals.columns) == list(triage.SIGNALS)
    assert len(signals) == len(test)


# --------------------------------------------------------------------------
# Combination
# --------------------------------------------------------------------------

def test_combine_uses_ranks_so_scales_cannot_dominate():
    """Raw signals are megawatts, degrees and a probability share. Without
    ranking, whichever has the largest numbers would swamp the rest."""
    idx = hours(4)
    signals = pd.DataFrame(
        {
            "interval_width": [1e6, 2e6, 3e6, 4e6],   # huge
            "temp_extremity": [30.0, 20.0, 10.0, 0.0],  # small, and reversed
        },
        index=idx,
    )

    score = triage.combine(signals, ("interval_width", "temp_extremity"))

    # Perfectly opposed signals of wildly different scale must cancel.
    assert score.nunique() == 1 or score.std() < 0.01


def test_combine_rejects_an_empty_selection():
    signals = pd.DataFrame({"a": [1.0]}, index=hours(1))
    with pytest.raises(ValueError, match="no signals"):
        triage.combine(signals, ())


# --------------------------------------------------------------------------
# RISK-COVERAGE
# --------------------------------------------------------------------------

def test_reviewing_nothing_reproduces_the_full_error():
    n = 500
    idx = hours(n)
    rng = np.random.default_rng(2)
    actual = pd.Series(rng.uniform(40000, 80000, n), index=idx)
    pred = actual + rng.normal(0, 1500, n)
    score = pd.Series(rng.random(n), index=idx)

    curve = triage.risk_coverage(score, actual, pred, (0.0,))

    assert curve.iloc[0]["reviewed_hours"] == 0
    assert curve.iloc[0]["auto_hours"] == n
    assert curve.iloc[0]["err_removed_pct"] == pytest.approx(0.0)


def test_a_perfect_score_removes_error_fastest():
    """Upper bound check. If the score were the actual error -- which is not
    available in practice -- reviewing 10% should remove far more than 10% of
    total error, because error is concentrated."""
    n = 1000
    idx = hours(n)
    rng = np.random.default_rng(3)
    actual = pd.Series(np.full(n, 60000.0), index=idx)
    err = rng.exponential(1000, n)
    pred = actual + err

    oracle = triage.risk_coverage(pd.Series(err, index=idx), actual, pred, (0.10,))
    random = triage.risk_coverage(pd.Series(rng.random(n), index=idx), actual, pred, (0.10,))

    assert oracle.iloc[0]["err_removed_pct"] > random.iloc[0]["err_removed_pct"] * 2


def test_error_on_accepted_hours_falls_as_review_increases():
    n = 1000
    idx = hours(n)
    rng = np.random.default_rng(4)
    actual = pd.Series(np.full(n, 60000.0), index=idx)
    err = rng.exponential(1200, n)
    pred = actual + err
    score = pd.Series(err + rng.normal(0, 200, n), index=idx)   # informative but noisy

    curve = triage.risk_coverage(score, actual, pred, (0.0, 0.05, 0.10, 0.20))

    assert curve["auto_mape"].is_monotonic_decreasing


def test_reviewed_and_auto_hours_always_sum_to_the_total():
    n = 777
    idx = hours(n)
    rng = np.random.default_rng(5)
    actual = pd.Series(np.full(n, 60000.0), index=idx)
    pred = actual + rng.normal(0, 1000, n)
    score = pd.Series(rng.random(n), index=idx)

    curve = triage.risk_coverage(score, actual, pred, (0.0, 0.03, 0.17, 0.5))

    assert (curve["reviewed_hours"] + curve["auto_hours"] == n).all()


# --------------------------------------------------------------------------
# ABLATION
# --------------------------------------------------------------------------

def test_ablation_includes_a_random_baseline_and_every_variant():
    n = 800
    idx = hours(n)
    rng = np.random.default_rng(6)

    signals = pd.DataFrame(
        {
            "interval_width": rng.random(n),
            "lead_disagreement": rng.random(n),
            "temp_extremity": rng.random(n),
            "training_density": rng.random(n),
        },
        index=idx,
    )
    actual = pd.Series(np.full(n, 60000.0), index=idx)
    pred = actual + rng.normal(0, 1200, n)

    out = triage.ablation(signals, actual, pred)

    variants = set(out["variant"])
    assert "random selection" in variants
    assert "ALL SIGNALS" in variants
    for sig in signals.columns:
        assert f"without {sig}" in variants
        assert f"only {sig}" in variants


def test_a_useless_signal_shows_no_loss_when_removed():
    """The property the ablation exists to detect. Pure noise contributes
    nothing, and dropping it should not hurt."""
    n = 2000
    idx = hours(n)
    rng = np.random.default_rng(7)

    actual = pd.Series(np.full(n, 60000.0), index=idx)
    err = rng.exponential(1000, n)
    pred = actual + err

    signals = pd.DataFrame(
        {
            "useful": err + rng.normal(0, 100, n),
            "useful2": err + rng.normal(0, 300, n),
            "noise": rng.random(n),
        },
        index=idx,
    )

    out = triage.ablation(signals, actual, pred, review_fraction=0.10).set_index("variant")

    without_noise = out.loc["without noise", "auto_mape"]
    without_useful = out.loc["without useful", "auto_mape"]

    assert without_useful > without_noise, "dropping the real signal must hurt more"


# --------------------------------------------------------------------------
# Review budget
# --------------------------------------------------------------------------

def test_review_budget_converts_hours_per_week_into_fractions():
    n = 24 * 7 * 10   # ten weeks
    idx = hours(n)
    rng = np.random.default_rng(8)
    actual = pd.Series(np.full(n, 60000.0), index=idx)
    pred = actual + rng.exponential(1000, n)
    score = pd.Series(rng.random(n), index=idx)

    out = triage.review_budget(score, actual, pred, hours_per_week=(4,))

    # 4 hours a week out of 168 is about 2.4% of all hours.
    assert out.iloc[0]["pct_of_all_hours"] == pytest.approx(100 * 4 / 168, rel=0.01)