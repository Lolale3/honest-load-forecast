"""S8 -- selective forecasting. Knowing which hours not to trust.

S7 established that even honestly-calibrated intervals fail where it matters:
below freezing, a nominal 90% interval covered 68%. Those hours cannot be
predicted well and cannot be covered by widening -- but they CAN be flagged.

So the output of this stage is not a better forecast. It is a triage rule:
route the least trustworthy hours to a human, let the rest through
automatically, and quantify the trade.

THE CONSTRAINT THAT MAKES IT HONEST
Every scoring signal must be computable at decision time, 24 hours ahead. A
score built from actual error would rank perfectly and be unusable -- the same
failure as `recoveries` in a credit model, or a 1-hour load lag here.

THE SIGNALS

  interval_width      from S7's conformal calibration: the model's own
                      band-conditional uncertainty
  lead_disagreement   spread across the day-1, day-2 and day-3 weather
                      forecasts for the same hour. If three forecasts issued
                      on different days agree, the atmosphere is predictable;
                      if they scatter, it is not. This measures METEOROLOGICAL
                      uncertainty and is entirely independent of the load
                      model
  temp_extremity      distance from the flat middle of the load-temperature
                      curve, where S6 measured 47 MW/degree against 767 at
                      the extremes
  training_density    how many similar hours the model actually learned from.
                      Sub-zero hours are 0.9% of training data

AND THEN THE ABLATION
A combined score is worthless if some of its parts do nothing. So each signal
is knocked out in turn and the curve re-measured. Publishing which signals
turned out to be redundant is the point, not an embarrassment -- a score you
have not ablated is a score you do not understand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Centre of the flat region of the load-temperature curve, in Celsius.
# From the S4 histogram: demand bottoms out between 15 and 20 C.
COMFORT_CENTRE = 17.5

# Bins used to estimate how much training data resembled each test hour.
DENSITY_BINS = np.arange(-15, 50, 2.5)

SIGNALS = ("interval_width", "lead_disagreement", "temp_extremity", "training_density")


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

def lead_disagreement(test: pd.DataFrame) -> pd.Series:
    """Spread across forecasts issued 1, 2 and 3 days before the same hour.

    Three independent attempts to predict one hour's temperature. Tight
    agreement means a predictable atmosphere; disagreement means the weather
    itself is uncertain, which is exactly when the load forecast should be
    trusted least.

    Available at decision time -- all three forecasts predate it.
    """
    cols = [c for c in ("fc1_h_base", "fc2_h_base", "fc3_h_base") if c in test.columns]

    if len(cols) < 2:
        # Only day-1 was built as features. Fall back to the raw forecast
        # columns if the aligned frame carried them.
        raw = [c for c in ("temp_fcst_day1", "temp_fcst_day2", "temp_fcst_day3") if c in test.columns]
        if len(raw) < 2:
            return pd.Series(np.nan, index=test.index)
        cols = raw

    return test[cols].std(axis=1)


def temp_extremity(test: pd.DataFrame, centre: float = COMFORT_CENTRE) -> pd.Series:
    """How far the forecast temperature sits from the comfort zone.

    Uses the FORECAST temperature, not the observed one -- the observed value
    is not known when the decision is made.
    """
    return (test["fc1_h_base"] - centre).abs()


def training_density(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Fraction of training hours at a similar temperature.

    Inverted and returned as a scarcity score: high means the model saw few
    comparable hours. Sub-zero temperatures are under 1% of training data,
    and the model's behaviour there is extrapolation more than learning.
    """
    counts, edges = np.histogram(train["act_h_base"].dropna(), bins=DENSITY_BINS)
    share = counts / max(counts.sum(), 1)

    idx = np.clip(np.digitize(test["fc1_h_base"], edges) - 1, 0, len(share) - 1)
    density = share[idx]

    return pd.Series(-density, index=test.index)


def build_signals(
    train: pd.DataFrame,
    test: pd.DataFrame,
    interval_width: pd.Series,
) -> pd.DataFrame:
    """All four signals, on their raw scales."""
    return pd.DataFrame(
        {
            "interval_width": interval_width,
            "lead_disagreement": lead_disagreement(test),
            "temp_extremity": temp_extremity(test),
            "training_density": training_density(train, test),
        },
        index=test.index,
    )


def combine(signals: pd.DataFrame, use: tuple[str, ...] = SIGNALS) -> pd.Series:
    """Combine signals into one uncertainty score.

    Each is converted to a percentile rank before averaging, because the raw
    scales are incomparable -- megawatts, degrees, and a probability share.
    Ranking also makes the score robust to the long tails these signals have.
    """
    chosen = [s for s in use if s in signals.columns]
    if not chosen:
        raise ValueError("no signals selected")

    ranked = signals[chosen].rank(pct=True, na_option="bottom")
    return ranked.mean(axis=1)


# --------------------------------------------------------------------------
# Risk-coverage
# --------------------------------------------------------------------------

def risk_coverage(
    score: pd.Series,
    actual: pd.Series,
    predicted: pd.Series,
    review_fractions: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Error on auto-accepted hours as a function of how many are reviewed.

    The highest-scoring hours are sent to a human and excluded. What remains
    is what the system would have shipped unattended, and its error is what
    the operator actually experiences.

    A useful score makes this curve fall steeply: reviewing a few hours should
    remove a disproportionate share of the error.
    """
    frame = pd.DataFrame({"score": score, "actual": actual, "pred": predicted}).dropna()
    frame["abs_err"] = (frame["pred"] - frame["actual"]).abs()
    frame["ape"] = frame["abs_err"] / frame["actual"]

    order = frame["score"].rank(ascending=False, method="first")

    rows = []
    for fraction in review_fractions:
        n_review = int(round(len(frame) * fraction))
        auto = frame[order > n_review]

        rows.append(
            {
                "reviewed_pct": 100 * fraction,
                "reviewed_hours": n_review,
                "auto_hours": len(auto),
                "auto_mape": 100 * auto["ape"].mean(),
                "auto_mae_mw": auto["abs_err"].mean(),
                "auto_worst_mw": auto["abs_err"].max(),
                "err_removed_pct": 100
                * (1 - auto["abs_err"].sum() / frame["abs_err"].sum()),
            }
        )

    return pd.DataFrame(rows)


def ablation(
    signals: pd.DataFrame,
    actual: pd.Series,
    predicted: pd.Series,
    review_fraction: float = 0.05,
) -> pd.DataFrame:
    """Knock out each signal in turn and measure what changes.

    The comparison is against a random-selection baseline. A signal whose
    removal does not move the number contributed nothing, and saying so is
    more useful than shipping it and implying it mattered.
    """
    available = tuple(signals.columns)
    if len(available) < 2:
        raise ValueError("ablation needs at least two signals to compare")

    full = combine(signals, available)
    base = risk_coverage(full, actual, predicted, (review_fraction,)).iloc[0]

    rng = np.random.default_rng(0)
    random_score = pd.Series(rng.random(len(signals)), index=signals.index)
    rand = risk_coverage(random_score, actual, predicted, (review_fraction,)).iloc[0]

    rows = [
        {
            "variant": "random selection",
            "auto_mape": rand["auto_mape"],
            "err_removed_pct": rand["err_removed_pct"],
            "vs_full_mape": rand["auto_mape"] - base["auto_mape"],
        },
        {
            "variant": "ALL SIGNALS",
            "auto_mape": base["auto_mape"],
            "err_removed_pct": base["err_removed_pct"],
            "vs_full_mape": 0.0,
        },
    ]

    for dropped in signals.columns:
        remaining = tuple(c for c in signals.columns if c != dropped)
        if not remaining:
            continue
        score = combine(signals, remaining)
        res = risk_coverage(score, actual, predicted, (review_fraction,)).iloc[0]
        rows.append(
            {
                "variant": f"without {dropped}",
                "auto_mape": res["auto_mape"],
                "err_removed_pct": res["err_removed_pct"],
                "vs_full_mape": res["auto_mape"] - base["auto_mape"],
            }
        )

    for only in signals.columns:
        score = combine(signals, (only,))
        res = risk_coverage(score, actual, predicted, (review_fraction,)).iloc[0]
        rows.append(
            {
                "variant": f"only {only}",
                "auto_mape": res["auto_mape"],
                "err_removed_pct": res["err_removed_pct"],
                "vs_full_mape": res["auto_mape"] - base["auto_mape"],
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The business question
# --------------------------------------------------------------------------

def review_budget(
    score: pd.Series,
    actual: pd.Series,
    predicted: pd.Series,
    hours_per_week: tuple[int, ...] = (2, 4, 8, 16),
) -> pd.DataFrame:
    """Translate the curve into something a planner can staff.

    "Review 5% of hours" is a statistician's unit. "Your analyst looks at
    eight hours a week" is a decision someone can actually approve, and
    whether it is worth it depends on the cost of a review against the cost
    of an error -- the same arithmetic as a credit cutoff.
    """
    total_weeks = len(score) / (24 * 7)

    rows = []
    for per_week in hours_per_week:
        fraction = (per_week * total_weeks) / len(score)
        if fraction >= 1:
            continue
        res = risk_coverage(score, actual, predicted, (fraction,)).iloc[0]
        rows.append(
            {
                "review_hours_per_week": per_week,
                "pct_of_all_hours": 100 * fraction,
                "auto_mape": res["auto_mape"],
                "auto_worst_mw": res["auto_worst_mw"],
                "err_removed_pct": res["err_removed_pct"],
            }
        )

    return pd.DataFrame(rows)