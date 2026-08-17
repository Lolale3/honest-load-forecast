"""S6 -- the dual evaluation. This is what the project exists for.

Everything before this stage was making one comparison possible: take a load
forecasting model and score it twice, once on the weather that actually
happened and once on the weather that was genuinely available a day ahead.

THREE CONFIGURATIONS, NOT TWO

    A   train on observed,  test on observed    the standard, flattering result
    B   train on observed,  test on forecast    the naive model meeting reality
    C   train on forecast,  test on forecast    the honest model, trained honestly

A vs B is the headline. A model built and validated the conventional way,
then fed the inputs it would really have had. The gap between them is the
cost of an evaluation that was never realistic.

B vs C is the constructive part. C learns the relationship between FORECAST
temperature and demand -- including the forecast's own biases -- so it should
beat B. If you must forecast, train on forecasts.

C is also the only fair comparison against ERCOT's own day-ahead number,
because C is the configuration ERCOT was actually operating in.

WHY THIS IS NOT CIRCULAR
The features for both sources are built by the same function with the same
parameters (see features.temperature_features). Feeding it identical inputs
produces identical outputs -- there is a test asserting exactly that. So any
difference measured here comes from the weather data, not from our code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features, models


# --------------------------------------------------------------------------
# The three configurations
# --------------------------------------------------------------------------

CONFIGS = {
    "A_observed_observed": ("act", "act"),
    "B_observed_forecast": ("act", "fc1"),
    "C_forecast_forecast": ("fc1", "fc1"),
}


# --------------------------------------------------------------------------
# Config D -- the operational split
# --------------------------------------------------------------------------
#
# Standing at T-24 and predicting hour T, some temperature features describe
# the past and some describe the target hour. Only the second group has to be
# forecast; the first is already observed.
#
#   NOT KNOWN at T-24 -- must come from the forecast:
#     the current-hour terms (hinges, interactions)
#     t_lag1, t_lag2, t_lag3   -- these cover T-1..T-3, i.e. 21-23 hours
#                                 into OUR future, despite the name
#     t_mean24, t_mean48       -- windows dominated by unobserved hours
#     hours_cold, hours_hot    -- duration counters ending at T-1
#
#   KNOWN at T-24 -- use the observed value:
#     t_lag24                  -- exactly the decision moment
#     t_mean72, t_mean168      -- mostly historical
#
# APPROXIMATION, STATED RATHER THAN HIDDEN: t_mean72 and t_mean168 are shifted
# by one hour, not by the horizon, so roughly 33% and 14% of those windows
# technically postdate the decision point. Fixing that properly means
# re-shifting in S4 and rebuilding. We accept the approximation here and note
# it, because the contamination is in the OLDEST, least influential part of a
# long average -- but it is a compromise, not a clean result.

KNOWN_AT_DECISION_TIME = ("t_lag24", "t_mean72", "t_mean168")


def operational_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split features into (forecast-sourced, observed-sourced) for config D.

    Returns two lists whose concatenation is the model's feature set, with
    each temperature feature drawn from whichever source would genuinely be
    available at decision time.
    """
    fc1_cols = features.feature_columns(df, "fc1")

    forecast_side, observed_side = [], []
    for col in fc1_cols:
        if col.startswith("fc1_") and any(col.endswith(k) for k in KNOWN_AT_DECISION_TIME):
            observed_side.append(col.replace("fc1_", "act_", 1))
        else:
            forecast_side.append(col)

    return forecast_side, observed_side


def run_operational(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.Series, models.Scores]:
    """Config D: forecast weather for the target hour, observed for the past.

    Trained the same way it is served -- the feature set is identical at fit
    and predict time, so there is no train/serve mismatch and no need to
    neutralise column names.
    """
    forecast_side, observed_side = operational_columns(train)
    cols = forecast_side + observed_side

    missing = [c for c in cols if c not in train.columns]
    if missing:
        raise RuntimeError(f"operational split references missing columns: {missing[:5]}")

    model = models.fit(train, cols)
    pred = pd.Series(model.predict(test[cols]), index=test.index)

    return pred, models.score(test["demand"], pred)


def _neutralise(df: pd.DataFrame, columns: list[str], source: str) -> pd.DataFrame:
    """Strip the source prefix so both builds present identical column names.

    sklearn records feature names at fit time and refuses to predict with
    different ones -- correctly, since silently accepting them is how you get
    coefficients applied to the wrong variables. We want one fitted model to
    accept either source, so we rename `act_h20` and `fc1_h20` both to `h20`.

    This is safe only because the two builds are term-by-term aligned, which
    _assert_aligned verifies immediately before this is used.
    """
    subset = df[columns].copy()
    subset.columns = [c.replace(f"{source}_", "", 1) for c in columns]
    return subset


def run_config(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_source: str,
    test_source: str,
    *,
    vanilla: bool = False,
) -> tuple[pd.Series, models.Scores]:
    """Fit on `train_source` features, predict using `test_source` features.

    The two column lists are positionally aligned -- act_h_base sits at the
    same index as fc1_h_base -- so the fitted coefficients apply correctly to
    either. We assert that alignment rather than assume it.
    """
    train_cols = features.feature_columns(train, train_source)
    test_cols = features.feature_columns(test, test_source)

    if vanilla:
        train_cols = models.vanilla_columns(train_cols)
        test_cols = models.vanilla_columns(test_cols)

    _assert_aligned(train_cols, test_cols, train_source, test_source)

    X_train = _neutralise(train, train_cols, train_source)
    X_test = _neutralise(test, test_cols, test_source)

    fit_frame = X_train.copy()
    fit_frame["demand"] = train["demand"]

    model = models.fit(fit_frame, list(X_train.columns))
    pred = pd.Series(model.predict(X_test), index=test.index)

    return pred, models.score(test["demand"], pred)


def _assert_aligned(train_cols: list[str], test_cols: list[str], src_a: str, src_b: str) -> None:
    """The two feature lists must correspond term by term.

    If they did not, coefficient i would be applied to a different variable at
    prediction time and the whole comparison would be meaningless -- while
    still producing plausible numbers. Silent, and fatal to the argument.
    """
    if len(train_cols) != len(test_cols):
        raise RuntimeError(
            f"feature counts differ: {src_a} has {len(train_cols)}, {src_b} has {len(test_cols)}"
        )

    normalised_a = [c.replace(f"{src_a}_", "", 1) for c in train_cols]
    normalised_b = [c.replace(f"{src_b}_", "", 1) for c in test_cols]

    if normalised_a != normalised_b:
        mismatch = [(a, b) for a, b in zip(normalised_a, normalised_b) if a != b][:3]
        raise RuntimeError(f"feature lists are not term-by-term aligned. Examples: {mismatch}")


# --------------------------------------------------------------------------
# The mechanism
# --------------------------------------------------------------------------

def weather_error_vs_demand_error(
    test: pd.DataFrame,
    pred_forecast: pd.Series,
    pred_observed: pd.Series,
) -> pd.DataFrame:
    """Does demand error track weather error, and where?

    The project's claim is a causal chain: the weather forecast is wrong by a
    degree or two; the load-temperature curve is steep at the extremes; so the
    same weather error costs far more MW at the extremes than in the middle.

    This measures each link. `degradation` is the extra demand error caused by
    using forecast weather instead of observed -- per band, so we can see
    whether it concentrates where the curve is steep.
    """
    frame = pd.DataFrame(
        {
            "demand": test["demand"],
            "temp_actual": test["act_h_base"],
            "temp_error": test["fc1_h_base"] - test["act_h_base"],
            "err_observed": (pred_observed - test["demand"]).abs(),
            "err_forecast": (pred_forecast - test["demand"]).abs(),
        }
    ).dropna()

    frame["degradation"] = frame["err_forecast"] - frame["err_observed"]
    frame["band"] = pd.cut(frame["temp_actual"], bins=[-20, 0, 10, 20, 30, 35, 50])

    rows = []
    for band, group in frame.groupby("band", observed=True):
        rows.append(
            {
                "band": str(band),
                "hours": len(group),
                "mean_abs_temp_err_C": group["temp_error"].abs().mean(),
                "mae_observed_mw": group["err_observed"].mean(),
                "mae_forecast_mw": group["err_forecast"].mean(),
                "degradation_mw": group["degradation"].mean(),
                "degradation_pct": 100 * group["degradation"].mean() / group["err_observed"].mean(),
                "mw_per_deg_of_weather_err": (
                    group["degradation"].mean() / group["temp_error"].abs().mean()
                    if group["temp_error"].abs().mean() > 0.01
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def worst_weather_misses(
    test: pd.DataFrame,
    pred_forecast: pd.Series,
    pred_observed: pd.Series,
    n: int = 10,
) -> pd.DataFrame:
    """The hours where the weather forecast was most wrong.

    If the mechanism is real, these should also be hours where the demand
    forecast degraded sharply -- and they should cluster at the extremes
    rather than being scattered across mild days.
    """
    temp_err = (test["fc1_h_base"] - test["act_h_base"]).abs()
    worst = temp_err.nlargest(n).index

    return pd.DataFrame(
        {
            "temp_actual": test.loc[worst, "act_h_base"],
            "temp_forecast": test.loc[worst, "fc1_h_base"],
            "temp_err_C": (test.loc[worst, "fc1_h_base"] - test.loc[worst, "act_h_base"]),
            "demand": test.loc[worst, "demand"],
            "err_observed_mw": (pred_observed[worst] - test.loc[worst, "demand"]),
            "err_forecast_mw": (pred_forecast[worst] - test.loc[worst, "demand"]),
        }
    ).round(1)


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------

def run(
    feats: pd.DataFrame,
    test_start: str = models.DEFAULT_TEST_START,
    *,
    verbose: bool = True,
) -> dict:
    """Run all three configurations and compare."""
    train, test = models.time_split(feats, test_start)

    results = {}
    for name, (train_src, test_src) in CONFIGS.items():
        pred, scores = run_config(train, test, train_src, test_src)
        results[name] = {"pred": pred, "scores": scores}

    pred_d, scores_d = run_operational(train, test)
    results["D_operational"] = {"pred": pred_d, "scores": scores_d}

    bench = models.score(test["demand"], test["day_ahead_forecast"])

    if verbose:
        print(f"  train  {train.index.min():%Y-%m-%d} .. {train.index.max():%Y-%m-%d}  ({len(train):,} hours)")
        print(f"  test   {test.index.min():%Y-%m-%d} .. {test.index.max():%Y-%m-%d}  ({len(test):,} hours)")

        fc_side, obs_side = operational_columns(train)
        print(f"\n  config D feature split: {len(fc_side)} forecast-sourced, {len(obs_side)} observed-sourced")
        print(f"    observed: {[c.replace('act_', '') for c in obs_side]}")

        print("\n  " + "-" * 74)
        print("  CONFIGURATION                                   MAPE      MAE       RMSE")
        print("  " + "-" * 74)

        labels = {
            "A_observed_observed": "A  train observed, test observed   (standard)",
            "B_observed_forecast": "B  train observed, test forecast   (naive -> reality)",
            "C_forecast_forecast": "C  train forecast, test forecast   (all-forecast)",
            "D_operational": "D  forecast target, observed past  (operational)",
        }
        for key, label in labels.items():
            s = results[key]["scores"]
            print(f"  {label:<46} {s.mape:5.2f}%  {s.mae:6,.0f}    {s.rmse:6,.0f}")

        print(f"  {'ERCOT day-ahead (operational)':<46} {bench.mape:5.2f}%  {bench.mae:6,.0f}    {bench.rmse:6,.0f}")
        print("  " + "-" * 74)

        a = results["A_observed_observed"]["scores"].mape
        b = results["B_observed_forecast"]["scores"].mape
        c = results["C_forecast_forecast"]["scores"].mape
        d = results["D_operational"]["scores"].mape

        print(f"\n  THE HEADLINE")
        print(f"    A -> B   {a:.2f}% -> {b:.2f}%   the same model, fed the weather it would")
        print(f"                              actually have had: {100*(b-a)/a:+.1f}% worse")

        best_honest = min(b, c, d)
        best_name = {b: "B", c: "C", d: "D"}[best_honest]
        print(f"\n    best honest configuration: {best_name} at {best_honest:.2f}%")
        print(f"    irreducible cost of not knowing the weather: {100*(best_honest-a)/a:+.1f}%")
        print(f"\n    vs ERCOT  {best_honest:.2f}% vs {bench.mape:.2f}%")

    return {"train": train, "test": test, "results": results, "benchmark": bench}