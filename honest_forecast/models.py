"""S5 -- the baseline model, and the evaluation that flatters it.

This stage deliberately produces a GOOD number using the standard method. S6
then re-scores the same model honestly and the number gets worse. You need
the flattering result before you can knock it down.

THE SPLIT IS THE WHOLE DISCIPLINE HERE
A random split would be a catastrophe. Shuffling puts July 2026 in training
and March 2022 in test, so the model learns the future in order to predict
the past. Load grew from 73 GW (2021) to 91 GW (2026), so a shuffled model
knows about data-centre growth before it happened, and scores brilliantly on
a task nobody could actually perform.

So the split is by TIME. Everything in test comes strictly after everything
in train.

WHY ORDINARY LEAST SQUARES
Not because it is the most accurate option -- it is not. Because the claim of
this project is about EVALUATION, not about model class. With a plain linear
regression reproducing a published benchmark, nobody can dismiss the finding
as an artifact of an exotic model. The same reasoning that makes logistic
regression correct for credit scoring: defensibility beats a decimal place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features

# Everything from this date onward is held out and never seen during fitting.
DEFAULT_TEST_START = "2026-01-01"

# Vanilla (Hong 2010) uses only the coincident temperature: polynomials and
# their interactions. No lags, no rolling means, no load history.
VANILLA_EXCLUDE = ("_t_lag", "_t_mean", "load_lag", "load_mean", "_hours_cold", "_hours_hot")

# The temperature basis knot count, used only for reporting.


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def time_split(df: pd.DataFrame, test_start: str = DEFAULT_TEST_START) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time. Never shuffle a time series.

    Returns (train, test) where every test timestamp is later than every
    train timestamp.
    """
    cutoff = pd.Timestamp(test_start, tz="UTC")

    train = df.loc[df.index < cutoff]
    test = df.loc[df.index >= cutoff]

    if train.empty or test.empty:
        raise ValueError(
            f"test_start={test_start} gives {len(train):,} train and {len(test):,} test rows"
        )
    if train.index.max() >= test.index.min():
        raise RuntimeError("train and test overlap -- the split is broken")

    return train, test


# --------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------

def vanilla_columns(all_columns: list[str]) -> list[str]:
    """Hong's Vanilla Benchmark: coincident temperature only, no history."""
    return [c for c in all_columns if not any(tag in c for tag in VANILLA_EXCLUDE)]


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def fit(train: pd.DataFrame, columns: list[str], target: str = "demand") -> Pipeline:
    """Fit OLS on the given columns.

    Scaling first because the cubic terms are enormous -- temperature cubed
    reaches ~69,000 while sine terms sit in [-1, 1]. Unscaled, that spread
    makes the normal equations numerically nasty. Scaling changes nothing
    about the fit, only its conditioning.
    """
    model = Pipeline([("scale", StandardScaler()), ("ols", LinearRegression())])
    model.fit(train[columns], train[target])
    return model


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

@dataclass
class Scores:
    mape: float          # mean absolute percentage error
    mae: float           # mean absolute error, MW
    rmse: float          # root mean squared error, MW -- punishes big misses
    n: int

    def __str__(self) -> str:
        return f"MAPE {self.mape:5.2f}%   MAE {self.mae:6,.0f} MW   RMSE {self.rmse:6,.0f} MW   n={self.n:,}"


def score(actual: pd.Series, predicted: pd.Series) -> Scores:
    """Standard point-forecast metrics.

    MAPE is what the field reports, so it makes our number comparable to
    ERCOT's published 2.40%. MAE in MW is included because a percentage hides
    the physical size: 2.4% of 90 GW is 2,160 MW, roughly two power plants.
    """
    both = pd.DataFrame({"a": actual, "p": predicted}).dropna()
    err = both["p"] - both["a"]

    return Scores(
        mape=float((err.abs() / both["a"]).mean() * 100),
        mae=float(err.abs().mean()),
        rmse=float(np.sqrt((err ** 2).mean())),
        n=len(both),
    )


def score_by_temperature(
    actual: pd.Series,
    predicted: pd.Series,
    temps: pd.Series,
    bins: tuple[float, ...] = (-20, 0, 10, 20, 30, 35, 40, 50),
) -> pd.DataFrame:
    """Error broken out by temperature band.

    This is where S6's story begins. A single average MAPE hides everything
    interesting: what matters is whether the model fails uniformly or fails
    hardest on the steep arms of the load-temperature curve, where a small
    weather error becomes a large demand error.
    """
    frame = pd.DataFrame({"a": actual, "p": predicted, "t": temps}).dropna()
    frame["band"] = pd.cut(frame["t"], bins=list(bins))

    rows = []
    for band, group in frame.groupby("band", observed=True):
        err = group["p"] - group["a"]
        rows.append(
            {
                "band": str(band),
                "hours": len(group),
                "mape": float((err.abs() / group["a"]).mean() * 100),
                "mae_mw": float(err.abs().mean()),
                "bias_mw": float(err.mean()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------

def run(
    feats: pd.DataFrame,
    test_start: str = DEFAULT_TEST_START,
    source: str = "act",
    *,
    verbose: bool = True,
) -> dict:
    """Fit vanilla and recency models, evaluate the standard way.

    `source="act"` means the features come from observed weather -- the
    flattering evaluation. S6 will re-run this with the archived forecast.
    """
    train, test = time_split(feats, test_start)

    all_cols = features.feature_columns(feats, source)
    van_cols = vanilla_columns(all_cols)

    if verbose:
        print(f"  train  {train.index.min():%Y-%m-%d} .. {train.index.max():%Y-%m-%d}  ({len(train):,} hours)")
        print(f"  test   {test.index.min():%Y-%m-%d} .. {test.index.max():%Y-%m-%d}  ({len(test):,} hours)")
        print(f"\n  vanilla features  {len(van_cols)}")
        print(f"  recency features  {len(all_cols)}")

    models, results = {}, {}

    for name, cols in (("vanilla", van_cols), ("recency", all_cols)):
        model = fit(train, cols)
        pred = pd.Series(model.predict(test[cols]), index=test.index)

        models[name] = (model, cols)
        results[name] = {"pred": pred, "scores": score(test["demand"], pred)}

    if verbose:
        print("\n  test-set performance (evaluated on OBSERVED weather):")
        for name in ("vanilla", "recency"):
            print(f"    {name:<9} {results[name]['scores']}")

        v, r = results["vanilla"]["scores"].mape, results["recency"]["scores"].mape
        improvement = 100 * (v - r) / v
        print(f"\n  recency improves on vanilla by {improvement:.1f}%")
        if 10 <= improvement <= 35:
            print("  -> in the range Hong reports (18-21%). Implementation looks sound.")
        else:
            print("  -> OUTSIDE the expected 18-21%. Worth investigating before trusting this.")

        bench = score(test["demand"], test["day_ahead_forecast"])
        print(f"\n  ERCOT's own day-ahead forecast on the same hours:")
        print(f"    {bench}")
        print("\n  NOTE: this is not yet a fair comparison. ERCOT forecast with real")
        print("  forecast weather; we used the weather that actually happened. S6 fixes that.")

    return {"train": train, "test": test, "models": models, "results": results}