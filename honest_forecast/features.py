"""S4 -- features.

We build on Tao's Vanilla Benchmark (Hong 2010), the standard reference model
in short-term load forecasting, plus the recency effect (Wang, Liu & Hong
2016):

    y = b0 + b1*Trend + Hour + Weekday + Month + Hour:Weekday + f(T)

with one deliberate departure from Hong, described below.

WHY NOT A CUBIC IN TEMPERATURE
Hong's f(T) is a cubic polynomial. We tried that first and it failed in a
specific, instructive way: the fitted model over-predicted sub-zero demand by
+3,700 MW on average, and missed five consecutive hours in January 2026 by
11-13 GW.

The cause is extrapolation. ERCOT has only ~375 sub-zero hours in five years,
and most belong to Winter Storms Elliott and Kingston -- genuine emergencies
with electric heating at full tilt. A single cubic fitted across the whole
range cannot be flat in the middle, steep on the hot side and MODERATELY
steep on the cold side; it bends once and keeps going. So it learned "below
freezing means crisis" from a sample consisting almost entirely of crises.

Polynomials misbehave at the edges of their data, which is exactly where the
data is thinnest. The fix is a PIECEWISE-LINEAR basis: hinge functions that
give each temperature range its own slope, fitted independently. Sparse cold
data then produces a flat cold slope rather than a wild extrapolation. This
is the standard approach in the semi-parametric load forecasting literature.

Why the piecewise basis still bends. ERCOT demand is flat between roughly 15
and 25 C and steep at both ends -- air conditioning above, electric heating
below. The hinges reproduce that shape without a polynomial's tail behaviour,
and that curvature is precisely what turns a 2-degree weather miss into a
multi-gigawatt demand miss at the extremes.

Why the interactions. 35 C at 3pm in August is not 35 C at 3am in April. The
same temperature produces different load depending on when it lands, so the
response curve itself has to vary by hour and season.

Recency: buildings have thermal mass. A hot afternoon still drives demand at
9pm, so lagged temperatures and rolling means matter. Hong reports 18-21%
improvement over vanilla at the aggregated level -- a useful correctness
check at S5.

THE ONE THING THIS MODULE DOES THAT HONG DOES NOT
Every temperature feature is built TWICE from identical code: once from
`temp_actual` (what happened) and once from `temp_fcst_day1` (what was
predicted a day ahead). Same construction, different source. That symmetry is
what makes the dual evaluation at S6 possible, and it is the entire point of
the project -- so the builder takes the source column as an argument rather
than hardcoding it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Region, get_region

# Recency: lags in hours, and rolling-mean windows in hours.
#
# The 72h and 168h windows were added after diagnosing a +3,100 MW cold bias.
# ERCOT's sub-zero training hours are ~0.9% of the data and concentrated in
# two multi-day emergencies (Elliott 2022, Kingston 2025). During a sustained
# freeze, demand is elevated for reasons the thermometer alone does not
# capture: buildings lose stored heat over days, heat pumps fall back to
# resistance heating, and behaviour changes. With only 24-48h of memory the
# model attributed all of that to "being cold" and then applied the same
# relationship to any cold hour -- over-predicting an ordinary January 2026
# cold snap by 13 GW.
TEMP_LAGS = (1, 2, 3, 24)
TEMP_WINDOWS = (24, 48, 72, 168)

# Thresholds for the freeze/heat duration counters, in Celsius.
COLD_THRESHOLD = 5.0
HOT_THRESHOLD = 30.0
DURATION_CAP = 120  # hours; beyond this the counter saturates

# Load lags. All are >= the 24-hour forecast horizon, so all are genuinely
# available at decision time. See _check_horizon_safety.
#
# 336h (two weeks) was added alongside the temperature windows: the model was
# also missing level shifts, under-predicting a mild May 2026 hour by 11.5 GW.
# ERCOT load grew from 73 GW (2021) to 91 GW (2026), and a linear trend term
# tracks that only crudely. Recent load anchors the level directly.
LOAD_LAGS = (24, 168, 336)
LOAD_WINDOWS = (168,)

FORECAST_HORIZON_HOURS = 24


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------

def _cyclical(values: pd.Series, period: int, name: str) -> pd.DataFrame:
    """Encode a cyclical variable as sine and cosine.

    Hour 23 and hour 0 are one hour apart, but as raw numbers they are 23
    apart. A model fed the raw integer learns a discontinuity that does not
    exist in the world. Projecting onto a circle removes it.
    """
    angle = 2 * np.pi * values / period
    return pd.DataFrame(
        {f"{name}_sin": np.sin(angle), f"{name}_cos": np.cos(angle)},
        index=values.index,
    )


def calendar_features(index: pd.DatetimeIndex, region: Region) -> pd.DataFrame:
    """Hour, weekday, month, trend -- computed in LOCAL time.

    Local time, not UTC. Human behaviour follows the local clock: people wake,
    work and sleep on Central Time regardless of what UTC says. Getting this
    wrong shifts every calendar feature by 5-6 hours and quietly wrecks the
    interactions.
    """
    local = index.tz_convert(region.local_timezone)

    df = pd.DataFrame(index=index)

    df["hour"] = local.hour
    df["weekday"] = local.dayofweek
    df["month"] = local.month
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)

    # Chronological trend. Hong includes one; some practitioners drop it. We
    # keep it because ERCOT's peak went from 73 GW (2021) to 91 GW (2026) --
    # largely data-centre growth. Without a trend the model learns an average
    # of five years and systematically underpredicts recent load.
    df["trend"] = np.arange(len(index), dtype=float) / (365.25 * 24)  # in years

    df = pd.concat(
        [
            df,
            _cyclical(df["hour"], 24, "hour"),
            _cyclical(df["weekday"], 7, "weekday"),
            _cyclical(pd.Series(local.dayofyear, index=index), 365.25, "doy"),
        ],
        axis=1,
    )

    return df


# --------------------------------------------------------------------------
# Temperature -- the part built twice
# --------------------------------------------------------------------------

# Knot positions for the piecewise-linear temperature basis, in Celsius.
# Chosen from the observed load-temperature curve: the comfort zone sits
# around 15-20 C, and the response steepens on both sides. Knots are placed
# where the slope changes, and more densely on the hot side because that is
# where ERCOT has both the most data and the steepest response.
TEMP_KNOTS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0)


def hinge_basis(temps: pd.Series, knots: tuple[float, ...] = TEMP_KNOTS) -> pd.DataFrame:
    """Piecewise-linear basis: one hinge per knot, plus the raw temperature.

    Each hinge is max(0, T - knot). Summed with fitted coefficients, these
    produce a continuous curve whose slope may change at every knot -- so the
    flat middle, the steep hot arm and a separately-fitted cold arm can all
    coexist.

    Compare a cubic, where one set of three coefficients has to describe the
    entire range at once. Beyond the data the cubic keeps accelerating; a
    hinge basis simply continues the last fitted slope, which is a far safer
    thing to do where observations are sparse.
    """
    out = pd.DataFrame(index=temps.index)
    t = temps.astype(float)

    out["h_base"] = t
    for knot in knots:
        label = f"h{int(knot)}" if knot >= 0 else f"hm{int(abs(knot))}"
        out[label] = (t - knot).clip(lower=0)

    return out


def _duration_below(temps: pd.Series, threshold: float, cap: int = DURATION_CAP) -> pd.Series:
    """Consecutive hours the temperature has been below `threshold`.

    This is the feature that distinguishes "cold since this morning" from
    "day three of a freeze" -- a distinction no instantaneous temperature can
    make, and one that matters enormously for demand. Capped so a single
    extraordinary event cannot dominate the fitted coefficient.
    """
    below = (temps < threshold).astype(int)
    # Cumulative count that resets whenever the condition breaks.
    groups = (below == 0).cumsum()
    run = below.groupby(groups).cumsum()
    return run.clip(upper=cap).astype(float)


def _duration_above(temps: pd.Series, threshold: float, cap: int = DURATION_CAP) -> pd.Series:
    """Consecutive hours above `threshold`. The summer counterpart."""
    above = (temps > threshold).astype(int)
    groups = (above == 0).cumsum()
    run = above.groupby(groups).cumsum()
    return run.clip(upper=cap).astype(float)


def temperature_features(
    temps: pd.Series,
    calendar: pd.DataFrame,
    prefix: str,
    *,
    lags: tuple[int, ...] = TEMP_LAGS,
    windows: tuple[int, ...] = TEMP_WINDOWS,
    knots: tuple[float, ...] = TEMP_KNOTS,
) -> pd.DataFrame:
    """Temperature basis, interactions and recency, for ONE source.

    `prefix` distinguishes the two builds -- "act" from observed weather,
    "fc1" from the day-ahead forecast. Everything else is identical, and it
    must stay identical: any asymmetry between the two would show up at S6 as
    a difference we would wrongly attribute to forecast error.
    """
    out = pd.DataFrame(index=temps.index)

    t = temps.astype(float)

    # Piecewise-linear response, replacing Hong's cubic. See module docstring.
    basis = hinge_basis(t, knots)
    for col in basis.columns:
        out[f"{prefix}_{col}"] = basis[col]

    # Interactions: the response curve varies by time of day and season. We
    # interact the raw temperature and the two steepest hinges rather than
    # every hinge, to keep the feature count manageable.
    for base_col in ("h_base", "h25", "h30"):
        if base_col not in basis.columns:
            continue
        base = basis[base_col]
        out[f"{prefix}_{base_col}_x_hour_sin"] = base * calendar["hour_sin"]
        out[f"{prefix}_{base_col}_x_hour_cos"] = base * calendar["hour_cos"]
        out[f"{prefix}_{base_col}_x_doy_sin"] = base * calendar["doy_sin"]
        out[f"{prefix}_{base_col}_x_doy_cos"] = base * calendar["doy_cos"]

    # Recency -- thermal mass. Yesterday's heat is still in the buildings.
    for lag in lags:
        out[f"{prefix}_t_lag{lag}"] = t.shift(lag)

    for window in windows:
        out[f"{prefix}_t_mean{window}"] = t.shift(1).rolling(window, min_periods=window).mean()

    # Duration: how long has this spell been going on? Shifted by one hour so
    # the counter describes the past, not the present.
    out[f"{prefix}_hours_cold"] = _duration_below(t, COLD_THRESHOLD).shift(1)
    out[f"{prefix}_hours_hot"] = _duration_above(t, HOT_THRESHOLD).shift(1)

    return out


# --------------------------------------------------------------------------
# Load lags -- where leakage would enter
# --------------------------------------------------------------------------

def _check_horizon_safety(lags: tuple[int, ...], horizon: int) -> None:
    """Refuse any load lag shorter than the forecast horizon.

    THIS IS THE LEAKAGE GUARD. Forecasting 24 hours ahead, you know what load
    was 24 hours ago -- that value exists before you commit. You do NOT know
    what it was 1 hour ago, because "1 hour before the target" is 23 hours
    into the future from where you stand.

    A 1-hour load lag would make the model look extraordinary and be
    unusable. Same failure as `recoveries` in a credit model: a feature that
    only exists after the fact.
    """
    unsafe = [lag for lag in lags if lag < horizon]
    if unsafe:
        raise ValueError(
            f"Load lags {unsafe} are shorter than the {horizon}h forecast horizon. "
            "Those values do not exist at decision time -- this is leakage."
        )


def load_features(
    demand: pd.Series,
    *,
    lags: tuple[int, ...] = LOAD_LAGS,
    windows: tuple[int, ...] = LOAD_WINDOWS,
    horizon: int = FORECAST_HORIZON_HOURS,
) -> pd.DataFrame:
    """Lagged demand and recent load level, guarded against the horizon."""
    _check_horizon_safety(lags, horizon)

    out = pd.DataFrame(index=demand.index)
    for lag in lags:
        out[f"load_lag{lag}"] = demand.shift(lag)

    # Rolling mean of recent load, shifted by the full horizon so every hour
    # in the window predates the decision point. A trend term captures growth
    # only as a straight line; this tracks the actual recent level, which
    # matters when load jumps (2026 peak is 7.5 GW above 2025).
    for window in windows:
        out[f"load_mean{window}"] = (
            demand.shift(horizon).rolling(window, min_periods=window).mean()
        )

    return out


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def build(
    df: pd.DataFrame,
    region: Region | str = "ercot",
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Aligned table (S3) -> model-ready feature table.

    Returns one frame containing the target, the calendar and load features,
    and BOTH temperature builds side by side. S5 and S6 select which
    temperature block to use; the frame itself stays neutral.
    """
    if isinstance(region, str):
        region = get_region(region)

    if not df.index.is_monotonic_increasing:
        raise ValueError("index must be sorted -- lags would be meaningless otherwise")

    calendar = calendar_features(df.index, region)

    actual = temperature_features(df["temp_actual"], calendar, "act")
    forecast = temperature_features(df["temp_fcst_day1"], calendar, "fc1")

    if len(actual.columns) != len(forecast.columns):
        raise RuntimeError("the two temperature builds must be symmetric")

    loads = load_features(df["demand"])

    out = pd.concat(
        [
            df[["demand", "day_ahead_forecast"]],
            calendar,
            loads,
            actual,
            forecast,
        ],
        axis=1,
    )

    if verbose:
        n_act = sum(c.startswith("act_") for c in out.columns)
        n_fc1 = sum(c.startswith("fc1_") for c in out.columns)
        print(f"  {len(out.columns)} columns: {n_act} actual-temp, {n_fc1} forecast-temp")
        print(f"  {len(out):,} rows before dropping warm-up")

    return out


def feature_columns(df: pd.DataFrame, source: str) -> list[str]:
    """Column names for one temperature source, plus the shared features.

    `source` is "act" or "fc1". This is the switch that makes the dual
    evaluation a one-word change rather than a second codebase.
    """
    if source not in ("act", "fc1"):
        raise ValueError(f"source must be 'act' or 'fc1', got {source!r}")

    other = "fc1" if source == "act" else "act"

    return [
        c
        for c in df.columns
        if c not in ("demand", "day_ahead_forecast")
        and not c.startswith(f"{other}_")
        and c not in ("hour", "weekday", "month")  # raw versions; cyclical ones are used
    ]


def drop_warmup(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Drop rows where any lag or rolling window is undefined.

    The longest window is 48 hours and the longest load lag is 168, so the
    first week of any run has incomplete features. Dropping is correct:
    filling them would be inventing history.
    """
    before = len(df)
    out = df.dropna()
    if verbose:
        print(f"  dropped {before - len(out):,} warm-up rows, {len(out):,} remain")
    return out


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> None:
    """Checks that would catch a feature bug.

    The important one is the temperature-response curve. We bin demand by
    temperature and print the shape. If the cubic is going to earn its place,
    the relationship must be visibly non-monotonic -- high at both ends, low
    in the middle. If it comes out as a straight line, something is wrong with
    either the data or our understanding of it.
    """
    print("\n  demand by temperature band (this is the shape the cubic must fit):")
    bands = pd.cut(df["act_h_base"], bins=[-20, 0, 5, 10, 15, 20, 25, 30, 35, 40, 50])
    by_band = df.groupby(bands, observed=True)["demand"].agg(["mean", "count"])

    peak = by_band["mean"].max()
    for band, row in by_band.iterrows():
        bar = "#" * int(40 * row["mean"] / peak)
        print(f"    {str(band):>14}  {row['mean']:>7,.0f} MW  {bar}  n={int(row['count']):,}")

    lo = by_band["mean"].idxmin()
    print(f"\n  minimum demand occurs in the {lo} band")
    print("  -> demand should rise on BOTH sides of that. If it does not, check the join.")

    # The two temperature builds must differ only by their source.
    act_cols = sorted(c.replace("act_", "") for c in df.columns if c.startswith("act_"))
    fc1_cols = sorted(c.replace("fc1_", "") for c in df.columns if c.startswith("fc1_"))
    if act_cols == fc1_cols:
        print(f"\n  the two temperature builds are symmetric ({len(act_cols)} features each)")
    else:
        print("\n  ASYMMETRIC BUILDS -- S6 would compare apples to oranges")
        print(f"    only in act: {sorted(set(act_cols) - set(fc1_cols))}")
        print(f"    only in fc1: {sorted(set(fc1_cols) - set(act_cols))}")

    diff = (df["act_h_base"] - df["fc1_h_base"]).abs()
    print(f"\n  |actual - forecast| temperature: mean {diff.mean():.2f} C, max {diff.max():.2f} C")
    print("  -> this gap is what S6 measures the cost of")