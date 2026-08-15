"""S3 -- put demand and weather on one hourly index, honestly.

Two tables arrive from S1 and S2. This module makes them one, and it is the
stage where quiet corruption is easiest to introduce, so almost everything
here exists to make a problem VISIBLE rather than to fix it.

Three ideas do the work:

THE SPINE. We generate every hour in the window first, then attach both
sources to it. The naive alternative -- joining demand to weather directly --
silently drops any hour missing from both, so the result looks complete while
being short. You cannot count what is not there.

ORDER OF OPERATIONS. Interpolation happens AFTER the spine exists, never
before. On a frame with rows missing, pandas fills across a hole as if the
neighbouring rows were adjacent: a 6-hour gap gets treated as a 1-hour gap
and filled with a straight line that never happened. Plausible, silent, wrong.

TARGETS ARE NOT INPUTS. A missing temperature is a missing input, and a short
one can be reasonably estimated. A missing demand is a missing ANSWER --
filling it would mean inventing the thing we are trying to predict. So those
rows are dropped, never imputed.
"""

from __future__ import annotations

import pandas as pd

from . import eia, weather
from .config import Region, get_region

# Temperature gaps at or below this are interpolated; longer ones are left as
# NaN. Three hours is short enough that it can only ever repair a hiccup --
# it cannot paper over a real outage.
MAX_FILL_HOURS = 3

# Open-Meteo's archived GFS runs are missing here. Found empirically (see
# scripts/probe_archive2.py), not documented. Excluded by name so that a
# reader sees we found it rather than wondering why the data has a bite out
# of it.
KNOWN_OUTAGES: tuple[tuple[str, str], ...] = (
    ("2023-12-30", "2024-01-19"),  # 20 days, day-1 runs resume mid-day on the 19th
)


# --------------------------------------------------------------------------
# The spine
# --------------------------------------------------------------------------

def hourly_spine(start: str, end: str) -> pd.DatetimeIndex:
    """Every hour in [start, end], inclusive, UTC.

    `start` and `end` are dates (YYYY-MM-DD). The spine runs from 00:00 on the
    first to 23:00 on the last.
    """
    first = pd.Timestamp(f"{start} 00:00", tz="UTC")
    last = pd.Timestamp(f"{end} 23:00", tz="UTC")
    if last <= first:
        raise ValueError(f"end ({end}) must be after start ({start})")
    return pd.date_range(first, last, freq="h", name="period_utc")


def _year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split a date range into calendar-year pieces, clipped to the range.

    Chunks are what make a five-year pull survivable: each year is cached
    independently, so a failure costs one year rather than the whole window.
    """
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []
    for year in range(first.year, last.year + 1):
        lo = max(first, pd.Timestamp(f"{year}-01-01"))
        hi = min(last, pd.Timestamp(f"{year}-12-31"))
        if lo <= hi:
            chunks.append((lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")))
    return chunks


# --------------------------------------------------------------------------
# Gap analysis
# --------------------------------------------------------------------------

def gap_runs(series: pd.Series) -> pd.DataFrame:
    """Group missing values into consecutive runs.

    The SHAPE of missingness matters more than the count. 500 scattered single
    hours and one 500-hour block are the same number and completely different
    problems -- the first is noise, the second is an outage. A null count
    cannot tell them apart; this can.
    """
    missing = series.isna()
    if not missing.any():
        return pd.DataFrame(columns=["start", "end", "hours"])

    # A new run begins wherever the missing flag flips on.
    run_id = (missing != missing.shift()).cumsum()
    runs = (
        series.index.to_series()
        .groupby(run_id[missing])
        .agg(["min", "max", "count"])
        .rename(columns={"min": "start", "max": "end", "count": "hours"})
        .reset_index(drop=True)
    )
    return runs.sort_values("hours", ascending=False).reset_index(drop=True)


def audit(df: pd.DataFrame, max_runs: int = 5) -> None:
    """Print a full null audit -- every hour, every column.

    Deliberately not a sample. The archive probe found a 20-day outage only
    because it happened to straddle the 15th of a month; had it ended on the
    14th we would have walked into modelling with a silent hole in the data.
    Sampling can prove presence. It can never prove absence.
    """
    print(f"  {'column':<24} {'nulls':>8} {'%':>7}   largest gaps")
    print("  " + "-" * 74)

    for col in df.columns:
        if col.endswith("_filled") or col.endswith("_n_cities"):
            continue
        nulls = int(df[col].isna().sum())
        pct = 100 * nulls / len(df) if len(df) else 0.0

        runs = gap_runs(df[col])
        if runs.empty:
            detail = "none"
        else:
            top = runs.head(max_runs)
            detail = ", ".join(
                f"{r.start:%Y-%m-%d %H:%M}+{int(r.hours)}h" for r in top.itertuples()
            )
            if len(runs) > max_runs:
                detail += f", ... ({len(runs)} runs total)"

        print(f"  {col:<24} {nulls:>8,} {pct:>6.2f}%   {detail}")


# --------------------------------------------------------------------------
# Gap handling
# --------------------------------------------------------------------------

def fill_short_gaps(
    df: pd.DataFrame,
    columns: list[str],
    max_hours: int = MAX_FILL_HOURS,
) -> pd.DataFrame:
    """Interpolate gaps of `max_hours` or fewer; leave longer ones untouched.

    NOTE ON WHY THIS IS HAND-ROLLED. The obvious implementation is
    `.interpolate(limit=max_hours)`, and it is wrong twice over:

      * With limit_direction="both", `limit` applies from EACH end. A 4-hour
        gap with limit=3 gets filled 3-from-the-front and 3-from-the-back,
        i.e. completely. The cap silently doubles.
      * Even one-directionally, `limit` fills the FIRST 3 hours of a 10-hour
        gap and leaves the rest. That is not our rule. Our rule is that a long
        gap is not fillable at all -- partially filling it is worse than
        leaving it, because it hides the outage behind plausible numbers.

    So we identify runs first, decide per run, and only then interpolate.

    Every filled value is recorded in a `<column>_filled` column. That matters
    as much as the filling: it lets you re-run any final result on real-only
    rows and confirm the fills were not load-bearing. An assumption you can
    test is not really an assumption.

    Requires a complete index -- see the module docstring.
    """
    out = df.copy()

    for col in columns:
        if col not in out.columns:
            continue

        original = out[col]
        was_missing = original.isna()

        # Interpolate everything inside the observed range, then put back the
        # values belonging to gaps that are too long to fill.
        candidate = original.interpolate(method="time", limit_area="inside")

        too_long = pd.Series(False, index=out.index)
        for run in gap_runs(original).itertuples():
            if run.hours > max_hours:
                too_long.loc[run.start:run.end] = True

        filled = candidate.where(~too_long, original)

        out[col] = filled
        out[f"{col}_filled"] = was_missing & filled.notna()

    return out


def drop_known_outages(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove date ranges we know have no forecast data.

    Named explicitly rather than left to the null rules, so the exclusion is a
    documented decision rather than a side effect.
    """
    mask = pd.Series(False, index=df.index)
    for start, end in KNOWN_OUTAGES:
        lo = pd.Timestamp(f"{start} 00:00", tz="UTC")
        hi = pd.Timestamp(f"{end} 23:00", tz="UTC")
        mask |= (df.index >= lo) & (df.index <= hi)
    return df.loc[~mask], int(mask.sum())


# --------------------------------------------------------------------------
# The join
# --------------------------------------------------------------------------

def build(
    region: Region | str = "ercot",
    start: str = "2023-01-01",
    end: str = "2023-12-31",
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """One hourly table: demand, the operator's forecast, and both temperatures.

    Long windows are fetched one calendar year at a time. Each year caches
    independently, so a failure in year four costs you year four rather than
    the whole pull -- and neither API enjoys a single enormous request.
    """
    if isinstance(region, str):
        region = get_region(region)

    spine = hourly_spine(start, end)

    demand_parts, weather_parts = [], []
    for chunk_start, chunk_end in _year_chunks(start, end):
        if verbose:
            print(f"  fetching {chunk_start[:4]} ...", flush=True)
        demand_parts.append(eia.load_demand(region, f"{chunk_start}T00", f"{chunk_end}T23"))
        weather_parts.append(weather.load_weather(region, chunk_start, chunk_end))

    demand = pd.concat(demand_parts).sort_index()
    temps = pd.concat(weather_parts).sort_index()

    # Chunk boundaries are the classic place to introduce a duplicated hour.
    # Check rather than trust -- a duplicate would silently break the reindex.
    for name, frame in (("demand", demand), ("weather", temps)):
        dupes = frame.index.duplicated()
        if dupes.any():
            raise RuntimeError(
                f"{int(dupes.sum())} duplicated hours in {name} after stitching year chunks"
            )

    # Reindex onto the spine BEFORE joining. Now every hour exists as a row,
    # and anything absent from a source is a visible NaN.
    df = pd.DataFrame(index=spine)
    df = df.join(demand.reindex(spine)).join(temps.reindex(spine))

    if verbose:
        print(f"\nRAW -- {len(df):,} hours on the spine, before any handling\n")
        audit(df)

    df, dropped_outage = drop_known_outages(df)

    temp_cols = ["temp_actual"] + weather._forecast_columns(df)
    df = fill_short_gaps(df, temp_cols)

    before = len(df)
    df = df.dropna(subset=["demand"])
    dropped_demand = before - len(df)

    if verbose:
        print(f"\n  excluded {dropped_outage:,} hours in known outage windows")
        print(f"  dropped  {dropped_demand:,} hours with no demand value (missing target)")
        fills = {c: int(df[c].sum()) for c in df.columns if c.endswith("_filled")}
        for col, n in fills.items():
            if n:
                print(f"  filled   {n:,} values in {col.replace('_filled', '')}")
        print(f"\nALIGNED -- {len(df):,} hours\n")
        audit(df)

    return df


# --------------------------------------------------------------------------
# Verification -- does the join actually line up?
# --------------------------------------------------------------------------

def check_alignment(df: pd.DataFrame, region: Region | str = "ercot") -> None:
    """Test the join using physics rather than row counts.

    If demand and temperature were joined off by even one hour, everything
    downstream is subtly wrong and nothing raises. What catches it is the
    SHAPE of the relationship between them:

      summer  hot -> more demand   (air conditioning)   positive correlation
      winter  cold -> more demand  (electric heating)   negative correlation

    Both are strong. A misaligned join blurs them toward zero, because you
    would be comparing this hour's temperature against a different hour's
    demand.
    """
    if isinstance(region, str):
        region = get_region(region)

    local = df.tz_convert(region.local_timezone)
    valid = local[["demand", "temp_actual"]].dropna()

    summer = valid[valid.index.month.isin([6, 7, 8])]
    winter = valid[valid.index.month.isin([12, 1, 2])]

    print("  temperature vs demand correlation:")
    ok = True

    if len(summer) > 100:
        r = summer["demand"].corr(summer["temp_actual"])
        print(f"    summer (Jun-Aug)   {r:+.3f}   ({len(summer):,} hours)")
        if r < 0.5:
            ok = False
            print("      SUSPICIOUS -- expected strongly positive (air conditioning)")
    else:
        print("    summer            too few hours to judge")

    if len(winter) > 100:
        r = winter["demand"].corr(winter["temp_actual"])
        print(f"    winter (Dec-Feb)   {r:+.3f}   ({len(winter):,} hours)")
        if r > -0.3:
            ok = False
            print("      SUSPICIOUS -- expected negative (electric heating)")
    else:
        print("    winter            too few hours to judge")

    if ok:
        print("  -> both signs as physics requires. The join lines up.")

    # The operator's own forecast is a second, independent check. ERCOT's
    # day-ahead number should track actual demand very closely -- if it does
    # not, our two EIA columns are misaligned with each other.
    both = df[["demand", "day_ahead_forecast"]].dropna()
    if len(both) > 100:
        r = both["demand"].corr(both["day_ahead_forecast"])
        mape = ((both["day_ahead_forecast"] - both["demand"]).abs() / both["demand"]).mean() * 100
        print(f"\n  ERCOT's own day-ahead forecast: corr {r:+.4f}, MAPE {mape:.2f}%")
        print("  -> this is the benchmark to beat at S6")