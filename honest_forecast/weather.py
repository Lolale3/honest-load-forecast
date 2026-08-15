"""S2 -- weather ingestion. Two series, two meanings.

    ERA5 reanalysis        what the temperature ACTUALLY was
    Previous Runs          what the temperature was FORECAST to be

The gap between them is the whole project. A model trained and evaluated on
the first column looks good; the same model fed the second column is the one
you would actually have had to run.

Two design decisions worth knowing before you read the code:

MODEL PINNING. Open-Meteo's default `best_match` picks whichever model it
considers best for a location, and that choice can change as models are added
or improved. For measuring forecast SKILL that is poison -- you would be
measuring a series whose underlying source silently shifted. So we pin one
model. GFS is NOAA's operational global model, appropriate for a US grid, and
its 2 m temperature is archived from March 2021 rather than January 2024.

RENORMALISING WEIGHTS. A region's temperature is a population-weighted average
of several cities. If one city is missing an hour, a naive weighted mean sums
to 0.83 instead of 1.0 and the region looks colder than it was -- silently,
plausibly, wrongly. So we renormalise over whichever cities are present and
record how many contributed.
"""

from __future__ import annotations

import pandas as pd
import requests

from . import cache
from .config import Region, WeatherPoint, get_region

ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# See module docstring. Do not change this to best_match without understanding
# what it does to a skill comparison.
MODEL = "gfs_seamless"

# Lead times to fetch, in days before valid time. day1 is canonical -- it is
# the day-ahead decision. The others let us show skill degrading with lead
# time later, at almost no extra cost.
LEAD_DAYS = (1, 2, 3)

REQUEST_TIMEOUT = 90

# Earliest date the GFS temperature archive covers.
FORECAST_ARCHIVE_START = "2021-03-01"


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _request(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

    # Open-Meteo returns a JSON body with a human-readable reason on a 400.
    # Surfacing that beats a bare HTTPError.
    if response.status_code >= 400:
        # Note: don't use .get(key, response.text) -- Python evaluates the
        # default eagerly, so it would read the body even when we don't need to.
        try:
            body = response.json()
            reason = body.get("reason") or str(body)[:300]
        except ValueError:
            reason = response.text[:300]
        raise RuntimeError(f"Open-Meteo {response.status_code}: {reason}")

    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason')}")
    if "hourly" not in payload:
        raise RuntimeError(f"Unexpected Open-Meteo response: {str(payload)[:300]}")
    return payload


def _hourly_to_frame(payload: dict) -> pd.DataFrame:
    """Open-Meteo returns COLUMNAR data: parallel arrays under `hourly`.

        {"hourly": {"time": [...], "temperature_2m": [...], ...}}

    Unlike EIA's row-per-observation format, this is already wide. But the
    arrays must be the same length or the rows silently misalign -- so we
    check rather than trust.
    """
    hourly = payload["hourly"]
    if "time" not in hourly:
        raise RuntimeError("Open-Meteo response has no `time` array")

    lengths = {k: len(v) for k, v in hourly.items()}
    if len(set(lengths.values())) > 1:
        raise RuntimeError(f"Open-Meteo arrays have mismatched lengths: {lengths}")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    df.index.name = "period_utc"

    # JSON nulls arrive as None and give the column `object` dtype, which
    # then quietly breaks arithmetic and min/max later. Coerce everything to
    # float so a missing value is a proper NaN.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_index()


# --------------------------------------------------------------------------
# The two fetches
# --------------------------------------------------------------------------

def fetch_actual(point: WeatherPoint, start: str, end: str, *, force: bool = False) -> pd.DataFrame:
    """ERA5 reanalysis -- our stand-in for what the temperature actually was.

    Note "stand-in". ERA5 is a reanalysis: a model that assimilates
    observations, not a thermometer at this coordinate. It is the standard
    ground-truth proxy in this field, and it is still a proxy.
    """
    params = {
        "latitude": point.latitude,
        "longitude": point.longitude,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m",
        "timezone": "UTC",
    }
    key = f"era5_{point.name}_{start}_{end}"
    payload = cache.cached_json(key, lambda: _request(ERA5_URL, params), force=force)

    df = _hourly_to_frame(payload)
    return df[["temperature_2m"]].rename(columns={"temperature_2m": "temp_actual"})


def fetch_forecast(
    point: WeatherPoint,
    start: str,
    end: str,
    lead_days: tuple[int, ...] = LEAD_DAYS,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Archived forecasts at fixed lead times -- what we WOULD have had.

    `temperature_2m_previous_day1` is the value predicted 24 hours before
    valid time, day2 48 hours before, and so on.
    """
    variables = [f"temperature_2m_previous_day{d}" for d in lead_days]
    params = {
        "latitude": point.latitude,
        "longitude": point.longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "models": MODEL,
        "timezone": "UTC",
    }
    key = f"fcst_{MODEL}_{point.name}_{start}_{end}_lead{'-'.join(map(str, lead_days))}"
    payload = cache.cached_json(key, lambda: _request(PREVIOUS_RUNS_URL, params), force=force)

    df = _hourly_to_frame(payload)
    rename = {f"temperature_2m_previous_day{d}": f"temp_fcst_day{d}" for d in lead_days}
    missing = set(rename) - set(df.columns)
    if missing:
        raise RuntimeError(f"Open-Meteo did not return: {sorted(missing)}")
    return df[list(rename)].rename(columns=rename)


# --------------------------------------------------------------------------
# Per city, then the region
# --------------------------------------------------------------------------

def load_city(
    point: WeatherPoint,
    start: str,
    end: str,
    lead_days: tuple[int, ...] = LEAD_DAYS,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Actual and forecast temperature for one city, joined on the hour."""
    def build() -> pd.DataFrame:
        actual = fetch_actual(point, start, end, force=force)
        forecast = fetch_forecast(point, start, end, lead_days, force=force)
        # Outer join: we want to SEE where one source has hours the other
        # lacks, not quietly drop them.
        return actual.join(forecast, how="outer")

    key = f"weather_{point.name}_{start}_{end}"
    return cache.cached_frame(key, build, force=force)


def weighted_temperature(
    per_city: dict[str, pd.DataFrame],
    points: tuple[WeatherPoint, ...],
    column: str,
) -> pd.DataFrame:
    """Combine cities into one regional series, renormalising over what's there.

    Returns two columns: the weighted value, and how many cities contributed.
    That second column is the point -- it makes a partially-missing hour
    visible instead of letting it pass as a normal reading.
    """
    values = pd.DataFrame({p.name: per_city[p.name][column] for p in points})
    weights = pd.Series({p.name: p.weight for p in points})

    present = values.notna()
    live_weight = present.mul(weights, axis=1).sum(axis=1)

    weighted_sum = values.mul(weights, axis=1).sum(axis=1, skipna=True)

    # Renormalise. Hours with no city at all stay NaN rather than becoming 0.
    result = weighted_sum.where(live_weight > 0) / live_weight.where(live_weight > 0)

    return pd.DataFrame({column: result, f"{column}_n_cities": present.sum(axis=1)})


def load_weather(
    region: Region | str = "ercot",
    start: str = "2024-01-01",
    end: str = "2024-01-31",
    lead_days: tuple[int, ...] = LEAD_DAYS,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Regional actual and forecast temperature, UTC-indexed.

    Dates are YYYY-MM-DD here, not the hour strings S1 uses -- that is
    Open-Meteo's interface, not a choice of ours.
    """
    if isinstance(region, str):
        region = get_region(region)

    per_city = {p.name: load_city(p, start, end, lead_days, force=force) for p in region.weather_points}

    columns = ["temp_actual"] + [f"temp_fcst_day{d}" for d in lead_days]
    parts = [weighted_temperature(per_city, region.weather_points, c) for c in columns]
    return pd.concat(parts, axis=1).sort_index()


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def _forecast_columns(df: pd.DataFrame) -> list[str]:
    """Forecast value columns only.

    An earlier version used `startswith("temp_fcst_day")`, which also matched
    the `_n_cities` bookkeeping columns and broke the monotonic check below
    with counts masquerading as temperatures. Be specific about what you
    select; prefix matching is a loose net.
    """
    return sorted(
        c for c in df.columns if c.startswith("temp_fcst_day") and not c.endswith("_n_cities")
    )


def summarize(df: pd.DataFrame, region: Region | str = "ercot") -> None:
    """Print the checks that catch a plumbing bug.

    The one that matters most: forecast error should GROW with lead time.
    Tomorrow is easier to predict than three days out. If day3 looks more
    accurate than day1, something is wired backwards -- and no null count
    would ever tell you.
    """
    if isinstance(region, str):
        region = get_region(region)

    print(f"{region.name} weather")
    print(f"  rows            {len(df):,}")
    print(f"  from            {df.index.min()}")
    print(f"  to              {df.index.max()}")

    for col in ["temp_actual"] + _forecast_columns(df):
        nulls = int(df[col].isna().sum())
        pct = 100 * nulls / len(df) if len(df) else 0
        print(f"  {col:<20} nulls={nulls:<5} ({pct:4.1f}%)  min={df[col].min():6.1f}C  max={df[col].max():6.1f}C")

    full = len(region.weather_points)
    partial = int((df["temp_actual_n_cities"] < full).sum())
    print(f"\n  hours with fewer than {full} cities reporting: {partial:,}")

    print("\n  forecast error vs ERA5 (mean absolute, degrees C):")
    errors = {}
    for col in _forecast_columns(df):
        overlap = df[[col, "temp_actual"]].dropna()
        if overlap.empty:
            print(f"    {col:<20} no overlapping hours")
            continue
        mae = (overlap[col] - overlap["temp_actual"]).abs().mean()
        errors[col] = mae
        print(f"    {col:<20} {mae:.2f}   (over {len(overlap):,} hours)")

    if len(errors) >= 2:
        leads = sorted(errors)
        if all(errors[leads[i]] <= errors[leads[i + 1]] for i in range(len(leads) - 1)):
            print("  -> error grows with lead time, as it must. Wiring looks right.")
        else:
            print("  -> SUSPICIOUS. Longer lead times should be LESS accurate. Check the columns.")