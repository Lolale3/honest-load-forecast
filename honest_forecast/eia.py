"""S1 -- ingest hourly demand and the operator's day-ahead forecast from EIA-930.

We ask EIA for two series at once:

    D   actual demand, as reported by the balancing authority
    DF  the day-ahead demand FORECAST that the operator published

That second one is the reason we use EIA rather than pulling from ERCOT
directly. It is a free external benchmark: at S6 we compare our model against
the number the grid operator actually committed to, rather than grading our
own homework.

Two things this module is careful about:

1. PAGINATION. The API caps a response at 5,000 rows. A paginated pull that
   quietly stops early produces a dataframe that looks completely normal and
   is missing months. So we compare what we collected against the `total` the
   API reports, and raise if they disagree.

2. THE PIVOT. Demand and forecast arrive as separate ROWS. Reshaping them into
   separate COLUMNS is safe only if each (period, type) appears exactly once.
   We check that explicitly, because the alternative is pandas silently
   averaging a duplicate and never telling us.
"""

from __future__ import annotations

import pandas as pd
import requests

from . import cache
from .config import Region, eia_api_key, get_region

BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

# The API's hard cap per response. Not a preference -- exceeding it silently
# truncates, which is why fetch_all() exists.
PAGE_SIZE = 5000

# EIA's codes for the two series we want, and the names we give them.
TYPE_MAP = {
    "D": "demand",
    "DF": "day_ahead_forecast",
}

REQUEST_TIMEOUT = 60


# --------------------------------------------------------------------------
# Layer 1: one HTTP request. The ONLY place in this module that hits network.
# --------------------------------------------------------------------------

def _request(params: dict) -> dict:
    response = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    # EIA can return HTTP 200 with an error inside the body. Catch that here
    # rather than letting it become a confusing KeyError later.
    if "response" not in payload:
        raise RuntimeError(f"Unexpected EIA response: {str(payload)[:400]}")
    return payload


def _params(region: Region, start: str, end: str, offset: int) -> dict:
    """Build the query.

    `start` and `end` are hour strings like "2024-01-01T00". End is exclusive
    of nothing in particular -- EIA treats it inclusively -- so we always pass
    the last hour we actually want.
    """
    return {
        "api_key": eia_api_key(),
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": region.eia_respondent,
        "facets[type][]": list(TYPE_MAP),
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": offset,
        "length": PAGE_SIZE,
    }


def _fetch_page(region: Region, start: str, end: str, offset: int, *, force: bool = False) -> dict:
    """One page, cached to disk so re-runs cost nothing."""
    key = f"eia_{region.key}_{start}_{end}_offset{offset}"
    return cache.cached_json(key, lambda: _request(_params(region, start, end, offset)), force=force)


# --------------------------------------------------------------------------
# Layer 2: all pages, with the guard
# --------------------------------------------------------------------------

def fetch_all(region: Region, start: str, end: str, *, force: bool = False) -> list[dict]:
    """Collect every row for the window, or raise.

    The guard is the point of this function. `total` is what the API says
    matches our query; we keep paginating until we have that many. If we end
    up short, that is a silent-data-loss bug and it stops the pipeline.
    """
    rows: list[dict] = []
    offset = 0
    total: int | None = None

    while True:
        payload = _fetch_page(region, start, end, offset, force=force)
        body = payload["response"]

        if total is None:
            total = int(body.get("total", 0))
            if total == 0:
                raise RuntimeError(
                    f"EIA reports 0 rows for {region.key} between {start} and {end}. "
                    "Check the dates and the respondent code."
                )
            if total > 200_000:
                raise RuntimeError(
                    f"EIA reports {total:,} rows -- that is {total // PAGE_SIZE + 1} requests. "
                    "Narrow the date range."
                )

        page = body.get("data", [])
        if not page:
            break

        rows.extend(page)
        offset += len(page)

        if offset >= total:
            break

    if len(rows) != total:
        raise RuntimeError(
            f"Pagination lost rows: collected {len(rows):,} of {total:,} reported by EIA. "
            "Refusing to continue with incomplete data."
        )

    return rows


# --------------------------------------------------------------------------
# Layer 3: the only function anything else should call
# --------------------------------------------------------------------------

def _to_frame(rows: list[dict]) -> pd.DataFrame:
    """Long rows -> wide frame, with the duplicate check that makes it safe."""
    raw = pd.DataFrame(rows)

    missing = {"period", "type", "value"} - set(raw.columns)
    if missing:
        raise RuntimeError(f"EIA response is missing columns: {sorted(missing)}")

    raw = raw[raw["type"].isin(TYPE_MAP)].copy()

    # Periods look like "2024-01-01T00". We parse as UTC -- and then VERIFY
    # that assumption in summarize(), rather than trusting it.
    raw["period"] = pd.to_datetime(raw["period"], format="%Y-%m-%dT%H", utc=True)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")

    # THE CHECK. If a (period, type) pair repeats, pivot would silently
    # aggregate it and we would never know.
    dupes = raw.duplicated(subset=["period", "type"], keep=False)
    if dupes.any():
        sample = raw.loc[dupes, ["period", "type"]].head(5).to_dict("records")
        raise RuntimeError(
            f"{int(dupes.sum())} duplicated (period, type) rows. Example: {sample}. "
            "Refusing to pivot -- the result would silently average them."
        )

    wide = raw.pivot(index="period", columns="type", values="value")
    wide = wide.rename(columns=TYPE_MAP)
    wide.columns.name = None
    wide.index.name = "period_utc"

    for col in TYPE_MAP.values():
        if col not in wide.columns:
            wide[col] = pd.NA

    return wide[list(TYPE_MAP.values())].sort_index()


def load_demand(
    region: Region | str = "ercot",
    start: str = "2024-01-01T00",
    end: str = "2024-01-31T23",
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Hourly demand and day-ahead forecast, UTC-indexed, cached."""
    if isinstance(region, str):
        region = get_region(region)

    key = f"demand_{region.key}_{start}_{end}"
    return cache.cached_frame(
        key,
        lambda: _to_frame(fetch_all(region, start, end, force=force)),
        force=force,
    )


# --------------------------------------------------------------------------
# Verification -- run this and READ it. Do not skip to modelling.
# --------------------------------------------------------------------------

def _timezone_looks_right(local: pd.DataFrame) -> tuple[bool, list[str]]:
    """Judge whether our timezone handling is correct, from the shape of the day.

    A first version of this checked that demand peaks in the late afternoon.
    That was wrong: it is a SUMMER invariant. ERCOT peaks in the morning in
    winter, because the driver is electric heating rather than air
    conditioning -- the January 2024 record was set in the 7-8 a.m. hour.

    The genuinely season-independent invariant is the TROUGH. Demand bottoms
    out in the small hours in every season, in every grid, because people are
    asleep. So that is the primary check. The peak is reported with seasonal
    context, and only flagged if it lands somewhere no load curve ever goes.
    """
    by_hour = local.groupby(local.index.hour)["demand"].mean()
    peak, trough = int(by_hour.idxmax()), int(by_hour.idxmin())

    notes = [f"average demand peaks at {peak:02d}:00 local, troughs at {trough:02d}:00"]
    ok = True

    # Primary check: season-independent.
    if 1 <= trough <= 6:
        notes.append("  trough is in the small hours -- timezone handling looks right")
    else:
        ok = False
        notes.append("  TROUGH IS WRONG. Demand always bottoms out overnight. Check the timezone.")

    # Secondary: which regime are we in?
    if 5 <= peak <= 10:
        notes.append("  morning peak -- winter heating pattern")
    elif 14 <= peak <= 21:
        notes.append("  afternoon/evening peak -- summer cooling pattern")
    else:
        ok = False
        notes.append("  PEAK IS WRONG. Expected a morning (heating) or afternoon (cooling) peak.")

    return ok, notes


def summarize(df: pd.DataFrame, region: Region | str = "ercot") -> None:
    """Print the checks that would catch a plumbing bug.

    Read the output. A row count cannot catch a timezone error; the shape of
    the daily curve can.
    """
    if isinstance(region, str):
        region = get_region(region)

    print(f"{region.name}")
    print(f"  rows            {len(df):,}")
    print(f"  from            {df.index.min()}")
    print(f"  to              {df.index.max()}")

    expected = int((df.index.max() - df.index.min()).total_seconds() // 3600) + 1
    print(f"  hours expected  {expected:,}")
    print(f"  hours missing   {expected - len(df):,}")

    for col in df.columns:
        nulls = int(df[col].isna().sum())
        print(f"  {col:<20} nulls={nulls:<6} min={df[col].min():,.0f}  max={df[col].max():,.0f}")

    local = df.tz_convert(region.local_timezone)

    print()
    _, notes = _timezone_looks_right(local)
    for line in notes:
        print(f"  {line}")

    # The single most checkable fact in the whole pull: when was the maximum?
    # Grid operators publish their peaks, so you can verify this against the
    # real world rather than against your own code.
    peak_at = local["demand"].idxmax()
    print(
        f"\n  maximum demand {local['demand'].max():,.0f} MW "
        f"at {peak_at:%Y-%m-%d %H:%M} {region.local_timezone.split('/')[-1]}"
    )
    print("  -> look this up in the operator's published peak records")