"""Diagnose the two stubborn failures. Look at the data, stop theorising.

    py scripts/diagnose.py

Two fixes aimed at these hours and neither moved them:

  2026-01-26/27  ~0 C, over-predicted by 11-13 GW
  2026-05-22     24.9 C, under-predicted by 11.5 GW

That pattern -- unmoved by both longer weather memory and a load-level anchor
-- says the cause is not what we assumed. So print the surrounding hours and
the comparable hours from history, and let the data say what is going on.
"""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

from honest_forecast import align  # noqa: E402
from honest_forecast.config import ensure_dirs, get_region  # noqa: E402

ensure_dirs()

region = get_region("ercot")
df = align.build("ercot", "2021-04-01", "2026-07-31", verbose=False)
local = df.tz_convert(region.local_timezone)


def window(center: str, hours: int = 36):
    c = pd.Timestamp(center, tz=region.local_timezone)
    lo, hi = c - pd.Timedelta(hours=hours), c + pd.Timedelta(hours=hours)
    return local.loc[lo:hi]


# --------------------------------------------------------------------------
print("=" * 78)
print("CASE 1  -- late January 2026, model over-predicted by 11-13 GW")
print("=" * 78)

jan = window("2026-01-26 18:00", 30)
print("\nActual demand vs ERCOT's own day-ahead forecast, hour by hour:\n")
show = jan[["demand", "day_ahead_forecast", "temp_actual"]].copy()
show["ercot_err"] = show["day_ahead_forecast"] - show["demand"]
print(show.round(1).to_string())

print("\n  KEY QUESTION: did ERCOT also over-forecast these hours?")
print("  If ERCOT was close and we were 13 GW out, the problem is our model.")
print("  If ERCOT ALSO missed high, something unusual happened to demand.")

# --------------------------------------------------------------------------
print("\n\n" + "=" * 78)
print("CASE 2  -- 22 May 2026, model under-predicted by 11.5 GW")
print("=" * 78)

may = window("2026-05-22 18:00", 30)
show = may[["demand", "day_ahead_forecast", "temp_actual"]].copy()
show["ercot_err"] = show["day_ahead_forecast"] - show["demand"]
print("\n")
print(show.round(1).to_string())

# --------------------------------------------------------------------------
print("\n\n" + "=" * 78)
print("CONTEXT -- what does demand normally look like at these temperatures?")
print("=" * 78)

for label, lo, hi in [("near 0 C", -2.0, 2.0), ("near 25 C", 23.0, 27.0)]:
    band = local[(local["temp_actual"] >= lo) & (local["temp_actual"] <= hi)]
    by_year = band.groupby(band.index.year)["demand"].agg(["mean", "max", "count"])
    print(f"\n  {label}  ({len(band):,} hours total)")
    print(by_year.round(0).to_string())

print("\n  -> if the mean at a given temperature is RISING year over year,")
print("     the model is fighting a level shift, not a weather relationship.")

# --------------------------------------------------------------------------
print("\n\n" + "=" * 78)
print("LEVEL -- how much has baseline load moved?")
print("=" * 78)

monthly = local.groupby([local.index.year, local.index.month])["demand"].mean().unstack(0)
print("\nMean demand by month and year (MW):\n")
print(monthly.round(0).to_string())