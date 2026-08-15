"""Pin down the archive: how far back does it go, and where exactly is the gap?

    py scripts/probe_archive2.py

The first probe found two surprises:

  * All of 2023 is covered, though the docs say the archive starts January 2024.
  * 2024-01-15 is empty while its neighbours are fine -- a HOLE, not a boundary.

This script does two things the first could not:

  PART 1  extends the monthly sweep back towards March 2021, to find the real
          start of coverage.
  PART 2  probes daily across December 2023 and January 2024 to find the exact
          edges of the gap.

Note the methodological lesson from probe 1: sampling the 15th of each month
found this gap only by luck. Sampling proves presence, never absence. A full
null audit at S3 is still required.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from honest_forecast import weather  # noqa: E402
from honest_forecast.config import ensure_dirs, get_region  # noqa: E402

ensure_dirs()

city = get_region("ercot").weather_points[0]


def day1_hours(date: str) -> int:
    """How many of the 24 hours on `date` have a day-1 forecast? -1 on error."""
    try:
        df = weather.fetch_forecast(city, date, date, lead_days=(1,))
        return int(df["temp_fcst_day1"].notna().sum())
    except RuntimeError:
        return -1


# --------------------------------------------------------------------------
# PART 1 -- how far back?
# --------------------------------------------------------------------------

print("PART 1: monthly sweep, 2021-2023\n")

early = [
    f"{year}-{month:02d}-15"
    for year in (2021, 2022)
    for month in range(1, 13)
] + [f"2023-{month:02d}-15" for month in (1,)]
early = [d for d in early if d >= "2021-01-15"]

earliest_covered = None
for date in early:
    n = day1_hours(date)
    status = f"{n}/24" if n >= 0 else "error"
    flag = ""
    if n > 0 and earliest_covered is None:
        earliest_covered = date
        flag = "  <-- earliest coverage found"
    print(f"  {date}   {status}{flag}")

print(f"\n  earliest date with day-1 coverage: {earliest_covered or 'none in 2021-2022'}")


# --------------------------------------------------------------------------
# PART 2 -- where exactly is the gap?
# --------------------------------------------------------------------------

print("\n\nPART 2: daily sweep across the suspected gap\n")

window = pd.date_range("2023-12-01", "2024-02-01", freq="D").strftime("%Y-%m-%d")

coverage = {}
for date in window:
    coverage[date] = day1_hours(date)

# Print compactly: one line per week.
for i in range(0, len(window), 7):
    chunk = window[i:i + 7]
    line = "  ".join(f"{d[5:]}:{coverage[d]:>3}" for d in chunk)
    print(f"  {line}")

empty = [d for d, n in coverage.items() if n == 0]
if empty:
    print(f"\n  gap runs {empty[0]} .. {empty[-1]}  ({len(empty)} days)")
    print("  (day-1 forecasts absent; the underlying model runs are missing)")
else:
    print("\n  no fully-empty days found in this window")

print(
    "\nRecord the gap in the README, and drop these dates from the analysis"
    "\nwindow rather than letting them become silent NaNs in the model."
)