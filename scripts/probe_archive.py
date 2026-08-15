"""Find where the archived-forecast coverage ACTUALLY starts.

    py scripts/probe_archive.py

The documentation disagrees with itself. The Previous Runs page says GFS 2 m
temperature reaches back to March 2021; Open-Meteo's features page says the
Previous Runs API begins January 2024. Our January 2024 pull came back 60%
empty, with each extra lead day losing exactly 24 more hours at the start --
the signature of running into an archive boundary.

So stop reading and measure. This samples one day per month per model and
reports whether day-1 forecasts exist. Cheap, and it settles the question.

The lesson generalises: when docs and data disagree, the data wins.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from honest_forecast import weather  # noqa: E402
from honest_forecast.config import ensure_dirs, get_region  # noqa: E402

ensure_dirs()

MODELS = ["gfs_seamless", "best_match"]

# One probe date per month. Mid-month avoids month-boundary oddities.
PROBE_DATES = [
    f"{year}-{month:02d}-15"
    for year in (2023, 2024, 2025, 2026)
    for month in range(1, 13)
]
PROBE_DATES = [d for d in PROBE_DATES if d <= "2026-07-15"]

city = get_region("ercot").weather_points[0]
original_model = weather.MODEL

print(f"Probing {city.name} for day-1 forecast coverage\n")
print(f"{'date':<14}" + "".join(f"{m:<18}" for m in MODELS))
print("-" * (14 + 18 * len(MODELS)))

results = {}

for date in PROBE_DATES:
    row = []
    for model in MODELS:
        weather.MODEL = model
        try:
            df = weather.fetch_forecast(city, date, date, lead_days=(1,))
            n = int(df["temp_fcst_day1"].notna().sum())
            row.append(f"{n}/24 hours" if n else "empty")
            results[(date, model)] = n
        except RuntimeError as exc:
            row.append(f"error: {str(exc)[:20]}")
            results[(date, model)] = 0
    print(f"{date:<14}" + "".join(f"{cell:<18}" for cell in row))

weather.MODEL = original_model

print("\nEarliest date with any day-1 coverage:")
for model in MODELS:
    covered = [d for d in PROBE_DATES if results.get((d, model), 0) > 0]
    print(f"  {model:<18} {covered[0] if covered else 'none found'}")

print(
    "\nUse the earliest fully-covered month as the start of the analysis window,"
    "\nand write the boundary into the README as a stated limitation."
)