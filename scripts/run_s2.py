"""Run S2 against the real APIs and print the checks.

    py scripts/run_s2.py

No API key needed -- Open-Meteo is open.

There is one assumption in S2 I could not verify from the docs: whether the
Previous Runs endpoint accepts start_date/end_date for a window 18 months in
the past, or only the recent `past_days` window. So this script PROBES with a
single city and two days before pulling the full month. If the probe fails,
you get a clear error instead of four confusing ones.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from honest_forecast import weather  # noqa: E402
from honest_forecast.config import ensure_dirs, get_region  # noqa: E402

ensure_dirs()

REGION = "ercot"
START = "2024-01-01"
END = "2024-01-31"

region = get_region(REGION)
probe_city = region.weather_points[0]

print(f"Probe: {probe_city.name}, 2 days, archived forecast ...")
try:
    probe = weather.fetch_forecast(probe_city, "2024-01-15", "2024-01-16")
    print(probe.head(3))
    print("Probe OK -- start_date/end_date works on the Previous Runs endpoint.\n")
except RuntimeError as exc:
    print(f"\nPROBE FAILED: {exc}")
    print("\nThe date-range assumption is wrong. Paste this error and we'll adapt.")
    sys.exit(1)

print(f"Pulling {REGION.upper()} weather {START} -> {END}, {len(region.weather_points)} cities ...\n")

df = weather.load_weather(REGION, START, END)

weather.summarize(df, REGION)

print("\nFirst 5 hours:")
print(df.head())

print("\nThe 5 hours where the day-ahead forecast was most wrong:")
err = (df["temp_fcst_day1"] - df["temp_actual"]).abs().nlargest(5)
print(df.loc[err.index].assign(error_c=df["temp_fcst_day1"] - df["temp_actual"]).round(2))