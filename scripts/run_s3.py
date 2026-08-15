"""Run S3 -- align demand and weather onto one hourly table.

    py scripts/run_s3.py

Starts with 2023 only. Once the alignment checks pass, widen to the full
window (April 2021 onward) by editing START below.

Read the RAW audit before the ALIGNED one. The difference between them is
every decision this stage made on your behalf.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from honest_forecast import align  # noqa: E402
from honest_forecast.config import ensure_dirs  # noqa: E402

ensure_dirs()

REGION = "ercot"
START = "2023-01-01"
END = "2023-12-31"

print(f"Building aligned table for {REGION.upper()}, {START} -> {END}")
print("(first run downloads ~12 months from two APIs; later runs read the cache)")

df = align.build(REGION, START, END)

print("\nALIGNMENT CHECK -- physics, not row counts\n")
align.check_alignment(df, REGION)

print("\nFirst 3 rows:")
with_cols = ["demand", "day_ahead_forecast", "temp_actual", "temp_fcst_day1"]
print(df[with_cols].head(3).round(2))

print("\nHottest and coldest hours in the year:")
print(df.loc[[df["temp_actual"].idxmax(), df["temp_actual"].idxmin()], with_cols].round(2))

print(f"\nPeak demand hour: {df['demand'].idxmax()}  ({df['demand'].max():,.0f} MW)")