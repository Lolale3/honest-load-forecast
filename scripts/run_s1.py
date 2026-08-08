"""Run S1 against the real API and print the checks.

    py scripts/run_s1.py

Reads your key from .env. First run hits EIA; every run after reads from
data/raw and costs nothing.

READ THE OUTPUT. Especially the peak-hour line at the bottom -- that is the
check that catches a timezone bug, and no row count ever would.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from honest_forecast import eia  # noqa: E402
from honest_forecast.config import ensure_dirs  # noqa: E402

ensure_dirs()

REGION = "ercot"
START = "2024-01-01T00"
END = "2024-01-31T23"

print(f"Pulling {REGION.upper()} {START} -> {END} ...\n")

df = eia.load_demand(REGION, START, END)

eia.summarize(df, REGION)

print("\nFirst 5 hours:")
print(df.head())

print("\nThe 5 hours with the largest gap between forecast and actual:")
gap = (df["day_ahead_forecast"] - df["demand"]).abs().nlargest(5)
print(df.loc[gap.index].assign(error=df["day_ahead_forecast"] - df["demand"]))