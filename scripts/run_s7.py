"""Run S7 -- prediction intervals, and whether they are honest.

    py scripts/run_s7.py

Two sets of intervals, both deployed on forecast-weather predictions:

    NAIVE   calibrated on residuals computed with OBSERVED weather
    HONEST  calibrated on residuals computed with FORECAST weather

If the naive intervals under-cover, then the conventional approach overstates
not just accuracy but confidence -- the error that persuades you to skip the
review.
"""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

from honest_forecast import align, features, intervals  # noqa: E402
from honest_forecast.config import ensure_dirs  # noqa: E402

ensure_dirs()

REGION = "ercot"
START = "2021-04-01"
END = "2026-07-31"
TEST_START = "2026-01-01"
LEVEL = 0.90

print("Loading and building features ...")
aligned = align.build(REGION, START, END, verbose=False)
feats = features.drop_warmup(features.build(aligned, REGION, verbose=False), verbose=False)
print(f"  {len(feats):,} hours\n")

out = intervals.run(feats, TEST_START, LEVEL)


def show(name: str) -> None:
    print(f"\n\n{name.upper()} INTERVALS -- coverage and sharpness by temperature\n")
    rep = out["reports"][name]
    fmt = {
        "coverage": "{:.1%}".format,
        "gap": "{:+.1%}".format,
        "nominal": "{:.0%}".format,
        "mean_width_mw": "{:,.0f}".format,
        "width_pct_of_load": "{:.1f}".format,
    }
    print(rep.to_string(index=False, formatters=fmt))


show("naive")
show("honest")

print("\n\n  A band covering well below the nominal level is the dangerous case:")
print("  the interval claims a confidence it does not have, in the conditions")
print("  where being wrong costs the most.\n")

# The hours that escaped even the honest interval.
test = out["test"]
iv = out["intervals"]["honest"]
missed = ~iv.covers(test["demand"])

print(f"\nHours outside the honest {LEVEL:.0%} interval: {int(missed.sum()):,} of {len(test):,}")
if missed.any():
    worst = (test.loc[missed, "demand"] - out["point"][missed]).abs().nlargest(8).index
    table = pd.DataFrame(
        {
            "demand": test.loc[worst, "demand"],
            "point": out["point"][worst],
            "lower": iv.lower[worst],
            "upper": iv.upper[worst],
            "temp_C": test.loc[worst, "act_h_base"],
        }
    )
    print("\nThe 8 largest escapes:\n")
    print(table.round(0).to_string())