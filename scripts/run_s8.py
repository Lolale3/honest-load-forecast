"""Run S8 -- selective forecasting and the review queue.

    py scripts/run_s8.py

The output is not a better forecast. It is a triage rule: which hours should
a human look at before anyone commits capacity, and what does reviewing them
actually buy.

Read the ablation. A combined score whose parts have not been tested
individually is a score nobody understands.
"""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

from honest_forecast import align, features, intervals, triage  # noqa: E402
from honest_forecast.config import ensure_dirs  # noqa: E402

ensure_dirs()

REGION = "ercot"
START = "2021-04-01"
END = "2026-07-31"
TEST_START = "2026-01-01"

print("Loading, building features, fitting, calibrating ...")
aligned = align.build(REGION, START, END, verbose=False)
feats = features.drop_warmup(features.build(aligned, REGION, verbose=False), verbose=False)

s7 = intervals.run(feats, TEST_START, level=0.90, verbose=False)

test = s7["test"]
point = s7["point"]
iv = s7["intervals"]["honest"]

# The aligned frame carries the day-2 and day-3 forecasts; attach them so the
# lead-disagreement signal has something to work with.
for col in ("temp_fcst_day2", "temp_fcst_day3"):
    if col in aligned.columns:
        test = test.assign(**{col: aligned.loc[test.index, col]})
test = test.assign(temp_fcst_day1=aligned.loc[test.index, "temp_fcst_day1"])

train = feats.loc[feats.index < pd.Timestamp(TEST_START, tz="UTC")]

print(f"  {len(test):,} test hours\n")

signals = triage.build_signals(train, test, iv.width())
print("Signal availability:")
for col in signals.columns:
    n = int(signals[col].notna().sum())
    print(f"  {col:<20} {n:,} / {len(signals):,} hours")

score = triage.combine(signals)

print("\n\nRISK-COVERAGE -- error on hours shipped without review\n")
curve = triage.risk_coverage(score, test["demand"], point)
print(curve.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

print("\n\nABLATION -- which signals actually contribute?")
print("(reviewing the top 5% of hours; positive vs_full_mape means worse)\n")
abl = triage.ablation(signals, test["demand"], point, review_fraction=0.05)
print(abl.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

print("\n\nWHAT AN OPERATOR WOULD ACTUALLY STAFF\n")
budget = triage.review_budget(score, test["demand"], point)
print(budget.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

print("\n\nTHE 10 HOURS THE SYSTEM FLAGS MOST URGENTLY\n")
top = score.nlargest(10).index
flagged = pd.DataFrame(
    {
        "score": score[top],
        "temp_C": test.loc[top, "fc1_h_base"],
        "demand": test.loc[top, "demand"],
        "point": point[top],
        "abs_err_mw": (point[top] - test.loc[top, "demand"]).abs(),
        "interval_mw": iv.width()[top],
    }
)
print(flagged.round(1).to_string())

print(f"\n  Mean absolute error on these 10 flagged hours: "
      f"{flagged['abs_err_mw'].mean():,.0f} MW")
print(f"  Mean absolute error across all test hours:      "
      f"{(point - test['demand']).abs().mean():,.0f} MW")