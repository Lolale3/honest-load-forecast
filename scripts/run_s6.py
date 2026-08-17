"""Run S6 -- the dual evaluation.

    py scripts/run_s6.py

Three configurations of the same model:

    A   train observed, test observed    the standard result
    B   train observed, test forecast    that model, fed reality
    C   train forecast, test forecast    trained honestly

A is what gets published. B is what would actually have happened. C is what a
careful practitioner would do instead.
"""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

from honest_forecast import align, evaluate, features  # noqa: E402
from honest_forecast.config import ensure_dirs  # noqa: E402

ensure_dirs()

REGION = "ercot"
START = "2021-04-01"
END = "2026-07-31"
TEST_START = "2026-01-01"

print("Loading and building features ...")
aligned = align.build(REGION, START, END, verbose=False)
feats = features.drop_warmup(features.build(aligned, REGION, verbose=False), verbose=False)
print(f"  {len(feats):,} hours\n")

out = evaluate.run(feats, TEST_START)

test = out["test"]
pred_a = out["results"]["A_observed_observed"]["pred"]
pred_b = out["results"]["B_observed_forecast"]["pred"]
pred_d = out["results"]["D_operational"]["pred"]

print("\n\nTHE MECHANISM -- where does the degradation happen?\n")
table = evaluate.weather_error_vs_demand_error(test, pred_b, pred_a)
print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

print("\n  Read the last column: MW of extra demand error per degree of")
print("  weather error. If the curve's steepness is the mechanism, that")
print("  number should be much larger in the extreme bands.\n")

print("\nTHE 10 HOURS WHERE THE WEATHER FORECAST WAS MOST WRONG\n")
print(evaluate.worst_weather_misses(test, pred_b, pred_a, n=10).to_string())

print("\n\nSAME MECHANISM TABLE, BUT FOR THE OPERATIONAL CONFIG D\n")
table_d = evaluate.weather_error_vs_demand_error(test, pred_d, pred_a)
print(table_d.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))