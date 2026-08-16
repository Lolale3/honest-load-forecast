"""Run S5 -- fit the baseline models and evaluate the standard way.

    py scripts/run_s5.py

This produces a GOOD number, using observed weather. That is the convention
in the field, and it is the number S6 will knock down.

Read the temperature breakdown at the end. A single average MAPE hides
whether the model fails uniformly or fails hardest where the stakes are.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from honest_forecast import align, features, models  # noqa: E402
from honest_forecast.config import ensure_dirs  # noqa: E402

ensure_dirs()

REGION = "ercot"
START = "2021-04-01"
END = "2026-07-31"
TEST_START = "2026-01-01"

print("Loading and building features ...")
aligned = align.build(REGION, START, END, verbose=False)
feats = features.drop_warmup(features.build(aligned, REGION, verbose=False), verbose=False)
print(f"  {len(feats):,} hours, {len(feats.columns)} columns\n")

print("Fitting on OBSERVED weather (the standard evaluation)\n")
out = models.run(feats, TEST_START, source="act")

test = out["test"]
pred = out["results"]["recency"]["pred"]

print("\n\nRecency model error by temperature band:\n")
breakdown = models.score_by_temperature(test["demand"], pred, test["act_h_base"])
print(breakdown.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

print("\nWorst 5 hours for the recency model:")
err = (pred - test["demand"]).abs().nlargest(5)
worst = test.loc[err.index, ["demand", "act_h_base", "fc1_h_base"]].copy()
worst["predicted"] = pred[err.index]
worst["error_mw"] = pred[err.index] - test.loc[err.index, "demand"]
print(worst.round(1).to_string())