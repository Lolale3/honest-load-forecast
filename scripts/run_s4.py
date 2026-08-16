"""Run S4 -- turn the aligned table into model-ready features.

    py scripts/run_s4.py

Reads from cache, so this is fast after S3 has run once.

The output to study is the temperature-response histogram. That shape --
high at both ends, low in the middle -- is why the model needs a cubic, and
it is the mechanism behind the whole project: on the steep parts of that
curve, a small weather error becomes a large demand error.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from honest_forecast import align, features  # noqa: E402
from honest_forecast.config import ensure_dirs  # noqa: E402

ensure_dirs()

REGION = "ercot"
START = "2021-04-01"
END = "2026-07-31"

print("Loading aligned data ...")
aligned = align.build(REGION, START, END, verbose=False)
print(f"  {len(aligned):,} hours\n")

print("Building features ...")
feats = features.build(aligned, REGION)
clean = features.drop_warmup(feats)

features.summarize(clean)

print("\nFeature columns for each source:")
act = features.feature_columns(clean, "act")
fc1 = features.feature_columns(clean, "fc1")
print(f"  act: {len(act)} features")
print(f"  fc1: {len(fc1)} features")

print("\nA few temperature features at the hottest hour:")
hottest = clean["act_t1"].idxmax()
show = ["act_t1", "act_t2", "act_t3", "act_t_lag24", "act_t_mean24", "fc1_t1", "demand"]
print(clean.loc[[hottest], show].T.round(1))