"""Generate the figures for the write-up.

    py scripts/make_figures.py

Writes PNGs to figures/. Everything reads from cache, so this is fast once
S3 has run.

Four figures, one per finding:

    1  the load-temperature curve          why the mechanism exists
    2  degradation by temperature band     where the cost lands
    3  interval coverage, naive vs honest  overstated confidence
    4  risk-coverage and the ablation      what triage does and does not buy
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from honest_forecast import align, evaluate, features, intervals, triage  # noqa: E402
from honest_forecast.config import ensure_dirs  # noqa: E402

ensure_dirs()

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

INK = "#1a1a1a"
ACCENT = "#c1440e"
COOL = "#2c5f8a"
GREY = "#9a9a9a"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

REGION, START, END, TEST_START = "ercot", "2021-04-01", "2026-07-31", "2026-01-01"

print("Loading ...")
aligned = align.build(REGION, START, END, verbose=False)
feats = features.drop_warmup(features.build(aligned, REGION, verbose=False), verbose=False)

s6 = evaluate.run(feats, TEST_START, verbose=False)
test = s6["test"]
pred_a = s6["results"]["A_observed_observed"]["pred"]
pred_b = s6["results"]["B_observed_forecast"]["pred"]

s7 = intervals.run(feats, TEST_START, level=0.90, verbose=False)


# --------------------------------------------------------------------------
# 1  The load-temperature curve
# --------------------------------------------------------------------------

print("figure 1 ...")
fig, ax = plt.subplots(figsize=(7, 4.2))

sample = feats.sample(min(6000, len(feats)), random_state=0)
ax.scatter(sample["act_h_base"], sample["demand"] / 1000, s=2, alpha=0.10, color=COOL,
           edgecolors="none")

bands = pd.cut(feats["act_h_base"], bins=np.arange(-10, 46, 2.5))
means = feats.groupby(bands, observed=True)["demand"].mean() / 1000
centres = [b.mid for b in means.index]
ax.plot(centres, means.values, color=ACCENT, linewidth=2.2, zorder=5)

ax.axvspan(15, 20, color=GREY, alpha=0.15, zorder=0)
ax.annotate("flat: 47 MW per °C\nof forecast error", xy=(17.5, 44), xytext=(0, 34),
            fontsize=8, color=INK,
            arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
ax.annotate("steep: 569 MW per °C", xy=(33, 70), xytext=(22, 84),
            fontsize=8, color=ACCENT,
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
ax.annotate("steep: 767 MW per °C", xy=(-4, 66), xytext=(-9, 82),
            fontsize=8, color=ACCENT,
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))

ax.set_xlabel("Temperature (°C), population-weighted across four metros")
ax.set_ylabel("ERCOT demand (GW)")
ax.set_title("The same weather error costs ten times more at the extremes",
             loc="left", fontsize=10.5, weight="bold", pad=12)
fig.text(0.01, -0.02, "ERCOT hourly demand, Apr 2021 – Jul 2026 (46,247 hours)",
         fontsize=7.5, color=GREY)
fig.tight_layout()
fig.savefig(FIG_DIR / "1_load_temperature_curve.png", bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------------------
# 2  Degradation by band
# --------------------------------------------------------------------------

print("figure 2 ...")
mech = evaluate.weather_error_vs_demand_error(test, pred_b, pred_a)

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(mech))
colours = [ACCENT if v > 200 else COOL for v in mech["mw_per_deg_of_weather_err"]]
ax.bar(x, mech["mw_per_deg_of_weather_err"], color=colours, width=0.62)

for i, v in enumerate(mech["mw_per_deg_of_weather_err"]):
    ax.text(i, v + 20, f"{v:,.0f}", ha="center", fontsize=8, color=INK)

ax.set_xticks(x)
ax.set_xticklabels([b.replace("(", "").replace("]", "").replace(", ", " to ") + " °C"
                    for b in mech["band"]], fontsize=8)
ax.set_ylabel("MW of extra demand error\nper °C of weather error")
ax.set_title("The cost of not knowing the weather, by temperature band",
             loc="left", fontsize=10.5, weight="bold", pad=12)
fig.text(0.01, -0.03,
         "Same model, same hours. Difference is observed vs archived day-ahead weather. "
         "Test period Jan–Jul 2026.",
         fontsize=7.5, color=GREY)
fig.tight_layout()
fig.savefig(FIG_DIR / "2_degradation_by_band.png", bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------------------
# 3  Interval coverage
# --------------------------------------------------------------------------

print("figure 3 ...")
naive = s7["reports"]["naive"].iloc[1:]
honest = s7["reports"]["honest"].iloc[1:]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(naive))
w = 0.36

ax.bar(x - w / 2, naive["coverage"] * 100, w, label="calibrated on observed weather",
       color=GREY)
ax.bar(x + w / 2, honest["coverage"] * 100, w, label="calibrated on forecast weather",
       color=COOL)
ax.axhline(90, color=ACCENT, linestyle="--", linewidth=1.3, zorder=5)
ax.text(len(x) - 0.4, 91, "nominal 90%", fontsize=8, color=ACCENT, ha="right")

ax.set_xticks(x)
ax.set_xticklabels([b.replace("(", "").replace("]", "").replace(", ", " to ")
                    for b in naive["band"]], fontsize=8)
ax.set_ylabel("Actual coverage (%)")
ax.set_ylim(50, 105)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("A 90% interval that covers 68% when it is freezing",
             loc="left", fontsize=10.5, weight="bold", pad=12)
fig.text(0.01, -0.03,
         "Both interval sets deployed on forecast-weather predictions. "
         "Honest calibration fixes the mild bands and not the cold ones.",
         fontsize=7.5, color=GREY)
fig.tight_layout()
fig.savefig(FIG_DIR / "3_interval_coverage.png", bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------------------
# 4  Risk-coverage and ablation
# --------------------------------------------------------------------------

print("figure 4 ...")
test_s8 = test.assign(
    temp_fcst_day1=aligned.loc[test.index, "temp_fcst_day1"],
    temp_fcst_day2=aligned.loc[test.index, "temp_fcst_day2"],
    temp_fcst_day3=aligned.loc[test.index, "temp_fcst_day3"],
)
train = feats.loc[feats.index < pd.Timestamp(TEST_START, tz="UTC")]
signals = triage.build_signals(train, test_s8, s7["intervals"]["honest"].width())
score = triage.combine(signals)

fractions = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
curve = triage.risk_coverage(score, test["demand"], s7["point"], fractions)

rng = np.random.default_rng(0)
rand_curve = triage.risk_coverage(
    pd.Series(rng.random(len(test)), index=test.index),
    test["demand"], s7["point"], fractions,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))

ax1.plot(curve["reviewed_pct"], curve["auto_mape"], "o-", color=COOL,
         linewidth=1.8, markersize=4, label="uncertainty score")
ax1.plot(rand_curve["reviewed_pct"], rand_curve["auto_mape"], "s--", color=GREY,
         linewidth=1.3, markersize=3.5, label="random selection")
ax1.set_xlabel("Hours sent for human review (%)")
ax1.set_ylabel("MAPE on hours shipped unattended")
ax1.legend(frameon=False, fontsize=8)
ax1.set_title("Triage buys little", loc="left", fontsize=10, weight="bold")

abl = triage.ablation(signals, test["demand"], s7["point"], review_fraction=0.05)
drops = abl[abl["variant"].str.startswith("without")].copy()
drops["signal"] = drops["variant"].str.replace("without ", "")
colours2 = [ACCENT if v > 0 else GREY for v in drops["vs_full_mape"]]

ax2.barh(np.arange(len(drops)), drops["vs_full_mape"], color=colours2, height=0.6)
ax2.axvline(0, color=INK, linewidth=0.8)
ax2.set_yticks(np.arange(len(drops)))
ax2.set_yticklabels(drops["signal"], fontsize=8)
ax2.set_xlabel("Change in MAPE when this signal is removed\n(negative = removing it HELPS)")
ax2.set_title("Three of four signals were harmful", loc="left", fontsize=10, weight="bold")

fig.tight_layout()
fig.savefig(FIG_DIR / "4_triage_and_ablation.png", bbox_inches="tight")
plt.close(fig)

print(f"\nWrote 4 figures to {FIG_DIR}")
for p in sorted(FIG_DIR.glob("*.png")):
    print(f"  {p.name}")