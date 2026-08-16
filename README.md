# Honest load forecasting

Most electricity demand forecasts are backtested against **the weather that
actually happened**. In production you never have that — you have a forecast
that's wrong by a degree or two. And because demand responds non-linearly to
temperature, a small weather miss becomes a large demand miss precisely on the
extreme days that matter most.

So the published accuracy number is measured on information the model will
never have, and it is optimistic exactly where you needed it not to be.

This project builds a load forecast for ERCOT and evaluates it twice: once the
usual way, and once on the weather forecast that was genuinely available at
decision time. Then it puts a calibrated uncertainty layer on top, so the hours
the model is least sure about can be routed to a human before anyone commits
capital or capacity.

## Data

| Source | What it gives us |
|---|---|
| EIA-930 | Hourly demand **and** the operator's own day-ahead forecast, by balancing authority |
| Open-Meteo Historical (ERA5) | The weather that actually happened |
| Open-Meteo Previous Runs (GFS) | The weather that was *forecast*, at fixed lead times of 1–3 days |

That third source is the whole project.

The EIA day-ahead forecast column gives a free external benchmark — we are not
grading our own homework. Over the 2026 test period ERCOT's own day-ahead
forecast achieves **1.92% MAPE**.

## Window

**April 2021 – July 2026**, 46,247 aligned hours.

The Open-Meteo docs give conflicting archive start dates (March 2021 on one
page, January 2024 on another). Neither is right for this endpoint. We probed
it empirically (`scripts/probe_archive.py`, `probe_archive2.py`) and found
coverage beginning **April 2021**, with a **20-day outage from 2023-12-30 to
2024-01-18** that no documentation mentions. That window is excluded by name.

Sampling found the outage by luck — it happened to straddle the 15th of a
month. So S3 runs a full null audit over every hour rather than spot checks.
Sampling proves presence; it can never prove absence.

## Known limitations

- **Weather is sampled at four metros** (Dallas, Houston, San Antonio, Austin)
  with fixed population-share weights. That is an assumption, not a
  measurement. ERCOT publishes load by eight weather zones; a zonal model
  would be more accurate and is a v2 candidate.
- **The 24-hour horizon is an approximation.** ERCOT's day-ahead market closes
  late morning for the whole of the following day, so real lead times run
  roughly 14–38 hours. We use a flat 24.
- **Baseline load moved ~11 GW during the training window.** At 25 °C, mean
  demand rose from 46,491 MW (2021) to 57,682 MW (2026) — monotonically, every
  year. This is large flexible load: data centres, crypto mining, industrial.
  Peak demand actually *fell* in 2025, which hides the growth; controlling for
  temperature reveals it. We track the level with a chronological trend and a
  weekly rolling mean of load, but a linear trend cannot follow a step change
  well. Residual level error is part of what S7 measures.
- **Sub-zero hours are 0.9% of the data** and concentrated in two multi-day
  emergencies (Elliott 2022, Kingston 2025). Uri (Feb 2021) predates the
  forecast archive.

## What we found before modelling

The temperature–demand curve is a lopsided U: flat between 15–20 °C, steep on
both arms. Roughly 900 MW/°C in the comfort zone, ~1,800 MW/°C above 35 °C.
Mean day-ahead weather error is 1.19 °C, maximum 7.58 °C. The same weather
error therefore costs several times more on the steep arms — and a backtest
fed observed weather has *zero* weather error everywhere, so it never sees any
of this.

Cooling and heating are not symmetric because air conditioning is entirely
electric while much Texas heating is gas. ERCOT is summer-peaking.

## Model

Tao's Vanilla Benchmark (Hong 2010) plus the recency effect
(Wang, Liu & Hong 2016), with one departure: a **piecewise-linear hinge basis**
in temperature rather than Hong's cubic. The cubic extrapolated badly into the
sparse cold tail; hinges continue the last fitted slope instead.

Ordinary least squares, split by time — train through 2025, test on 2026.
Never shuffled: a random split would let the model see the future.

Recency improves on vanilla by **26.2%**, consistent with the 18–21% Hong
reports. Test MAPE **2.47%** evaluated on observed weather.

## An honest negative result

The model's worst hours are 10–13 GW out. Two fixes aimed at them — longer
temperature memory with freeze-duration counters, and a load-level anchor —
moved the error almost not at all.

The diagnostic explains why: **ERCOT's own operational forecast missed the same
hours, in the same direction, by a comparable margin** (up to 9,616 MW on
2026-01-26). On that day demand *fell* through the morning while temperature
stayed near −6 °C. Demand dropping during a freeze is not a weather response;
it is behaviour — a conservation appeal or large-load curtailment.

No feature engineering fixes that, which is why two reasonable attempts
failed. It is the argument for the uncertainty layer rather than a defect to
be tuned away: these are precisely the hours a human should see before anyone
commits capacity.

## Stages

- [x] **S0** — repo, config, cache, tests
- [x] **S1** — EIA-930 ingestion, pagination guard, timezone verification
- [x] **S2** — weather ingestion, pinned GFS, renormalised city weights
- [x] **S3** — hourly spine, full null audit, per-gap fill rules
- [x] **S4** — hinge basis, interactions, recency, dual temperature builds
- [x] **S5** — vanilla and recency baselines, time-ordered split
- [ ] **S6** — dual evaluation against the archived forecast
- [ ] **S7** — quantile models and interval coverage
- [ ] **S8** — selective forecasting / review triage
- [ ] **S9** — second region (ISO-NE), stretch
- [ ] **S10** — write-up

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then paste your free key from eia.gov/opendata
pytest
```

Tests run entirely offline — no network, no API key required.

```bash
py scripts/run_s3.py      # build the aligned table
py scripts/run_s5.py      # fit and evaluate
py scripts/diagnose.py    # inspect the failing hours
```