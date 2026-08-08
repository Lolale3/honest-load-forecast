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
| Open-Meteo Previous Runs | The weather that was *forecast*, at fixed lead times |

That third one is the whole project. It exists back to January 2024, which
bounds the honest-evaluation window.

The EIA day-ahead forecast column gives us a free external benchmark — we are
not grading our own homework.

## Known limitations

- Archived forecasts begin **January 2024**, so Winter Storm Uri (Feb 2021) is
  out of scope. The extreme-weather narrative rests on the 2024–25 winter
  events and the July 2026 record summer peak.
- Weather is sampled at four metros with fixed population-share weights. That
  is an assumption, not a measurement.

## Stages

- [x] **S0** — repo, config, cache, tests
- [ ] **S1** — EIA load + day-ahead forecast ingestion
- [ ] **S2** — weather ingestion (ERA5 and archived forecasts)
- [ ] **S3** — alignment to one UTC hourly table
- [ ] **S4** — features
- [ ] **S5** — baseline model, naive evaluation
- [ ] **S6** — dual evaluation vs the EIA benchmark
- [ ] **S7** — quantile models and coverage
- [ ] **S8** — selective forecasting / triage
- [ ] **S9** — second region (ISO-NE), stretch
- [ ] **S10** — the write-up

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then paste your free key from eia.gov/opendata
pytest
```

Tests run entirely offline. No network, no API key required.
