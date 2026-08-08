"""Configuration for the honest load forecasting project.

Design rule: the region is a PARAMETER, never a hardcoded string. Adding
ISO-NE later should be a config change, not a code change. Everything below
follows from that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"        # untouched API responses, cached as JSON
INTERIM_DIR = DATA_DIR / "interim"  # parsed frames, cached as parquet


# --------------------------------------------------------------------------
# Weather sampling points
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WeatherPoint:
    """One city we pull weather for, and how much of the region's load it stands for.

    A grid's demand is driven by the weather where the PEOPLE are, not by a
    single airport. So we sample several metros and take a weighted average.
    The weights are rough population shares -- an assumption, and one we
    should write down rather than bury.
    """

    name: str
    latitude: float
    longitude: float
    weight: float


@dataclass(frozen=True)
class Region:
    """Everything that differs between one grid and another."""

    key: str                 # short id used in filenames
    name: str                # human label
    eia_respondent: str      # EIA-930 balancing authority code
    local_timezone: str      # IANA tz, for calendar features only
    weather_points: tuple[WeatherPoint, ...] = field(default_factory=tuple)

    def total_weight(self) -> float:
        return sum(p.weight for p in self.weather_points)


ERCOT = Region(
    key="ercot",
    name="ERCOT (Texas)",
    eia_respondent="ERCO",
    local_timezone="America/Chicago",
    weather_points=(
        WeatherPoint("Dallas-Fort Worth", 32.7767, -96.7970, 0.34),
        WeatherPoint("Houston", 29.7604, -95.3698, 0.32),
        WeatherPoint("San Antonio", 29.4241, -98.4936, 0.17),
        WeatherPoint("Austin", 30.2672, -97.7431, 0.17),
    ),
)

ISONE = Region(
    key="isone",
    name="ISO New England",
    eia_respondent="ISNE",
    local_timezone="America/New_York",
    weather_points=(
        WeatherPoint("Boston", 42.3601, -71.0589, 0.55),
        WeatherPoint("Hartford", 41.7658, -72.6734, 0.20),
        WeatherPoint("Providence", 41.8240, -71.4128, 0.15),
        WeatherPoint("Portland ME", 43.6591, -70.2568, 0.10),
    ),
)

REGIONS: dict[str, Region] = {r.key: r for r in (ERCOT, ISONE)}

DEFAULT_REGION = "ercot"


def get_region(key: str = DEFAULT_REGION) -> Region:
    """Look up a region, failing loudly on a typo."""
    try:
        return REGIONS[key]
    except KeyError:
        known = ", ".join(sorted(REGIONS))
        raise KeyError(f"Unknown region {key!r}. Known regions: {known}") from None


# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------

def eia_api_key() -> str:
    """Read the EIA key from the environment.

    Never hardcode it, never commit it. Put it in a .env file (gitignored)
    and load it with python-dotenv, or export it in your shell.
    """
    key = os.environ.get("EIA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "EIA_API_KEY is not set. Get a free key at "
            "https://www.eia.gov/opendata/ and put it in a .env file as:\n"
            "    EIA_API_KEY=your_key_here"
        )
    return key


def ensure_dirs() -> None:
    """Create the data directories if they don't exist."""
    for d in (RAW_DIR, INTERIM_DIR):
        d.mkdir(parents=True, exist_ok=True)