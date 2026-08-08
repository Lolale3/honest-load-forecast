"""Tests for configuration. These run offline -- no network, no API key."""

import pytest

from honest_forecast import config


def test_default_region_is_ercot():
    assert config.get_region().key == "ercot"


def test_unknown_region_fails_loudly():
    with pytest.raises(KeyError, match="Unknown region"):
        config.get_region("caiso")


@pytest.mark.parametrize("key", sorted(config.REGIONS))
def test_weather_weights_sum_to_one(key):
    """If the weights don't sum to 1 the weighted temperature is silently wrong.

    This is the kind of bug that produces a model that ALMOST works, which is
    much worse than one that obviously doesn't.
    """
    region = config.get_region(key)
    assert region.total_weight() == pytest.approx(1.0)


@pytest.mark.parametrize("key", sorted(config.REGIONS))
def test_weather_points_are_plausible_us_coordinates(key):
    region = config.get_region(key)
    assert region.weather_points, f"{key} has no weather points"
    for p in region.weather_points:
        assert 24 < p.latitude < 50, f"{p.name} latitude looks wrong"
        assert -125 < p.longitude < -66, f"{p.name} longitude looks wrong"


def test_missing_api_key_gives_a_useful_error(monkeypatch):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="eia.gov/opendata"):
        config.eia_api_key()