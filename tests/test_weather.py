"""Tests for S2. All offline.

The important ones are the weighted-average tests. That function is where a
silent, plausible, wrong answer is easiest to produce.
"""

import pandas as pd
import pytest

from honest_forecast import cache, weather
from honest_forecast.config import WeatherPoint, get_region


@pytest.fixture(autouse=True)
def temp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(cache, "INTERIM_DIR", tmp_path / "interim")


def payload(n: int = 3, **series) -> dict:
    times = pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {"hourly": {"time": times, **series}}


# --------------------------------------------------------------------------
# Columnar parsing
# --------------------------------------------------------------------------

def test_parallel_arrays_become_rows():
    df = weather._hourly_to_frame(payload(3, temperature_2m=[1.0, 2.0, 3.0]))

    assert len(df) == 3
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert df["temperature_2m"].tolist() == [1.0, 2.0, 3.0]


def test_mismatched_array_lengths_raise():
    """If the arrays differ in length the rows misalign -- every temperature
    attaches to the wrong hour. That must never pass silently."""
    bad = {"hourly": {"time": ["2024-01-01T00:00", "2024-01-01T01:00"], "temperature_2m": [1.0]}}
    with pytest.raises(RuntimeError, match="mismatched lengths"):
        weather._hourly_to_frame(bad)


def test_missing_time_array_raises():
    with pytest.raises(RuntimeError, match="no `time` array"):
        weather._hourly_to_frame({"hourly": {"temperature_2m": [1.0]}})


# --------------------------------------------------------------------------
# Weighted aggregation -- the trap
# --------------------------------------------------------------------------

def two_cities() -> tuple[WeatherPoint, ...]:
    return (
        WeatherPoint("Big", 32.0, -96.0, 0.75),
        WeatherPoint("Small", 29.0, -98.0, 0.25),
    )


def frames(big: list, small: list) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=len(big), freq="h", tz="UTC")
    return {
        "Big": pd.DataFrame({"temp_actual": big}, index=idx),
        "Small": pd.DataFrame({"temp_actual": small}, index=idx),
    }


def test_weights_applied_correctly():
    out = weather.weighted_temperature(frames([10.0], [20.0]), two_cities(), "temp_actual")
    # 0.75*10 + 0.25*20 = 12.5
    assert out["temp_actual"].iloc[0] == pytest.approx(12.5)
    assert out["temp_actual_n_cities"].iloc[0] == 2


def test_missing_city_renormalises_instead_of_dragging_the_average_down():
    """THE BUG THIS FUNCTION EXISTS TO PREVENT.

    Small city is missing. A naive weighted sum gives 0.75*10 = 7.5 -- as if
    the region were 7.5C when every reading available said 10C. Plausible,
    silent, wrong. Correct behaviour is to renormalise: the answer is 10.
    """
    out = weather.weighted_temperature(frames([10.0], [None]), two_cities(), "temp_actual")

    assert out["temp_actual"].iloc[0] == pytest.approx(10.0)
    assert out["temp_actual"].iloc[0] != pytest.approx(7.5), "naive weighted sum bug"
    assert out["temp_actual_n_cities"].iloc[0] == 1, "the count is how you SEE this happened"


def test_all_cities_missing_gives_nan_not_zero():
    out = weather.weighted_temperature(frames([None], [None]), two_cities(), "temp_actual")

    assert pd.isna(out["temp_actual"].iloc[0]), "no data must be NaN, never 0"
    assert out["temp_actual_n_cities"].iloc[0] == 0


def test_partial_coverage_is_visible_across_hours():
    out = weather.weighted_temperature(
        frames([10.0, 10.0, 10.0], [20.0, None, 20.0]), two_cities(), "temp_actual"
    )
    assert out["temp_actual"].tolist() == pytest.approx([12.5, 10.0, 12.5])
    assert out["temp_actual_n_cities"].tolist() == [2, 1, 2]


def test_real_region_weights_are_used():
    """Guard against the config and the aggregation drifting apart."""
    region = get_region("ercot")
    assert region.total_weight() == pytest.approx(1.0)
    assert len(region.weather_points) == 4


# --------------------------------------------------------------------------
# Error surfacing
# --------------------------------------------------------------------------

def test_api_error_body_is_surfaced(monkeypatch):
    class FakeResponse:
        status_code = 400

        @staticmethod
        def json():
            return {"error": True, "reason": "Value error: invalid date"}

    monkeypatch.setattr(weather.requests, "get", lambda *a, **k: FakeResponse())

    with pytest.raises(RuntimeError, match="invalid date"):
        weather._request("http://example.com", {})


def test_missing_forecast_variable_raises(monkeypatch):
    """If Open-Meteo silently omits a lead time we asked for, stop."""
    monkeypatch.setattr(
        weather, "_request", lambda url, params: payload(2, temperature_2m_previous_day1=[1.0, 2.0])
    )
    point = WeatherPoint("Nowhere", 32.0, -96.0, 1.0)

    with pytest.raises(RuntimeError, match="did not return"):
        weather.fetch_forecast(point, "2024-01-01", "2024-01-02", lead_days=(1, 2))


# --------------------------------------------------------------------------
# Column selection and dtype
#
# Both of these are bugs that got through the first time. The first produced a
# false "SUSPICIOUS" alarm by comparing city COUNTS against temperatures. The
# second let JSON nulls sit as object-dtype None instead of NaN.
# --------------------------------------------------------------------------

def test_forecast_columns_exclude_bookkeeping():
    df = pd.DataFrame(
        columns=[
            "temp_actual", "temp_actual_n_cities",
            "temp_fcst_day1", "temp_fcst_day1_n_cities",
            "temp_fcst_day2", "temp_fcst_day2_n_cities",
        ]
    )
    assert weather._forecast_columns(df) == ["temp_fcst_day1", "temp_fcst_day2"]


def test_json_nulls_become_float_nan_not_object_none():
    df = weather._hourly_to_frame(payload(3, temperature_2m=[1.0, None, 3.0]))

    assert df["temperature_2m"].dtype.kind == "f", "must be float, not object"
    assert df["temperature_2m"].isna().sum() == 1
    # The real symptom: arithmetic on an object column misbehaves.
    assert (df["temperature_2m"] * 2).sum() == pytest.approx(8.0)


def test_all_null_forecast_column_is_still_float():
    """A lead time entirely outside the archive comes back all nulls. It must
    still be a float column so downstream maths does not explode."""
    df = weather._hourly_to_frame(payload(3, temperature_2m_previous_day3=[None, None, None]))
    assert df["temperature_2m_previous_day3"].dtype.kind == "f"
    assert df["temperature_2m_previous_day3"].isna().all()