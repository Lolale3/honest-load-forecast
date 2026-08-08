"""Tests for S1. All offline -- we never call EIA here.

We build fake API responses in the shape EIA documents, then check that our
parsing does the right thing, and more importantly that it REFUSES when the
data is wrong.
"""

import pandas as pd
import pytest

from honest_forecast import cache, eia
from honest_forecast.config import get_region


@pytest.fixture(autouse=True)
def temp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(cache, "INTERIM_DIR", tmp_path / "interim")


def make_rows(n_hours: int, start: str = "2024-01-01T00") -> list[dict]:
    """Two rows per hour -- one demand, one forecast -- as EIA sends them."""
    periods = pd.date_range(start.replace("T", " "), periods=n_hours, freq="h")
    rows = []
    for i, p in enumerate(periods):
        stamp = p.strftime("%Y-%m-%dT%H")
        rows.append({"period": stamp, "respondent": "ERCO", "type": "D", "value": 40000 + i})
        rows.append({"period": stamp, "respondent": "ERCO", "type": "DF", "value": 39900 + i})
    return rows


# --------------------------------------------------------------------------
# The pivot
# --------------------------------------------------------------------------

def test_long_rows_become_two_columns():
    df = eia._to_frame(make_rows(3))

    assert list(df.columns) == ["demand", "day_ahead_forecast"]
    assert len(df) == 3, "3 hours x 2 types = 6 long rows -> 3 wide rows"
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing


def test_values_land_in_the_right_columns():
    df = eia._to_frame(make_rows(1))
    assert df["demand"].iloc[0] == 40000
    assert df["day_ahead_forecast"].iloc[0] == 39900


def test_unwanted_types_are_dropped():
    rows = make_rows(2)
    rows.append({"period": "2024-01-01T00", "type": "NG", "value": 12345})  # net generation
    df = eia._to_frame(rows)
    assert list(df.columns) == ["demand", "day_ahead_forecast"]
    assert len(df) == 2


def test_duplicate_period_type_raises_instead_of_averaging():
    """The whole reason we check before pivoting.

    Without this, pandas would quietly average 40000 and 99999 and hand us a
    frame that looks perfectly fine.
    """
    rows = make_rows(2)
    rows.append({"period": "2024-01-01T00", "type": "D", "value": 99999})

    with pytest.raises(RuntimeError, match="duplicated"):
        eia._to_frame(rows)


def test_missing_columns_raise():
    with pytest.raises(RuntimeError, match="missing columns"):
        eia._to_frame([{"period": "2024-01-01T00", "value": 1}])


def test_missing_series_becomes_a_null_column_not_a_crash():
    """If EIA has demand but no forecast for a window, we still want a frame
    with both columns -- otherwise downstream code breaks in a confusing way."""
    rows = [r for r in make_rows(2) if r["type"] == "D"]
    df = eia._to_frame(rows)
    assert list(df.columns) == ["demand", "day_ahead_forecast"]
    assert df["day_ahead_forecast"].isna().all()


# --------------------------------------------------------------------------
# The pagination guard
# --------------------------------------------------------------------------

def fake_api(rows: list[dict], total: int | None = None, page_size: int = 4):
    """Stand in for EIA. Serves `rows` in pages, reporting `total`."""
    reported = len(rows) if total is None else total
    calls = {"n": 0}

    def _fetch_page(region, start, end, offset, *, force=False):
        calls["n"] += 1
        return {"response": {"total": str(reported), "data": rows[offset:offset + page_size]}}

    return _fetch_page, calls


def test_pagination_collects_every_row(monkeypatch):
    rows = make_rows(5)  # 10 long rows, served 4 at a time
    fetch, calls = fake_api(rows)
    monkeypatch.setattr(eia, "_fetch_page", fetch)

    got = eia.fetch_all(get_region("ercot"), "2024-01-01T00", "2024-01-01T04")

    assert len(got) == 10
    assert calls["n"] == 3, "10 rows at 4 per page = 3 requests"


def test_short_pull_raises_rather_than_returning_partial_data(monkeypatch):
    """The failure this whole design exists to prevent.

    EIA claims 100 rows; the pages run dry at 10. Without the guard we would
    return a perfectly normal-looking frame missing 90% of the data.
    """
    fetch, _ = fake_api(make_rows(5), total=100)
    monkeypatch.setattr(eia, "_fetch_page", fetch)

    with pytest.raises(RuntimeError, match="Pagination lost rows"):
        eia.fetch_all(get_region("ercot"), "2024-01-01T00", "2024-01-01T04")


def test_zero_rows_raises(monkeypatch):
    fetch, _ = fake_api([], total=0)
    monkeypatch.setattr(eia, "_fetch_page", fetch)

    with pytest.raises(RuntimeError, match="0 rows"):
        eia.fetch_all(get_region("ercot"), "2024-01-01T00", "2024-01-01T04")


def test_absurdly_large_request_is_refused(monkeypatch):
    fetch, _ = fake_api(make_rows(2), total=999_999)
    monkeypatch.setattr(eia, "_fetch_page", fetch)

    with pytest.raises(RuntimeError, match="Narrow the date range"):
        eia.fetch_all(get_region("ercot"), "2019-01-01T00", "2026-01-01T00")


# --------------------------------------------------------------------------
# The timezone sanity check itself
#
# The first version of this check asserted a late-afternoon peak. That is a
# SUMMER invariant, and it produced a false alarm on real January data where
# ERCOT peaks in the morning. These tests pin down the corrected behaviour so
# the same mistake cannot come back.
# --------------------------------------------------------------------------

def daily_curve(peak_hour: int, trough_hour: int, tz: str = "America/Chicago") -> pd.DataFrame:
    """A week of synthetic demand with a chosen peak and trough hour."""
    idx = pd.date_range("2024-01-01", periods=24 * 7, freq="h", tz="UTC").tz_convert(tz)
    hours = idx.hour
    demand = 50000 + 10000 * ((hours == peak_hour).astype(float)) - 8000 * ((hours == trough_hour).astype(float))
    return pd.DataFrame({"demand": demand}, index=idx)


def test_winter_morning_peak_is_accepted():
    """The real January 2024 shape: ERCOT's record was set in the 7-8am hour."""
    ok, notes = eia._timezone_looks_right(daily_curve(peak_hour=8, trough_hour=3))
    assert ok
    assert any("winter heating" in n for n in notes)


def test_summer_afternoon_peak_is_accepted():
    ok, notes = eia._timezone_looks_right(daily_curve(peak_hour=17, trough_hour=4))
    assert ok
    assert any("summer cooling" in n for n in notes)


def test_overnight_peak_is_rejected():
    """A genuine timezone bug shifts the whole curve. The trough moving out of
    the small hours is what gives it away, in any season."""
    ok, notes = eia._timezone_looks_right(daily_curve(peak_hour=2, trough_hour=14))
    assert not ok
    assert any("TROUGH IS WRONG" in n for n in notes)