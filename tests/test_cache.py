"""Tests for the cache layer.

The point of these is to prove that a second call does NOT hit the network --
that's the whole reason the cache exists. We never make a real request here;
we count how many times a fake fetch function was called.
"""

import pandas as pd
import pytest

from honest_forecast import cache


@pytest.fixture(autouse=True)
def temp_dirs(tmp_path, monkeypatch):
    """Point the cache at a throwaway directory for each test."""
    monkeypatch.setattr(cache, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(cache, "INTERIM_DIR", tmp_path / "interim")


def test_json_fetches_once_then_serves_from_disk():
    calls = []

    def fetch():
        calls.append(1)
        return {"hello": "world"}

    first = cache.cached_json("demo", fetch)
    second = cache.cached_json("demo", fetch)

    assert first == second == {"hello": "world"}
    assert len(calls) == 1, "second call should have come from disk"


def test_force_refetches():
    calls = []

    def fetch():
        calls.append(1)
        return {"n": len(calls)}

    cache.cached_json("demo", fetch)
    result = cache.cached_json("demo", fetch, force=True)

    assert len(calls) == 2
    assert result == {"n": 2}


def test_frame_round_trips_and_caches():
    calls = []

    def build():
        calls.append(1)
        idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        return pd.DataFrame({"load": [1.0, 2.0, 3.0]}, index=idx)

    first = cache.cached_frame("frame_demo", build)
    second = cache.cached_frame("frame_demo", build)

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)
    assert str(second.index.tz) == "UTC", "timezone must survive the round trip"


def test_build_returning_wrong_type_fails_fast():
    with pytest.raises(TypeError, match="not a DataFrame"):
        cache.cached_frame("bad", lambda: {"not": "a frame"})


def test_unsafe_names_do_not_escape_the_cache_dir():
    cache.cached_json("../../etc/passwd", lambda: {"ok": True})
    written = list(cache.RAW_DIR.glob("*.json"))
    assert len(written) == 1
    assert ".." not in written[0].name