"""Tests for S3. All offline.

The interpolation tests are the important ones. Filling is the only place in
this pipeline where we CREATE data, so the rules around it need to be pinned
down hard.
"""

import pandas as pd
import pytest

from honest_forecast import align


# --------------------------------------------------------------------------
# The spine
# --------------------------------------------------------------------------

def test_spine_covers_every_hour_inclusive():
    spine = align.hourly_spine("2023-01-01", "2023-01-02")

    assert len(spine) == 48
    assert spine[0] == pd.Timestamp("2023-01-01 00:00", tz="UTC")
    assert spine[-1] == pd.Timestamp("2023-01-02 23:00", tz="UTC")


def test_spine_rejects_backwards_range():
    with pytest.raises(ValueError, match="must be after"):
        align.hourly_spine("2023-06-01", "2023-01-01")


def test_spine_is_hourly_and_utc():
    spine = align.hourly_spine("2023-03-11", "2023-03-13")
    assert str(spine.tz) == "UTC"
    # Spring-forward is a LOCAL clock event. In UTC every day is 24 hours,
    # which is exactly why we keep the spine in UTC and convert only for display.
    assert len(spine) == 72


# --------------------------------------------------------------------------
# Gap runs -- shape, not just count
# --------------------------------------------------------------------------

def series_with(pattern: list) -> pd.Series:
    idx = pd.date_range("2023-01-01", periods=len(pattern), freq="h", tz="UTC")
    return pd.Series(pattern, index=idx)


def test_no_gaps_returns_empty():
    assert align.gap_runs(series_with([1.0, 2.0, 3.0])).empty


def test_consecutive_missing_values_form_one_run():
    runs = align.gap_runs(series_with([1.0, None, None, None, 5.0]))

    assert len(runs) == 1
    assert runs.iloc[0]["hours"] == 3


def test_scattered_and_blocked_gaps_are_distinguished():
    """Same null count, completely different problems. This is why gap_runs
    exists rather than just calling .isna().sum()."""
    scattered = align.gap_runs(series_with([None, 1.0, None, 1.0, None, 1.0]))
    blocked = align.gap_runs(series_with([None, None, None, 1.0, 1.0, 1.0]))

    assert scattered["hours"].sum() == blocked["hours"].sum() == 3
    assert len(scattered) == 3, "three separate single-hour gaps"
    assert len(blocked) == 1, "one three-hour outage"


def test_runs_are_sorted_largest_first():
    runs = align.gap_runs(series_with([None, 1.0, None, None, None, 1.0]))
    assert runs.iloc[0]["hours"] == 3


# --------------------------------------------------------------------------
# Filling -- the only place we create data
# --------------------------------------------------------------------------

def frame_with(values: list) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame({"temp_actual": values}, index=idx)


def test_short_gap_is_interpolated():
    df = align.fill_short_gaps(frame_with([10.0, None, 12.0]), ["temp_actual"])

    assert df["temp_actual"].iloc[1] == pytest.approx(11.0)
    assert df["temp_actual_filled"].tolist() == [False, True, False]


def test_gap_at_the_limit_is_filled():
    df = align.fill_short_gaps(frame_with([10.0, None, None, None, 14.0]), ["temp_actual"])
    assert df["temp_actual"].notna().all()
    assert int(df["temp_actual_filled"].sum()) == 3


def test_gap_beyond_the_limit_is_left_alone():
    """A 4-hour gap exceeds MAX_FILL_HOURS=3, so NOTHING in it is filled.

    This test caught a real bug. The first implementation used pandas'
    `.interpolate(limit=3, limit_direction="both")`, where `limit` applies
    from each end -- so a 4-hour gap was filled 3-from-the-front and
    3-from-the-back, i.e. entirely. The cap silently doubled.
    """
    df = align.fill_short_gaps(frame_with([10.0, None, None, None, None, 15.0]), ["temp_actual"])

    assert int(df["temp_actual"].isna().sum()) == 4
    assert int(df["temp_actual_filled"].sum()) == 0


def test_leading_and_trailing_gaps_are_never_extrapolated():
    """Filling between two known points is estimation. Filling before the
    first or after the last is invention."""
    df = align.fill_short_gaps(frame_with([None, 10.0, 11.0, None]), ["temp_actual"])

    assert pd.isna(df["temp_actual"].iloc[0])
    assert pd.isna(df["temp_actual"].iloc[-1])
    assert int(df["temp_actual_filled"].sum()) == 0


def test_fill_flag_marks_only_created_values():
    df = align.fill_short_gaps(frame_with([10.0, None, 12.0, 13.0]), ["temp_actual"])
    assert df["temp_actual_filled"].tolist() == [False, True, False, False]


def test_interpolation_is_time_aware_not_positional():
    """With an irregular index, a positional fill would put the midpoint in
    the wrong place. method='time' weights by actual elapsed time."""
    idx = pd.DatetimeIndex(
        ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 04:00"], tz="UTC"
    )
    df = pd.DataFrame({"temp_actual": [10.0, None, 40.0]}, index=idx)

    out = align.fill_short_gaps(df, ["temp_actual"])
    # One hour into a four-hour span: 10 + (40-10) * 1/4 = 17.5
    assert out["temp_actual"].iloc[1] == pytest.approx(17.5)


# --------------------------------------------------------------------------
# Outage exclusion
# --------------------------------------------------------------------------

def test_known_outage_window_is_removed():
    idx = pd.date_range("2023-12-28", "2024-01-22", freq="h", tz="UTC")
    df = pd.DataFrame({"demand": 1.0}, index=idx)

    out, removed = align.drop_known_outages(df)

    # Dec 30 00:00 .. Jan 19 23:00 inclusive is 21 days, not 20. The gap of
    # MISSING days is 20 (Dec 30 .. Jan 18); the exclusion window extends
    # through the 19th because that day is only half covered.
    assert removed == 21 * 24, "2023-12-30 00:00 through 2024-01-19 23:00"
    assert not ((out.index >= "2023-12-30") & (out.index <= "2024-01-19")).any()
    assert len(out) == len(idx) - removed


def test_data_outside_the_outage_is_untouched():
    idx = pd.date_range("2023-06-01", "2023-06-30", freq="h", tz="UTC")
    df = pd.DataFrame({"demand": 1.0}, index=idx)

    out, removed = align.drop_known_outages(df)

    assert removed == 0
    assert len(out) == len(df)


def test_long_gap_is_not_partially_filled():
    """Even one-directionally, pandas' `limit` would fill the first 3 hours of
    a 10-hour gap. That is worse than not filling: it hides an outage behind
    plausible numbers. Long gaps must be left wholly intact."""
    values = [10.0] + [None] * 10 + [20.0]
    df = align.fill_short_gaps(frame_with(values), ["temp_actual"])

    assert int(df["temp_actual"].isna().sum()) == 10, "all ten hours stay missing"
    assert int(df["temp_actual_filled"].sum()) == 0


def test_short_and_long_gaps_in_one_series_are_handled_separately():
    """The rule is per-gap, not per-column."""
    values = [10.0, None, 12.0] + [None] * 5 + [20.0]
    df = align.fill_short_gaps(frame_with(values), ["temp_actual"])

    assert df["temp_actual"].iloc[1] == pytest.approx(11.0), "1-hour gap filled"
    assert df["temp_actual"].iloc[3:8].isna().all(), "5-hour gap untouched"
    assert int(df["temp_actual_filled"].sum()) == 1


# --------------------------------------------------------------------------
# Year chunking
#
# Chunk boundaries are a classic source of duplicated or dropped hours: an
# off-by-one at a year end silently doubles 31 December or loses 1 January.
# --------------------------------------------------------------------------

def test_single_year_is_one_chunk():
    assert align._year_chunks("2023-01-01", "2023-12-31") == [("2023-01-01", "2023-12-31")]


def test_multi_year_splits_on_calendar_years():
    assert align._year_chunks("2021-04-01", "2023-06-15") == [
        ("2021-04-01", "2021-12-31"),
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-06-15"),
    ]


def test_chunks_are_contiguous_with_no_overlap():
    """Every day in the range appears in exactly one chunk. An overlap
    duplicates hours; a gap loses them."""
    chunks = align._year_chunks("2021-04-01", "2026-08-01")

    days = pd.DatetimeIndex([])
    for lo, hi in chunks:
        days = days.append(pd.date_range(lo, hi, freq="D"))

    expected = pd.date_range("2021-04-01", "2026-08-01", freq="D")
    assert len(days) == len(expected), "chunks must tile the range exactly"
    assert not days.duplicated().any(), "chunks must not overlap"
    assert days.equals(expected)


def test_partial_year_at_both_ends_is_clipped():
    assert align._year_chunks("2022-06-01", "2022-06-30") == [("2022-06-01", "2022-06-30")]