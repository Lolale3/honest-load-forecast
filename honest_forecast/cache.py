"""Disk caching.

Why this exists, and why it's the second file in the project:

You are going to re-run the ingestion code dozens of times while you get the
parsing right. If every run hits the API you will be slow, rate-limited, and
tempted to test on less data than you should.

So there are two layers, and the split matters:

  RAW  (json)     -- exactly what the API returned, untouched.
  INTERIM (parquet) -- what your code made of it.

When your parsing logic is wrong -- and it will be -- you fix the code and
re-parse from RAW. No network call. That separation is the single highest
return-on-effort thing in a data project.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import INTERIM_DIR, RAW_DIR


def _safe(name: str) -> str:
    """Make a string safe to use as a filename.

    Dots are stripped, not kept. We add our own extension, so a dot in the key
    serves no purpose -- and allowing them means a key like "../../etc/passwd"
    turns into a filename containing "..", which is the kind of thing that
    looks harmless until it isn't.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# --------------------------------------------------------------------------
# Layer 1: raw API responses
# --------------------------------------------------------------------------

def raw_path(name: str) -> Path:
    return RAW_DIR / f"{_safe(name)}.json"


def cached_json(
    name: str,
    fetch: Callable[[], Any],
    *,
    force: bool = False,
) -> Any:
    """Return cached JSON for `name`, calling `fetch()` only if we must.

    Parameters
    ----------
    name : cache key; becomes the filename.
    fetch : zero-arg callable that does the network request and returns
            something JSON-serialisable.
    force : ignore any existing cache and re-fetch.
    """
    path = raw_path(name)
    if path.exists() and not force:
        with path.open() as fh:
            return json.load(fh)

    payload = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first, then move. If the process dies mid-write we
    # end up with no cache rather than a corrupt one that looks valid.
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh)
    tmp.replace(path)
    return payload


# --------------------------------------------------------------------------
# Layer 2: parsed dataframes
# --------------------------------------------------------------------------

def frame_path(name: str) -> Path:
    return INTERIM_DIR / f"{_safe(name)}.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Make a frame look the same whether it came from build() or from disk.

    Parquet does not preserve a DatetimeIndex's `freq` attribute. So a freshly
    built frame has `index.freq == <Hour>` and the same frame reloaded has
    `index.freq is None`. That asymmetry is poison: your code works on a cold
    run and breaks on a warm one, or the reverse, and the cause is invisible.

    We resolve it by always dropping freq. That is also the honest choice --
    this data WILL have missing hours, so a frame claiming a regular hourly
    frequency would be lying. Stage S3 puts the data on an explicit complete
    index instead, where the gaps are visible rather than assumed away.
    """
    if isinstance(df.index, pd.DatetimeIndex) and df.index.freq is not None:
        df = df.copy()
        df.index = df.index._with_freq(None)
    return df


def cached_frame(
    name: str,
    build: Callable[[], pd.DataFrame],
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Return a cached DataFrame for `name`, calling `build()` only if we must."""
    path = frame_path(name)
    if path.exists() and not force:
        return _normalize(pd.read_parquet(path))

    df = build()
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"build() for {name!r} returned {type(df).__name__}, not a DataFrame")
    df = _normalize(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=True)
    tmp.replace(path)
    return df


def clear(name: str) -> None:
    """Delete both cache layers for one key. Useful when you suspect bad data."""
    for p in (raw_path(name), frame_path(name)):
        p.unlink(missing_ok=True)