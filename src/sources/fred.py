"""FRED macro series.

Two timing details matter here.

Publication lag: a value FRED dates as day t is released the following
morning, so at the close of day t only the value for t-1 is known. Every
series is shifted by one observation before it is aligned to the trading
calendar; using it unshifted would hand the model one day of the future.

Licensing: FRED's ICE BofA spread series now return only the last three years,
which rules them out for a study starting in 2005. The Moody's corporate
yield series below run daily since the 1980s and are the classic credit
predictors in the literature (the Baa-Aaa default spread of Fama and French,
1989).
"""

from __future__ import annotations

import io
import logging
import os
import time

import pandas as pd
import requests

from src import cache

logger = logging.getLogger(__name__)

API = "https://api.stlouisfed.org/fred/series/observations"
GRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ATTEMPTS = 3

CREDIT_SERIES = {
    "baa10y": "BAA10Y",   # Moody's Baa yield minus 10-year Treasury
    "aaa10y": "AAA10Y",   # Moody's Aaa yield minus 10-year Treasury
    "dbaa": "DBAA",       # Moody's Baa yield
    "daaa": "DAAA",       # Moody's Aaa yield
}


def load_series(series_id: str, start: str, use_cache: bool = True) -> pd.Series:
    """One FRED series as a float Series indexed by observation date."""
    key = f"fred_{series_id}"
    if use_cache:
        cached = cache.read(key)
        if cached is not None and not cached.empty and cached.index.min() <= pd.Timestamp(start):
            return cached["value"].loc[start:]

    series = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            series = _from_api(series_id, start)
            break
        except Exception as e:
            logger.info("FRED API attempt %d for %s failed: %s", attempt, series_id, e)
            time.sleep(2 * attempt)
    if series is None:
        # The API has outages of its own; the public graph export serves the
        # same observations without a key.
        series = _from_graph_csv(series_id, start)

    series = series.dropna().rename(series_id)
    series.index.name = "date"
    if use_cache and not series.empty:
        cache.write(key, series.to_frame("value"))
    return series


def _from_api(series_id: str, start: str) -> pd.Series:
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is not set. Add it to .env at the project root.")

    resp = requests.get(API, params={
        "series_id": series_id, "api_key": api_key,
        "file_type": "json", "observation_start": start,
    }, timeout=60)
    resp.raise_for_status()

    obs = pd.DataFrame(resp.json().get("observations", []))
    if obs.empty:
        return pd.Series(dtype=float)
    return pd.Series(
        pd.to_numeric(obs["value"], errors="coerce").values,  # FRED marks gaps with "."
        index=pd.to_datetime(obs["date"]),
    )


def _from_graph_csv(series_id: str, start: str) -> pd.Series:
    resp = requests.get(GRAPH_CSV, params={"id": series_id, "cosd": start}, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # First column is the date (its header has changed over time), second the value.
    return pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                     index=pd.to_datetime(df.iloc[:, 0]))


def load_credit_state(start: str) -> pd.DataFrame:
    """Credit spread series, lagged one observation for publication timing."""
    columns = {}
    for name, series_id in CREDIT_SERIES.items():
        try:
            columns[name] = load_series(series_id, start).shift(1)
        except Exception as e:
            logger.warning("FRED series unavailable: %s (%s): %s", name, series_id, e)
    return pd.DataFrame(columns).sort_index()
