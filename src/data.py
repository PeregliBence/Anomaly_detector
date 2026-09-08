"""Market data ingestion with an on-disk parquet cache.

The cache makes walk-forward experiments repeatable without re-hitting the
Yahoo API, which matters because a full backfill re-reads the same history
hundreds of times.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(ticker: str, interval: str) -> Path:
    safe = ticker.replace("/", "_").replace("^", "IDX_").replace("=", "_")
    return CACHE_DIR / f"{safe}_{interval}.parquet"


def _download(ticker: str, start: str, end: str | None, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end, interval=interval,
        progress=False, auto_adjust=True,
    )
    if df.empty:
        return pd.DataFrame()

    if getattr(df.columns, "nlevels", 1) > 1:
        df.columns = df.columns.get_level_values(0)

    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        logger.error("%s: missing columns %s", ticker, missing)
        return pd.DataFrame()

    df = df[OHLCV].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df[~df.index.duplicated(keep="first")].sort_index().dropna(subset=["Close"])


def load(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return OHLCV data, refreshing the cache only for the missing tail."""
    path = _cache_path(ticker, interval)
    cached = pd.DataFrame()

    if use_cache and path.exists():
        cached = pd.read_parquet(path)
        covers_start = cached.index.min() <= pd.Timestamp(start)
        fresh_enough = end is not None and cached.index.max() >= pd.Timestamp(end)
        if covers_start and fresh_enough:
            return cached.loc[start:end].copy()

    fetch_start = start
    if not cached.empty and cached.index.min() <= pd.Timestamp(start):
        fetch_start = (cached.index.max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    logger.info("Downloading %s [%s -> %s]", ticker, fetch_start, end or "today")
    fresh = _download(ticker, fetch_start, end, interval)

    if fresh.empty and cached.empty:
        logger.warning("No data available for %s", ticker)
        return pd.DataFrame()

    combined = fresh if cached.empty else pd.concat([cached, fresh])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path)

    return combined.loc[start:end].copy()
