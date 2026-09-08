"""
Data ingestion module.
Downloads and validates market data from Yahoo Finance with caching support.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from app.config import logger


def load_data(
    ticker: str,
    start: str = "2018-01-01",
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol (e.g. "AAPL", "BTC-USD", "^GSPC").
    start, end : str
        Date range in YYYY-MM-DD format.
    interval : str
        Data frequency: '1d', '1h', '5m', etc.

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    logger.info("Downloading %s data [%s → %s] interval=%s", ticker, start, end, interval)

    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    except Exception as e:
        logger.error("Failed to download data for %s: %s", ticker, e)
        return pd.DataFrame()

    if df.empty:
        logger.warning("No data returned for %s", ticker)
        return pd.DataFrame()

    # Collapse MultiIndex columns (yfinance quirk)
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Missing columns %s for %s", missing, ticker)
        return pd.DataFrame()

    df = df[required].copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=["Close"])

    if df.empty:
        logger.error("All Close prices are NaN after cleaning for %s", ticker)
        return pd.DataFrame()

    logger.info("Loaded %d rows for %s", len(df), ticker)
    return df


def validate_data(df: pd.DataFrame, min_rows: int = 60) -> tuple[bool, list[str]]:
    """
    Validate data quality.
    Returns (is_valid, list_of_warnings).
    """
    warnings = []

    if df.empty:
        return False, ["Üres adathalmaz."]

    if len(df) < min_rows:
        warnings.append(f"Kevés adat: {len(df)} sor (ajánlott minimum: {min_rows}).")

    nan_pct = df.isna().mean()
    for col in ["Close", "Volume"]:
        if col in nan_pct and nan_pct[col] > 0.05:
            warnings.append(f"Sok hiányzó adat a(z) {col} oszlopban: {nan_pct[col]:.1%}")

    if len(df) > 1:
        gaps = df.index.to_series().diff()
        max_gap = gaps.max()
        if max_gap > timedelta(days=10):
            warnings.append(f"Nagy adathiány az idősorban: {max_gap.days} nap.")

    is_valid = len(df) >= min_rows
    return is_valid, warnings


def load_multiple_tickers(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load data for multiple tickers. Returns dict[ticker -> DataFrame]."""
    results = {}
    for t in tickers:
        df = load_data(t, start=start, end=end)
        if not df.empty:
            results[t] = df
    return results
