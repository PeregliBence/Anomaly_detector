"""Yahoo Finance source.

Free and deep, but survivor-only: tickers that delisted are largely absent, so
a universe drawn from here is biased toward companies that made it. This
module therefore serves two roles — the cross-asset series, where survivorship
does not apply, and an interim equity universe until Sharadar's delisting-aware
data replaces it.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from src import cache

logger = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Cross-asset state. These carry information no single equity's own history
# does: the price of risk, the shape of the curve, the dollar, and credit.
# Every series here has history back to 2003, so the macro features are
# complete from the start of the study and 2008 falls in the test period.
# ^VIX3M (from 2006) and the HYG/LQD credit ETFs (from 2007) were dropped for
# that reason; credit now comes from FRED — see sources.fred.
MARKET_SERIES = {
    "vix": "^VIX",
    "yield_10y": "^TNX",
    "yield_13w": "^IRX",
    "yield_30y": "^TYX",
    "dollar": "DX-Y.NYB",
    "gold": "GC=F",
    "oil": "CL=F",
    "spx": "^GSPC",
    "russell": "IWM",
}

# Sector ETFs, used to describe where the market's leadership sits rather than
# as cross-section members. Their dispersion measures rotation, which single
# names cannot report about themselves.
SECTOR_SERIES = {
    "tech": "XLK", "financials": "XLF", "health": "XLV", "energy": "XLE",
    "industrials": "XLI", "discretionary": "XLY", "staples": "XLP",
    "utilities": "XLU", "materials": "XLB",
}


def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=True, interval="1d")
    if df.empty:
        return pd.DataFrame()

    if getattr(df.columns, "nlevels", 1) > 1:
        df.columns = df.columns.get_level_values(0)
    if any(c not in df.columns for c in OHLCV):
        return pd.DataFrame()

    df = df[OHLCV].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df[~df.index.duplicated(keep="first")].sort_index().dropna(subset=["Close"])


def load_prices(
    ticker: str, start: str, end: str | None = None, use_cache: bool = True
) -> pd.DataFrame:
    """Daily OHLCV for one instrument, cached on disk."""
    key = f"yahoo_{ticker}"

    if use_cache:
        cached = cache.read(key)
        if cached is not None and not cached.empty:
            covers = cached.index.min() <= pd.Timestamp(start)
            fresh = end is not None and cached.index.max() >= pd.Timestamp(end)
            if covers and fresh:
                return cached.loc[start:end].copy()

    fresh_df = _download(ticker, start, end)
    if fresh_df.empty:
        return pd.DataFrame()

    if use_cache:
        cache.write(key, fresh_df)
    return fresh_df.loc[start:end].copy()


def load_many(
    tickers: list[str], start: str, end: str | None = None, min_rows: int = 300
) -> dict[str, pd.DataFrame]:
    """Load a set of instruments, dropping those with too little history."""
    out = {}
    for i, ticker in enumerate(tickers, 1):
        df = load_prices(ticker, start, end)
        if len(df) >= min_rows:
            out[ticker] = df
        if i % 50 == 0:
            logger.info("Loaded %d/%d instruments", i, len(tickers))
    logger.info("Loaded %d of %d requested instruments", len(out), len(tickers))
    return out


def load_market_state(start: str, end: str | None = None) -> pd.DataFrame:
    """Cross-asset and sector series aligned on one daily index.

    Forward-filled because these markets keep different calendars; a stale
    value is what an observer would have seen, so filling forward stays causal.
    Sector columns are prefixed so the feature layer can find them.
    """
    columns = {}

    for name, ticker in MARKET_SERIES.items():
        df = load_prices(ticker, start, end)
        if not df.empty:
            columns[name] = df["Close"]
        else:
            logger.warning("Market series unavailable: %s (%s)", name, ticker)

    for name, ticker in SECTOR_SERIES.items():
        df = load_prices(ticker, start, end)
        if not df.empty:
            columns[f"sector_{name}"] = df["Close"]

    if not columns:
        return pd.DataFrame()

    market = pd.DataFrame(columns).sort_index()

    try:
        from src.sources.fred import load_credit_state
        credit = load_credit_state(start)
        if not credit.empty:
            market = market.join(credit, how="outer")
    except Exception as e:
        logger.warning("Credit state unavailable: %s", e)

    # Restrict to days the equity market traded, after forward-filling the
    # series that keep other calendars.
    trading_days = market["spx"].dropna().index if "spx" in market else market.index
    market = market.sort_index().ffill().reindex(trading_days)
    market.index.name = "date"
    return market
