"""
Alternative Data Sources — macro, sentiment, and options-derived features.

Data sources (all free, no API key required unless noted):
  - FRED (Federal Reserve Economic Data): macro indicators via fredapi
  - CNN Fear & Greed Index: market sentiment scraping
  - yfinance options chain: implied volatility, put/call ratio
  - yfinance info: short interest data

All functions are designed for graceful degradation — if a source is
unavailable, the feature columns are filled with NaN and a warning is logged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from app.config import logger, _env

# ═══════════════════════════════════════════════════════════════════════════
# 1. FRED — Macroeconomic indicators
# ═══════════════════════════════════════════════════════════════════════════

try:
    from fredapi import Fred as _Fred
    HAS_FRED = True
except ImportError:
    HAS_FRED = False


# Series IDs for common macro indicators
FRED_SERIES = {
    "fed_funds_rate": "DFF",           # effective federal funds rate (daily)
    "treasury_10y": "DGS10",           # 10-year treasury yield (daily)
    "treasury_2y": "DGS2",             # 2-year treasury yield (daily)
    "treasury_3m": "DGS3MO",           # 3-month treasury yield (daily)
    "vix": "VIXCLS",                   # CBOE VIX (daily)
    "unemployment": "UNRATE",          # unemployment rate (monthly)
    "cpi_yoy": "CPIAUCSL",             # CPI (monthly, needs yoy calc)
    "initial_claims": "ICSA",          # initial jobless claims (weekly)
}


def fetch_fred_data(
    start: str = "2015-01-01",
    end: str | None = None,
    series: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Fetch macroeconomic data from FRED.

    Requires FRED_API_KEY in environment.  Free registration at
    https://fred.stlouisfed.org/docs/api/api_key.html

    Returns a DataFrame indexed by date with one column per series.
    Missing values are forward-filled (macro data is released at lower
    frequencies than daily).
    """
    api_key = _env("FRED_API_KEY")
    if not api_key:
        logger.info("FRED_API_KEY not set — skipping macro data")
        return pd.DataFrame()

    if not HAS_FRED:
        logger.warning("fredapi not installed (pip install fredapi)")
        return pd.DataFrame()

    if series is None:
        series = FRED_SERIES
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    fred = _Fred(api_key=api_key)
    frames: dict[str, pd.Series] = {}

    for name, series_id in series.items():
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            if s is not None and not s.empty:
                frames[name] = s
        except Exception as e:
            logger.warning("FRED fetch failed for %s (%s): %s", name, series_id, e)

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Derived: yield curve spread (10y - 2y), classic recession indicator
    if "treasury_10y" in df.columns and "treasury_2y" in df.columns:
        df["yield_curve_10y2y"] = df["treasury_10y"] - df["treasury_2y"]

    # Derived: term spread (10y - 3m)
    if "treasury_10y" in df.columns and "treasury_3m" in df.columns:
        df["term_spread_10y3m"] = df["treasury_10y"] - df["treasury_3m"]

    # CPI year-over-year change
    if "cpi_yoy" in df.columns:
        df["cpi_yoy"] = df["cpi_yoy"].pct_change(periods=12) * 100

    # Forward-fill lower-frequency data to daily
    df = df.ffill()

    logger.info("FRED data loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


def merge_fred_features(df: pd.DataFrame, fred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge FRED macro features into the main OHLCV DataFrame.
    Uses date-based join with forward fill for alignment.
    """
    if fred_df.empty:
        return df

    # Prefix columns to avoid collisions
    fred_cols = {c: f"macro_{c}" for c in fred_df.columns}
    fred_renamed = fred_df.rename(columns=fred_cols)

    df = df.join(fred_renamed, how="left")

    # Forward fill since macro data may not cover every trading day
    macro_cols = list(fred_cols.values())
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    logger.info("Merged %d FRED features", len(macro_cols))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. CNN Fear & Greed Index
# ═══════════════════════════════════════════════════════════════════════════

def fetch_fear_greed_index() -> dict:
    """
    Fetch the current CNN Fear & Greed Index.

    Returns dict with keys: value (0-100), label, timestamp.
    Uses the public CNN API endpoint.

    Gracefully returns empty dict on failure.
    """
    try:
        import urllib.request
        import json

        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        score = data.get("fear_and_greed", {}).get("score")
        rating = data.get("fear_and_greed", {}).get("rating")
        timestamp = data.get("fear_and_greed", {}).get("timestamp")

        if score is not None:
            result = {
                "value": float(score),
                "label": str(rating) if rating else _fg_label(float(score)),
                "timestamp": timestamp,
            }
            logger.info("Fear & Greed Index: %.0f (%s)", result["value"], result["label"])
            return result

    except Exception as e:
        logger.warning("Failed to fetch Fear & Greed Index: %s", e)

    return {}


def _fg_label(value: float) -> str:
    if value <= 25:
        return "Extreme Fear"
    elif value <= 45:
        return "Fear"
    elif value <= 55:
        return "Neutral"
    elif value <= 75:
        return "Greed"
    else:
        return "Extreme Greed"


def add_fear_greed_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a static Fear & Greed column (current snapshot) to the DataFrame.
    Since historical F&G data isn't freely available, this fills the latest
    value into the most recent rows (useful for live/current analysis).
    """
    fg = fetch_fear_greed_index()
    if not fg:
        df["fear_greed"] = np.nan
        df["fear_greed_label"] = "unknown"
        return df

    df["fear_greed"] = np.nan
    df["fear_greed_label"] = "unknown"

    # Fill in last 5 rows with current value (for current-state analysis)
    if len(df) >= 1:
        df.iloc[-min(5, len(df)):, df.columns.get_loc("fear_greed")] = fg["value"]
        df.iloc[-min(5, len(df)):, df.columns.get_loc("fear_greed_label")] = fg["label"]

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. Options-derived features (yfinance — no API key needed)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_options_features(ticker: str) -> dict:
    """
    Extract options-derived features from yfinance for current analysis.

    Returns dict with:
      - implied_vol_atm: at-the-money implied volatility (avg of nearest call & put)
      - put_call_ratio: total put OI / total call OI
      - put_call_volume_ratio: total put volume / total call volume
      - max_pain: strike with maximum pain (max total OI)
      - iv_skew: OTM put IV - OTM call IV (positive = fear)

    All values are snapshots (current).  Gracefully returns empty dict on failure.
    """
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            logger.info("No options data for %s", ticker)
            return {}

        # Use nearest expiration for most liquid data
        exp = expirations[0]
        chain = tk.option_chain(exp)
        calls = chain.calls
        puts = chain.puts

        if calls.empty or puts.empty:
            return {}

        # Current price approximation
        spot = tk.fast_info.get("lastPrice") or tk.fast_info.get("previousClose")
        if spot is None or spot <= 0:
            # fallback: midpoint of ATM strikes
            spot = calls["strike"].median()

        result = {}

        # Put/Call ratio (open interest)
        total_call_oi = calls["openInterest"].sum()
        total_put_oi = puts["openInterest"].sum()
        if total_call_oi > 0:
            result["put_call_ratio"] = float(total_put_oi / total_call_oi)
        else:
            result["put_call_ratio"] = np.nan

        # Put/Call volume ratio
        total_call_vol = calls["volume"].sum()
        total_put_vol = puts["volume"].sum()
        if total_call_vol > 0:
            result["put_call_volume_ratio"] = float(total_put_vol / total_call_vol)
        else:
            result["put_call_volume_ratio"] = np.nan

        # ATM implied volatility
        atm_call_idx = (calls["strike"] - spot).abs().idxmin()
        atm_put_idx = (puts["strike"] - spot).abs().idxmin()
        call_iv = calls.loc[atm_call_idx, "impliedVolatility"]
        put_iv = puts.loc[atm_put_idx, "impliedVolatility"]
        result["implied_vol_atm"] = float((call_iv + put_iv) / 2)

        # IV Skew: OTM put IV vs OTM call IV
        # OTM puts: strike < spot * 0.95, OTM calls: strike > spot * 1.05
        otm_puts = puts[puts["strike"] < spot * 0.95]
        otm_calls = calls[calls["strike"] > spot * 1.05]
        if not otm_puts.empty and not otm_calls.empty:
            avg_put_iv = otm_puts["impliedVolatility"].mean()
            avg_call_iv = otm_calls["impliedVolatility"].mean()
            result["iv_skew"] = float(avg_put_iv - avg_call_iv)
        else:
            result["iv_skew"] = np.nan

        # Max pain: strike where total (call + put) OI is maximum
        all_strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
        max_oi = 0
        max_pain_strike = spot
        for strike in all_strikes:
            c_oi = calls.loc[calls["strike"] == strike, "openInterest"].sum()
            p_oi = puts.loc[puts["strike"] == strike, "openInterest"].sum()
            total = c_oi + p_oi
            if total > max_oi:
                max_oi = total
                max_pain_strike = strike
        result["max_pain"] = float(max_pain_strike)
        result["max_pain_distance_pct"] = float((spot - max_pain_strike) / spot * 100) if spot > 0 else 0.0

        logger.info("Options features for %s: P/C=%.2f, IV_ATM=%.2f, skew=%.3f",
                     ticker,
                     result.get("put_call_ratio", 0),
                     result.get("implied_vol_atm", 0),
                     result.get("iv_skew", 0))

        return result

    except Exception as e:
        logger.warning("Failed to fetch options data for %s: %s", ticker, e)
        return {}


def add_options_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add options-derived features as static columns (current snapshot)."""
    opts = fetch_options_features(ticker)
    if not opts:
        for col in ["opt_put_call_ratio", "opt_put_call_vol_ratio", "opt_iv_atm",
                     "opt_iv_skew", "opt_max_pain", "opt_max_pain_dist_pct"]:
            df[col] = np.nan
        return df

    col_map = {
        "put_call_ratio": "opt_put_call_ratio",
        "put_call_volume_ratio": "opt_put_call_vol_ratio",
        "implied_vol_atm": "opt_iv_atm",
        "iv_skew": "opt_iv_skew",
        "max_pain": "opt_max_pain",
        "max_pain_distance_pct": "opt_max_pain_dist_pct",
    }

    for src_key, dst_col in col_map.items():
        val = opts.get(src_key, np.nan)
        # Fill only the last N rows with current snapshot
        df[dst_col] = np.nan
        if len(df) >= 1:
            df.iloc[-min(5, len(df)):, df.columns.get_loc(dst_col)] = val

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. Short Interest (yfinance info — no API key needed)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_short_interest(ticker: str) -> dict:
    """
    Fetch short interest data from yfinance .info property.

    Returns dict with:
      - short_ratio: days to cover
      - short_pct_float: short % of float
      - short_pct_shares: short % of shares outstanding
    """
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        info = tk.info

        result = {}
        if "shortRatio" in info and info["shortRatio"]:
            result["short_ratio"] = float(info["shortRatio"])
        if "shortPercentOfFloat" in info and info["shortPercentOfFloat"]:
            result["short_pct_float"] = float(info["shortPercentOfFloat"])
        if "sharesShortPriorMonth" in info and info.get("sharesOutstanding"):
            result["short_pct_shares"] = float(
                info["sharesShortPriorMonth"] / info["sharesOutstanding"]
            )

        if result:
            logger.info("Short interest for %s: ratio=%.1f, pct_float=%.2f%%",
                         ticker,
                         result.get("short_ratio", 0),
                         result.get("short_pct_float", 0) * 100)
        return result

    except Exception as e:
        logger.warning("Failed to fetch short interest for %s: %s", ticker, e)
        return {}


def add_short_interest_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add short interest data as static columns."""
    si = fetch_short_interest(ticker)

    col_map = {
        "short_ratio": "si_short_ratio",
        "short_pct_float": "si_short_pct_float",
        "short_pct_shares": "si_short_pct_shares",
    }

    for src_key, dst_col in col_map.items():
        val = si.get(src_key, np.nan)
        df[dst_col] = np.nan
        if len(df) >= 1:
            df.iloc[-min(5, len(df)):, df.columns.get_loc(dst_col)] = val

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. Master integration function
# ═══════════════════════════════════════════════════════════════════════════

def add_all_alternative_data(
    df: pd.DataFrame,
    ticker: str,
    enable_fred: bool = True,
    enable_fear_greed: bool = True,
    enable_options: bool = True,
    enable_short_interest: bool = True,
) -> pd.DataFrame:
    """
    Add all available alternative data features to the DataFrame.

    Each source is independent and optional — failures in one do not
    affect others.
    """
    if enable_fred:
        try:
            start = df.index.min().strftime("%Y-%m-%d") if not df.empty else "2015-01-01"
            end = df.index.max().strftime("%Y-%m-%d") if not df.empty else None
            fred_df = fetch_fred_data(start=start, end=end)
            df = merge_fred_features(df, fred_df)
        except Exception as e:
            logger.warning("FRED integration failed: %s", e)

    if enable_fear_greed:
        try:
            df = add_fear_greed_feature(df)
        except Exception as e:
            logger.warning("Fear & Greed integration failed: %s", e)

    if enable_options:
        try:
            df = add_options_features(df, ticker)
        except Exception as e:
            logger.warning("Options integration failed: %s", e)

    if enable_short_interest:
        try:
            df = add_short_interest_features(df, ticker)
        except Exception as e:
            logger.warning("Short interest integration failed: %s", e)

    return df
