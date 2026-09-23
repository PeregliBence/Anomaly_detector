"""Per-instrument causal features.

Every value on row t is computed from data available at t. Statistics that
would otherwise summarise the whole sample — a volatility regime from a global
quantile, a fitted scaler — are either expanding-window here or deferred to the
model, where they are fit inside the training fold.

Feature choice follows the cross-sectional literature rather than the usual
technical-indicator grab bag: momentum at several lookbacks, short-horizon
reversal, volatility, and liquidity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal feature matrix for one instrument, indexed by date."""
    close, volume, high = df["Close"], df["Volume"], df["High"]
    out = pd.DataFrame(index=df.index)

    log_ret = np.log(close / close.shift(1))
    out["log_return"] = log_ret

    # --- volatility -------------------------------------------------------
    for w in (5, 20, 60, 252):
        out[f"vol_{w}"] = log_ret.rolling(w).std()
    out["vol_ratio"] = out["vol_20"] / (out["vol_252"] + 1e-9)

    # --- momentum ---------------------------------------------------------
    # 12-1 is the standard momentum definition: the last month is excluded
    # because short-horizon reversal runs against the longer trend.
    out["mom_12_1"] = np.log(close.shift(21) / close.shift(252))
    out["mom_6_1"] = np.log(close.shift(21) / close.shift(126))
    for p in (20, 60, 120):
        out[f"mom_{p}"] = np.log(close / close.shift(p))

    # Risk-adjusted momentum travels better across instruments and regimes.
    out["mom_12_1_vol_adj"] = out["mom_12_1"] / (out["vol_252"] * np.sqrt(252) + 1e-9)

    # --- reversal ---------------------------------------------------------
    out["reversal_5"] = np.log(close / close.shift(5))
    out["reversal_21"] = np.log(close / close.shift(21))

    # --- trend position ---------------------------------------------------
    for w in (20, 50, 200):
        sma = close.rolling(w).mean()
        out[f"dist_sma_{w}"] = (close - sma) / (sma + 1e-9)

    out["dist_52w_high"] = close / (high.rolling(252).max() + 1e-9) - 1
    roll_max = close.rolling(252, min_periods=20).max()
    out["drawdown"] = (close - roll_max) / (roll_max + 1e-9)

    # --- oscillators ------------------------------------------------------
    out["rsi_14"] = _rsi(close)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (close + 1e-9)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_pct"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    out["atr_norm"] = _atr(df) / (close + 1e-9)

    # --- liquidity --------------------------------------------------------
    dollar_volume = close * volume
    out["dollar_volume"] = dollar_volume.rolling(20).mean()
    out["log_dollar_volume"] = np.log1p(out["dollar_volume"])
    out["volume_spike"] = volume / (volume.rolling(20).median() + 1e-9)
    # Amihud illiquidity: price impact per unit of traded value.
    out["illiquidity"] = (log_ret.abs() / (dollar_volume + 1)).rolling(20).mean() * 1e9

    # --- lags -------------------------------------------------------------
    for lag in (1, 2, 5):
        out[f"ret_lag_{lag}"] = log_ret.shift(lag)

    # Expanding rank keeps the regime label causal; a global quantile would
    # leak the whole sample's distribution into every historical row.
    out["vol_regime"] = out["vol_20"].expanding(min_periods=252).rank(pct=True)

    return out.replace([np.inf, -np.inf], np.nan)


def build_market_features(market: pd.DataFrame) -> pd.DataFrame:
    """State-of-the-world features shared by every instrument on a date."""
    out = pd.DataFrame(index=market.index)

    if "vix" in market:
        out["vix"] = market["vix"]
        out["vix_change_20"] = market["vix"].pct_change(20)
        out["vix_rank"] = market["vix"].expanding(min_periods=252).rank(pct=True)

    # Term structure of volatility: backwardation marks acute stress.
    if "vix" in market and "vix3m" in market:
        out["vix_term"] = market["vix"] / (market["vix3m"] + 1e-9)

    # Curve slope is the classic macro state variable.
    if "yield_10y" in market and "yield_13w" in market:
        out["yield_slope"] = market["yield_10y"] - market["yield_13w"]
        out["yield_slope_change_60"] = out["yield_slope"].diff(60)
    if "yield_10y" in market:
        out["yield_10y_change_20"] = market["yield_10y"].diff(20)

    # Credit: high yield relative to investment grade widens before equities.
    if "hy_credit" in market and "ig_credit" in market:
        ratio = market["hy_credit"] / (market["ig_credit"] + 1e-9)
        out["credit_ratio"] = ratio
        out["credit_momentum_60"] = np.log(ratio / ratio.shift(60))

    for name in ("dollar", "gold", "oil"):
        if name in market:
            out[f"{name}_momentum_60"] = np.log(
                market[name] / market[name].shift(60)
            )

    if "spx" in market:
        spx_ret = np.log(market["spx"] / market["spx"].shift(1))
        out["market_vol_20"] = spx_ret.rolling(20).std()
        out["market_mom_60"] = np.log(market["spx"] / market["spx"].shift(60))
        out["market_drawdown"] = (
            market["spx"] / market["spx"].rolling(252, min_periods=20).max() - 1
        )

    return out.replace([np.inf, -np.inf], np.nan)


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Realised log return from t to t+horizon."""
    return np.log(df["Close"].shift(-horizon) / df["Close"])
