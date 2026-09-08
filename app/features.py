"""
Comprehensive feature engineering for market time series.

Features include: returns, volatility, technical indicators (RSI, MACD, Bollinger,
ATR), momentum, drawdown, volume analysis, lag features, calendar features,
trend labels, and forecasting targets.
"""

import numpy as np
import pandas as pd

from app.config import logger


# ---------------------------------------------------------------------------
# Basic returns and volatility
# ---------------------------------------------------------------------------

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Simple and log returns."""
    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def compute_rolling_volatility(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Annualized rolling volatility for multiple windows."""
    if windows is None:
        windows = [10, 20, 60]
    for w in windows:
        df[f"vol_{w}"] = df["log_return"].rolling(w).std() * np.sqrt(252)
    return df


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def compute_sma(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    if windows is None:
        windows = [20, 50, 200]
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(w).mean()
    return df


def compute_ema(df: pd.DataFrame, spans: list[int] | None = None) -> pd.DataFrame:
    if spans is None:
        spans = [12, 26]
    for s in spans:
        df[f"EMA_{s}"] = df["Close"].ewm(span=s, adjust=False).mean()
    return df


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return df


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def compute_bollinger(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    sma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["BB_upper"] = sma + num_std * std
    df["BB_lower"] = sma - num_std * std
    df["BB_mid"] = sma
    df["BB_pct"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / (sma + 1e-9)
    return df


# ---------------------------------------------------------------------------
# ATR (Average True Range)
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df[f"ATR_{window}"] = tr.rolling(window).mean()
    return df


# ---------------------------------------------------------------------------
# Momentum & Rate of Change
# ---------------------------------------------------------------------------

def compute_momentum(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    if periods is None:
        periods = [5, 10, 20]
    for p in periods:
        df[f"momentum_{p}"] = df["Close"] - df["Close"].shift(p)
        df[f"roc_{p}"] = df["Close"].pct_change(p)
    return df


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def compute_drawdown(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Rolling max drawdown."""
    roll_max = df["Close"].rolling(window, min_periods=1).max()
    df["drawdown"] = (df["Close"] - roll_max) / (roll_max + 1e-9)
    return df


# ---------------------------------------------------------------------------
# Volume features
# ---------------------------------------------------------------------------

def compute_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    vol = df["Volume"]
    vol_median = vol.rolling(window).median()
    df["volume_spike"] = vol / (vol_median + 1e-9)
    df["volume_sma"] = vol.rolling(window).mean()
    df["volume_std"] = vol.rolling(window).std()

    # Price-volume divergence: price up but volume down (or inverse)
    price_dir = np.sign(df["Close"].diff())
    vol_dir = np.sign(vol.diff())
    df["pv_divergence"] = (price_dir != vol_dir).astype(int)

    return df


# ---------------------------------------------------------------------------
# Distance from moving averages
# ---------------------------------------------------------------------------

def compute_ma_distances(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    for w in [20, 50, 200]:
        col = f"SMA_{w}"
        if col in df.columns:
            df[f"dist_SMA_{w}"] = (close - df[col]) / (df[col] + 1e-9)
    return df


# ---------------------------------------------------------------------------
# Volatility-adjusted return
# ---------------------------------------------------------------------------

def compute_vol_adj_return(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_adj_return"] = df["log_return"] / (df.get("vol_20", df["log_return"].rolling(20).std()) + 1e-9)
    return df


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def compute_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    for lag in lags:
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)
        # Use vol_20 if already computed, otherwise compute on the fly
        if "vol_20" in df.columns:
            df[f"vol_lag_{lag}"] = df["vol_20"].shift(lag)
        else:
            vol_20 = df["log_return"].rolling(20).std()
            df[f"vol_lag_{lag}"] = vol_20.shift(lag)
    return df


# ---------------------------------------------------------------------------
# Calendar / seasonal features
# ---------------------------------------------------------------------------

def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["is_month_start"] = idx.is_month_start.astype(int)
    df["is_month_end"] = idx.is_month_end.astype(int)
    df["is_quarter_end"] = idx.is_quarter_end.astype(int)
    return df


# ---------------------------------------------------------------------------
# Trend direction label
# ---------------------------------------------------------------------------

def compute_trend_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify trend based on SMA positions:
      - bullish: Close > SMA_50 > SMA_200
      - bearish: Close < SMA_50 < SMA_200
      - sideways: otherwise
    """
    close = df["Close"]
    sma50 = df.get("SMA_50", close)
    sma200 = df.get("SMA_200", close)

    conditions = [
        (close > sma50) & (sma50 > sma200),
        (close < sma50) & (sma50 < sma200),
    ]
    choices = ["bullish", "bearish"]
    df["trend"] = np.select(conditions, choices, default="sideways")
    return df


# ---------------------------------------------------------------------------
# Volatility regime
# ---------------------------------------------------------------------------

def compute_volatility_regime(df: pd.DataFrame, vol_col: str = "vol_20") -> pd.DataFrame:
    """
    Classify volatility into low / medium / high based on percentiles
    of the rolling volatility distribution.
    """
    if vol_col not in df.columns:
        df["vol_regime"] = "unknown"
        return df

    vol = df[vol_col]
    q33 = vol.quantile(0.33)
    q66 = vol.quantile(0.66)
    df["vol_regime"] = np.select(
        [vol <= q33, vol <= q66],
        ["low", "medium"],
        default="high",
    )
    return df


# ---------------------------------------------------------------------------
# Forecast targets
# ---------------------------------------------------------------------------

def compute_targets(df: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    """
    Create target variables for supervised forecasting:
      - target_return_N: cumulative log return over next N days
      - target_dir_N: 1 if positive, 0 if negative
    """
    if horizons is None:
        horizons = [1, 5]
    for h in horizons:
        if h == 1:
            fwd = df["log_return"].shift(-1)
        else:
            # Sum of next h log returns: r_{t+1} + r_{t+2} + ... + r_{t+h}
            fwd = df["log_return"].shift(-1).rolling(h, min_periods=h).sum()
        df[f"target_return_{h}"] = fwd
        df[f"target_dir_{h}"] = (fwd > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def add_all_features(df: pd.DataFrame, include_targets: bool = False) -> pd.DataFrame:
    """
    Compute all features on the OHLCV DataFrame.
    Returns a copy with NaN rows from warm-up periods dropped.
    """
    logger.info("Computing features on %d rows", len(df))
    df = df.copy()

    df = compute_returns(df)
    df = compute_rolling_volatility(df)
    df = compute_sma(df)
    df = compute_ema(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_bollinger(df)
    df = compute_atr(df)
    df = compute_momentum(df)
    df = compute_drawdown(df)
    df = compute_volume_features(df)
    df = compute_ma_distances(df)
    df = compute_vol_adj_return(df)
    df = compute_lag_features(df)
    df = compute_calendar_features(df)
    df = compute_trend_label(df)
    df = compute_volatility_regime(df)

    if include_targets:
        df = compute_targets(df)

    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows where core indicators are NaN (warm-up period)
    core_cols = ["log_return", "vol_20", "RSI_14", "MACD", "BB_pct"]
    existing_core = [c for c in core_cols if c in df.columns]
    df = df.dropna(subset=existing_core)

    logger.info("Features computed: %d rows, %d columns retained", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# ML feature matrix preparation
# ---------------------------------------------------------------------------

ML_FEATURE_COLS = [
    "log_return", "vol_20", "vol_adj_return",
    "dist_SMA_20", "dist_SMA_50", "dist_SMA_200",
    "RSI_14", "ATR_14", "MACD_hist", "BB_pct", "BB_width",
    "volume_spike", "momentum_5", "roc_10", "drawdown",
    "return_lag_1", "return_lag_2", "return_lag_3",
]


def prepare_ml_features(df: pd.DataFrame, feature_cols: list[str] | None = None):
    """
    Build a clean feature matrix for unsupervised/supervised models.

    Returns
    -------
    X_raw : np.ndarray
    X_scaled : np.ndarray
    feature_cols : list[str]
    scaler : RobustScaler
    index_used : pd.DatetimeIndex
    """
    from sklearn.preprocessing import RobustScaler

    if feature_cols is None:
        feature_cols = [c for c in ML_FEATURE_COLS if c in df.columns]

    X_df = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(X_df) < 10:
        scaler = RobustScaler()
        empty = np.empty((0, len(feature_cols)))
        return empty, empty, feature_cols, scaler, X_df.index

    X_raw = X_df.values.astype(np.float64)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_raw, X_scaled, feature_cols, scaler, X_df.index
