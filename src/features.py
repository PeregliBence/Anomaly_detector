"""Causal feature engineering and h-day directional targets.

Every feature at row t is computed only from data available at t. Statistics
that would otherwise summarise the whole sample (volatility regime, scaling)
are either expanding-window or deferred to the model layer, where they are fit
inside the training fold.
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
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the causal feature matrix. Index is preserved."""
    close, volume = df["Close"], df["Volume"]
    out = pd.DataFrame(index=df.index)

    log_ret = np.log(close / close.shift(1))
    out["log_return"] = log_ret

    for lag in (1, 2, 3, 5, 10):
        out[f"ret_lag_{lag}"] = log_ret.shift(lag)

    for w in (5, 10, 20, 60):
        out[f"ret_mean_{w}"] = log_ret.rolling(w).mean()
        out[f"vol_{w}"] = log_ret.rolling(w).std()

    # Volatility-scaled momentum: comparable across assets and regimes.
    for p in (5, 10, 20, 60):
        out[f"mom_{p}"] = np.log(close / close.shift(p)) / (out["vol_20"] * np.sqrt(p) + 1e-9)

    for w in (20, 50, 200):
        sma = close.rolling(w).mean()
        out[f"dist_sma_{w}"] = (close - sma) / (sma + 1e-9)

    out["rsi_14"] = _rsi(close)

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (close + 1e-9)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_pct"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)
    out["bb_width"] = (4 * bb_std) / (bb_mid + 1e-9)

    out["atr_norm"] = _atr(df) / (close + 1e-9)

    roll_max = close.rolling(252, min_periods=20).max()
    out["drawdown"] = (close - roll_max) / (roll_max + 1e-9)

    out["volume_spike"] = volume / (volume.rolling(20).median() + 1e-9)
    out["volume_trend"] = np.log(
        (volume.rolling(5).mean() + 1) / (volume.rolling(60).mean() + 1)
    )

    # Expanding rank keeps the regime label causal; a global quantile would
    # leak the future distribution into every historical row.
    out["vol_regime"] = out["vol_20"].expanding(min_periods=252).rank(pct=True)
    out["vol_ratio"] = out["vol_20"] / (out["vol_60"] + 1e-9)

    out["day_of_week"] = df.index.dayofweek
    out["month"] = df.index.month

    return out.replace([np.inf, -np.inf], np.nan)


def drift_estimate(df: pd.DataFrame, horizon: int = 5, window: int = 252) -> pd.Series:
    """Expected h-day return implied by the trailing mean daily return.

    Causal by construction: only returns realised up to t enter the estimate.
    """
    daily = np.log(df["Close"] / df["Close"].shift(1))
    return daily.rolling(window).mean() * horizon


def make_target(df: pd.DataFrame, horizon: int = 5, mode: str = "absolute") -> pd.Series:
    """Binary target for the move from t to t+horizon.

    mode="absolute" : did the price rise? Base rate follows the asset's drift,
        so on equities "always up" already scores ~58% and is the bar to beat.
    mode="excess"   : did the move beat the trailing drift? Base rate sits near
        50% by construction, so any edge above it is genuine skill rather than
        a restatement of the upward drift.

    NaN where the outcome has not resolved yet, keeping unlabelled rows out of
    training.
    """
    fwd_return = np.log(df["Close"].shift(-horizon) / df["Close"])

    if mode == "absolute":
        threshold = 0.0
    elif mode == "excess":
        threshold = drift_estimate(df, horizon)
    else:
        raise ValueError(f"Unknown target mode '{mode}'")

    return (fwd_return > threshold).astype(float).where(fwd_return.notna())


def forward_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Realised cumulative log return from t to t+horizon."""
    return np.log(df["Close"].shift(-horizon) / df["Close"])


def build_dataset(
    df: pd.DataFrame, horizon: int = 5, target_mode: str = "absolute"
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return aligned (features, target, forward_return) with warm-up rows dropped."""
    X = build_features(df)
    y = make_target(df, horizon, mode=target_mode)
    fwd = forward_return(df, horizon)

    # Rows with an unresolved target are kept: they carry no label for training
    # but are exactly the rows a live run needs to predict on.
    X = X.loc[X.notna().all(axis=1)]
    return X, y.loc[X.index], fwd.loc[X.index]
