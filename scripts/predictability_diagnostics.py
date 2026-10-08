"""Why direction is hard and volatility is not.

Quantifies, on the project's own data, the three facts that set the ceiling for
any directional model:

  1. Autocorrelation of returns (~0) vs of absolute returns (strongly positive).
  2. The accuracy ceiling implied by the drift-to-noise ratio.
  3. Out-of-sample R^2 for predicting the next 5-day return vs the next 5-day
     realised volatility, using the same naive predictor in both cases.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sources import yahoo

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s | %(message)s")

TICKERS = ["AAPL", "MSFT", "^GSPC", "BTC-USD"]
HORIZON = 5
LAGS = [1, 2, 5, 10, 20, 60]


def autocorrelations(returns: pd.Series) -> dict:
    out = {}
    for lag in LAGS:
        out[f"ret_lag{lag}"] = returns.autocorr(lag)
        out[f"abs_lag{lag}"] = returns.abs().autocorr(lag)
    return out


def accuracy_ceiling(returns: pd.Series, horizon: int) -> dict:
    """Best achievable directional accuracy if only the drift were known.

    For returns that are roughly normal with mean mu and deviation sigma over
    the horizon, always betting on the drift direction is correct with
    probability Phi(mu / sigma).
    """
    mu = returns.mean() * horizon
    sigma = returns.std() * np.sqrt(horizon)
    return {
        "drift_per_window_pct": mu * 100,
        "noise_per_window_pct": sigma * 100,
        "signal_to_noise": mu / sigma,
        "accuracy_ceiling": stats.norm.cdf(mu / sigma),
    }


def naive_r2(df: pd.DataFrame, horizon: int) -> dict:
    """R^2 of a naive 'tomorrow looks like the recent past' predictor.

    Applied to two targets: the forward return, and the forward realised
    volatility. Same predictor family, same data, drastically different result.
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    # Target A: cumulative return over the next `horizon` days.
    fwd_return = log_ret.shift(-horizon).rolling(horizon).sum()
    pred_return = log_ret.rolling(horizon).sum()  # trailing return as predictor

    # Target B: realised volatility over the next `horizon` days.
    fwd_vol = log_ret.shift(-horizon).rolling(horizon).std()
    pred_vol = log_ret.rolling(horizon).std()  # trailing volatility as predictor

    def r2(actual: pd.Series, predicted: pd.Series) -> float:
        joined = pd.concat([actual, predicted], axis=1).dropna()
        if len(joined) < 100:
            return float("nan")
        a, p = joined.iloc[:, 0], joined.iloc[:, 1]
        return 1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum()

    def correlation(actual: pd.Series, predicted: pd.Series) -> float:
        joined = pd.concat([actual, predicted], axis=1).dropna()
        return joined.iloc[:, 0].corr(joined.iloc[:, 1])

    return {
        "return_r2": r2(fwd_return, pred_return),
        "return_corr": correlation(fwd_return, pred_return),
        "vol_r2": r2(fwd_vol, pred_vol),
        "vol_corr": correlation(fwd_vol, pred_vol),
    }


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    ac_rows, ceiling_rows, r2_rows = [], [], []

    for ticker in TICKERS:
        df = yahoo.load_prices(ticker, start="2010-01-01", end="2026-09-01")
        if df.empty:
            continue
        log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()

        ac_rows.append({"ticker": ticker, **autocorrelations(log_ret)})
        ceiling_rows.append({"ticker": ticker, **accuracy_ceiling(log_ret, HORIZON)})
        r2_rows.append({"ticker": ticker, **naive_r2(df, HORIZON)})

    print("\n=== 1. Autocorrelation: sign carries no memory, magnitude does ===")
    ac = pd.DataFrame(ac_rows).set_index("ticker").round(3)
    print("\nReturns (direction):")
    print(ac[[c for c in ac.columns if c.startswith("ret_")]].to_string())
    print("\nAbsolute returns (magnitude):")
    print(ac[[c for c in ac.columns if c.startswith("abs_")]].to_string())

    print(f"\n=== 2. Accuracy ceiling from drift alone ({HORIZON}-day horizon) ===")
    print(pd.DataFrame(ceiling_rows).set_index("ticker").round(4).to_string())

    print(f"\n=== 3. Naive predictor R^2: return vs volatility ({HORIZON}-day) ===")
    print(pd.DataFrame(r2_rows).set_index("ticker").round(4).to_string())


if __name__ == "__main__":
    main()
