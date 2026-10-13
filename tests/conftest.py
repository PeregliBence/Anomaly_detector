"""Synthetic market data for the test suite.

Tests never touch the Sharadar snapshot or the network, so they run anywhere,
including the CI runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_prices(n_days: int = 600, seed: int = 0, start: str = "2015-01-01") -> pd.DataFrame:
    """One instrument's OHLCV as a random walk on business days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days, name="date")
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
    spread = np.abs(rng.normal(0, 0.01, n_days))
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.005, n_days)),
        "High": close * (1 + spread),
        "Low": close * (1 - spread),
        "Close": close,
        "Volume": rng.integers(100_000, 5_000_000, n_days).astype(float),
    }, index=dates)


@pytest.fixture
def frames() -> dict[str, pd.DataFrame]:
    return {f"T{i:02d}": make_prices(seed=i) for i in range(30)}


@pytest.fixture
def market() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2014-01-01", periods=900, name="date")

    def walk(level, vol):
        return level * np.exp(np.cumsum(rng.normal(0, vol, len(dates))))

    cols = {name: walk(level, 0.01) for name, level in {
        "spx": 2000, "russell": 120, "dollar": 95, "gold": 1200, "oil": 60}.items()}
    cols.update({"vix": 15 + np.abs(rng.normal(0, 5, len(dates))),
                 "yield_10y": 2 + rng.normal(0, 0.1, len(dates)).cumsum() / 20,
                 "yield_13w": 0.5 + rng.normal(0, 0.05, len(dates)).cumsum() / 20,
                 "baa10y": 2 + rng.normal(0, 0.02, len(dates)).cumsum() / 10,
                 "dbaa": 5 + rng.normal(0, 0.02, len(dates)).cumsum() / 10,
                 "daaa": 4 + rng.normal(0, 0.02, len(dates)).cumsum() / 10})
    for s in ("tech", "financials", "health", "energy", "industrials"):
        cols[f"sector_{s}"] = walk(100, 0.012)
    return pd.DataFrame(cols, index=dates)


def toy_panel(n_dates: int = 400, n_names: int = 30, seed: int = 0):
    """A (date, ticker) panel with one feature, a label and a forward return."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_dates)
    index = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(n_names)]],
                                       names=["date", "ticker"])
    X = pd.DataFrame({"x": rng.normal(size=len(index))}, index=index)
    fwd = pd.Series(rng.normal(size=len(index)), index=index)
    y = (fwd > 0).astype(float)
    return X, y, fwd
