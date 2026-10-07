"""Instrument universe.

Two definitions live here, on purpose.

`liquid_membership` is the research universe: built from Sharadar's
delisting-aware prices, it selects the most liquid common stocks as of each
rebalance date using only data available then. Companies that later failed
are in it for as long as they were liquid, exactly as an investor at the time
would have seen them.

`LARGE_CAP` is today's survivors. Results computed on it are optimistic,
because companies that failed over the period are simply absent. It is kept
for the live daily run — where today's list is legitimately known — and as
the biased comparison the survivorship measurement needs.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

SURVIVORSHIP_BIASED = True  # applies to LARGE_CAP only


def liquid_membership(
    prices: pd.DataFrame,
    size: int = 1000,
    rebalance_days: int = 63,
    min_price: float = 5.0,
    adv_window: int = 60,
    start: str | None = None,
) -> pd.MultiIndex:
    """Point-in-time liquid universe as a (date, ticker) index.

    On each rebalance date the `size` names with the highest trailing average
    dollar volume, among those trading at or above `min_price` unadjusted, are
    selected and held until the next rebalance. Everything the selection
    looks at is known at that date's close.
    """
    dollar_vol = prices.pivot(index="date", columns="ticker", values="DollarVolume")
    adv = dollar_vol.rolling(adv_window, min_periods=adv_window // 2).mean().astype("float32")
    del dollar_vol

    price = prices.pivot(index="date", columns="ticker", values="CloseUnadj")
    eligible = adv.where(price >= min_price)
    del price, adv

    dates = eligible.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]

    rebalance_dates = dates[::rebalance_days]
    pairs = []
    for i, as_of in enumerate(rebalance_dates):
        selected = eligible.loc[as_of].dropna().nlargest(size).index
        end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        held = dates[(dates >= as_of) & ((dates < end) if end is not None else True)]
        pairs.append(pd.MultiIndex.from_product([held, selected], names=["date", "ticker"]))

    membership = pairs[0].append(pairs[1:]) if len(pairs) > 1 else pairs[0]
    logger.info("Universe: %d rebalances, %d distinct names, %d (date, ticker) slots",
                len(rebalance_dates), membership.get_level_values("ticker").nunique(),
                len(membership))
    return membership

# Large-cap US equities across sectors.
LARGE_CAP = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "CRM",
    "AMD", "INTC", "CSCO", "ADBE", "QCOM", "TXN", "IBM", "NOW", "INTU", "MU",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI",
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "MDT", "ISRG",
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "LOW",
    "HD", "MO", "CL", "KMB", "GIS",
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "OXY", "HAL",
    "BA", "CAT", "GE", "HON", "UNP", "UPS", "RTX", "LMT", "DE", "MMM",
    "FDX", "NSC", "EMR", "ETN", "ITW",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
    "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "DOW",
    "AMT", "PLD", "CCI", "SPG", "O", "PSA",
    "DIS", "CMCSA", "NFLX", "T", "VZ", "TMUS", "PARA", "WBD",
]

# ETFs are deliberately NOT panel members.
#
# A cross-sectional target asks whether an instrument beat the field's median.
# A diversified ETF holds dozens of the field's own members, so its return is
# close to their average by construction and it lands near the median almost
# every period. Mixing ETFs into the cross-section therefore adds rows that
# are nearly unrankable, diluting the signal the target is meant to measure.
#
# They belong in the market-state features instead, where breadth and sector
# rotation are exactly what they describe — see sources.yahoo.MARKET_SERIES.
CRYPTO = ["BTC-USD", "ETH-USD"]


def default(include_crypto: bool = False) -> list[str]:
    """The working universe for an experiment run: single names only."""
    tickers = list(LARGE_CAP)
    if include_crypto:
        tickers += CRYPTO
    return tickers


def small(n: int = 60) -> list[str]:
    """A reduced universe for quick iteration, spread across sectors."""
    step = max(1, len(LARGE_CAP) // n)
    return LARGE_CAP[::step][:n]
