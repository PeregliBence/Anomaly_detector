"""Instrument universe.

Interim definition until Sharadar's delisting-aware data is available. This
list holds today's large-cap survivors, which means results computed on it are
optimistic: companies that failed over the period are simply absent, so the
sample is conditioned on survival.

That bias is one of the errors the thesis measures, so the list is explicitly
marked rather than quietly used. `SURVIVORSHIP_BIASED` exists to be asserted
against in the final run.
"""

from __future__ import annotations

SURVIVORSHIP_BIASED = True

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

# Sector and asset-class ETFs. These widen the cross-section beyond single
# stocks and give the panel instruments whose risk is not idiosyncratic.
ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "MDY",
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
    "EFA", "EEM", "EWJ", "EWG", "EWU", "FXI", "INDA",
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG",
    "GLD", "SLV", "USO", "DBC", "UNG",
    "VNQ", "XBI", "SMH", "ITB", "KRE",
]

CRYPTO = ["BTC-USD", "ETH-USD"]


def default(include_etfs: bool = True, include_crypto: bool = False) -> list[str]:
    """The working universe for an experiment run."""
    tickers = list(LARGE_CAP)
    if include_etfs:
        tickers += ETFS
    if include_crypto:
        tickers += CRYPTO
    return tickers


def small(n: int = 60) -> list[str]:
    """A reduced universe for quick iteration, spread across sectors."""
    step = max(1, len(LARGE_CAP) // n)
    return LARGE_CAP[::step][:n] + ETFS[:10]
