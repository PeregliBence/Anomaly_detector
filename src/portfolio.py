"""Long-short portfolio statistics, in the form the literature reports them.

Papers in this field rarely report information coefficients; they report the
Sharpe ratio of a long-top-decile, short-bottom-decile portfolio. To put this
work beside them, predictions are turned into that portfolio:

- each rebalance date, buy the top decile and sell the bottom decile by score,
  equally weighted;
- rebalance every `horizon` trading days, holding each portfolio for exactly
  its forecast horizon. Rebalancing daily on a multi-day horizon would stack
  overlapping positions whose returns share most of their days, and a Sharpe
  ratio computed on those overlapping returns is inflated;
- returns are gross of trading costs, as the comparison papers report them.
  Weekly turnover at the 5-day horizon would cost real money, which is
  stated as a limitation rather than modelled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def long_short_returns(preds: pd.DataFrame, horizon: int, n_buckets: int = 10,
                       min_names: int = 50, offset: int = 0) -> pd.DataFrame:
    """Per-period simple returns of the top, bottom and long-short portfolios.

    `preds` needs columns score and fwd_return (a log return over `horizon`),
    indexed by (date, ticker). `offset` shifts which dates are rebalance days,
    so results can be checked for dependence on the starting point.
    """
    usable = preds.dropna(subset=["score", "fwd_return"])
    dates = usable.index.get_level_values("date").unique().sort_values()
    rebalance = dates[offset::horizon]

    day = usable[usable.index.get_level_values("date").isin(rebalance)].reset_index()
    counts = day.groupby("date")["score"].transform("size")
    day = day[counts >= min_names]
    if day.empty:
        return pd.DataFrame(columns=["top", "bottom", "long_short", "market"])

    pct = day.groupby("date")["score"].rank(method="first", pct=True)
    day["bucket"] = np.minimum((pct * n_buckets).apply(np.ceil).astype(int) - 1, n_buckets - 1)
    day["simple"] = np.expm1(day["fwd_return"])

    by_bucket = day.groupby(["date", "bucket"])["simple"].mean().unstack()
    out = pd.DataFrame({
        "top": by_bucket[n_buckets - 1],
        "bottom": by_bucket[0],
        "market": day.groupby("date")["simple"].mean(),
    })
    out["long_short"] = out["top"] - out["bottom"]
    return out


def summarise(returns: pd.DataFrame, horizon: int) -> dict:
    """Annualised mean, volatility and Sharpe ratio of each leg."""
    periods_per_year = TRADING_DAYS / horizon
    out = {"n_periods": len(returns), "years": round(len(returns) / periods_per_year, 1)}
    for col in ("long_short", "top", "bottom", "market"):
        r = returns[col].dropna()
        mean, std = r.mean() * periods_per_year, r.std() * np.sqrt(periods_per_year)
        out[f"{col}_ann_return"] = mean
        out[f"{col}_ann_vol"] = std
        out[f"{col}_sharpe"] = mean / std if std > 0 else float("nan")
    ls = returns["long_short"].dropna()
    wealth = (1 + ls).cumprod()
    out["long_short_max_drawdown"] = float((wealth / wealth.cummax() - 1).min())
    out["long_short_hit_rate"] = float((ls > 0).mean())
    return out


def sharpe_over_offsets(preds: pd.DataFrame, horizon: int, max_offsets: int = 20) -> pd.Series:
    """Long-short Sharpe for every possible rebalance offset.

    With a 20-day hold there are 20 equally valid choices of rebalance day;
    reporting only one invites picking the luckiest. The spread across them
    shows how much of a Sharpe ratio is the calendar rather than the model.
    Long horizons are sampled at `max_offsets` evenly spaced offsets.
    """
    offsets = np.unique(np.linspace(0, horizon - 1, min(horizon, max_offsets)).astype(int))
    values = {}
    for offset in offsets:
        stats = summarise(long_short_returns(preds, horizon, offset=offset), horizon)
        values[offset] = stats["long_short_sharpe"]
    return pd.Series(values, name="long_short_sharpe")
