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
- returns are reported gross, as the comparison papers report them, and net
  of a proportional trading cost. Each rebalance replaces some fraction of
  each leg; replacing a name is one sale and one purchase, so a period costs
  2 x one-way cost x (long-leg turnover + short-leg turnover). The first
  period buys both legs from cash.
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

    members = day.groupby(["date", "bucket"])["ticker"].agg(frozenset).unstack()
    out["turnover_top"] = _turnover(members[n_buckets - 1])
    out["turnover_bottom"] = _turnover(members[0])
    return out


def _turnover(holdings: pd.Series) -> pd.Series:
    """Fraction of a leg replaced at each rebalance; 1.0 when bought from cash."""
    values, previous = [], None
    for current in holdings:
        if not isinstance(current, frozenset) or not current:
            values.append(np.nan)
            continue
        values.append(1.0 if previous is None else 1 - len(current & previous) / len(current))
        previous = current
    return pd.Series(values, index=holdings.index)


def net_long_short(returns: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Long-short return after a proportional one-way trading cost in basis points."""
    cost = 2 * (cost_bps / 1e4) * (returns["turnover_top"] + returns["turnover_bottom"])
    return returns["long_short"] - cost.fillna(0)


def summarise(returns: pd.DataFrame, horizon: int,
              cost_levels_bps: tuple[float, ...] = (5, 10, 20)) -> dict:
    """Annualised mean, volatility and Sharpe ratio of each leg, gross and net."""
    periods_per_year = TRADING_DAYS / horizon
    out = {"n_periods": len(returns), "years": round(len(returns) / periods_per_year, 1)}

    if "turnover_top" in returns:
        out["turnover_per_rebalance"] = float(
            returns[["turnover_top", "turnover_bottom"]].iloc[1:].mean().mean())
        for bps in cost_levels_bps:
            net = net_long_short(returns, bps).dropna()
            std = net.std() * np.sqrt(periods_per_year)
            out[f"long_short_sharpe_net_{bps:g}bps"] = (
                net.mean() * periods_per_year / std if std > 0 else float("nan"))
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
