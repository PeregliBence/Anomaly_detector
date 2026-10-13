"""Scoring, significance and portfolio statistics."""

import numpy as np
import pandas as pd
import pytest

from conftest import toy_panel
from src import portfolio
from src.metrics import newey_west_tstat, rank_ic, wilson_interval


def preds_with(score_fn):
    X, _, fwd = toy_panel()
    return pd.DataFrame({"score": score_fn(fwd), "fwd_return": fwd}, index=X.index)


def test_rank_ic_is_one_for_a_perfect_ranking_and_minus_one_for_its_reverse():
    assert rank_ic(preds_with(lambda f: f)).mean() == pytest.approx(1.0)
    assert rank_ic(preds_with(lambda f: -f)).mean() == pytest.approx(-1.0)


def test_newey_west_t_is_calibrated_on_noise():
    rng = np.random.default_rng(0)
    ts = [newey_west_tstat(pd.Series(rng.normal(size=500)), lags=10)[0] for _ in range(300)]
    assert 0.85 < np.std(ts) < 1.2
    assert abs(np.mean(ts)) < 0.2


def test_wilson_interval_brackets_the_observed_rate():
    lo, hi = wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_perfect_scores_give_a_positive_spread_and_full_rebalance_dates():
    preds = preds_with(lambda f: f)
    returns = portfolio.long_short_returns(preds, horizon=5, min_names=10)
    dates = preds.index.get_level_values("date").unique().sort_values()
    assert list(returns.index) == list(dates[::5])
    assert (returns["long_short"] > 0).all()


def test_turnover_is_one_from_cash_and_zero_when_holdings_repeat():
    X, _, fwd = toy_panel()
    tickers = X.index.get_level_values("ticker")
    fixed = pd.Series(tickers.str[1:].astype(float), index=X.index)  # same order every day
    preds = pd.DataFrame({"score": fixed, "fwd_return": fwd})
    returns = portfolio.long_short_returns(preds, horizon=5, min_names=10)
    assert returns["turnover_top"].iloc[0] == 1.0
    assert (returns["turnover_top"].iloc[1:] == 0).all()


def test_costs_reduce_returns_by_turnover_times_rate():
    returns = pd.DataFrame({"long_short": [0.01, 0.01], "turnover_top": [1.0, 0.5],
                            "turnover_bottom": [1.0, 0.5]})
    net = portfolio.net_long_short(returns, cost_bps=10)
    assert net.tolist() == pytest.approx([0.01 - 2 * 0.001 * 2, 0.01 - 2 * 0.001 * 1])
