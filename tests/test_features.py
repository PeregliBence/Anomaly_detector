"""Features at date t may depend only on data up to t."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_prices
from src.features import build_features, build_market_features, forward_return
from src.panel import sequence_columns


def assert_same_up_to(full: pd.DataFrame, truncated: pd.DataFrame):
    common = truncated.index
    pd.testing.assert_frame_equal(full.loc[common], truncated, check_exact=False, rtol=1e-9)


@pytest.mark.parametrize("cut", [300, 450, 599])
def test_price_features_do_not_change_when_the_future_is_removed(cut):
    df = make_prices(n_days=600)
    assert_same_up_to(build_features(df), build_features(df.iloc[:cut]))


@pytest.mark.parametrize("cut", [400, 700])
def test_market_features_do_not_change_when_the_future_is_removed(market, cut):
    assert_same_up_to(build_market_features(market), build_market_features(market.iloc[:cut]))


def test_sequence_columns_are_plain_lags():
    df = make_prices()
    feats = build_features(df)
    seq = sequence_columns(df, feats, seq_len=5)
    pd.testing.assert_series_equal(seq["seq_ret_03"], feats["log_return"].shift(3), check_names=False)


def test_forward_return_looks_exactly_horizon_days_ahead():
    df = make_prices()
    fwd = forward_return(df, 5)
    expected = np.log(df["Close"].iloc[15] / df["Close"].iloc[10])
    assert fwd.iloc[10] == pytest.approx(expected)
    assert fwd.iloc[-5:].isna().all()


def test_delisted_tail_resolves_at_the_last_price():
    df = make_prices(n_days=100)
    fwd = forward_return(df, 5, delisted=True)
    last = df["Close"].iloc[-1]
    assert fwd.notna().all()
    assert fwd.iloc[-3] == pytest.approx(np.log(last / df["Close"].iloc[-3]))


def test_negative_oil_price_does_not_produce_missing_values(market):
    m = market.copy()
    m.iloc[500, m.columns.get_loc("oil")] = -37.63
    out = build_market_features(m)
    assert out["oil_momentum_60"].iloc[500] < -1          # an extreme collapse, not a gap
    assert np.isfinite(out["oil_momentum_60"].iloc[560])  # 60 days later the lag is defined too
