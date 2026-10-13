"""Panel targets and the point-in-time joins behind the alternative features."""

import numpy as np
import pandas as pd

from src.alt_features import _asof, _windowed_sum
from src.config import MIN_NAMES_PER_DATE
from src.panel import build_panel, cross_sectional_target


def test_cross_sectional_target_is_half_ones_per_date(frames):
    panel = build_panel(frames, horizons=[5])
    y = cross_sectional_target(panel["fwd_return_5"])
    share = y.groupby(level="date").mean().dropna()
    assert share.between(0.4, 0.6).all()


def test_cross_sectional_target_is_unlabelled_for_a_thin_field(frames):
    few = dict(list(frames.items())[: MIN_NAMES_PER_DATE - 1])
    panel = build_panel(few, horizons=[5])
    assert cross_sectional_target(panel["fwd_return_5"]).isna().all()


def test_panel_keeps_only_member_rows_but_uses_full_history(frames):
    dates = next(iter(frames.values())).index
    keep = pd.MultiIndex.from_product([dates[400:410], ["T00", "T01"]], names=["date", "ticker"])
    panel = build_panel(frames, horizons=[5], keep=keep)
    assert set(panel.index) == set(keep)
    # 12-1 momentum needs a year of history; it exists because features were
    # computed before the rows were filtered.
    assert panel["mom_12_1"].notna().all()


def keys(*rows):
    return pd.DataFrame(rows, columns=["date", "ticker"]).astype({"date": "datetime64[ns]"})


def test_asof_hides_a_value_until_it_is_available():
    table = pd.DataFrame({"ticker": ["A"], "available": pd.to_datetime(["2020-01-10"]), "v": [1.0]})
    out = _asof(keys(("2020-01-09", "A"), ("2020-01-10", "A"), ("2020-02-01", "A")),
                table, "available", ["v"])
    assert np.isnan(out["v"].iloc[0])
    assert out["v"].iloc[1:].tolist() == [1.0, 1.0]


def test_asof_returns_rows_in_the_callers_order_despite_ties():
    table = pd.DataFrame({"ticker": ["A", "B"], "available": pd.to_datetime(["2020-01-01"] * 2),
                          "v": [1.0, 2.0]})
    k = keys(("2020-01-05", "B"), ("2020-01-05", "A"), ("2020-01-03", "B"))
    out = _asof(k, table, "available", ["v"])
    assert out["v"].tolist() == [2.0, 1.0, 2.0]
    assert list(out.index.get_level_values("ticker")) == ["B", "A", "B"]


def test_asof_drops_a_stale_value():
    table = pd.DataFrame({"ticker": ["A"], "available": pd.to_datetime(["2019-01-01"]), "v": [1.0]})
    out = _asof(keys(("2020-06-01", "A")), table, "available", ["v"], tolerance=pd.Timedelta(days=400))
    assert np.isnan(out["v"].iloc[0])


def test_windowed_sum_counts_only_events_inside_the_window():
    events = pd.DataFrame({
        "ticker": ["A", "A", "A", "B"],
        "available": pd.to_datetime(["2020-01-01", "2020-03-01", "2020-03-01", "2020-03-01"]),
        "n": [1.0, 1.0, 1.0, 1.0],
    })
    out = _windowed_sum(keys(("2020-03-15", "A"), ("2020-05-15", "A"), ("2020-03-15", "B")),
                        events, "available", ["n"], window=pd.Timedelta(days=60))
    # 2020-03-15 looks back to 2020-01-15: two same-day events, not the January one.
    assert out["n"].tolist() == [2.0, 0.0, 1.0]
