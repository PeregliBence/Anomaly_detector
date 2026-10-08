"""Features from fundamentals, earnings, corporate events and insider trades.

Every one of these is only knowable from the day it became public, which is
rarely the day it refers to: a quarter's accounts appear 25-90 days after the
quarter ends. Each source is therefore aligned on its publication date, with
a margin:

- Fundamentals (Sharadar SF1, as-reported dimensions) from the day after the
  filing date. As-reported values are never revised, so a later restatement
  cannot leak backwards. A filing older than 400 days counts as missing.
- Earnings reaction from two trading days after the announcement, so the
  three-day window around it has fully closed.
- Other 8-K events seven days after their recorded date: an 8-K may be filed
  up to four business days after the event, and the table's date may be the
  event's rather than the filing's.
- Insider trades from the day after their Form 4 filing date.

Every feature enters the model as a within-date percentile rank. A name with
no data gets the neutral rank 0.5, plus a missing-data flag, as Gu, Kelly and
Xiu (2020) do — no row is dropped for lacking an alternative source.

Live data for these sources is not available after the subscription ends, so
they are used to measure what alternative data adds, not in production.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.sources import sharadar

logger = logging.getLogger(__name__)

STALE_FILING = pd.Timedelta(days=400)
EVENT_DELAY = pd.Timedelta(days=7)
DAY = pd.Timedelta(days=1)

EARNINGS_CODE = "22"
RED_FLAG_CODES = {"13", "26", "31", "36", "40", "41", "42"}  # bankruptcy, impairment,
# delisting notice, late filing, auditor change (x2), non-reliance on past statements
DEPARTURE_CODE = "52"


def _asof(keys: pd.DataFrame, table: pd.DataFrame, on: str, cols: list[str],
          tolerance: pd.Timedelta | None = None) -> pd.DataFrame:
    """Latest `table` row per ticker with `on` <= each key date.

    The result is aligned row for row with `keys` and indexed by the keys'
    (date, ticker). merge_asof needs both sides sorted, so the original row
    order is carried through and restored rather than trusted to survive.
    """
    left = keys[["date", "ticker"]].reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    left = left.sort_values("date", kind="stable")
    right = table.dropna(subset=[on]).sort_values(on, kind="stable")[["ticker", on] + cols]
    merged = pd.merge_asof(left, right, left_on="date", right_on=on, by="ticker",
                           direction="backward", tolerance=tolerance)
    merged = merged.sort_values("_row")
    index = pd.MultiIndex.from_frame(keys[["date", "ticker"]].reset_index(drop=True))
    return pd.DataFrame(merged[cols].values, index=index, columns=cols)


def _windowed_sum(keys: pd.DataFrame, events: pd.DataFrame, on: str, cols: list[str],
                  window: pd.Timedelta) -> pd.DataFrame:
    """Per ticker, sum of `cols` over events with `on` in (date - window, date].

    Exact and cheap: a running total looked up at both ends of the window.
    Events sharing a ticker and date are summed first, so the running total
    at that date is unambiguous.
    """
    ev = events.groupby(["ticker", on], as_index=False)[cols].sum().sort_values(["ticker", on])
    cum_cols = [f"_cum_{c}" for c in cols]
    ev[cum_cols] = ev.groupby("ticker")[cols].cumsum().values

    now = _asof(keys, ev, on, cum_cols)
    before = _asof(keys.assign(date=keys["date"] - window), ev, on, cum_cols)
    out = now.fillna(0).values - before.fillna(0).values
    return pd.DataFrame(out, index=now.index, columns=cols)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def fundamental_features(keys: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    art = sharadar.load(
        "fundamentals",
        columns=["ticker", "dimension", "date", "reportperiod", "assets", "equity",
                 "netinc", "revenue", "ncfo", "gp", "debt"],
        filters=[("dimension", "==", "ART"), ("ticker", "in", tickers)],
    )
    art["available"] = pd.to_datetime(art["date"]) + DAY
    art = art.sort_values(["ticker", "reportperiod"])
    art["assets_lag4"] = art.groupby("ticker")["assets"].shift(4)

    daily = sharadar.load("daily", columns=["ticker", "date", "marketcap"],
                          filters=[("ticker", "in", tickers)])
    daily["date"] = pd.to_datetime(daily["date"])

    f = _asof(keys, art, "available",
              ["assets", "equity", "netinc", "revenue", "ncfo", "gp", "debt", "assets_lag4"],
              tolerance=STALE_FILING)
    mcap = keys.merge(daily, on=["date", "ticker"], how="left").set_index(["date", "ticker"])["marketcap"]
    mcap = mcap.reindex(f.index) * 1e6  # DAILY reports millions of dollars

    def ratio(a, b):
        return (a / b.where(b > 0)).replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame({
        "size": np.log(mcap.where(mcap > 0)),
        "book_to_market": ratio(f["equity"], mcap),
        "earnings_yield": ratio(f["netinc"], mcap),
        "sales_to_price": ratio(f["revenue"], mcap),
        "cashflow_yield": ratio(f["ncfo"], mcap),
        "gross_profitability": ratio(f["gp"], f["assets"]),
        "roe": ratio(f["netinc"], f["equity"]),
        "accruals": ratio(f["netinc"] - f["ncfo"], f["assets"]),
        "leverage": ratio(f["debt"], f["assets"]),
        "asset_growth": ratio(f["assets"], f["assets_lag4"]) - 1,
        "fund_missing": f["assets"].isna().astype(float),
    }, index=f.index)


def earnings_surprise(keys: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Standardised unexpected earnings: seasonal EPS change over its own volatility."""
    arq = sharadar.load("fundamentals", columns=["ticker", "dimension", "date", "reportperiod", "eps"],
                        filters=[("dimension", "==", "ARQ"), ("ticker", "in", tickers)])
    arq = arq.sort_values(["ticker", "reportperiod"])
    change = arq["eps"] - arq.groupby("ticker")["eps"].shift(4)
    arq["eps_change"] = change
    # Scale by the volatility of *past* changes only.
    arq["sue"] = change / arq.assign(c=change).groupby("ticker")["c"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=4).std())
    arq["available"] = pd.to_datetime(arq["date"]) + DAY
    return _asof(keys, arq, "available", ["sue"], tolerance=STALE_FILING)


def earnings_reaction(keys: pd.DataFrame, prices: pd.DataFrame, market: pd.DataFrame,
                      tickers: list[str]) -> pd.DataFrame:
    """Return over the three days around each earnings release, net of the market."""
    ev = sharadar.load("events", columns=["ticker", "date", "eventcodes"],
                       filters=[("ticker", "in", tickers)])
    ev = ev[ev["eventcodes"].astype(str).str.split("|").map(lambda c: EARNINGS_CODE in c)]
    ev["date"] = pd.to_datetime(ev["date"])
    spx = np.log(market["spx"]).reindex(pd.DatetimeIndex(prices["date"].unique()).sort_values()).ffill()

    rows = []
    by_ticker = {t: g for t, g in ev.groupby("ticker")}
    for ticker, px in prices.groupby("ticker", sort=False):
        if ticker not in by_ticker:
            continue
        dates = px["date"].values
        logp = np.log(px["Close"].values)
        for e in by_ticker[ticker]["date"].values:
            i = np.searchsorted(dates, e)
            if i - 2 < 0 or i + 2 >= len(dates):
                continue
            start, end = pd.Timestamp(dates[i - 2]), pd.Timestamp(dates[i + 1])
            stock = logp[i + 1] - logp[i - 2]
            mkt = spx.get(end, np.nan) - spx.get(start, np.nan)
            rows.append((ticker, pd.Timestamp(e), pd.Timestamp(dates[i + 2]), stock - mkt))

    er = pd.DataFrame(rows, columns=["ticker", "announced", "available", "ear"])
    out = _asof(keys, er, "available", ["ear", "announced"], tolerance=pd.Timedelta(days=120))
    announced = pd.to_datetime(out["announced"])
    days = (out.index.get_level_values("date") - pd.DatetimeIndex(announced)).days
    return pd.DataFrame({"ear": pd.to_numeric(out["ear"]), "days_since_earnings": days},
                        index=out.index)


def event_flags(keys: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    ev = sharadar.load("events", columns=["ticker", "date", "eventcodes"],
                       filters=[("ticker", "in", tickers)])
    codes = ev["eventcodes"].astype(str).str.split("|")
    ev["red_flag"] = codes.map(lambda c: float(bool(RED_FLAG_CODES & set(c))))
    ev["departure"] = codes.map(lambda c: float(DEPARTURE_CODE in c))
    ev["available"] = pd.to_datetime(ev["date"]) + EVENT_DELAY
    ev = ev[(ev["red_flag"] > 0) | (ev["departure"] > 0)]
    red = _windowed_sum(keys, ev, "available", ["red_flag"], pd.Timedelta(days=180))
    dep = _windowed_sum(keys, ev, "available", ["departure"], pd.Timedelta(days=90))
    return pd.DataFrame({"red_flags_180": red["red_flag"], "departures_90": dep["departure"]})


def insider_features(keys: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    ins = sharadar.load("insiders", columns=["ticker", "date", "transactioncode", "transactionvalue"],
                        filters=[("ticker", "in", tickers), ("transactioncode", "in", ["P", "S"])])
    ins["available"] = pd.to_datetime(ins["date"]) + DAY
    value = pd.to_numeric(ins["transactionvalue"], errors="coerce").fillna(0).abs()
    ins["buy_n"] = (ins["transactioncode"] == "P").astype(float)
    ins["sell_n"] = (ins["transactioncode"] == "S").astype(float)
    ins["net_value"] = np.where(ins["transactioncode"] == "P", value, -value)

    w90 = _windowed_sum(keys, ins, "available", ["buy_n", "sell_n"], pd.Timedelta(days=90))
    w180 = _windowed_sum(keys, ins, "available", ["net_value"], pd.Timedelta(days=180))
    return pd.DataFrame({
        "insider_buys_90": w90["buy_n"],
        "insider_net_count_90": w90["buy_n"] - w90["sell_n"],
        "insider_net_value_180": w180["net_value"],
    })


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def add_alternative_features(panel: pd.DataFrame, prices: pd.DataFrame,
                             market: pd.DataFrame) -> pd.DataFrame:
    """Join every alternative source onto the panel as within-date ranks."""
    keys = panel.index.to_frame(index=False)
    keys["date"] = pd.to_datetime(keys["date"])
    tickers = keys["ticker"].unique().tolist()

    parts = [
        fundamental_features(keys, tickers),
        earnings_surprise(keys, tickers),
        earnings_reaction(keys, prices, market, tickers),
        event_flags(keys, tickers),
        insider_features(keys, tickers),
    ]
    raw = pd.concat([p.reindex(panel.index) for p in parts], axis=1)

    # Insider value is only comparable across firms relative to their size.
    if "size" in raw:
        raw["insider_net_value_180"] = raw["insider_net_value_180"] / np.exp(raw["size"])

    out = panel
    by_date = raw.groupby(level="date")
    for col in raw.columns:
        if col == "fund_missing":
            out["alt_fund_missing"] = raw[col].fillna(1.0).astype("float32")
            continue
        out[f"alt_{col}"] = by_date[col].rank(pct=True).fillna(0.5).astype("float32")

    coverage = raw.notna().mean().round(3).to_dict()
    logger.info("Alternative features added; coverage: %s", coverage)
    return out
