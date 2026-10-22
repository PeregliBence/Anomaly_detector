"""The daily live run: predict today, and score past predictions as they mature.

Each day's predictions are written to live/predictions/<date>.csv and
committed. The git history then shows that every forecast existed before its
outcome did, so the live track record cannot be suspected of look-ahead.

Live prices come from yfinance. The production models were trained on
Sharadar data; the features are built by the same code from the same kind of
input (total-return adjusted prices), but small differences between vendors
remain and are a stated limitation. Today's universe is a list fixed from the
Sharadar snapshot — live, the current constituents are legitimately known, so
survivorship bias does not arise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from src import production
from src.config import ROOT
from src.panel import add_cross_sectional_features, build_panel
from src.sources import yahoo

logger = logging.getLogger(__name__)

LIVE_DIR = ROOT / "live"
UNIVERSE_FILE = LIVE_DIR / "universe.csv"
CANDIDATES_FILE = LIVE_DIR / "candidates.csv"
PRED_DIR = LIVE_DIR / "predictions"
SCORECARD_FILE = LIVE_DIR / "scorecard.csv"

# The research universe was re-selected every 63 trading days; live does the
# same on a calendar schedule.
REFRESH_CALENDAR_DAYS = 91
UNIVERSE_SIZE = 1000
CANDIDATE_MULTIPLE = 1.5
MIN_PRICE = 5.0
ADV_WINDOW = 60

# Enough trading days for the slowest feature (252 days) plus warm-up, and
# for scoring predictions up to the longest horizon after they were made.
HISTORY_CALENDAR_DAYS = 600
DOWNLOAD_CHUNK = 100


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def to_yahoo(ticker: str) -> str:
    return ticker.replace(".", "-")


def build_universe(size: int = UNIVERSE_SIZE) -> pd.DataFrame:
    """Freeze today's liquid universe, and a wider candidate list, from Sharadar.

    The candidate list is what the live run later re-ranks from free data,
    once the subscription that produced it has ended.
    """
    from src import universe
    from src.sources import sharadar

    meta = sharadar.equity_meta()
    last = pd.Timestamp(sharadar.load("stocks", columns=["date"])["date"].max())
    prices = sharadar.load_prices(start=str((last - pd.Timedelta(days=200)).date()),
                                  tickers=meta.index.tolist())

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    n_candidates = int(size * CANDIDATE_MULTIPLE)
    for path, n in ((UNIVERSE_FILE, size), (CANDIDATES_FILE, n_candidates)):
        members = universe.todays_list_membership(prices, size=n).get_level_values("ticker").unique()
        pd.DataFrame({
            "ticker": [to_yahoo(t) for t in members],
            "sharadar_ticker": list(members),
            "as_of": str(last.date()),
        }).sort_values("ticker").to_csv(path, index=False)
        logger.info("%s: %d names as of %s", path.name, len(members), last.date())
    return pd.read_csv(UNIVERSE_FILE)


def load_universe() -> list[str]:
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError("live/universe.csv not found. Run `python -m src.cli live-universe`.")
    return pd.read_csv(UNIVERSE_FILE)["ticker"].tolist()


def universe_is_stale(today: pd.Timestamp | None = None) -> bool:
    if not UNIVERSE_FILE.exists():
        return False
    as_of = pd.Timestamp(pd.read_csv(UNIVERSE_FILE, nrows=1)["as_of"].iloc[0])
    return ((today or pd.Timestamp.today()) - as_of).days >= REFRESH_CALENDAR_DAYS


def refresh_universe(size: int = UNIVERSE_SIZE) -> pd.DataFrame:
    """Re-rank the candidate list by recent dollar volume, as the research did.

    Same rule as the research universe: trailing 60-day average dollar volume,
    price at least $5, top `size`. Only names on the candidate list can enter
    — a stated limitation, since a company listed after the snapshot never
    appears on it.
    """
    candidates = pd.read_csv(CANDIDATES_FILE)
    start = str((pd.Timestamp.today() - pd.Timedelta(days=ADV_WINDOW * 2)).date())
    frames = download_prices(candidates["ticker"].tolist(), start)

    rows = []
    for ticker, df in frames.items():
        recent = df.tail(ADV_WINDOW)
        if len(recent) >= ADV_WINDOW // 2 and recent["Close"].iloc[-1] >= MIN_PRICE:
            rows.append((ticker, recent["DollarVolume"].mean(), recent.index[-1]))
    ranked = pd.DataFrame(rows, columns=["ticker", "adv", "last"])
    latest = ranked["last"].max()
    ranked = ranked[ranked["last"] == latest].nlargest(size, "adv")

    out = candidates.set_index("ticker").loc[ranked["ticker"], ["sharadar_ticker"]].reset_index()
    out["as_of"] = str(latest.date())
    out = out.sort_values("ticker")
    out.to_csv(UNIVERSE_FILE, index=False)
    logger.info("Universe refreshed from %d candidates: %d names as of %s",
                len(candidates), len(out), latest.date())
    return out


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def download_prices(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    """Daily OHLCV per ticker in the column layout the feature code expects."""
    frames: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), DOWNLOAD_CHUNK):
        chunk = tickers[i:i + DOWNLOAD_CHUNK]
        raw = yf.download(chunk, start=start, auto_adjust=True, progress=False,
                          group_by="ticker", threads=True)
        for t in chunk:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            if df.empty:
                continue
            df = df[df["Close"] > 0].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            df["DollarVolume"] = df["Close"] * df["Volume"]
            df["CloseUnadj"] = df["Close"]
            frames[t] = df
        logger.info("Downloaded %d/%d", min(i + DOWNLOAD_CHUNK, len(tickers)), len(tickers))
    return frames


# ---------------------------------------------------------------------------
# Predict and resolve
# ---------------------------------------------------------------------------

DERIVED_FILE = production.MODELS_DIR / "drift_plus_rank.json"


def derived_coefficients() -> dict:
    """Intercept and slope of the drift-plus-rank forecast, per horizon."""
    if not DERIVED_FILE.exists():
        return {}
    return json.loads(DERIVED_FILE.read_text())


def _complete_rows(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The rows with every input the model was trained on.

    A column missing for every name — a market series whose source failed —
    is an outage, not something to score around, so it stops the run.
    """
    X = panel.reindex(columns=columns)
    empty = [c for c in columns if X[c].isna().all()]
    if empty:
        raise RuntimeError(f"No data today for model inputs {empty}; not predicting.")
    return X.dropna()


def predict(frames: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    """Score every universe name on `as_of` with each production horizon."""
    tickers = [t for t, df in frames.items() if as_of in df.index]
    keep = pd.MultiIndex.from_product([[as_of], tickers], names=["date", "ticker"])
    market = yahoo.load_market_state(start="2003-01-01")

    panel = build_panel(frames, horizons=[], market=market, keep=keep,
                        seq_len=production.SEQ_LEN, min_rows=260)
    panel = add_cross_sectional_features(panel)

    out = pd.DataFrame(index=panel.index.get_level_values("ticker"))
    for h in production.HORIZONS:
        bundle = production.load(h)
        X = _complete_rows(panel, bundle["columns"])
        s = production.score(bundle, X).droplevel("date")
        out[f"score_h{h}"] = s
        out[f"rank_h{h}"] = s.rank(ascending=False, method="first")

        # Absolute direction, where an absolute model has been trained.
        if production.bundle_path(h, "absolute").exists():
            abs_bundle = production.load(h, "absolute")
            X = _complete_rows(panel, abs_bundle["columns"])
            p = production.score(abs_bundle, X).droplevel("date")
            out[f"prob_up_h{h}"] = p
            out[f"dir_h{h}"] = pd.Series(np.where(p > 0.5, "up", "down"), index=p.index)

        # Derived absolute direction: long-run share of rises shifted by the
        # relative rank — the same forecast without any market timing.
        coef = derived_coefficients().get(str(h))
        if coef is not None:
            p = 1 / (1 + np.exp(-(coef["intercept"] + coef["slope"] * (s - 0.5))))
            out[f"prob_up_dr_h{h}"] = p
            out[f"dir_dr_h{h}"] = pd.Series(np.where(p > 0.5, "up", "down"), index=p.index)

    out = out.dropna(how="all").sort_values(f"rank_h{production.HORIZONS[0]}")
    out.index.name = "ticker"
    return out


def _direction_scores(prob_up: pd.Series, fwd: pd.Series) -> dict:
    """How the absolute forecasts fared, next to the 'always up' rule."""
    joined = pd.concat([prob_up.rename("p"), fwd.rename("fwd")], axis=1, join="inner").dropna()
    rose = (joined["fwd"] > 0).astype(float)
    return {
        "direction_n": len(joined),
        "direction_accuracy": float(((joined["p"] > 0.5) == (rose > 0)).mean()),
        "always_up_accuracy": float(rose.mean()),
        "predicted_up_share": float((joined["p"] > 0.5).mean()),
        "brier": float(((joined["p"] - rose) ** 2).mean()),
    }


def resolve(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Score every past prediction whose horizon has elapsed and is not yet scored."""
    closes = pd.DataFrame({t: df["Close"] for t, df in frames.items()}).sort_index()
    calendar = closes.index

    done = pd.read_csv(SCORECARD_FILE) if SCORECARD_FILE.exists() else pd.DataFrame()
    seen = set(zip(done.get("date", []), done.get("horizon", [])))
    rows = []

    for path in sorted(PRED_DIR.glob("*.csv")):
        made = pd.Timestamp(path.stem)
        if made not in calendar:
            continue
        pos = calendar.get_loc(made)
        preds = pd.read_csv(path, index_col="ticker")

        for h in production.HORIZONS:
            col = f"score_h{h}"
            if (path.stem, h) in seen or col not in preds or pos + h >= len(calendar):
                continue
            fwd = np.log(closes.iloc[pos + h] / closes.iloc[pos])
            joined = pd.concat([preds[col], fwd.rename("fwd")], axis=1, join="inner").dropna()
            if len(joined) < 20:
                continue
            decile = pd.qcut(joined[col].rank(method="first"), 10, labels=False)
            row = {
                "date": path.stem,
                "horizon": h,
                "resolved_on": str(calendar[pos + h].date()),
                "n": len(joined),
                "ic": stats.spearmanr(joined[col], joined["fwd"]).statistic,
                "top_decile_return": joined["fwd"][decile == 9].mean(),
                "bottom_decile_return": joined["fwd"][decile == 0].mean(),
            }
            if f"prob_up_h{h}" in preds:
                row.update(_direction_scores(preds[f"prob_up_h{h}"], fwd))
            if f"prob_up_dr_h{h}" in preds:
                derived = _direction_scores(preds[f"prob_up_dr_h{h}"], fwd)
                row.update({f"dr_{k}": v for k, v in derived.items() if k != "always_up_accuracy"})
                row.setdefault("always_up_accuracy", derived["always_up_accuracy"])
            rows.append(row)

    if rows:
        new = pd.DataFrame(rows)
        new["spread"] = new["top_decile_return"] - new["bottom_decile_return"]
        done = pd.concat([done, new], ignore_index=True).sort_values(["date", "horizon"])
        done.to_csv(SCORECARD_FILE, index=False)
        logger.info("Scored %d matured predictions", len(rows))
    return done


def daily() -> dict:
    """One live step: score matured forecasts, then predict for the latest close."""
    refreshed = False
    if universe_is_stale() and CANDIDATES_FILE.exists():
        refresh_universe()
        refreshed = True
    tickers = load_universe()
    start = str((pd.Timestamp.today() - pd.Timedelta(days=HISTORY_CALENDAR_DAYS)).date())
    frames = download_prices(tickers, start)
    if not frames:
        raise RuntimeError("No prices downloaded; Yahoo may be blocking this machine.")

    counts = pd.Series([df.index.max() for df in frames.values()]).value_counts()
    as_of = counts.index[0]  # the latest date most of the universe has a close for
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRED_DIR / f"{as_of.date()}.csv"

    # Scored first, so a failed prediction does not cost the day's evaluation.
    scorecard = resolve(frames)

    # Never rewrite a committed forecast. On a market holiday the latest close
    # is a day already predicted; regenerating it from revised vendor data
    # would alter a forecast after the fact and void the timestamp guarantee.
    if out_path.exists():
        n_predictions, status = len(pd.read_csv(out_path)), "already predicted"
    else:
        preds = predict(frames, as_of)
        preds.round(6).to_csv(out_path)
        n_predictions, status = len(preds), "new"

    return {"as_of": str(as_of.date()), "status": status, "n_predictions": n_predictions,
            "file": str(out_path.relative_to(ROOT)), "scored_total": len(scorecard),
            "universe_refreshed": refreshed}
