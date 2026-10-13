"""Run one experiment: a universe, a set of horizons, a target, a set of models."""

from __future__ import annotations

import logging
from functools import partial
from typing import Sequence

import pandas as pd

from src import models as model_lib
from src.config import MIN_NAMES_PER_DATE, REFIT_EVERY, START_DATE
from src.metrics import evaluate
from src.panel import add_cross_sectional_features, apply_universe_filter, build_panel, split_xy
from src.sources import yahoo
from src.validation import tune_on_train, walk_forward

logger = logging.getLogger(__name__)

# Long enough for the slowest feature (12-1 momentum needs 252 days) to warm
# up before the first in-sample date.
WARMUP_YEARS = 2


def _warmup_start(start: str) -> str:
    return (pd.Timestamp(start) - pd.DateOffset(years=WARMUP_YEARS)).strftime("%Y-%m-%d")


UNIVERSE_MODES = ("point_in_time", "survivors", "todays_list")


def load_sharadar_panel(
    horizons: Sequence[int],
    start: str = START_DATE,
    size: int = 1000,
    rebalance_days: int = 63,
    universe_mode: str = "point_in_time",
    seq_len: int = 0,
    alt_features: bool = False,
) -> pd.DataFrame:
    """The research panel from Sharadar data.

    universe_mode:
      point_in_time — liquid names as of each rebalance, delisted included.
                      The correct universe and the default.
      survivors     — the same selection with every later-delisted company
                      removed, isolating the effect of the delistings.
      todays_list   — today's liquid names applied to all history, as a study
                      built on a current screener list would do.
    The last two exist only to measure survivorship bias.
    """
    from src import universe
    from src.sources import sharadar

    if universe_mode not in UNIVERSE_MODES:
        raise ValueError(f"universe_mode must be one of {UNIVERSE_MODES}")

    warmup = _warmup_start(start)
    meta = sharadar.equity_meta()
    tickers = meta.index
    if universe_mode == "survivors":
        tickers = meta.index[meta["isdelisted"] != "Y"]
    logger.info("Eligible common stocks (%s): %d", universe_mode, len(tickers))

    prices = sharadar.load_prices(start=warmup, tickers=tickers.tolist())
    logger.info("Loaded %d price rows", len(prices))

    if universe_mode == "todays_list":
        membership = universe.todays_list_membership(prices, size=size, start=start)
    else:
        membership = universe.liquid_membership(
            prices, size=size, rebalance_days=rebalance_days, start=start
        )
    members = set(membership.get_level_values("ticker"))
    prices = prices[prices["ticker"].isin(members)]

    frames = (
        (ticker, group.drop(columns="ticker").set_index("date"))
        for ticker, group in prices.groupby("ticker", sort=False)
    )
    delisted = set(meta.index[meta["isdelisted"] == "Y"])
    market = yahoo.load_market_state(start=warmup)

    panel = build_panel(
        frames, horizons=horizons, market=market, delisted=delisted,
        keep=membership, seq_len=seq_len,
    )
    if panel.empty:
        return panel
    if alt_features:
        from src.alt_features import add_alternative_features
        panel = add_alternative_features(panel, prices, market)
    del prices
    return add_cross_sectional_features(panel)


def load_panel(
    tickers: list[str],
    horizon: int,
    start: str = START_DATE,
    end: str | None = None,
    universe_size: int | None = None,
) -> pd.DataFrame:
    """yfinance panel over a fixed ticker list. Survivor-only — see universe.py."""
    frames = yahoo.load_many(tickers, start=start, end=end)
    market = yahoo.load_market_state(start=start, end=end)

    panel = build_panel(frames, horizons=[horizon], market=market)
    if panel.empty:
        return panel
    if universe_size:
        panel = apply_universe_filter(panel, size=universe_size)
    return add_cross_sectional_features(panel)


def run(
    panel: pd.DataFrame,
    horizon: int,
    model_names: list[str],
    target_mode: str = "cross_sectional",
    tune: bool = False,
    tune_splits: int = 3,
    **wf_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward every model over the panel for one horizon.

    Returns (predictions, scorecard). Predictions carry a `model` column so
    several models' output can be stored together.
    """
    X, y, fwd = split_xy(panel, horizon=horizon, target_mode=target_mode)
    if X.empty:
        logger.warning("No usable rows for target '%s'", target_mode)
        return pd.DataFrame(), pd.DataFrame()

    all_preds, rows = [], []

    for name in model_names:
        logger.info("Running %s (horizon=%d, target=%s)", name, horizon, target_mode)

        # partial binds the model name now; a lambda would capture the loop
        # variable and every fold would build the last model in the list.
        factory = partial(model_lib.build, name)

        params = {"horizon": horizon} if model_lib.needs_horizon(name) else {}
        base = partial(factory, **params)

        # Tuning happens inside every fold, on that fold's training rows only.
        tuner = None
        if tune and name in model_lib.PARAM_GRIDS:
            grid = model_lib.PARAM_GRIDS[name]
            tuner = partial(tune_on_train, model_factory=base, grid=grid,
                            horizon=horizon, n_splits=tune_splits)

        try:
            preds = walk_forward(X, y, fwd, base, horizon=horizon, tuner=tuner, **wf_kwargs)
        except Exception:
            logger.exception("  %s failed", name)
            continue

        if preds.empty:
            continue

        score = evaluate(
            preds, horizon=horizon, min_names=MIN_NAMES_PER_DATE,
            refit_every=wf_kwargs.get("refit_every", REFIT_EVERY),
        )
        rows.append({"model": name, "horizon": horizon, "target": target_mode, **score})

        preds["model"] = name
        preds["horizon"] = horizon
        preds["target"] = target_mode
        all_preds.append(preds)

    predictions = pd.concat(all_preds) if all_preds else pd.DataFrame()
    return predictions, pd.DataFrame(rows)
