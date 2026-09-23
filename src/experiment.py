"""Run one experiment: a universe, a horizon, a target, a set of models."""

from __future__ import annotations

import logging
from functools import partial

import pandas as pd

from src import models as model_lib
from src.config import MIN_NAMES_PER_DATE, START_DATE
from src.metrics import evaluate
from src.panel import add_cross_sectional_features, apply_universe_filter, build_panel, split_xy
from src.sources import yahoo
from src.validation import tune_on_train, walk_forward

logger = logging.getLogger(__name__)


def load_panel(
    tickers: list[str],
    horizon: int,
    start: str = START_DATE,
    end: str | None = None,
    universe_size: int | None = None,
) -> pd.DataFrame:
    """Download and assemble the panel, ready for `run`."""
    logger.info("Loading %d instruments from %s", len(tickers), start)
    frames = yahoo.load_many(tickers, start=start, end=end)
    market = yahoo.load_market_state(start=start, end=end)

    panel = build_panel(frames, horizon=horizon, market=market)
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
    **wf_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward every model over the panel.

    Returns (predictions, scorecard). Predictions carry a `model` column so
    several models' output can be stored together.
    """
    X, y, fwd = split_xy(panel, target_mode=target_mode)
    if X.empty:
        logger.warning("No usable rows for target '%s'", target_mode)
        return pd.DataFrame(), pd.DataFrame()

    all_preds, rows = [], []

    for name in model_names:
        logger.info("Running %s (horizon=%d, target=%s)", name, horizon, target_mode)

        # partial binds the model name now; a lambda would capture the loop
        # variable and every fold would build the last model in the list.
        factory = partial(model_lib.build, name)

        params = {}
        if tune and name in model_lib.PARAM_GRIDS:
            params = tune_on_train(
                X, y, factory, model_lib.PARAM_GRIDS[name], horizon=horizon
            )
            logger.info("  tuned params: %s", params)

        try:
            preds = walk_forward(
                X, y, fwd,
                partial(factory, **params),
                horizon=horizon,
                **wf_kwargs,
            )
        except Exception as e:
            logger.warning("  %s failed: %s", name, e)
            continue

        if preds.empty:
            continue

        score = evaluate(preds, horizon=horizon, min_names=MIN_NAMES_PER_DATE)
        rows.append({
            "model": name, "horizon": horizon, "target": target_mode, **score
        })

        preds["model"] = name
        preds["horizon"] = horizon
        preds["target"] = target_mode
        all_preds.append(preds)

    predictions = pd.concat(all_preds) if all_preds else pd.DataFrame()
    scorecard = pd.DataFrame(rows)
    return predictions, scorecard
