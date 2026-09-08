"""Purged walk-forward evaluation.

The purge is the part that most naive backtests get wrong. With an h-day
target, the label of a training sample at time s is only known at s+h. A model
predicting at time t may therefore train on samples up to t-h, never up to t —
otherwise the last h training labels encode the very window being predicted.

Models are refit periodically rather than at every step; between refits the
frozen model scores each day, which is what a deployed system would do.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from src.models import Model

logger = logging.getLogger(__name__)


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    fwd_return: pd.Series,
    model_factory: Callable[[], Model],
    horizon: int = 5,
    min_train: int = 756,
    refit_every: int = 21,
    train_window: int | None = None,
) -> pd.DataFrame:
    """Score every day after the warm-up period using only prior information.

    Parameters
    ----------
    min_train : first prediction is made once this many labelled rows exist.
    refit_every : trading days between refits.
    train_window : rolling training length; None trains on all history.

    Returns one row per prediction: date, prob_up, pred_dir, y_true, fwd_return.
    """
    n = len(X)
    records = []

    for block_start in range(min_train, n, refit_every):
        # Purge: labels after this cutoff overlap the block being predicted.
        train_end = block_start - horizon
        if train_end <= 0:
            continue

        train_start = 0 if train_window is None else max(0, train_end - train_window)
        train_slice = slice(train_start, train_end)

        X_train = X.iloc[train_slice]
        y_train = y.iloc[train_slice]

        labelled = y_train.notna()
        X_train, y_train = X_train[labelled], y_train[labelled]

        if len(X_train) < 100 or y_train.nunique() < 2:
            continue

        try:
            model = model_factory().fit(X_train, y_train)
        except Exception as e:
            logger.warning("Fit failed at %s: %s", X.index[block_start].date(), e)
            continue

        block = slice(block_start, min(block_start + refit_every, n))
        X_block = X.iloc[block]
        probs = model.predict_proba(X_block)

        for date, prob, truth, ret in zip(
            X_block.index, probs, y.iloc[block].values, fwd_return.iloc[block].values
        ):
            records.append({
                "date": date,
                "prob_up": float(prob),
                "pred_dir": int(prob > 0.5),
                "y_true": truth,
                "fwd_return": ret,
                "n_train": len(X_train),
            })

    return pd.DataFrame(records)


def run_experiment(
    df: pd.DataFrame,
    ticker: str,
    model_names: list[str],
    horizon: int = 5,
    target_mode: str = "absolute",
    **kwargs,
) -> pd.DataFrame:
    """Walk-forward every named model over one ticker; returns stacked predictions."""
    from src.features import build_dataset
    from src.models import build

    X, y, fwd = build_dataset(df, horizon=horizon, target_mode=target_mode)
    if len(X) < kwargs.get("min_train", 756) + horizon:
        logger.warning("%s: only %d usable rows, skipping", ticker, len(X))
        return pd.DataFrame()

    frames = []
    for name in model_names:
        preds = walk_forward(X, y, fwd, lambda n=name: build(n), horizon=horizon, **kwargs)
        if preds.empty:
            continue
        preds.insert(0, "ticker", ticker)
        preds.insert(1, "model", name)
        preds.insert(2, "horizon", horizon)
        preds.insert(3, "target_mode", target_mode)
        frames.append(preds)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
