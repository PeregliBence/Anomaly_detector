"""Purged walk-forward validation over a panel.

Two things make panel validation different from the single-series case, and
both are easy to get wrong:

Purging must be done by date, not by row position. A training row's label for
an h-day horizon is only known h days later, so a model predicting on date t
may train on rows dated up to t-h. In a panel the rows are interleaved across
instruments, so slicing by position would leave one instrument's future inside
another's past.

Hyperparameters must be chosen inside the training window. Tuning on the full
history and then "validating" out of sample leaks the test period through the
parameter choice — a quieter version of the same error the purge prevents.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.config import MIN_TRAIN_DAYS, REFIT_EVERY

logger = logging.getLogger(__name__)


def _dates(X: pd.DataFrame) -> pd.DatetimeIndex:
    return X.index.get_level_values("date").unique().sort_values()


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    fwd_return: pd.Series,
    model_factory: Callable[[], object],
    horizon: int,
    min_train_days: int = MIN_TRAIN_DAYS,
    refit_every: int = REFIT_EVERY,
    train_window_days: int | None = None,
) -> pd.DataFrame:
    """Score every date after the warm-up using only prior information.

    `X` is indexed by (date, ticker). The model is refit every `refit_every`
    trading days; between refits the frozen model scores each day, which is
    what a deployed system does.

    Returns one row per prediction, indexed by (date, ticker).
    """
    dates = _dates(X)
    if len(dates) <= min_train_days + horizon:
        logger.warning("Only %d dates available, need more than %d",
                       len(dates), min_train_days + horizon)
        return pd.DataFrame()

    date_level = X.index.get_level_values("date")
    records = []

    for start in range(min_train_days, len(dates), refit_every):
        # Purge: labels dated after this cutoff resolve inside the block being
        # predicted, so they encode its outcome.
        cutoff = dates[start - horizon]
        lower = None
        if train_window_days is not None:
            lower = dates[max(0, start - horizon - train_window_days)]

        train_mask = date_level <= cutoff
        if lower is not None:
            train_mask &= date_level >= lower

        X_train, y_train = X[train_mask], y[train_mask]
        labelled = y_train.notna()
        X_train, y_train = X_train[labelled], y_train[labelled]

        if len(X_train) < 500 or y_train.nunique() < 2:
            continue

        block_dates = dates[start:start + refit_every]
        block_mask = date_level.isin(block_dates)
        if not block_mask.any():
            continue

        try:
            model = model_factory().fit(X_train, y_train)
            scores = model.predict_proba(X[block_mask])
        except Exception as e:
            logger.warning("Fold at %s failed: %s", dates[start].date(), e)
            continue

        block = pd.DataFrame(
            {
                "score": np.asarray(scores, dtype=float),
                "y_true": y[block_mask].values,
                "fwd_return": fwd_return[block_mask].values,
                "train_rows": len(X_train),
                "train_end": cutoff,
            },
            index=X[block_mask].index,
        )
        records.append(block)

    if not records:
        return pd.DataFrame()

    out = pd.concat(records).sort_index()
    logger.info("Walk-forward produced %d predictions over %d folds",
                len(out), len(records))
    return out


def tune_on_train(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[..., object],
    grid: Sequence[dict],
    horizon: int,
    n_splits: int = 3,
) -> dict:
    """Pick hyperparameters using purged splits inside the training window.

    The last split's validation slice is the most recent data, so the chosen
    parameters reflect the regime the model is about to be deployed into.
    Returns the best parameter dict by mean validation log loss.
    """
    from sklearn.metrics import log_loss

    dates = _dates(X)
    date_level = X.index.get_level_values("date")
    if len(dates) < (n_splits + 1) * (horizon + 60):
        return dict(grid[0])

    bounds = np.linspace(len(dates) // 2, len(dates), n_splits + 1).astype(int)
    best_params, best_score = dict(grid[0]), np.inf

    for params in grid:
        losses = []
        for i in range(n_splits):
            split, end = bounds[i], bounds[i + 1]
            cutoff = dates[split - horizon]

            train = date_level <= cutoff
            valid = date_level.isin(dates[split:end])

            y_tr, y_va = y[train], y[valid]
            tr_ok, va_ok = y_tr.notna(), y_va.notna()
            if tr_ok.sum() < 500 or va_ok.sum() < 100 or y_tr[tr_ok].nunique() < 2:
                continue

            try:
                model = model_factory(**params).fit(X[train][tr_ok], y_tr[tr_ok])
                proba = model.predict_proba(X[valid][va_ok])
                losses.append(log_loss(y_va[va_ok], proba, labels=[0.0, 1.0]))
            except Exception as e:
                logger.debug("Tuning fold failed for %s: %s", params, e)

        if losses and np.mean(losses) < best_score:
            best_score, best_params = float(np.mean(losses)), dict(params)

    logger.debug("Selected params %s (log loss %.4f)", best_params, best_score)
    return best_params
