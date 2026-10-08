"""Production models: trained on all available history, saved, loaded, scored.

The production configuration is fixed in advance rather than picked from the
test results: an equal-weight rank average of the four non-linear models.
Choosing whichever ensemble scored best out of sample would quietly turn the
test period into a selection set.

Training needs the Sharadar snapshot, so it runs locally. The saved bundles
hold model weights and metadata only — no data — and are what the daily run
loads, on a machine with neither the snapshot nor a GPU.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src import experiment
from src.config import ROOT, START_DATE
from src.models import build, needs_horizon
from src.panel import split_xy
from src.store import _git_commit

logger = logging.getLogger(__name__)

MODELS_DIR = ROOT / "models"
MEMBERS = ["lightgbm", "xgboost", "mlp", "lstm"]
HORIZONS = (5, 20)
SEQ_LEN = 20
TRAIN_STRIDE = 5


def bundle_path(horizon: int):
    return MODELS_DIR / f"production_h{horizon}.joblib"


def train(horizons=HORIZONS, members=MEMBERS, start: str = START_DATE, size: int = 1000) -> None:
    """Fit every member on all labelled history and save one bundle per horizon."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    panel = experiment.load_sharadar_panel(horizons=list(horizons), start=start,
                                           size=size, seq_len=SEQ_LEN)

    for h in horizons:
        X, y, _ = split_xy(panel, horizon=h)
        dates = X.index.get_level_values("date")
        keep = y.notna() & dates.isin(dates.unique().sort_values()[::TRAIN_STRIDE])
        X_train, y_train = X[keep], y[keep]

        fitted = {}
        for name in members:
            params = {"horizon": h} if needs_horizon(name) else {}
            logger.info("Training %s for h=%d on %d rows", name, h, len(X_train))
            fitted[name] = build(name, **params).fit(X_train, y_train)

        train_dates = X_train.index.get_level_values("date")
        bundle = {
            "horizon": h,
            "members": fitted,
            "columns": list(X.columns),
            "seq_len": SEQ_LEN,
            "universe_size": size,
            "train_start": str(train_dates.min().date()),
            "train_end": str(train_dates.max().date()),
            "train_rows": int(len(X_train)),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
        }
        joblib.dump(bundle, bundle_path(h), compress=3)
        logger.info("Saved %s (%d members, trained to %s)", bundle_path(h).name,
                    len(fitted), bundle["train_end"])


def load(horizon: int) -> dict:
    path = bundle_path(horizon)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run `python -m src.cli train` first.")
    return joblib.load(path)


def score(bundle: dict, X: pd.DataFrame) -> pd.Series:
    """Equal-weight average of each member's within-date percentile rank."""
    missing = [c for c in bundle["columns"] if c not in X.columns]
    if missing:
        raise KeyError(f"Live features are missing {len(missing)} training columns: {missing[:5]}")
    X = X[bundle["columns"]]

    ranks = []
    for model in bundle["members"].values():
        raw = pd.Series(np.asarray(model.predict_proba(X), dtype=float), index=X.index)
        ranks.append(raw.groupby(level="date").rank(pct=True))
    return pd.concat(ranks, axis=1).mean(axis=1).rename("score")
