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


TARGETS = ("cross_sectional", "absolute")


def bundle_path(horizon: int, target: str = "cross_sectional"):
    prefix = "production" if target == "cross_sectional" else "production_abs"
    return MODELS_DIR / f"{prefix}_h{horizon}.joblib"


def train(horizons=HORIZONS, members=MEMBERS, start: str = START_DATE, size: int = 1000,
          targets=TARGETS) -> None:
    """Fit every member on all labelled history; one bundle per horizon and target.

    cross_sectional: will the stock beat the median of the field?
    absolute:        will the stock's price be higher after the horizon?
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    panel = experiment.load_sharadar_panel(horizons=list(horizons), start=start,
                                           size=size, seq_len=SEQ_LEN)
    for target in targets:
        for h in horizons:
            _train_one(panel, h, target, members, size)


def _train_one(panel: pd.DataFrame, h: int, target: str, members, size: int) -> None:
    X, y, _ = split_xy(panel, horizon=h, target_mode=target)
    dates = X.index.get_level_values("date")
    keep = y.notna() & dates.isin(dates.unique().sort_values()[::TRAIN_STRIDE])
    X_train, y_train = X[keep], y[keep]

    fitted = {}
    for name in members:
        params = {"horizon": h} if needs_horizon(name) else {}
        logger.info("Training %s for h=%d (%s) on %d rows", name, h, target, len(X_train))
        fitted[name] = build(name, **params).fit(X_train, y_train)

    train_dates = X_train.index.get_level_values("date")
    bundle = {
        "horizon": h,
        "target": target,
        "members": fitted,
        "columns": list(X.columns),
        "seq_len": SEQ_LEN,
        "universe_size": size,
        "train_start": str(train_dates.min().date()),
        "train_end": str(train_dates.max().date()),
        "train_rows": int(len(X_train)),
        "base_rate": float(y_train.mean()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }
    path = bundle_path(h, target)
    joblib.dump(bundle, path, compress=3)
    logger.info("Saved %s (%d members, trained to %s)", path.name, len(fitted), bundle["train_end"])


def load(horizon: int, target: str = "cross_sectional") -> dict:
    path = bundle_path(horizon, target)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run `python -m src.cli train` first.")
    return joblib.load(path)


def score(bundle: dict, X: pd.DataFrame) -> pd.Series:
    """Combine the members' outputs into one score per row.

    Cross-sectional bundles average each member's within-date percentile rank:
    only the ordering matters, and ranks put members on a common scale.
    Absolute bundles average the members' probabilities, because there the
    probability of a rise is itself the forecast and must keep its meaning.
    """
    missing = [c for c in bundle["columns"] if c not in X.columns]
    if missing:
        raise KeyError(f"Live features are missing {len(missing)} training columns: {missing[:5]}")
    X = X[bundle["columns"]]

    outputs = [pd.Series(np.asarray(m.predict_proba(X), dtype=float), index=X.index)
               for m in bundle["members"].values()]
    if bundle.get("target", "cross_sectional") == "absolute":
        return pd.concat(outputs, axis=1).mean(axis=1).rename("prob_up")
    ranks = [o.groupby(level="date").rank(pct=True) for o in outputs]
    return pd.concat(ranks, axis=1).mean(axis=1).rename("score")
