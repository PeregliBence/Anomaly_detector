"""The walk-forward must never let a model see the outcome it is scored on."""

import numpy as np
import pandas as pd
import pytest

from conftest import toy_panel
from src.validation import walk_forward


class Spy:
    """Records which dates it was trained on and which it was asked to score."""

    instances: list["Spy"] = []

    def __init__(self, **params):
        self.params = params
        Spy.instances.append(self)

    def fit(self, X, y):
        self.train_dates = X.index.get_level_values("date").unique()
        return self

    def predict_proba(self, X):
        self.scored_dates = X.index.get_level_values("date").unique()
        return np.full(len(X), 0.5)


def run(horizon=5, **kwargs):
    Spy.instances = []
    X, y, fwd = toy_panel()
    preds = walk_forward(X, y, fwd, Spy, horizon=horizon, min_train_days=100,
                         refit_every=20, **kwargs)
    dates = X.index.get_level_values("date").unique().sort_values()
    return preds, dates


def test_purge_leaves_a_horizon_gap_before_every_scored_block():
    horizon = 5
    _, dates = run(horizon=horizon)
    position = {d: i for i, d in enumerate(dates)}
    assert Spy.instances
    for spy in Spy.instances:
        last_trained = max(position[d] for d in spy.train_dates)
        first_scored = min(position[d] for d in spy.scored_dates)
        # A label on the last training date resolves `horizon` days later;
        # that must not reach into the block being scored.
        assert last_trained + horizon <= first_scored


def test_every_date_after_warm_up_is_scored_exactly_once():
    preds, dates = run()
    scored = preds.index.get_level_values("date")
    assert scored.min() == dates[100]
    assert not preds.index.duplicated().any()
    assert set(scored) == set(dates[100:])


def test_train_stride_only_trains_on_every_kth_date():
    _, dates = run(train_stride=5)
    allowed = set(dates[::5])
    for spy in Spy.instances:
        assert set(spy.train_dates) <= allowed


def test_tuner_sees_only_the_folds_training_rows():
    seen = []

    def tuner(X_train, y_train):
        seen.append(X_train.index.get_level_values("date").max())
        return {"chosen": len(seen)}

    preds, _ = run(tuner=tuner)
    assert len(seen) == len(Spy.instances)
    for spy, latest in zip(Spy.instances, seen):
        assert latest <= max(spy.train_dates)
        assert latest < min(spy.scored_dates)
    assert [s.params["chosen"] for s in Spy.instances] == list(range(1, len(seen) + 1))
    assert len(preds.attrs["fold_params"]) == len(seen)


def test_tree_learning_curves_have_one_point_per_tree():
    pytest.importorskip("lightgbm")
    from src import models
    from src.neural import _time_split

    X, y, _ = toy_panel(n_dates=120, n_names=30)
    tr, va = _time_split(X, 0.1, gap=5)
    for name in ("lightgbm", "xgboost"):
        m = models.build(name, n_estimators=25).fit_with_curve(X[tr], y[tr], X[va], y[va], X[tr], y[tr])
        for split in ("train", "valid"):
            assert len(m.history_[split]["logloss"]) == 25
            assert len(m.history_[split]["auc"]) == 25
