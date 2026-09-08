"""Directional classifiers and the baselines they must beat.

Every model exposes the same interface: fit on a training fold, then emit
P(up) for unseen rows. Any preprocessing that learns from data (scaling,
class priors) is fit inside `fit` so it can never see the test fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class Model:
    """Base interface: fit(X, y) then predict_proba(X) -> P(up)."""

    name = "base"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model":
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class AlwaysUp(Model):
    """Buy and hold. The bar any equity model has to clear, because of drift."""

    name = "always_up"

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.ones(len(X))


class BaseRate(Model):
    """Constant P(up) equal to the training-fold frequency of up moves."""

    name = "base_rate"

    def fit(self, X, y):
        self.p_up_ = float(y.mean())
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p_up_)


class Random(Model):
    """Coin flip. Sanity anchor for the evaluation harness itself.

    The generator is shared across instances so that refitting mid-walk-forward
    continues the stream instead of replaying the same draws each block.
    """

    name = "random"
    _rng = np.random.default_rng(42)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return Random._rng.random(len(X))


class Persistence(Model):
    """Assume the last 20-day move continues. Naive momentum reference."""

    name = "persistence"

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return (X["mom_20"].values > 0).astype(float)


# ---------------------------------------------------------------------------
# Learned models
# ---------------------------------------------------------------------------

class Logistic(Model):
    """L2 logistic regression on standardised features."""

    name = "logistic"

    def __init__(self, C: float = 0.1):
        self.C = C

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler_ = StandardScaler().fit(X.values)
        self.clf_ = LogisticRegression(C=self.C, max_iter=2000).fit(
            self.scaler_.transform(X.values), y.values
        )
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.scaler_.transform(X.values))[:, 1]


class XGBoost(Model):
    """Gradient boosted trees predicting the direction label directly.

    Shallow and heavily regularised: the signal-to-noise ratio on daily
    returns is low enough that a deeper model memorises the training fold.
    """

    name = "xgboost"

    def __init__(self, n_estimators: int = 300, max_depth: int = 3, learning_rate: float = 0.03):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

    def fit(self, X, y):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed")
        self.clf_ = xgb.XGBClassifier(**self.params).fit(X.values, y.values)
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(X.values)[:, 1]


REGISTRY: dict[str, type[Model]] = {
    "always_up": AlwaysUp,
    "base_rate": BaseRate,
    "random": Random,
    "persistence": Persistence,
    "logistic": Logistic,
    "xgboost": XGBoost,
}


def build(name: str) -> Model:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]()
