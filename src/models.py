"""Models and the baselines they have to beat.

For a cross-sectional problem the output that matters is a score used to rank
the field on each date, not a hard class. Every model therefore emits a number
in [0, 1]; the binary prediction is a threshold on it.

The baselines are the point of this module. `cs_momentum` is a published,
free strategy: rank by 12-1 momentum and buy the top. A learned model that
cannot beat it has added nothing, no matter what its accuracy looks like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class Model:
    name = "base"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model":
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class Random(Model):
    """Coin flip. Sanity anchor for the harness itself."""

    name = "random"
    _rng = np.random.default_rng(42)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return Random._rng.random(len(X))


class ConstantRate(Model):
    """Every instrument scored at the training base rate. No ranking at all."""

    name = "constant"

    def fit(self, X, y):
        self.p_ = float(y.mean())
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p_)


class _RankFeature(Model):
    """Score by an already cross-sectionally ranked column."""

    column = ""
    invert = False

    def fit(self, X, y):
        if self.column not in X.columns:
            raise KeyError(f"{self.name} requires the '{self.column}' column")
        return self

    def predict_proba(self, X):
        rank = X[self.column].values
        return 1.0 - rank if self.invert else rank


class CrossSectionalMomentum(_RankFeature):
    """Jegadeesh-Titman momentum: buy past winners over 12-1 months."""

    name = "cs_momentum"
    column = "cs_mom_12_1"


class CrossSectionalReversal(_RankFeature):
    """Short-horizon reversal: buy recent losers."""

    name = "cs_reversal"
    column = "cs_reversal_5"
    invert = True


class LowVolatility(_RankFeature):
    """The low-volatility anomaly: buy the calmest names."""

    name = "cs_low_vol"
    column = "cs_vol_20"
    invert = True


# ---------------------------------------------------------------------------
# Learned models
# ---------------------------------------------------------------------------

class Logistic(Model):
    """Regularised linear probe. Fast, and a fair test of whether the signal
    is linear before reaching for anything heavier."""

    name = "logistic"

    def __init__(self, C: float = 0.01):
        self.C = C

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler_ = StandardScaler().fit(X.values)
        self.clf_ = LogisticRegression(C=self.C, max_iter=3000).fit(
            self.scaler_.transform(X.values), y.values
        )
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.scaler_.transform(X.values))[:, 1]


class LightGBM(Model):
    """Gradient boosted trees. Still the strongest family on tabular panels,
    and fast enough to tune inside every training fold."""

    name = "lightgbm"

    def __init__(self, n_estimators=400, num_leaves=31, learning_rate=0.03,
                 min_child_samples=200, reg_lambda=5.0):
        self.params = dict(
            n_estimators=n_estimators, num_leaves=num_leaves,
            learning_rate=learning_rate, min_child_samples=min_child_samples,
            reg_lambda=reg_lambda, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, objective="binary",
            random_state=42, n_jobs=-1, verbose=-1,
        )

    def fit(self, X, y):
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed")
        self.clf_ = lgb.LGBMClassifier(**self.params).fit(X.values, y.values)
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(X.values)[:, 1]


class XGBoost(Model):
    """Second boosting implementation, as a check that results are not an
    artefact of one library's defaults."""

    name = "xgboost"

    def __init__(self, n_estimators=400, max_depth=5, learning_rate=0.03,
                 min_child_weight=200, reg_lambda=5.0):
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, min_child_weight=min_child_weight,
            reg_lambda=reg_lambda, subsample=0.8, colsample_bytree=0.7,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )

    def fit(self, X, y):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed")
        self.clf_ = xgb.XGBClassifier(**self.params).fit(X.values, y.values)
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(X.values)[:, 1]


class Ensemble(Model):
    """Average of several models' scores.

    Averaging helps here for the usual reason: the members' errors are only
    partly correlated, so the mean has less variance than any member.
    """

    name = "ensemble"

    def __init__(self, members: list[str] | None = None):
        self.member_names = members or ["logistic", "lightgbm", "xgboost"]

    def fit(self, X, y):
        self.members_ = []
        for name in self.member_names:
            try:
                self.members_.append(build(name).fit(X, y))
            except Exception:
                continue
        if not self.members_:
            raise RuntimeError("No ensemble member could be fitted")
        return self

    def predict_proba(self, X):
        scores = np.column_stack([m.predict_proba(X) for m in self.members_])
        return scores.mean(axis=1)


REGISTRY: dict[str, type[Model]] = {
    "random": Random,
    "constant": ConstantRate,
    "cs_momentum": CrossSectionalMomentum,
    "cs_reversal": CrossSectionalReversal,
    "cs_low_vol": LowVolatility,
    "logistic": Logistic,
    "lightgbm": LightGBM,
    "xgboost": XGBoost,
    "ensemble": Ensemble,
}

# Searched inside the training window only — see validation.tune_on_train.
PARAM_GRIDS: dict[str, list[dict]] = {
    "logistic": [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)],
    "lightgbm": [
        {"num_leaves": leaves, "learning_rate": lr, "min_child_samples": mcs}
        for leaves in (15, 31)
        for lr in (0.01, 0.03)
        for mcs in (100, 400)
    ],
    "xgboost": [
        {"max_depth": d, "learning_rate": lr}
        for d in (3, 5)
        for lr in (0.01, 0.03)
    ],
}

BASELINES = ["random", "constant", "cs_momentum", "cs_reversal", "cs_low_vol"]
LEARNED = ["logistic", "lightgbm", "xgboost"]


def build(name: str, **params) -> Model:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**params) if params else REGISTRY[name]()
