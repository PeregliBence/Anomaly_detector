"""Scoring for panel predictions.

The headline number for a cross-sectional model is the information
coefficient: the rank correlation, computed within each date, between the
score and the return that followed. Its mean over dates is the edge, and its
t-statistic says whether that edge is distinguishable from zero.

That t-statistic needs care. Predicting daily on an h-day horizon makes
consecutive windows overlap, so consecutive ICs are autocorrelated and the
ordinary standard error understates the noise. The Newey-West correction below
widens it to account for the overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Cross-sectional
# ---------------------------------------------------------------------------

def rank_ic(preds: pd.DataFrame, min_names: int = 20) -> pd.Series:
    """Per-date Spearman correlation between score and realised return."""
    usable = preds.dropna(subset=["score", "fwd_return"])

    def one_date(group: pd.DataFrame) -> float:
        if len(group) < min_names or group["score"].nunique() < 2:
            return np.nan
        return stats.spearmanr(group["score"], group["fwd_return"]).statistic

    return usable.groupby(level="date").apply(one_date).dropna()


def newey_west_tstat(series: pd.Series, lags: int) -> tuple[float, float]:
    """t-statistic for the mean of an autocorrelated series.

    Returns (t_stat, standard_error). `lags` should be the horizon minus one:
    that is how far the overlap between prediction windows reaches.
    """
    x = series.dropna().values
    n = len(x)
    if n < 10:
        return (float("nan"), float("nan"))

    centred = x - x.mean()
    variance = float(centred @ centred) / n

    for lag in range(1, min(lags, n - 1) + 1):
        cov = float(centred[lag:] @ centred[:-lag]) / n
        variance += 2 * (1 - lag / (lags + 1)) * cov

    variance = max(variance, 1e-12)
    se = np.sqrt(variance / n)
    return (float(x.mean() / se), float(se))


def ic_summary(preds: pd.DataFrame, horizon: int, min_names: int = 20) -> dict:
    """Mean IC with overlap-corrected significance."""
    ic = rank_ic(preds, min_names=min_names)
    if ic.empty:
        return {"n_dates": 0}

    t_stat, se = newey_west_tstat(ic, lags=max(horizon - 1, 1))
    mean = float(ic.mean())
    std = float(ic.std())

    return {
        "n_dates": len(ic),
        "ic_mean": mean,
        "ic_std": std,
        # Information ratio: edge per unit of its own variability.
        "ic_ir": mean / std if std > 0 else float("nan"),
        "ic_tstat": t_stat,
        "ic_se": se,
        "ic_pvalue": float(2 * (1 - stats.norm.cdf(abs(t_stat)))) if np.isfinite(t_stat) else float("nan"),
        "ic_positive_rate": float((ic > 0).mean()),
    }


def decile_spread(preds: pd.DataFrame, n_buckets: int = 10, min_names: int = 20) -> dict:
    """Return of a long-top / short-bottom bucket portfolio.

    Translates the rank correlation into the quantity the result would
    actually be judged on: what the ordering was worth.
    """
    usable = preds.dropna(subset=["score", "fwd_return"])
    rows = []

    for date, group in usable.groupby(level="date"):
        if len(group) < max(min_names, n_buckets * 2):
            continue
        buckets = pd.qcut(group["score"].rank(method="first"), n_buckets, labels=False)
        top = group["fwd_return"][buckets == n_buckets - 1].mean()
        bottom = group["fwd_return"][buckets == 0].mean()
        rows.append({"date": date, "top": top, "bottom": bottom, "spread": top - bottom})

    if not rows:
        return {"n_dates": 0}

    spreads = pd.DataFrame(rows).set_index("date")["spread"]
    return {
        "n_dates": len(spreads),
        "spread_mean": float(spreads.mean()),
        "spread_std": float(spreads.std()),
        "spread_positive_rate": float((spreads > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Pooled classification
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval — reliable for proportions near 0 or 1."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def classification_summary(preds: pd.DataFrame) -> dict:
    """Accuracy and calibration pooled across the panel.

    Reported alongside the IC because they answer different questions: the IC
    asks whether the ordering is right, this asks whether the probabilities
    are.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score

    d = preds.dropna(subset=["y_true", "score"])
    if d.empty:
        return {"n": 0}

    y_true = d["y_true"].astype(int).values
    score = d["score"].values
    y_pred = (score > 0.5).astype(int)

    n = len(d)
    correct = int((y_true == y_pred).sum())
    base_rate = float(y_true.mean())
    majority = max(base_rate, 1 - base_rate)
    lo, hi = wilson_interval(correct, n)

    out = {
        "n": n,
        "accuracy": correct / n,
        "accuracy_ci_low": lo,
        "accuracy_ci_high": hi,
        "majority_baseline": majority,
        "edge_vs_majority": correct / n - majority,
        "base_rate": base_rate,
        "brier": brier_score_loss(y_true, np.clip(score, 0, 1)),
    }
    out["auc"] = (
        roc_auc_score(y_true, score)
        if len(set(y_true)) > 1 and len(set(score)) > 1
        else float("nan")
    )
    return out


def evaluate(preds: pd.DataFrame, horizon: int, min_names: int = 20) -> dict:
    """Full scorecard for one model's panel predictions."""
    if preds.empty:
        return {"n": 0}
    return {
        **classification_summary(preds),
        **ic_summary(preds, horizon=horizon, min_names=min_names),
        **decile_spread(preds, min_names=min_names),
    }
