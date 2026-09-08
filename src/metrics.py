"""Scoring and significance testing for directional predictions.

A raw accuracy number means little on its own: equities drift upward, so
"always up" already scores above 50%. Every accuracy here is therefore paired
with a confidence interval and a test against the relevant null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


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


def score(preds: pd.DataFrame) -> dict:
    """Summarise one model's predictions on one ticker.

    `preds` needs columns: prob_up, pred_dir, y_true, fwd_return.
    Rows whose outcome has not resolved yet are dropped.
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, brier_score_loss,
        matthews_corrcoef, roc_auc_score,
    )

    d = preds.dropna(subset=["y_true"])
    if d.empty:
        return {"n": 0}

    y_true = d["y_true"].astype(int).values
    y_pred = d["pred_dir"].astype(int).values
    prob = d["prob_up"].values
    n = len(d)
    correct = int((y_true == y_pred).sum())
    accuracy = correct / n

    base_rate = float(y_true.mean())
    # The harder of the two trivial strategies: always-up or always-down.
    majority_rate = max(base_rate, 1 - base_rate)

    lo, hi = wilson_interval(correct, n)

    out = {
        "n": n,
        "accuracy": accuracy,
        "accuracy_ci_low": lo,
        "accuracy_ci_high": hi,
        "base_rate_up": base_rate,
        "majority_baseline": majority_rate,
        "edge_vs_majority": accuracy - majority_rate,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0,
        "brier": brier_score_loss(y_true, prob),
        "pred_up_rate": float(y_pred.mean()),
        # Binomial test against a coin flip, and against the majority class.
        "p_value_vs_coin": stats.binomtest(correct, n, 0.5, alternative="greater").pvalue,
        "p_value_vs_majority": stats.binomtest(correct, n, majority_rate, alternative="greater").pvalue,
    }

    if len(set(y_true)) > 1 and len(set(prob)) > 1:
        out["auc"] = roc_auc_score(y_true, prob)
    else:
        out["auc"] = float("nan")

    return out


def score_all(preds: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Score every experiment group, keyed by ticker/model/horizon/target_mode."""
    if preds.empty:
        return pd.DataFrame()

    if by is None:
        by = [c for c in ("ticker", "model", "horizon", "target_mode") if c in preds.columns]

    rows = []
    for keys, group in preds.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(by, keys)), **score(group)})
    return pd.DataFrame(rows)


def mcnemar(preds_a: pd.DataFrame, preds_b: pd.DataFrame) -> dict:
    """Test whether two models differ on the same days (paired, exact binomial)."""
    a = preds_a.dropna(subset=["y_true"]).set_index("date")
    b = preds_b.dropna(subset=["y_true"]).set_index("date")
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {"n": 0, "p_value": float("nan")}

    a_ok = (a.loc[common, "pred_dir"] == a.loc[common, "y_true"]).values
    b_ok = (b.loc[common, "pred_dir"] == b.loc[common, "y_true"]).values

    a_only = int((a_ok & ~b_ok).sum())
    b_only = int((~a_ok & b_ok).sum())
    discordant = a_only + b_only

    if discordant == 0:
        return {"n": len(common), "a_only": 0, "b_only": 0, "p_value": 1.0}

    return {
        "n": len(common),
        "a_only": a_only,
        "b_only": b_only,
        "p_value": stats.binomtest(a_only, discordant, 0.5).pvalue,
    }
