"""
Ensemble anomaly scoring and severity classification.

Combines multiple anomaly detection methods into a unified score with:
  - Percentile-rank normalized scores
  - Weighted combination
  - K-of-N voting
  - Severity levels: normal / mild / strong
"""

import numpy as np
import pandas as pd

from app.anomaly_models import percentile_rank
from app.config import logger


def compute_ensemble_score(
    df: pd.DataFrame,
    weights: dict | None = None,
    contamination: float = 0.01,
    vote_k: int = 2,
    combine: str = "and",
) -> pd.DataFrame:
    """
    Build unified ensemble anomaly score from individual model scores.

    Parameters
    ----------
    weights : dict
        Weights for each model's score (e.g. {"zscore": 1, "iso": 1, "lof": 1, "lstm": 1, "ocsvm": 0.5}).
    contamination : float
        Expected proportion of anomalies.
    vote_k : int
        Minimum number of models that must flag a point for the vote to trigger.
    combine : str
        "and" = score_flag AND vote_flag (conservative)
        "or"  = score_flag OR vote_flag (aggressive)
    """
    df = df.copy()

    if weights is None:
        weights = {"zscore": 1.0, "iso": 1.0, "lof": 1.0, "lstm": 1.0, "ocsvm": 0.5,
                    "stl": 0.5, "cusum": 0.3, "volume": 0.3}

    # Map score columns
    score_map = {
        "zscore": ("zscore", True),    # higher abs = more anomalous
        "iso": ("iso_score", False),    # lower = more anomalous (invert)
        "lof": ("lof_score", True),     # higher = more anomalous
        "lstm": ("lstm_error", True),   # higher = more anomalous
        "ocsvm": ("ocsvm_score", False), # lower = more anomalous (invert)
        "stl": ("stl_residual", True),  # higher abs = more anomalous
    }

    # Compute percentile ranks for available scores
    ranked = {}
    for key, (col, higher_is_worse) in score_map.items():
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        if key in ("zscore", "stl"):
            vals = np.abs(vals)
        elif not higher_is_worse:
            vals = -vals  # invert so higher = more anomalous
        ranked[key] = percentile_rank(vals)

    if not ranked:
        df["ensemble_score"] = 0.0
        df["anomaly_ensemble"] = 0
        df["anomaly_severity"] = "normal"
        return df

    # Weighted combination
    w_total = np.zeros(len(df), dtype=float)
    score_total = np.zeros(len(df), dtype=float)
    default_weight = 0.5

    for key, p_rank in ranked.items():
        if key not in weights:
            logger.warning("Model '%s' not in weights dict; using default %.1f", key, default_weight)
            w = default_weight
        else:
            w = float(weights[key])
        valid = np.isfinite(p_rank)
        score_total[valid] += p_rank[valid] * w
        w_total[valid] += w

    ens_score = np.where(w_total > 0, score_total / w_total, 0.0)
    df["ensemble_score"] = ens_score

    # Score-based flag
    valid_scores = ens_score[np.isfinite(ens_score)]
    if len(valid_scores) > 0:
        cutoff = np.percentile(valid_scores, 100.0 * (1.0 - contamination))
        score_flag = (ens_score >= cutoff).astype(int)
    else:
        score_flag = np.zeros(len(df), dtype=int)

    # K-of-N voting
    anomaly_cols = [c for c in [
        "anomaly_zscore", "anomaly_iso", "anomaly_lof", "anomaly_lstm",
        "anomaly_ocsvm", "anomaly_stl", "anomaly_cusum", "anomaly_iqr",
        "anomaly_volume", "anomaly_vol_shift",
    ] if c in df.columns]

    vote_sum = np.zeros(len(df), dtype=int)
    for c in anomaly_cols:
        vote_sum += df[c].fillna(0).astype(int).values

    vote_flag = (vote_sum >= vote_k).astype(int)

    # Combine
    if combine == "or":
        df["anomaly_ensemble"] = ((score_flag == 1) | (vote_flag == 1)).astype(int)
    else:
        df["anomaly_ensemble"] = ((score_flag == 1) & (vote_flag == 1)).astype(int)

    # Severity classification
    df["anomaly_severity"] = classify_severity(df, ens_score, vote_sum)
    df["vote_count"] = vote_sum

    logger.info("Ensemble: %d anomalies detected", df["anomaly_ensemble"].sum())
    return df


def classify_severity(
    df: pd.DataFrame,
    ens_score: np.ndarray,
    vote_sum: np.ndarray,
) -> pd.Series:
    """
    Classify anomalies into severity levels:
    - 'normal': not anomalous
    - 'mild': flagged by ensemble but lower score or fewer votes
    - 'strong': high score AND multiple model agreement
    """
    valid = ens_score[np.isfinite(ens_score)]
    if len(valid) == 0:
        return pd.Series("normal", index=df.index)

    p90 = np.percentile(valid, 90)
    p99 = np.percentile(valid, 99)

    severity = np.where(
        (ens_score >= p99) & (vote_sum >= 3), "strong",
        np.where(
            (ens_score >= p90) & (vote_sum >= 2), "mild",
            "normal",
        )
    )
    return pd.Series(severity, index=df.index)


def get_anomaly_summary(df: pd.DataFrame) -> dict:
    """
    Summary statistics for the anomaly detection results.
    Used for KPI cards in the dashboard.
    """
    summary = {
        "total_points": len(df),
        "total_anomalies": 0,
        "mild_anomalies": 0,
        "strong_anomalies": 0,
        "latest_anomaly_date": None,
        "anomaly_rate": 0.0,
        "model_counts": {},
    }

    if "anomaly_ensemble" in df.columns:
        anomalies = df[df["anomaly_ensemble"] == 1]
        summary["total_anomalies"] = len(anomalies)
        summary["anomaly_rate"] = len(anomalies) / max(len(df), 1)

        if not anomalies.empty:
            summary["latest_anomaly_date"] = anomalies.index[-1]

    if "anomaly_severity" in df.columns:
        summary["mild_anomalies"] = int((df["anomaly_severity"] == "mild").sum())
        summary["strong_anomalies"] = int((df["anomaly_severity"] == "strong").sum())

    for model_col in ["anomaly_zscore", "anomaly_iso", "anomaly_lof", "anomaly_lstm",
                       "anomaly_ocsvm", "anomaly_stl", "anomaly_cusum"]:
        if model_col in df.columns:
            name = model_col.replace("anomaly_", "")
            summary["model_counts"][name] = int(df[model_col].sum())

    return summary
