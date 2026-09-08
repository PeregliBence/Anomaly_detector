"""
Backtesting Engine — historical evaluation of anomaly detection and forecasting.

Provides:
  1. Walk-forward validation with rolling train/test splits
  2. Anomaly detection metrics: Precision, Recall, F1
  3. Forecast metrics: directional accuracy, MAE, RMSE, Sharpe ratio
  4. Maximum drawdown calculation
  5. Signal quality assessment

All evaluation is strictly out-of-sample to avoid look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Callable

from app.config import logger


@dataclass
class BacktestResult:
    """Container for backtesting results."""
    # General
    ticker: str = ""
    period_start: str = ""
    period_end: str = ""
    n_windows: int = 0

    # Forecast metrics
    directional_accuracy: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    mean_return_following_signal: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0

    # Anomaly metrics
    anomaly_precision: float = 0.0
    anomaly_recall: float = 0.0
    anomaly_f1: float = 0.0
    anomaly_count: int = 0

    # Confidence calibration data
    confidence_bins: list = field(default_factory=list)
    accuracy_bins: list = field(default_factory=list)

    # Detailed results per window
    window_results: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


# ============================================================================
# 1. WALK-FORWARD VALIDATION
# ============================================================================

def walk_forward_forecast(
    df: pd.DataFrame,
    forecast_fn: Callable,
    window_size: int = 252,
    step_size: int = 20,
    horizon: int = 5,
    min_train_size: int = 200,
) -> BacktestResult:
    """
    Walk-forward validation for forecast models.

    Splits the data into rolling windows:
      [train: window_size] → [test: horizon] → step forward by step_size → repeat

    Parameters
    ----------
    df : DataFrame with features already computed
    forecast_fn : callable(df_train) -> ForecastResult or None
    window_size : training window length
    step_size : how many bars to advance between windows
    horizon : forecast horizon (test period length)
    min_train_size : minimum training data required

    Returns BacktestResult with aggregated metrics.
    """
    result = BacktestResult()
    if len(df) < min_train_size + horizon:
        logger.warning("Not enough data for backtesting (need %d, got %d)",
                       min_train_size + horizon, len(df))
        return result

    result.period_start = str(df.index[0].date())
    result.period_end = str(df.index[-1].date())

    predictions = []
    actuals = []
    confidences = []
    directions_pred = []
    directions_actual = []
    equity = [1.0]

    i = min_train_size
    while i + horizon <= len(df):
        train_start = max(0, i - window_size)
        train_df = df.iloc[train_start:i].copy()
        test_df = df.iloc[i:i + horizon].copy()

        try:
            forecast = forecast_fn(train_df)
        except Exception as e:
            logger.debug("Forecast failed at window %d: %s", i, e)
            i += step_size
            continue

        if forecast is None:
            i += step_size
            continue

        # Compare prediction vs actual
        actual_prices = test_df["Close"].values
        if len(actual_prices) == 0:
            i += step_size
            continue

        pred_prices = forecast.values[:len(actual_prices)]

        # Directional accuracy
        actual_return = (actual_prices[-1] / float(train_df["Close"].iloc[-1])) - 1
        pred_return = (pred_prices[-1] / float(train_df["Close"].iloc[-1])) - 1

        actual_dir = "up" if actual_return > 0.005 else "down" if actual_return < -0.005 else "sideways"
        is_correct = forecast.direction == actual_dir

        # Track for metrics
        predictions.append(pred_prices[-1])
        actuals.append(actual_prices[-1])
        confidences.append(forecast.confidence)
        directions_pred.append(forecast.direction)
        directions_actual.append(actual_dir)

        # Simple equity tracking: if we follow the signal
        signal_return = 0.0
        if forecast.direction == "up":
            signal_return = actual_return
        elif forecast.direction == "down":
            signal_return = -actual_return
        # sideways = no trade

        equity.append(equity[-1] * (1 + signal_return))

        window_result = {
            "train_end": str(train_df.index[-1].date()),
            "test_end": str(test_df.index[-1].date()),
            "predicted_dir": forecast.direction,
            "actual_dir": actual_dir,
            "correct": is_correct,
            "confidence": forecast.confidence,
            "actual_return_pct": round(actual_return * 100, 2),
            "signal_return_pct": round(signal_return * 100, 2),
        }
        result.window_results.append(window_result)

        i += step_size

    result.n_windows = len(result.window_results)
    result.equity_curve = equity

    if not result.window_results:
        return result

    # Aggregate metrics
    correct_count = sum(1 for w in result.window_results if w["correct"])
    result.directional_accuracy = correct_count / len(result.window_results)

    predictions = np.array(predictions)
    actuals_arr = np.array(actuals)
    errors = predictions - actuals_arr
    result.mae = float(np.mean(np.abs(errors)))
    result.rmse = float(np.sqrt(np.mean(errors ** 2)))

    # Sharpe ratio from signal returns
    signal_returns = [w["signal_return_pct"] / 100 for w in result.window_results]
    if len(signal_returns) > 1:
        mean_ret = np.mean(signal_returns)
        std_ret = np.std(signal_returns)
        result.sharpe_ratio = float(mean_ret / (std_ret + 1e-9) * np.sqrt(252 / step_size))
        result.mean_return_following_signal = float(mean_ret * 100)

    # Profit factor
    gains = sum(r for r in signal_returns if r > 0)
    losses = abs(sum(r for r in signal_returns if r < 0))
    result.profit_factor = float(gains / (losses + 1e-9))

    # Max drawdown
    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - peak) / (peak + 1e-9)
    result.max_drawdown = float(drawdowns.min())

    # Confidence calibration data
    if confidences:
        conf_arr = np.array(confidences)
        correct_arr = np.array([w["correct"] for w in result.window_results]).astype(float)
        bins = np.linspace(0, 1, 6)
        conf_bins = []
        acc_bins = []
        for j in range(len(bins) - 1):
            mask = (conf_arr >= bins[j]) & (conf_arr < bins[j + 1])
            if mask.sum() > 0:
                conf_bins.append(float(conf_arr[mask].mean()))
                acc_bins.append(float(correct_arr[mask].mean()))
        result.confidence_bins = conf_bins
        result.accuracy_bins = acc_bins

    logger.info(
        "Backtest complete: %d windows, dir_acc=%.1f%%, Sharpe=%.2f, maxDD=%.1f%%",
        result.n_windows, result.directional_accuracy * 100,
        result.sharpe_ratio, result.max_drawdown * 100,
    )

    return result


# ============================================================================
# 2. ANOMALY BACKTESTING
# ============================================================================

def backtest_anomalies(
    df: pd.DataFrame,
    forward_window: int = 5,
    return_threshold: float = 0.03,
) -> dict:
    """
    Evaluate anomaly detection quality by checking what happened after each
    flagged anomaly.

    A "true positive" anomaly is one followed by a significant price move
    (|return| > threshold within forward_window days).

    Parameters
    ----------
    df : DataFrame with anomaly_ensemble and Close columns
    forward_window : days to look ahead
    return_threshold : minimum |return| for a "significant" move

    Returns dict with precision, recall, f1, and details.
    """
    if "anomaly_ensemble" not in df.columns:
        return {"precision": 0, "recall": 0, "f1": 0, "message": "No anomaly data"}

    close = df["Close"].values
    anomaly_flags = df["anomaly_ensemble"].values

    # Calculate forward returns for all points
    forward_returns = np.full(len(df), np.nan)
    for i in range(len(df) - forward_window):
        forward_returns[i] = (close[i + forward_window] / close[i]) - 1

    # "Ground truth": significant moves
    significant = np.abs(forward_returns) > return_threshold

    # Exclude last forward_window points (can't evaluate)
    eval_mask = np.isfinite(forward_returns)

    anomaly_eval = anomaly_flags[eval_mask].astype(bool)
    significant_eval = significant[eval_mask]

    tp = int(np.sum(anomaly_eval & significant_eval))
    fp = int(np.sum(anomaly_eval & ~significant_eval))
    fn = int(np.sum(~anomaly_eval & significant_eval))
    tn = int(np.sum(~anomaly_eval & ~significant_eval))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Post-anomaly return analysis
    anomaly_indices = np.where(anomaly_eval)[0]
    post_anomaly_returns = forward_returns[eval_mask][anomaly_indices]
    valid_returns = post_anomaly_returns[np.isfinite(post_anomaly_returns)]

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_anomalies": int(anomaly_eval.sum()),
        "total_significant_moves": int(significant_eval.sum()),
        "avg_post_anomaly_return": float(np.mean(np.abs(valid_returns))) if len(valid_returns) > 0 else 0.0,
        "median_post_anomaly_return": float(np.median(np.abs(valid_returns))) if len(valid_returns) > 0 else 0.0,
        "threshold": return_threshold,
        "forward_window": forward_window,
    }


# ============================================================================
# 3. SIGNAL QUALITY ASSESSMENT
# ============================================================================

def assess_signal_quality(backtest: BacktestResult) -> dict:
    """
    Provide a qualitative assessment of signal quality based on backtest results.

    Returns dict with:
      - grade: "A" | "B" | "C" | "D" | "F"
      - assessment: human-readable summary
      - strengths: list of strengths
      - weaknesses: list of weaknesses
      - recommendations: list of recommendations
    """
    strengths = []
    weaknesses = []
    recommendations = []
    score = 0.0

    # Directional accuracy (weight: 30%)
    da = backtest.directional_accuracy
    if da >= 0.6:
        strengths.append(f"Jó iránybecslés ({da:.0%})")
        score += 30 * min(da, 0.8) / 0.8
    elif da >= 0.5:
        score += 15
    else:
        weaknesses.append(f"Gyenge iránybecslés ({da:.0%})")
        recommendations.append("Több feature vagy más modell szükséges az irány-predikció javításához")

    # Sharpe ratio (weight: 25%)
    sr = backtest.sharpe_ratio
    if sr >= 1.5:
        strengths.append(f"Kiváló Sharpe ratio ({sr:.2f})")
        score += 25
    elif sr >= 0.5:
        strengths.append(f"Elfogadható Sharpe ratio ({sr:.2f})")
        score += 15
    elif sr >= 0:
        score += 5
    else:
        weaknesses.append(f"Negatív Sharpe ratio ({sr:.2f})")
        recommendations.append("A jelzések nem hoznak hozzáadott értéket — kockázatkezelés szükséges")

    # Max drawdown (weight: 20%)
    dd = abs(backtest.max_drawdown)
    if dd < 0.1:
        strengths.append(f"Alacsony max drawdown ({dd:.0%})")
        score += 20
    elif dd < 0.2:
        score += 10
    else:
        weaknesses.append(f"Magas max drawdown ({dd:.0%})")
        recommendations.append("Pozícióméret csökkentése vagy stop-loss bevezetése javasolt")

    # Profit factor (weight: 15%)
    pf = backtest.profit_factor
    if pf >= 2.0:
        strengths.append(f"Kiváló profit faktor ({pf:.2f})")
        score += 15
    elif pf >= 1.2:
        score += 8
    elif pf >= 1.0:
        score += 3
    else:
        weaknesses.append(f"Profit faktor < 1 ({pf:.2f}) — nettó veszteséges")

    # Sample size (weight: 10%)
    if backtest.n_windows >= 50:
        score += 10
    elif backtest.n_windows >= 20:
        score += 5
        recommendations.append("Több adat szükséges a statisztikailag megbízható kiértékeléshez")
    else:
        weaknesses.append("Kevés kiértékelési ablak — az eredmények nem megbízhatók")
        recommendations.append("Hosszabb historikus időszak használata javasolt")

    # Grade
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 50:
        grade = "C"
    elif score >= 30:
        grade = "D"
    else:
        grade = "F"

    assessment = {
        "A": "Kiváló jelminőség — megbízható, konzisztens szignálok",
        "B": "Jó jelminőség — használható, de van fejlődési lehetőség",
        "C": "Közepes jelminőség — óvatossággal használható",
        "D": "Gyenge jelminőség — jelentős javítás szükséges",
        "F": "Nem megfelelő — a jelzések nem megbízhatók",
    }

    return {
        "grade": grade,
        "score": round(score, 1),
        "assessment": assessment[grade],
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recommendations,
    }


# ============================================================================
# 4. QUICK BACKTEST — convenience function
# ============================================================================

def run_quick_backtest(
    df: pd.DataFrame,
    ticker: str = "",
    horizon: int = 5,
    models: list[str] | None = None,
) -> dict:
    """
    Run a complete backtest using the standard forecast pipeline.

    Returns dict with:
      - forecast_backtest: BacktestResult
      - anomaly_backtest: dict
      - signal_quality: dict
    """
    from app.forecasting import forecast_ensemble

    if models is None:
        models = ["arima", "xgboost"]

    def forecast_fn(train_df):
        result, _ = forecast_ensemble(train_df, horizon=horizon, models=models)
        return result

    # Forecast backtest
    bt = walk_forward_forecast(
        df, forecast_fn,
        window_size=252,
        step_size=horizon,
        horizon=horizon,
        min_train_size=200,
    )
    bt.ticker = ticker

    # Anomaly backtest
    anom_bt = backtest_anomalies(df)

    bt.anomaly_precision = anom_bt.get("precision", 0)
    bt.anomaly_recall = anom_bt.get("recall", 0)
    bt.anomaly_f1 = anom_bt.get("f1", 0)
    bt.anomaly_count = anom_bt.get("total_anomalies", 0)

    # Signal quality
    quality = assess_signal_quality(bt)

    return {
        "forecast_backtest": bt,
        "anomaly_backtest": anom_bt,
        "signal_quality": quality,
    }
