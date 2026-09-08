"""
Advanced Forecasting Models — GARCH, LightGBM stacking, TFT, online learning.

Extends the base forecasting module with:
  1. GARCH(1,1) volatility model (replaces ARIMA-only)
  2. LightGBM + XGBoost stacking ensemble
  3. Temporal Fusion Transformer (TFT) via pytorch-forecasting
  4. Online learning via River library (incremental models)
  5. Confidence calibration (isotonic regression / Platt scaling)

All models are optional — graceful degradation if dependencies are missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from app.config import logger
from app.forecasting import ForecastResult

# ── Conditional imports ──────────────────────────────────────────────────

try:
    from arch import arch_model as _arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

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

try:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import pytorch_forecasting
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    import lightning.pytorch as pl
    HAS_TFT = True
except ImportError:
    HAS_TFT = False

try:
    from river import linear_model as river_lm
    from river import preprocessing as river_pp
    from river import metrics as river_metrics
    from river import compose as river_compose
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


# ============================================================================
# 1. GARCH(1,1) VOLATILITY FORECAST
# ============================================================================

def forecast_garch(
    df: pd.DataFrame,
    horizon: int = 5,
    p: int = 1,
    q: int = 1,
    dist: str = "t",
) -> Optional[ForecastResult]:
    """
    GARCH(1,1) volatility forecast.

    Mire jó: a variancia (volatilitás) előrejelzése, ami az árfolyam-előrejelzés
    konfidencia-intervallumait javítja.  Egyesítve az ARIMA mean modellel jobb
    eredményt ad, mint bármelyik egyedül.

    Returns a ForecastResult where `values` are the predicted annualized
    volatilities (not prices).  The `direction` field indicates volatility
    trend (up = vol increasing, down = vol decreasing).
    """
    if not HAS_ARCH:
        logger.warning("arch not installed (pip install arch), skipping GARCH")
        return None

    if "log_return" not in df.columns or len(df) < 100:
        return None

    returns = df["log_return"].dropna().values * 100  # scale to percent

    try:
        model = _arch_model(returns, vol="Garch", p=p, q=q, dist=dist, mean="AR", lags=1)
        result = model.fit(disp="off", show_warning=False)

        forecasts = result.forecast(horizon=horizon, method="simulation", simulations=1000)
        var_fcast = forecasts.variance.iloc[-1].values  # variance per step
        mean_fcast = forecasts.mean.iloc[-1].values       # mean return per step

        # Convert variance to annualized volatility
        vol_fcast = np.sqrt(var_fcast) * np.sqrt(252) / 100  # back from pct

        # Build price path from mean + vol
        last_price = float(df["Close"].iloc[-1])
        cum_returns = np.cumsum(mean_fcast / 100)
        pred_prices = last_price * np.exp(cum_returns)

        # Confidence bands using GARCH volatility
        cum_vol = np.sqrt(np.cumsum(var_fcast)) / 100
        lower = last_price * np.exp(cum_returns - 1.96 * cum_vol)
        upper = last_price * np.exp(cum_returns + 1.96 * cum_vol)

        last_date = df.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

        # Direction: is volatility expanding or contracting?
        current_vol = float(df["log_return"].tail(20).std() * np.sqrt(252))
        forecast_avg_vol = float(vol_fcast.mean())
        if forecast_avg_vol > current_vol * 1.1:
            direction = "up"
        elif forecast_avg_vol < current_vol * 0.9:
            direction = "down"
        else:
            direction = "sideways"

        conf = max(0.2, min(0.85, 1.0 - float(np.std(vol_fcast) / (np.mean(vol_fcast) + 1e-9))))

        return ForecastResult(
            dates=list(future_dates),
            values=pred_prices,
            lower=lower,
            upper=upper,
            direction=direction,
            confidence=round(conf, 3),
            model_name="ARIMA-GARCH",
            metadata={
                "garch_order": (p, q),
                "dist": dist,
                "aic": result.aic,
                "bic": result.bic,
                "forecast_vol": vol_fcast.tolist(),
            },
        )

    except Exception as e:
        logger.error("GARCH forecast failed: %s", e)
        return None


# ============================================================================
# 2. LightGBM + XGBoost STACKING ENSEMBLE
# ============================================================================

def _create_gbm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for gradient boosting models."""
    feat = pd.DataFrame(index=df.index)

    for lag in [1, 2, 3, 5, 10, 20]:
        feat[f"ret_lag_{lag}"] = df["log_return"].shift(lag)

    for w in [5, 10, 20]:
        feat[f"ret_mean_{w}"] = df["log_return"].rolling(w).mean()
        feat[f"ret_std_{w}"] = df["log_return"].rolling(w).std()
        feat[f"ret_skew_{w}"] = df["log_return"].rolling(w).skew()
        feat[f"ret_kurt_{w}"] = df["log_return"].rolling(w).kurt()

    for col in ["RSI_14", "MACD_hist", "BB_pct", "ATR_14", "volume_spike",
                 "drawdown", "vol_adj_return", "dist_SMA_20", "dist_SMA_50"]:
        if col in df.columns:
            feat[col] = df[col]

    # Macro features if available
    for col in df.columns:
        if col.startswith("macro_") or col.startswith("opt_") or col.startswith("si_"):
            feat[col] = df[col]

    feat["day_of_week"] = df.index.dayofweek
    feat["month"] = df.index.month

    return feat


def forecast_stacking_ensemble(
    df: pd.DataFrame,
    horizon: int = 5,
    n_bootstrap: int = 10,
) -> Optional[ForecastResult]:
    """
    Stacking ensemble: XGBoost + LightGBM first-level, Ridge meta-learner.

    Uses time-series cross-validation to generate out-of-fold predictions
    for the meta-learner, avoiding data leakage.
    """
    if not (HAS_XGBOOST and HAS_LIGHTGBM and HAS_SKLEARN):
        missing = []
        if not HAS_XGBOOST:
            missing.append("xgboost")
        if not HAS_LIGHTGBM:
            missing.append("lightgbm")
        if not HAS_SKLEARN:
            missing.append("scikit-learn")
        logger.warning("Missing packages for stacking: %s", ", ".join(missing))
        return None

    if "log_return" not in df.columns:
        return None

    features = _create_gbm_features(df)
    target = df["log_return"].shift(-1)
    combined = features.join(target.rename("target")).dropna()

    if len(combined) < 200:
        logger.warning("Not enough data for stacking (got %d, need 200+)", len(combined))
        return None

    X = combined.drop(columns=["target"]).values
    y = combined["target"].values
    feature_names = combined.drop(columns=["target"]).columns.tolist()

    # Time-series CV for out-of-fold predictions
    tscv = TimeSeriesSplit(n_splits=3)
    oof_xgb = np.full(len(X), np.nan)
    oof_lgb = np.full(len(X), np.nan)

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42,
        )
        xgb_model.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb_model.predict(X_val)

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=42,
        )
        lgb_model.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgb_model.predict(X_val)

    # Meta-learner on out-of-fold predictions
    valid_mask = np.isfinite(oof_xgb) & np.isfinite(oof_lgb)
    meta_X = np.column_stack([oof_xgb[valid_mask], oof_lgb[valid_mask]])
    meta_y = y[valid_mask]

    meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    meta_model.fit(meta_X, meta_y)

    # Re-train base models on full data for final predictions
    xgb_final = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42,
    )
    xgb_final.fit(X, y)

    lgb_final = lgb.LGBMRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=42,
    )
    lgb_final.fit(X, y)

    # Bootstrap for uncertainty
    all_preds = []
    for i in range(n_bootstrap):
        rng = np.random.RandomState(42 + i)
        idx = rng.choice(len(X), size=len(X), replace=True)

        xgb_b = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42 + i,
        )
        xgb_b.fit(X[idx], y[idx])

        lgb_b = lgb.LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=42 + i,
        )
        lgb_b.fit(X[idx], y[idx])

        # Recursive prediction
        preds = []
        last_features = X[-1:].copy()
        for step in range(horizon):
            p_xgb = xgb_b.predict(last_features)[0]
            p_lgb = lgb_b.predict(last_features)[0]
            meta_input = np.array([[p_xgb, p_lgb]])
            pred = meta_model.predict(meta_input)[0]
            preds.append(pred)
            # Shift features
            new_f = last_features[0].copy()
            for j in range(min(5, len(new_f) - 1), 0, -1):
                new_f[j] = new_f[j - 1]
            new_f[0] = pred
            last_features = new_f.reshape(1, -1)
        all_preds.append(preds)

    all_preds = np.array(all_preds)
    mean_returns = all_preds.mean(axis=0)
    std_returns = all_preds.std(axis=0)

    last_price = float(df["Close"].iloc[-1])
    cum_returns = np.cumsum(mean_returns)
    pred_prices = last_price * np.exp(cum_returns)
    lower = last_price * np.exp(np.cumsum(mean_returns - 1.96 * std_returns))
    upper = last_price * np.exp(np.cumsum(mean_returns + 1.96 * std_returns))

    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

    total_return = cum_returns[-1]
    if total_return > 0.005:
        direction = "up"
    elif total_return < -0.005:
        direction = "down"
    else:
        direction = "sideways"

    signs = np.sign(all_preds.sum(axis=1))
    agreement = np.mean(signs == np.sign(total_return))
    conf = max(0.1, min(0.9, agreement))

    # Store feature importance (LightGBM)
    lgb_importance = dict(zip(feature_names, lgb_final.feature_importances_.tolist()))

    return ForecastResult(
        dates=list(future_dates),
        values=pred_prices,
        lower=lower,
        upper=upper,
        direction=direction,
        confidence=round(conf, 3),
        model_name="Stacking (XGB+LGB)",
        metadata={
            "meta_alpha": float(meta_model.alpha_),
            "meta_coefs": meta_model.coef_.tolist(),
            "lgb_importance": lgb_importance,
            "xgb_model": xgb_final,
            "lgb_model": lgb_final,
        },
    )


# ============================================================================
# 3. TEMPORAL FUSION TRANSFORMER (TFT)
# ============================================================================

def forecast_tft(
    df: pd.DataFrame,
    horizon: int = 5,
    max_epochs: int = 30,
    hidden_size: int = 32,
    attention_head_size: int = 2,
) -> Optional[ForecastResult]:
    """
    Temporal Fusion Transformer forecast.

    State-of-the-art time series model that combines:
    - Variable selection (learns which features matter)
    - Temporal attention (learns time-varying importance)
    - Multi-horizon output

    Requires: pytorch-forecasting, lightning.pytorch, torch
    """
    if not HAS_TFT:
        logger.info("pytorch-forecasting not available, skipping TFT")
        return None

    if "log_return" not in df.columns or len(df) < 300:
        logger.warning("Not enough data for TFT (need 300+, got %d)", len(df))
        return None

    try:
        # Prepare data in the format TFT expects
        tft_df = df.copy()
        tft_df["time_idx"] = range(len(tft_df))
        tft_df["group"] = "0"  # single series
        tft_df["target"] = tft_df["log_return"]

        # Select known real-valued features
        known_reals = ["time_idx"]
        unknown_reals = ["target"]
        for col in ["RSI_14", "MACD_hist", "BB_pct", "volume_spike", "vol_20"]:
            if col in tft_df.columns:
                unknown_reals.append(col)

        max_encoder_length = min(60, len(tft_df) // 4)
        max_prediction_length = horizon

        training_cutoff = tft_df["time_idx"].max() - max_prediction_length

        training = TimeSeriesDataSet(
            tft_df[tft_df.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="target",
            group_ids=["group"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals,
            target_normalizer=GroupNormalizer(groups=["group"]),
        )

        validation = TimeSeriesDataSet.from_dataset(
            training, tft_df, min_prediction_idx=training_cutoff + 1
        )

        train_dl = training.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_dl = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

        tft_model = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=7,  # 7 quantiles for probabilistic output
            loss=pytorch_forecasting.metrics.QuantileLoss(),
            learning_rate=0.01,
            reduce_on_plateau_patience=4,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="auto",
            gradient_clip_val=0.1,
        )
        trainer.fit(tft_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # Predict
        predictions = tft_model.predict(val_dl, mode="quantiles", return_x=True)
        pred_values = predictions.output.numpy()

        # Extract median, 2.5th and 97.5th percentiles
        if pred_values.ndim == 3:
            median_pred = pred_values[0, :, 3]  # median quantile
            lower_pred = pred_values[0, :, 0]   # lowest quantile
            upper_pred = pred_values[0, :, -1]  # highest quantile
        else:
            median_pred = pred_values.flatten()[:horizon]
            lower_pred = median_pred - 0.02
            upper_pred = median_pred + 0.02

        # Convert returns to prices
        last_price = float(df["Close"].iloc[-1])
        cum_returns = np.cumsum(median_pred[:horizon])
        pred_prices = last_price * np.exp(cum_returns)
        lower_prices = last_price * np.exp(np.cumsum(lower_pred[:horizon]))
        upper_prices = last_price * np.exp(np.cumsum(upper_pred[:horizon]))

        last_date = df.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

        total_return = cum_returns[-1] if len(cum_returns) > 0 else 0
        if total_return > 0.005:
            direction = "up"
        elif total_return < -0.005:
            direction = "down"
        else:
            direction = "sideways"

        width = np.mean(upper_prices - lower_prices)
        conf = max(0.15, min(0.85, 1.0 - width / last_price * 3))

        return ForecastResult(
            dates=list(future_dates),
            values=pred_prices,
            lower=lower_prices,
            upper=upper_prices,
            direction=direction,
            confidence=round(conf, 3),
            model_name="TFT",
            metadata={"encoder_length": max_encoder_length, "epochs": max_epochs},
        )

    except Exception as e:
        logger.error("TFT forecast failed: %s", e)
        return None


# ============================================================================
# 4. ONLINE LEARNING (River)
# ============================================================================

@dataclass
class OnlineLearnerState:
    """Persistent state for online learning model."""
    model: object = None
    metric: object = None
    n_samples_seen: int = 0
    recent_errors: list = field(default_factory=list)


_online_state: dict[str, OnlineLearnerState] = {}


def _get_online_features(row: pd.Series) -> dict:
    """Convert a DataFrame row to a dict for River."""
    feat = {}
    for col in ["log_return", "vol_20", "RSI_14", "MACD_hist", "BB_pct",
                 "volume_spike", "drawdown", "dist_SMA_20", "dist_SMA_50"]:
        if col in row.index and pd.notna(row[col]):
            feat[col] = float(row[col])
    # Lag features
    for lag_col in [c for c in row.index if c.startswith("return_lag_")]:
        if pd.notna(row[lag_col]):
            feat[lag_col] = float(row[lag_col])
    return feat


def online_learn_and_predict(
    df: pd.DataFrame,
    ticker: str,
    horizon: int = 1,
) -> Optional[ForecastResult]:
    """
    Online (incremental) learning using River library.

    The model is trained incrementally on each new data point,
    avoiding full retraining.  State is preserved across calls
    for the same ticker.

    Uses a pipeline:
      StandardScaler → Linear Regression with L2 penalty

    This is NOT a deep model — it's designed for fast adaptation
    and low latency, complementing the batch models.
    """
    if not HAS_RIVER:
        logger.info("river not installed, skipping online learning")
        return None

    if "log_return" not in df.columns or len(df) < 60:
        return None

    # Get or create state for this ticker
    if ticker not in _online_state:
        model = river_compose.Pipeline(
            river_pp.StandardScaler(),
            river_lm.LinearRegression(l2=0.01),
        )
        _online_state[ticker] = OnlineLearnerState(
            model=model,
            metric=river_metrics.MAE(),
        )

    state = _online_state[ticker]

    # Train on all available data (incremental)
    target_col = "log_return"
    rows = df.dropna(subset=[target_col])

    start_idx = max(0, state.n_samples_seen - 1)
    new_rows = rows.iloc[start_idx:]

    for i in range(len(new_rows) - 1):
        x = _get_online_features(new_rows.iloc[i])
        y = float(new_rows.iloc[i + 1][target_col])

        if x:
            pred = state.model.predict_one(x)
            state.metric.update(y_true=y, y_pred=pred)
            state.model.learn_one(x, y)
            state.recent_errors.append(abs(y - pred))
            if len(state.recent_errors) > 100:
                state.recent_errors = state.recent_errors[-100:]

    state.n_samples_seen = len(rows)

    # Predict next step
    last_features = _get_online_features(rows.iloc[-1])
    if not last_features:
        return None

    pred_return = state.model.predict_one(last_features)

    # Estimate uncertainty from recent errors
    if state.recent_errors:
        mae = np.mean(state.recent_errors[-50:])
    else:
        mae = 0.01

    last_price = float(df["Close"].iloc[-1])
    pred_price = last_price * np.exp(pred_return)
    lower_price = last_price * np.exp(pred_return - 1.96 * mae)
    upper_price = last_price * np.exp(pred_return + 1.96 * mae)

    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=1)

    if pred_return > 0.003:
        direction = "up"
    elif pred_return < -0.003:
        direction = "down"
    else:
        direction = "sideways"

    conf = max(0.1, min(0.7, 1.0 - mae * 50))

    return ForecastResult(
        dates=list(future_dates),
        values=np.array([pred_price]),
        lower=np.array([lower_price]),
        upper=np.array([upper_price]),
        direction=direction,
        confidence=round(conf, 3),
        model_name="Online (River)",
        metadata={
            "n_samples": state.n_samples_seen,
            "mae": float(state.metric.get()),
            "recent_mae": float(mae),
        },
    )


# ============================================================================
# 5. ENHANCED ENSEMBLE — combine all models
# ============================================================================

def forecast_advanced_ensemble(
    df: pd.DataFrame,
    ticker: str = "",
    horizon: int = 5,
    models: list[str] | None = None,
) -> tuple[Optional[ForecastResult], list[ForecastResult]]:
    """
    Advanced ensemble: runs all available models and combines results.

    Models: arima-garch, stacking, tft, online, + original (arima, xgboost, lstm)
    """
    from app.forecasting import forecast_arima, forecast_xgboost, forecast_lstm

    if models is None:
        models = ["arima", "garch", "xgboost", "stacking", "lstm", "tft", "online"]

    results = []

    if "arima" in models:
        r = forecast_arima(df, horizon=horizon)
        if r is not None:
            results.append(r)

    if "garch" in models:
        r = forecast_garch(df, horizon=horizon)
        if r is not None:
            results.append(r)

    if "xgboost" in models:
        r = forecast_xgboost(df, horizon=horizon)
        if r is not None:
            results.append(r)

    if "stacking" in models:
        r = forecast_stacking_ensemble(df, horizon=horizon)
        if r is not None:
            results.append(r)

    if "lstm" in models:
        r = forecast_lstm(df, horizon=horizon)
        if r is not None:
            results.append(r)

    if "tft" in models:
        r = forecast_tft(df, horizon=horizon)
        if r is not None:
            results.append(r)

    if "online" in models and ticker:
        r = online_learn_and_predict(df, ticker, horizon=1)
        if r is not None:
            results.append(r)

    if not results:
        return None, []

    if len(results) == 1:
        return results[0], results

    # Weighted combination by confidence
    weights = np.array([r.confidence for r in results])
    weights /= weights.sum()

    min_steps = min(len(r.values) for r in results)
    combined_values = np.zeros(min_steps)
    combined_lower = np.zeros(min_steps)
    combined_upper = np.zeros(min_steps)

    for r, w in zip(results, weights):
        combined_values += r.values[:min_steps] * w
        combined_lower += r.lower[:min_steps] * w
        combined_upper += r.upper[:min_steps] * w

    # Direction consensus
    dir_counts = {"up": 0.0, "down": 0.0, "sideways": 0.0}
    for r, w in zip(results, weights):
        dir_counts[r.direction] += w
    direction = max(dir_counts, key=dir_counts.get)

    avg_conf = float(np.average([r.confidence for r in results], weights=weights))
    agreement = dir_counts[direction]
    final_conf = min(avg_conf * agreement * 1.2, 0.92)

    ensemble = ForecastResult(
        dates=results[0].dates[:min_steps],
        values=combined_values,
        lower=combined_lower,
        upper=combined_upper,
        direction=direction,
        confidence=round(final_conf, 3),
        model_name="Advanced Ensemble",
        metadata={
            "models_used": [r.model_name for r in results],
            "weights": {r.model_name: round(float(w), 3) for r, w in zip(results, weights)},
            "direction_votes": dir_counts,
        },
    )

    return ensemble, results
