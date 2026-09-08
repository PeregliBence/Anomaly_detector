"""
Forecasting module for stock price prediction.

Models implemented:
  - ARIMA: classical statistical model with confidence intervals
  - XGBoost: gradient boosted trees with time-series feature engineering
  - LSTM: recurrent neural network for sequential pattern learning
  - Ensemble: combines models for robust directional prediction

All models provide probability/confidence-based outputs, never deterministic claims.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from app.config import logger

# Conditional imports for optional dependencies
try:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models as keras_models
    HAS_TF = True
except ImportError:
    HAS_TF = False


# ============================================================================
# FORECAST RESULT
# ============================================================================

@dataclass
class ForecastResult:
    """Container for forecast outputs."""
    dates: list                   # future dates
    values: np.ndarray            # predicted prices
    lower: np.ndarray             # lower confidence bound
    upper: np.ndarray             # upper confidence bound
    direction: str                # "up", "down", "sideways"
    confidence: float             # 0.0 - 1.0
    model_name: str
    metadata: dict = field(default_factory=dict)

    @property
    def expected_return_pct(self) -> float:
        if len(self.values) >= 2 and self.values[0] > 0:
            return (self.values[-1] / self.values[0] - 1) * 100
        return 0.0


# ============================================================================
# 1. ARIMA FORECAST
# ============================================================================

def forecast_arima(
    df: pd.DataFrame,
    horizon: int = 5,
    max_p: int = 5,
    max_q: int = 5,
    confidence: float = 0.95,
) -> Optional[ForecastResult]:
    """
    ARIMA előrejelzés automatikus paraméterválasztással (AIC minimalizálás).

    Mire jó: rövid távú lineáris trendek és autokorrelációs minták megragadása.
    Limitáció: nem lineáris mintákat nem kezel jól, stacionaritást igényel.
    """
    if not HAS_STATSMODELS:
        logger.warning("statsmodels not installed, skipping ARIMA forecast")
        return None

    close = df["Close"].astype(float).dropna()
    if len(close) < 60:
        logger.warning("Not enough data for ARIMA (need 60+, got %d)", len(close))
        return None

    # Use log prices for better stationarity.
    # ARIMA receives a numeric index to avoid repeated date-frequency warnings
    # during walk-forward backtests where trading calendars are irregular.
    log_prices = pd.Series(np.log(close.values), index=pd.RangeIndex(len(close)))

    best_aic = np.inf
    best_order = (1, 1, 1)

    # Grid search over small parameter space
    max_p_search = min(max_p, 3)
    max_q_search = min(max_q, 3)
    for p in range(max_p_search + 1):
        for q in range(max_q_search + 1):
            try:
                model = _ARIMA(log_prices, order=(p, 1, q))
                result = model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, 1, q)
            except Exception:
                continue

    try:
        model = _ARIMA(log_prices, order=best_order)
        result = model.fit()
        forecast_obj = result.get_forecast(steps=horizon)
        pred_log = forecast_obj.predicted_mean
        ci = forecast_obj.conf_int(alpha=1 - confidence)

        pred_prices = np.exp(pred_log.values)
        lower = np.exp(ci.iloc[:, 0].values)
        upper = np.exp(ci.iloc[:, 1].values)

        # Generate future dates (business days)
        last_date = close.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

        # Determine direction
        last_price = float(close.iloc[-1])
        final_pred = float(pred_prices[-1])
        pct_change = (final_pred / last_price - 1)

        if pct_change > 0.005:
            direction = "up"
        elif pct_change < -0.005:
            direction = "down"
        else:
            direction = "sideways"

        # Confidence based on prediction interval width relative to price
        avg_width = np.mean(upper - lower)
        relative_width = avg_width / last_price
        conf = max(0.1, min(0.95, 1.0 - relative_width * 5))

        return ForecastResult(
            dates=list(future_dates),
            values=pred_prices,
            lower=lower,
            upper=upper,
            direction=direction,
            confidence=round(conf, 3),
            model_name="ARIMA",
            metadata={"order": best_order, "aic": best_aic},
        )

    except Exception as e:
        logger.error("ARIMA forecast failed: %s", e)
        return None


# ============================================================================
# 2. XGBOOST FORECAST
# ============================================================================

def _create_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build supervised learning features for XGBoost."""
    feat = pd.DataFrame(index=df.index)

    # Lag returns
    for lag in [1, 2, 3, 5, 10, 20]:
        feat[f"ret_lag_{lag}"] = df["log_return"].shift(lag)

    # Rolling statistics
    for w in [5, 10, 20]:
        feat[f"ret_mean_{w}"] = df["log_return"].rolling(w).mean()
        feat[f"ret_std_{w}"] = df["log_return"].rolling(w).std()
        feat[f"ret_min_{w}"] = df["log_return"].rolling(w).min()
        feat[f"ret_max_{w}"] = df["log_return"].rolling(w).max()

    # Technical indicators (if available)
    for col in ["RSI_14", "MACD_hist", "BB_pct", "ATR_14", "volume_spike", "drawdown"]:
        if col in df.columns:
            feat[col] = df[col]

    # Calendar
    feat["day_of_week"] = df.index.dayofweek
    feat["month"] = df.index.month

    # Trend distance
    for col in ["dist_SMA_20", "dist_SMA_50"]:
        if col in df.columns:
            feat[col] = df[col]

    return feat


def forecast_xgboost(
    df: pd.DataFrame,
    horizon: int = 5,
    n_estimators: int = 200,
    n_bootstrap: int = 10,
) -> Optional[ForecastResult]:
    """
    XGBoost előrejelzés time-series feature engineeringgel.

    Mire jó: komplex nemlineáris mintázatok tanulása feature-ökből, gyors tanítás.
    Limitáció: nem natívan szekvenciális, recursive forecast bias felhalmozódhat.
    """
    if not HAS_XGBOOST:
        logger.warning("xgboost not installed, skipping XGBoost forecast")
        return None

    if "log_return" not in df.columns:
        return None

    features = _create_xgb_features(df)
    target = df["log_return"].shift(-1)  # next-day return

    combined = features.join(target.rename("target")).dropna()
    if len(combined) < 100:
        logger.warning("Not enough data for XGBoost (got %d)", len(combined))
        return None

    X = combined.drop(columns=["target"]).values
    y = combined["target"].values

    # Time-series split: proper train/validation to avoid data leakage
    split = max(int(0.8 * len(X)), len(X) - max(horizon * 5, 50))
    split = min(split, len(X) - horizon)  # ensure at least 'horizon' test points
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Bootstrap ensemble for uncertainty estimation
    all_preds = []
    for i in range(n_bootstrap):
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + i,
            verbosity=0,
        )
        # Bootstrap sampling of training data
        rng = np.random.RandomState(42 + i)
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        model.fit(X_train[idx], y_train[idx])

        # Recursive multi-step forecast
        preds = []
        last_features = X[-1:].copy()
        for step in range(horizon):
            pred = model.predict(last_features)[0]
            preds.append(pred)
            # Properly shift lag features: newest pred becomes lag_1, old lags shift right
            new_features = last_features[0].copy()
            # ret_lag columns are at indices 0..5 (lags 1,2,3,5,10,20)
            # Shift: lag_20 <- lag_10, lag_10 <- lag_5, lag_5 <- lag_3, lag_3 <- lag_2, lag_2 <- lag_1, lag_1 <- pred
            new_features[5] = new_features[4]  # ret_lag_20 <- ret_lag_10
            new_features[4] = new_features[3]  # ret_lag_10 <- ret_lag_5
            new_features[3] = new_features[2]  # ret_lag_5 <- ret_lag_3
            new_features[2] = new_features[1]  # ret_lag_3 <- ret_lag_2
            new_features[1] = new_features[0]  # ret_lag_2 <- ret_lag_1
            new_features[0] = pred              # ret_lag_1 <- current prediction
            last_features = new_features.reshape(1, -1)
        all_preds.append(preds)

    all_preds = np.array(all_preds)
    mean_returns = all_preds.mean(axis=0)
    std_returns = all_preds.std(axis=0)

    # Convert returns to prices
    last_price = float(df["Close"].iloc[-1])
    cum_returns = np.cumsum(mean_returns)
    pred_prices = last_price * np.exp(cum_returns)

    cum_lower = np.cumsum(mean_returns - 1.96 * std_returns)
    cum_upper = np.cumsum(mean_returns + 1.96 * std_returns)
    lower_prices = last_price * np.exp(cum_lower)
    upper_prices = last_price * np.exp(cum_upper)

    # Future dates
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

    # Direction
    total_return = cum_returns[-1]
    if total_return > 0.005:
        direction = "up"
    elif total_return < -0.005:
        direction = "down"
    else:
        direction = "sideways"

    # Confidence: based on model agreement
    signs = np.sign(all_preds.sum(axis=1))
    agreement = np.mean(signs == np.sign(total_return))
    conf = max(0.1, min(0.9, agreement))

    return ForecastResult(
        dates=list(future_dates),
        values=pred_prices,
        lower=lower_prices,
        upper=upper_prices,
        direction=direction,
        confidence=round(conf, 3),
        model_name="XGBoost",
        metadata={"n_bootstrap": n_bootstrap, "mean_return": float(total_return)},
    )


# ============================================================================
# 3. LSTM FORECAST
# ============================================================================

def forecast_lstm(
    df: pd.DataFrame,
    horizon: int = 5,
    lookback: int = 30,
    epochs: int = 20,
    units: int = 64,
) -> Optional[ForecastResult]:
    """
    LSTM alapú előrejelzés.

    Mire jó: szekvenciális mintázatok (momentum, mean reversion) tanulása.
    Limitáció: számításigényes, overfitting kis adaton, aktívan hangolni kell.
    """
    if not HAS_TF:
        logger.warning("TensorFlow not available, skipping LSTM forecast")
        return None

    if "log_return" not in df.columns or len(df) < lookback + 60:
        return None

    # Prepare features
    feat_cols = ["log_return"]
    for col in ["vol_20", "RSI_14", "MACD_hist", "BB_pct", "volume_spike"]:
        if col in df.columns:
            feat_cols.append(col)

    data = df[feat_cols].dropna().values.astype(np.float32)
    if len(data) < lookback + 20:
        return None

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled) - 1):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i, 0])  # predict next log_return
    X, y = np.array(X), np.array(y)

    # Train/val split
    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    tf.keras.utils.set_random_seed(42)

    with tf.device("/CPU:0"):
        model = tf.keras.Sequential([
            layers.LSTM(units, return_sequences=True, input_shape=(lookback, len(feat_cols))),
            layers.Dropout(0.2),
            layers.LSTM(units // 2, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=32,
            shuffle=False, verbose=0,
        )

        # Multi-step recursive forecast with MC dropout for uncertainty
        n_mc = 20
        all_forecasts = []

        for _ in range(n_mc):
            current_seq = scaled[-lookback:].copy()
            preds = []
            for _ in range(horizon):
                inp = current_seq.reshape(1, lookback, len(feat_cols))
                pred = model(inp, training=True).numpy()[0, 0]  # training=True enables dropout
                preds.append(pred)
                # Roll sequence forward
                new_row = current_seq[-1].copy()
                new_row[0] = pred
                current_seq = np.vstack([current_seq[1:], new_row])
            all_forecasts.append(preds)

    all_forecasts = np.array(all_forecasts)

    # Inverse scale the return predictions
    mean_preds = all_forecasts.mean(axis=0)
    std_preds = all_forecasts.std(axis=0)

    # Convert scaled returns back
    ret_mean = scaler.mean_[0]
    ret_scale = scaler.scale_[0]
    mean_returns = mean_preds * ret_scale + ret_mean
    std_returns = std_preds * ret_scale

    # Convert to prices
    last_price = float(df["Close"].iloc[-1])
    cum_returns = np.cumsum(mean_returns)
    pred_prices = last_price * np.exp(cum_returns)

    lower_prices = last_price * np.exp(np.cumsum(mean_returns - 1.96 * std_returns))
    upper_prices = last_price * np.exp(np.cumsum(mean_returns + 1.96 * std_returns))

    # Future dates
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

    # Direction & confidence
    total_return = cum_returns[-1]
    if total_return > 0.005:
        direction = "up"
    elif total_return < -0.005:
        direction = "down"
    else:
        direction = "sideways"

    directional_agreement = np.mean(np.sign(all_forecasts.sum(axis=1)) == np.sign(total_return))
    conf = max(0.1, min(0.85, directional_agreement * 0.9))

    return ForecastResult(
        dates=list(future_dates),
        values=pred_prices,
        lower=lower_prices,
        upper=upper_prices,
        direction=direction,
        confidence=round(conf, 3),
        model_name="LSTM",
        metadata={"n_mc_samples": n_mc, "lookback": lookback},
    )


# ============================================================================
# 4. DIRECTION CLASSIFIER
# ============================================================================

def classify_direction(
    df: pd.DataFrame,
    horizon: int = 5,
) -> dict:
    """
    Irány-klasszifikáció: up / down / sideways valószínűségi becslés.
    Több jelzés kombinálásával ad probabilisztikus eredményt.

    Returns dict with keys: direction, probabilities, signals.
    """
    signals = {}
    close = df["Close"]
    last_price = float(close.iloc[-1])

    # 1. Trend (SMA position)
    if "SMA_50" in df.columns and "SMA_200" in df.columns:
        sma50 = float(df["SMA_50"].iloc[-1])
        sma200 = float(df["SMA_200"].iloc[-1])
        if last_price > sma50 > sma200:
            signals["trend"] = 1.0
        elif last_price < sma50 < sma200:
            signals["trend"] = -1.0
        else:
            signals["trend"] = 0.0

    # 2. RSI
    if "RSI_14" in df.columns:
        rsi = float(df["RSI_14"].iloc[-1])
        if rsi > 70:
            signals["rsi"] = -0.5  # overbought → potential down
        elif rsi < 30:
            signals["rsi"] = 0.5   # oversold → potential up
        else:
            signals["rsi"] = (50 - rsi) / 100  # mild signal

    # 3. MACD
    if "MACD_hist" in df.columns:
        macd_hist = float(df["MACD_hist"].iloc[-1])
        macd_hist_prev = float(df["MACD_hist"].iloc[-2]) if len(df) > 1 else 0
        if macd_hist > 0 and macd_hist > macd_hist_prev:
            signals["macd"] = 0.7
        elif macd_hist < 0 and macd_hist < macd_hist_prev:
            signals["macd"] = -0.7
        else:
            signals["macd"] = np.sign(macd_hist) * 0.3

    # 4. Momentum
    if "momentum_5" in df.columns:
        mom = float(df["momentum_5"].iloc[-1])
        signals["momentum"] = np.clip(mom / (last_price * 0.05 + 1e-9), -1, 1)

    # 5. Bollinger Band position
    if "BB_pct" in df.columns:
        bb = float(df["BB_pct"].iloc[-1])
        if bb > 0.95:
            signals["bollinger"] = -0.5  # near upper band
        elif bb < 0.05:
            signals["bollinger"] = 0.5   # near lower band
        else:
            signals["bollinger"] = 0.0

    # 6. Recent volatility trend
    if "vol_20" in df.columns and "vol_60" in df.columns:
        vol_short = float(df["vol_20"].iloc[-1])
        vol_long = float(df["vol_60"].iloc[-1])
        if vol_short > vol_long * 1.5:
            signals["vol_expansion"] = -0.3  # high vol = uncertainty
        else:
            signals["vol_expansion"] = 0.1

    if not signals:
        return {"direction": "sideways", "probabilities": {"up": 0.33, "down": 0.33, "sideways": 0.34}, "signals": {}}

    # Aggregate signals
    avg_signal = np.mean(list(signals.values()))

    # Convert to probabilities using softmax-like approach
    up_score = max(0, avg_signal) + 0.2
    down_score = max(0, -avg_signal) + 0.2
    side_score = 0.3 + 0.2 * (1 - abs(avg_signal))

    total = up_score + down_score + side_score
    probs = {
        "up": round(up_score / total, 3),
        "down": round(down_score / total, 3),
        "sideways": round(side_score / total, 3),
    }

    direction = max(probs, key=probs.get)

    return {"direction": direction, "probabilities": probs, "signals": signals}


# ============================================================================
# 5. ENSEMBLE FORECAST
# ============================================================================

def forecast_ensemble(
    df: pd.DataFrame,
    horizon: int = 5,
    models: list[str] | None = None,
) -> tuple[Optional[ForecastResult], list[ForecastResult]]:
    """
    Ensemble forecast: futtat több modellt és kombinálja az eredményeket.

    Returns
    -------
    ensemble_result : ForecastResult or None
    individual_results : list of ForecastResult
    """
    if models is None:
        models = ["arima", "xgboost", "lstm"]

    results = []

    if "arima" in models:
        r = forecast_arima(df, horizon=horizon)
        if r is not None:
            results.append(r)

    if "xgboost" in models:
        r = forecast_xgboost(df, horizon=horizon)
        if r is not None:
            results.append(r)

    if "lstm" in models:
        r = forecast_lstm(df, horizon=horizon)
        if r is not None:
            results.append(r)

    if not results:
        return None, []

    if len(results) == 1:
        return results[0], results

    # Combine forecasts: weighted average by confidence
    weights = np.array([r.confidence for r in results])
    weights /= weights.sum()

    # Ensure all results have the same number of steps
    min_steps = min(len(r.values) for r in results)
    combined_values = np.zeros(min_steps)
    combined_lower = np.zeros(min_steps)
    combined_upper = np.zeros(min_steps)

    for r, w in zip(results, weights):
        combined_values += r.values[:min_steps] * w
        combined_lower += r.lower[:min_steps] * w
        combined_upper += r.upper[:min_steps] * w

    # Direction from consensus
    directions = [r.direction for r in results]
    dir_counts = {"up": 0, "down": 0, "sideways": 0}
    for d, w in zip(directions, weights):
        dir_counts[d] += w
    direction = max(dir_counts, key=dir_counts.get)

    # Confidence: based on agreement
    agreement = dir_counts[direction]
    avg_conf = np.average([r.confidence for r in results], weights=weights)
    final_conf = min(avg_conf * agreement, 0.9)

    ensemble = ForecastResult(
        dates=results[0].dates[:min_steps],
        values=combined_values,
        lower=combined_lower,
        upper=combined_upper,
        direction=direction,
        confidence=round(final_conf, 3),
        model_name="Ensemble",
        metadata={
            "models_used": [r.model_name for r in results],
            "weights": {r.model_name: round(w, 3) for r, w in zip(results, weights)},
        },
    )

    return ensemble, results


# ============================================================================
# 6. FORECAST RELIABILITY ASSESSMENT
# ============================================================================

def assess_forecast_reliability(df: pd.DataFrame) -> dict:
    """
    Evaluates whether the current market conditions support reliable forecasting.

    Returns dict with:
    - reliable: bool
    - risk_level: "low" / "medium" / "high"
    - warnings: list[str]
    - volatility_state: str
    """
    warnings = []

    # Check data sufficiency
    if len(df) < 100:
        warnings.append("Kevés történelmi adat áll rendelkezésre (< 100 nap).")

    # Check recent volatility
    vol_state = "normal"
    if "vol_20" in df.columns and "vol_60" in df.columns:
        vol_short = float(df["vol_20"].iloc[-1])
        vol_long = float(df["vol_60"].iloc[-1])
        if vol_short > vol_long * 2:
            vol_state = "elevated"
            warnings.append("Emelkedett volatilitás — az előrejelzés megbízhatósága csökken.")
        if vol_short > vol_long * 3:
            vol_state = "extreme"
            warnings.append("Extrém volatilitás — az előrejelzés nagyon bizonytalan!")

    # Check for recent anomalies
    if "anomaly_ensemble" in df.columns:
        recent = df["anomaly_ensemble"].tail(5).sum()
        if recent >= 2:
            warnings.append("Több anomália az elmúlt napokban — instabil piaci környezet.")

    # Risk level
    if vol_state == "extreme" or len(warnings) >= 3:
        risk_level = "high"
    elif vol_state == "elevated" or len(warnings) >= 2:
        risk_level = "medium"
    else:
        risk_level = "low"

    reliable = risk_level != "high"

    return {
        "reliable": reliable,
        "risk_level": risk_level,
        "warnings": warnings,
        "volatility_state": vol_state,
    }
