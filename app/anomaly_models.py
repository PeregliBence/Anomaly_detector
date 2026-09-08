"""
Anomaly detection models for financial time series.

Methods implemented:
  Statistical:  Z-score (standard/robust/rolling), IQR
  ML:           Isolation Forest (single + ensemble), LOF, One-Class SVM
  Deep:         LSTM Autoencoder
  Decomposition: STL residual anomalies, CUSUM change-point detection
  Domain:       Volume anomaly, Volatility regime shift
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    HAS_TF = True
except Exception:
    tf = None
    layers = None
    models = None
    HAS_TF = False

from app.config import logger

np.random.seed(42)
if HAS_TF:
    tf.random.set_seed(42)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ============================================================================
# 1. Z-SCORE ANOMALY DETECTION
# ============================================================================

def detect_zscore(
    df: pd.DataFrame,
    col: str = "log_return",
    threshold: float = 3.0,
    method: str = "rolling",
    window: int = 60,
) -> pd.DataFrame:
    """
    Z-score based anomaly detection.

    Methods:
    - 'standard': global mean/std — jó stabil piacon, érzékeny outlierekre
    - 'robust': median/MAD — kevésbé érzékeny szélsőségekre
    - 'rolling': gördülő ablak — adaptív, lokális eltéréseket is detektálja

    Limitáció: egyváltozós, nem veszi figyelembe a feature-ök közötti kapcsolatokat.
    """
    s = df[col].astype(float)

    if method == "standard":
        mu, sigma = s.mean(), s.std()
        z = (s - mu) / (sigma + 1e-9)
    elif method == "robust":
        median = s.median()
        mad = np.median(np.abs(s - median)) + 1e-9
        z = 0.6745 * (s - median) / mad
    elif method == "rolling":
        roll_mu = s.rolling(window=window, min_periods=10).mean()
        roll_std = s.rolling(window=window, min_periods=10).std()
        z = (s - roll_mu) / (roll_std + 1e-9)
    else:
        raise ValueError(f"Unknown z-score method: {method}")

    df["zscore"] = z
    df["anomaly_zscore"] = ((z.abs() > threshold) & np.isfinite(z)).astype(int)
    return df


# ============================================================================
# 2. IQR OUTLIER DETECTION
# ============================================================================

def detect_iqr(
    df: pd.DataFrame,
    col: str = "log_return",
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Interkvartilis terjedelem (IQR) alapú outlier detektálás.

    Mire jó: egyszerű, robosztus, nem parametrikus módszer.
    Limitáció: statikus küszöb, nem adaptív.
    """
    s = df[col].astype(float)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    df["anomaly_iqr"] = ((s < lower) | (s > upper)).astype(int)
    return df


# ============================================================================
# 3. ISOLATION FOREST
# ============================================================================

def detect_isolation_forest(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    contamination: float = 0.01,
    n_estimators: int = 200,
) -> pd.DataFrame:
    """
    Isolation Forest: random partíciókkal izolál outliereket.
    Mire jó: magas dimenziós adatokon, nem kell normális eloszlás.
    Limitáció: contamination paraméter előzetes meghatározást igényel.
    """
    if X_scaled.shape[0] < 30:
        df["anomaly_iso"] = 0
        df["iso_score"] = np.nan
        return df

    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    iso.fit(X_scaled)
    df["iso_score"] = pd.Series(iso.decision_function(X_scaled), index=df.index)
    df["anomaly_iso"] = pd.Series((iso.predict(X_scaled) == -1).astype(int), index=df.index)
    return df


def detect_isolation_forest_ensemble(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    contamination: float = 0.01,
    n_estimators: int = 200,
    n_runs: int = 5,
) -> pd.DataFrame:
    """
    Ensemble Isolation Forest: több futtatás átlagolt score-ja stabilabb eredményt ad.
    """
    if X_scaled.shape[0] < 30:
        df["anomaly_iso"] = 0
        df["iso_score"] = np.nan
        return df

    all_scores = np.zeros((n_runs, X_scaled.shape[0]))
    for i in range(n_runs):
        iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42 + i)
        iso.fit(X_scaled)
        all_scores[i] = iso.decision_function(X_scaled)

    avg_scores = all_scores.mean(axis=0)
    df["iso_score"] = pd.Series(avg_scores, index=df.index)
    cutoff = np.percentile(avg_scores, 100.0 * contamination)
    df["anomaly_iso"] = pd.Series((avg_scores < cutoff).astype(int), index=df.index)
    return df


# ============================================================================
# 4. LOCAL OUTLIER FACTOR
# ============================================================================

def detect_lof(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    contamination: float = 0.01,
    n_neighbors: int = 20,
) -> pd.DataFrame:
    """
    LOF: lokális sűrűség alapú anomália detektálás.
    Mire jó: nem-gömbszerű klaszterekben is talál outliereket, lokálisan adaptív.
    Limitáció: n_neighbors érzékeny, nagy adaton lassú.
    """
    if X_scaled.shape[0] < (n_neighbors + 1):
        df["anomaly_lof"] = 0
        df["lof_score"] = np.nan
        return df

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof.fit_predict(X_scaled)
    scores = -lof.negative_outlier_factor_
    df["lof_score"] = pd.Series(scores, index=df.index)
    cutoff = np.percentile(scores, 100.0 * (1 - contamination))
    df["anomaly_lof"] = pd.Series((scores > cutoff).astype(int), index=df.index)
    return df


# ============================================================================
# 5. ONE-CLASS SVM
# ============================================================================

def detect_ocsvm(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    nu: float = 0.01,
    kernel: str = "rbf",
    gamma: str = "scale",
) -> pd.DataFrame:
    """
    One-Class SVM: hipersíkkal szeparálja a normális mintákat.
    Mire jó: nemlineáris határok kezelése RBF kernellel.
    Limitáció: lassú nagy adathalmazon, nu érzékenyen behangolható.
    """
    if X_scaled.shape[0] < 30:
        df["anomaly_ocsvm"] = 0
        df["ocsvm_score"] = np.nan
        return df

    clf = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    clf.fit(X_scaled)
    scores = clf.decision_function(X_scaled)
    preds = clf.predict(X_scaled)
    df["ocsvm_score"] = pd.Series(scores, index=df.index)
    df["anomaly_ocsvm"] = pd.Series((preds == -1).astype(int), index=df.index)
    return df


# ============================================================================
# 6. STL DECOMPOSITION + RESIDUAL ANOMALY
# ============================================================================

def detect_stl_residual(
    df: pd.DataFrame,
    col: str = "Close",
    period: int = 5,
    threshold_sigma: float = 3.0,
) -> pd.DataFrame:
    """
    STL dekompozíció: trend + szezonalitás + reziduál.
    A nagy reziduálok anomáliát jeleznek.
    Mire jó: szezonális mintázatok kiszűrése után lát tisztábban.
    Limitáció: periódust ismerni kell; hiányzó napok zavarhatják.
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        logger.warning("statsmodels not available, skipping STL decomposition")
        df["anomaly_stl"] = 0
        df["stl_residual"] = np.nan
        return df

    series = df[col].astype(float).copy()
    if len(series) < 2 * period + 1:
        df["anomaly_stl"] = 0
        df["stl_residual"] = np.nan
        return df

    try:
        stl = STL(series, period=period, robust=True)
        result = stl.fit()
        residual = result.resid
        df["stl_residual"] = residual
        df["stl_trend"] = result.trend
        df["stl_seasonal"] = result.seasonal

        mu, sigma = residual.mean(), residual.std()
        z = (residual - mu) / (sigma + 1e-9)
        df["anomaly_stl"] = (z.abs() > threshold_sigma).astype(int)
    except Exception as e:
        logger.warning("STL decomposition failed: %s", e)
        df["anomaly_stl"] = 0
        df["stl_residual"] = np.nan

    return df


# ============================================================================
# 7. CUSUM CHANGE-POINT DETECTION
# ============================================================================

def detect_cusum(
    df: pd.DataFrame,
    col: str = "log_return",
    threshold: float = 1.0,
    drift: float = 0.0,
) -> pd.DataFrame:
    """
    CUSUM (Cumulative Sum) change-point detection.
    Felismeri a sorozat átlagának tartós eltolódását.
    Mire jó: regime shift detektálás, trendtörés.
    Limitáció: egy-irányú shift-ekre érzékeny, küszöb valasztás szubjektív.
    """
    s = df[col].astype(float).values
    mu = np.nanmean(s)
    std = np.nanstd(s) + 1e-9

    s_norm = (s - mu) / std

    s_high = np.zeros(len(s))
    s_low = np.zeros(len(s))
    change_points = np.zeros(len(s), dtype=int)

    for i in range(1, len(s)):
        if np.isnan(s_norm[i]):
            s_high[i] = s_high[i - 1]
            s_low[i] = s_low[i - 1]
            continue
        s_high[i] = max(0, s_high[i - 1] + s_norm[i] - drift)
        s_low[i] = max(0, s_low[i - 1] - s_norm[i] - drift)
        if s_high[i] > threshold or s_low[i] > threshold:
            change_points[i] = 1
            s_high[i] = 0
            s_low[i] = 0

    df["cusum_high"] = pd.Series(s_high, index=df.index)
    df["cusum_low"] = pd.Series(s_low, index=df.index)
    df["anomaly_cusum"] = pd.Series(change_points, index=df.index)
    return df


# ============================================================================
# 8. VOLUME ANOMALY DETECTION
# ============================================================================

def detect_volume_anomaly(
    df: pd.DataFrame,
    spike_col: str = "volume_spike",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Volume spike anomaly: volume / rolling_median(volume) > threshold.
    Mire jó: szokatlan forgalom detektálása (intézményi kereskedés, hír).
    """
    if spike_col not in df.columns:
        df["anomaly_volume"] = 0
        return df

    df["anomaly_volume"] = (df[spike_col] > threshold).astype(int)
    return df


# ============================================================================
# 9. VOLATILITY REGIME SHIFT DETECTION
# ============================================================================

def detect_volatility_shift(
    df: pd.DataFrame,
    vol_col: str = "vol_20",
    window: int = 60,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Volatilitás rezsimváltás: mikor ugrik a volatilitás az elmúlt időszak átlagának
    multiplier-szörösére.
    """
    if vol_col not in df.columns:
        df["anomaly_vol_shift"] = 0
        return df

    vol = df[vol_col].astype(float)
    vol_roll_mean = vol.rolling(window, min_periods=10).mean()
    df["anomaly_vol_shift"] = (vol > vol_roll_mean * multiplier).astype(int)
    return df


# ============================================================================
# 10. LSTM AUTOENCODER
# ============================================================================

def _create_sequences(X: np.ndarray, seq_len: int = 30) -> np.ndarray:
    n = len(X) - seq_len + 1
    if n <= 0:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
    shape = (n, seq_len) + X.shape[1:]
    strides = (X.strides[0],) + X.strides
    view = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    return np.array(view, dtype=np.float32)  # copy to own memory


def _build_lstm_autoencoder(timesteps: int, n_features: int, latent_dim: int = 32):
    if not HAS_TF:
        return None

    inputs = layers.Input(shape=(timesteps, n_features))
    x = layers.LSTM(64, activation="relu", return_sequences=True)(inputs)
    x = layers.LSTM(latent_dim, activation="relu", return_sequences=False)(x)
    x = layers.RepeatVector(timesteps)(x)
    x = layers.LSTM(latent_dim, activation="relu", return_sequences=True)(x)
    x = layers.LSTM(64, activation="relu", return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm_autoencoder(
    X_raw: np.ndarray,
    seq_len: int = 30,
    epochs: int = 7,
    batch_size: int = 64,
    latent_dim: int = 32,
):
    """
    LSTM Autoencoder anomáliadetektáláshoz.
    Mire jó: időbeli mintázatokból tanul, szekvenciális anomáliákat is detektál (pl. flash crash).
    Limitáció: tanítási idő, kis adaton overfit, hiperparaméter-érzékeny.

    Returns: (model, threshold, seq_len, history, scaler, train_loss)
    """
    if not HAS_TF:
        logger.warning("TensorFlow not available, skipping LSTM autoencoder training")
        return None, None, seq_len, None, None, None

    tf.keras.utils.set_random_seed(42)

    seqs = _create_sequences(X_raw, seq_len=seq_len)
    if len(seqs) < 10:
        return None, None, seq_len, None, None, None

    n_seqs, timesteps, n_features = seqs.shape
    split = int(0.8 * n_seqs)
    X_train, X_val = seqs[:split], seqs[split:]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))
    X_train_s = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_s = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)

    with tf.device("/CPU:0"):
        model = _build_lstm_autoencoder(timesteps, n_features, latent_dim)
        history = model.fit(
            X_train_s, X_train_s,
            validation_data=(X_val_s, X_val_s),
            epochs=epochs, batch_size=batch_size,
            shuffle=False, verbose=0,
        )
        train_recon = model.predict(X_train_s, verbose=0)

    train_loss = np.mean(np.mean((X_train_s - train_recon) ** 2, axis=2), axis=1)
    threshold = np.percentile(train_loss, 99)

    logger.info("LSTM AE trained: threshold=%.6f, train_loss_mean=%.6f", threshold, train_loss.mean())
    return model, threshold, seq_len, history, scaler, train_loss


def detect_lstm_autoencoder(
    df: pd.DataFrame,
    model,
    threshold: float,
    X_raw: np.ndarray,
    seq_len: int = 30,
    scaler=None,
    thresh_method: str = "median_mad",
    mad_multiplier: float = 3.0,
    smooth_window: int = 3,
    train_loss: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Run trained LSTM autoencoder for anomaly detection.
    """
    if (not HAS_TF) or model is None or threshold is None:
        df["anomaly_lstm"] = 0
        df["lstm_error"] = np.nan
        return df

    seqs = _create_sequences(X_raw, seq_len=seq_len)
    if scaler is not None:
        n_seqs, ts, nf = seqs.shape
        seqs = scaler.transform(seqs.reshape(-1, nf)).reshape(n_seqs, ts, nf)

    with tf.device("/CPU:0"):
        recon = model.predict(seqs, verbose=0)

    errors = np.mean(np.mean((seqs - recon) ** 2, axis=2), axis=1)

    if smooth_window and smooth_window > 1:
        errors = pd.Series(errors).rolling(window=smooth_window, center=True, min_periods=1).median().values

    if thresh_method == "median_mad":
        base = train_loss if train_loss is not None else errors
        med = np.median(base)
        mad = np.median(np.abs(base - med))
        actual_threshold = med + mad_multiplier * mad
    else:
        actual_threshold = threshold

    anomaly_flags = np.zeros(len(df), dtype=int)
    error_series = np.full(len(df), np.nan)
    # Map each sequence's error to the LAST point in that sequence
    # Sequence i covers X[i : i+seq_len], so its last point is at index i+seq_len-1
    for i, e in enumerate(errors):
        idx = i + seq_len - 1
        if 0 <= idx < len(df):
            anomaly_flags[idx] = int(e > actual_threshold)
            error_series[idx] = e

    df["anomaly_lstm"] = pd.Series(anomaly_flags, index=df.index).astype(int)
    df["lstm_error"] = pd.Series(error_series, index=df.index)
    return df


def collapse_lstm_events(
    df: pd.DataFrame,
    anomaly_col: str = "anomaly_lstm",
    error_col: str = "lstm_error",
    merge_window: int = 1,
) -> pd.DataFrame:
    """Collapse contiguous LSTM anomaly runs to single peak events."""
    df = df.copy()
    df["anomaly_lstm_peak"] = 0

    mask = df[anomaly_col].fillna(0).astype(int)
    if mask.sum() == 0:
        return df

    group_id = (mask != mask.shift(fill_value=0)).cumsum()
    for _, sub in df[mask.astype(bool)].groupby(group_id[mask.astype(bool)]):
        errs = sub[error_col].dropna()
        pick = errs.idxmax() if not errs.empty else sub.index[0]
        df.loc[pick, "anomaly_lstm_peak"] = 1

    return df


# ============================================================================
# UTILITY: Percentile rank normalization
# ============================================================================

def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Convert values to percentile ranks in [0, 1]."""
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() == 0:
        return out
    v = values[mask]
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(v), dtype=float)
    out[mask] = (ranks + 1.0) / (len(v) + 1.0)
    return out


def add_feature_deviation_scores(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Add robust z-like deviation scores (dev_<feature>) for explainability."""
    df = df.copy()
    for c in feature_cols:
        if c not in df.columns:
            continue
        s = df[c].astype(float).replace([np.inf, -np.inf], np.nan)
        med = np.nanmedian(s.values)
        mad = np.nanmedian(np.abs(s.values - med))
        if mad < 1e-6:
            # Near-constant feature → no meaningful deviation
            df[f"dev_{c}"] = 0.0
        else:
            dev = 0.6745 * (s - med) / mad
            df[f"dev_{c}"] = np.clip(dev, -10, 10)
    return df
