"""
Application configuration with .env support and structured logging.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


@dataclass
class AnomalyConfig:
    """All tuneable parameters for the anomaly detection pipeline."""

    # Data
    default_ticker: str = _env("DEFAULT_TICKER", "AAPL")
    default_start_date: str = _env("DEFAULT_START_DATE", "2018-01-01")
    lookback_years: int = _env_int("DEFAULT_LOOKBACK_YEARS", 6)

    # Z-score
    zscore_threshold: float = _env_float("ZSCORE_THRESHOLD", 3.0)
    zscore_method: str = _env("ZSCORE_METHOD", "rolling")
    zscore_window: int = _env_int("ZSCORE_WINDOW", 60)

    # Isolation Forest
    if_contamination: float = _env_float("IF_CONTAMINATION", 0.01)
    if_n_estimators: int = _env_int("IF_N_ESTIMATORS", 200)
    if_ensemble_runs: int = _env_int("IF_ENSEMBLE_RUNS", 5)

    # LOF
    lof_contamination: float = _env_float("LOF_CONTAMINATION", 0.01)
    lof_n_neighbors: int = _env_int("LOF_N_NEIGHBORS", 20)

    # One-Class SVM
    ocsvm_nu: float = _env_float("OCSVM_NU", 0.01)

    # LSTM Autoencoder
    lstm_seq_len: int = _env_int("LSTM_SEQ_LEN", 30)
    lstm_epochs: int = _env_int("LSTM_EPOCHS", 7)
    lstm_batch_size: int = _env_int("LSTM_BATCH_SIZE", 64)
    lstm_latent_dim: int = _env_int("LSTM_LATENT_DIM", 32)
    lstm_thresh_method: str = _env("LSTM_THRESH_METHOD", "median_mad")
    lstm_mad_mult: float = _env_float("LSTM_MAD_MULT", 3.0)

    # Forecasting
    forecast_horizon: int = _env_int("FORECAST_HORIZON", 5)
    arima_max_p: int = _env_int("ARIMA_MAX_P", 5)
    arima_max_q: int = _env_int("ARIMA_MAX_Q", 5)
    xgb_n_estimators: int = _env_int("XGBOOST_N_ESTIMATORS", 200)
    lstm_forecast_epochs: int = _env_int("LSTM_FORECAST_EPOCHS", 20)

    # Ensemble
    ensemble_vote_k: int = _env_int("ENSEMBLE_VOTE_K", 2)
    ensemble_combine: str = _env("ENSEMBLE_COMBINE", "and")

    # Live
    live_refresh_seconds: int = _env_int("LIVE_REFRESH_SECONDS", 300)

    # Alternative data toggles
    enable_fred: bool = _env("ENABLE_FRED", "true").lower() == "true"
    enable_fear_greed: bool = _env("ENABLE_FEAR_GREED", "true").lower() == "true"
    enable_options: bool = _env("ENABLE_OPTIONS_DATA", "true").lower() == "true"
    enable_short_interest: bool = _env("ENABLE_SHORT_INTEREST", "true").lower() == "true"
    enable_llm_sentiment: bool = _env("ENABLE_LLM_SENTIMENT", "true").lower() == "true"

    # Advanced forecasting toggles
    enable_garch: bool = _env("ENABLE_GARCH", "true").lower() == "true"
    enable_stacking: bool = _env("ENABLE_STACKING", "true").lower() == "true"
    enable_tft: bool = _env("ENABLE_TFT", "false").lower() == "true"
    enable_online_learning: bool = _env("ENABLE_ONLINE_LEARNING", "true").lower() == "true"
    enable_backtesting: bool = _env("ENABLE_BACKTESTING", "true").lower() == "true"
    enable_shap: bool = _env("ENABLE_SHAP", "true").lower() == "true"

    # Backtesting
    backtest_window_size: int = _env_int("BACKTEST_WINDOW_SIZE", 252)
    backtest_step_size: int = _env_int("BACKTEST_STEP_SIZE", 20)


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure structured logging and return the app logger."""
    log_level = level or _env("LOG_LEVEL", "INFO")
    numeric = getattr(logging, log_level.upper(), logging.INFO)

    fmt = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"
    logging.basicConfig(format=fmt, level=numeric, force=True)

    logger = logging.getLogger("anomaly_detector")
    logger.setLevel(numeric)
    return logger


# Singleton instances
CONFIG = AnomalyConfig()
logger = setup_logging()
