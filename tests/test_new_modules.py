"""
Tests for the new app/ package modules.
Run with: python -m pytest tests/test_new_modules.py -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Features
# ============================================================================

class TestFeatures:
    @pytest.fixture
    def sample_df(self):
        """Create a realistic sample OHLCV DataFrame."""
        rng = np.random.RandomState(42)
        n = 300
        dates = pd.bdate_range("2023-01-01", periods=n)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        df = pd.DataFrame({
            "Open": close + rng.normal(0, 0.3, n),
            "High": close + np.abs(rng.normal(0, 1, n)),
            "Low": close - np.abs(rng.normal(0, 1, n)),
            "Close": close,
            "Volume": (rng.exponential(1_000_000, n)).astype(int),
        }, index=dates)
        return df

    def test_add_all_features(self, sample_df):
        from app.features import add_all_features
        df = add_all_features(sample_df)
        assert "log_return" in df.columns
        assert "RSI_14" in df.columns
        assert "MACD" in df.columns
        assert "BB_upper" in df.columns
        assert "ATR_14" in df.columns
        assert "momentum_10" in df.columns
        assert "drawdown" in df.columns
        assert "trend" in df.columns
        assert "vol_regime" in df.columns
        assert not df["log_return"].isnull().all()

    def test_prepare_ml_features(self, sample_df):
        from app.features import add_all_features, prepare_ml_features
        df = add_all_features(sample_df)
        X_raw, X_scaled, cols, scaler, index = prepare_ml_features(df)
        assert X_raw.shape[0] == X_scaled.shape[0]
        assert X_raw.shape[1] == len(cols)
        assert not np.isnan(X_scaled).any()
        assert len(index) == X_raw.shape[0]


# ============================================================================
# Anomaly Models
# ============================================================================

class TestAnomalyModels:
    @pytest.fixture
    def data_with_outlier(self):
        rng = np.random.RandomState(0)
        n = 200
        dates = pd.bdate_range("2023-01-01", periods=n)
        data = rng.normal(0, 1, n)
        data[100] = 10.0  # obvious outlier
        df = pd.DataFrame({"log_return": data, "Close": 100 + np.cumsum(data)}, index=dates)
        df["Volume"] = rng.exponential(1e6, n).astype(int)
        df["rolling_vol_20"] = df["log_return"].rolling(20).std()
        df["rolling_vol_60"] = df["log_return"].rolling(60).std()
        df = df.dropna()
        return df

    def test_zscore_detects_outlier(self, data_with_outlier):
        from app.anomaly_models import detect_zscore
        df = detect_zscore(data_with_outlier, threshold=3.0, method="standard")
        assert df["anomaly_zscore"].sum() >= 1

    def test_iqr_detects_outlier(self, data_with_outlier):
        from app.anomaly_models import detect_iqr
        df = detect_iqr(data_with_outlier)
        assert df["anomaly_iqr"].sum() >= 1

    def test_isolation_forest(self, data_with_outlier):
        from app.anomaly_models import detect_isolation_forest_ensemble
        rng = np.random.RandomState(0)
        X = rng.normal(size=(len(data_with_outlier), 4))
        X[80] += 8.0  # outlier in feature space
        df = detect_isolation_forest_ensemble(data_with_outlier, X, contamination=0.02)
        assert "anomaly_iso" in df.columns
        assert "iso_score" in df.columns

    def test_lof(self, data_with_outlier):
        from app.anomaly_models import detect_lof
        rng = np.random.RandomState(0)
        X = rng.normal(size=(len(data_with_outlier), 4))
        X[80] += 8.0
        df = detect_lof(data_with_outlier, X, contamination=0.02)
        assert "anomaly_lof" in df.columns

    def test_cusum(self, data_with_outlier):
        from app.anomaly_models import detect_cusum
        df = detect_cusum(data_with_outlier, threshold=1.0)
        assert "anomaly_cusum" in df.columns

    def test_volume_anomaly(self, data_with_outlier):
        from app.anomaly_models import detect_volume_anomaly
        df = data_with_outlier.copy()
        # The function expects a precomputed 'volume_spike' column
        vol_median = df["Volume"].rolling(20, min_periods=1).median()
        df["volume_spike"] = df["Volume"] / vol_median
        df.loc[df.index[100], "volume_spike"] = 5.0  # well above threshold
        df = detect_volume_anomaly(df, threshold=3.0)
        assert df["anomaly_volume"].sum() >= 1


# ============================================================================
# Ensemble
# ============================================================================

class TestEnsemble:
    def test_compute_ensemble_score(self):
        from app.ensemble import compute_ensemble_score
        n = 100
        dates = pd.bdate_range("2023-01-01", periods=n)
        rng = np.random.RandomState(42)
        df = pd.DataFrame(index=dates)
        df["anomaly_zscore"] = 0
        df["anomaly_iqr"] = 0
        df["anomaly_iso"] = 0
        df["anomaly_lof"] = 0
        df["zscore"] = rng.randn(n) * 0.5
        df["iso_score"] = rng.rand(n) * 0.5
        df["lof_score"] = rng.rand(n) * 0.3

        # Make row 50 anomalous across all models AND give it extreme scores
        for col in ["anomaly_zscore", "anomaly_iqr", "anomaly_iso", "anomaly_lof"]:
            df.loc[df.index[50], col] = 1
        df.loc[df.index[50], "zscore"] = 8.0
        df.loc[df.index[50], "iso_score"] = -0.5  # lower = more anomalous for IF
        df.loc[df.index[50], "lof_score"] = 5.0

        # Use combine="or" so voting alone suffices, vote_k=2
        df = compute_ensemble_score(df, vote_k=2, combine="or")
        assert "ensemble_score" in df.columns
        assert "anomaly_ensemble" in df.columns
        assert "anomaly_severity" in df.columns
        assert df.loc[df.index[50], "anomaly_ensemble"] == 1

    def test_get_anomaly_summary(self):
        from app.ensemble import get_anomaly_summary
        n = 50
        dates = pd.bdate_range("2023-01-01", periods=n)
        df = pd.DataFrame(index=dates)
        df["anomaly_ensemble"] = 0
        df["anomaly_severity"] = "normal"
        df["ensemble_score"] = 0.1
        df.loc[df.index[10], "anomaly_ensemble"] = 1
        df.loc[df.index[10], "anomaly_severity"] = "strong"

        summary = get_anomaly_summary(df)
        assert summary["total_anomalies"] == 1
        assert summary["strong_anomalies"] == 1


# ============================================================================
# Forecasting
# ============================================================================

class TestForecasting:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.RandomState(42)
        n = 300
        dates = pd.bdate_range("2023-01-01", periods=n)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        df = pd.DataFrame({
            "Close": close,
            "Volume": rng.exponential(1e6, n).astype(int),
        }, index=dates)
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["rolling_vol_20"] = df["log_return"].rolling(20).std()
        df["rsi"] = 50 + rng.normal(0, 10, n)
        df["macd"] = rng.normal(0, 1, n)
        df["macd_signal"] = rng.normal(0, 1, n)
        df["momentum_10"] = rng.normal(0, 2, n)
        df["bb_pct_b"] = rng.uniform(0, 1, n)
        df["trend"] = np.random.choice(["bullish", "bearish", "sideways"], n)
        df["vol_regime"] = np.random.choice(["low", "medium", "high"], n)
        df["anomaly_ensemble"] = 0
        df["ensemble_score"] = rng.rand(n) * 0.3
        df = df.dropna()
        return df

    def test_classify_direction(self, sample_df):
        from app.forecasting import classify_direction
        result = classify_direction(sample_df)
        assert "direction" in result
        assert "probabilities" in result
        assert result["direction"] in ("up", "down", "sideways")
        probs = result["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_assess_reliability(self, sample_df):
        from app.forecasting import assess_forecast_reliability
        result = assess_forecast_reliability(sample_df)
        assert "reliable" in result
        assert "risk_level" in result
        assert "warnings" in result


# ============================================================================
# Visualization
# ============================================================================

class TestVisualization:
    @pytest.fixture
    def chart_df(self):
        rng = np.random.RandomState(42)
        n = 100
        dates = pd.bdate_range("2023-01-01", periods=n)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        df = pd.DataFrame({
            "Close": close,
            "Volume": rng.exponential(1e6, n).astype(int),
            "sma_20": pd.Series(close).rolling(20).mean().values,
            "sma_50": pd.Series(close).rolling(50).mean().values,
            "bb_upper": close + 2,
            "bb_lower": close - 2,
            "rolling_vol_20": rng.uniform(0.01, 0.05, n),
            "rolling_vol_60": rng.uniform(0.01, 0.04, n),
            "vol_regime": np.random.choice(["low", "medium", "high"], n),
            "ensemble_score": rng.rand(n) * 0.5,
            "anomaly_ensemble": np.zeros(n, dtype=int),
            "anomaly_severity": ["normal"] * n,
            "rsi": 30 + rng.uniform(0, 40, n),
        }, index=dates)
        df.loc[df.index[50], "anomaly_ensemble"] = 1
        df.loc[df.index[50], "anomaly_severity"] = "strong"
        return df

    def test_create_main_chart(self, chart_df):
        from app.visualization import create_main_chart
        fig = create_main_chart(chart_df, "TEST")
        assert fig is not None

    def test_create_volume_chart(self, chart_df):
        from app.visualization import create_volume_chart
        fig = create_volume_chart(chart_df)
        assert fig is not None

    def test_create_anomaly_score_chart(self, chart_df):
        from app.visualization import create_anomaly_score_chart
        fig = create_anomaly_score_chart(chart_df)
        assert fig is not None


# ============================================================================
# Data Loader
# ============================================================================

class TestDataLoader:
    def test_validate_data(self):
        from app.data_loader import validate_data
        dates = pd.bdate_range("2023-01-01", periods=100)
        df = pd.DataFrame({
            "Open": np.random.rand(100) * 100,
            "High": np.random.rand(100) * 100 + 50,
            "Low": np.random.rand(100) * 100 - 50,
            "Close": np.random.rand(100) * 100,
            "Volume": np.random.randint(1e5, 1e7, 100),
        }, index=dates)
        is_valid, warnings = validate_data(df)
        assert isinstance(is_valid, bool)
        assert isinstance(warnings, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
