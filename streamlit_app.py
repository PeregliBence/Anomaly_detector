"""
📈 Stock Market Anomaly Detector & Forecaster — Streamlit Application

Professional anomaly detection and forecasting dashboard for financial time series.
Supports stocks, indices, ETFs, forex, and crypto via Yahoo Finance.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st

from app.config import CONFIG, logger
from app.data_loader import load_data, validate_data
from app.features import add_all_features, prepare_ml_features, ML_FEATURE_COLS
from app.anomaly_models import (
    detect_zscore, detect_iqr,
    detect_isolation_forest_ensemble, detect_lof, detect_ocsvm,
    detect_stl_residual, detect_cusum,
    detect_volume_anomaly, detect_volatility_shift,
    train_lstm_autoencoder, detect_lstm_autoencoder, collapse_lstm_events,
    add_feature_deviation_scores,
)
from app.ensemble import compute_ensemble_score, get_anomaly_summary
from app.forecasting import (
    forecast_ensemble, classify_direction, assess_forecast_reliability, ForecastResult,
)
from app.explainability import (
    build_anomaly_explanation, compute_feature_importance_for_anomalies,
    generate_data_quality_report,
)
from app.visualization import (
    create_main_chart, create_volume_chart, create_volatility_chart,
    create_anomaly_score_chart, create_rsi_chart, create_forecast_chart,
    create_dashboard_figure, create_anomaly_zoom, create_model_comparison_chart,
)
from app.search import yahoo_symbol_search, TOP_TICKERS

# New modules
from app.alternative_data import add_all_alternative_data, fetch_fear_greed_index
from app.advanced_forecasting import (
    forecast_garch, forecast_stacking_ensemble, forecast_tft,
    online_learn_and_predict, forecast_advanced_ensemble,
)
from app.calibration import ConfidenceCalibrator, calibrate_forecast_results
from app.backtesting import run_quick_backtest, backtest_anomalies, assess_signal_quality
from app.shap_explainer import (
    compute_shap_for_forecast, generate_llm_explanation,
    get_feature_importance_data,
)
from app.llm_sentiment import run_sentiment_pipeline, get_aggregate_sentiment

try:
    from streamlit_searchbox import st_searchbox
    HAS_SEARCHBOX = True
except ImportError:
    HAS_SEARCHBOX = False


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Stock Anomaly Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #1a1a2e 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
    }
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar controls and return configuration dict."""
    with st.sidebar:
        st.header("⚙️ Beállítások")

        # Ticker selection
        st.subheader("📊 Eszköz kiválasztás")
        if HAS_SEARCHBOX:
            selected_ticker = st_searchbox(
                yahoo_symbol_search,
                key="ticker_searchbox",
                placeholder="Keresés: AAPL, Bitcoin, S&P 500...",
                default=None,
            )
            ticker = selected_ticker if selected_ticker else TOP_TICKERS[0][1]
        else:
            ticker = st.text_input("Ticker szimbólum", value=CONFIG.default_ticker)

        # Date range
        st.subheader("📅 Időszak")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Kezdő dátum", value=pd.to_datetime(CONFIG.default_start_date))
        with col2:
            end_date = st.date_input("Záró dátum", value=pd.to_datetime("today"))

        st.markdown("---")

        # Model parameters
        st.subheader("🧠 Modell paraméterek")

        z_threshold = st.slider("Z-score küszöb", 2.0, 5.0, CONFIG.zscore_threshold, 0.1,
                                help="Magasabb = kevesebb, de biztosabb anomália")
        contamination = st.slider("Contamination (IF/LOF)", 0.001, 0.10, CONFIG.if_contamination, 0.001,
                                  help="Várt anomáliák aránya")
        seq_len = st.slider("LSTM szekvencia hossz", 10, 60, CONFIG.lstm_seq_len, 5)

        # Advanced settings in expander
        with st.expander("🔧 Haladó beállítások"):
            ensemble_k = st.slider("Ensemble szavazási küszöb (K)", 1, 5, CONFIG.ensemble_vote_k)
            ensemble_combine = st.selectbox("Ensemble kombináció", ["and", "or"], index=0,
                                            help="'and' = konzervatív, 'or' = agresszív")
            run_ocsvm = st.checkbox("One-Class SVM futtatása", value=False,
                                    help="Lassabb, de extra perspektívát ad")
            run_stl = st.checkbox("STL dekompozíció futtatása", value=True)
            run_cusum = st.checkbox("CUSUM change-point detektálás", value=True)
            lstm_epochs = st.slider("LSTM epochák", 3, 20, CONFIG.lstm_epochs)
            volume_spike_threshold = st.slider("Volume spike küszöb", 2.0, 10.0, 3.0, 0.5)

        st.markdown("---")

        # Forecast settings
        st.subheader("🔮 Előrejelzés")
        run_forecast = st.checkbox("Előrejelzés futtatása", value=True)
        forecast_horizon = st.slider("Forecast horizon (nap)", 1, 20, CONFIG.forecast_horizon)
        forecast_models = st.multiselect(
            "Forecast modellek",
            ["arima", "xgboost", "lstm", "garch", "stacking", "tft", "online"],
            default=["arima", "xgboost"],
        )

        st.markdown("---")

        # Alternative data & advanced features
        st.subheader("📡 Kiegészítő modulok")
        with st.expander("🌐 Alternatív adatok"):
            enable_alt_data = st.checkbox("Alternatív adatok betöltése", value=False,
                                          help="FRED makró, Fear&Greed, opciós adat, short interest")
            enable_sentiment = st.checkbox("LLM szentiment elemzés", value=False,
                                           help="OpenAI-alapú hír szentiment (API kulcs szükséges)")

        with st.expander("🧪 Haladó modellek"):
            enable_backtesting = st.checkbox("Backtesting futtatása", value=False,
                                             help="Walk-forward validáció az előrejelzés minőségéhez")
            enable_shap = st.checkbox("SHAP magyarázatok", value=False,
                                      help="Feature importance elemzés SHAP-pal")
            enable_calibration = st.checkbox("Konfidencia kalibráció", value=False,
                                             help="Isotonic regression a megbízhatóbb konfidenciáért")

        st.markdown("---")

        # Live mode
        st.subheader("🔴 Live mód")
        live_mode = st.checkbox("Automatikus frissítés", value=False)
        refresh_interval = st.slider("Frissítési intervallum (mp)", 60, 600, CONFIG.live_refresh_seconds, 30,
                                     help="Milyen gyakran frissüljön az adat") if live_mode else 300

        st.markdown("---")

        run_btn = st.button("🚀 Elemzés indítása", type="primary", width='stretch')

        return {
            "ticker": ticker,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "z_threshold": z_threshold,
            "contamination": contamination,
            "seq_len": seq_len,
            "ensemble_k": ensemble_k,
            "ensemble_combine": ensemble_combine,
            "run_ocsvm": run_ocsvm,
            "run_stl": run_stl,
            "run_cusum": run_cusum,
            "lstm_epochs": lstm_epochs,
            "volume_spike_threshold": volume_spike_threshold,
            "run_forecast": run_forecast,
            "forecast_horizon": forecast_horizon,
            "forecast_models": forecast_models,
            "enable_alt_data": enable_alt_data,
            "enable_sentiment": enable_sentiment,
            "enable_backtesting": enable_backtesting,
            "enable_shap": enable_shap,
            "enable_calibration": enable_calibration,
            "live_mode": live_mode,
            "refresh_interval": refresh_interval,
            "run_btn": run_btn,
        }


# ============================================================================
# PIPELINE
# ============================================================================

def run_pipeline(cfg: dict) -> dict | None:
    """Execute the full analysis pipeline. Returns results dict or None on failure."""

    progress = st.progress(0, text="Adatok letöltése...")

    # 1. Load data
    df = load_data(cfg["ticker"], start=cfg["start_date"], end=cfg["end_date"])
    if df.empty:
        st.error("❌ Nem érkezett adat. Ellenőrizd a tickert és a dátumokat!")
        return None

    is_valid, warnings = validate_data(df)
    for w in warnings:
        st.warning(w)

    progress.progress(10, text="Feature-ök számítása...")

    # 2. Feature engineering
    df = add_all_features(df, include_targets=True)
    if len(df) < 60:
        st.error("❌ Túl kevés adat a feature-ök kiszámítása után.")
        return None

    # 2b. Alternative data (optional)
    if cfg.get("enable_alt_data"):
        progress.progress(15, text="Alternatív adatok betöltése...")
        try:
            df = add_all_alternative_data(df, cfg["ticker"])
        except Exception as e:
            logger.warning("Alt data error: %s", e)
            st.warning(f"⚠️ Alternatív adatok részben nem elérhetők: {e}")

    st.sidebar.success(f"✅ {len(df)} adatpont betöltve")

    progress.progress(20, text="ML feature mátrix előkészítése...")

    # 3. Prepare ML features
    X_raw, X_scaled, feature_cols, scaler, ml_index = prepare_ml_features(df)
    df = df.loc[ml_index].copy()

    progress.progress(25, text="Z-score anomáliák...")

    # 4. Z-score
    df = detect_zscore(df, threshold=cfg["z_threshold"], method="rolling", window=60)

    progress.progress(30, text="IQR outlierek...")

    # 5. IQR
    df = detect_iqr(df)

    progress.progress(35, text="Isolation Forest (ensemble)...")

    # 6. Isolation Forest ensemble
    df = detect_isolation_forest_ensemble(df, X_scaled, contamination=cfg["contamination"])

    progress.progress(45, text="LOF...")

    # 7. LOF
    df = detect_lof(df, X_scaled, contamination=cfg["contamination"])

    progress.progress(50, text="One-Class SVM..." if cfg["run_ocsvm"] else "OCSVM kihagyva...")

    # 8. One-Class SVM (optional)
    if cfg["run_ocsvm"]:
        df = detect_ocsvm(df, X_scaled, nu=cfg["contamination"])
    else:
        df["anomaly_ocsvm"] = 0
        df["ocsvm_score"] = np.nan

    progress.progress(55, text="Dekompozíció és change-point...")

    # 9. STL decomposition
    if cfg["run_stl"]:
        df = detect_stl_residual(df, period=5)
    else:
        df["anomaly_stl"] = 0

    # 10. CUSUM
    if cfg["run_cusum"]:
        df = detect_cusum(df, threshold=1.0, drift=0.0)
    else:
        df["anomaly_cusum"] = 0

    progress.progress(60, text="Volume és volatilitás anomáliák...")

    # 11. Volume anomaly
    df = detect_volume_anomaly(df, threshold=cfg["volume_spike_threshold"])

    # 12. Volatility regime shift
    df = detect_volatility_shift(df)

    progress.progress(65, text="LSTM Autoencoder tanítása...")

    # 13. LSTM Autoencoder
    model, threshold, seq_used, history, lstm_scaler, train_loss = train_lstm_autoencoder(
        X_raw, seq_len=cfg["seq_len"], epochs=cfg["lstm_epochs"], batch_size=64,
    )

    if model is not None:
        df = detect_lstm_autoencoder(
            df, model, threshold, X_raw,
            seq_len=cfg["seq_len"], scaler=lstm_scaler,
            thresh_method="median_mad", mad_multiplier=3.0,
            smooth_window=3, train_loss=train_loss,
        )
        df = collapse_lstm_events(df)
        df["anomaly_lstm"] = df["anomaly_lstm_peak"].astype(int)
    else:
        df["anomaly_lstm"] = 0
        df["lstm_error"] = np.nan
        st.warning("⚠️ Túl kevés adat az LSTM modellhez.")

    progress.progress(80, text="Ensemble scoring...")

    # 14. Feature deviation scores for explainability
    df = add_feature_deviation_scores(df, feature_cols)

    # 15. Ensemble
    df = compute_ensemble_score(
        df,
        contamination=cfg["contamination"],
        vote_k=cfg["ensemble_k"],
        combine=cfg["ensemble_combine"],
    )

    progress.progress(85, text="Előrejelzés..." if cfg["run_forecast"] else "Előrejelzés kihagyva...")

    # 16. Forecasting
    forecast_result = None
    individual_forecasts = []
    direction_info = classify_direction(df, horizon=cfg["forecast_horizon"])

    base_models = [m for m in cfg["forecast_models"] if m in ("arima", "xgboost", "lstm")]
    advanced_models = [m for m in cfg["forecast_models"] if m in ("garch", "stacking", "tft", "online")]

    if cfg["run_forecast"] and base_models:
        forecast_result, individual_forecasts = forecast_ensemble(
            df, horizon=cfg["forecast_horizon"], models=base_models,
        )

    # 16b. Advanced forecasting models
    if cfg["run_forecast"] and advanced_models:
        progress.progress(88, text="Haladó előrejelzési modellek...")
        try:
            adv_result, adv_forecasts = forecast_advanced_ensemble(
                df, horizon=cfg["forecast_horizon"], models=advanced_models,
            )
            individual_forecasts.extend(adv_forecasts)
            # If we had no base forecast, use advanced as main result
            if forecast_result is None:
                forecast_result = adv_result
        except Exception as e:
            logger.warning("Advanced forecasting error: %s", e)
            st.warning(f"⚠️ Haladó modellek hibája: {e}")

    # 16c. Confidence calibration
    calibration_data = None
    if cfg.get("enable_calibration") and individual_forecasts:
        try:
            individual_forecasts = calibrate_forecast_results(individual_forecasts, df)
            calibration_data = True
        except Exception as e:
            logger.warning("Calibration error: %s", e)

    # 17. Reliability assessment
    reliability = assess_forecast_reliability(df)

    progress.progress(90, text="Kiegészítő elemzések...")

    # 18. Sentiment analysis (optional)
    sentiment_report = None
    if cfg.get("enable_sentiment"):
        progress.progress(91, text="Szentiment elemzés...")
        try:
            sentiment_report = run_sentiment_pipeline(cfg["ticker"])
        except Exception as e:
            logger.warning("Sentiment error: %s", e)
            st.warning(f"⚠️ Szentiment elemzés nem elérhető: {e}")

    # 19. Backtesting (optional)
    backtest_result = None
    signal_quality = None
    if cfg.get("enable_backtesting") and cfg["run_forecast"]:
        progress.progress(93, text="Backtesting futtatása...")
        try:
            bt_results = run_quick_backtest(
                df, ticker=cfg["ticker"], horizon=cfg["forecast_horizon"],
                models=[m for m in cfg["forecast_models"] if m in ("arima", "xgboost", "lstm")],
            )
            backtest_result = bt_results["forecast_backtest"]
            signal_quality = bt_results["signal_quality"]
        except Exception as e:
            logger.warning("Backtest error: %s", e)
            st.warning(f"⚠️ Backtesting hiba: {e}")

    # 20. SHAP explainability (optional)
    shap_data = None
    llm_explanation = None
    if cfg.get("enable_shap") and individual_forecasts:
        progress.progress(95, text="SHAP magyarázatok számítása...")
        try:
            for fr in individual_forecasts:
                shap_result = compute_shap_for_forecast(fr, df)
                if shap_result is not None:
                    shap_data = shap_result
                    break
            if shap_data is not None:
                sentiment_dict = None
                if sentiment_report:
                    sentiment_dict = {
                        "overall": sentiment_report.overall,
                        "score": sentiment_report.score,
                        "high_impact_count": sentiment_report.high_impact_news,
                    }
                llm_explanation = generate_llm_explanation(
                    shap_data, forecast_result,
                    sentiment_dict,
                )
        except Exception as e:
            logger.warning("SHAP error: %s", e)

    progress.progress(97, text="Vizualizáció előkészítése...")

    # 18. Summary
    summary = get_anomaly_summary(df)

    progress.progress(100, text="Kész!")
    time.sleep(0.3)
    progress.empty()

    return {
        "df": df,
        "feature_cols": feature_cols,
        "summary": summary,
        "forecast_result": forecast_result,
        "individual_forecasts": individual_forecasts,
        "direction_info": direction_info,
        "reliability": reliability,
        "sentiment_report": sentiment_report,
        "backtest_result": backtest_result,
        "signal_quality": signal_quality,
        "shap_data": shap_data,
        "llm_explanation": llm_explanation,
        "calibration_data": calibration_data,
    }


# ============================================================================
# KPI CARDS
# ============================================================================

def render_kpi_cards(summary: dict, direction_info: dict, reliability: dict, df: pd.DataFrame):
    """Render the KPI metrics row."""
    cols = st.columns(6)

    with cols[0]:
        trend = df["trend"].iloc[-1] if "trend" in df.columns else "?"
        trend_icon = {"bullish": "📈", "bearish": "📉", "sideways": "➡️"}.get(trend, "❓")
        st.metric("Aktuális trend", f"{trend_icon} {trend.capitalize()}")

    with cols[1]:
        st.metric("Anomáliák száma", summary["total_anomalies"],
                   delta=f"{summary.get('strong_anomalies', 0)} erős")

    with cols[2]:
        latest = summary.get("latest_anomaly_date")
        latest_str = latest.strftime("%Y-%m-%d") if latest else "—"
        st.metric("Utolsó anomália", latest_str)

    with cols[3]:
        dir_info = direction_info
        dir_icon = {"up": "📈", "down": "📉", "sideways": "➡️"}.get(dir_info["direction"], "")
        probs = dir_info["probabilities"]
        st.metric("Forecast irány",
                   f"{dir_icon} {dir_info['direction'].upper()}",
                   delta=f"{probs.get(dir_info['direction'], 0):.0%} valószínűség")

    with cols[4]:
        vol_state = reliability.get("volatility_state", "?")
        vol_icon = {"normal": "🟢", "elevated": "🟡", "extreme": "🔴"}.get(vol_state, "⚪")
        st.metric("Volatilitás", f"{vol_icon} {vol_state.capitalize()}")

    with cols[5]:
        risk = reliability.get("risk_level", "?")
        risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")
        st.metric("Kockázati szint", f"{risk_icon} {risk.capitalize()}")


# ============================================================================
# TAB RENDERERS
# ============================================================================

def render_dashboard_tab(results: dict, cfg: dict):
    """Dashboard tab: KPIs + charts."""
    df = results["df"]
    summary = results["summary"]

    render_kpi_cards(summary, results["direction_info"], results["reliability"], df)

    st.markdown("---")

    # Main chart
    st.plotly_chart(
        create_main_chart(df, cfg["ticker"]),
        width='stretch', key="main_chart",
    )

    # Volume and Volatility side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_volume_chart(df), width='stretch', key="vol_chart")
    with col2:
        st.plotly_chart(create_volatility_chart(df), width='stretch', key="volatility_chart")

    # Anomaly score and RSI
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_anomaly_score_chart(df), width='stretch', key="score_chart")
    with col2:
        st.plotly_chart(create_rsi_chart(df), width='stretch', key="rsi_chart")


def render_anomaly_tab(results: dict, cfg: dict):
    """Anomaly analysis tab."""
    df = results["df"]
    summary = results["summary"]

    # Model comparison
    if summary.get("model_counts"):
        st.plotly_chart(
            create_model_comparison_chart(summary["model_counts"]),
            width='stretch',
        )

    # Anomaly table
    anomalies = df[df["anomaly_ensemble"] == 1].copy()
    if anomalies.empty:
        st.info("Nem találtunk anomáliát a megadott paraméterekkel.")
        return

    anomalies["date_str"] = anomalies.index.strftime("%Y-%m-%d")

    st.subheader(f"🔍 Anomáliák részletes elemzése ({len(anomalies)} db)")

    # Anomaly selector
    selected_date = st.selectbox(
        "Válassz egy anomáliás napot:",
        options=anomalies.index.tolist(),
        format_func=lambda x: x.strftime("%Y-%m-%d"),
        key="anomaly_selector",
    )

    if selected_date is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.plotly_chart(
                create_anomaly_zoom(df, selected_date, window=15),
                width='stretch',
            )

        with col2:
            row = df.loc[selected_date]
            explanation = build_anomaly_explanation(row)
            st.markdown(explanation)

    # Full anomaly table
    st.subheader("📋 Anomáliák táblázata")

    display_cols = ["Close", "log_return", "ensemble_score", "anomaly_severity", "vote_count"]
    model_cols = [c for c in ["anomaly_zscore", "anomaly_iso", "anomaly_lof", "anomaly_lstm",
                               "anomaly_ocsvm", "anomaly_stl", "anomaly_cusum"] if c in anomalies.columns]
    display_cols.extend(model_cols)
    existing_cols = [c for c in display_cols if c in anomalies.columns]

    st.dataframe(
        anomalies[existing_cols].tail(100),
        width='stretch',
        height=400,
    )

    # Export
    col1, col2 = st.columns(2)
    with col1:
        csv = anomalies[existing_cols].to_csv()
        st.download_button(
            "📥 Anomáliák exportálása (CSV)",
            data=csv,
            file_name=f"{cfg['ticker']}_anomalies.csv",
            mime="text/csv",
        )


def render_forecast_tab(results: dict, cfg: dict):
    """Forecast tab."""
    df = results["df"]
    forecast_result = results["forecast_result"]
    individual = results["individual_forecasts"]
    direction_info = results["direction_info"]
    reliability = results["reliability"]

    # Reliability warnings
    if reliability["warnings"]:
        for w in reliability["warnings"]:
            st.warning(w)

    if not reliability["reliable"]:
        st.error("⚠️ A jelenlegi piaci helyzet túl bizonytalan a megbízható előrejelzéshez. "
                 "Az alábbi eredményeket fokozott óvatossággal kezeld!")

    st.info("⚠️ Ez az elemzés nem minősül befektetési tanácsnak. "
            "Az előrejelzések kizárólag adatvezérelt becslések, nem garanciák.")

    # Main forecast chart
    if forecast_result:
        st.plotly_chart(
            create_forecast_chart(df, forecast_result, cfg["ticker"]),
            width='stretch',
        )

        # Forecast details
        col1, col2, col3 = st.columns(3)
        with col1:
            dir_icon = {"up": "📈", "down": "📉", "sideways": "➡️"}.get(forecast_result.direction, "")
            st.metric("Várható irány", f"{dir_icon} {forecast_result.direction.upper()}")
        with col2:
            st.metric("Confidence", f"{forecast_result.confidence:.0%}")
        with col3:
            st.metric("Várt elmozdulás", f"{forecast_result.expected_return_pct:+.2f}%")

        # Individual model results
        if len(individual) > 1:
            st.subheader("📊 Modellek összehasonlítása")
            model_data = []
            for r in individual:
                model_data.append({
                    "Modell": r.model_name,
                    "Irány": r.direction,
                    "Confidence": f"{r.confidence:.0%}",
                    "Várt hozam (%)": f"{r.expected_return_pct:+.2f}",
                })
            st.table(pd.DataFrame(model_data))
    else:
        st.info("Előrejelzés nem elérhető. Ellenőrizd a beállításokat vagy az adatmennyiséget.")

    # Direction analysis
    st.subheader("🧭 Irány-klasszifikáció")
    probs = direction_info["probabilities"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Emelkedés", f"{probs.get('up', 0):.0%}")
    with col2:
        st.metric("📉 Csökkenés", f"{probs.get('down', 0):.0%}")
    with col3:
        st.metric("➡️ Oldalazás", f"{probs.get('sideways', 0):.0%}")

    if direction_info.get("signals"):
        with st.expander("🔍 Jelzések részletei"):
            for sig_name, sig_val in direction_info["signals"].items():
                direction = "bullish" if sig_val > 0.1 else "bearish" if sig_val < -0.1 else "neutral"
                icon = "🟢" if direction == "bullish" else "🔴" if direction == "bearish" else "⚪"
                st.write(f"{icon} **{sig_name}**: {sig_val:+.2f} ({direction})")


def render_details_tab(results: dict, cfg: dict):
    """Details tab: model info, feature importance, data quality."""
    df = results["df"]
    feature_cols = results["feature_cols"]

    # Data quality report
    st.subheader("📋 Adat minőségi elemzés")
    quality_warnings = generate_data_quality_report(df)
    if quality_warnings:
        for w in quality_warnings:
            st.markdown(w)
    else:
        st.success("✅ Az adatminőség megfelelő.")

    # Feature importance
    st.subheader("📊 Feature fontosság (anomáliák szempontjából)")
    importance = compute_feature_importance_for_anomalies(df, feature_cols)
    if not importance.empty:
        import plotly.express as px
        fig = px.bar(
            importance.head(10).reset_index(),
            x="importance", y="index",
            orientation="h",
            labels={"index": "Feature", "importance": "Korreláció az anomália score-ral"},
            template="plotly_dark",
            height=350,
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Feature fontosság nem elérhető.")

    # Model info
    st.subheader("🧠 Alkalmazott modellek")

    model_descriptions = {
        "Z-score (rolling)": "Gördülő ablakos z-score a napi log-hozamokra. Gyors, interpretálható, de egyváltozós.",
        "IQR": "Interkvartilis terjedelem alapú outlier detektálás. Robosztus és nem-parametrikus.",
        "Isolation Forest": "Random partíciókkal izolál outliereket magas dimenziós térben. Ensemble változat stabilabb.",
        "LOF (Local Outlier Factor)": "Lokális sűrűségkülönbség alapú detektálás. Adaptív nemlineáris klaszterekhez.",
        "One-Class SVM": "Hipersíkkal szeparálja a normál adatokat. RBF kernel nemlineáris boundary-t tanul.",
        "LSTM Autoencoder": "Szekvenciális mintázatokból tanul, rekonstrukciós hibából detektál. Flash crash, regime shift.",
        "STL Decomposition": "Trend + szezonalitás + reziduál felbontás. Szezonális anomáliák kiszűrése.",
        "CUSUM": "Kumulatív összeg monitoring. Regime shift és tartós trendváltás detektálás.",
        "Volume Anomaly": "Forgalmi csúcspont detektálás. Intézményi kereskedés, hír-alapú spike.",
        "Volatility Shift": "Volatilitás rezsimváltás érzékelése. Piaci stressz korai jelzése.",
    }

    for name, desc in model_descriptions.items():
        with st.expander(f"🔹 {name}"):
            st.write(desc)

    # Raw data
    st.subheader("📊 Nyers adatok")
    st.dataframe(df.tail(50), width='stretch', height=400)

    # Full export
    csv_full = df.to_csv()
    st.download_button(
        "📥 Teljes adathalmaz exportálása (CSV)",
        data=csv_full,
        file_name=f"{cfg['ticker']}_full_analysis.csv",
        mime="text/csv",
    )


def render_backtesting_tab(results: dict, cfg: dict):
    """Backtesting tab: walk-forward validation results and signal quality."""
    backtest_result = results.get("backtest_result")
    signal_quality = results.get("signal_quality")

    if backtest_result is None:
        st.info("🔧 Backtesting nincs bekapcsolva. Engedélyezd a bal oldali panelen a '🧪 Haladó modellek' szekcióban.")
        return

    st.subheader("📈 Walk-forward Backtesting Eredmények")

    # Key metrics
    cols = st.columns(5)
    with cols[0]:
        st.metric("Irány-pontosság", f"{backtest_result.directional_accuracy:.1%}")
    with cols[1]:
        st.metric("Sharpe-ráta", f"{backtest_result.sharpe_ratio:.2f}")
    with cols[2]:
        st.metric("Max lehívás", f"{backtest_result.max_drawdown:.1%}")
    with cols[3]:
        st.metric("Profit Factor", f"{backtest_result.profit_factor:.2f}")
    with cols[4]:
        st.metric("Windows", f"{backtest_result.n_windows}")

    st.markdown("---")

    # Equity curve
    if backtest_result.equity_curve:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=backtest_result.equity_curve,
            mode="lines",
            name="Equity",
            line=dict(color="#00d4aa", width=2),
        ))
        fig.update_layout(
            title="Equity görbe (backtesting)",
            template="plotly_dark",
            xaxis_title="Window",
            yaxis_title="Kumulatív hozam",
            height=350,
        )
        st.plotly_chart(fig, width='stretch')

    # Signal quality
    if signal_quality:
        st.subheader("📊 Jelzés minőség")
        grade = signal_quality.get("grade", "?")
        grade_colors = {"A": "🟢", "B": "🟢", "C": "🟡", "D": "🟠", "F": "🔴"}
        st.markdown(f"### Osztályzat: {grade_colors.get(grade, '⚪')} **{grade}**")
        st.write(signal_quality.get("description", ""))

        if signal_quality.get("details"):
            with st.expander("Részletes értékelés"):
                for key, val in signal_quality["details"].items():
                    st.write(f"- **{key}**: {val}")

    # Anomaly backtest results
    df = results["df"]
    anom_backtest = backtest_anomalies(df)
    if anom_backtest:
        st.subheader("🔍 Anomália-detektálás pontossága")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Precision", f"{anom_backtest.get('precision', 0):.1%}")
        with cols[1]:
            st.metric("Recall", f"{anom_backtest.get('recall', 0):.1%}")
        with cols[2]:
            st.metric("F1-score", f"{anom_backtest.get('f1', 0):.1%}")
        with cols[3]:
            avg_ret = anom_backtest.get("avg_post_anomaly_return")
            if avg_ret is not None:
                st.metric("Átl. anomália utáni hozam", f"{avg_ret:+.2%}")


def render_shap_tab(results: dict, cfg: dict):
    """SHAP explainability tab."""
    shap_data = results.get("shap_data")
    llm_explanation = results.get("llm_explanation")

    if shap_data is None and llm_explanation is None:
        st.info("🔧 SHAP nincs bekapcsolva. Engedélyezd a bal oldali panelen a '🧪 Haladó modellek' szekcióban.")
        return

    st.subheader("🧠 SHAP Feature Importance")

    if shap_data is not None:
        importance_df = get_feature_importance_data(shap_data)
        if importance_df is not None and not importance_df.empty:
            import plotly.express as px
            fig = px.bar(
                importance_df.head(15),
                x="importance",
                y="feature",
                orientation="h",
                template="plotly_dark",
                height=450,
                labels={"importance": "SHAP hatás", "feature": "Feature"},
                color="importance",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
            st.plotly_chart(fig, width='stretch')

    if llm_explanation:
        st.markdown("---")
        st.subheader("💬 AI magyarázat")
        st.markdown(llm_explanation)


def render_sentiment_tab(results: dict, cfg: dict):
    """Sentiment analysis tab."""
    sentiment_report = results.get("sentiment_report")

    if sentiment_report is None:
        st.info("🔧 Szentiment elemzés nincs bekapcsolva. Engedélyezd a bal panelen az 'Alternatív adatok' szekcióban.")
        return

    st.subheader("📰 Szentiment elemzés")

    # Aggregate score
    score = sentiment_report.score
    sentiment_label = "Pozitív 📈" if score > 0.2 else "Negatív 📉" if score < -0.2 else "Semleges ➡️"
    cols = st.columns(4)
    with cols[0]:
        st.metric("Aggregált szentiment", f"{score:+.2f}")
    with cols[1]:
        st.metric("Hangulat", sentiment_label)
    with cols[2]:
        st.metric("Elemzési mód", sentiment_report.method.upper())
    with cols[3]:
        st.metric("Magas hatású hírek", sentiment_report.high_impact_news)

    # Individual article sentiments
    if sentiment_report.headline_analyses:
        st.markdown("---")
        st.subheader("📃 Elemzett cikkek")
        for item in sentiment_report.headline_analyses[:15]:
            item_score = item.get("score", 0)
            icon = "🟢" if item_score > 0.2 else "🔴" if item_score < -0.2 else "⚪"
            title = item.get("headline", "Ismeretlen cím")
            st.write(f"{icon} **{title}** — szentiment: {item_score:+.2f}")

    # SEC filing analysis
    if sentiment_report.filing_analysis:
        st.markdown("---")
        st.subheader("📄 SEC beadvány")
        fa = sentiment_report.filing_analysis
        st.write(f"- **{fa.get('form_type', '?')}** ({fa.get('date', '?')})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.title("📈 Tőzsdei Anomáliadetektálás & Előrejelzés")
    st.caption("Adatvezérelt anomáliaészlelés, mintafelismerés és valószínűségi előrejelzés")

    # Initialize session state
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = None
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None

    # Render sidebar and get config
    cfg = render_sidebar()

    # Run pipeline
    if cfg["run_btn"]:
        st.session_state["results"] = run_pipeline(cfg)
        st.session_state["cfg"] = cfg
        st.session_state["last_run"] = time.time()

    # Live mode auto-refresh
    if cfg["live_mode"] and st.session_state["last_run"]:
        elapsed = time.time() - st.session_state["last_run"]
        if elapsed > cfg["refresh_interval"]:
            st.session_state["results"] = run_pipeline(cfg)
            st.session_state["cfg"] = cfg
            st.session_state["last_run"] = time.time()
            st.rerun()

    # Show results or welcome message
    results = st.session_state["results"]
    saved_cfg = st.session_state.get("cfg", cfg)

    if results is None:
        st.markdown("""
        ### 👋 Üdvözöllek!

        Ez az alkalmazás tőzsdei idősorok anomáliadetektálására és előrejelzésére szolgál.

        **Használat:**
        1. Válaszd ki az eszközt a bal oldali panelen
        2. Állítsd be az időszakot és a modell paramétereket
        3. Kattints a **🚀 Elemzés indítása** gombra

        **Támogatott eszközök:**
        - 🏢 Részvények (AAPL, MSFT, TSLA, ...)
        - 📊 Indexek (S&P 500, NASDAQ, DAX, BUX, ...)
        - 📦 ETF-ek (SPY, QQQ, GLD, ...)
        - 💱 Devizapárok (EUR/USD, EUR/HUF, ...)
        - ₿ Kriptovaluták (BTC, ETH, SOL, ...)

        **Alkalmazott anomáliadetektálási módszerek:**
        Z-score, IQR, Isolation Forest, LOF, One-Class SVM, LSTM Autoencoder,
        STL dekompozíció, CUSUM change-point, volume anomália, volatilitás rezsimváltás.

        **Előrejelzési módszerek:**
        ARIMA, XGBoost, LSTM, GARCH, LightGBM Stacking, TFT, Online tanulás — valószínűségi kimenettel és konfidencia intervallummal.

        **Kiegészítő modulok:**
        📡 Alternatív adatok (FRED, Fear&Greed, opciók, short interest) |
        📰 LLM szentiment | 📊 Backtesting | 🧠 SHAP magyarázatok
        """)
        return

    # Live mode indicator
    if cfg["live_mode"]:
        last = st.session_state.get("last_run", 0)
        next_refresh = max(0, cfg["refresh_interval"] - (time.time() - last))
        st.sidebar.info(f"🔴 Live mód aktív\nKövetkező frissítés: {int(next_refresh)}s")

    # Tab layout — build dynamic tab list based on enabled features
    tab_names = [
        "📊 Dashboard",
        "🔍 Anomáliák",
        "🔮 Előrejelzés",
        "📋 Részletek",
    ]
    # Always show extra tabs if results contain the data (user enabled them)
    has_backtest = results.get("backtest_result") is not None
    has_shap = results.get("shap_data") is not None or results.get("llm_explanation") is not None
    has_sentiment = results.get("sentiment_report") is not None

    if has_backtest:
        tab_names.append("📈 Backtesting")
    if has_shap:
        tab_names.append("🧠 SHAP")
    if has_sentiment:
        tab_names.append("📰 Szentiment")

    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_dashboard_tab(results, saved_cfg)

    with tabs[1]:
        render_anomaly_tab(results, saved_cfg)

    with tabs[2]:
        render_forecast_tab(results, saved_cfg)

    with tabs[3]:
        render_details_tab(results, saved_cfg)

    idx = 4
    if has_backtest:
        with tabs[idx]:
            render_backtesting_tab(results, saved_cfg)
        idx += 1
    if has_shap:
        with tabs[idx]:
            render_shap_tab(results, saved_cfg)
        idx += 1
    if has_sentiment:
        with tabs[idx]:
            render_sentiment_tab(results, saved_cfg)


if __name__ == "__main__":
    main()
