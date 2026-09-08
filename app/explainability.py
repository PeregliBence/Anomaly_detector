"""
Explainability module for anomaly detection results.

Provides:
  - Per-anomaly explanations with contributing factors
  - Model contribution breakdown
  - Feature importance analysis
  - Data quality and confidence warnings
"""

import numpy as np
import pandas as pd

from app.config import logger


def _safe_float(row, key: str, default=np.nan) -> float:
    """Safely extract a float value from a row."""
    try:
        val = row[key]
        if isinstance(val, (pd.Series, np.ndarray)):
            val = val.iloc[0] if hasattr(val, "iloc") else val[0]
        return float(val)
    except Exception:
        return default


def build_anomaly_explanation(row: pd.Series) -> str:
    """
    Generate a Markdown explanation for why a specific data point was flagged as anomalous.
    Includes model contributions, feature context, and interpretive notes.
    """
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    date_str = row.name.strftime("%Y-%m-%d") if hasattr(row.name, "strftime") else str(row.name)

    price = _safe_float(row, "Close")
    z = _safe_float(row, "zscore")
    iso = _safe_float(row, "iso_score")
    lof = _safe_float(row, "lof_score")
    lstm_err = _safe_float(row, "lstm_error")
    ocsvm = _safe_float(row, "ocsvm_score")
    ens = _safe_float(row, "ensemble_score")
    severity = row.get("anomaly_severity", "unknown")

    lines = [
        f"### Anomália elemzés — {date_str}",
        "",
        f"**Záróár:** {price:.2f}" if np.isfinite(price) else "",
        f"**Súlyosság:** {'🔴 Erős' if severity == 'strong' else '🟡 Enyhe' if severity == 'mild' else '⚪ Normál'}",
        f"**Ensemble score:** {ens:.3f}" if np.isfinite(ens) else "",
        "",
        "---",
        "#### Modellek jelzései",
    ]

    # Model contributions
    model_flags = {
        "Z-score": ("anomaly_zscore", z, "z-score = {:.2f}"),
        "Isolation Forest": ("anomaly_iso", iso, "score = {:.4f} (alacsonyabb = gyanúsabb)"),
        "LOF": ("anomaly_lof", lof, "score = {:.4f} (magasabb = gyanúsabb)"),
        "LSTM Autoencoder": ("anomaly_lstm", lstm_err, "rekonstrukciós hiba = {:.6f}"),
        "One-Class SVM": ("anomaly_ocsvm", ocsvm, "score = {:.4f}"),
    }

    for name, (flag_col, score_val, fmt) in model_flags.items():
        flagged = int(_safe_float(row, flag_col, 0))
        icon = "✅" if flagged else "❌"
        score_str = fmt.format(score_val) if np.isfinite(score_val) else "n/a"
        lines.append(f"- {icon} **{name}**: {score_str}")

    # Additional model flags
    for col, name in [("anomaly_stl", "STL Decomposition"), ("anomaly_cusum", "CUSUM Change-Point"),
                       ("anomaly_volume", "Volume Anomaly"), ("anomaly_vol_shift", "Volatility Shift")]:
        if col in row.index:
            flagged = int(_safe_float(row, col, 0))
            icon = "✅" if flagged else "❌"
            lines.append(f"- {icon} **{name}**")

    # Context indicators
    lines.extend(["", "---", "#### Piaci kontextus"])

    context = {
        "20 napos volatilitás": ("vol_20", ".4f"),
        "RSI (14)": ("RSI_14", ".1f"),
        "MACD hisztogram": ("MACD_hist", ".4f"),
        "Bollinger %B": ("BB_pct", ".2f"),
        "ATR (14)": ("ATR_14", ".4f"),
        "Volume spike": ("volume_spike", ".2f"),
        "Drawdown": ("drawdown", ".3f"),
        "Volatilitás-igazított hozam": ("vol_adj_return", ".4f"),
    }

    for label, (col, fmt) in context.items():
        val = _safe_float(row, col)
        if np.isfinite(val):
            lines.append(f"- **{label}:** {val:{fmt}}")

    # Trend info
    trend = row.get("trend", None)
    if trend and isinstance(trend, str):
        trend_icon = {"bullish": "📈", "bearish": "📉", "sideways": "➡️"}.get(trend, "")
        lines.append(f"- **Trend:** {trend_icon} {trend}")

    vol_regime = row.get("vol_regime", None)
    if vol_regime and isinstance(vol_regime, str):
        lines.append(f"- **Volatilitás rezsim:** {vol_regime}")

    # Top deviating features
    dev_cols = [c for c in row.index if isinstance(c, str) and c.startswith("dev_")]
    dev_pairs = [(c.replace("dev_", ""), float(_safe_float(row, c))) for c in dev_cols]
    dev_pairs = [(n, v) for n, v in dev_pairs if np.isfinite(v)]

    if dev_pairs:
        dev_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        lines.extend(["", "#### Leginkább kilógó feature-ök"])
        for name, val in dev_pairs[:5]:
            bar = "🔺" if val > 0 else "🔻"
            lines.append(f"- {bar} **{name}**: {val:+.2f}σ")

    # Interpretation
    lines.extend(["", "---", "#### Értelmezés"])

    interpretations = []
    if np.isfinite(z) and abs(z) > 3:
        direction = "pozitív" if z > 0 else "negatív"
        interpretations.append(f"A napi hozam rendkívül {direction} irányú ({z:.1f}σ az átlagtól).")

    vol_val = _safe_float(row, "volume_spike")
    if np.isfinite(vol_val) and vol_val > 3:
        interpretations.append(f"Szokatlanul magas forgalom ({vol_val:.1f}× a medián felett).")

    rsi_val = _safe_float(row, "RSI_14")
    if np.isfinite(rsi_val):
        if rsi_val > 70:
            interpretations.append(f"Az RSI ({rsi_val:.0f}) túlvett állapotot jelez.")
        elif rsi_val < 30:
            interpretations.append(f"Az RSI ({rsi_val:.0f}) túladott állapotot jelez.")

    bb_val = _safe_float(row, "BB_pct")
    if np.isfinite(bb_val):
        if bb_val > 1:
            interpretations.append("Az árfolyam a felső Bollinger sáv felett van.")
        elif bb_val < 0:
            interpretations.append("Az árfolyam az alsó Bollinger sáv alatt van.")

    if interpretations:
        for interp in interpretations:
            lines.append(f"- {interp}")
    else:
        lines.append("- A modellek együttes jelzése alapján ez a pont anomáliának minősül, "
                      "de nincs egyetlen kimagaslóan kilógó faktor.")

    return "\n".join(lines)


def compute_feature_importance_for_anomalies(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Compute which features contribute most to anomaly detection.
    Uses correlation between feature deviation and ensemble score.
    """
    dev_cols = [f"dev_{c}" for c in feature_cols if f"dev_{c}" in df.columns]
    if not dev_cols or "ensemble_score" not in df.columns:
        return pd.DataFrame()

    correlations = {}
    for dc in dev_cols:
        feat_name = dc.replace("dev_", "")
        valid = df[[dc, "ensemble_score"]].dropna()
        if len(valid) > 10:
            correlations[feat_name] = valid[dc].abs().corr(valid["ensemble_score"])

    if not correlations:
        return pd.DataFrame()

    importance = pd.DataFrame.from_dict(correlations, orient="index", columns=["importance"])
    importance = importance.sort_values("importance", ascending=False)
    return importance


def generate_data_quality_report(df: pd.DataFrame) -> list[str]:
    """Generate warnings about data quality issues that may affect model reliability."""
    warnings = []

    if len(df) < 200:
        warnings.append(f"⚠️ Kevés adat ({len(df)} pont). Az ML modellek megbízhatósága korlátozott.")
    elif len(df) < 500:
        warnings.append(f"ℹ️ Közepes adatmennyiség ({len(df)} pont). Több adat javítaná a pontosságot.")

    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        nan_pct = nan_count / (len(df) * len(df.columns))
        if nan_pct > 0.01:
            warnings.append(f"⚠️ {nan_pct:.1%} hiányzó adat az adathalmazban.")

    if "vol_20" in df.columns:
        vol = df["vol_20"].iloc[-1] if not df["vol_20"].empty else 0
        vol_mean = df["vol_20"].mean()
        if vol > vol_mean * 2:
            warnings.append("⚠️ A jelenlegi volatilitás jelentősen meghaladja az átlagot. "
                            "Az előrejelzések megbízhatósága romolhat.")

    if "log_return" in df.columns:
        kurt = df["log_return"].kurtosis()
        if kurt > 10:
            warnings.append(f"ℹ️ A hozameloszlás vastagfarkú (kurtosis: {kurt:.1f}). "
                            "Ez normális tőzsdei adatoknál, de a statisztikai módszerek érzékenysége csökken.")

    return warnings
