"""
Plotly-based interactive visualization for financial time series analysis.

Charts:
  - Main price chart with anomaly markers and trend overlays
  - Volume chart with anomaly highlighting
  - Rolling volatility panel with regime shading
  - Anomaly score heatmap
  - RSI panel with overbought/oversold zones
  - Forecast chart with confidence intervals
  - Combined multi-panel dashboard figure
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.config import logger


# Color scheme
COLORS = {
    "price": "#3b82f6",
    "sma20": "#f59e0b",
    "sma50": "#10b981",
    "sma200": "#8b5cf6",
    "volume_up": "#22c55e",
    "volume_down": "#ef4444",
    "anomaly_mild": "#fbbf24",
    "anomaly_strong": "#dc2626",
    "forecast": "#6366f1",
    "forecast_fill": "rgba(99, 102, 241, 0.15)",
    "bg": "#0e1117",
    "grid": "rgba(255,255,255,0.08)",
    "text": "#e5e7eb",
}


def create_main_chart(
    df: pd.DataFrame,
    ticker: str,
    show_sma: bool = True,
    show_bollinger: bool = True,
) -> go.Figure:
    """
    Main price chart with SMA overlays, Bollinger Bands, and anomaly markers.
    """
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="Záróár",
        line=dict(color=COLORS["price"], width=1.5),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Ár: %{y:.2f}<extra></extra>",
    ))

    # Moving averages
    if show_sma:
        for col, color, name in [
            ("SMA_20", COLORS["sma20"], "SMA 20"),
            ("SMA_50", COLORS["sma50"], "SMA 50"),
            ("SMA_200", COLORS["sma200"], "SMA 200"),
        ]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    mode="lines", name=name,
                    line=dict(color=color, width=1, dash="dot"),
                    visible="legendonly",
                ))

    # Bollinger Bands
    if show_bollinger and "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"],
            mode="lines", name="BB Upper",
            line=dict(color="rgba(139,92,246,0.3)", width=1),
            visible="legendonly",
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"],
            mode="lines", name="BB Lower",
            line=dict(color="rgba(139,92,246,0.3)", width=1),
            fill="tonexty", fillcolor="rgba(139,92,246,0.05)",
            visible="legendonly",
        ))

    # Anomaly markers
    if "anomaly_severity" in df.columns:
        # Mild anomalies
        mild = df[df["anomaly_severity"] == "mild"]
        if not mild.empty:
            fig.add_trace(go.Scatter(
                x=mild.index, y=mild["Close"],
                mode="markers", name="Enyhe anomália",
                marker=dict(color=COLORS["anomaly_mild"], size=8, symbol="circle",
                            line=dict(width=1, color="#000")),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Ár: %{y:.2f}<br>⚠️ Enyhe anomália<extra></extra>",
            ))

        # Strong anomalies
        strong = df[df["anomaly_severity"] == "strong"]
        if not strong.empty:
            fig.add_trace(go.Scatter(
                x=strong.index, y=strong["Close"],
                mode="markers", name="Erős anomália",
                marker=dict(color=COLORS["anomaly_strong"], size=12, symbol="diamond",
                            line=dict(width=1.5, color="#fff")),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Ár: %{y:.2f}<br>🔴 Erős anomália<extra></extra>",
            ))
    elif "anomaly_ensemble" in df.columns:
        anom = df[df["anomaly_ensemble"] == 1]
        if not anom.empty:
            fig.add_trace(go.Scatter(
                x=anom.index, y=anom["Close"],
                mode="markers", name="Anomália",
                marker=dict(color=COLORS["anomaly_strong"], size=8, symbol="circle"),
            ))

    # CUSUM change points
    if "anomaly_cusum" in df.columns:
        cp = df[df["anomaly_cusum"] == 1]
        if not cp.empty:
            fig.add_trace(go.Scatter(
                x=cp.index, y=cp["Close"],
                mode="markers", name="Trendváltás (CUSUM)",
                marker=dict(color="#06b6d4", size=10, symbol="triangle-up",
                            line=dict(width=1, color="#000")),
                visible="legendonly",
            ))

    fig.update_layout(
        title=f"{ticker} — Árfolyam és anomáliák",
        xaxis_title="", yaxis_title="Árfolyam",
        template="plotly_dark",
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=20),
    )

    return fig


def create_volume_chart(df: pd.DataFrame) -> go.Figure:
    """Volume bar chart with up/down coloring and anomaly highlights."""
    fig = go.Figure()

    colors = [COLORS["volume_up"] if df["Close"].iloc[i] >= df["Close"].iloc[max(0, i-1)]
              else COLORS["volume_down"] for i in range(len(df))]

    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, name="Volume",
        opacity=0.6,
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Volume: %{y:,.0f}<extra></extra>",
    ))

    # Highlight volume anomalies
    if "anomaly_volume" in df.columns:
        vol_anom = df[df["anomaly_volume"] == 1]
        if not vol_anom.empty:
            fig.add_trace(go.Bar(
                x=vol_anom.index, y=vol_anom["Volume"],
                marker_color="#f59e0b", name="Volume anomália",
                opacity=0.9,
            ))

    fig.update_layout(
        title="Forgalom",
        xaxis_title="", yaxis_title="Volume",
        template="plotly_dark", height=200,
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=20),
    )

    return fig


def create_volatility_chart(df: pd.DataFrame) -> go.Figure:
    """Rolling volatility with regime shading."""
    fig = go.Figure()

    for col, name, color in [
        ("vol_20", "20 napos", COLORS["price"]),
        ("vol_60", "60 napos", COLORS["sma50"]),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                mode="lines", name=name,
                line=dict(color=color, width=1.5),
            ))

    # Volatility regime background shading
    if "vol_regime" in df.columns:
        regimes = df["vol_regime"]
        for regime, color in [("high", "rgba(239,68,68,0.1)"), ("medium", "rgba(251,191,36,0.05)")]:
            mask = regimes == regime
            if mask.any():
                segments = _find_segments(mask)
                for start, end in segments:
                    fig.add_vrect(
                        x0=df.index[start], x1=df.index[min(end, len(df) - 1)],
                        fillcolor=color, layer="below", line_width=0,
                    )

    fig.update_layout(
        title="Rolling volatilitás (annualizált)",
        xaxis_title="", yaxis_title="Volatilitás",
        template="plotly_dark", height=220,
        margin=dict(l=60, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    )

    return fig


def create_anomaly_score_chart(df: pd.DataFrame) -> go.Figure:
    """Ensemble anomaly score as area chart with threshold."""
    fig = go.Figure()

    if "ensemble_score" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ensemble_score"],
            mode="lines", name="Ensemble Score",
            line=dict(color="#f97316", width=1),
            fill="tozeroy", fillcolor="rgba(249,115,22,0.15)",
        ))

        # Threshold line
        valid = df["ensemble_score"].dropna()
        if len(valid) > 0:
            p95 = valid.quantile(0.95)
            fig.add_hline(y=p95, line_dash="dash", line_color="#ef4444",
                          annotation_text="95. percentilis", annotation_position="top right")

    fig.update_layout(
        title="Anomália score",
        xaxis_title="", yaxis_title="Score",
        template="plotly_dark", height=200,
        margin=dict(l=60, r=20, t=40, b=20),
    )

    return fig


def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """RSI chart with overbought/oversold zones."""
    fig = go.Figure()

    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI_14"],
            mode="lines", name="RSI (14)",
            line=dict(color="#a78bfa", width=1.5),
        ))

        # Overbought/oversold zones
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.1)", line_width=0)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.1)", line_width=0)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,68,68,0.5)")
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(34,197,94,0.5)")
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)")

    fig.update_layout(
        title="RSI (14)",
        xaxis_title="", yaxis_title="RSI",
        template="plotly_dark", height=200,
        yaxis=dict(range=[0, 100]),
        margin=dict(l=60, r=20, t=40, b=20),
    )

    return fig


def create_forecast_chart(
    df: pd.DataFrame,
    forecast_result,
    ticker: str = "",
) -> go.Figure:
    """
    Forecast chart showing historical prices and future predictions
    with confidence intervals.
    """
    fig = go.Figure()

    # Last N days of historical data for context
    n_context = 60
    hist = df.tail(n_context)

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        mode="lines", name="Historikus ár",
        line=dict(color=COLORS["price"], width=2),
    ))

    if forecast_result is not None:
        fc = forecast_result
        dates = pd.DatetimeIndex(fc.dates)

        # Connection line from last historical to first forecast
        bridge_dates = [hist.index[-1]] + list(dates)
        bridge_values = [float(hist["Close"].iloc[-1])] + list(fc.values)

        fig.add_trace(go.Scatter(
            x=bridge_dates, y=bridge_values,
            mode="lines", name=f"Előrejelzés ({fc.model_name})",
            line=dict(color=COLORS["forecast"], width=2, dash="dash"),
        ))

        # Confidence interval
        upper_bridge = [float(hist["Close"].iloc[-1])] + list(fc.upper)
        lower_bridge = [float(hist["Close"].iloc[-1])] + list(fc.lower)

        fig.add_trace(go.Scatter(
            x=bridge_dates, y=upper_bridge,
            mode="lines", name="Felső határ",
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=bridge_dates, y=lower_bridge,
            mode="lines", name="Konfidencia intervallum",
            line=dict(width=0),
            fill="tonexty", fillcolor=COLORS["forecast_fill"],
        ))

        # Direction annotation
        dir_icon = {"up": "📈", "down": "📉", "sideways": "➡️"}.get(fc.direction, "")
        fig.add_annotation(
            x=dates[-1], y=fc.values[-1],
            text=f"{dir_icon} {fc.direction.upper()}<br>Conf: {fc.confidence:.0%}",
            showarrow=True, arrowhead=2, arrowcolor=COLORS["forecast"],
            font=dict(size=12, color=COLORS["text"]),
            bgcolor="rgba(30,30,30,0.8)", bordercolor=COLORS["forecast"],
        )

    fig.update_layout(
        title=f"Előrejelzés — {ticker}",
        xaxis_title="", yaxis_title="Árfolyam",
        template="plotly_dark", height=400,
        margin=dict(l=60, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    )

    return fig


def create_dashboard_figure(
    df: pd.DataFrame,
    ticker: str,
    forecast_result=None,
) -> go.Figure:
    """
    Combined multi-panel dashboard figure with shared x-axis:
    Panel 1: Price + anomalies
    Panel 2: Volume
    Panel 3: Anomaly score
    Panel 4: RSI
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.18, 0.18, 0.19],
        subplot_titles=[f"{ticker} — Árfolyam", "Forgalom", "Anomália Score", "RSI (14)"],
    )

    # Panel 1: Price
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="Záróár",
        line=dict(color=COLORS["price"], width=1.5),
    ), row=1, col=1)

    for col, color, name in [("SMA_20", COLORS["sma20"], "SMA 20"), ("SMA_50", COLORS["sma50"], "SMA 50")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode="lines", name=name,
                line=dict(color=color, width=1, dash="dot"), visible="legendonly",
            ), row=1, col=1)

    # Anomaly markers
    if "anomaly_severity" in df.columns:
        mild = df[df["anomaly_severity"] == "mild"]
        strong = df[df["anomaly_severity"] == "strong"]
        if not mild.empty:
            fig.add_trace(go.Scatter(
                x=mild.index, y=mild["Close"],
                mode="markers", name="Enyhe",
                marker=dict(color=COLORS["anomaly_mild"], size=7, symbol="circle"),
            ), row=1, col=1)
        if not strong.empty:
            fig.add_trace(go.Scatter(
                x=strong.index, y=strong["Close"],
                mode="markers", name="Erős",
                marker=dict(color=COLORS["anomaly_strong"], size=10, symbol="diamond"),
            ), row=1, col=1)

    # Panel 2: Volume
    colors = [COLORS["volume_up"] if df["Close"].iloc[i] >= df["Close"].iloc[max(0, i-1)]
              else COLORS["volume_down"] for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, name="Volume", showlegend=False, opacity=0.6,
    ), row=2, col=1)

    # Panel 3: Anomaly score
    if "ensemble_score" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ensemble_score"],
            mode="lines", name="Score", showlegend=False,
            line=dict(color="#f97316", width=1),
            fill="tozeroy", fillcolor="rgba(249,115,22,0.15)",
        ), row=3, col=1)

    # Panel 4: RSI
    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI_14"],
            mode="lines", name="RSI", showlegend=False,
            line=dict(color="#a78bfa", width=1.5),
        ), row=4, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,68,68,0.4)", row=4, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(34,197,94,0.4)", row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=900,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=1, xanchor="right"),
        margin=dict(l=60, r=20, t=40, b=20),
    )
    fig.update_yaxes(title_text="Ár", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=3, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=4, col=1)

    return fig


def create_anomaly_zoom(df: pd.DataFrame, date, window: int = 15) -> go.Figure:
    """Zoom into a specific anomaly point with ±window context."""
    try:
        center = df.index.get_loc(date)
    except KeyError:
        fig = go.Figure()
        fig.add_annotation(text="A kiválasztott dátum nem található az adatban.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", height=300)
        return fig
    start = max(0, center - window)
    end = min(len(df) - 1, center + window)
    zoom = df.iloc[start:end + 1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=zoom.index, y=zoom["Close"],
        mode="lines+markers", name="Záróár",
        line=dict(color=COLORS["price"], width=2),
        marker=dict(size=4),
    ))

    # Highlight the anomaly point
    fig.add_trace(go.Scatter(
        x=[date], y=[float(df.loc[date, "Close"])],
        mode="markers", name="Anomália",
        marker=dict(color=COLORS["anomaly_strong"], size=14, symbol="diamond",
                    line=dict(width=2, color="#fff")),
    ))

    fig.update_layout(
        title="Ráközelítés az anomáliára",
        template="plotly_dark", height=300,
        margin=dict(l=60, r=20, t=40, b=20),
        yaxis=dict(tickformat=".2f"),
    )

    return fig


def create_model_comparison_chart(model_counts: dict) -> go.Figure:
    """Bar chart comparing anomaly counts across models."""
    names = list(model_counts.keys())
    counts = list(model_counts.values())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=counts,
        marker_color=["#3b82f6", "#f97316", "#a78bfa", "#10b981", "#ef4444", "#06b6d4", "#f59e0b"][:len(names)],
        text=counts, textposition="auto",
    ))

    fig.update_layout(
        title="Anomáliák modellenként",
        template="plotly_dark", height=300,
        xaxis_title="Modell", yaxis_title="Anomáliák száma",
        margin=dict(l=60, r=20, t=40, b=20),
    )

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_segments(mask: pd.Series) -> list[tuple[int, int]]:
    """Find contiguous True segments in a boolean series. Returns list of (start, end) indices."""
    segments = []
    in_segment = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_segment:
            start = i
            in_segment = True
        elif not val and in_segment:
            segments.append((start, i - 1))
            in_segment = False
    if in_segment:
        segments.append((start, len(mask) - 1))
    return segments
