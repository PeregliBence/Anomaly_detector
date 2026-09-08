"""
SHAP-based Explainability & LLM-powered Decision Support.

Provides:
  1. SHAP values for XGBoost/LightGBM feature importance
  2. LLM-generated natural language explanations
  3. Visual feature importance data for charts
  4. "Why did the model predict this?" answers

Dependencies: shap (optional), openai (optional for LLM explanations)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field

from app.config import logger, _env

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


@dataclass
class ExplainabilityResult:
    """Container for SHAP-based explanation."""
    feature_names: list[str] = field(default_factory=list)
    shap_values: Optional[np.ndarray] = None
    base_value: float = 0.0
    top_features: list[dict] = field(default_factory=list)
    llm_explanation: str = ""
    prediction_value: float = 0.0
    prediction_direction: str = "sideways"


# ============================================================================
# 1. SHAP VALUES COMPUTATION
# ============================================================================

def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list[str],
    model_type: str = "tree",
    max_samples: int = 200,
) -> Optional[ExplainabilityResult]:
    """
    Compute SHAP values for a trained model.

    Parameters
    ----------
    model : trained XGBoost/LightGBM/sklearn model
    X : feature matrix (n_samples, n_features)
    feature_names : column names
    model_type : "tree" for tree-based, "kernel" for others
    max_samples : subsample for speed

    Returns ExplainabilityResult with SHAP values and top features.
    """
    if not HAS_SHAP:
        logger.info("shap not installed (pip install shap)")
        return None

    try:
        # Subsample for performance
        if len(X) > max_samples:
            idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sample)
        else:
            explainer = shap.KernelExplainer(model.predict, X_sample[:50])
            shap_vals = explainer.shap_values(X_sample)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0] if len(shap_vals) > 0 else shap_vals

        result = ExplainabilityResult(
            feature_names=list(feature_names),
            shap_values=shap_vals,
            base_value=float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0]),
        )

        # Top features by mean absolute SHAP value
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]

        top_features = []
        for i in sorted_idx[:15]:
            top_features.append({
                "feature": feature_names[i],
                "importance": float(mean_abs_shap[i]),
                "direction": "positive" if np.mean(shap_vals[:, i]) > 0 else "negative",
                "last_shap": float(shap_vals[-1, i]) if len(shap_vals) > 0 else 0.0,
            })
        result.top_features = top_features

        logger.info("SHAP values computed: %d features, top=%s",
                     len(feature_names), top_features[0]["feature"] if top_features else "N/A")
        return result

    except Exception as e:
        logger.error("SHAP computation failed: %s", e)
        return None


def compute_shap_for_forecast(
    forecast_result,
    df: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> Optional[ExplainabilityResult]:
    """
    Extract SHAP values from a forecast result that contains a model.

    Works with ForecastResult objects from stacking ensemble or XGBoost
    that store the trained model in metadata.
    """
    if not HAS_SHAP:
        return None

    metadata = getattr(forecast_result, "metadata", {})

    # Try to get stored model
    model = metadata.get("lgb_model") or metadata.get("xgb_model")
    if model is None:
        return None

    # Rebuild feature matrix from the forecast's feature builder
    from app.advanced_forecasting import _create_gbm_features
    features = _create_gbm_features(df).dropna()

    if features.empty:
        return None

    X = features.values
    names = features.columns.tolist()

    result = compute_shap_values(model, X, names, model_type="tree")
    if result:
        result.prediction_direction = forecast_result.direction
        result.prediction_value = float(forecast_result.values[-1]) if len(forecast_result.values) > 0 else 0.0

    return result


# ============================================================================
# 2. LLM-POWERED EXPLANATION GENERATOR
# ============================================================================

def generate_llm_explanation(
    shap_result: Optional[ExplainabilityResult] = None,
    forecast_result=None,
    news_sentiment: Optional[dict] = None,
    ticker: str = "",
    model: str = "gemini-2.0-flash",
) -> str:
    """
    Generate a human-readable explanation using Gemini.

    Combines:
    - SHAP feature importance
    - Forecast prediction and confidence
    - News sentiment (if available)

    Falls back to a template-based explanation if Gemini is not available.
    """
    # Build context string
    context_parts = [f"Ticker: {ticker}"]

    if forecast_result:
        context_parts.append(
            f"Forecast: direction={forecast_result.direction}, "
            f"confidence={forecast_result.confidence:.0%}, "
            f"expected return={forecast_result.expected_return_pct:+.2f}%"
        )

    if shap_result and shap_result.top_features:
        top5 = shap_result.top_features[:5]
        feat_str = "; ".join(
            f"{f['feature']} (importance={f['importance']:.4f}, dir={f['direction']})"
            for f in top5
        )
        context_parts.append(f"Top SHAP features: {feat_str}")

    if news_sentiment:
        context_parts.append(
            f"News sentiment: {news_sentiment.get('overall', 'neutral')}, "
            f"score={news_sentiment.get('score', 0):.2f}, "
            f"high-impact news={news_sentiment.get('high_impact_count', 0)}"
        )

    context = "\n".join(context_parts)

    # Try Gemini explanation
    api_key = _env("GEMINI_API_KEY")
    if api_key and HAS_GEMINI:
        try:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model)

            prompt = (
                "You are a financial analysis explainer. Given model analysis data, "
                "write a clear 3-5 sentence explanation in Hungarian for a trader. "
                "Explain WHY the model made this prediction, which factors were most "
                "important, and what risks to watch. Be specific and actionable. "
                "Do not give investment advice — only explain the model's reasoning.\n\n"
                + context
            )

            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=400,
                ),
            )
            return response.text.strip()

        except Exception as e:
            logger.warning("Gemini explanation failed: %s", e)

    # Template fallback
    return _template_explanation(shap_result, forecast_result, news_sentiment, ticker)


def _template_explanation(
    shap_result: Optional[ExplainabilityResult],
    forecast_result,
    news_sentiment: Optional[dict],
    ticker: str,
) -> str:
    """Generate a template-based explanation without LLM."""
    lines = [f"### Modell magyarázat — {ticker}", ""]

    if forecast_result:
        dir_hu = {"up": "emelkedés", "down": "csökkenés", "sideways": "oldalazás"}
        lines.append(
            f"**Előrejelzés:** {dir_hu.get(forecast_result.direction, forecast_result.direction)} "
            f"({forecast_result.confidence:.0%} konfidencia, "
            f"várt elmozdulás: {forecast_result.expected_return_pct:+.2f}%)"
        )
        lines.append("")

    if shap_result and shap_result.top_features:
        lines.append("**Legfontosabb tényezők (SHAP):**")
        for f in shap_result.top_features[:5]:
            dir_icon = "📈" if f["direction"] == "positive" else "📉"
            lines.append(
                f"- {dir_icon} **{f['feature']}**: fontosság = {f['importance']:.4f}"
            )
        lines.append("")

    if news_sentiment:
        sent_hu = {"bullish": "pozitív", "bearish": "negatív", "neutral": "semleges"}
        lines.append(
            f"**Hírek hangulata:** {sent_hu.get(news_sentiment.get('overall', ''), 'ismeretlen')} "
            f"(score: {news_sentiment.get('score', 0):+.2f})"
        )

    if not any(x for x in [shap_result, forecast_result, news_sentiment]):
        lines.append("Nem áll rendelkezésre elegendő adat a magyarázathoz.")

    lines.append("")
    lines.append("*Ez az elemzés automatikusan generált, nem minősül befektetési tanácsnak.*")

    return "\n".join(lines)


# ============================================================================
# 3. FEATURE IMPORTANCE DATA FOR VISUALIZATION
# ============================================================================

def get_feature_importance_data(
    shap_result: Optional[ExplainabilityResult],
) -> pd.DataFrame:
    """
    Convert SHAP results to a DataFrame suitable for bar chart visualization.

    Returns DataFrame with columns: feature, importance, direction
    """
    if shap_result is None or not shap_result.top_features:
        return pd.DataFrame(columns=["feature", "importance", "direction"])

    return pd.DataFrame(shap_result.top_features[:15])
