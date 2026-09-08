"""
Confidence Calibration — isotonic regression and Platt scaling.

Ensures that when the model says "70% confident", actual accuracy is ~70%.
Applies post-hoc calibration to raw confidence scores from forecasting models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field

from app.config import logger

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class CalibrationResult:
    """Result of confidence calibration."""
    raw_confidence: float
    calibrated_confidence: float
    method: str
    reliability_curve: dict = field(default_factory=dict)


class ConfidenceCalibrator:
    """
    Post-hoc confidence calibration for forecast models.

    Methods:
      - isotonic: non-parametric, fits a monotone function
      - platt: parametric (logistic sigmoid), good with small samples

    Usage:
      calibrator = ConfidenceCalibrator()
      calibrator.fit(raw_confidences, actual_outcomes)
      calibrated = calibrator.calibrate(0.75)
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._fitted = False
        self._isotonic = None
        self._platt = None

    def fit(
        self,
        confidences: np.ndarray,
        actuals: np.ndarray,
    ) -> "ConfidenceCalibrator":
        """
        Fit the calibrator.

        Parameters
        ----------
        confidences : array of shape (n,)
            Raw confidence scores from models (0 to 1).
        actuals : array of shape (n,)
            Binary outcomes: 1 if the prediction was correct, 0 otherwise.
        """
        if not HAS_SKLEARN:
            logger.warning("scikit-learn required for calibration")
            return self

        confidences = np.asarray(confidences, dtype=float)
        actuals = np.asarray(actuals, dtype=float)

        valid = np.isfinite(confidences) & np.isfinite(actuals)
        confidences = confidences[valid]
        actuals = actuals[valid]

        if len(confidences) < 10:
            logger.warning("Not enough data for calibration (need 10+, got %d)", len(confidences))
            return self

        if self.method == "isotonic":
            self._isotonic = IsotonicRegression(
                y_min=0.05, y_max=0.95, out_of_bounds="clip"
            )
            self._isotonic.fit(confidences, actuals)
            self._fitted = True

        elif self.method == "platt":
            self._platt = LogisticRegression(C=1.0, solver="lbfgs")
            self._platt.fit(confidences.reshape(-1, 1), actuals)
            self._fitted = True

        logger.info("Calibrator fitted (%s) on %d samples", self.method, len(confidences))
        return self

    def calibrate(self, confidence: float) -> float:
        """Transform a raw confidence score to a calibrated one."""
        if not self._fitted:
            return confidence

        conf_arr = np.array([confidence])

        if self.method == "isotonic" and self._isotonic is not None:
            return float(self._isotonic.predict(conf_arr)[0])
        elif self.method == "platt" and self._platt is not None:
            return float(self._platt.predict_proba(conf_arr.reshape(-1, 1))[0, 1])

        return confidence

    def compute_reliability_curve(
        self,
        confidences: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """
        Compute reliability diagram data (calibration curve).

        Returns dict with:
          - bin_edges: array of bin boundaries
          - mean_predicted: avg predicted confidence per bin
          - mean_actual: fraction of correct predictions per bin
          - counts: number of samples per bin
          - ece: Expected Calibration Error
        """
        confidences = np.asarray(confidences, dtype=float)
        actuals = np.asarray(actuals, dtype=float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        mean_predicted = []
        mean_actual = []
        counts = []

        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() == 0:
                mean_predicted.append(np.nan)
                mean_actual.append(np.nan)
                counts.append(0)
            else:
                mean_predicted.append(confidences[mask].mean())
                mean_actual.append(actuals[mask].mean())
                counts.append(int(mask.sum()))

        # ECE (Expected Calibration Error)
        total = sum(counts)
        ece = 0.0
        for mp, ma, c in zip(mean_predicted, mean_actual, counts):
            if c > 0 and np.isfinite(mp) and np.isfinite(ma):
                ece += (c / total) * abs(ma - mp)

        return {
            "bin_edges": bin_edges.tolist(),
            "mean_predicted": mean_predicted,
            "mean_actual": mean_actual,
            "counts": counts,
            "ece": round(ece, 4),
        }


def calibrate_forecast_results(
    results: list,
    calibrator: Optional[ConfidenceCalibrator] = None,
) -> list:
    """
    Apply calibration to a list of ForecastResult objects.
    Modifies confidence in-place and returns the list.
    """
    if calibrator is None or not calibrator._fitted:
        return results

    for r in results:
        raw = r.confidence
        r.confidence = round(calibrator.calibrate(raw), 3)
        if "raw_confidence" not in r.metadata:
            r.metadata["raw_confidence"] = raw

    return results
