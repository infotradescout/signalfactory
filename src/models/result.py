"""
src/models/result.py — Unified prediction result container.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    """
    Returned by every model's .predict() method.
    Contains both the prediction and interpretability metadata.
    """

    # Core outputs
    predictions: pd.Series           # predicted values / labels
    probabilities: Optional[pd.DataFrame] = None  # class probas for classifiers
    forecast: Optional[pd.DataFrame] = None       # future forecast (forecasters only)

    # Model metadata
    model_type: str = ""
    target_label: str = ""
    feature_importance: Optional[pd.Series] = None

    # Evaluation metrics (if y_true was available)
    metrics: dict = field(default_factory=dict)

    # ── helpers ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"Target      : {self.target_label}",
            f"Model       : {self.model_type}",
            f"Predictions : {len(self.predictions)} rows",
        ]
        if self.metrics:
            lines.append("Metrics     :")
            for k, v in self.metrics.items():
                lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if self.feature_importance is not None and not self.feature_importance.empty:
            top5 = self.feature_importance.nlargest(5)
            lines.append("Top features:")
            for feat, score in top5.items():
                lines.append(f"  {feat}: {score:.4f}")
        return "\n".join(lines)

    def latest_prediction(self):
        """Return the most recent prediction value."""
        if self.predictions.empty:
            return None
        return self.predictions.iloc[-1]
