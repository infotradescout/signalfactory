"""
src/outputs/report.py — Build structured human- and machine-readable reports.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..models.result import PredictionResult

logger = logging.getLogger(__name__)


class ReportBuilder:
    """Converts a PredictionResult into various output formats."""

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self, result: PredictionResult) -> dict:
        """Serialise a PredictionResult to a plain dict."""
        d = {
            "generated_at": datetime.utcnow().isoformat(),
            "target": result.target_label,
            "model": result.model_type,
            "metrics": result.metrics,
            "latest_prediction": _safe_scalar(result.latest_prediction()),
        }
        if result.feature_importance is not None:
            d["top_features"] = result.feature_importance.head(10).to_dict()
        if result.forecast is not None:
            d["forecast"] = result.forecast.to_dict(orient="records")
        if result.probabilities is not None:
            d["latest_probabilities"] = result.probabilities.iloc[-1].to_dict()
        return d

    def save_json(self, result: PredictionResult, filename: Optional[str] = None) -> Path:
        data = self.to_dict(result)
        fname = filename or f"{result.target_label.replace(' ', '_')}_{_ts()}.json"
        path = self.output_dir / fname
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Report saved → %s", path)
        return path

    def save_csv(self, result: PredictionResult, filename: Optional[str] = None) -> Path:
        preds_df = result.predictions.rename("prediction").to_frame()
        if result.probabilities is not None:
            preds_df = preds_df.join(result.probabilities)
        if result.forecast is not None:
            preds_df = pd.concat(
                [preds_df, result.forecast.add_prefix("fc_")], axis=0
            )
        fname = filename or f"{result.target_label.replace(' ', '_')}_{_ts()}.csv"
        path = self.output_dir / fname
        preds_df.to_csv(path)
        logger.info("CSV report saved → %s", path)
        return path

    def print_summary(self, result: PredictionResult) -> None:
        print("\n" + "=" * 60)
        print(result.summary())
        print("=" * 60 + "\n")


# ── helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _safe_scalar(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)
