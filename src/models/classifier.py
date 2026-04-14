"""
src/models/classifier.py — Classification models for categorical/ordinal targets.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .result import PredictionResult

logger = logging.getLogger(__name__)

_MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, solver="lbfgs", C=1.0
    ),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    ),
}


class Classifier:
    """Train and predict a categorical target."""

    def __init__(self, model_name: str = "random_forest", target_label: str = "",
                 classes: Optional[list[str]] = None):
        self.model_name = model_name
        self.target_label = target_label
        self.classes = classes
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._model = _MODELS.get(model_name, _MODELS["random_forest"])
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Classifier":
        y_enc = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        self._model.fit(X_scaled, y_enc)
        self._is_fitted = True
        logger.info(
            "Classifier '%s' trained on %d samples, classes: %s",
            self.model_name, len(y), list(self.label_encoder.classes_)
        )
        return self

    def predict(self, X: pd.DataFrame, y_true: Optional[pd.Series] = None) -> PredictionResult:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict().")

        X_scaled = self.scaler.transform(X)
        pred_enc = self._model.predict(X_scaled)
        pred_labels = self.label_encoder.inverse_transform(pred_enc)
        preds = pd.Series(pred_labels, index=X.index, name="prediction")

        # Probability matrix
        class_names = list(self.label_encoder.classes_)
        if hasattr(self._model, "predict_proba"):
            proba_arr = self._model.predict_proba(X_scaled)
            probabilities = pd.DataFrame(proba_arr, index=X.index, columns=class_names)
        else:
            probabilities = None

        metrics = {}
        if y_true is not None:
            y_aligned = y_true.reindex(X.index).dropna()
            known_mask = y_aligned.isin(self.label_encoder.classes_)
            y_known = y_aligned[known_mask]
            if not y_known.empty:
                y_true_enc = self.label_encoder.transform(y_known)
                pred_known = pred_enc[known_mask.values]
                metrics = {
                    "accuracy": accuracy_score(y_true_enc, pred_known),
                    "f1_weighted": f1_score(y_true_enc, pred_known, average="weighted"),
                    "eval_rows": int(len(y_known)),
                }

        return PredictionResult(
            predictions=preds,
            probabilities=probabilities,
            model_type=f"classifier/{self.model_name}",
            target_label=self.target_label,
            feature_importance=self._get_feature_importance(X.columns.tolist()),
            metrics=metrics,
        )

    def _get_feature_importance(self, feature_names: list[str]) -> Optional[pd.Series]:
        if hasattr(self._model, "feature_importances_"):
            return pd.Series(
                self._model.feature_importances_, index=feature_names
            ).sort_values(ascending=False)
        if hasattr(self._model, "coef_"):
            coef = np.abs(self._model.coef_).mean(axis=0)
            return pd.Series(coef, index=feature_names).sort_values(ascending=False)
        return None
