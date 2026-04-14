"""
src/models/regressor.py — Regression models for continuous numeric targets.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .result import PredictionResult

logger = logging.getLogger(__name__)

_MODELS = {
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    ),
    "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "ridge": Ridge(alpha=1.0),
    "ensemble": None,  # built dynamically below
}


def _build_ensemble() -> VotingRegressor:
    return VotingRegressor([
        ("gbm", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("ridge", Ridge(alpha=1.0)),
    ])


class Regressor:
    """Train and predict a continuous numeric target."""

    def __init__(self, model_name: str = "gradient_boosting", target_label: str = ""):
        self.model_name = model_name
        self.target_label = target_label
        self.scaler = StandardScaler()
        self._model = _MODELS.get(model_name) or _build_ensemble()
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Regressor":
        X_scaled = self.scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._is_fitted = True
        logger.info("Regressor '%s' trained on %d samples", self.model_name, len(y))
        return self

    def predict(self, X: pd.DataFrame, y_true: Optional[pd.Series] = None) -> PredictionResult:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        X_scaled = self.scaler.transform(X)
        preds = pd.Series(self._model.predict(X_scaled), index=X.index, name="prediction")

        metrics = {}
        if y_true is not None:
            aligned_y = y_true.reindex(X.index).dropna()
            aligned_preds = preds.reindex(aligned_y.index)
            metrics = {
                "MAE": mean_absolute_error(aligned_y, aligned_preds),
                "RMSE": np.sqrt(mean_squared_error(aligned_y, aligned_preds)),
                "R2": r2_score(aligned_y, aligned_preds),
            }

        importance = self._get_feature_importance(X.columns.tolist())

        return PredictionResult(
            predictions=preds,
            model_type=f"regressor/{self.model_name}",
            target_label=self.target_label,
            feature_importance=importance,
            metrics=metrics,
        )

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
        """Time-series cross-validation metrics."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mae_scores, r2_scores = [], []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            import copy
            model = copy.deepcopy(self._model)
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_te_s)
            mae_scores.append(mean_absolute_error(y_te, preds))
            r2_scores.append(r2_score(y_te, preds))
        return {
            "cv_mae_mean": float(np.mean(mae_scores)),
            "cv_mae_std": float(np.std(mae_scores)),
            "cv_r2_mean": float(np.mean(r2_scores)),
        }

    def _get_feature_importance(self, feature_names: list[str]) -> Optional[pd.Series]:
        if hasattr(self._model, "feature_importances_"):
            return pd.Series(
                self._model.feature_importances_, index=feature_names
            ).sort_values(ascending=False)
        if hasattr(self._model, "coef_"):
            return pd.Series(
                np.abs(self._model.coef_), index=feature_names
            ).sort_values(ascending=False)
        return None
