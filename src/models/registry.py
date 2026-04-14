"""
src/models/registry.py
One-stop factory: given a target config, return the right fitted model.
Also handles serialisation (save / load) via joblib.
"""
import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from .classifier import Classifier
from .forecaster import TimeSeriesForecaster
from .regressor import Regressor
from .result import PredictionResult

logger = logging.getLogger(__name__)

_MODEL_DIR = Path("data/models")


class ModelRegistry:
    """
    Build, train, evaluate and persist models based on target config.

    Usage
    -----
    reg = ModelRegistry()
    result = reg.run(target_cfg, X, y)
    """

    def __init__(self, model_dir: str = "data/models"):
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, object] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        target_cfg: dict,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        horizon: int = 12,
    ) -> PredictionResult:
        """
        Train (if needed) and run inference for the given target.

        Parameters
        ----------
        target_cfg : one block from targets.yaml
        X          : feature matrix (from FeaturePipeline)
        y          : label series (may be None for pure forecast)
        horizon    : forecast horizon in periods (forecast targets only)
        """
        target_type = target_cfg.get("type", "regression")
        model_name = target_cfg.get("model", "gradient_boosting")
        label = target_cfg.get("label", "target")
        target_id = target_cfg.get("id", label.lower().replace(" ", "_"))

        if target_type == "forecast":
            return self._run_forecast(target_cfg, X, y, label, horizon)
        elif target_type == "classification":
            return self._run_classifier(target_cfg, X, y, model_name, label)
        else:
            return self._run_regressor(target_cfg, X, y, model_name, label)

    def save(self, key: str) -> Path:
        """Persist a trained model to disk."""
        model = self._models.get(key)
        if model is None:
            raise KeyError(f"No model found for key: {key!r}")
        path = self._model_dir / f"{key}.pkl"
        joblib.dump(model, path)
        logger.info("Model saved → %s", path)
        return path

    def load(self, key: str) -> object:
        """Load a previously saved model."""
        path = self._model_dir / f"{key}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved model at: {path}")
        model = joblib.load(path)
        self._models[key] = model
        return model

    # ── internal ──────────────────────────────────────────────────────────────

    def _run_regressor(
        self, cfg: dict, X: pd.DataFrame, y: Optional[pd.Series],
        model_name: str, label: str
    ) -> PredictionResult:
        if X.empty or y is None:
            raise ValueError("Regression requires non-empty X and y.")

        split = max(1, int(len(X) * 0.8))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        reg = Regressor(model_name=model_name, target_label=label)
        reg.fit(X_train, y_train)
        self._models[label] = reg

        result = reg.predict(X_test, y_true=y_test)
        cv = reg.cross_validate(X, y)
        result.metrics.update(cv)
        logger.info("Regression complete.\n%s", result.summary())
        return result

    def _run_classifier(
        self, cfg: dict, X: pd.DataFrame, y: Optional[pd.Series],
        model_name: str, label: str
    ) -> PredictionResult:
        if X.empty or y is None:
            raise ValueError("Classification requires non-empty X and y.")

        classes = cfg.get("classes")
        split = max(1, int(len(X) * 0.8))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        clf = Classifier(model_name=model_name, target_label=label, classes=classes)
        clf.fit(X_train, y_train)
        self._models[label] = clf

        result = clf.predict(X_test, y_true=y_test)
        logger.info("Classification complete.\n%s", result.summary())
        return result

    def _run_forecast(
        self, cfg: dict, X: pd.DataFrame, y: Optional[pd.Series],
        label: str, horizon: int
    ) -> PredictionResult:
        if y is None and not X.empty:
            # Use first column of X as the series to forecast
            y = X.iloc[:, 0]

        if y is None:
            raise ValueError("Forecast requires at least one numeric time series.")

        order = TimeSeriesForecaster.auto_order(y)
        forecaster = TimeSeriesForecaster(order=order, target_label=label)
        # Use univariate forecasting here to avoid requiring future exogenous inputs
        # during out-of-sample prediction in the dashboard batch workflow.
        forecaster.fit(y, None)
        self._models[label] = forecaster

        result = forecaster.predict(horizon=horizon)
        logger.info("Forecast complete.\n%s", result.summary())
        return result
