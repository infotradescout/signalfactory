"""
src/models/forecaster.py — Time-series forecasting with ARIMA / SARIMAX.
Used when target type == 'forecast'.
"""
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .result import PredictionResult

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not installed — ARIMA forecaster unavailable")


class TimeSeriesForecaster:
    """
    Fit ARIMA (or SARIMAX when exogenous regressors are supplied) and
    produce a rolling in-sample fit + out-of-sample forecast.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (2, 1, 2),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        target_label: str = "",
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.target_label = target_label
        self._result = None
        self._series: Optional[pd.Series] = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "TimeSeriesForecaster":
        if not _STATSMODELS_AVAILABLE:
            raise RuntimeError("statsmodels is required for time-series forecasting. "
                               "Run: pip install statsmodels")
        self._series = y.dropna()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if X is not None and not X.empty:
                model = SARIMAX(
                    self._series,
                    exog=X.reindex(self._series.index),
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            else:
                model = ARIMA(self._series, order=self.order)

            self._result = model.fit()

        logger.info("ARIMA%s fitted. AIC=%.2f", self.order, self._result.aic)
        return self

    def predict(
        self,
        horizon: int = 12,
        X_future: Optional[pd.DataFrame] = None,
    ) -> PredictionResult:
        if self._result is None:
            raise RuntimeError("Model must be fitted before predict().")

        # In-sample fitted values
        fitted = pd.Series(self._result.fittedvalues, name="fitted")

        # Out-of-sample forecast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self._result.get_forecast(
                steps=horizon,
                exog=X_future if X_future is not None else None,
            )

        fc_mean = fc.predicted_mean
        fc_ci = fc.conf_int()

        # Build a tidy forecast dataframe
        forecast_df = pd.DataFrame({
            "forecast": fc_mean.values,
            "lower_95": fc_ci.iloc[:, 0].values,
            "upper_95": fc_ci.iloc[:, 1].values,
        })

        metrics = {
            "AIC": float(self._result.aic),
            "BIC": float(self._result.bic),
        }

        return PredictionResult(
            predictions=fitted,
            forecast=forecast_df,
            model_type=f"forecast/arima{self.order}",
            target_label=self.target_label,
            metrics=metrics,
        )

    @staticmethod
    def auto_order(y: pd.Series) -> tuple[int, int, int]:
        """
        Heuristic order selection: test stationarity and suggest (p,d,q).
        Falls back to (2,1,2) which works for most macro series.
        """
        if not _STATSMODELS_AVAILABLE:
            return (2, 1, 2)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_pval = adfuller(y.dropna())[1]
            d = 0 if adf_pval < 0.05 else 1
            return (2, d, 2)
        except Exception:
            return (2, 1, 2)
