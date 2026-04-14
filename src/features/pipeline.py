"""
src/features/pipeline.py
Orchestrates all feature assembly for a given target config.
Produces a clean wide DataFrame ready for modelling.
"""
import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .engineering import (
    encode_country,
    interaction_terms,
    lag_features,
    pct_change_features,
    rolling_features,
)

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Given raw DataFrames from various connectors, assemble and engineer
    a modelling-ready feature matrix.

    Usage
    -----
    pipe = FeaturePipeline(target_cfg)
    X, y = pipe.fit_transform(wb_df=..., fred_df=..., extra_dfs=[...])
    """

    def __init__(self, target_cfg: dict):
        self.target_cfg = target_cfg
        self.target_name = target_cfg.get("label", "target")
        self.feature_groups: dict[str, list[str]] = target_cfg.get("features", {})
        self._feature_columns: list[str] = []

    # ── public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        wb_df: Optional[pd.DataFrame] = None,
        fred_df: Optional[pd.DataFrame] = None,
        extra_dfs: Optional[list[pd.DataFrame]] = None,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Merge all data sources, engineer features, handle missingness.
        Returns (X, y) where y may be None if no label column is found.
        """
        combined = self._merge_sources(wb_df, fred_df, extra_dfs or [])
        if combined.empty:
            logger.warning("No data available to build feature matrix.")
            return pd.DataFrame(), None

        combined = self._engineer(combined)
        combined = self._clean(combined)

        predictable_cols = self._get_all_feature_names()
        label_col = self._infer_label_col(combined)

        feature_cols = [c for c in predictable_cols if c in combined.columns]
        self._feature_columns = feature_cols

        X = combined[feature_cols].copy()
        y = combined[label_col] if label_col and label_col in combined.columns else None

        logger.info(
            "Feature matrix: %d rows × %d features (label: %s)",
            len(X), len(feature_cols), label_col
        )
        return X, y

    @property
    def feature_names(self) -> list[str]:
        return self._feature_columns

    # ── internal ──────────────────────────────────────────────────────────────

    def _merge_sources(
        self,
        wb_df: Optional[pd.DataFrame],
        fred_df: Optional[pd.DataFrame],
        extra_dfs: list[pd.DataFrame],
    ) -> pd.DataFrame:
        frames = []

        if wb_df is not None and not wb_df.empty:
            frames.append(wb_df)

        if fred_df is not None and not fred_df.empty:
            # FRED data is typically time-indexed — align to year for merging
            fred_df = fred_df.copy()
            date_col = next((c for c in fred_df.columns if "date" in c.lower()), None)
            if date_col:
                fred_df[date_col] = pd.to_datetime(fred_df[date_col])
                fred_df["year"] = fred_df[date_col].dt.year
                fred_df = fred_df.drop(columns=[date_col])
            frames.append(fred_df)

        for edf in extra_dfs:
            if not edf.empty:
                frames.append(edf)

        if not frames:
            return pd.DataFrame()

        # Merge all on year (and optionally country)
        result = frames[0]
        for frame in frames[1:]:
            merge_keys = [k for k in ["year", "country"] if k in result.columns and k in frame.columns]
            if merge_keys:
                result = result.merge(frame, on=merge_keys, how="outer")
            else:
                result = pd.concat([result, frame], axis=1)

        return result

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        all_indicators = self._get_all_feature_names()
        available = [c for c in all_indicators if c in df.columns]

        # Sort by time before computing lags/rolling
        if "year" in df.columns:
            df = df.sort_values("year")

        df = lag_features(df, available, lags=[1, 2, 3])
        df = rolling_features(df, available, windows=[3, 5])
        df = pct_change_features(df, available)

        # Add composite interaction terms for common pairs
        pairs = [
            ("gdp_growth", "unemployment"),
            ("inflation", "interest_rate"),
            ("m2_money_supply", "inflation"),
        ]
        for a, b in pairs:
            df = interaction_terms(df, a, b)

        # One-hot encode country if present
        if "country" in df.columns:
            df = encode_country(df)

        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill then drop rows with too many missing values."""
        df = df.ffill().bfill()
        # Drop columns that are >60% missing
        threshold = 0.6
        df = df.dropna(axis=1, thresh=int(len(df) * (1 - threshold)))
        # Drop rows that still have any NaN in remaining columns
        df = df.dropna()
        return df

    def _get_all_feature_names(self) -> list[str]:
        names = []
        for group_features in self.feature_groups.values():
            names.extend(group_features)
        return list(dict.fromkeys(names))  # deduplicate, preserve order

    def _infer_label_col(self, df: pd.DataFrame) -> Optional[str]:
        """
        The label is the primary economic feature of the target.
        For gdp_growth target, gdp_growth column is the label.
        """
        configured_target = self.target_cfg.get("target_column")
        if configured_target and configured_target in df.columns:
            return configured_target

        target_id = self.target_cfg.get("id", "")
        all_features = self._get_all_feature_names()
        # Try exact match against target id
        if target_id in df.columns:
            return target_id
        # Try first economic feature as label
        economic_features = self.feature_groups.get("economic", [])
        if economic_features and economic_features[0] in df.columns:
            return economic_features[0]
        return None
