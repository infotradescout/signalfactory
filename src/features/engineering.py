"""
src/features/engineering.py
Shared transformations used across feature domains.
"""
import numpy as np
import pandas as pd


def lag_features(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    """Add lagged versions of selected columns."""
    out = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = df[col].shift(lag)
    return out


def rolling_features(
    df: pd.DataFrame, columns: list[str], windows: list[int]
) -> pd.DataFrame:
    """Add rolling-mean and rolling-std of selected columns."""
    out = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            out[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
            out[f"{col}_roll_std_{w}"] = df[col].rolling(w).std()
    return out


def pct_change_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add year-on-year / period-on-period % change."""
    out = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        out[f"{col}_pct_chg"] = df[col].pct_change() * 100
    return out


def interaction_terms(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    """Multiply two numeric columns to capture interaction."""
    out = df.copy()
    if col_a in df.columns and col_b in df.columns:
        out[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]
    return out


def normalize_z(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Z-score normalise columns in-place (returns copy)."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        mu = out[col].mean()
        sigma = out[col].std()
        if sigma > 0:
            out[col] = (out[col] - mu) / sigma
    return out


def encode_country(df: pd.DataFrame, country_col: str = "country") -> pd.DataFrame:
    """One-hot encode country column if present."""
    if country_col not in df.columns:
        return df
    dummies = pd.get_dummies(df[country_col], prefix="country")
    return pd.concat([df.drop(columns=[country_col]), dummies], axis=1)
