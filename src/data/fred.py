"""
src/data/fred.py — Federal Reserve Economic Data (FRED) connector.
Requires FRED_API_KEY in environment / .env file.
Free key: https://fred.stlouisfed.org/docs/api/api_key.html
"""
import logging
import os
from datetime import date
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from .cache import DataCache

logger = logging.getLogger(__name__)
load_dotenv()

try:
    from fredapi import Fred
    _FRED_AVAILABLE = True
except ImportError:
    _FRED_AVAILABLE = False
    logger.warning("fredapi not installed — FRED connector disabled")


class FREDConnector:
    """Fetch time-series economic data from the St. Louis Fed (FRED)."""

    def __init__(self, series: dict[str, str], cache: Optional[DataCache] = None,
                 cache_days: int = 1):
        """
        Parameters
        ----------
        series     : mapping of feature_name -> FRED series ID
                     e.g. {"interest_rate": "FEDFUNDS"}
        cache      : DataCache instance (optional)
        cache_days : how long to trust cached data (default 1 — daily data)
        """
        self.series_map = series
        self.cache = cache
        self.cache_days = cache_days
        self._fred: Optional[object] = None

    # ── public API ────────────────────────────────────────────────────────────

    def fetch(self, start_date: str, end_date: str, frequency: str = "a") -> pd.DataFrame:
        """
        Return a wide DataFrame indexed by date with one column per series.

        Parameters
        ----------
        frequency : 'a' annual | 'q' quarterly | 'm' monthly | 'w' weekly | 'd' daily
        """
        if not _FRED_AVAILABLE:
            raise RuntimeError("fredapi not installed. Run: pip install fredapi")

        api_key = os.environ.get("FRED_API_KEY", "")
        if not api_key or api_key == "your_fred_api_key_here":
            logger.warning(
                "FRED_API_KEY not set — skipping FRED connector. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            return pd.DataFrame()

        if self._fred is None:
            self._fred = Fred(api_key=api_key)

        frames = {}
        for feature_name, series_id in self.series_map.items():
            s = self._fetch_series(feature_name, series_id, start_date, end_date, frequency)
            if s is not None:
                frames[feature_name] = s

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index.name = "date"
        df = df.reset_index()
        return df

    # ── internal ──────────────────────────────────────────────────────────────

    def _fetch_series(
        self, feature_name: str, series_id: str,
        start_date: str, end_date: str, frequency: str
    ) -> Optional[pd.Series]:
        cache_key = ("fred", series_id, start_date, end_date, frequency)

        if self.cache:
            cached = self.cache.get(*cache_key, max_age_days=self.cache_days)
            if cached is not None:
                logger.debug("Cache hit: FRED %s", series_id)
                return cached.set_index("date").iloc[:, 0]

        try:
            logger.info("Fetching FRED: %s (%s)", feature_name, series_id)
            s = self._fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
                frequency=frequency,
            )
            s.name = feature_name
            s.dropna(inplace=True)

            if self.cache:
                self.cache.set(*cache_key, df=s.reset_index().rename(columns={"index": "date"}))

            return s

        except Exception as exc:
            logger.warning("FRED fetch failed for %s: %s", series_id, exc)
            return None
