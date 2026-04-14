"""
src/data/world_bank.py — World Bank Open Data connector.
No API key required. Uses wbgapi for clean access.
"""
import logging
from typing import Optional

import pandas as pd

from .cache import DataCache

logger = logging.getLogger(__name__)

try:
    import wbgapi as wb
    _WB_AVAILABLE = True
except ImportError:
    _WB_AVAILABLE = False
    logger.warning("wbgapi not installed — World Bank connector disabled")


class WorldBankConnector:
    """Fetch annual indicator data for one or more countries from the World Bank."""

    def __init__(self, indicators: dict[str, str], cache: Optional[DataCache] = None,
                 cache_days: int = 7):
        """
        Parameters
        ----------
        indicators : mapping of feature_name -> WB indicator code
                     e.g. {"gdp_growth": "NY.GDP.MKTP.KD.ZG"}
        cache      : DataCache instance (optional)
        cache_days : how long to trust cached data
        """
        self.indicators = indicators
        self.cache = cache
        self.cache_days = cache_days

    # ── public API ────────────────────────────────────────────────────────────

    def fetch(
        self,
        countries: list[str],
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """
        Return a tidy DataFrame with columns:
        [country, year, <indicator_name>, ...]
        """
        if not _WB_AVAILABLE:
            raise RuntimeError("wbgapi is not installed. Run: pip install wbgapi")

        frames = []
        for feature_name, wb_code in self.indicators.items():
            df = self._fetch_indicator(feature_name, wb_code, countries, start_year, end_year)
            if df is not None:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        result = frames[0]
        for df in frames[1:]:
            result = result.merge(df, on=["country", "year"], how="outer")

        result = result.sort_values(["country", "year"]).reset_index(drop=True)
        return result

    # ── internal ──────────────────────────────────────────────────────────────

    def _fetch_indicator(
        self, feature_name: str, wb_code: str,
        countries: list[str], start_year: int, end_year: int
    ) -> Optional[pd.DataFrame]:
        cache_key = ("wb", wb_code, ",".join(sorted(countries)), str(start_year), str(end_year))

        if self.cache:
            cached = self.cache.get(*cache_key, max_age_days=self.cache_days)
            if cached is not None:
                logger.debug("Cache hit: WB %s", wb_code)
                return cached

        try:
            logger.info("Fetching World Bank: %s (%s)", feature_name, wb_code)
            raw = wb.data.DataFrame(
                wb_code,
                economy=countries,
                time=range(start_year, end_year + 1),
                numericTimeKeys=True,
            )
            # raw shape: (countries) x (years) — reshape to tidy
            raw.index.name = "country"
            tidy = raw.stack().reset_index()
            tidy.columns = ["country", "year", feature_name]
            tidy["year"] = tidy["year"].astype(int)

            if self.cache:
                self.cache.set(*cache_key, df=tidy)
            return tidy

        except Exception as exc:
            logger.warning("World Bank fetch failed for %s: %s", wb_code, exc)
            return None
