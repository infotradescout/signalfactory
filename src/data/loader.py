"""
src/data/loader.py — Load user-supplied CSV, Excel, and SQLite data files.
Drop files into data/uploads/ and they are automatically discoverable.
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .cache import DataCache

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}


class FileLoader:
    """
    Scan a directory for CSV / Excel / Parquet files and expose them as
    DataFrames. Also reads named tables from the SQLite cache store.
    """

    def __init__(self, upload_dir: str = "data/uploads", cache: Optional[DataCache] = None):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.cache = cache

    # ── public API ────────────────────────────────────────────────────────────

    def list_files(self) -> list[str]:
        """Return relative paths of all loadable files in the upload directory."""
        return [
            str(p.relative_to(self.upload_dir))
            for p in self.upload_dir.rglob("*")
            if p.suffix.lower() in _SUPPORTED_EXTENSIONS
        ]

    def load_file(self, filename: str) -> pd.DataFrame:
        """Load a file from the upload directory into a DataFrame."""
        path = self.upload_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".tsv":
            df = pd.read_csv(path, sep="\t")
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        logger.info("Loaded %s: %d rows × %d cols", filename, len(df), len(df.columns))
        return df

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load every file in the upload directory; keys are filenames (no extension)."""
        result: dict[str, pd.DataFrame] = {}
        for fname in self.list_files():
            stem = Path(fname).stem
            try:
                result[stem] = self.load_file(fname)
            except Exception as exc:
                logger.warning("Could not load %s: %s", fname, exc)
        return result

    def save_to_cache(self, name: str, df: pd.DataFrame) -> None:
        """Persist a DataFrame to the SQLite cache as a named table."""
        if self.cache:
            self.cache.store_dataframe(name, df)
            logger.info("Saved '%s' (%d rows) to cache", name, len(df))

    def load_from_cache(self, name: str) -> Optional[pd.DataFrame]:
        """Load a previously persisted named table from SQLite cache."""
        if self.cache:
            return self.cache.load_table(name)
        return None

    def infer_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Heuristic: find the most likely date/time column."""
        date_hints = {"date", "year", "time", "period", "month", "quarter"}
        for col in df.columns:
            if col.lower() in date_hints:
                return col
            try:
                pd.to_datetime(df[col])
                return col
            except Exception:
                pass
        return None
