"""
src/data/cache.py — SQLite-backed local data cache.
All connectors write through here so network calls are minimised.
"""
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """Persistent SQLite cache with expiry-based invalidation."""

    def __init__(self, db_path: str = "data/cache/store.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key        TEXT PRIMARY KEY,
                    data       TEXT NOT NULL,
                    stored_at  TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _make_key(*parts: str) -> str:
        raw = "|".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, *key_parts: str, max_age_days: int = 7) -> Optional[pd.DataFrame]:
        """Return cached DataFrame or None if missing / expired."""
        key = self._make_key(*key_parts)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data, stored_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        stored_at = datetime.fromisoformat(row[1])
        if datetime.utcnow() - stored_at > timedelta(days=max_age_days):
            logger.debug("Cache expired for key %s", key)
            return None
        try:
            return pd.read_json(row[0], orient="split")
        except Exception:
            return None

    def set(self, *key_parts: str, df: pd.DataFrame) -> None:
        """Store a DataFrame in the cache."""
        key = self._make_key(*key_parts)
        data = df.to_json(orient="split", date_format="iso")
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, data, stored_at) VALUES (?, ?, ?)",
                (key, data, now),
            )

    def invalidate(self, *key_parts: str) -> None:
        key = self._make_key(*key_parts)
        with self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def store_dataframe(self, table: str, df: pd.DataFrame) -> None:
        """Persist a DataFrame as a named table (for user-supplied data)."""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(table, conn, if_exists="replace", index=True)

    def load_table(self, table: str) -> Optional[pd.DataFrame]:
        """Load a previously stored named table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql(f"SELECT * FROM \"{table}\"", conn)
        except Exception:
            return None

    def list_tables(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT IN ('cache')"
            ).fetchall()
        return [r[0] for r in rows]
