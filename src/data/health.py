"""Source health checks for ingestion connectors."""

from __future__ import annotations

import os
from datetime import datetime, timezone


def source_health_snapshot(sources_cfg: dict) -> dict:
    """Return a lightweight source health snapshot for operations dashboards."""
    world_bank_enabled = bool(sources_cfg.get("world_bank", {}).get("enabled", False))
    fred_enabled = bool(sources_cfg.get("fred", {}).get("enabled", False))
    csv_enabled = bool(sources_cfg.get("csv", {}).get("enabled", False))
    scraper_enabled = bool(sources_cfg.get("web_scraper", {}).get("enabled", False))
    fred_key_present = bool(os.environ.get("FRED_API_KEY", "").strip())

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "world_bank": {
                "enabled": world_bank_enabled,
                "status": "ready" if world_bank_enabled else "disabled",
            },
            "fred": {
                "enabled": fred_enabled,
                "status": "ready" if fred_enabled and fred_key_present else "needs_key" if fred_enabled else "disabled",
                "api_key_present": fred_key_present,
            },
            "csv": {
                "enabled": csv_enabled,
                "status": "ready" if csv_enabled else "disabled",
            },
            "web_scraper": {
                "enabled": scraper_enabled,
                "status": "ready" if scraper_enabled else "disabled",
            },
        },
    }