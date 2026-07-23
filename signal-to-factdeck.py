#!/usr/bin/env python3
"""Headless SignalFactory runner that writes lane-indexed feeds for FactDeck."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("signal-to-factdeck")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FACTDECK_DIR = Path(
    os.getenv("FACTDECK_FEEDS_DIR", str(PROJECT_ROOT.parent / "FactDeck" / "feeds"))
)

sys.path.insert(0, str(PROJECT_ROOT))

from src.analyzers import AnalyzerRegistry
from src.configuration import load_signal_catalog
from src.data import DataCache, FREDConnector, FileLoader, WorldBankConnector
from src.features import FeaturePipeline
from src.signals import build_signal_packet_from_result

SCAFFOLD_TEMPLATE_FILES = {
    "food_demand_weekly_template.csv",
    "construction_material_cost_direction_template.csv",
    "restaurant_failure_risk_template.csv",
    "construction_project_delay_risk_template.csv",
}


def _parse_year(value: str | int) -> int:
    if isinstance(value, int):
        return value
    raw = str(value).strip().lower()
    if raw == "today":
        return date.today().year
    return int(raw[:4])


def _needs_feature(feature_groups: dict, name: str) -> bool:
    return any(name in group for group in feature_groups.values())


def _load_extra_frames(upload_dir: Path, cache: DataCache, allow_template_data: bool) -> list:
    loader = FileLoader(upload_dir=str(upload_dir), cache=cache)
    files = loader.list_files()
    if not allow_template_data:
        filtered = []
        for file_name in files:
            if Path(file_name).name in SCAFFOLD_TEMPLATE_FILES:
                continue
            filtered.append(file_name)
        files = filtered

    frames = []
    for file_name in files:
        try:
            frames.append(loader.load_file(file_name))
        except Exception as exc:
            logger.warning("Failed to load upload %s: %s", file_name, exc)
    return frames


def _run_one_signal_pack(
    signal_cfg: dict,
    sources_cfg: dict,
    cache: DataCache,
    allow_template_data: bool,
) -> dict | None:
    features = signal_cfg.get("features", {})
    date_range = signal_cfg.get("date_range", {})
    countries = signal_cfg.get("countries", ["US"])
    start_year = _parse_year(date_range.get("start", "2000"))
    end_year = _parse_year(date_range.get("end", "today"))
    horizon = int(signal_cfg.get("forecast_horizon", 12))

    wb_df = None
    wb_cfg = sources_cfg.get("world_bank", {})
    if wb_cfg.get("enabled", False):
        indicators = {
            name: code
            for name, code in wb_cfg.get("indicators", {}).items()
            if _needs_feature(features, name)
        }
        if indicators:
            wb = WorldBankConnector(
                indicators=indicators,
                cache=cache,
                cache_days=int(wb_cfg.get("cache_days", 7)),
            )
            wb_df = wb.fetch(countries=countries, start_year=start_year, end_year=end_year)

    fred_df = None
    fred_cfg = sources_cfg.get("fred", {})
    if fred_cfg.get("enabled", False):
        series = {
            name: sid
            for name, sid in fred_cfg.get("series", {}).items()
            if _needs_feature(features, name)
        }
        if series:
            fred = FREDConnector(
                series=series,
                cache=cache,
                cache_days=int(fred_cfg.get("cache_days", 1)),
            )
            fred_df = fred.fetch(
                start_date=f"{start_year}-01-01",
                end_date=f"{end_year}-12-31",
                frequency="a",
            )

    extra_dfs = []
    csv_cfg = sources_cfg.get("csv", {})
    if csv_cfg.get("enabled", False):
        upload_dir = PROJECT_ROOT / csv_cfg.get("directory", "data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        extra_dfs = _load_extra_frames(upload_dir, cache, allow_template_data)

    pipeline = FeaturePipeline(signal_cfg)
    X, y = pipeline.fit_transform(wb_df=wb_df, fred_df=fred_df, extra_dfs=extra_dfs)
    if X.empty:
        logger.warning("No features produced for signal pack %s", signal_cfg.get("id"))
        return None

    result = AnalyzerRegistry().run(signal_cfg, X, y, horizon=horizon)
    signal_packet = build_signal_packet_from_result(
        signal_cfg,
        result,
        raw_df=wb_df if wb_df is not None and not wb_df.empty else fred_df,
        lane_cfg=None,
        source_name="signalfactory-headless",
    )
    top_score = 0.0
    if signal_packet.get("signals"):
        top_score = float(signal_packet["signals"][0].get("overall_score") or 0.0)

    lane = signal_cfg.get("lane", "unclassified")
    signal_kind = signal_cfg.get("signal_kind", "raw_event")

    # Derive trend from score movement direction stub (overridable from payload).
    trend = "rising" if top_score > 0.65 else "falling" if top_score < 0.35 else "neutral"

    # Impact level: risk/macro lanes carry higher default weight.
    if lane in ("risk",):
        impact_level = "critical" if top_score > 0.8 else "high"
    elif lane in ("macro", "market"):
        impact_level = "high" if top_score > 0.7 else "medium"
    else:
        impact_level = "medium" if top_score > 0.5 else "low"

    # Lane-specific action hints.
    _action_hints = {
        "macro":          "Recalibrate local demand forecasts with updated macro inputs.",
        "business":       "Identify which local business category is affected; surface to operators.",
        "risk":           "Flag affected clients in region. Monitor escalation path.",
        "infrastructure": "Track downstream impact on construction and logistics clients.",
        "community":      "Adjust long-range presence strategy based on demographic/policy shift.",
        "presence":       "Audit listing completeness and review velocity. Claim unclaimed profiles.",
        "market":         "Check sector exposure for regional business portfolio.",
        "supply":         "Alert procurement-dependent clients; evaluate alternative sourcing paths.",
    }

    now = datetime.now(timezone.utc).isoformat()
    return {
        "id":           signal_cfg.get("id"),
        "lane":         lane,
        "signal_kind":  signal_kind,
        "confidence":   round(top_score, 4),
        "score":        round(top_score * 100, 1),
        "impact_level": impact_level,
        "trend":        trend,
        "velocity":     None,
        "receivedAt":   now,
        "source":       "signalfactory",
        "source_class": "model_output",
        "entity":       ",".join(signal_cfg.get("countries", ["global"])),
        "location":     ",".join(signal_cfg.get("countries", ["global"])),
        "action_hint":  _action_hints.get(lane),
        "tags":         [lane, signal_kind],
        "observed_fact": signal_cfg.get("label", signal_cfg.get("id")),
        "signal":       lane,
        "payload":      signal_packet,
    }


def _build_lane_indexed_packet(items: list[dict], failures: list[dict]) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    lanes: dict[str, list] = {}
    for item in items:
        lane = str(item.get("lane") or item.get("signal") or "unclassified")
        lanes.setdefault(lane, []).append(item)

    lane_order = sorted(lanes.keys())
    lane_counts = {lane: len(lanes[lane]) for lane in lane_order}

    # Per-lane summaries.
    lane_summaries: dict[str, dict] = {}
    for lane, lane_items in lanes.items():
        confidences = [float(i.get("confidence") or 0) for i in lane_items]
        scores      = [float(i.get("score") or 0) for i in lane_items]
        impacts     = [i.get("impact_level", "medium") for i in lane_items]
        trends      = [i.get("trend", "neutral") for i in lane_items]
        top_item    = max(lane_items, key=lambda i: float(i.get("confidence") or 0), default={})
        lane_summaries[lane] = {
            "count":           len(lane_items),
            "avg_confidence":  round(sum(confidences) / len(confidences), 4) if confidences else 0,
            "avg_score":       round(sum(scores) / len(scores), 1) if scores else 0,
            "dominant_impact": max(set(impacts), key=impacts.count) if impacts else "medium",
            "dominant_trend":  max(set(trends), key=trends.count) if trends else "neutral",
            "top_signal_id":   top_item.get("id"),
            "top_signal_kind": top_item.get("signal_kind"),
        }

    # Overall feed quality score (0-100).
    all_confidences = [float(i.get("confidence") or 0) for i in items]
    failure_penalty = min(len(failures) * 5, 30)
    quality_score = round(
        (sum(all_confidences) / len(all_confidences) * 100 - failure_penalty)
        if all_confidences else 0,
        1
    )

    return {
        "schema_version":  "2.0",
        "generated_at":    now,
        "source":          "signalfactory-headless",
        "format":          "lane-indexed-v1",
        "count":           len(items),
        "quality_score":   quality_score,
        "lane_counts":     lane_counts,
        "lane_order":      lane_order,
        "lane_summaries":  lane_summaries,
        "lanes":           lanes,
        "items":           items,
        "failures":        failures,
    }


def _parse_beta_regions(raw: str | None) -> list[str]:
    if not raw:
        return []
    values = [part.strip() for part in raw.split(",")]
    return [value for value in values if value]


def _write_packet(feeds_dir: Path, packet: dict) -> tuple[Path, Path]:
    feeds_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    packet_path = feeds_dir / f"signalfactory_{stamp}.json"
    latest_path = feeds_dir / "latest.json"

    packet_with_path = dict(packet)
    packet_with_path["json_path"] = str(packet_path)
    packet_path.write_text(json.dumps(packet_with_path, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(packet_with_path, indent=2), encoding="utf-8")
    return packet_path, latest_path


def run_signal_pipeline(
    signal_pack: str | None,
    feeds_dir: Path,
    allow_template_data: bool,
    beta_regions: list[str] | None = None,
) -> int:
    catalog, sources_cfg = load_signal_catalog(PROJECT_ROOT)
    specs = catalog.get("signal_specs", {})
    if not specs:
        logger.error("No signal specs found in config/signal_specs.yaml")
        return 2

    selected_ids = [signal_pack] if signal_pack else list(specs.keys())
    cache = DataCache(str(PROJECT_ROOT / "data" / "cache" / "store.db"))
    results: list[dict] = []
    failures: list[dict] = []

    for signal_id in selected_ids:
        cfg = specs.get(signal_id)
        if not cfg:
            failures.append({"signal_pack": signal_id, "error": "not found in catalog"})
            continue

        try:
            logger.info("Running signal pack: %s", signal_id)
            item = _run_one_signal_pack(cfg, sources_cfg, cache, allow_template_data)
            if item is None:
                failures.append({"signal_pack": signal_id, "error": "no usable feature matrix"})
                continue
            results.append(item)
        except Exception as exc:
            failures.append({"signal_pack": signal_id, "error": str(exc)})
            logger.exception("Signal pack failed: %s", signal_id)

    packet = _build_lane_indexed_packet(results, failures)
    if beta_regions:
        packet["beta_test"] = {
            "enabled": True,
            "regions": beta_regions,
            "run_label": "gulf-coast-beta",
        }
        for item in packet.get("items", []):
            item["beta_regions"] = beta_regions
    else:
        packet["beta_test"] = {
            "enabled": False,
            "regions": [],
        }

    packet_path, latest_path = _write_packet(feeds_dir, packet)
    logger.info("Wrote lane-indexed packet: %s", packet_path)
    logger.info("Updated latest feed: %s", latest_path)
    print(json.dumps(packet, indent=2))

    if not results:
        logger.error("No signal items were produced. Check data sources and API keys.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SignalFactory headless and publish lane-indexed output to FactDeck feeds."
    )
    parser.add_argument("--signal-pack", type=str, help="Run only one signal pack id")
    parser.add_argument(
        "--factdeck-dir",
        type=Path,
        default=DEFAULT_FACTDECK_DIR,
        help=f"FactDeck feeds directory (default: {DEFAULT_FACTDECK_DIR})",
    )
    parser.add_argument(
        "--allow-template-data",
        action="store_true",
        help="Allow *_template.csv files from data/uploads to be used in batch runs.",
    )
    parser.add_argument(
        "--beta-regions",
        type=str,
        default="",
        help="Comma-separated beta test regions to stamp into output metadata.",
    )
    args = parser.parse_args()
    beta_regions = _parse_beta_regions(args.beta_regions)
    return run_signal_pipeline(args.signal_pack, args.factdeck_dir, args.allow_template_data, beta_regions)


if __name__ == "__main__":
    raise SystemExit(main())
