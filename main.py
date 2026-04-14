"""
main.py — CLI entry point for the SignalFactory LISA ingestion and scoring engine.

Usage examples
--------------
# Use the active signal pack from config/signal_specs.yaml
python main.py

# Specify a signal pack by key
python main.py --signal-pack inflation_forecast

# Override countries and date range
python main.py --signal-pack gdp_growth --countries US,CN,DE --start 2005

# Run a support forecast signal with custom horizon
python main.py --signal-pack inflation_forecast --horizon 24

# Save outputs to disk
python main.py --signal-pack market_trend --save
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

# ── imports ───────────────────────────────────────────────────────────────────
from src.analyzers import AnalyzerRegistry
from src.data import DataCache, FREDConnector, FileLoader, WorldBankConnector
from src.data.health import source_health_snapshot
from src.configuration import load_signal_catalog
from src.features import FeaturePipeline
from src.outputs import ReportBuilder
from src.pipeline import SignalPipelineEngine


# ─── argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SignalFactory — ingest, score, and package signals for LISA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--signal-pack", "--target",
        dest="signal_pack",
        default=None,
        help="Signal pack key from signal_specs.yaml",
    )
    parser.add_argument(
        "--countries", default=None,
        help="Comma-separated ISO-2 country codes, e.g. US,CN,DE"
    )
    parser.add_argument("--start", type=int, default=None, help="Start year (e.g. 2000)")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon (periods)")
    parser.add_argument("--save", action="store_true", help="Save JSON + CSV reports to disk")
    parser.add_argument(
        "--list-signal-packs", "--list-targets",
        dest="list_signal_packs",
        action="store_true",
        help="List all available signal packs",
    )
    return parser.parse_args()


# ─── config loading ───────────────────────────────────────────────────────────

def load_configs():
    return load_signal_catalog(Path(__file__).resolve().parent)


# ─── data fetching ────────────────────────────────────────────────────────────

def fetch_data(signal_cfg: dict, sources_cfg: dict, countries: list[str],
               start_year: int, end_year: int):
    cache = DataCache("data/cache/store.db")
    feature_groups = signal_cfg.get("features", {})

    # Flatten all needed feature names
    needed_features = set(f for g in feature_groups.values() for f in g)

    # ── World Bank ────────────────────────────────────────────────────────
    wb_df = None
    wb_cfg = sources_cfg.get("world_bank", {})
    if wb_cfg.get("enabled", False):
        needed_wb = {
            k: v for k, v in wb_cfg.get("indicators", {}).items()
            if k in needed_features
        }
        if needed_wb:
            wb = WorldBankConnector(
                indicators=needed_wb, cache=cache,
                cache_days=wb_cfg.get("cache_days", 7)
            )
            logger.info("Fetching World Bank data for: %s", list(needed_wb.keys()))
            wb_df = wb.fetch(countries, start_year, end_year)
            logger.info("World Bank: %d rows", len(wb_df) if wb_df is not None else 0)

    # ── FRED ──────────────────────────────────────────────────────────────
    fred_df = None
    fred_cfg = sources_cfg.get("fred", {})
    if fred_cfg.get("enabled", False):
        needed_fred = {
            k: v for k, v in fred_cfg.get("series", {}).items()
            if k in needed_features
        }
        if needed_fred:
            fred = FREDConnector(
                series=needed_fred, cache=cache,
                cache_days=fred_cfg.get("cache_days", 1)
            )
            logger.info("Fetching FRED data for: %s", list(needed_fred.keys()))
            fred_df = fred.fetch(f"{start_year}-01-01", f"{end_year}-12-31", frequency="a")
            if fred_df is not None:
                logger.info("FRED: %d rows", len(fred_df))

    # ── User-uploaded files ───────────────────────────────────────────────
    csv_cfg = sources_cfg.get("csv", {})
    extra_dfs = []
    if csv_cfg.get("enabled", False):
        loader = FileLoader(upload_dir=csv_cfg.get("directory", "data/uploads"), cache=cache)
        for name, df in loader.load_all().items():
            logger.info("Loaded user file '%s': %d rows", name, len(df))
            extra_dfs.append(df)

    return wb_df, fred_df, extra_dfs


# ─── main ─────────────────────────────────────────────────────────────────────

def run():
    args = parse_args()
    catalog_cfg, sources_cfg = load_configs()
    signal_specs = catalog_cfg["signal_specs"]
    lane_cfgs = catalog_cfg.get("lanes", {})

    health = source_health_snapshot(sources_cfg)
    logger.info("Source health snapshot: %s", health["sources"])

    if args.list_signal_packs:
        print("\nAvailable signal packs:")
        for key, cfg in signal_specs.items():
            marker = " ← active" if key == catalog_cfg.get("active_signal_pack") else ""
            lane = cfg.get("lane", "opportunity")
            print(f"  {key:30s}  {cfg['label']} [{lane}] ({cfg['type']}){marker}")
        return

    signal_pack_key = args.signal_pack or catalog_cfg.get("active_signal_pack")
    if signal_pack_key not in signal_specs:
        logger.error("Unknown signal pack '%s'. Use --list-signal-packs to see options.", signal_pack_key)
        sys.exit(1)

    signal_cfg = signal_specs[signal_pack_key]
    signal_cfg["id"] = signal_pack_key
    logger.info(
        "Signal pack: %s [%s] (%s)",
        signal_cfg["label"],
        signal_cfg.get("lane", "opportunity"),
        signal_cfg["type"],
    )

    countries = (
        args.countries.upper().split(",")
        if args.countries else signal_cfg.get("countries", ["US"])
    )
    start_year = args.start or int(
        signal_cfg.get("date_range", {}).get("start", "2000")[:4]
    )
    end_year = date.today().year
    horizon = args.horizon or int(signal_cfg.get("forecast_horizon", 12))

    logger.info(
        "Entity scope: %s | Years: %d–%d | Analyzer horizon: %d",
        countries, start_year, end_year, horizon
    )

    wb_df, fred_df, extra_dfs = fetch_data(signal_cfg, sources_cfg, countries, start_year, end_year)

    pipeline_engine = SignalPipelineEngine(lane_cfg=lane_cfgs.get(signal_cfg.get("lane", "opportunity"), {}))
    pipeline_result = pipeline_engine.run(
        signal_cfg,
        wb_df=wb_df,
        fred_df=fred_df,
        extra_dfs=extra_dfs,
    )
    logger.info(
        "Signal pipeline: %d raw events -> %d normalized signals -> lane %s packet",
        len(pipeline_result["raw_events"]),
        len(pipeline_result["normalized_signals"]),
        pipeline_result["packet"].get("lane", signal_cfg.get("lane", "opportunity")),
    )

    pipe = FeaturePipeline(signal_cfg)
    X, y = pipe.fit_transform(wb_df=wb_df, fred_df=fred_df, extra_dfs=extra_dfs)

    if X.empty:
        logger.error(
            "Signal matrix is empty. No evidence sources produced a usable signal table.\n"
            "• Check your internet connection.\n"
            "• For FRED data, set FRED_API_KEY in .env (free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html)\n"
            "• Drop CSV files in data/uploads/ for offline signal ingestion."
        )
        sys.exit(1)

    logger.info("Signal matrix: %d rows × %d cols", *X.shape)

    registry = AnalyzerRegistry()
    result = registry.run(signal_cfg, X, y, horizon=horizon)

    reporter = ReportBuilder()
    reporter.print_summary(result)

    if args.save:
        json_path = reporter.save_json(result)
        csv_path = reporter.save_csv(result)
        packet_path = Path("data/reports") / f"{signal_cfg['id']}_lane_packet.json"
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        packet_path.write_text(json.dumps(pipeline_result["packet"], indent=2, default=str), encoding="utf-8")
        trace_path = Path("data/reports") / f"{signal_cfg['id']}_pipeline_trace.json"
        trace_path.write_text(json.dumps(pipeline_result, indent=2, default=str), encoding="utf-8")
        logger.info("Saved: %s", json_path)
        logger.info("Saved: %s", csv_path)
        logger.info("Saved: %s", packet_path)
        logger.info("Saved: %s", trace_path)


if __name__ == "__main__":
    run()
