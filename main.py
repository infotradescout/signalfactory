"""
main.py — CLI entry point for the Predictive Analytics System.

Usage examples
--------------
# Use the active target from config/targets.yaml
python main.py

# Specify a target by key
python main.py --target inflation_forecast

# Override countries and date range
python main.py --target gdp_growth --countries US,CN,DE --start 2005

# Run in forecast mode with custom horizon
python main.py --target inflation_forecast --horizon 24

# Save outputs to disk
python main.py --target market_trend --save
"""

import argparse
import logging
import sys
import warnings
from datetime import date
from pathlib import Path

import yaml
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
from src.data import DataCache, FREDConnector, FileLoader, WorldBankConnector
from src.features import FeaturePipeline
from src.models import ModelRegistry
from src.outputs import ReportBuilder


# ─── argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predictive Analytics System — run from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--target", default=None, help="Target key from targets.yaml")
    parser.add_argument(
        "--countries", default=None,
        help="Comma-separated ISO-2 country codes, e.g. US,CN,DE"
    )
    parser.add_argument("--start", type=int, default=None, help="Start year (e.g. 2000)")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon (periods)")
    parser.add_argument("--save", action="store_true", help="Save JSON + CSV reports to disk")
    parser.add_argument("--list-targets", action="store_true", help="List all available targets")
    return parser.parse_args()


# ─── config loading ───────────────────────────────────────────────────────────

def load_configs():
    with open("config/targets.yaml") as f:
        targets_cfg = yaml.safe_load(f)
    with open("config/sources.yaml") as f:
        sources_cfg = yaml.safe_load(f)
    return targets_cfg, sources_cfg


# ─── data fetching ────────────────────────────────────────────────────────────

def fetch_data(target_cfg: dict, sources_cfg: dict, countries: list[str],
               start_year: int, end_year: int):
    cache = DataCache("data/cache/store.db")
    feature_groups = target_cfg.get("features", {})

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
    targets_cfg, sources_cfg = load_configs()

    if args.list_targets:
        print("\nAvailable prediction targets:")
        for key, cfg in targets_cfg["targets"].items():
            marker = " ← active" if key == targets_cfg.get("active_target") else ""
            print(f"  {key:30s}  {cfg['label']} ({cfg['type']}){marker}")
        return

    # Resolve target
    target_key = args.target or targets_cfg.get("active_target")
    if target_key not in targets_cfg["targets"]:
        logger.error("Unknown target '%s'. Use --list-targets to see options.", target_key)
        sys.exit(1)

    target_cfg = targets_cfg["targets"][target_key]
    target_cfg["id"] = target_key
    logger.info("Target: %s (%s)", target_cfg["label"], target_cfg["type"])

    # Resolve parameters
    countries = (
        args.countries.upper().split(",")
        if args.countries else target_cfg.get("countries", ["US"])
    )
    start_year = args.start or int(
        target_cfg.get("date_range", {}).get("start", "2000")[:4]
    )
    end_year = date.today().year
    horizon = args.horizon or int(target_cfg.get("forecast_horizon", 12))

    logger.info(
        "Countries: %s | Years: %d–%d | Horizon: %d",
        countries, start_year, end_year, horizon
    )

    # Fetch data
    wb_df, fred_df, extra_dfs = fetch_data(target_cfg, sources_cfg, countries, start_year, end_year)

    # Build features
    pipe = FeaturePipeline(target_cfg)
    X, y = pipe.fit_transform(wb_df=wb_df, fred_df=fred_df, extra_dfs=extra_dfs)

    if X.empty:
        logger.error(
            "Feature matrix is empty. No data was retrieved.\n"
            "• Check your internet connection.\n"
            "• For FRED data, set FRED_API_KEY in .env (free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html)\n"
            "• Drop CSV files in data/uploads/ for offline use."
        )
        sys.exit(1)

    logger.info("Feature matrix: %d rows × %d cols", *X.shape)

    # Run model
    registry = ModelRegistry()
    result = registry.run(target_cfg, X, y, horizon=horizon)

    # Print summary
    reporter = ReportBuilder()
    reporter.print_summary(result)

    # Save outputs
    if args.save:
        json_path = reporter.save_json(result)
        csv_path = reporter.save_csv(result)
        logger.info("Saved: %s", json_path)
        logger.info("Saved: %s", csv_path)


if __name__ == "__main__":
    run()
