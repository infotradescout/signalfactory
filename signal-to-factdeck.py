#!/usr/bin/env python3
"""
signal-to-factdeck.py — Bridge SignalFactory output to FactDeck feeds.

Runs the signal generation pipeline and writes output to FactDeck's feed directory.
Designed to run periodically (e.g., via Task Scheduler or cron).

Usage:
    python signal-to-factdeck.py
    python signal-to-factdeck.py --signal-pack inflation_forecast --save

Environment:
    FACTDECK_FEEDS_DIR - Path to FactDeck feeds directory (default: ../FactDeck/feeds)
    SIGNALFACTORY_SAVE - Write outputs to disk (default: true)
"""

import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
import os
import argparse

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("signal-to-factdeck")

# Default paths
FACTDECK_FEEDS_DIR = os.getenv("FACTDECK_FEEDS_DIR", str(Path(__file__).parent.parent / "FactDeck" / "feeds"))

# Import SignalFactory modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from src.analyzers import AnalyzerRegistry
    from src.configuration import load_signal_catalog
    from src.data import DataCache, FREDConnector, FileLoader, WorldBankConnector
    from src.features import FeaturePipeline
    from src.outputs import ReportBuilder
    from src.pipeline import SignalPipelineEngine
except ImportError as e:
    logger.error(f"Failed to import SignalFactory modules: {e}")
    logger.error("Make sure you have run: pip install -r requirements.txt")
    sys.exit(1)


def run_signal_pipeline(signal_pack: str = None, save: bool = True):
    """Run the signal pipeline and return results."""
    try:
        catalog = load_signal_catalog()
        if signal_pack:
            if signal_pack not in catalog:
                logger.error(f"Signal pack '{signal_pack}' not found in catalog")
                return None
            specs = [catalog[signal_pack]]
        else:
            specs = list(catalog.values())

        logger.info(f"Running {len(specs)} signal pack(s)")
        results = []
        
        for spec in specs:
            logger.info(f"Processing signal pack: {spec.name}")
            engine = SignalPipelineEngine(spec)
            result = engine.run()
            results.append(result)
        
        logger.info(f"Generated {len(results)} result(s)")
        return results
    except Exception as e:
        logger.error(f"Signal pipeline failed: {e}", exc_info=True)
        return None


def format_feed_packet(results):
    """Format signal results into FactDeck feed packet format."""
    items = []
    now = datetime.now(timezone.utc).isoformat()
    
    for result in (results or []):
        try:
            # Extract signal data from result
            signal_data = {
                "signal_name": getattr(result, 'name', 'unknown'),
                "confidence": getattr(result, 'confidence', 0.0),
                "generated_at": now,
                "source": "signalfactory",
                "payload": result.__dict__ if hasattr(result, '__dict__') else str(result),
            }
            items.append(signal_data)
        except Exception as e:
            logger.warning(f"Failed to format result: {e}")
            continue
    
    packet = {
        "generated_at": now,
        "source": "signalfactory-bridge",
        "items": items,
        "count": len(items),
    }
    return packet


def write_to_factdeck(packet: dict):
    """Write signal packet to FactDeck feeds directory."""
    try:
        feeds_dir = Path(FACTDECK_FEEDS_DIR)
        feeds_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename based on timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_filename = f"signal_{timestamp}.json"
        json_path = feeds_dir / json_filename
        
        # Write packet
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(packet, f, indent=2)
        logger.info(f"Wrote packet to {json_path}")
        
        # Update latest.json pointer
        latest_pointer = {
            "json_path": str(json_path),
            "generated_at": packet["generated_at"],
            "items_count": len(packet["items"]),
        }
        latest_path = feeds_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(latest_pointer, f, indent=2)
        logger.info(f"Updated {latest_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to write to FactDeck: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Bridge SignalFactory output to FactDeck feeds")
    parser.add_argument("--signal-pack", type=str, help="Specific signal pack to run")
    parser.add_argument("--factdeck-dir", type=str, default=FACTDECK_FEEDS_DIR, 
                       help=f"FactDeck feeds directory (default: {FACTDECK_FEEDS_DIR})")
    args = parser.parse_args()
    
    global FACTDECK_FEEDS_DIR
    FACTDECK_FEEDS_DIR = args.factdeck_dir
    
    logger.info("Starting SignalFactory → FactDeck bridge")
    logger.info(f"FactDeck feeds directory: {FACTDECK_FEEDS_DIR}")
    
    # Run signal pipeline
    results = run_signal_pipeline(signal_pack=args.signal_pack, save=True)
    if not results:
        logger.error("Signal pipeline produced no results")
        sys.exit(1)
    
    # Format and write to FactDeck
    packet = format_feed_packet(results)
    if not write_to_factdeck(packet):
        logger.error("Failed to write to FactDeck")
        sys.exit(1)
    
    logger.info("Bridge completed successfully")
    print(json.dumps(packet, indent=2))


if __name__ == "__main__":
    main()
