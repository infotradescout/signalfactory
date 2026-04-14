"""Configuration helpers for SignalFactory's LISA-oriented signal catalog."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def load_signal_catalog(project_root: str | Path = ".") -> tuple[dict, dict]:
    """
    Load the signal catalog and source configuration.

    Supports the new signal-centric config (`signal_specs.yaml`) and the legacy
    target-centric config (`targets.yaml`) so the migration can happen in place.
    """
    root = Path(project_root)
    signal_specs_path = root / "config" / "signal_specs.yaml"
    legacy_targets_path = root / "config" / "targets.yaml"
    sources_path = root / "config" / "sources.yaml"
    lanes_path = root / "config" / "lanes.yaml"

    if signal_specs_path.exists():
        with open(signal_specs_path, encoding="utf-8") as handle:
            raw_catalog = yaml.safe_load(handle) or {}
    elif legacy_targets_path.exists():
        with open(legacy_targets_path, encoding="utf-8") as handle:
            raw_catalog = yaml.safe_load(handle) or {}
    else:
        raise FileNotFoundError("Expected config/signal_specs.yaml or config/targets.yaml")

    with open(sources_path, encoding="utf-8") as handle:
        sources = yaml.safe_load(handle) or {}

    lanes = {}
    if lanes_path.exists():
        with open(lanes_path, encoding="utf-8") as handle:
            lanes = yaml.safe_load(handle) or {}

    catalog = _normalize_catalog(raw_catalog)
    if lanes:
        catalog["lanes"] = lanes.get("lanes", {})
        catalog["default_lane"] = lanes.get("default_lane")
    return catalog, sources


def _normalize_catalog(raw_catalog: dict) -> dict:
    signal_specs = deepcopy(raw_catalog.get("signal_specs") or raw_catalog.get("targets") or {})
    active_signal_pack = raw_catalog.get("active_signal_pack") or raw_catalog.get("active_target")

    normalized_specs: dict[str, dict] = {}
    for key, cfg in signal_specs.items():
        spec = deepcopy(cfg)
        spec["id"] = key
        spec.setdefault("label", key.replace("_", " ").title())
        spec.setdefault("lane", _infer_lane(key, spec))
        spec.setdefault("signal_kind", spec.get("type", "observation"))
        spec.setdefault("consumer", "LISA")
        normalized_specs[key] = spec

    if not active_signal_pack and normalized_specs:
        active_signal_pack = next(iter(normalized_specs))

    return {
        "signal_specs": normalized_specs,
        "active_signal_pack": active_signal_pack,
        # Legacy aliases kept during migration.
        "targets": normalized_specs,
        "active_target": active_signal_pack,
    }


def _infer_lane(key: str, spec: dict) -> str:
    feature_blob = " ".join(
        str(item)
        for group in spec.get("features", {}).values()
        for item in group
    ).lower()
    key_blob = f"{key} {spec.get('label', '')} {feature_blob}".lower()

    if any(word in key_blob for word in ["inflation", "gdp", "macro", "rate", "unemployment"]):
        return "macro"
    if any(word in key_blob for word in ["market", "stock", "equity", "volatility"]):
        return "market"
    if any(word in key_blob for word in ["restaurant", "business", "demand"]):
        return "business"
    if any(word in key_blob for word in ["community", "social", "unrest", "sentiment"]):
        return "community"
    if any(word in key_blob for word in ["construction", "material", "delay", "infrastructure"]):
        return "infrastructure"
    if any(word in key_blob for word in ["risk", "failure"]):
        return "risk"
    return "opportunity"