"""Adapters that map current analyzer outputs into signal packets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd

from ..models.result import PredictionResult
from .packager import build_lane_packet
from .schema import NormalizedSignal
from .scorer import score_signal


def build_signal_packet_from_result(
    signal_cfg: dict,
    result: PredictionResult,
    *,
    raw_df: pd.DataFrame | None = None,
    lane_cfg: dict | None = None,
    source_name: str = "analyzer_registry",
) -> dict[str, Any]:
    """Wrap a current analyzer result in the new LISA lane packet contract."""
    lane = signal_cfg.get("lane", "opportunity")
    signal_id = f"sig-{uuid4().hex[:12]}"
    event_time = _resolve_event_time(raw_df, result)
    latest_value = _latest_signal_value(result)
    metric = signal_cfg.get("signal_metric") or signal_cfg.get("target_column") or signal_cfg.get("id", "signal_value")
    entity = ",".join(signal_cfg.get("countries", ["global"]))
    source_type = "model_output"
    reliability = _reliability_from_result(result)
    novelty_hint = 0.55 if result.forecast is not None else 0.65
    relevance_hint = 0.8

    normalized = NormalizedSignal(
        signal_id=signal_id,
        lane=lane,
        signal_pack=signal_cfg.get("id", "signal_pack"),
        entity=entity,
        metric=metric,
        value=latest_value,
        unit=signal_cfg.get("output_unit", "signal"),
        event_time=event_time,
        publish_time=datetime.now(timezone.utc),
        source_id=f"{signal_cfg.get('id', 'signal_pack')}:{result.model_type}",
        source_type=source_type,
        source_name=source_name,
        raw_event_ref=signal_id,
        source_credibility_hints={
            "historical_reliability": reliability,
            "model_type": result.model_type,
            "metric_snapshot": result.metrics,
        },
        evidence_refs=_top_feature_refs(result),
        extraction_path=["ingest", "normalize", "analyze", "score", "package"],
        metadata={
            "consumer": signal_cfg.get("consumer", "LISA"),
            "signal_kind": signal_cfg.get("signal_kind", signal_cfg.get("type", "observation")),
        },
    )
    scored = score_signal(
        normalized,
        lane_cfg=lane_cfg,
        historical_reliability=reliability,
        source_type_weight=0.75,
        corroboration_count=1 if result.probabilities is not None else 0,
        contradiction_count=0,
        relevance_hint=relevance_hint,
        novelty_hint=novelty_hint,
    )
    packet = build_lane_packet(
        lane,
        [scored],
        lane_cfg=lane_cfg,
        summary=(
            f"SignalFactory packaged {signal_cfg.get('label', signal_cfg.get('id', 'signal'))} "
            f"for LISA as a {lane} lane evidence packet."
        ),
    )
    return packet.to_dict()


def _latest_signal_value(result: PredictionResult):
    if result.forecast is not None and not result.forecast.empty and "forecast" in result.forecast.columns:
        return _safe_scalar(result.forecast.iloc[0]["forecast"])
    return _safe_scalar(result.latest_prediction())


def _resolve_event_time(raw_df: pd.DataFrame | None, result: PredictionResult) -> datetime:
    if raw_df is not None and not raw_df.empty:
        for col in ["date", "timestamp", "logged_at", "year"]:
            if col in raw_df.columns:
                parsed = pd.to_datetime(raw_df[col], errors="coerce")
                if parsed.notna().any():
                    dt = parsed.dropna().iloc[-1].to_pydatetime()
                    if dt.tzinfo is None:
                        return dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc)
    if result.predictions is not None and len(result.predictions.index):
        idx_value = result.predictions.index[-1]
        parsed_index = pd.to_datetime(idx_value, errors="coerce")
        if not pd.isna(parsed_index):
            dt = parsed_index.to_pydatetime()
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _reliability_from_result(result: PredictionResult) -> float:
    if "accuracy" in result.metrics:
        return _safe_float(result.metrics["accuracy"], 0.7)
    if "f1_weighted" in result.metrics:
        return _safe_float(result.metrics["f1_weighted"], 0.68)
    if "R2" in result.metrics:
        return max(0.2, min(0.9, (_safe_float(result.metrics["R2"], 0.0) + 1.0) / 2.0))
    return 0.6


def _top_feature_refs(result: PredictionResult) -> list[str]:
    if result.feature_importance is None or result.feature_importance.empty:
        return []
    return [str(name) for name in result.feature_importance.head(5).index.tolist()]


def _safe_scalar(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default