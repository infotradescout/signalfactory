"""Explicit signal pipeline orchestration for LISA packet preparation."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from ..signals.packager import build_lane_packet
from ..signals.schema import NormalizedSignal, RawEvent
from ..signals.scorer import score_signal


class SignalPipelineEngine:
    """Builds lane packets from raw source tables using explicit pipeline stages."""

    def __init__(self, lane_cfg: dict | None = None):
        self.lane_cfg = lane_cfg or {}

    def run(
        self,
        signal_cfg: dict,
        *,
        wb_df: pd.DataFrame | None = None,
        fred_df: pd.DataFrame | None = None,
        extra_dfs: list[pd.DataFrame] | None = None,
    ) -> dict:
        raw_events = self._extract_raw_events(wb_df=wb_df, fred_df=fred_df, extra_dfs=extra_dfs or [])
        normalized = self._normalize_events(raw_events, signal_cfg)
        scored = self._score_signals(normalized)
        lane = signal_cfg.get("lane", "opportunity")
        packet = build_lane_packet(lane, scored, self.lane_cfg)
        return {
            "raw_events": [event.to_dict() for event in raw_events],
            "normalized_signals": [signal.to_dict() for signal in normalized],
            "packet": packet.to_dict(),
        }

    def _extract_raw_events(
        self,
        *,
        wb_df: pd.DataFrame | None,
        fred_df: pd.DataFrame | None,
        extra_dfs: list[pd.DataFrame],
    ) -> list[RawEvent]:
        events: list[RawEvent] = []
        events.extend(self._events_from_df("world_bank", wb_df))
        events.extend(self._events_from_df("fred", fred_df))
        for idx, frame in enumerate(extra_dfs, start=1):
            events.extend(self._events_from_df(f"upload_{idx}", frame))
        return events

    def _events_from_df(self, source_name: str, frame: pd.DataFrame | None) -> list[RawEvent]:
        if frame is None or frame.empty:
            return []

        timestamp_col = self._guess_timestamp_col(frame)
        events: list[RawEvent] = []
        sample = frame.tail(min(50, len(frame)))
        for idx, row in sample.reset_index(drop=True).iterrows():
            event_time = self._event_time_from_row(row, timestamp_col)
            payload = {
                key: value
                for key, value in row.to_dict().items()
                if not self._is_nan(value)
            }
            events.append(
                RawEvent(
                    event_id=f"{source_name}-{uuid4().hex[:12]}-{idx}",
                    source_name=source_name,
                    timestamp=event_time,
                    raw_payload=payload,
                    source_metadata={
                        "rows_total": int(len(frame)),
                        "columns": list(frame.columns),
                    },
                )
            )
        return events

    def _normalize_events(self, events: list[RawEvent], signal_cfg: dict) -> list[NormalizedSignal]:
        lane = signal_cfg.get("lane", "opportunity")
        signal_pack = signal_cfg.get("id", "signal_pack")
        output_unit = signal_cfg.get("output_unit", "signal")
        target_col = signal_cfg.get("target_column")
        countries = signal_cfg.get("countries", ["global"])
        entity_default = ",".join(countries)

        normalized: list[NormalizedSignal] = []
        for event in events:
            signal_id = f"ns-{uuid4().hex[:14]}"
            metric, value = self._pick_metric_value(event.raw_payload, target_col)
            entity = str(event.raw_payload.get("country") or event.raw_payload.get("entity") or entity_default)
            normalized.append(
                NormalizedSignal(
                    signal_id=signal_id,
                    lane=lane,
                    signal_pack=signal_pack,
                    entity=entity,
                    metric=metric,
                    value=value,
                    unit=output_unit,
                    event_time=event.timestamp,
                    publish_time=datetime.now(timezone.utc),
                    source_id=event.event_id,
                    source_type="structured_data" if event.source_name in {"world_bank", "fred"} else "uploaded_data",
                    source_name=event.source_name,
                    raw_event_ref=event.event_id,
                    source_credibility_hints={
                        "source_name": event.source_name,
                        "structured": event.source_name in {"world_bank", "fred"},
                    },
                    evidence_refs=[event.event_id],
                    extraction_path=["extract", "normalize"],
                    metadata={"signal_kind": signal_cfg.get("signal_kind", signal_cfg.get("type", "observation"))},
                )
            )
        return normalized

    def _score_signals(self, signals: list[NormalizedSignal]):
        scored = []
        for signal in signals:
            source_type_weight = 0.82 if signal.source_type == "structured_data" else 0.65
            scored.append(
                score_signal(
                    signal,
                    lane_cfg=self.lane_cfg,
                    historical_reliability=0.7,
                    source_type_weight=source_type_weight,
                    corroboration_count=0,
                    contradiction_count=0,
                    relevance_hint=0.75,
                    novelty_hint=0.55,
                )
            )
        return scored

    @staticmethod
    def _guess_timestamp_col(frame: pd.DataFrame) -> str | None:
        for candidate in ["date", "timestamp", "logged_at", "year"]:
            if candidate in frame.columns:
                return candidate
        return None

    @staticmethod
    def _event_time_from_row(row: pd.Series, timestamp_col: str | None) -> datetime:
        if not timestamp_col:
            return datetime.now(timezone.utc)

        value = row.get(timestamp_col)
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return datetime.now(timezone.utc)
        if timestamp_col == "year":
            parsed = pd.Timestamp(year=int(parsed.year), month=1, day=1)
        dt = parsed.to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _pick_metric_value(payload: dict, preferred_metric: str | None) -> tuple[str, object]:
        if preferred_metric and preferred_metric in payload and not SignalPipelineEngine._is_nan(payload.get(preferred_metric)):
            return preferred_metric, payload.get(preferred_metric)

        for key, value in payload.items():
            if key.lower() in {"country", "year", "date", "timestamp"}:
                continue
            if isinstance(value, (int, float)) and not SignalPipelineEngine._is_nan(value):
                return key, value

        for key, value in payload.items():
            if key.lower() in {"country", "year", "date", "timestamp"}:
                continue
            if not SignalPipelineEngine._is_nan(value):
                return key, value

        return "signal_value", None

    @staticmethod
    def _is_nan(value) -> bool:
        try:
            return pd.isna(value)
        except Exception:
            return False