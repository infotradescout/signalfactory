"""Canonical signal objects for ingestion, scoring, and LISA packaging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class RawEvent:
    event_id: str
    source_name: str
    timestamp: datetime
    raw_payload: dict[str, Any]
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(slots=True)
class NormalizedSignal:
    signal_id: str
    lane: str
    signal_pack: str
    entity: str
    metric: str
    value: Any
    unit: str
    event_time: datetime
    publish_time: datetime
    source_id: str
    source_type: str
    source_name: str
    raw_event_ref: str
    source_credibility_hints: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    extraction_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_time"] = self.event_time.isoformat()
        data["publish_time"] = self.publish_time.isoformat()
        return data


@dataclass(slots=True)
class ScoredSignal:
    normalized_signal: NormalizedSignal
    truth_score: float
    novelty_score: float
    recency_score: float
    relevance_score: float
    corroboration_score: float
    contradiction_risk: float
    overall_score: float
    confidence_reasons: list[str] = field(default_factory=list)
    corroborating_signals: list[str] = field(default_factory=list)
    contradictory_signals: list[str] = field(default_factory=list)
    unresolved_tensions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.normalized_signal.to_dict(),
            "truth_score": self.truth_score,
            "novelty_score": self.novelty_score,
            "recency_score": self.recency_score,
            "relevance_score": self.relevance_score,
            "corroboration_score": self.corroboration_score,
            "contradiction_risk": self.contradiction_risk,
            "overall_score": self.overall_score,
            "confidence_reasons": self.confidence_reasons,
            "corroborating_signals": self.corroborating_signals,
            "contradictory_signals": self.contradictory_signals,
            "unresolved_tensions": self.unresolved_tensions,
        }


@dataclass(slots=True)
class LanePacket:
    packet_id: str
    lane: str
    generation_timestamp: datetime
    priority: float
    summary: str
    signals: list[ScoredSignal]
    evidence_chain: list[str] = field(default_factory=list)
    recommended_downstream_handling: str = "queue_for_lisa_review"
    unresolved_tensions: list[str] = field(default_factory=list)
    retention_decay_hours: int = 72

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "lane": self.lane,
            "generation_timestamp": self.generation_timestamp.isoformat(),
            "priority": self.priority,
            "summary": self.summary,
            "signals": [signal.to_dict() for signal in self.signals],
            "evidence_chain": self.evidence_chain,
            "recommended_downstream_handling": self.recommended_downstream_handling,
            "unresolved_tensions": self.unresolved_tensions,
            "retention_decay_hours": self.retention_decay_hours,
        }