"""Assemble scored signals into lane packets for LISA consumption."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from .schema import LanePacket, ScoredSignal


def build_lane_packet(
    lane: str,
    scored_signals: list[ScoredSignal],
    lane_cfg: dict | None = None,
    *,
    summary: str | None = None,
) -> LanePacket:
    lane_cfg = lane_cfg or {}
    ranked = sorted(scored_signals, key=lambda item: item.overall_score, reverse=True)
    priority = ranked[0].overall_score if ranked else 0.0
    evidence_chain = [signal.normalized_signal.signal_id for signal in ranked]
    unresolved = [
        tension
        for signal in ranked
        for tension in signal.unresolved_tensions
    ]

    if summary is None:
        if ranked:
            top = ranked[0].normalized_signal
            summary = (
                f"{lane.title()} packet for LISA with {len(ranked)} scored signal(s); "
                f"top signal is {top.metric} for {top.entity}."
            )
        else:
            summary = f"{lane.title()} packet is empty; no scored signals were available."

    return LanePacket(
        packet_id=f"{lane}-{uuid4().hex[:12]}",
        lane=lane,
        generation_timestamp=datetime.now(timezone.utc),
        priority=priority,
        summary=summary,
        signals=ranked,
        evidence_chain=evidence_chain,
        recommended_downstream_handling=lane_cfg.get(
            "recommended_downstream_handling",
            "queue_for_lisa_review",
        ),
        unresolved_tensions=unresolved,
        retention_decay_hours=lane_cfg.get("retention_decay_hours", 72),
    )