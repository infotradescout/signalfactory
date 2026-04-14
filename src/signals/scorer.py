"""Deterministic baseline scoring for normalized signals."""

from __future__ import annotations

from datetime import datetime, timezone

from .schema import NormalizedSignal, ScoredSignal


def score_signal(
    signal: NormalizedSignal,
    lane_cfg: dict | None = None,
    *,
    historical_reliability: float = 0.65,
    source_type_weight: float = 0.7,
    corroboration_count: int = 0,
    contradiction_count: int = 0,
    relevance_hint: float = 0.75,
    novelty_hint: float = 0.6,
) -> ScoredSignal:
    lane_cfg = lane_cfg or {}
    weights = lane_cfg.get(
        "scoring_weights",
        {
            "truth": 0.28,
            "novelty": 0.17,
            "recency": 0.18,
            "relevance": 0.22,
            "corroboration": 0.10,
            "contradiction": 0.05,
        },
    )
    recency_half_life_hours = lane_cfg.get("recency_half_life_hours", 72)

    truth_score = _clamp((historical_reliability + source_type_weight) / 2.0)
    novelty_score = _clamp(novelty_hint)
    recency_score = _recency_score(signal.event_time, recency_half_life_hours)
    relevance_score = _clamp(relevance_hint)
    corroboration_score = _clamp(min(1.0, corroboration_count / 3.0))
    contradiction_risk = _clamp(min(1.0, contradiction_count / 2.0))

    overall_score = _clamp(
        truth_score * weights.get("truth", 0.28)
        + novelty_score * weights.get("novelty", 0.17)
        + recency_score * weights.get("recency", 0.18)
        + relevance_score * weights.get("relevance", 0.22)
        + corroboration_score * weights.get("corroboration", 0.10)
        - contradiction_risk * weights.get("contradiction", 0.05)
    )

    reasons = [
        f"truth={truth_score:.2f} from source reliability and source type hints",
        f"novelty={novelty_score:.2f} from current signal delta placeholder",
        f"recency={recency_score:.2f} using {recency_half_life_hours}h lane decay",
        f"relevance={relevance_score:.2f} for lane {signal.lane}",
    ]
    if corroboration_count:
        reasons.append(f"corroborated by {corroboration_count} independent signal(s)")
    if contradiction_count:
        reasons.append(f"contradiction risk raised by {contradiction_count} conflicting signal(s)")

    return ScoredSignal(
        normalized_signal=signal,
        truth_score=truth_score,
        novelty_score=novelty_score,
        recency_score=recency_score,
        relevance_score=relevance_score,
        corroboration_score=corroboration_score,
        contradiction_risk=contradiction_risk,
        overall_score=overall_score,
        confidence_reasons=reasons,
        corroborating_signals=[],
        contradictory_signals=[],
        unresolved_tensions=["No contradiction graph linked yet"] if contradiction_count else [],
    )


def _recency_score(event_time: datetime, half_life_hours: int) -> float:
    now = datetime.now(timezone.utc)
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    age_hours = max(0.0, (now - event_time).total_seconds() / 3600.0)
    if half_life_hours <= 0:
        return 1.0
    decay_steps = age_hours / half_life_hours
    return _clamp(0.5 ** decay_steps)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))