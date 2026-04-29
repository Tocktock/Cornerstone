from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from cornerstone.schemas import FreshnessState


@dataclass(frozen=True)
class FreshnessPolicy:
    fresh_days: int = 7
    stale_days: int = 30

    def classify(self, reference_time: datetime | None, now: datetime | None = None) -> FreshnessState:
        if reference_time is None:
            return FreshnessState.UNKNOWN

        comparison_time = now or datetime.now(timezone.utc)
        reference_time = _ensure_aware(reference_time)
        comparison_time = _ensure_aware(comparison_time)

        age_days = (comparison_time - reference_time).total_seconds() / 86_400
        if age_days <= self.fresh_days:
            return FreshnessState.FRESH
        if age_days <= self.stale_days:
            return FreshnessState.AGING
        return FreshnessState.STALE


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
