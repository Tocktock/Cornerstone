from __future__ import annotations

from datetime import UTC, datetime

from cornerstone.config import Settings


def utcnow(settings: Settings | None = None) -> datetime:
    if settings and settings.fixed_now:
        return datetime.fromisoformat(settings.fixed_now).astimezone(UTC)
    return datetime.now(UTC)
