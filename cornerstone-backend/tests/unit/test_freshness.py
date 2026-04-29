from __future__ import annotations

from datetime import datetime, timedelta, timezone

from cornerstone.schemas import FreshnessState
from cornerstone.services.freshness import FreshnessPolicy


def test_freshness_policy_unknown_without_timestamp() -> None:
    assert FreshnessPolicy().classify(None) == FreshnessState.UNKNOWN


def test_freshness_policy_fresh_within_threshold() -> None:
    now = datetime(2026, 4, 24, tzinfo=timezone.utc)
    assert FreshnessPolicy().classify(now - timedelta(days=2), now=now) == FreshnessState.FRESH


def test_freshness_policy_aging_after_fresh_before_stale() -> None:
    now = datetime(2026, 4, 24, tzinfo=timezone.utc)
    assert FreshnessPolicy().classify(now - timedelta(days=14), now=now) == FreshnessState.AGING


def test_freshness_policy_stale_after_stale_threshold() -> None:
    now = datetime(2026, 4, 24, tzinfo=timezone.utc)
    assert FreshnessPolicy().classify(now - timedelta(days=31), now=now) == FreshnessState.STALE


def test_freshness_policy_future_timestamp_is_fresh() -> None:
    now = datetime(2026, 4, 24, tzinfo=timezone.utc)
    assert FreshnessPolicy().classify(now + timedelta(hours=1), now=now) == FreshnessState.FRESH
