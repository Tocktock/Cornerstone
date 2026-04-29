from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from cornerstone.observability import log_event, parse_log_message
from cornerstone.schemas import FreshnessState

pytestmark = [pytest.mark.unit, pytest.mark.observability]


def test_log_event_emits_parseable_structured_json(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="cornerstone")

    log_event(
        "test.event",
        freshnessState=FreshnessState.FRESH,
        occurredAt=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
        count=2,
    )

    events = [_parse(record.message) for record in caplog.records if record.name == "cornerstone"]
    assert events == [
        {
            "event": "test.event",
            "schemaVersion": 1,
            "freshnessState": "fresh",
            "occurredAt": "2026-04-25T00:00:00+00:00",
            "count": 2,
        }
    ]


def _parse(message: str) -> dict[str, object]:
    return parse_log_message(message)
