from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from cornerstone.observability import parse_log_message

pytestmark = [pytest.mark.integration, pytest.mark.observability]


def test_source_sync_emits_structured_operational_logs(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="cornerstone")

    source_response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Manual Pilot", "productionEnabled": True},
    )
    source_id = source_response.json()["id"]
    sync_response = client.post(
        f"/v1/manual-sources/{source_id}/sync",
        json={
            "objects": [
                {
                    "sourceExternalId": "doc-1",
                    "title": "Cornerstone Overview",
                    "content": "Cornerstone is a shared organizational context layer. Official context must preserve provenance.",
                    "sourceUrl": "https://example.internal/doc-1",
                }
            ]
        },
    )

    assert sync_response.status_code == 200
    events = _cornerstone_events(caplog)
    event_names = [event["event"] for event in events]
    assert "source.created" in event_names
    assert "source.sync_started" in event_names
    assert "artifact.extracted" in event_names
    assert "source.sync_succeeded" in event_names
    assert "http.request.completed" in event_names

    sync_succeeded = _event(events, "source.sync_succeeded")
    assert sync_succeeded["artifactCount"] == 1
    assert sync_succeeded["evidenceFragmentCount"] == 2
    assert sync_succeeded["freshnessState"] == "fresh"


def test_officialization_logs_blocked_and_success_paths(
    client: TestClient,
    synced_evidence: dict[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="cornerstone")

    unsupported_response = client.post(
        "/v1/concepts",
        json={
            "name": "Unsupported Concept",
            "shortDefinition": "A concept without supporting evidence.",
            "createdBy": "reviewer@example.com",
        },
    )
    unsupported_id = unsupported_response.json()["id"]
    blocked_response = client.post(
        f"/v1/concepts/{unsupported_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert blocked_response.status_code == 409

    supported_response = client.post(
        "/v1/concepts",
        json={
            "name": "Cornerstone",
            "shortDefinition": "Shared organizational context layer.",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    supported_id = supported_response.json()["id"]
    success_response = client.post(
        f"/v1/concepts/{supported_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert success_response.status_code == 200

    events = _cornerstone_events(caplog)
    assert _event(events, "concept.officialize_blocked")["conceptId"] == unsupported_id
    assert _event(events, "concept.officialized")["conceptId"] == supported_id


def test_context_query_logs_trust_label(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="cornerstone")

    response = client.get("/v1/context/query", params={"q": "What is MissingThing?"})

    assert response.status_code == 200
    event = _event(_cornerstone_events(caplog), "context.query_completed")
    assert event["trustLabel"] == "unsupported"
    assert event["freshnessState"] == "unknown"
    assert event["limitationCount"] >= 1


def _cornerstone_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for record in caplog.records:
        if record.name != "cornerstone":
            continue
        try:
            events.append(parse_log_message(record.message))
        except ValueError:
            continue
    return events


def _event(events: list[dict[str, object]], event_name: str) -> dict[str, object]:
    for event in events:
        if event.get("event") == event_name:
            return event
    available = [event.get("event") for event in events]
    raise AssertionError(f"Missing event {event_name!r}. Available events: {available!r}")
