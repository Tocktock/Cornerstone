from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_audit_events_are_exposed_after_review_and_officialization(
    client: TestClient, synced_evidence: dict[str, str]
) -> None:
    concept_response = client.post(
        "/v1/concepts",
        json={
            "name": "Cornerstone",
            "shortDefinition": "Shared organizational context layer.",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    concept_id = concept_response.json()["id"]

    officialize_response = client.post(
        f"/v1/concepts/{concept_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert officialize_response.status_code == 200

    audit_response = client.get("/v1/audit-events")

    assert audit_response.status_code == 200
    events = audit_response.json()
    event_types = [event["eventType"] for event in events]
    assert "evidence.reviewed" in event_types
    assert "concept.officialized" in event_types

    officialized = next(event for event in events if event["eventType"] == "concept.officialized")
    assert officialized["actor"] == "reviewer@example.com"
    assert officialized["entityType"] == "Concept"
    assert officialized["entityId"] == concept_id


def test_blocked_officialization_creates_audit_event(client: TestClient) -> None:
    concept_response = client.post(
        "/v1/concepts",
        json={
            "name": "Unsupported Concept",
            "shortDefinition": "A concept without support.",
            "createdBy": "reviewer@example.com",
        },
    )
    concept_id = concept_response.json()["id"]

    officialize_response = client.post(
        f"/v1/concepts/{concept_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert officialize_response.status_code == 409

    events = client.get("/v1/audit-events").json()
    blocked = next(event for event in events if event["eventType"] == "concept.officialization_blocked")
    assert blocked["entityId"] == concept_id
    assert "reason" in blocked["metadata"]
