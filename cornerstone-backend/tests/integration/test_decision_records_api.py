from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_decision_record_creation_requires_reviewed_evidence(client: TestClient) -> None:
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
                    "title": "Decision Source",
                    "content": "We decided to use FastAPI for the backend.",
                    "sourceUrl": "https://example.internal/doc-1",
                }
            ]
        },
    )
    evidence_id = sync_response.json()["evidenceFragments"][0]["id"]

    response = client.post(
        "/v1/decision-records",
        json={
            "title": "Use FastAPI",
            "decision": "Use FastAPI for backend APIs.",
            "reason": "Python ecosystem and async API ergonomics.",
            "decidedBy": "reviewer@example.com",
            "evidenceFragmentIds": [evidence_id],
        },
    )

    assert response.status_code == 409
    assert "must be reviewed" in response.json()["detail"]


def test_decision_record_can_support_concept_officialization(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    decision_response = client.post(
        "/v1/decision-records",
        json={
            "title": "Define Cornerstone",
            "decision": "Cornerstone is the shared organizational context layer.",
            "reason": "This definition appears in the reviewed source evidence.",
            "decidedBy": "reviewer@example.com",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
        },
    )
    assert decision_response.status_code == 201
    decision_id = decision_response.json()["id"]

    concept_response = client.post(
        "/v1/concepts",
        json={
            "name": "Cornerstone",
            "shortDefinition": "Shared organizational context layer.",
            "decisionRecordIds": [decision_id],
            "createdBy": "reviewer@example.com",
        },
    )
    assert concept_response.status_code == 201

    officialize_response = client.post(
        f"/v1/concepts/{concept_response.json()['id']}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )

    assert officialize_response.status_code == 200
    assert officialize_response.json()["status"] == "official"


def test_decision_record_creation_rejects_unauthorized_decider(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    response = client.post(
        "/v1/decision-records",
        json={
            "title": "Define Cornerstone",
            "decision": "Cornerstone is the shared organizational context layer.",
            "reason": "This definition appears in the reviewed source evidence.",
            "decidedBy": "random@example.com",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
        },
    )

    assert response.status_code == 403


def test_decision_record_list_and_read(client: TestClient, synced_evidence: dict[str, str]) -> None:
    response = client.post(
        "/v1/decision-records",
        json={
            "title": "Listable Decision",
            "decision": "Cornerstone keeps decisions attached to reviewed evidence.",
            "reason": "The decision is part of the official context audit trail.",
            "decidedBy": "reviewer@example.com",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
        },
    )
    assert response.status_code == 201
    decision_id = response.json()["id"]

    list_response = client.get("/v1/decision-records")
    read_response = client.get(f"/v1/decision-records/{decision_id}")

    assert list_response.status_code == 200
    assert any(item["id"] == decision_id for item in list_response.json())
    assert read_response.status_code == 200
    assert read_response.json()["title"] == "Listable Decision"


def test_decision_record_read_missing_returns_404(client: TestClient) -> None:
    response = client.get("/v1/decision-records/missing-decision")

    assert response.status_code == 404


def test_decision_record_creation_rejects_missing_affected_concept(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    response = client.post(
        "/v1/decision-records",
        json={
            "title": "Missing affected concept",
            "decision": "This decision references a concept that does not exist.",
            "reason": "Regression coverage for decision record validation.",
            "decidedBy": "reviewer@example.com",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "affectedConceptIds": ["missing-concept"],
        },
    )

    assert response.status_code == 404
    assert "Concept not found" in response.json()["detail"]
