from __future__ import annotations

from fastapi.testclient import TestClient


def test_officialization_without_support_returns_conflict(client: TestClient) -> None:
    concept_response = client.post(
        "/v1/concepts",
        json={
            "name": "Unsupported Concept",
            "shortDefinition": "A concept without support.",
            "createdBy": "reviewer@example.com",
        },
    )
    assert concept_response.status_code == 201
    concept_id = concept_response.json()["id"]

    officialize_response = client.post(
        f"/v1/concepts/{concept_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert officialize_response.status_code == 409


def test_create_and_officialize_supported_concept(
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
    assert concept_response.status_code == 201
    concept = concept_response.json()
    assert concept["status"] == "candidate"

    officialize_response = client.post(
        f"/v1/concepts/{concept['id']}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert officialize_response.status_code == 200
    official = officialize_response.json()
    assert official["status"] == "official"
    assert official["lastReviewedAt"] is not None


def test_officialization_with_unreviewed_evidence_returns_conflict(client: TestClient) -> None:
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
                    "content": "Cornerstone is a shared organizational context layer.",
                    "sourceUrl": "https://example.internal/doc-1",
                }
            ]
        },
    )
    evidence_id = sync_response.json()["evidenceFragments"][0]["id"]
    concept_response = client.post(
        "/v1/concepts",
        json={
            "name": "Cornerstone",
            "shortDefinition": "Shared organizational context layer.",
            "evidenceFragmentIds": [evidence_id],
            "createdBy": "reviewer@example.com",
        },
    )

    officialize_response = client.post(
        f"/v1/concepts/{concept_response.json()['id']}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )

    assert officialize_response.status_code == 409
    assert "must be reviewed" in officialize_response.json()["detail"]


def test_non_production_source_evidence_cannot_officialize(client: TestClient) -> None:
    source_response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Demo Manual", "productionEnabled": False},
    )
    source_id = source_response.json()["id"]
    sync_response = client.post(
        f"/v1/manual-sources/{source_id}/sync",
        json={
            "objects": [
                {
                    "sourceExternalId": "demo-doc-1",
                    "title": "Demo Doc",
                    "content": "Demo Concept is not production context.",
                    "sourceUrl": "https://example.internal/demo-doc-1",
                }
            ]
        },
    )
    evidence_id = sync_response.json()["evidenceFragments"][0]["id"]
    review_response = client.post(
        f"/v1/evidence/{evidence_id}/review",
        json={"trustState": "reviewed", "reviewedBy": "reviewer@example.com"},
    )
    assert review_response.status_code == 200
    concept_response = client.post(
        "/v1/concepts",
        json={
            "name": "Demo Concept",
            "shortDefinition": "Demo-only content.",
            "evidenceFragmentIds": [evidence_id],
            "createdBy": "reviewer@example.com",
        },
    )

    officialize_response = client.post(
        f"/v1/concepts/{concept_response.json()['id']}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )

    assert officialize_response.status_code == 409
    assert "non-production source" in officialize_response.json()["detail"]


def test_unauthorized_reviewer_cannot_officialize(
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

    officialize_response = client.post(
        f"/v1/concepts/{concept_response.json()['id']}/officialize",
        json={"reviewedBy": "random@example.com"},
    )

    assert officialize_response.status_code == 403


def test_create_concept_with_fake_decision_record_id_returns_not_found(client: TestClient) -> None:
    response = client.post(
        "/v1/concepts",
        json={
            "name": "Cornerstone",
            "shortDefinition": "Shared organizational context layer.",
            "decisionRecordIds": ["fake-decision-record"],
            "createdBy": "reviewer@example.com",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "DecisionRecord not found: fake-decision-record"
