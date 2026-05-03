from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def _create_unreviewed_evidence(client: TestClient) -> str:
    source_response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Review Queue Manual", "productionEnabled": True},
    )
    assert source_response.status_code == 201
    source_id = source_response.json()["id"]
    sync_response = client.post(
        f"/v1/manual-sources/{source_id}/sync",
        json={
            "objects": [
                {
                    "sourceExternalId": "queue-doc-1",
                    "title": "Review Queue Source",
                    "content": "Cornerstone evidence should enter a reviewer queue before it becomes official context.",
                    "sourceUrl": "https://example.internal/review-queue",
                }
            ]
        },
    )
    assert sync_response.status_code == 200
    return sync_response.json()["evidenceFragments"][0]["id"]


def test_review_queue_lists_unreviewed_evidence_with_source_context(client: TestClient) -> None:
    evidence_id = _create_unreviewed_evidence(client)

    response = client.get("/v1/evidence/review-queue")

    assert response.status_code == 200
    body = response.json()
    assert body["totalCount"] >= 1
    item = next(item for item in body["items"] if item["evidenceFragment"]["id"] == evidence_id)
    assert item["artifact"]["title"] == "Review Queue Source"
    assert item["dataSource"]["name"] == "Review Queue Manual"
    assert "review_evidence" in item["suggestedNextActions"]
    assert "create_concept_candidate" in item["suggestedNextActions"]


def test_review_queue_excludes_reviewed_evidence_by_default(client: TestClient) -> None:
    evidence_id = _create_unreviewed_evidence(client)
    review_response = client.post(
        f"/v1/evidence/{evidence_id}/review",
        json={"trustState": "reviewed", "reviewedBy": "reviewer@example.com"},
    )
    assert review_response.status_code == 200

    default_queue = client.get("/v1/evidence/review-queue")
    reviewed_queue = client.get("/v1/evidence/review-queue?trustState=reviewed")

    assert all(item["evidenceFragment"]["id"] != evidence_id for item in default_queue.json()["items"])
    assert any(item["evidenceFragment"]["id"] == evidence_id for item in reviewed_queue.json()["items"])


def test_reviewer_can_mark_evidence_conflicted(client: TestClient) -> None:
    evidence_id = _create_unreviewed_evidence(client)

    response = client.post(
        f"/v1/evidence/{evidence_id}/review",
        json={
            "trustState": "conflicted",
            "reviewedBy": "reviewer@example.com",
            "reviewNote": "This conflicts with a newer decision.",
        },
    )
    queue = client.get("/v1/evidence/review-queue?trustState=conflicted")

    assert response.status_code == 200
    assert response.json()["trustState"] == "conflicted"
    assert queue.status_code == 200
    assert queue.json()["conflictedCount"] == 1
    assert any(item["evidenceFragment"]["id"] == evidence_id for item in queue.json()["items"])


def test_reviewer_can_create_concept_candidate_from_evidence(client: TestClient) -> None:
    evidence_id = _create_unreviewed_evidence(client)

    response = client.post(
        f"/v1/evidence/{evidence_id}/concept-candidates",
        json={
            "name": "Evidence Review Queue",
            "shortDefinition": "A reviewer workflow for evidence fragments.",
            "createdBy": "reviewer@example.com",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "reviewing"
    assert body["evidenceFragmentIds"] == [evidence_id]


def test_unauthorized_actor_cannot_create_concept_candidate_from_evidence(client: TestClient) -> None:
    evidence_id = _create_unreviewed_evidence(client)

    response = client.post(
        f"/v1/evidence/{evidence_id}/concept-candidates",
        json={
            "name": "Unauthorized Candidate",
            "shortDefinition": "Should be blocked.",
            "createdBy": "random@example.com",
        },
    )

    assert response.status_code == 403
