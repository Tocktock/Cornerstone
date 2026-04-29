from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_authorized_reviewer_can_mark_evidence_reviewed(client: TestClient) -> None:
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
                    "content": "Cornerstone is a context layer.",
                    "sourceUrl": "https://example.internal/doc-1",
                }
            ]
        },
    )
    evidence_id = sync_response.json()["evidenceFragments"][0]["id"]

    response = client.post(
        f"/v1/evidence/{evidence_id}/review",
        json={"trustState": "reviewed", "reviewedBy": "reviewer@example.com"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trustState"] == "reviewed"
    assert body["reviewedBy"] == "reviewer@example.com"
    assert body["reviewedAt"]


def test_unauthorized_reviewer_cannot_review_evidence(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    response = client.post(
        f"/v1/evidence/{synced_evidence['evidence_id']}/review",
        json={"trustState": "reviewed", "reviewedBy": "random@example.com"},
    )

    assert response.status_code == 403
