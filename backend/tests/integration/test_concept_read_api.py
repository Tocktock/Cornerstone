from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_concept_list_and_read_return_created_candidate(client: TestClient) -> None:
    concept_response = client.post(
        "/v1/concepts",
        json={
            "name": "Cornerstone",
            "shortDefinition": "Shared organizational context layer.",
            "createdBy": "reviewer@example.com",
        },
    )
    assert concept_response.status_code == 201
    concept_id = concept_response.json()["id"]

    list_response = client.get("/v1/concepts")
    read_response = client.get(f"/v1/concepts/{concept_id}")

    assert list_response.status_code == 200
    assert [concept["id"] for concept in list_response.json()] == [concept_id]
    assert read_response.status_code == 200
    assert read_response.json()["name"] == "Cornerstone"


def test_create_concept_with_missing_evidence_returns_not_found(client: TestClient) -> None:
    response = client.post(
        "/v1/concepts",
        json={
            "name": "Cornerstone",
            "shortDefinition": "Shared organizational context layer.",
            "evidenceFragmentIds": ["missing-evidence"],
            "createdBy": "reviewer@example.com",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "EvidenceFragment not found: missing-evidence"
