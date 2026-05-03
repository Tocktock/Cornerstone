from __future__ import annotations

from fastapi.testclient import TestClient


def test_connector_support_matrix_exposes_live_proof_boundaries_without_secrets(client: TestClient) -> None:
    response = client.get("/v1/connectors/support-matrix")

    assert response.status_code == 200
    body = response.json()
    items = body["items"]
    assert items
    assert all(item["mutatesOfficialGraph"] is False for item in items)
    assert any(item["provider"] == "notion" and item["objectType"] == "page" for item in items)
    assert any(item["provider"] == "google_drive" and item["createsEvidence"] is True for item in items)
    assert "ONTOLOGY_LIVE_LLM_ENABLED" in body["liveProofEnvGuards"]
    assert "never persisted" in body["secretRedactionPolicy"]
    assert "mock-notion-access-token" not in response.text
    assert "Authorization:" not in response.text
