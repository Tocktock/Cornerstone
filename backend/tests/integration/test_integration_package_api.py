from __future__ import annotations

from fastapi.testclient import TestClient


def test_integration_package_manifest_declares_external_package_path(client: TestClient) -> None:
    response = client.get("/v1/integration/package/manifest")

    assert response.status_code == 200
    body = response.json()
    assert body["chosenPath"] == "external_integration_package"
    assert body["nonChosenPath"] == "frontend_mvp"
    assert "GET /v1/integration/ontology/{concept}" in body["stableEndpoints"]
    assert "review gates" in body["trustBoundary"]
    assert body["quickstart"]


def test_integration_ontology_endpoint_wraps_official_graph_and_readiness(client: TestClient) -> None:
    proof = client.post(
        "/v1/ontology/proof-runs",
        json={
            "focusConcept": "Settlement",
            "reviewer": "reviewer@example.com",
            "createdBy": "operator@example.com",
            "confirmMutation": True,
            "runEvaluation": True,
        },
    )
    assert proof.status_code == 201, proof.text

    response = client.get("/v1/integration/ontology/Settlement")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["concept"] == "Settlement"
    assert body["reviewGateBypassAllowed"] is False
    assert body["trustState"] == "official"
    assert body["unsupportedState"] is None
    assert body["officialGraph"]["officialGraphAvailable"] is True
    assert body["ssotReadiness"]["status"] == "passed"
    assert body["ssotReadiness"]["graph"] is not None
    assert body["evidenceCitations"]
    assert "Candidate counts are summarized" in body["candidateVsOfficialBoundary"]


def test_integration_ontology_rejects_candidate_bypass(client: TestClient) -> None:
    response = client.get("/v1/integration/ontology/Settlement", params={"includeCandidates": "true"})

    assert response.status_code == 409
    assert "review gates cannot be bypassed" in response.json()["detail"]
