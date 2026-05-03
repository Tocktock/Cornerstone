from __future__ import annotations

from fastapi.testclient import TestClient


def test_ontology_ssot_readiness_reports_not_ready_before_official_graph(client: TestClient) -> None:
    response = client.get("/v1/ontology/ssot/readiness", params={"focusConcept": "settlement"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["releaseVersion"] == "2.5.0"
    assert body["status"] == "failed"
    assert body["officialGraphAvailable"] is False
    assert body["officialGraphSafe"] is False
    assert body["recommendedActions"]
    assert {check["key"] for check in body["checks"]} >= {
        "source_ingestion_available",
        "official_graph_available",
        "official_graph_safe",
        "evidence_citations_valid",
        "review_provenance_present",
        "candidate_boundary_respected",
        "ontology_evaluation_available",
    }

    sources = client.get("/v1/sources")
    assert sources.status_code == 200
    assert sources.json()["sources"] == []


def test_ontology_ssot_readiness_passes_after_operator_proof(client: TestClient) -> None:
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
    assert proof.json()["status"] == "passed"

    response = client.get(
        "/v1/ontology/ssot/readiness",
        params={
            "focusConcept": "Settlement",
            "depth": 1,
            "mode": "official",
            "includeGraph": True,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "passed"
    assert body["officialGraphAvailable"] is True
    assert body["officialGraphSafe"] is True
    assert body["trustLabel"] == "official"
    assert body["latestEvaluationSuccess"] is True
    assert body["graph"] is not None
    assert body["graph"]["officialGraphAvailable"] is True
    assert body["graph"]["depth"] == 1
    assert body["nodeCount"] >= 3
    assert body["edgeCount"] >= 1
    assert body["evidenceCount"] >= 1
    assert all(check["status"] == "passed" for check in body["checks"] if check["required"])
    assert body["recommendedActions"] == []
