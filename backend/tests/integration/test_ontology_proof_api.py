from __future__ import annotations

from fastapi.testclient import TestClient


def test_ontology_proof_dry_run_returns_planned_checklist_without_mutation(client: TestClient) -> None:
    response = client.post(
        "/v1/ontology/proof-runs",
        json={"dryRun": True, "focusConcept": "Settlement"},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "planned"
    assert body["dryRun"] is True
    assert body["sourceId"] is None
    assert body["artifactIds"] == []
    assert body["summary"]["plannedCount"] == len(body["checklist"])
    assert {step["key"] for step in body["checklist"]} == {
        "create_manual_source",
        "sync_manual_seed",
        "run_reextraction",
        "review_evidence",
        "approve_concepts",
        "approve_relations",
        "serve_explainable_graph",
        "run_ontology_evaluation",
    }

    sources = client.get("/v1/sources")
    assert sources.status_code == 200
    assert sources.json()["sources"] == []


def test_ontology_proof_requires_explicit_mutation_confirmation(client: TestClient) -> None:
    response = client.post(
        "/v1/ontology/proof-runs",
        json={"focusConcept": "Settlement", "reviewer": "reviewer@example.com"},
    )

    assert response.status_code == 409
    assert "confirmMutation=true" in response.text


def test_ontology_proof_runs_complete_operator_checklist(client: TestClient) -> None:
    response = client.post(
        "/v1/ontology/proof-runs",
        json={
            "focusConcept": "Settlement",
            "reviewer": "reviewer@example.com",
            "createdBy": "operator@example.com",
            "confirmMutation": True,
            "runEvaluation": True,
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "passed"
    assert body["summary"]["requiredFailed"] == 0
    assert body["summary"]["requiredPassed"] == len(body["checklist"])
    assert body["summary"]["officialGraphAvailable"] is True
    assert body["summary"]["officialGraphMutated"] is False
    assert body["summary"]["evaluationSuccess"] is True
    assert body["sourceId"]
    assert body["artifactIds"]
    assert body["evidenceFragmentIds"]
    assert body["reextractionRunId"]
    assert body["extractionRunIds"]
    assert body["conceptCandidateIds"]
    assert body["relationCandidateIds"]
    assert body["approvedConceptIds"]
    assert body["approvedRelationIds"]
    assert body["graphResponseId"]
    assert body["evaluationTaskId"]
    assert body["evaluationResultId"]
    assert all(step["status"] == "passed" for step in body["checklist"])

    graph = client.get("/v1/ontology/graph", params={"concept": "Settlement", "depth": 1, "mode": "official"})
    assert graph.status_code == 200
    graph_body = graph.json()
    assert graph_body["officialGraphAvailable"] is True
    assert graph_body["trustLabel"] == "official"
    assert len(graph_body["edges"]) >= 1


def test_ontology_proof_preserves_authorized_reviewer_gate(client: TestClient) -> None:
    response = client.post(
        "/v1/ontology/proof-runs",
        json={
            "focusConcept": "Settlement",
            "reviewer": "intruder@example.com",
            "confirmMutation": True,
        },
    )

    assert response.status_code == 409
    assert "Reviewer is not authorized" in response.text
