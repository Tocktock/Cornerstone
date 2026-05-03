from __future__ import annotations

from fastapi.testclient import TestClient


def _create_official_settlement_graph(client: TestClient) -> dict[str, str]:
    source = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Ontology Eval", "productionEnabled": True},
    )
    assert source.status_code == 201
    source_id = source.json()["id"]
    sync = client.post(
        f"/v1/manual-sources/{source_id}/sync",
        json={
            "objects": [
                {
                    "sourceExternalId": "settlement-eval-doc",
                    "title": "Settlement Evaluation Notes",
                    "content": "Settlement is the process of finalizing obligations. Settlement updates the ledger.",
                    "sourceUrl": "https://example.internal/settlement-eval-doc",
                }
            ]
        },
    )
    assert sync.status_code == 200
    evidence_id = sync.json()["evidenceFragments"][0]["id"]
    review = client.post(
        f"/v1/evidence/{evidence_id}/review",
        json={"trustState": "reviewed", "reviewedBy": "reviewer@example.com"},
    )
    assert review.status_code == 200

    settlement = client.post(
        "/v1/concepts",
        json={
            "name": "Settlement",
            "aliases": ["payment settlement"],
            "shortDefinition": "Settlement is the process of finalizing obligations.",
            "evidenceFragmentIds": [evidence_id],
            "createdBy": "reviewer@example.com",
        },
    )
    assert settlement.status_code == 201
    settlement_id = settlement.json()["id"]
    assert client.post(
        f"/v1/concepts/{settlement_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    ).status_code == 200

    ledger = client.post(
        "/v1/concepts",
        json={
            "name": "Ledger",
            "shortDefinition": "A ledger records settled obligations.",
            "evidenceFragmentIds": [evidence_id],
            "createdBy": "reviewer@example.com",
        },
    )
    assert ledger.status_code == 201
    ledger_id = ledger.json()["id"]
    assert client.post(
        f"/v1/concepts/{ledger_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    ).status_code == 200

    relation = client.post(
        "/v1/concept-relations",
        json={
            "sourceConceptId": settlement_id,
            "targetConceptId": ledger_id,
            "relationType": "updates",
            "evidenceFragmentIds": [evidence_id],
            "createdBy": "reviewer@example.com",
        },
    )
    assert relation.status_code == 201
    relation_id = relation.json()["id"]
    assert client.post(
        f"/v1/concept-relations/{relation_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    ).status_code == 200

    return {
        "evidence_id": evidence_id,
        "settlement_id": settlement_id,
        "ledger_id": ledger_id,
        "relation_id": relation_id,
    }


def test_ontology_graph_evaluation_task_run_succeeds_for_official_graph(client: TestClient) -> None:
    graph = _create_official_settlement_graph(client)

    create = client.post(
        "/v1/evaluations/ontology/tasks",
        json={
            "name": "Settlement official graph",
            "conceptQuery": "payment settlement",
            "expectedTrustLabel": "official",
            "requiredConceptIds": [graph["settlement_id"], graph["ledger_id"]],
            "requiredRelationIds": [graph["relation_id"]],
            "requiredEvidenceFragmentIds": [graph["evidence_id"]],
            "requireOfficialGraph": True,
            "minEvidenceCount": 1,
            "minNodeCount": 2,
            "minEdgeCount": 1,
            "maxPendingCandidateCount": 0,
            "createdBy": "qa@example.com",
            "tags": ["ontology", "official"],
        },
    )
    assert create.status_code == 201, create.text
    task_id = create.json()["id"]

    run = client.post(f"/v1/evaluations/ontology/tasks/{task_id}/run", json={"evaluatedBy": "qa@example.com"})

    assert run.status_code == 200, run.text
    result = run.json()
    assert result["success"] is True
    assert result["graphFound"] is True
    assert result["graphDepthRespected"] is True
    assert result["nodeRequirementsMet"] is True
    assert result["edgeRequirementsMet"] is True
    assert result["evidenceValid"] is True
    assert result["provenancePresent"] is True
    assert result["officialGraphSafe"] is True
    assert result["candidateBoundaryRespected"] is True
    assert result["relationIntegrityValid"] is True
    assert result["citationValidityRate"] == 1.0
    assert result["response"]["trustLabel"] == "official"

    summary = client.get("/v1/evaluations/ontology/summary")
    assert summary.status_code == 200
    assert summary.json()["ontologyGraphTaskSuccessRate"] == 1.0


def test_ontology_graph_evaluation_detects_missing_official_graph(client: TestClient) -> None:
    create = client.post(
        "/v1/evaluations/ontology/tasks",
        json={
            "name": "Missing settlement graph",
            "conceptQuery": "Missing Settlement",
            "expectedTrustLabel": "official",
            "requireOfficialGraph": True,
            "minEvidenceCount": 1,
            "minNodeCount": 1,
            "createdBy": "qa@example.com",
        },
    )
    assert create.status_code == 201, create.text

    run = client.post(f"/v1/evaluations/ontology/tasks/{create.json()['id']}/run")

    assert run.status_code == 200
    result = run.json()
    assert result["success"] is False
    assert result["graphFound"] is False
    assert result["trustLabelCorrect"] is False
    assert "graph_not_found" in result["failureReasons"]
    assert "official_graph_safety_violation" in result["failureReasons"]


def test_ontology_graph_evaluation_allows_explicit_unsupported_graph_task(client: TestClient) -> None:
    create = client.post(
        "/v1/evaluations/ontology/tasks",
        json={
            "name": "Unsupported graph expected",
            "conceptQuery": "Missing Settlement",
            "expectedTrustLabel": "unsupported",
            "requireOfficialGraph": False,
            "requireEvidence": False,
            "minEvidenceCount": 0,
            "minNodeCount": 0,
            "minEdgeCount": 0,
            "requireReviewProvenance": False,
            "createdBy": "qa@example.com",
        },
    )
    assert create.status_code == 201, create.text

    run = client.post(f"/v1/evaluations/ontology/tasks/{create.json()['id']}/run")

    assert run.status_code == 200, run.text
    result = run.json()
    assert result["success"] is True
    assert result["graphFound"] is False
    assert result["trustLabel"] == "unsupported"
    assert result["candidateBoundaryRespected"] is True


def test_ontology_graph_evaluation_rejects_invalid_unsupported_contract(client: TestClient) -> None:
    response = client.post(
        "/v1/evaluations/ontology/tasks",
        json={
            "name": "Invalid unsupported graph task",
            "conceptQuery": "Missing Settlement",
            "expectedTrustLabel": "unsupported",
            "requireEvidence": True,
            "createdBy": "qa@example.com",
        },
    )

    assert response.status_code == 422
    assert "Unsupported ontology graph evaluation tasks" in response.text


def test_ontology_graph_evaluation_bulk_run_and_results_filter(client: TestClient) -> None:
    first = client.post(
        "/v1/evaluations/ontology/tasks",
        json={
            "name": "Unsupported graph one",
            "conceptQuery": "Missing Settlement One",
            "expectedTrustLabel": "unsupported",
            "requireOfficialGraph": False,
            "requireEvidence": False,
            "minEvidenceCount": 0,
            "minNodeCount": 0,
            "minEdgeCount": 0,
            "requireReviewProvenance": False,
            "createdBy": "qa@example.com",
        },
    )
    assert first.status_code == 201
    second = client.post(
        "/v1/evaluations/ontology/tasks",
        json={
            "name": "Unsupported graph two",
            "conceptQuery": "Missing Settlement Two",
            "expectedTrustLabel": "unsupported",
            "requireOfficialGraph": False,
            "requireEvidence": False,
            "minEvidenceCount": 0,
            "minNodeCount": 0,
            "minEdgeCount": 0,
            "requireReviewProvenance": False,
            "createdBy": "qa@example.com",
        },
    )
    assert second.status_code == 201

    run = client.post("/v1/evaluations/ontology/run", json={"evaluatedBy": "qa@example.com"})

    assert run.status_code == 200
    body = run.json()
    assert body["totalCount"] == 2
    assert body["successCount"] == 2
    assert body["ontologyGraphTaskSuccessRate"] == 1.0

    task_results = client.get(f"/v1/evaluations/ontology/results?taskId={first.json()['id']}")
    assert task_results.status_code == 200
    assert len(task_results.json()) == 1


def test_ontology_graph_evaluation_list_read_and_missing_routes(client: TestClient) -> None:
    create = client.post(
        "/v1/evaluations/ontology/tasks",
        json={
            "name": "Listable unsupported graph",
            "conceptQuery": "Missing Settlement",
            "expectedTrustLabel": "unsupported",
            "requireOfficialGraph": False,
            "requireEvidence": False,
            "minEvidenceCount": 0,
            "minNodeCount": 0,
            "minEdgeCount": 0,
            "requireReviewProvenance": False,
            "createdBy": "qa@example.com",
        },
    )
    assert create.status_code == 201
    task_id = create.json()["id"]

    list_response = client.get("/v1/evaluations/ontology/tasks")
    read_response = client.get(f"/v1/evaluations/ontology/tasks/{task_id}")
    run_response = client.post(f"/v1/evaluations/ontology/tasks/{task_id}/run")
    result_id = run_response.json()["id"]
    result_response = client.get(f"/v1/evaluations/ontology/results/{result_id}")
    missing_task_response = client.get("/v1/evaluations/ontology/tasks/missing-task")
    missing_result_response = client.get("/v1/evaluations/ontology/results/missing-result")

    assert list_response.status_code == 200
    assert any(item["id"] == task_id for item in list_response.json())
    assert read_response.status_code == 200
    assert run_response.status_code == 200
    assert result_response.status_code == 200
    assert missing_task_response.status_code == 404
    assert missing_result_response.status_code == 404
