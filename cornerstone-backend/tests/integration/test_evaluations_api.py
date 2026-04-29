from __future__ import annotations

from fastapi.testclient import TestClient


def _officialize_cornerstone(client: TestClient, evidence_id: str) -> str:
    concept_response = client.post(
        "/v1/concepts",
        json={
            "name": "Cornerstone",
            "shortDefinition": "Cornerstone is a shared organizational context layer.",
            "evidenceFragmentIds": [evidence_id],
            "createdBy": "reviewer@example.com",
        },
    )
    assert concept_response.status_code == 201
    concept_id = concept_response.json()["id"]
    officialize_response = client.post(
        f"/v1/concepts/{concept_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert officialize_response.status_code == 200
    return concept_id


def test_evaluation_task_run_succeeds_for_official_grounded_context(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    concept_id = _officialize_cornerstone(client, synced_evidence["evidence_id"])

    create_response = client.post(
        "/v1/evaluations/tasks",
        json={
            "name": "Cornerstone official answer",
            "query": "What is Cornerstone?",
            "expectedAnswerContains": ["shared organizational context layer"],
            "expectedTrustLabel": "official",
            "requiredEvidenceFragmentIds": [synced_evidence["evidence_id"]],
            "requiredConceptIds": [concept_id],
            "requireOfficialAnswer": True,
            "createdBy": "qa@example.com",
            "tags": ["pilot", "official"],
        },
    )
    assert create_response.status_code == 201
    task_id = create_response.json()["id"]

    run_response = client.post(
        f"/v1/evaluations/tasks/{task_id}/run",
        json={"evaluatedBy": "qa@example.com", "clarificationReduced": True},
    )

    assert run_response.status_code == 200
    result = run_response.json()
    assert result["success"] is True
    assert result["answerCorrect"] is True
    assert result["evidenceValid"] is True
    assert result["provenancePresent"] is True
    assert result["trustLabelCorrect"] is True
    assert result["freshnessPolicyRespected"] is True
    assert result["unsupportedOfficialClaim"] is False
    assert result["citationValidityRate"] == 1.0
    assert result["response"]["trustLabel"] == "official"

    summary_response = client.get("/v1/evaluations/summary")
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert summary["totalTaskCount"] == 1
    assert summary["evaluatedResultCount"] == 1
    assert summary["groundedContextTaskSuccessRate"] == 1.0


def test_evaluation_detects_unsupported_official_claim(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    _officialize_cornerstone(client, synced_evidence["evidence_id"])
    create_response = client.post(
        "/v1/evaluations/tasks",
        json={
            "name": "Should be unsupported but official exists",
            "query": "What is Cornerstone?",
            "expectedTrustLabel": "unsupported",
            "requireEvidence": False,
            "minEvidenceCount": 0,
            "createdBy": "qa@example.com",
        },
    )
    assert create_response.status_code == 201

    run_response = client.post(f"/v1/evaluations/tasks/{create_response.json()['id']}/run")

    assert run_response.status_code == 200
    result = run_response.json()
    assert result["success"] is False
    assert result["trustLabelCorrect"] is False
    assert result["unsupportedOfficialClaim"] is True
    assert "unsupported_official_claim" in result["failureReasons"]


def test_bulk_evaluation_run_and_results_filter(client: TestClient) -> None:
    first = client.post(
        "/v1/evaluations/tasks",
        json={
            "name": "Unsupported missing thing",
            "query": "What is MissingThing?",
            "expectedTrustLabel": "unsupported",
            "requireEvidence": False,
            "minEvidenceCount": 0,
            "createdBy": "qa@example.com",
        },
    )
    assert first.status_code == 201
    second = client.post(
        "/v1/evaluations/tasks",
        json={
            "name": "Unsupported other thing",
            "query": "What is OtherMissingThing?",
            "expectedTrustLabel": "unsupported",
            "requireEvidence": False,
            "minEvidenceCount": 0,
            "createdBy": "qa@example.com",
        },
    )
    assert second.status_code == 201

    run_response = client.post("/v1/evaluations/run", json={"evaluatedBy": "qa@example.com"})

    assert run_response.status_code == 200
    body = run_response.json()
    assert body["totalCount"] == 2
    assert body["successCount"] == 2
    assert body["groundedContextTaskSuccessRate"] == 1.0

    task_results = client.get(f"/v1/evaluations/results?taskId={first.json()['id']}")
    assert task_results.status_code == 200
    assert len(task_results.json()) == 1


def test_evaluation_task_rejects_unsupported_with_evidence_requirement(client: TestClient) -> None:
    response = client.post(
        "/v1/evaluations/tasks",
        json={
            "name": "Invalid unsupported task",
            "query": "What is MissingThing?",
            "expectedTrustLabel": "unsupported",
            "requireEvidence": True,
            "createdBy": "qa@example.com",
        },
    )

    assert response.status_code == 422


def test_persistent_evaluation_api_stores_tasks_and_results(persistent_client: TestClient) -> None:
    create_response = persistent_client.post(
        "/v1/evaluations/tasks",
        json={
            "name": "Persistent unsupported task",
            "query": "What is MissingThing?",
            "expectedTrustLabel": "unsupported",
            "requireEvidence": False,
            "minEvidenceCount": 0,
            "createdBy": "qa@example.com",
        },
    )
    assert create_response.status_code == 201
    task_id = create_response.json()["id"]

    run_response = persistent_client.post(f"/v1/evaluations/tasks/{task_id}/run")
    assert run_response.status_code == 200
    result_id = run_response.json()["id"]

    assert persistent_client.get(f"/v1/evaluations/tasks/{task_id}").status_code == 200
    assert persistent_client.get(f"/v1/evaluations/results/{result_id}").json()["success"] is True


def test_evaluation_task_rejects_vague_success_contract(client: TestClient) -> None:
    response = client.post(
        "/v1/evaluations/tasks",
        json={
            "name": "Vague task",
            "query": "What is Cornerstone?",
            "createdBy": "qa@example.com",
        },
    )

    assert response.status_code == 422
    assert "explicit success condition" in response.text


def test_evaluation_task_rejects_unsupported_with_nonzero_evidence_count(client: TestClient) -> None:
    response = client.post(
        "/v1/evaluations/tasks",
        json={
            "name": "Invalid unsupported task",
            "query": "What is MissingThing?",
            "expectedTrustLabel": "unsupported",
            "requireEvidence": False,
            "minEvidenceCount": 1,
            "createdBy": "qa@example.com",
        },
    )

    assert response.status_code == 422
    assert "minEvidenceCount=0" in response.text


def test_evaluation_task_rejects_underspecified_supported_task(client: TestClient) -> None:
    response = client.post(
        "/v1/evaluations/tasks",
        json={
            "name": "Weak task",
            "query": "What is Cornerstone?",
            "requireEvidence": False,
            "minEvidenceCount": 0,
            "createdBy": "qa@example.com",
        },
    )

    assert response.status_code == 422


def test_evaluation_list_read_and_missing_routes(client: TestClient) -> None:
    create_response = client.post(
        "/v1/evaluations/tasks",
        json={
            "name": "Listable unsupported task",
            "query": "What is MissingThing?",
            "expectedTrustLabel": "unsupported",
            "requireEvidence": False,
            "minEvidenceCount": 0,
            "createdBy": "qa@example.com",
        },
    )
    assert create_response.status_code == 201
    task_id = create_response.json()["id"]

    list_response = client.get("/v1/evaluations/tasks")
    read_response = client.get(f"/v1/evaluations/tasks/{task_id}")
    run_response = client.post(f"/v1/evaluations/tasks/{task_id}/run")
    result_id = run_response.json()["id"]
    result_response = client.get(f"/v1/evaluations/results/{result_id}")
    missing_task_response = client.get("/v1/evaluations/tasks/missing-task")
    missing_result_response = client.get("/v1/evaluations/results/missing-result")

    assert list_response.status_code == 200
    assert any(item["id"] == task_id for item in list_response.json())
    assert read_response.status_code == 200
    assert run_response.status_code == 200
    assert result_response.status_code == 200
    assert missing_task_response.status_code == 404
    assert missing_result_response.status_code == 404
