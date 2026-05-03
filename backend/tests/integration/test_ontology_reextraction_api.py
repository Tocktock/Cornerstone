from __future__ import annotations

from fastapi.testclient import TestClient


def _create_manual_source(client: TestClient) -> str:
    response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Ontology Re-extraction", "productionEnabled": True},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def _upload_settlement_text(
    client: TestClient,
    source_id: str,
    *,
    content: str = "Settlement is the process of finalizing obligations. Clearing precedes settlement.",
    run_inline: bool = False,
) -> dict:
    response = client.post(
        f"/v1/manual-sources/{source_id}/uploads/text",
        json={
            "objects": [
                {
                    "title": "Settlement Notes",
                    "content": content,
                }
            ],
            "queueOntologyReExtraction": True,
            "runOntologyReExtractionInline": run_inline,
            "ontologyFocusConcept": "settlement",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_manual_text_upload_queues_reextraction_without_promoting_official_graph(client: TestClient) -> None:
    source_id = _create_manual_source(client)

    body = _upload_settlement_text(client, source_id)

    assert body["artifactCreatedCount"] == 1
    assert body["artifactReusedCount"] == 0
    assert body["artifactChangedCount"] == 0
    assert len(body["createdArtifactIds"]) == 1
    assert body["reusedArtifactIds"] == []
    assert body["changedArtifactIds"] == []
    assert body["ontologyReextractionRunId"] is not None
    assert body["ontologyReextractionStatus"] == "queued"

    run_id = body["ontologyReextractionRunId"]
    read = client.get(f"/v1/ontology/re-extraction-runs/{run_id}")
    assert read.status_code == 200
    run = read.json()["run"]
    assert run["status"] == "queued"
    assert run["trigger"] == "manual_upload"
    assert run["focusConcept"] == "settlement"
    assert run["artifactIds"] == body["createdArtifactIds"]
    assert run["officialGraphMutated"] is False
    assert read.json()["conceptCandidates"] == []
    assert read.json()["relationCandidates"] == []

    graph = client.get("/v1/ontology/graph", params={"concept": "settlement", "mode": "official"})
    assert graph.status_code == 200
    assert graph.json()["officialGraphAvailable"] is False


def test_running_reextraction_creates_candidates_only(client: TestClient) -> None:
    source_id = _create_manual_source(client)
    body = _upload_settlement_text(
        client,
        source_id,
        content=(
            "Settlement is the process of finalizing obligations. "
            "Clearing precedes settlement. Reconciliation validates settlement."
        ),
    )
    run_id = body["ontologyReextractionRunId"]

    run = client.post(
        f"/v1/ontology/re-extraction-runs/{run_id}/run",
        json={"requestedBy": "operator@example.com"},
    )

    assert run.status_code == 200, run.text
    response = run.json()
    reextraction = response["run"]
    assert reextraction["status"] == "completed"
    assert reextraction["officialGraphMutated"] is False
    assert reextraction["extractionRunIds"]
    assert reextraction["conceptCandidateCount"] >= 2
    assert reextraction["relationCandidateCount"] >= 2

    concept_names = {candidate["name"] for candidate in response["conceptCandidates"]}
    assert {"Settlement", "Clearing"} <= concept_names
    assert all(candidate["status"] == "pending" for candidate in response["conceptCandidates"])

    relation_tuples = {
        (candidate["sourceName"], candidate["relationType"], candidate["targetName"])
        for candidate in response["relationCandidates"]
    }
    assert ("Clearing", "precedes", "Settlement") in relation_tuples
    assert ("Reconciliation", "validates", "Settlement") in relation_tuples
    assert all(candidate["status"] == "pending" for candidate in response["relationCandidates"])

    graph = client.get("/v1/ontology/graph", params={"concept": "settlement", "mode": "official"})
    assert graph.status_code == 200
    assert graph.json()["officialGraphAvailable"] is False


def test_exact_reused_manual_text_does_not_queue_second_reextraction(client: TestClient) -> None:
    source_id = _create_manual_source(client)
    first = _upload_settlement_text(client, source_id)
    second = _upload_settlement_text(client, source_id)

    assert first["ontologyReextractionRunId"] is not None
    assert second["artifactCreatedCount"] == 0
    assert second["artifactReusedCount"] == 1
    assert second["createdArtifactIds"] == []
    assert second["reusedArtifactIds"] == [first["createdArtifactIds"][0]]
    assert second["ontologyReextractionRunId"] is None
    assert second["ontologyReextractionStatus"] is None

    list_response = client.get("/v1/ontology/re-extraction-runs", params={"datasourceId": source_id})
    assert list_response.status_code == 200
    assert len(list_response.json()["runs"]) == 1


def test_changed_manual_text_queues_new_reextraction_with_changed_artifact(client: TestClient) -> None:
    source_id = _create_manual_source(client)
    first = _upload_settlement_text(client, source_id)
    changed = _upload_settlement_text(
        client,
        source_id,
        content="Settlement is the process of finalizing obligations. Ledger updates follow settlement.",
    )

    assert changed["artifactCreatedCount"] == 1
    assert changed["artifactReusedCount"] == 0
    assert changed["artifactChangedCount"] == 1
    assert changed["createdArtifactIds"] != first["createdArtifactIds"]
    assert changed["changedArtifactIds"] == changed["createdArtifactIds"]
    assert changed["ontologyReextractionRunId"] is not None

    list_response = client.get("/v1/ontology/re-extraction-runs", params={"datasourceId": source_id})
    assert list_response.status_code == 200
    assert len(list_response.json()["runs"]) == 2


def test_manual_reextraction_request_can_scope_to_datasource_and_run_inline(client: TestClient) -> None:
    source_id = _create_manual_source(client)
    upload = _upload_settlement_text(client, source_id, run_inline=False)

    create = client.post(
        "/v1/ontology/re-extraction-runs",
        json={
            "datasourceId": source_id,
            "focusConcept": "settlement",
            "trigger": "manual_request",
            "createdBy": "operator@example.com",
            "reason": "Operator requested source-wide ontology refresh.",
            "runInline": True,
        },
    )

    assert create.status_code == 201, create.text
    body = create.json()
    assert body["run"]["status"] == "completed"
    assert body["run"]["artifactIds"] == upload["createdArtifactIds"]
    assert body["run"]["officialGraphMutated"] is False
    assert body["conceptCandidates"]
    assert body["relationCandidates"]


def _connect_test_discover_select_notion(client: TestClient) -> str:
    intent = client.post(
        "/v1/connections/intents",
        json={"provider": "notion", "sourceName": "Team Notion", "createdBy": "admin@example.com"},
    ).json()
    connect_response = client.get(f"/v1/oauth/notion/callback?state={intent['stateNonce']}&code=code-123")
    assert connect_response.status_code == 200
    source_id = connect_response.json()["dataSource"]["id"]
    assert client.post(f"/v1/sources/{source_id}/test").status_code == 200
    assert client.post(f"/v1/sources/{source_id}/discover", json={"pageSize": 25}).status_code == 200
    select_response = client.put(
        f"/v1/sources/{source_id}/selections",
        json={"syncMode": "selected_only", "selectedExternalObjectIds": ["notion-page-1"]},
    )
    assert select_response.status_code == 200
    return source_id


def test_connector_sync_worker_queues_reextraction_event(client: TestClient) -> None:
    source_id = _connect_test_discover_select_notion(client)
    create_response = client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com", "ontologyFocusConcept": "settlement"},
    )
    assert create_response.status_code == 201

    run_response = client.post("/v1/sync-worker/run", json={"maxJobs": 1})

    assert run_response.status_code == 200, run_response.text
    body = run_response.json()
    assert body["processedJobCount"] == 1
    detail = body["jobs"][0]
    assert detail["job"]["status"] == "succeeded"
    event_types = [event["eventType"] for event in detail["events"]]
    assert "ontology.reextraction_queued" in event_types
    reextraction_event = next(event for event in detail["events"] if event["eventType"] == "ontology.reextraction_queued")
    assert reextraction_event["metadata"]["officialGraphMutated"] is False
    assert reextraction_event["metadata"]["status"] == "queued"

    list_response = client.get("/v1/ontology/re-extraction-runs", params={"datasourceId": source_id})
    assert list_response.status_code == 200
    runs = list_response.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["trigger"] == "connector_sync"
    assert runs[0]["syncJobId"] == detail["job"]["id"]
    assert runs[0]["focusConcept"] == "settlement"
