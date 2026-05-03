from __future__ import annotations

import json

from fastapi.testclient import TestClient

from cornerstone.config import Settings
from cornerstone.main import create_app
from cornerstone.store import InMemoryStore


def _create_manual_source(client: TestClient) -> str:
    response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Ontology Extraction", "productionEnabled": True},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def _upload_settlement_text(client: TestClient) -> dict[str, str]:
    source_id = _create_manual_source(client)
    response = client.post(
        f"/v1/manual-sources/{source_id}/uploads/text",
        json={
            "objects": [
                {
                    "title": "Settlement Notes",
                    "content": (
                        "Settlement is the process of finalizing obligations. "
                        "Clearing precedes settlement. "
                        "Reconciliation validates settlement."
                    ),
                }
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    return {
        "source_id": source_id,
        "artifact_id": body["artifacts"][0]["id"],
        "evidence_id": body["evidenceFragments"][0]["id"],
    }


def test_ontology_extraction_run_creates_candidate_only_concepts_and_relations(
    client: TestClient,
) -> None:
    ids = _upload_settlement_text(client)

    response = client.post(
        "/v1/ontology/extraction-runs",
        json={
            "artifactIds": [ids["artifact_id"]],
            "focusConcept": "settlement",
            "requestedBy": "reviewer@example.com",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["run"]["status"] == "completed"
    assert body["run"]["conceptCandidateCount"] >= 2
    assert body["run"]["relationCandidateCount"] >= 2

    concept_names = {candidate["name"] for candidate in body["conceptCandidates"]}
    assert {"Settlement", "Clearing"} <= concept_names
    settlement = next(candidate for candidate in body["conceptCandidates"] if candidate["name"] == "Settlement")
    assert settlement["status"] == "pending"
    assert settlement["proposedDefinition"].startswith("the process of finalizing obligations")
    assert settlement["matchedExistingConceptId"] is None

    relation_tuples = {
        (candidate["sourceName"], candidate["relationType"], candidate["targetName"])
        for candidate in body["relationCandidates"]
    }
    assert ("Clearing", "precedes", "Settlement") in relation_tuples
    assert ("Reconciliation", "validates", "Settlement") in relation_tuples

    graph_response = client.get("/v1/ontology/graph", params={"concept": "settlement"})
    assert graph_response.status_code == 200
    assert graph_response.json()["officialGraphAvailable"] is False


def test_ontology_extraction_run_can_be_read_and_candidates_listed(client: TestClient) -> None:
    ids = _upload_settlement_text(client)
    create = client.post(
        "/v1/ontology/extraction-runs",
        json={"artifactIds": [ids["artifact_id"]], "requestedBy": "reviewer@example.com"},
    )
    assert create.status_code == 201
    run_id = create.json()["run"]["id"]

    read = client.get(f"/v1/ontology/extraction-runs/{run_id}")
    assert read.status_code == 200
    assert read.json()["run"]["id"] == run_id

    concepts = client.get("/v1/ontology/concept-candidates", params={"runId": run_id, "status": "pending"})
    assert concepts.status_code == 200
    assert concepts.json()["candidates"]

    relations = client.get("/v1/ontology/relation-candidates", params={"runId": run_id, "status": "pending"})
    assert relations.status_code == 200
    assert relations.json()["candidates"]

    runs = client.get("/v1/ontology/extraction-runs")
    assert runs.status_code == 200
    assert any(item["id"] == run_id for item in runs.json()["runs"])


def test_ontology_extraction_matches_existing_concept_by_name_or_alias(
    client: TestClient,
) -> None:
    ids = _upload_settlement_text(client)
    concept = client.post(
        "/v1/concepts",
        json={
            "name": "Settlement",
            "aliases": ["payment settlement"],
            "shortDefinition": "Official settlement definition.",
            "evidenceFragmentIds": [ids["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    assert concept.status_code == 201
    concept_id = concept.json()["id"]

    response = client.post(
        "/v1/ontology/extraction-runs",
        json={
            "artifactIds": [ids["artifact_id"]],
            "focusConcept": "payment settlement",
            "requestedBy": "reviewer@example.com",
        },
    )

    assert response.status_code == 201
    candidates = response.json()["conceptCandidates"]
    assert candidates
    assert any(candidate["name"] == "Settlement" and candidate["matchedExistingConceptId"] == concept_id for candidate in candidates)


def test_ontology_extraction_rejects_empty_scope(client: TestClient) -> None:
    response = client.post(
        "/v1/ontology/extraction-runs",
        json={"requestedBy": "reviewer@example.com"},
    )

    assert response.status_code == 422


def test_ontology_extraction_candidates_round_trip_in_persistent_store(
    persistent_client: TestClient,
) -> None:
    ids = _upload_settlement_text(persistent_client)

    response = persistent_client.post(
        "/v1/ontology/extraction-runs",
        json={"artifactIds": [ids["artifact_id"]], "requestedBy": "reviewer@example.com"},
    )

    assert response.status_code == 201
    run_id = response.json()["run"]["id"]
    read = persistent_client.get(f"/v1/ontology/extraction-runs/{run_id}")
    assert read.status_code == 200
    assert read.json()["conceptCandidates"]
    assert read.json()["relationCandidates"]


def test_live_llm_provider_uses_fixture_defaults_and_keeps_candidates_pending(store: InMemoryStore) -> None:
    settings = Settings(
        ontology_live_llm_enabled=True,
        ontology_live_llm_fixture_response_json=json.dumps(
            {
                "concepts": [
                    {
                        "name": "Settlement",
                        "definition": "A reviewed process for finalizing obligations.",
                        "concept_type": "process",
                        "evidence_fragment_ids": ["placeholder"],
                        "confidence": 0.91,
                    }
                ],
                "relations": [
                    {
                        "source_name": "Clearing",
                        "target_name": "Settlement",
                        "relation_type": "precedes",
                        "evidence_fragment_ids": ["placeholder"],
                        "confidence": 0.86,
                        "rationale": "The source text states the ordering explicitly.",
                    }
                ],
            }
        ),
    )
    client = TestClient(create_app(store=store, settings=settings))
    ids = _upload_settlement_text(client)
    evidence_id = ids["evidence_id"]
    settings.ontology_live_llm_fixture_response_json = settings.ontology_live_llm_fixture_response_json.replace(
        "placeholder",
        evidence_id,
    )

    response = client.post(
        "/v1/ontology/extraction-runs",
        json={
            "artifactIds": [ids["artifact_id"]],
            "provider": "live_llm",
            "focusConcept": "settlement",
            "requestedBy": "reviewer@example.com",
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["run"]["provider"] == "live_llm"
    assert body["run"]["modelName"] == "cornerstone-live-ontology-model"
    assert body["run"]["promptVersion"] == "ontology-extraction-v2.1.0"
    assert body["conceptCandidates"][0]["status"] == "pending"
    assert body["conceptCandidates"][0]["evidenceFragmentIds"] == [evidence_id]
    assert body["relationCandidates"][0]["status"] == "pending"

    graph = client.get("/v1/ontology/graph", params={"concept": "Settlement"})
    assert graph.status_code == 200
    assert graph.json()["officialGraphAvailable"] is False


def test_live_llm_provider_is_explicitly_gated(client: TestClient) -> None:
    ids = _upload_settlement_text(client)

    response = client.post(
        "/v1/ontology/extraction-runs",
        json={"artifactIds": [ids["artifact_id"]], "provider": "live_llm"},
    )

    assert response.status_code == 422
    assert "Live ontology provider is disabled" in response.json()["detail"]


def test_live_llm_provider_rejects_unknown_evidence_and_self_relations(store: InMemoryStore) -> None:
    settings = Settings(
        ontology_live_llm_enabled=True,
        ontology_live_llm_fixture_response_json=json.dumps(
            {
                "concepts": [
                    {
                        "name": "Settlement",
                        "definition": "A process.",
                        "concept_type": "process",
                        "evidence_fragment_ids": ["missing-evidence"],
                    }
                ],
                "relations": [
                    {
                        "source_name": "Settlement",
                        "target_name": "Settlement",
                        "relation_type": "precedes",
                        "evidence_fragment_ids": ["missing-evidence"],
                    }
                ],
            }
        ),
    )
    client = TestClient(create_app(store=store, settings=settings))
    ids = _upload_settlement_text(client)

    response = client.post(
        "/v1/ontology/extraction-runs",
        json={"artifactIds": [ids["artifact_id"]], "provider": "live_llm"},
    )

    assert response.status_code == 422
    assert "unknown evidence ids" in response.json()["detail"]


def test_live_llm_provider_rejects_self_relations_after_evidence_validation(store: InMemoryStore) -> None:
    settings = Settings(
        ontology_live_llm_enabled=True,
        ontology_live_llm_fixture_response_json=json.dumps(
            {
                "concepts": [],
                "relations": [
                    {
                        "source_name": "Settlement",
                        "target_name": "Settlement",
                        "relation_type": "precedes",
                        "evidence_fragment_ids": ["placeholder"],
                    }
                ],
            }
        ),
    )
    client = TestClient(create_app(store=store, settings=settings))
    ids = _upload_settlement_text(client)
    settings.ontology_live_llm_fixture_response_json = settings.ontology_live_llm_fixture_response_json.replace(
        "placeholder",
        ids["evidence_id"],
    )

    response = client.post(
        "/v1/ontology/extraction-runs",
        json={"artifactIds": [ids["artifact_id"]], "provider": "live_llm"},
    )

    assert response.status_code == 422
    assert "Relation source and target must differ" in response.json()["detail"]
