from __future__ import annotations

from fastapi.testclient import TestClient


def _context_space_id(client: TestClient) -> str:
    response = client.get("/api/v1/context-spaces")
    response.raise_for_status()
    return response.json()[0]["id"]


def _reviewer_id(client: TestClient, context_space_id: str) -> str:
    response = client.get("/api/v1/actors", params={"context_space_id": context_space_id})
    response.raise_for_status()
    return next(actor["id"] for actor in response.json() if "reviewer" in [role.lower() for role in actor["roles"]])


def test_sync_is_idempotent_and_creates_curated_knowledge(client: TestClient):
    context_space_id = _context_space_id(client)
    connection = client.get("/api/v1/source-connections", params={"context_space_id": context_space_id}).json()[0]

    before_artifacts = client.get("/api/v1/artifacts", params={"context_space_id": context_space_id}).json()
    before_concepts = client.get("/api/v1/concepts", params={"context_space_id": context_space_id}).json()

    sync_response = client.post(f"/api/v1/source-connections/{connection['id']}/sync")
    assert sync_response.status_code == 200
    assert sync_response.json()["artifact_count"] == len(before_artifacts)

    after_artifacts = client.get("/api/v1/artifacts", params={"context_space_id": context_space_id}).json()
    after_concepts = client.get("/api/v1/concepts", params={"context_space_id": context_space_id}).json()
    decisions = client.get("/api/v1/decisions", params={"context_space_id": context_space_id}).json()
    relations = client.get("/api/v1/relations", params={"context_space_id": context_space_id}).json()

    assert len(after_artifacts) == len(before_artifacts)
    assert len(after_concepts) == len(before_concepts)
    assert len(after_artifacts) >= 7
    assert all(item["evidence_count"] >= 1 for item in after_artifacts)
    assert len(decisions) >= 2
    assert len(relations) >= 2


def test_ungrounded_concept_cannot_be_officialized(client: TestClient):
    context_space_id = _context_space_id(client)
    reviewer_id = _reviewer_id(client, context_space_id)

    create_response = client.post(
        "/api/v1/concepts",
        json={
            "context_space_id": context_space_id,
            "concept_type": "TERM",
            "canonical_name": "Ungrounded Concept",
            "aliases": [],
            "definition": "No evidence is linked to this concept.",
            "owner_actor_id": None,
            "evidence_fragment_ids": [],
            "linked_decision_ids": [],
        },
    )
    assert create_response.status_code == 200
    concept_id = create_response.json()["id"]

    review_response = client.post(
        f"/api/v1/concepts/{concept_id}/review",
        json={"actor_id": reviewer_id, "action": "approve"},
    )
    assert review_response.status_code == 400
    assert "cannot become official" in review_response.json()["detail"].lower()


def test_relation_with_decision_lineage_can_be_officialized_without_direct_evidence(client: TestClient):
    context_space_id = _context_space_id(client)
    reviewer_id = _reviewer_id(client, context_space_id)

    concepts = client.get("/api/v1/concepts", params={"context_space_id": context_space_id}).json()
    decisions = client.get("/api/v1/decisions", params={"context_space_id": context_space_id}).json()
    cornerstone_id = next(item["id"] for item in concepts if item["canonical_name"] == "Cornerstone")
    decision_record_id = next(item["id"] for item in concepts if item["canonical_name"] == "Decision Record")
    accepted_decision_id = decisions[0]["id"]

    create_response = client.post(
        "/api/v1/relations",
        json={
            "context_space_id": context_space_id,
            "subject_concept_id": decision_record_id,
            "predicate": "INFORMS",
            "object_concept_id": cornerstone_id,
            "description": "Decision Record informs Cornerstone.",
            "evidence_fragment_ids": [],
            "linked_decision_ids": [accepted_decision_id],
        },
    )
    assert create_response.status_code == 200
    relation_id = create_response.json()["id"]

    review_response = client.post(
        f"/api/v1/relations/{relation_id}/review",
        json={"actor_id": reviewer_id, "action": "approve"},
    )
    assert review_response.status_code == 200
    assert review_response.json()["status"] == "OFFICIAL"




def test_relation_with_only_proposed_decision_lineage_cannot_be_officialized(client: TestClient):
    context_space_id = _context_space_id(client)
    reviewer_id = _reviewer_id(client, context_space_id)

    concepts = client.get("/api/v1/concepts", params={"context_space_id": context_space_id}).json()
    cornerstone_id = next(item["id"] for item in concepts if item["canonical_name"] == "Cornerstone")
    decision_record_id = next(item["id"] for item in concepts if item["canonical_name"] == "Decision Record")

    create_decision_response = client.post(
        "/api/v1/decisions",
        json={
            "context_space_id": context_space_id,
            "title": "Proposed but not accepted",
            "problem": "Need a draft rationale only.",
            "decision": "Create a proposed decision.",
            "rationale": "Used to verify officialization rules.",
            "constraints": [],
            "impact": [],
            "evidence_fragment_ids": [],
            "linked_concept_ids": [],
            "linked_relation_ids": [],
        },
    )
    assert create_decision_response.status_code == 200
    proposed_decision_id = create_decision_response.json()["id"]

    create_relation_response = client.post(
        "/api/v1/relations",
        json={
            "context_space_id": context_space_id,
            "subject_concept_id": decision_record_id,
            "predicate": "DEPENDS_ON",
            "object_concept_id": cornerstone_id,
            "description": "A draft relation backed only by a proposed decision.",
            "evidence_fragment_ids": [],
            "linked_decision_ids": [proposed_decision_id],
        },
    )
    assert create_relation_response.status_code == 200
    relation_id = create_relation_response.json()["id"]

    review_response = client.post(
        f"/api/v1/relations/{relation_id}/review",
        json={"actor_id": reviewer_id, "action": "approve"},
    )
    assert review_response.status_code == 400
    assert "accepted decision lineage" in review_response.json()["detail"].lower()

def test_structured_answer_is_source_backed(client: TestClient):
    context_space_id = _context_space_id(client)
    response = client.get(
        "/api/v1/answers",
        params={"q": "Cornerstone", "context_space_id": context_space_id},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "Cornerstone"
    assert "summary" in data
    assert any(item["canonical_name"] == "Cornerstone" for item in data["concepts"])
    assert len(data["evidence"]) >= 1
    assert all("artifact_url" in item for item in data["evidence"])


def test_unstructured_artifact_answer_fallback_is_source_backed(unstructured_client: TestClient):
    context_space_id = _context_space_id(unstructured_client)
    response = unstructured_client.get(
        "/api/v1/answers",
        params={"q": "translation backend", "context_space_id": context_space_id},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["concepts"] == []
    assert data["relations"] == []
    assert data["decisions"] == []
    assert len(data["evidence"]) >= 1
    assert "source-backed evidence fragment" in data["summary"].lower()
    assert any("Cargo Translation Backend" in item["artifact_title"] for item in data["evidence"])
