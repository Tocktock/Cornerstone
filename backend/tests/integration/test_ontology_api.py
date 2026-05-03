from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def _create_concept(
    client: TestClient,
    *,
    name: str,
    evidence_id: str,
    aliases: list[str] | None = None,
    official: bool = True,
) -> str:
    response = client.post(
        "/v1/concepts",
        json={
            "name": name,
            "aliases": aliases or [],
            "shortDefinition": f"Official definition for {name}.",
            "evidenceFragmentIds": [evidence_id],
            "createdBy": "reviewer@example.com",
        },
    )
    assert response.status_code == 201
    concept_id = response.json()["id"]
    if official:
        official_response = client.post(
            f"/v1/concepts/{concept_id}/officialize",
            json={"reviewedBy": "reviewer@example.com"},
        )
        assert official_response.status_code == 200
    return concept_id


def test_ontology_search_finds_official_concept_by_alias(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    concept_id = _create_concept(
        client,
        name="Settlement",
        aliases=["payment settlement", "settlements"],
        evidence_id=synced_evidence["evidence_id"],
    )

    response = client.get("/v1/ontology/search", params={"q": "payment settlement"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "official"
    assert body["results"][0]["id"] == concept_id
    assert body["results"][0]["matchedBy"] == "alias"
    assert "payment settlement" in body["results"][0]["aliases"]


def test_ontology_graph_returns_depth_one_official_nodes_edges_and_citations(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    settlement_id = _create_concept(
        client,
        name="Settlement",
        aliases=["payment settlement"],
        evidence_id=synced_evidence["evidence_id"],
    )
    ledger_id = _create_concept(
        client,
        name="Ledger",
        evidence_id=synced_evidence["evidence_id"],
    )
    relation = client.post(
        "/v1/concept-relations",
        json={
            "sourceConceptId": settlement_id,
            "targetConceptId": ledger_id,
            "relationType": "updates",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    assert relation.status_code == 201
    relation_id = relation.json()["id"]
    official_relation = client.post(
        f"/v1/concept-relations/{relation_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert official_relation.status_code == 200

    response = client.get(
        "/v1/ontology/graph",
        params={"concept": "payment settlement", "depth": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trustLabel"] == "official"
    assert body["officialGraphAvailable"] is True
    assert body["focusConcept"]["id"] == settlement_id
    assert {node["id"] for node in body["nodes"]} == {settlement_id, ledger_id}
    assert body["edges"][0]["id"] == relation_id
    assert body["edges"][0]["relationType"] == "updates"
    assert body["evidence"][0]["evidenceFragmentId"] == synced_evidence["evidence_id"]
    support_types = {support["entityType"] for support in body["evidence"][0]["supports"]}
    assert {"concept", "concept_relation"} <= support_types


def test_official_ontology_graph_excludes_candidate_relations(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    settlement_id = _create_concept(client, name="Settlement", evidence_id=synced_evidence["evidence_id"])
    clearing_id = _create_concept(client, name="Clearing", evidence_id=synced_evidence["evidence_id"])
    candidate_relation = client.post(
        "/v1/concept-relations",
        json={
            "sourceConceptId": settlement_id,
            "targetConceptId": clearing_id,
            "relationType": "follows",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    assert candidate_relation.status_code == 201

    official_response = client.get("/v1/ontology/graph", params={"concept": "Settlement"})
    mixed_response = client.get("/v1/ontology/graph", params={"concept": "Settlement", "mode": "mixed"})

    assert official_response.status_code == 200
    assert official_response.json()["edges"] == []
    assert {node["id"] for node in official_response.json()["nodes"]} == {settlement_id}
    assert mixed_response.status_code == 200
    assert {edge["relationType"] for edge in mixed_response.json()["edges"]} == {"follows"}


def test_ontology_graph_unknown_concept_returns_unsupported_response(client: TestClient) -> None:
    response = client.get("/v1/ontology/graph", params={"concept": "Missing concept"})

    assert response.status_code == 200
    body = response.json()
    assert body["trustLabel"] == "unsupported"
    assert body["focusConcept"] is None
    assert body["nodes"] == []
    assert body["edges"] == []
    assert body["limitations"]


def test_ontology_graph_rejects_depth_above_one(client: TestClient) -> None:
    response = client.get("/v1/ontology/graph", params={"concept": "Settlement", "depth": 2})

    assert response.status_code == 422


def test_ontology_aliases_round_trip_in_sqlalchemy_store(persistent_client: TestClient) -> None:
    source_response = persistent_client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Persistent Pilot", "productionEnabled": True},
    )
    assert source_response.status_code == 201
    source_id = source_response.json()["id"]
    sync_response = persistent_client.post(
        f"/v1/manual-sources/{source_id}/sync",
        json={
            "objects": [
                {
                    "sourceExternalId": "doc-1",
                    "title": "Settlement Guide",
                    "content": "Settlement is completed after clearing. Settlement updates the ledger.",
                    "sourceUrl": "https://example.internal/settlement-guide",
                }
            ]
        },
    )
    assert sync_response.status_code == 200
    evidence_id = sync_response.json()["evidenceFragments"][0]["id"]
    review = persistent_client.post(
        f"/v1/evidence/{evidence_id}/review",
        json={"trustState": "reviewed", "reviewedBy": "reviewer@example.com"},
    )
    assert review.status_code == 200
    concept = persistent_client.post(
        "/v1/concepts",
        json={
            "name": "Settlement",
            "aliases": ["payment settlement"],
            "shortDefinition": "The process of finalizing a transaction obligation.",
            "evidenceFragmentIds": [evidence_id],
            "createdBy": "reviewer@example.com",
        },
    )
    assert concept.status_code == 201
    concept_id = concept.json()["id"]
    official = persistent_client.post(
        f"/v1/concepts/{concept_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert official.status_code == 200

    search = persistent_client.get("/v1/ontology/search", params={"q": "payment settlement"})

    assert search.status_code == 200
    assert search.json()["results"][0]["id"] == concept_id
    assert search.json()["results"][0]["matchedBy"] == "alias"
