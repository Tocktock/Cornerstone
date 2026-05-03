from __future__ import annotations

from fastapi.testclient import TestClient


def test_unknown_query_returns_unsupported(client: TestClient) -> None:
    response = client.get("/v1/context/query", params={"q": "What is MissingThing?"})

    assert response.status_code == 200
    body = response.json()
    assert body["trustLabel"] == "unsupported"
    assert body["evidence"] == []
    assert body["limitations"]


def test_official_concept_query_returns_citation(
    client: TestClient, synced_evidence: dict[str, str]
) -> None:
    concept_response = client.post(
        "/v1/concepts",
        json={
            "name": "Cornerstone",
            "shortDefinition": "Shared organizational context layer.",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    concept_id = concept_response.json()["id"]
    client.post(
        f"/v1/concepts/{concept_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )

    response = client.get("/v1/context/query", params={"q": "What is Cornerstone?"})

    assert response.status_code == 200
    body = response.json()
    assert body["trustLabel"] == "official"
    assert body["concepts"][0]["name"] == "Cornerstone"
    assert body["evidence"][0]["evidenceFragmentId"] == synced_evidence["evidence_id"]
    assert body["freshness"]["state"] == "fresh"



def test_reviewed_evidence_without_concept_returns_evidence_supported(
    client: TestClient, synced_evidence: dict[str, str]
) -> None:
    evidence_response = client.get("/v1/evidence")
    assert evidence_response.status_code == 200
    for fragment in evidence_response.json():
        client.post(
            f"/v1/evidence/{fragment['id']}/review",
            json={"trustState": "reviewed", "reviewedBy": "reviewer@example.com"},
        )

    response = client.get("/v1/context/query", params={"q": "organizational context"})

    assert response.status_code == 200
    body = response.json()
    assert body["trustLabel"] == "evidence_supported"
    assert body["officialAnswerAvailable"] is False
    assert body["concepts"] == []
    assert body["evidence"][0]["evidenceFragmentId"] == synced_evidence["evidence_id"]
    assert body["evidence"][0]["supports"][0]["entityType"] == "evidence_fragment"


def test_grounded_response_includes_relation_and_support_metadata(
    client: TestClient, synced_evidence: dict[str, str]
) -> None:
    cornerstone_response = client.post(
        "/v1/concepts",
        json={
            "name": "Cornerstone",
            "shortDefinition": "Shared organizational context layer.",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    source_studio_response = client.post(
        "/v1/concepts",
        json={
            "name": "Source Studio",
            "shortDefinition": "Connector administration surface.",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    cornerstone_id = cornerstone_response.json()["id"]
    source_studio_id = source_studio_response.json()["id"]
    client.post(
        f"/v1/concepts/{cornerstone_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    client.post(
        f"/v1/concepts/{source_studio_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    relation_response = client.post(
        "/v1/concept-relations",
        json={
            "sourceConceptId": cornerstone_id,
            "targetConceptId": source_studio_id,
            "relationType": "depends_on",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    relation_id = relation_response.json()["id"]
    client.post(
        f"/v1/concept-relations/{relation_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )

    response = client.get("/v1/context/query", params={"q": "Cornerstone"})

    assert response.status_code == 200
    body = response.json()
    assert body["trustLabel"] == "official"
    assert body["officialAnswerAvailable"] is True
    assert body["relations"][0]["id"] == relation_id
    support_types = {support["entityType"] for support in body["evidence"][0]["supports"]}
    assert {"concept", "concept_relation"} <= support_types
    assert body["evidence"][0]["isValid"] is True
    assert body["evidence"][0]["validityErrors"] == []

def test_official_context_survives_degraded_source_with_limitation(
    client: TestClient, store, synced_evidence: dict[str, str]
) -> None:
    from cornerstone.schemas import DataSourceStatus, DataSourceSyncStatus, ErrorInfo

    concept_response = client.post(
        "/v1/concepts",
        json={
            "name": "Cornerstone",
            "shortDefinition": "Shared organizational context layer.",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    concept_id = concept_response.json()["id"]
    client.post(
        f"/v1/concepts/{concept_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    source = store.get_data_source(synced_evidence["source_id"])
    store.update_data_source(
        source.model_copy(
            update={
                "status": DataSourceStatus.DEGRADED,
                "sync_status": DataSourceSyncStatus.DEGRADED,
                "last_error": ErrorInfo(code="provider_unavailable", message="Provider temporarily unavailable."),
            },
            deep=True,
        )
    )

    response = client.get("/v1/context/query", params={"q": "What is Cornerstone?"})

    assert response.status_code == 200
    body = response.json()
    assert body["trustLabel"] == "official"
    assert body["evidence"][0]["evidenceFragmentId"] == synced_evidence["evidence_id"]
    assert any("degraded" in limitation for limitation in body["limitations"])
    assert any("latest sync or connection error" in limitation for limitation in body["limitations"])
