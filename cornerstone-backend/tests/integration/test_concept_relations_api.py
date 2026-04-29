from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def _official_concept(client: TestClient, name: str, evidence_id: str) -> str:
    concept_response = client.post(
        "/v1/concepts",
        json={
            "name": name,
            "shortDefinition": f"Official definition for {name}.",
            "evidenceFragmentIds": [evidence_id],
            "createdBy": "reviewer@example.com",
        },
    )
    assert concept_response.status_code == 201
    concept_id = concept_response.json()["id"]
    official_response = client.post(
        f"/v1/concepts/{concept_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert official_response.status_code == 200
    return concept_id


def test_relation_creation_and_officialization_requires_reviewed_support(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    source_concept_id = _official_concept(client, "Cornerstone", synced_evidence["evidence_id"])
    target_concept_id = _official_concept(client, "Evidence Layer", synced_evidence["evidence_id"])

    create_response = client.post(
        "/v1/concept-relations",
        json={
            "sourceConceptId": source_concept_id,
            "targetConceptId": target_concept_id,
            "relationType": "depends_on",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    assert create_response.status_code == 201
    relation_id = create_response.json()["id"]

    officialize_response = client.post(
        f"/v1/concept-relations/{relation_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert officialize_response.status_code == 200
    assert officialize_response.json()["status"] == "official"
    assert officialize_response.json()["lastReviewedAt"] is not None


def test_relation_officialization_without_support_is_blocked(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    source_concept_id = _official_concept(client, "Source Concept", synced_evidence["evidence_id"])
    target_concept_id = _official_concept(client, "Target Concept", synced_evidence["evidence_id"])
    create_response = client.post(
        "/v1/concept-relations",
        json={
            "sourceConceptId": source_concept_id,
            "targetConceptId": target_concept_id,
            "relationType": "depends_on",
            "createdBy": "reviewer@example.com",
        },
    )
    assert create_response.status_code == 201

    officialize_response = client.post(
        f"/v1/concept-relations/{create_response.json()['id']}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )

    assert officialize_response.status_code == 409
    assert "ConceptRelations require" in officialize_response.json()["detail"]


def test_relation_officialization_requires_both_concepts_official(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    source_concept_id = _official_concept(client, "Official Source", synced_evidence["evidence_id"])
    target_response = client.post(
        "/v1/concepts",
        json={
            "name": "Candidate Target",
            "shortDefinition": "Not official yet.",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    assert target_response.status_code == 201
    create_response = client.post(
        "/v1/concept-relations",
        json={
            "sourceConceptId": source_concept_id,
            "targetConceptId": target_response.json()["id"],
            "relationType": "depends_on",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    assert create_response.status_code == 201

    officialize_response = client.post(
        f"/v1/concept-relations/{create_response.json()['id']}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )

    assert officialize_response.status_code == 409
    assert "both Concepts to be official" in officialize_response.json()["detail"]


def test_unauthorized_reviewer_cannot_create_or_officialize_relation(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    source_concept_id = _official_concept(client, "Authorized Source", synced_evidence["evidence_id"])
    target_concept_id = _official_concept(client, "Authorized Target", synced_evidence["evidence_id"])
    create_response = client.post(
        "/v1/concept-relations",
        json={
            "sourceConceptId": source_concept_id,
            "targetConceptId": target_concept_id,
            "relationType": "depends_on",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "random@example.com",
        },
    )
    assert create_response.status_code == 403

    valid_create = client.post(
        "/v1/concept-relations",
        json={
            "sourceConceptId": source_concept_id,
            "targetConceptId": target_concept_id,
            "relationType": "depends_on",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    assert valid_create.status_code == 201
    blocked_officialize = client.post(
        f"/v1/concept-relations/{valid_create.json()['id']}/officialize",
        json={"reviewedBy": "random@example.com"},
    )
    assert blocked_officialize.status_code == 403


def test_relation_route_reports_missing_relation_evidence_and_decision_record(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    source_concept_id = _official_concept(client, "Missing Source", synced_evidence["evidence_id"])
    target_concept_id = _official_concept(client, "Missing Target", synced_evidence["evidence_id"])

    assert client.get("/v1/concept-relations/missing-relation-id").status_code == 404

    missing_evidence = client.post(
        "/v1/concept-relations",
        json={
            "sourceConceptId": source_concept_id,
            "targetConceptId": target_concept_id,
            "relationType": "depends_on",
            "evidenceFragmentIds": ["missing-evidence-id"],
            "createdBy": "reviewer@example.com",
        },
    )
    assert missing_evidence.status_code == 404
    assert "EvidenceFragment not found" in missing_evidence.json()["detail"]

    missing_decision = client.post(
        "/v1/concept-relations",
        json={
            "sourceConceptId": source_concept_id,
            "targetConceptId": target_concept_id,
            "relationType": "depends_on",
            "decisionRecordId": "missing-decision-id",
            "createdBy": "reviewer@example.com",
        },
    )
    assert missing_decision.status_code == 404
    assert "DecisionRecord not found" in missing_decision.json()["detail"]


def test_relation_can_be_officialized_with_valid_decision_record_support(
    client: TestClient,
    synced_evidence: dict[str, str],
) -> None:
    source_concept_id = _official_concept(client, "Decision Source", synced_evidence["evidence_id"])
    target_concept_id = _official_concept(client, "Decision Target", synced_evidence["evidence_id"])
    decision = client.post(
        "/v1/decision-records",
        json={
            "title": "Relation decision",
            "decision": "Decision Source depends on Decision Target.",
            "reason": "Reviewed source evidence supports the relation.",
            "decidedBy": "reviewer@example.com",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "affectedConceptIds": [source_concept_id, target_concept_id],
        },
    )
    assert decision.status_code == 201
    relation = client.post(
        "/v1/concept-relations",
        json={
            "sourceConceptId": source_concept_id,
            "targetConceptId": target_concept_id,
            "relationType": "governed_by",
            "decisionRecordId": decision.json()["id"],
            "createdBy": "reviewer@example.com",
        },
    )
    assert relation.status_code == 201

    official = client.post(
        f"/v1/concept-relations/{relation.json()['id']}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )

    assert official.status_code == 200
    assert official.json()["status"] == "official"
    assert official.json()["decisionRecordId"] == decision.json()["id"]
