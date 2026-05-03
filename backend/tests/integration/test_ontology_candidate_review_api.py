from __future__ import annotations

from fastapi.testclient import TestClient


def _create_reviewed_settlement_extraction(client: TestClient) -> dict[str, object]:
    source = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Candidate Review", "productionEnabled": True},
    )
    assert source.status_code == 201
    source_id = source.json()["id"]
    upload = client.post(
        f"/v1/manual-sources/{source_id}/uploads/text",
        json={
            "objects": [
                {
                    "title": "Settlement Review Notes",
                    "content": (
                        "Settlement is the process of finalizing obligations. "
                        "Clearing precedes settlement. "
                        "Reconciliation validates settlement."
                    ),
                }
            ]
        },
    )
    assert upload.status_code == 200
    artifact_id = upload.json()["artifacts"][0]["id"]
    evidence_fragments = upload.json()["evidenceFragments"]
    evidence_id = evidence_fragments[0]["id"]
    for fragment in evidence_fragments:
        review = client.post(
            f"/v1/evidence/{fragment['id']}/review",
            json={"trustState": "reviewed", "reviewedBy": "reviewer@example.com"},
        )
        assert review.status_code == 200
    extraction = client.post(
        "/v1/ontology/extraction-runs",
        json={"artifactIds": [artifact_id], "focusConcept": "settlement", "requestedBy": "reviewer@example.com"},
    )
    assert extraction.status_code == 201
    return {
        "source_id": source_id,
        "artifact_id": artifact_id,
        "evidence_id": evidence_id,
        "run": extraction.json()["run"],
        "concept_candidates": extraction.json()["conceptCandidates"],
        "relation_candidates": extraction.json()["relationCandidates"],
    }


def _candidate_by_name(candidates: list[dict[str, object]], name: str) -> dict[str, object]:
    return next(candidate for candidate in candidates if candidate["name"] == name)


def _relation_candidate(
    candidates: list[dict[str, object]],
    source: str,
    relation_type: str,
    target: str,
) -> dict[str, object]:
    return next(
        candidate
        for candidate in candidates
        if candidate["sourceName"] == source
        and candidate["relationType"] == relation_type
        and candidate["targetName"] == target
    )


def _approve_concept_candidate(client: TestClient, candidate_id: str, **overrides: object) -> dict[str, object]:
    payload = {"reviewedBy": "reviewer@example.com"}
    payload.update(overrides)
    response = client.post(f"/v1/ontology/concept-candidates/{candidate_id}/approve", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_concept_candidate_approval_promotes_to_official_concept(client: TestClient) -> None:
    data = _create_reviewed_settlement_extraction(client)
    settlement = _candidate_by_name(data["concept_candidates"], "Settlement")

    response = client.post(
        f"/v1/ontology/concept-candidates/{settlement['id']}/approve",
        json={
            "reviewedBy": "reviewer@example.com",
            "aliases": ["payment settlement"],
            "reviewNote": "Definition is supported by reviewed manual evidence.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["candidate"]["status"] == "approved"
    assert body["candidate"]["promotedConceptId"] == body["concept"]["id"]
    assert body["concept"]["status"] == "official"
    assert "payment settlement" in body["concept"]["aliases"]
    assert body["auditEventIds"]

    search = client.get("/v1/ontology/search", params={"q": "payment settlement"})
    assert search.status_code == 200
    assert search.json()["results"][0]["id"] == body["concept"]["id"]


def test_relation_candidate_approval_uses_approved_endpoint_concepts_and_updates_graph(client: TestClient) -> None:
    data = _create_reviewed_settlement_extraction(client)
    concept_candidates = data["concept_candidates"]
    settlement = _approve_concept_candidate(client, str(_candidate_by_name(concept_candidates, "Settlement")["id"]))
    clearing = _approve_concept_candidate(client, str(_candidate_by_name(concept_candidates, "Clearing")["id"]))
    relation = _relation_candidate(data["relation_candidates"], "Clearing", "precedes", "Settlement")

    approval = client.post(
        f"/v1/ontology/relation-candidates/{relation['id']}/approve",
        json={"reviewedBy": "reviewer@example.com"},
    )

    assert approval.status_code == 200, approval.text
    body = approval.json()
    assert body["candidate"]["status"] == "approved"
    assert body["candidate"]["promotedRelationId"] == body["relation"]["id"]
    assert body["relation"]["status"] == "official"
    assert body["relation"]["sourceConceptId"] == clearing["concept"]["id"]
    assert body["relation"]["targetConceptId"] == settlement["concept"]["id"]

    graph = client.get("/v1/ontology/graph", params={"concept": "Settlement"})
    assert graph.status_code == 200
    graph_body = graph.json()
    assert graph_body["officialGraphAvailable"] is True
    assert {node["id"] for node in graph_body["nodes"]} == {settlement["concept"]["id"], clearing["concept"]["id"]}
    assert graph_body["edges"][0]["id"] == body["relation"]["id"]
    assert graph_body["edges"][0]["relationType"] == "precedes"


def test_candidate_rejection_blocks_later_approval(client: TestClient) -> None:
    data = _create_reviewed_settlement_extraction(client)
    settlement = _candidate_by_name(data["concept_candidates"], "Settlement")

    reject = client.post(
        f"/v1/ontology/concept-candidates/{settlement['id']}/reject",
        json={"reviewedBy": "reviewer@example.com", "reviewNote": "Wrong scope."},
    )
    assert reject.status_code == 200
    assert reject.json()["candidate"]["status"] == "rejected"

    approve = client.post(
        f"/v1/ontology/concept-candidates/{settlement['id']}/approve",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert approve.status_code == 409


def test_concept_candidate_edit_then_approve(client: TestClient) -> None:
    data = _create_reviewed_settlement_extraction(client)
    settlement = _candidate_by_name(data["concept_candidates"], "Settlement")

    edit = client.patch(
        f"/v1/ontology/concept-candidates/{settlement['id']}",
        json={
            "editedBy": "reviewer@example.com",
            "name": "Settlement Process",
            "aliases": ["settlement"],
            "proposedDefinition": "The reviewed process of finalizing obligations.",
        },
    )
    assert edit.status_code == 200
    assert edit.json()["candidate"]["name"] == "Settlement Process"

    approval = client.post(
        f"/v1/ontology/concept-candidates/{settlement['id']}/approve",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert approval.status_code == 200
    assert approval.json()["concept"]["name"] == "Settlement Process"
    assert approval.json()["concept"]["shortDefinition"] == "The reviewed process of finalizing obligations."


def test_concept_candidate_merge_into_existing_official_concept(client: TestClient, synced_evidence: dict[str, str]) -> None:
    concept = client.post(
        "/v1/concepts",
        json={
            "name": "Settlement",
            "shortDefinition": "Existing official settlement definition.",
            "evidenceFragmentIds": [synced_evidence["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    assert concept.status_code == 201
    concept_id = concept.json()["id"]
    official = client.post(f"/v1/concepts/{concept_id}/officialize", json={"reviewedBy": "reviewer@example.com"})
    assert official.status_code == 200

    data = _create_reviewed_settlement_extraction(client)
    settlement = _candidate_by_name(data["concept_candidates"], "Settlement")

    approval = client.post(
        f"/v1/ontology/concept-candidates/{settlement['id']}/approve",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert approval.status_code == 409

    merge = client.post(
        f"/v1/ontology/concept-candidates/{settlement['id']}/merge",
        json={
            "reviewedBy": "reviewer@example.com",
            "targetConceptId": concept_id,
            "aliases": ["payment settlement"],
        },
    )
    assert merge.status_code == 200, merge.text
    body = merge.json()
    assert body["candidate"]["status"] == "merged"
    assert body["candidate"]["mergedIntoConceptId"] == concept_id
    assert "payment settlement" in body["concept"]["aliases"]
    assert body["concept"]["status"] == "official"


def test_candidate_review_requires_authorized_reviewer(client: TestClient) -> None:
    data = _create_reviewed_settlement_extraction(client)
    settlement = _candidate_by_name(data["concept_candidates"], "Settlement")

    response = client.post(
        f"/v1/ontology/concept-candidates/{settlement['id']}/approve",
        json={"reviewedBy": "intruder@example.com"},
    )

    assert response.status_code == 403


def test_review_queue_summary_groups_pending_candidates_and_supports_filters(client: TestClient) -> None:
    data = _create_reviewed_settlement_extraction(client)
    run_id = data["run"]["id"]
    source_id = data["source_id"]

    response = client.get(
        "/v1/ontology/review-queue/summary",
        params={"runId": run_id, "sourceId": source_id, "status": "pending", "minConfidence": 0.7},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["totalPendingConceptCandidates"] >= 1
    assert body["totalPendingRelationCandidates"] >= 1
    assert body["countsByRun"][run_id] >= 1
    assert body["countsBySource"][source_id] >= 1
    assert body["countsByStatus"]["pending"] >= 1
    assert "preview_candidate_action" in body["machineReadableNextActions"]
    groups = {group["focusConcept"]: group for group in body["groupedByFocusConcept"]}
    assert "Settlement" in groups
    assert groups["Settlement"]["pendingConceptCandidateCount"] >= 1
    assert groups["Settlement"]["pendingRelationCandidateCount"] >= 1
    assert "highConfidenceCount" in groups["Settlement"]
    assert "lowConfidenceCount" in groups["Settlement"]


def test_concept_candidate_preview_reports_apply_boundary_and_duplicate_blocker(client: TestClient) -> None:
    data = _create_reviewed_settlement_extraction(client)
    settlement = _candidate_by_name(data["concept_candidates"], "Settlement")

    preview = client.get(
        f"/v1/ontology/concept-candidates/{settlement['id']}/preview",
        params={"action": "approve"},
    )
    assert preview.status_code == 200
    body = preview.json()
    assert body["candidateType"] == "concept"
    assert body["action"] == "approve"
    assert body["canApply"] is True
    assert body["officialGraphWillChange"] is True
    assert body["evidencePreserved"] is True
    assert body["blockerReasons"] == []

    _approve_concept_candidate(client, str(settlement["id"]))
    other_data = _create_reviewed_settlement_extraction(client)
    duplicate = _candidate_by_name(other_data["concept_candidates"], "Settlement")
    duplicate_preview = client.get(
        f"/v1/ontology/concept-candidates/{duplicate['id']}/preview",
        params={"action": "approve"},
    )
    assert duplicate_preview.status_code == 200
    duplicate_body = duplicate_preview.json()
    assert duplicate_body["canApply"] is False
    assert duplicate_body["officialGraphWillChange"] is False
    assert {blocker["code"] for blocker in duplicate_body["blockerReasons"]} == {"duplicate_concept"}


def test_relation_candidate_preview_blocks_until_endpoints_are_official(client: TestClient) -> None:
    data = _create_reviewed_settlement_extraction(client)
    relation = _relation_candidate(data["relation_candidates"], "Clearing", "precedes", "Settlement")

    preview = client.get(
        f"/v1/ontology/relation-candidates/{relation['id']}/preview",
        params={"action": "approve"},
    )

    assert preview.status_code == 200
    body = preview.json()
    assert body["candidateType"] == "relation"
    assert body["canApply"] is False
    assert body["officialGraphWillChange"] is False
    assert {blocker["code"] for blocker in body["blockerReasons"]} == {
        "source_concept_unresolved",
        "target_concept_unresolved",
    }
