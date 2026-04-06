from __future__ import annotations

from fastapi.testclient import TestClient


def test_source_statuses_cover_symptom_states(client: TestClient, headers):
    response = client.get("/api/v1/source-connections", headers=headers["member"])
    response.raise_for_status()
    payload = response.json()

    source_states = {item["source_connection_state"] for item in payload}
    freshness_states = {item["freshness_state"] for item in payload}

    assert {"active", "degraded", "paused", "removed"} <= source_states
    assert {"current", "stale", "unknown"} <= freshness_states


def test_member_retrieval_hides_evidence_only_support_but_discloses_restriction(
    client: TestClient, headers
):
    response = client.get("/api/v1/concepts", headers=headers["member"])
    response.raise_for_status()

    restricted = next(
        item
        for item in response.json()
        if item["payload"]["canonical_name"] == "Private Escalation Trigger"
    )

    assert restricted["payload"]["support_visibility"] == "restricted_support"
    assert restricted["payload"]["visible_support_items"] == []
    assert restricted["warnings"] == ["restricted_support"]


def test_cross_domain_relation_requires_workspace_review(client: TestClient, headers):
    queue = client.get("/api/v1/review-queue", headers=headers["reviewer"])
    queue.raise_for_status()
    relation = next(
        item for item in queue.json() if item["resource_ref"]["resource_kind"] == "relation"
    )

    response = client.post(
        f"/api/v1/relations/{relation['resource_ref']['resource_id']}/review",
        headers=headers["reviewer"],
        json={"action": "officialize"},
    )

    assert response.status_code == 400
    assert "workspace" in response.json()["detail"]


def test_superseded_decisions_remain_readable_with_lineage(client: TestClient, headers):
    response = client.get("/api/v1/decisions", headers=headers["member"])
    response.raise_for_status()

    legacy = next(
        item for item in response.json() if item["payload"]["title"] == "Legacy Escalation Routing"
    )

    assert legacy["payload"]["lifecycle_state"] == "superseded"
    assert (
        legacy["payload"]["superseded_by_ref"]["resource_label"] == "Risk-Based Escalation Routing"
    )


def test_promoted_support_enters_workspace_without_private_origin_leakage(
    client: TestClient, headers
):
    concepts = client.get("/api/v1/concepts", headers=headers["member"])
    concepts.raise_for_status()
    promoted = next(
        item
        for item in concepts.json()
        if item["payload"]["canonical_name"] == "VIP Escalation Insight"
    )

    provenance = client.get(
        f"/api/v1/provenance/concept/{promoted['payload']['concept_id']}",
        headers=headers["member"],
    )
    provenance.raise_for_status()

    support_items = provenance.json()["payload"]["support_items"]
    assert any(item["support_item_kind"] == "promoted_support" for item in support_items)
    assert all(
        (item.get("source_locator") or "").startswith("workspace-promotion:")
        for item in support_items
    )
    assert "private:" not in str(provenance.json())


def test_no_match_returns_canonical_reason(client: TestClient, headers):
    response = client.get(
        "/api/v1/search", headers=headers["member"], params={"q": "totally-unmatched-query"}
    )
    response.raise_for_status()

    assert response.json()["response_kind"] == "no_match"
    assert response.json()["payload"]["reason"] == "no_official_match"


def test_rest_and_mcp_answer_surfaces_match_contract_semantics(client: TestClient, headers):
    rest = client.get("/api/v1/answers", headers=headers["member"], params={"q": "escalation"})
    rest.raise_for_status()
    mcp = client.post(
        "/api/v1/mcp/read",
        headers=headers["member"],
        json={"request_intent": "get_answer", "query": "escalation"},
    )
    mcp.raise_for_status()

    assert rest.json()["response_kind"] == mcp.json()["response_kind"] == "answer"
    assert (
        rest.json()["payload"]["support_visibility"] == mcp.json()["payload"]["support_visibility"]
    )
    assert (
        rest.json()["payload"]["verification_state"] == mcp.json()["payload"]["verification_state"]
    )
