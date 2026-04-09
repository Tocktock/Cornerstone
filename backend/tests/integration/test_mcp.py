from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from fastapi.testclient import TestClient

from ..mcp_testkit import mcp_session


def _normalize(value):
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, str) and value.startswith("file://"):
        return f"file://{urlparse(value).path.rsplit('/', maxsplit=1)[-1]}"
    return value


def test_mcp_tool_surface_is_read_only_and_discoverable(client: TestClient, headers):
    async def inspect_surface():
        async with mcp_session(client.app, headers["member"]) as (session, initialize_result):
            tools = await session.list_tools()
            return initialize_result, tools

    initialize_result, tools = asyncio.run(inspect_surface())

    assert [tool.name for tool in tools.tools] == [
        "search_context",
        "get_answer",
        "get_concept",
        "get_relation",
        "get_decision",
        "get_graph_slice",
        "follow_provenance",
    ]
    assert initialize_result.capabilities.tools is not None
    assert initialize_result.capabilities.prompts is None
    assert initialize_result.capabilities.resources is None


def test_mcp_contract_parity_matches_rest_surfaces(client: TestClient, headers):
    concept = _normalize(
        client.get("/api/v1/concepts/ops-playbook", headers=headers["member"]).json()
    )
    relations = client.get("/api/v1/relations", headers=headers["member"])
    relations.raise_for_status()
    relation_id = relations.json()[0]["payload"]["relation_id"]
    relation = _normalize(
        client.get(f"/api/v1/relations/{relation_id}", headers=headers["member"]).json()
    )
    decisions = client.get("/api/v1/decisions", headers=headers["member"])
    decisions.raise_for_status()
    decision_id = next(
        item["payload"]["decision_id"]
        for item in decisions.json()
        if item["payload"]["title"] == "Legacy Escalation Routing"
    )
    decision = _normalize(
        client.get(f"/api/v1/decisions/{decision_id}", headers=headers["member"]).json()
    )
    answer = _normalize(
        client.get("/api/v1/answers", headers=headers["member"], params={"q": "escalation"}).json()
    )
    search_results = _normalize(
        client.get("/api/v1/search", headers=headers["member"], params={"q": "playbook"}).json()
    )
    graph_slice = _normalize(
        client.get(
            "/api/v1/graph",
            headers=headers["member"],
            params={"root": "ops-playbook"},
        ).json()
    )
    concepts = client.get("/api/v1/concepts", headers=headers["member"])
    concepts.raise_for_status()
    vip_concept_id = next(
        item["payload"]["concept_id"]
        for item in concepts.json()
        if item["payload"]["canonical_name"] == "VIP Escalation Insight"
    )
    provenance = _normalize(
        client.get(
            f"/api/v1/provenance/concept/{vip_concept_id}",
            headers=headers["member"],
        ).json()
    )
    no_match = _normalize(
        client.get(
            "/api/v1/search",
            headers=headers["member"],
            params={"q": "totally-unmatched-query"},
        ).json()
    )

    async def collect_mcp_results():
        async with mcp_session(client.app, headers["member"]) as (session, _):
            return {
                "concept": _normalize(
                    (
                        await session.call_tool(
                            "get_concept",
                            {"resource_id": concept["payload"]["concept_id"]},
                        )
                    ).structuredContent
                ),
                "relation": _normalize(
                    (
                        await session.call_tool(
                            "get_relation",
                            {"resource_id": relation_id},
                        )
                    ).structuredContent
                ),
                "decision": _normalize(
                    (
                        await session.call_tool(
                            "get_decision",
                            {"resource_id": decision_id},
                        )
                    ).structuredContent
                ),
                "answer": _normalize(
                    (
                        await session.call_tool(
                            "get_answer",
                            {"query": "escalation"},
                        )
                    ).structuredContent
                ),
                "search_results": _normalize(
                    (
                        await session.call_tool(
                            "search_context",
                            {"query": "playbook"},
                        )
                    ).structuredContent
                ),
                "graph_slice": _normalize(
                    (
                        await session.call_tool(
                            "get_graph_slice",
                            {"root": "ops-playbook"},
                        )
                    ).structuredContent
                ),
                "provenance": _normalize(
                    (
                        await session.call_tool(
                            "follow_provenance",
                            {
                                "resource_kind": "concept",
                                "resource_id": vip_concept_id,
                            },
                        )
                    ).structuredContent
                ),
                "no_match": _normalize(
                    (
                        await session.call_tool(
                            "search_context",
                            {"query": "totally-unmatched-query"},
                        )
                    ).structuredContent
                ),
            }

    mcp_results = asyncio.run(collect_mcp_results())

    assert mcp_results["concept"] == concept
    assert mcp_results["relation"] == relation
    assert mcp_results["decision"] == decision
    assert mcp_results["answer"] == answer
    assert mcp_results["search_results"] == search_results
    assert mcp_results["graph_slice"] == graph_slice
    assert mcp_results["provenance"] == provenance
    assert mcp_results["no_match"] == no_match


def test_mcp_uses_actor_preferred_scope_when_consumer_scope_is_omitted(
    client: TestClient, headers
):
    rest = _normalize(
        client.get(
            "/api/v1/answers",
            headers=headers["reviewer"],
            params={"q": "escalation"},
        ).json()
    )

    async def call_without_scope():
        async with mcp_session(client.app, headers["reviewer"]) as (session, _):
            return _normalize(
                (await session.call_tool("get_answer", {"query": "escalation"})).structuredContent
            )

    mcp = asyncio.run(call_without_scope())

    assert mcp["consumer_scope"] == "review"
    assert mcp == rest


def test_mcp_rejects_missing_and_unknown_bearer_tokens(client: TestClient):
    missing = client.post("/mcp", json={})
    unknown = client.post("/mcp", headers={"Authorization": "Bearer not-a-real-token"}, json={})

    assert missing.status_code == 401
    assert missing.json()["detail"] == "Missing bearer token."
    assert unknown.status_code == 401
    assert unknown.json()["detail"] == "Unknown actor token."


def test_mcp_rejects_scope_escalation(client: TestClient, headers):
    async def attempt_scope_escalation():
        async with mcp_session(client.app, headers["member"]) as (session, _):
            return await session.call_tool(
                "get_answer",
                {"query": "escalation", "consumer_scope": "admin"},
            )

    result = asyncio.run(attempt_scope_escalation())

    assert result.isError is True
    assert any(
        "Requested consumer scope is not allowed." in getattr(item, "text", "")
        for item in result.content
    )
