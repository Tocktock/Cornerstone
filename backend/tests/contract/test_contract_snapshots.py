from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.parse import urlparse

from fastapi.testclient import TestClient

from ..mcp_testkit import mcp_session

GOLDEN_DIR = Path(__file__).with_name("golden")


def _normalize(value):
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, str) and value.startswith("file://"):
        return f"file://{Path(urlparse(value).path).name}"
    return value


def _load_golden(name: str):
    return json.loads((GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8"))


def test_canonical_contract_snapshots(client: TestClient, headers):
    concepts = client.get("/api/v1/concepts", headers=headers["member"])
    concepts.raise_for_status()
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
    decision = _normalize(
        next(
            item
            for item in decisions.json()
            if item["payload"]["title"] == "Legacy Escalation Routing"
        )
    )

    answer = _normalize(
        client.get("/api/v1/answers", headers=headers["member"], params={"q": "escalation"}).json()
    )
    search_results = _normalize(
        client.get("/api/v1/search", headers=headers["member"], params={"q": "playbook"}).json()
    )
    graph_slice = _normalize(
        client.get(
            "/api/v1/graph", headers=headers["member"], params={"root": "ops-playbook"}
        ).json()
    )

    vip = next(
        item
        for item in concepts.json()
        if item["payload"]["canonical_name"] == "VIP Escalation Insight"
    )
    provenance = _normalize(
        client.get(
            f"/api/v1/provenance/concept/{vip['payload']['concept_id']}", headers=headers["member"]
        ).json()
    )
    no_match = _normalize(
        client.get(
            "/api/v1/search", headers=headers["member"], params={"q": "totally-unmatched-query"}
        ).json()
    )

    assert concept == _load_golden("concept")
    assert relation == _load_golden("relation")
    assert decision == _load_golden("decision")
    assert answer == _load_golden("answer")
    assert search_results == _load_golden("search_results")
    assert graph_slice == _load_golden("graph_slice")
    assert provenance == _load_golden("provenance")
    assert no_match == _load_golden("no_match")


def test_mcp_answer_snapshot_matches_rest_answer(client: TestClient, headers):
    rest = _normalize(
        client.get("/api/v1/answers", headers=headers["member"], params={"q": "escalation"}).json()
    )
    
    async def fetch_mcp_answer():
        async with mcp_session(client.app, headers["member"]) as (session, _):
            return _normalize(
                (await session.call_tool("get_answer", {"query": "escalation"})).structuredContent
            )

    mcp = asyncio.run(fetch_mcp_answer())

    assert mcp == rest


def test_source_connection_contract_snapshots(client: TestClient, headers):
    created = client.post(
        "/api/v1/source-connections",
        headers=headers["operator"],
        json={
            "template_key": "notion_shared_page_tree",
            "source_label": "Snapshot Notion KB",
            "selected_scope_input": "11111111-1111-1111-1111-111111111111",
            "visibility_class": "member_visible",
            "sync_interval_seconds": 900,
        },
    )
    created.raise_for_status()
    connection_id = created.json()["id"]

    detail = _normalize(
        client.get(
            f"/api/v1/source-connections/{connection_id}",
            headers=headers["operator"],
        ).json()
    )
    runs = _normalize(
        client.get(
            f"/api/v1/source-connections/{connection_id}/runs",
            headers=headers["operator"],
        ).json()
    )

    assert detail == _load_golden("source_connection_detail")
    assert runs == _load_golden("source_connection_runs")
