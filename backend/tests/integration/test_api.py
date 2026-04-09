from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import httpx
from fastapi.testclient import TestClient


def test_source_statuses_cover_symptom_states(client: TestClient, headers):
    response = client.get("/api/v1/source-connections", headers=headers["member"])
    response.raise_for_status()
    payload = response.json()

    source_states = {item["source_connection_state"] for item in payload}
    freshness_states = {item["freshness_state"] for item in payload}

    assert {"active", "degraded", "paused", "removed"} <= source_states
    assert {"current", "stale", "unknown"} <= freshness_states


def test_connector_manager_can_complete_mocked_notion_oauth_and_sync_live_database(
    client: TestClient,
    headers,
    monkeypatch,
):
    settings = client.app.state.settings
    monkeypatch.setattr(settings, "notion_demo_oauth_mode", False)
    monkeypatch.setattr(settings, "notion_client_id", "notion-client-id")
    monkeypatch.setattr(settings, "notion_client_secret", "notion-client-secret")
    monkeypatch.setattr(settings, "notion_oauth_redirect_uri", "http://localhost:4173/sources")

    database_id = "77777777-7777-7777-7777-777777777777"
    data_source_id = "88888888-8888-8888-8888-888888888888"
    request_log: list[tuple[str, str, dict | None]] = []

    class MockResponse:
        def __init__(self, payload: dict, status_code: int = 200):
            self._payload = payload
            self.status_code = status_code
            self.request = httpx.Request("GET", "https://mock.notion.local")

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "Mock Notion request failed",
                    request=self.request,
                    response=httpx.Response(self.status_code, request=self.request),
                )

    def fake_post(url, *, auth, headers, json, timeout):
        request_log.append(("POST", url, json))
        assert url == settings.notion_oauth_token_url
        assert auth == (settings.notion_client_id, settings.notion_client_secret)
        assert json["grant_type"] == "authorization_code"
        return MockResponse(
            {
                "access_token": "oauth-access-token",
                "token_type": "bearer",
                "refresh_token": "oauth-refresh-token",
                "bot_id": "99999999-9999-9999-9999-999999999999",
                "workspace_icon": None,
                "workspace_name": "Mocked Notion Workspace",
                "workspace_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "owner": {"type": "workspace", "workspace": True},
                "duplicated_template_id": None,
                "request_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            }
        )

    def fake_request(method, url, *, headers, json=None, timeout):
        request_log.append((method, url, json))
        assert headers["Authorization"] == "Bearer oauth-access-token"
        assert headers["Notion-Version"] == settings.notion_version
        parsed = urlparse(url)
        if method == "GET" and parsed.path == f"/v1/databases/{database_id}":
            return MockResponse(
                {
                    "object": "database",
                    "id": database_id,
                    "title": [{"type": "text", "plain_text": "Engineering Runbooks"}],
                    "url": "https://www.notion.so/engineering-runbooks",
                    "data_sources": [{"id": data_source_id, "name": "Engineering Runbooks"}],
                }
            )
        if method == "POST" and parsed.path == f"/v1/data_sources/{data_source_id}/query":
            return MockResponse(
                {
                    "object": "list",
                    "results": [
                        {
                            "object": "page",
                            "id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                            "url": "https://www.notion.so/runbook-item",
                            "last_edited_time": "2026-04-06T00:00:00Z",
                            "properties": {
                                "Name": {
                                    "type": "title",
                                    "title": [{"plain_text": "Runbook Item"}],
                                },
                                "Status": {
                                    "type": "status",
                                    "status": {"name": "Published"},
                                },
                                "Owner": {
                                    "type": "people",
                                    "people": [{"name": "Tars"}],
                                },
                            },
                        }
                    ],
                    "has_more": False,
                    "next_cursor": None,
                }
            )
        raise AssertionError(f"Unexpected mocked Notion request: {method} {url}")

    monkeypatch.setattr("cornerstone.connectors.notion.httpx.post", fake_post)
    monkeypatch.setattr("cornerstone.connectors.notion.httpx.request", fake_request)

    start = client.post("/api/v1/provider-bindings/notion/start", headers=headers["operator"])
    start.raise_for_status()
    start_payload = start.json()
    query = parse_qs(urlparse(start_payload["authorization_url"]).query)

    assert start_payload["demo_mode"] is False
    assert query["client_id"] == ["notion-client-id"]
    assert query["owner"] == ["user"]
    assert query["state"] == [start_payload["binding_state"]]

    complete = client.post(
        "/api/v1/provider-bindings/notion/complete",
        headers=headers["operator"],
        json={
            "provider": "notion",
            "binding_state": start_payload["binding_state"],
            "code": "temporary-oauth-code",
        },
    )
    complete.raise_for_status()
    complete_payload = complete.json()

    assert complete_payload["account_label"] == "Mocked Notion Workspace"
    assert complete_payload["provider_credential_ref"] == start_payload["provider_credential_ref"]

    created = client.post(
        "/api/v1/source-connections",
        headers=headers["operator"],
        json={
            "template_key": "notion_shared_database",
            "provider_credential_ref": complete_payload["provider_credential_ref"],
            "source_label": "Live Notion Database",
            "selected_scope_input": database_id,
            "visibility_class": "member_visible",
            "sync_interval_seconds": 900,
        },
    )
    created.raise_for_status()
    created_payload = created.json()

    assert created_payload["provider"] == "notion"
    assert created_payload["provider_credential_ref"] == complete_payload["provider_credential_ref"]
    assert created_payload["selected_scope_json"]["object_id"] == database_id

    runs = client.get(
        f"/api/v1/source-connections/{created_payload['id']}/runs",
        headers=headers["operator"],
    )
    runs.raise_for_status()

    assert runs.json()[0]["trigger_kind"] == "initial"
    assert runs.json()[0]["run_status"] == "succeeded"
    assert runs.json()[0]["artifact_count"] == 1
    assert any(url.endswith("/oauth/token") for _, url, _ in request_log)
    assert any(url.endswith(f"/databases/{database_id}") for _, url, _ in request_log)
    assert any(url.endswith(f"/data_sources/{data_source_id}/query") for _, url, _ in request_log)


def test_connector_manager_can_bind_preview_create_and_manage_notion_connection(
    client: TestClient, headers
):
    binding = client.post("/api/v1/provider-bindings/notion/start", headers=headers["operator"])
    binding.raise_for_status()
    binding_payload = binding.json()

    preview = client.post(
        "/api/v1/source-connections/preview",
        headers=headers["operator"],
        json={
            "template_key": "notion_shared_page_tree",
            "provider_credential_ref": binding_payload["provider_credential_ref"],
            "source_label": "Notion knowledge base",
            "selected_scope_input": "11111111-1111-1111-1111-111111111111",
            "visibility_class": "member_visible",
        },
    )
    preview.raise_for_status()
    assert preview.json()["resolved_source_boundary_locator"].startswith("notion://page-tree/")

    created = client.post(
        "/api/v1/source-connections",
        headers=headers["operator"],
        json={
            "template_key": "notion_shared_page_tree",
            "provider_credential_ref": binding_payload["provider_credential_ref"],
            "source_label": "Notion knowledge base",
            "selected_scope_input": "11111111-1111-1111-1111-111111111111",
            "visibility_class": "member_visible",
            "sync_interval_seconds": 900,
        },
    )
    created.raise_for_status()
    connection = created.json()

    assert connection["provider"] == "notion"
    assert connection["template_key"] == "notion_shared_page_tree"
    assert connection["can_manage"] is True
    assert connection["provider_credential_ref"] == binding_payload["provider_credential_ref"]

    detail = client.get(
        f"/api/v1/source-connections/{connection['id']}",
        headers=headers["operator"],
    )
    detail.raise_for_status()
    assert (
        detail.json()["selected_scope_json"]["object_id"]
        == "11111111-1111-1111-1111-111111111111"
    )

    runs = client.get(
        f"/api/v1/source-connections/{connection['id']}/runs",
        headers=headers["operator"],
    )
    runs.raise_for_status()
    assert runs.json()[0]["trigger_kind"] == "initial"

    paused = client.post(
        f"/api/v1/source-connections/{connection['id']}/pause",
        headers=headers["operator"],
    )
    paused.raise_for_status()
    assert paused.json()["source_connection_state"] == "paused"

    resumed = client.post(
        f"/api/v1/source-connections/{connection['id']}/resume",
        headers=headers["operator"],
    )
    resumed.raise_for_status()
    assert resumed.json()["source_connection_state"] == "active"

    synced = client.post(
        f"/api/v1/source-connections/{connection['id']}/sync",
        headers=headers["operator"],
    )
    synced.raise_for_status()
    assert synced.json()["source_connection_state"] == "active"

    removed = client.delete(
        f"/api/v1/source-connections/{connection['id']}",
        headers=headers["operator"],
    )
    removed.raise_for_status()
    assert removed.json()["source_connection_state"] == "removed"


def test_member_cannot_manage_connector_routes(client: TestClient, headers):
    templates = client.get("/api/v1/connector-templates", headers=headers["member"])
    assert templates.status_code == 403

    preview = client.post(
        "/api/v1/source-connections/preview",
        headers=headers["member"],
        json={
            "template_key": "notion_shared_page_tree",
            "source_label": "Forbidden preview",
            "selected_scope_input": "11111111-1111-1111-1111-111111111111",
            "visibility_class": "member_visible",
        },
    )
    assert preview.status_code == 403

    sources = client.get("/api/v1/source-connections", headers=headers["member"])
    sources.raise_for_status()
    assert all(item["can_manage"] is False for item in sources.json())


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

