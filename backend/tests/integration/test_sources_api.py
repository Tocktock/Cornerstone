from __future__ import annotations

from fastapi.testclient import TestClient


def test_empty_production_state_has_no_demo_context(client: TestClient) -> None:
    response = client.get("/v1/sources")

    assert response.status_code == 200
    body = response.json()
    assert body["productionEnabled"] is True
    assert body["hasRealSources"] is False
    assert body["onboardingRequired"] is True
    assert body["sources"] == []


def test_provider_backed_source_direct_creation_is_rejected(client: TestClient) -> None:
    create_response = client.post(
        "/v1/sources",
        json={"type": "notion", "name": "Company Notion", "productionEnabled": True},
    )

    assert create_response.status_code == 409
    assert "Provider-backed sources must be created" in create_response.json()["detail"]


def test_fake_oauth_completion_route_is_removed(client: TestClient) -> None:
    source_response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Manual Pilot", "productionEnabled": True},
    )
    source_id = source_response.json()["id"]

    oauth_response = client.post(f"/v1/sources/{source_id}/oauth/complete")

    assert oauth_response.status_code == 404




def test_oauth_created_notion_source_does_not_allow_manual_sync(client: TestClient) -> None:
    intent_response = client.post(
        "/v1/connections/intents",
        json={"provider": "notion", "sourceName": "Company Notion", "createdBy": "admin@example.com"},
    )
    assert intent_response.status_code == 201
    intent = intent_response.json()
    oauth_response = client.get(f"/v1/oauth/notion/callback?state={intent['stateNonce']}&code=code-123")
    assert oauth_response.status_code == 200
    source_id = oauth_response.json()["dataSource"]["id"]

    sync_response = client.post(
        f"/v1/manual-sources/{source_id}/sync",
        json={
            "objects": [
                {
                    "sourceExternalId": "notion-page-1",
                    "title": "Product Principles",
                    "content": "Official means reviewed. Evidence must have provenance.",
                    "sourceUrl": "https://notion.example/page-1",
                }
            ]
        },
    )

    assert sync_response.status_code == 409
    assert "Manual sync is only available for manual sources" in sync_response.json()["detail"]


def test_legacy_source_sync_route_is_removed(client: TestClient) -> None:
    source_response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Manual Pilot", "productionEnabled": True},
    )
    source_id = source_response.json()["id"]

    sync_response = client.post(f"/v1/sources/{source_id}/sync", json={"objects": []})

    assert sync_response.status_code == 404


def test_sync_is_idempotent_for_same_source_external_id_and_hash(client: TestClient) -> None:
    source_response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Manual Pilot", "productionEnabled": True},
    )
    source_id = source_response.json()["id"]
    payload = {
        "objects": [
            {
                "sourceExternalId": "doc-1",
                "title": "Cornerstone Overview",
                "content": "Cornerstone is a context layer. Evidence must have provenance.",
                "sourceUrl": "https://example.internal/doc-1",
            }
        ]
    }

    first_sync = client.post(f"/v1/manual-sources/{source_id}/sync", json=payload)
    second_sync = client.post(f"/v1/manual-sources/{source_id}/sync", json=payload)

    assert first_sync.status_code == 200
    assert second_sync.status_code == 200
    first_body = first_sync.json()
    second_body = second_sync.json()
    assert second_body["dataSource"]["artifactCount"] == 1
    assert second_body["dataSource"]["evidenceFragmentCount"] == 2
    assert second_body["artifacts"][0]["id"] == first_body["artifacts"][0]["id"]
    assert second_body["evidenceFragments"][0]["id"] == first_body["evidenceFragments"][0]["id"]


def test_failed_sync_after_prior_success_marks_degraded_and_rolls_back(
    store,
    monkeypatch,
) -> None:
    from fastapi.testclient import TestClient as LocalTestClient

    from cornerstone.main import create_app
    from cornerstone.services import source_sync

    client = LocalTestClient(create_app(store=store), raise_server_exceptions=False)
    source_response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Manual Pilot", "productionEnabled": True},
    )
    source_id = source_response.json()["id"]
    first_sync = client.post(
        f"/v1/manual-sources/{source_id}/sync",
        json={
            "objects": [
                {
                    "sourceExternalId": "doc-1",
                    "title": "Stable Doc",
                    "content": "Cornerstone is a context layer.",
                    "sourceUrl": "https://example.internal/doc-1",
                }
            ]
        },
    )
    assert first_sync.status_code == 200

    def fail_extraction(*args, **kwargs):
        raise RuntimeError("forced extraction failure")

    monkeypatch.setattr(source_sync, "extract_evidence_fragments", fail_extraction)
    failed_sync = client.post(
        f"/v1/manual-sources/{source_id}/sync",
        json={
            "objects": [
                {
                    "sourceExternalId": "doc-2",
                    "title": "Broken Doc",
                    "content": "This extraction fails.",
                    "sourceUrl": "https://example.internal/doc-2",
                }
            ]
        },
    )

    assert failed_sync.status_code == 500
    source = client.get("/v1/sources").json()["sources"][0]
    assert source["status"] == "degraded"
    assert source["freshnessState"] == "unknown"
    assert source["artifactCount"] == 1
    artifacts = client.get("/v1/artifacts").json()
    assert [artifact["sourceExternalId"] for artifact in artifacts] == ["doc-1"]


def test_missing_source_read_and_manual_sync_return_404(client: TestClient) -> None:
    read_response = client.get("/v1/sources/missing-source")
    sync_response = client.post(
        "/v1/manual-sources/missing-source/sync",
        json={"objects": []},
    )

    assert read_response.status_code == 404
    assert sync_response.status_code == 404


def test_connector_catalog_detail_endpoint(client: TestClient) -> None:
    response = client.get("/v1/connectors/notion")

    assert response.status_code == 200
    assert response.json()["provider"] == "notion"
