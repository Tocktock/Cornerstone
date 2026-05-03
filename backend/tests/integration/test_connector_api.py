from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from cornerstone.persistence.store import SqlAlchemyStore
from cornerstone.store import InMemoryStore

pytestmark = pytest.mark.integration


def test_connector_catalog_endpoint_returns_notion_as_available(client: TestClient) -> None:
    response = client.get("/v1/connectors")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["provider"] == "notion"
    assert body[0]["availability"] == "available"
    assert body[0]["supportedObjects"] == ["page"]
    assert body[0]["discoverableObjects"] == ["page", "database", "data_source"]
    assert body[0]["ingestibleObjects"] == ["page"]
    assert body[0]["setupSteps"][0]["key"] == "authorize"


def test_create_intent_and_authorize_redirect_are_stateful(client: TestClient) -> None:
    intent_response = client.post(
        "/v1/connections/intents",
        json={"provider": "notion", "sourceName": "Team Notion", "createdBy": "admin@example.com"},
    )
    assert intent_response.status_code == 201
    intent = intent_response.json()
    assert intent["status"] == "created"
    assert intent["authorizationUrl"].startswith("https://api.notion.com/v1/oauth/authorize")

    redirect_response = client.get(
        f"/v1/oauth/notion/authorize?intentId={intent['id']}",
        follow_redirects=False,
    )

    assert redirect_response.status_code == 307
    assert "state=" in redirect_response.headers["location"]
    read_response = client.get(f"/v1/connections/intents/{intent['id']}")
    assert read_response.json()["status"] == "oauth_redirected"


def test_oauth_callback_bad_state_is_rejected(client: TestClient) -> None:
    response = client.get("/v1/oauth/notion/callback?state=missing-state&code=abc")

    assert response.status_code == 404
    assert response.json()["detail"] == "OAuth state not found."


def test_oauth_callback_provider_error_marks_intent_failed(client: TestClient) -> None:
    intent = _create_notion_intent(client)

    response = client.get(f"/v1/oauth/notion/callback?state={intent['stateNonce']}&error=access_denied")

    assert response.status_code == 200
    body = response.json()
    assert body["intent"]["status"] == "failed"
    assert body["intent"]["failureError"]["code"] == "oauth_failed"
    assert body["nextAction"] == "reconnect"


def test_oauth_success_creates_source_and_public_credential_without_plaintext(client: TestClient) -> None:
    intent = _create_notion_intent(client)

    response = client.get(f"/v1/oauth/notion/callback?state={intent['stateNonce']}&code=code-123")

    assert response.status_code == 200
    body = response.json()
    assert body["intent"]["status"] == "completed"
    assert body["dataSource"]["type"] == "notion"
    assert body["dataSource"]["status"] == "sync_pending"
    assert body["dataSource"]["authStatus"] == "authorized"
    assert body["dataSource"]["connectionStatus"] == "untested"
    assert body["dataSource"]["syncStatus"] == "never_synced"
    assert body["dataSource"]["nextAction"] == "test_connection"
    assert body["credential"]["provider"] == "notion"
    assert "encryptedAccessToken" not in body["credential"]
    assert "mock-notion-access-token-code-123" not in response.text
    assert body["nextAction"] == "test_connection"


def test_test_connection_marks_oauth_source_test_passed_and_discovery_next(client: TestClient) -> None:
    source_id = _connect_notion(client)["source_id"]

    response = client.post(f"/v1/sources/{source_id}/test")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "passed"
    assert body["canReadObjects"] is True
    assert body["sampleObjectCount"] == 2
    source = client.get(f"/v1/sources/{source_id}").json()
    assert source["status"] == "connected"
    assert source["connectionStatus"] == "test_passed"
    assert source["nextAction"] == "discover_sources"


def test_discovery_requires_test_connection(client: TestClient) -> None:
    source_id = _connect_notion(client)["source_id"]

    response = client.post(f"/v1/sources/{source_id}/discover", json={"pageSize": 25})

    assert response.status_code == 409
    assert response.json()["detail"] == "Test the source connection before discovery."


def test_discovery_persists_provider_object_snapshots_and_prompts_selection(client: TestClient) -> None:
    source_id = _connect_and_test_notion(client)

    response = client.post(f"/v1/sources/{source_id}/discover", json={"pageSize": 25})
    objects_response = client.get(f"/v1/sources/{source_id}/objects")

    assert response.status_code == 200
    body = response.json()
    assert body["dataSource"]["discoveredObjectCount"] == 2
    assert body["dataSource"]["nextAction"] == "select_sources"
    assert {item["externalId"] for item in body["objects"]} == {"notion-page-1", "notion-database-1"}
    objects_by_id = {item["externalId"]: item for item in body["objects"]}
    assert objects_by_id["notion-page-1"]["accessState"] == "accessible"
    assert objects_by_id["notion-page-1"]["ingestionSupported"] is True
    assert objects_by_id["notion-database-1"]["ingestionSupported"] is False
    assert "database ingestion" in objects_by_id["notion-database-1"]["ingestionUnsupportedReason"]
    assert objects_response.status_code == 200
    assert objects_response.json()["totalCount"] == 2
    assert objects_response.json()["accessibleCount"] == 2
    assert objects_response.json()["syncableCount"] == 1
    assert objects_response.json()["selectedCount"] == 0


def test_selection_requires_discovery_before_save(client: TestClient) -> None:
    source_id = _connect_and_test_notion(client)

    response = client.put(
        f"/v1/sources/{source_id}/selections",
        json={"syncMode": "selected_only", "selectedExternalObjectIds": ["notion-page-1"]},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Discover provider objects before saving a source selection."


def test_source_selection_rejects_unknown_or_inaccessible_object(client: TestClient) -> None:
    source_id = _connect_test_discover_notion(client)

    response = client.put(
        f"/v1/sources/{source_id}/selections",
        json={"syncMode": "selected_only", "selectedExternalObjectIds": ["missing-page"]},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "permission_denied"
    assert response.json()["detail"]["unknownExternalObjectIds"] == ["missing-page"]



def test_source_selection_rejects_unsupported_notion_database_object(client: TestClient) -> None:
    source_id = _connect_test_discover_notion(client)

    response = client.put(
        f"/v1/sources/{source_id}/selections",
        json={"syncMode": "selected_only", "selectedExternalObjectIds": ["notion-database-1"]},
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "unsupported_object_type"
    assert detail["unsupportedExternalObjectIds"] == ["notion-database-1"]
    assert "database ingestion" in detail["unsupportedReasons"]["notion-database-1"]


def test_workspace_limited_selection_is_rejected_until_implemented(client: TestClient) -> None:
    source_id = _connect_test_discover_notion(client)

    response = client.put(
        f"/v1/sources/{source_id}/selections",
        json={"syncMode": "workspace_limited", "selectedExternalObjectIds": []},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "unsupported_object_type"


def test_source_selection_can_be_saved_and_marks_snapshots_selected(client: TestClient) -> None:
    source_id = _connect_test_discover_notion(client)

    save_response = client.put(
        f"/v1/sources/{source_id}/selections",
        json={"syncMode": "selected_only", "selectedExternalObjectIds": ["notion-page-1"]},
    )
    read_response = client.get(f"/v1/sources/{source_id}/selections")
    objects_response = client.get(f"/v1/sources/{source_id}/objects")
    source_response = client.get(f"/v1/sources/{source_id}")

    assert save_response.status_code == 200
    assert read_response.status_code == 200
    assert read_response.json()["selectedExternalObjectIds"] == ["notion-page-1"]
    selected = [item for item in objects_response.json()["objects"] if item["selectedForSync"]]
    assert [item["externalId"] for item in selected] == ["notion-page-1"]
    assert source_response.json()["selectedObjectCount"] == 1
    assert source_response.json()["nextAction"] == "run_first_sync"


def test_all_accessible_selection_expands_only_to_syncable_objects(client: TestClient) -> None:
    source_id = _connect_test_discover_notion(client)

    response = client.put(
        f"/v1/sources/{source_id}/selections",
        json={"syncMode": "all_accessible", "selectedExternalObjectIds": []},
    )

    assert response.status_code == 200
    assert response.json()["selectedExternalObjectIds"] == ["notion-page-1"]


def test_sync_job_requires_source_selection_before_first_sync(client: TestClient) -> None:
    source_id = _connect_test_discover_notion(client)

    response = client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com", "runInline": True},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "source_selection_required"
    assert response.json()["detail"]["nextAction"] == "select_sources"


def test_sync_job_lifecycle_ingests_selected_notion_page_into_artifacts(client: TestClient) -> None:
    source_id = _connect_test_discover_notion(client)
    client.put(
        f"/v1/sources/{source_id}/selections",
        json={"syncMode": "selected_only", "selectedExternalObjectIds": ["notion-page-1"]},
    )

    response = client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com", "runInline": True},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["job"]["status"] == "succeeded"
    assert body["job"]["artifactCreatedCount"] == 1
    assert body["job"]["artifactReusedCount"] == 0
    assert body["job"]["evidenceCreatedCount"] > 0
    assert [event["eventType"] for event in body["events"]] == [
        "sync.job_queued",
        "sync.job_claimed",
        "sync.job_started",
        "sync.objects_selected",
        "sync.objects_normalized",
        "sync.cursor_advanced",
        "sync.job_succeeded",
        "ontology.reextraction_queued",
    ]
    claim_event = next(event for event in body["events"] if event["eventType"] == "sync.job_claimed")
    assert claim_event["metadata"]["workerId"] == "sync-worker"
    artifacts = client.get(f"/v1/artifacts?datasourceId={source_id}").json()
    assert len(artifacts) == 1
    assert artifacts[0]["sourceExternalId"] == "notion-page-1"
    assert artifacts[0]["sourceObjectType"] == "page"
    assert artifacts[0]["providerMetadata"]["provider"] == "notion"
    source = client.get(f"/v1/sources/{source_id}").json()
    assert source["syncStatus"] == "succeeded"
    assert source["syncFreshnessState"] == "fresh"
    assert source["contentFreshnessState"] == "fresh"
    assert source["artifactCount"] == 1
    assert source["nextAction"] == "review_evidence"


def test_disconnect_revokes_credential_and_blocks_future_test(
    client: TestClient,
    store: InMemoryStore,
) -> None:
    source_id = _connect_and_test_notion(client)

    disconnect_response = client.post(f"/v1/sources/{source_id}/disconnect")
    retry_test_response = client.post(f"/v1/sources/{source_id}/test")

    assert disconnect_response.status_code == 200
    assert disconnect_response.json()["status"] == "disconnected"
    assert disconnect_response.json()["authStatus"] == "revoked"
    assert disconnect_response.json()["nextAction"] == "reconnect"
    assert retry_test_response.status_code == 409
    credentials = store.list_connector_credentials(datasource_id=source_id)
    assert credentials[0].status == "revoked"
    assert credentials[0].revoked_at is not None


def test_persistent_connector_state_survives_store_instances(
    persistent_client: TestClient,
    sqlite_persistent_store: SqlAlchemyStore,
) -> None:
    source_id = _connect_test_discover_notion(persistent_client)
    persistent_client.put(
        f"/v1/sources/{source_id}/selections",
        json={"syncMode": "selected_only", "selectedExternalObjectIds": ["notion-page-1"]},
    )
    job_response = persistent_client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com", "runInline": True},
    )
    assert job_response.status_code == 201
    job_id = job_response.json()["job"]["id"]

    second_store = SqlAlchemyStore(sqlite_persistent_store.engine)

    assert second_store.get_data_source(source_id).type == "notion"
    assert second_store.get_data_source(source_id).discovered_object_count == 2
    assert second_store.get_active_credential_for_source(source_id).status == "active"
    assert second_store.get_source_selection(source_id).selected_external_object_ids == ["notion-page-1"]
    snapshots = second_store.list_provider_object_snapshots(datasource_id=source_id)
    assert len(snapshots) == 2
    assert [snapshot.external_id for snapshot in snapshots if snapshot.selected_for_sync] == ["notion-page-1"]
    assert second_store.get_sync_job(job_id).status == "succeeded"
    assert len(second_store.list_sync_job_events(job_id)) == 8


def _create_notion_intent(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/v1/connections/intents",
        json={"provider": "notion", "sourceName": "Team Notion", "createdBy": "admin@example.com"},
    )
    assert response.status_code == 201
    return response.json()


def _connect_notion(client: TestClient) -> dict[str, str]:
    intent = _create_notion_intent(client)
    response = client.get(f"/v1/oauth/notion/callback?state={intent['stateNonce']}&code=code-123")
    assert response.status_code == 200
    body = response.json()
    return {"source_id": body["dataSource"]["id"], "intent_id": body["intent"]["id"]}


def _connect_and_test_notion(client: TestClient) -> str:
    source_id = _connect_notion(client)["source_id"]
    test_response = client.post(f"/v1/sources/{source_id}/test")
    assert test_response.status_code == 200
    return source_id


def _connect_test_discover_notion(client: TestClient) -> str:
    source_id = _connect_and_test_notion(client)
    discovery_response = client.post(f"/v1/sources/{source_id}/discover", json={"pageSize": 25})
    assert discovery_response.status_code == 200
    return source_id
