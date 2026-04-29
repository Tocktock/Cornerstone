from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cornerstone.schemas import ConnectorError, ConnectorErrorCode, ConnectorNextAction, utc_now
from cornerstone.store import InMemoryStore

pytestmark = pytest.mark.integration


def test_sync_job_is_queued_by_default_and_worker_advances_cursor(client: TestClient) -> None:
    source_id = _connect_test_discover_select_notion(client)

    create_response = client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com"},
    )

    assert create_response.status_code == 201
    queued = create_response.json()
    assert queued["job"]["status"] == "queued"
    assert queued["job"]["attemptCount"] == 0
    assert client.get(f"/v1/artifacts?datasourceId={source_id}").json() == []

    run_response = client.post("/v1/sync-worker/run", json={"maxJobs": 1})

    assert run_response.status_code == 200
    body = run_response.json()
    assert body["processedJobCount"] == 1
    assert body["jobs"][0]["job"]["status"] == "succeeded"
    assert body["jobs"][0]["job"]["attemptCount"] == 1
    assert "sync.cursor_advanced" in [event["eventType"] for event in body["jobs"][0]["events"]]

    cursor_response = client.get(f"/v1/sources/{source_id}/sync-cursor")
    assert cursor_response.status_code == 200
    cursor = cursor_response.json()
    assert cursor["lastSuccessfulSyncJobId"] == queued["job"]["id"]
    assert cursor["processedExternalObjectIds"] == ["notion-page-1"]

    artifacts = client.get(f"/v1/artifacts?datasourceId={source_id}").json()
    assert len(artifacts) == 1


def test_cancelled_queued_sync_job_does_not_ingest_content(client: TestClient) -> None:
    source_id = _connect_test_discover_select_notion(client)
    queued = client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com"},
    ).json()

    cancel_response = client.post(f"/v1/sync-jobs/{queued['job']['id']}/cancel")
    run_response = client.post("/v1/sync-worker/run", json={"maxJobs": 1})

    assert cancel_response.status_code == 200
    assert cancel_response.json()["job"]["status"] == "cancelled"
    assert run_response.status_code == 200
    assert run_response.json()["processedJobCount"] == 0
    assert client.get(f"/v1/artifacts?datasourceId={source_id}").json() == []


def test_rate_limited_sync_job_waits_for_retry_and_does_not_advance_cursor(
    client: TestClient,
    store: InMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id = _connect_test_discover_select_notion(client)
    job = client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com", "maxAttempts": 2},
    ).json()["job"]

    class RateLimitedConnector:
        async def list_objects(self, **kwargs: Any) -> list[Any]:
            _ = kwargs
            error = ConnectorError(
                code=ConnectorErrorCode.RATE_LIMITED,
                user_message="Provider asked Cornerstone to wait.",
                retryable=True,
                next_action=ConnectorNextAction.WAIT_AND_RETRY,
                retry_after_seconds=60,
            )
            raise _FakeProviderError(error)

    monkeypatch.setattr("cornerstone.services.sync_worker.get_connector", lambda provider, settings: RateLimitedConnector())

    run_response = client.post("/v1/sync-worker/run", json={"jobId": job["id"]})

    assert run_response.status_code == 200
    body = run_response.json()
    assert body["processedJobCount"] == 1
    failed_job = body["jobs"][0]["job"]
    assert failed_job["status"] == "retry_waiting"
    assert failed_job["attemptCount"] == 1
    assert failed_job["nextAttemptAt"] is not None
    assert failed_job["error"]["code"] == "rate_limited"

    source = client.get(f"/v1/sources/{source_id}").json()
    assert source["syncStatus"] == "waiting_retry"
    assert source["syncFreshnessState"] == "unknown"

    assert client.get(f"/v1/sources/{source_id}/sync-cursor").status_code == 404
    assert store.list_artifacts(datasource_id=source_id) == []


def test_retry_waiting_job_is_skipped_until_due(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    source_id = _connect_test_discover_select_notion(client)
    job = client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com", "maxAttempts": 2},
    ).json()["job"]

    class RateLimitedConnector:
        async def list_objects(self, **kwargs: Any) -> list[Any]:
            _ = kwargs
            error = ConnectorError(
                code=ConnectorErrorCode.RATE_LIMITED,
                user_message="Provider asked Cornerstone to wait.",
                retryable=True,
                next_action=ConnectorNextAction.WAIT_AND_RETRY,
                retry_after_seconds=60,
            )
            raise _FakeProviderError(error)

    monkeypatch.setattr("cornerstone.services.sync_worker.get_connector", lambda provider, settings: RateLimitedConnector())
    assert client.post("/v1/sync-worker/run", json={"jobId": job["id"]}).json()["processedJobCount"] == 1

    second_run = client.post("/v1/sync-worker/run", json={"jobId": job["id"]})

    assert second_run.status_code == 200
    assert second_run.json()["processedJobCount"] == 0
    assert second_run.json()["skippedJobCount"] == 1


class _FakeProviderError(Exception):
    def __init__(self, connector_error: ConnectorError) -> None:
        super().__init__(connector_error.user_message)
        self.connector_error = connector_error


def _connect_test_discover_select_notion(client: TestClient) -> str:
    intent = client.post(
        "/v1/connections/intents",
        json={"provider": "notion", "sourceName": "Team Notion", "createdBy": "admin@example.com"},
    ).json()
    connect_response = client.get(f"/v1/oauth/notion/callback?state={intent['stateNonce']}&code=code-123")
    assert connect_response.status_code == 200
    source_id = connect_response.json()["dataSource"]["id"]
    assert client.post(f"/v1/sources/{source_id}/test").status_code == 200
    assert client.post(f"/v1/sources/{source_id}/discover", json={"pageSize": 25}).status_code == 200
    select_response = client.put(
        f"/v1/sources/{source_id}/selections",
        json={"syncMode": "selected_only", "selectedExternalObjectIds": ["notion-page-1"]},
    )
    assert select_response.status_code == 200
    return source_id


def test_sync_success_writes_roll_back_when_cursor_advance_fails(
    client: TestClient,
    store: InMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id = _connect_test_discover_select_notion(client)
    job = client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com"},
    ).json()["job"]

    def fail_cursor_write(cursor: Any) -> Any:
        _ = cursor
        raise RuntimeError("cursor write failed after artifact extraction")

    monkeypatch.setattr(store, "upsert_sync_cursor", fail_cursor_write)

    run_response = client.post("/v1/sync-worker/run", json={"jobId": job["id"]})

    assert run_response.status_code == 200
    body = run_response.json()
    assert body["processedJobCount"] == 1
    failed_job = body["jobs"][0]["job"]
    assert failed_job["status"] == "retry_waiting"
    assert failed_job["error"]["code"] == "unknown"
    assert "sync.retry_scheduled" in [event["eventType"] for event in body["jobs"][0]["events"]]
    assert "sync.cursor_advanced" not in [event["eventType"] for event in body["jobs"][0]["events"]]

    source = client.get(f"/v1/sources/{source_id}").json()
    assert source["syncStatus"] == "waiting_retry"
    assert source["artifactCount"] == 0
    assert source["evidenceFragmentCount"] == 0
    assert source["syncFreshnessState"] == "unknown"
    assert client.get(f"/v1/sources/{source_id}/sync-cursor").status_code == 404
    assert store.list_artifacts(datasource_id=source_id) == []
    assert store.list_evidence_fragments() == []


def test_persistent_sync_success_writes_roll_back_when_cursor_advance_fails(
    persistent_client: TestClient,
    sqlite_persistent_store: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id = _connect_test_discover_select_notion(persistent_client)
    job = persistent_client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com"},
    ).json()["job"]

    def fail_cursor_write(cursor: Any) -> Any:
        _ = cursor
        raise RuntimeError("cursor write failed after artifact extraction")

    monkeypatch.setattr(sqlite_persistent_store, "upsert_sync_cursor", fail_cursor_write)

    run_response = persistent_client.post("/v1/sync-worker/run", json={"jobId": job["id"]})

    assert run_response.status_code == 200
    body = run_response.json()
    assert body["processedJobCount"] == 1
    assert body["jobs"][0]["job"]["status"] == "retry_waiting"
    assert persistent_client.get(f"/v1/sources/{source_id}/sync-cursor").status_code == 404
    assert sqlite_persistent_store.list_artifacts(datasource_id=source_id) == []
    assert sqlite_persistent_store.list_evidence_fragments() == []
    source = persistent_client.get(f"/v1/sources/{source_id}").json()
    assert source["artifactCount"] == 0
    assert source["evidenceFragmentCount"] == 0
    assert source["syncStatus"] == "waiting_retry"


def test_worker_claim_event_records_worker_identity_and_clears_lease_on_success(client: TestClient) -> None:
    source_id = _connect_test_discover_select_notion(client)
    job = client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com"},
    ).json()["job"]

    run_response = client.post(
        "/v1/sync-worker/run",
        json={"jobId": job["id"], "workerId": "worker-a", "leaseSeconds": 120},
    )

    assert run_response.status_code == 200
    body = run_response.json()
    assert body["processedJobCount"] == 1
    saved_job = body["jobs"][0]["job"]
    assert saved_job["status"] == "succeeded"
    assert saved_job["leaseOwner"] is None
    assert saved_job["leaseAcquiredAt"] is None
    assert saved_job["leaseExpiresAt"] is None
    claim_events = [event for event in body["jobs"][0]["events"] if event["eventType"] == "sync.job_claimed"]
    assert len(claim_events) == 1
    assert claim_events[0]["metadata"]["workerId"] == "worker-a"


def test_already_claimed_job_is_skipped_by_second_worker(client: TestClient, store: InMemoryStore) -> None:
    source_id = _connect_test_discover_select_notion(client)
    job = client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com"},
    ).json()["job"]
    claimed = store.claim_sync_job(job["id"], worker_id="worker-a", lease_seconds=120)
    assert claimed is not None

    run_response = client.post(
        "/v1/sync-worker/run",
        json={"jobId": job["id"], "workerId": "worker-b", "leaseSeconds": 120},
    )

    assert run_response.status_code == 200
    body = run_response.json()
    assert body["processedJobCount"] == 0
    assert body["skippedJobCount"] == 1
    saved = store.get_sync_job(job["id"])
    assert saved.status == "running"
    assert saved.lease_owner == "worker-a"


def test_expired_running_job_lease_can_be_reclaimed_by_second_worker(
    client: TestClient,
    store: InMemoryStore,
) -> None:
    source_id = _connect_test_discover_select_notion(client)
    job = client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com"},
    ).json()["job"]
    claimed = store.claim_sync_job(job["id"], worker_id="worker-a", lease_seconds=60)
    assert claimed is not None
    expired = claimed.model_copy(
        update={"lease_expires_at": utc_now() - timedelta(seconds=1)},
        deep=True,
    )
    store.update_sync_job(expired)

    run_response = client.post(
        "/v1/sync-worker/run",
        json={"jobId": job["id"], "workerId": "worker-b", "leaseSeconds": 120},
    )

    assert run_response.status_code == 200
    body = run_response.json()
    assert body["processedJobCount"] == 1
    saved_job = body["jobs"][0]["job"]
    assert saved_job["status"] == "succeeded"
    claim_events = [event for event in body["jobs"][0]["events"] if event["eventType"] == "sync.job_claimed"]
    assert claim_events[-1]["metadata"]["workerId"] == "worker-b"
    assert claim_events[-1]["metadata"]["recoveredExpiredLease"] is True


def test_sync_job_lease_heartbeat_extends_active_worker_lease(
    client: TestClient,
    store: InMemoryStore,
) -> None:
    source_id = _connect_test_discover_select_notion(client)
    job = client.post(
        f"/v1/sources/{source_id}/sync-jobs",
        json={"createdBy": "admin@example.com"},
    ).json()["job"]
    claimed = store.claim_sync_job(job["id"], worker_id="worker-a", lease_seconds=60)
    assert claimed is not None

    wrong_worker_response = client.post(
        f"/v1/sync-jobs/{job['id']}/heartbeat",
        json={"workerId": "worker-b", "leaseSeconds": 120},
    )
    assert wrong_worker_response.status_code == 409

    heartbeat_response = client.post(
        f"/v1/sync-jobs/{job['id']}/heartbeat",
        json={"workerId": "worker-a", "leaseSeconds": 120},
    )

    assert heartbeat_response.status_code == 200
    heartbeat_job = heartbeat_response.json()["job"]
    assert heartbeat_job["status"] == "running"
    assert heartbeat_job["leaseOwner"] == "worker-a"
    assert heartbeat_job["leaseHeartbeatAt"] is not None
    assert heartbeat_job["leaseExpiresAt"] > claimed.lease_expires_at.isoformat()
    heartbeat_events = [
        event
        for event in heartbeat_response.json()["events"]
        if event["eventType"] == "sync.lease_heartbeat"
    ]
    assert heartbeat_events
    assert heartbeat_events[-1]["metadata"]["workerId"] == "worker-a"
