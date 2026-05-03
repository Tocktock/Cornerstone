from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from cornerstone.schemas import utc_now

pytestmark = pytest.mark.integration


def test_schedule_requires_selection_before_enabling(client: TestClient) -> None:
    source_id = _connect_test_discover_notion(client)

    response = client.put(
        f"/v1/sources/{source_id}/sync-schedule",
        json={"enabled": True, "intervalMinutes": 60, "createdBy": "admin@example.com"},
    )

    assert response.status_code == 409
    assert "selection" in response.json()["detail"].lower()


def test_scheduler_enqueues_due_schedule_and_worker_processes_it(client: TestClient) -> None:
    source_id = _connect_test_discover_select_notion(client)
    due_at = (utc_now() - timedelta(minutes=1)).isoformat()

    schedule_response = client.put(
        f"/v1/sources/{source_id}/sync-schedule",
        json={
            "enabled": True,
            "intervalMinutes": 30,
            "startAt": due_at,
            "maxAttempts": 2,
            "createdBy": "scheduler@example.com",
        },
    )
    assert schedule_response.status_code == 200
    schedule = schedule_response.json()
    assert schedule["status"] == "active"
    assert schedule["lastEnqueuedSyncJobId"] is None

    scheduler_response = client.post("/v1/sync-scheduler/run", json={"maxSchedules": 10})
    assert scheduler_response.status_code == 200
    scheduler_body = scheduler_response.json()
    assert scheduler_body["enqueuedJobCount"] == 1
    assert scheduler_body["schedulesCheckedCount"] == 1
    scheduled_job = scheduler_body["jobs"][0]
    assert scheduled_job["trigger"] == "scheduled"
    assert scheduled_job["status"] == "queued"
    assert scheduled_job["maxAttempts"] == 2
    assert scheduler_body["schedules"][0]["lastEnqueuedSyncJobId"] == scheduled_job["id"]

    run_response = client.post("/v1/sync-worker/run", json={"maxJobs": 1})
    assert run_response.status_code == 200
    assert run_response.json()["processedJobCount"] == 1
    assert run_response.json()["jobs"][0]["job"]["status"] == "succeeded"

    cursor = client.get(f"/v1/sources/{source_id}/sync-cursor").json()
    assert cursor["lastSuccessfulSyncJobId"] == scheduled_job["id"]


def test_scheduler_skips_not_due_schedule_until_forced(client: TestClient) -> None:
    source_id = _connect_test_discover_select_notion(client)
    future_at = (utc_now() + timedelta(hours=1)).isoformat()
    assert client.put(
        f"/v1/sources/{source_id}/sync-schedule",
        json={"enabled": True, "intervalMinutes": 60, "startAt": future_at},
    ).status_code == 200

    skipped = client.post("/v1/sync-scheduler/run", json={"maxSchedules": 10})
    assert skipped.status_code == 200
    assert skipped.json()["enqueuedJobCount"] == 0
    assert skipped.json()["schedulesCheckedCount"] == 0

    forced = client.post("/v1/sync-scheduler/run", json={"maxSchedules": 10, "includeNotDue": True})
    assert forced.status_code == 200
    assert forced.json()["enqueuedJobCount"] == 1


def test_scheduler_does_not_enqueue_duplicate_when_source_has_active_job(client: TestClient) -> None:
    source_id = _connect_test_discover_select_notion(client)
    due_at = (utc_now() - timedelta(minutes=1)).isoformat()
    assert client.put(
        f"/v1/sources/{source_id}/sync-schedule",
        json={"enabled": True, "intervalMinutes": 60, "startAt": due_at},
    ).status_code == 200
    assert client.post(f"/v1/sources/{source_id}/sync-jobs", json={"createdBy": "admin@example.com"}).status_code == 201

    scheduler_response = client.post("/v1/sync-scheduler/run", json={"maxSchedules": 10})

    assert scheduler_response.status_code == 200
    body = scheduler_response.json()
    assert body["enqueuedJobCount"] == 0
    assert body["skippedScheduleCount"] == 1


def test_paused_schedule_is_persisted_but_not_enqueued(client: TestClient) -> None:
    source_id = _connect_test_discover_select_notion(client)
    due_at = (utc_now() - timedelta(minutes=1)).isoformat()
    response = client.put(
        f"/v1/sources/{source_id}/sync-schedule",
        json={"enabled": False, "intervalMinutes": 60, "startAt": due_at},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "paused"

    fetched = client.get(f"/v1/sources/{source_id}/sync-schedule")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "paused"

    scheduler_response = client.post("/v1/sync-scheduler/run", json={"maxSchedules": 10, "includeNotDue": True})
    assert scheduler_response.status_code == 200
    assert scheduler_response.json()["enqueuedJobCount"] == 0


def _connect_test_discover_notion(client: TestClient) -> str:
    intent = client.post(
        "/v1/connections/intents",
        json={"provider": "notion", "sourceName": "Team Notion", "createdBy": "admin@example.com"},
    ).json()
    connect_response = client.get(f"/v1/oauth/notion/callback?state={intent['stateNonce']}&code=code-123")
    assert connect_response.status_code == 200
    source_id = connect_response.json()["dataSource"]["id"]
    assert client.post(f"/v1/sources/{source_id}/test").status_code == 200
    assert client.post(f"/v1/sources/{source_id}/discover", json={"pageSize": 25}).status_code == 200
    return source_id


def _connect_test_discover_select_notion(client: TestClient) -> str:
    source_id = _connect_test_discover_notion(client)
    select_response = client.put(
        f"/v1/sources/{source_id}/selections",
        json={"syncMode": "selected_only", "selectedExternalObjectIds": ["notion-page-1"]},
    )
    assert select_response.status_code == 200
    return source_id
