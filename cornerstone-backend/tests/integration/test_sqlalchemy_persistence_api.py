from __future__ import annotations

from fastapi.testclient import TestClient

from cornerstone.main import create_app
from cornerstone.persistence.store import SqlAlchemyStore


def _create_reviewed_evidence(client: TestClient) -> dict[str, str]:
    source_response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Persistent Manual", "productionEnabled": True},
    )
    assert source_response.status_code == 201
    source_id = source_response.json()["id"]
    sync_response = client.post(
        f"/v1/manual-sources/{source_id}/sync",
        json={
            "objects": [
                {
                    "sourceExternalId": "doc-persist-1",
                    "title": "Persistence PRD",
                    "content": "Cornerstone persists official context with evidence and provenance. Official context must be backed by reviewed evidence.",
                    "sourceUrl": "https://example.internal/persistence-prd",
                }
            ]
        },
    )
    assert sync_response.status_code == 200
    body = sync_response.json()
    evidence_id = body["evidenceFragments"][0]["id"]
    review_response = client.post(
        f"/v1/evidence/{evidence_id}/review",
        json={"trustState": "reviewed", "reviewedBy": "reviewer@example.com"},
    )
    assert review_response.status_code == 200
    return {"source_id": source_id, "evidence_id": evidence_id}


def test_sqlalchemy_store_persists_source_artifact_evidence_and_official_concept(
    persistent_client: TestClient,
    sqlite_persistent_store: SqlAlchemyStore,
) -> None:
    ids = _create_reviewed_evidence(persistent_client)
    concept_response = persistent_client.post(
        "/v1/concepts",
        json={
            "name": "Persistence",
            "shortDefinition": "Durable trust-state storage.",
            "evidenceFragmentIds": [ids["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    assert concept_response.status_code == 201
    concept_id = concept_response.json()["id"]
    official_response = persistent_client.post(
        f"/v1/concepts/{concept_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert official_response.status_code == 200

    # New app/store instance, same engine: verifies durable database-backed reads, not object memory.
    reloaded_client = TestClient(create_app(store=SqlAlchemyStore(sqlite_persistent_store.engine)))
    concept_read = reloaded_client.get(f"/v1/concepts/{concept_id}")
    source_read = reloaded_client.get("/v1/sources")
    evidence_read = reloaded_client.get("/v1/evidence")

    assert concept_read.status_code == 200
    assert concept_read.json()["status"] == "official"
    assert concept_read.json()["evidenceFragmentIds"] == [ids["evidence_id"]]
    assert source_read.json()["sources"][0]["artifactCount"] == 1
    assert len(evidence_read.json()) == 2


def test_sqlalchemy_store_sync_idempotency_uses_database_unique_identity(
    persistent_client: TestClient,
) -> None:
    source_response = persistent_client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Idempotent Manual", "productionEnabled": True},
    )
    source_id = source_response.json()["id"]
    payload = {
        "objects": [
            {
                "sourceExternalId": "doc-repeat",
                "title": "Repeatable Doc",
                "content": "The same source object should be reused when the hash is unchanged. Idempotent sync must not duplicate artifacts.",
                "sourceUrl": "https://example.internal/repeatable-doc",
            }
        ]
    }

    first_sync = persistent_client.post(f"/v1/manual-sources/{source_id}/sync", json=payload)
    second_sync = persistent_client.post(f"/v1/manual-sources/{source_id}/sync", json=payload)

    assert first_sync.status_code == 200
    assert second_sync.status_code == 200
    assert second_sync.json()["dataSource"]["artifactCount"] == 1
    assert second_sync.json()["dataSource"]["evidenceFragmentCount"] == 2
    assert second_sync.json()["artifacts"][0]["id"] == first_sync.json()["artifacts"][0]["id"]


def test_sqlalchemy_store_transaction_rolls_back_partial_sync(
    sqlite_persistent_store: SqlAlchemyStore,
    monkeypatch,
) -> None:
    from cornerstone.services import source_sync

    client = TestClient(create_app(store=sqlite_persistent_store), raise_server_exceptions=False)
    source_response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Rollback Manual", "productionEnabled": True},
    )
    source_id = source_response.json()["id"]

    def fail_extraction(*args, **kwargs):
        raise RuntimeError("forced extraction failure")

    monkeypatch.setattr(source_sync, "extract_evidence_fragments", fail_extraction)
    failed_sync = client.post(
        f"/v1/manual-sources/{source_id}/sync",
        json={
            "objects": [
                {
                    "sourceExternalId": "doc-fail",
                    "title": "Broken Doc",
                    "content": "This should not commit.",
                    "sourceUrl": "https://example.internal/broken-doc",
                }
            ]
        },
    )

    assert failed_sync.status_code == 500
    assert client.get("/v1/artifacts").json() == []
    source = client.get("/v1/sources").json()["sources"][0]
    assert source["status"] == "failed"
    assert source["artifactCount"] == 0


def test_sqlalchemy_store_persists_sync_schedule_and_scheduled_job(
    persistent_client: TestClient,
    sqlite_persistent_store: SqlAlchemyStore,
) -> None:
    from datetime import timedelta

    from cornerstone.schemas import utc_now

    intent = persistent_client.post(
        "/v1/connections/intents",
        json={"provider": "notion", "sourceName": "Persistent Notion", "createdBy": "admin@example.com"},
    ).json()
    callback = persistent_client.get(f"/v1/oauth/notion/callback?state={intent['stateNonce']}&code=code-123")
    assert callback.status_code == 200
    source_id = callback.json()["dataSource"]["id"]
    assert persistent_client.post(f"/v1/sources/{source_id}/test").status_code == 200
    assert persistent_client.post(f"/v1/sources/{source_id}/discover", json={"pageSize": 25}).status_code == 200
    assert persistent_client.put(
        f"/v1/sources/{source_id}/selections",
        json={"syncMode": "selected_only", "selectedExternalObjectIds": ["notion-page-1"]},
    ).status_code == 200
    due_at = (utc_now() - timedelta(minutes=1)).isoformat()
    schedule = persistent_client.put(
        f"/v1/sources/{source_id}/sync-schedule",
        json={"enabled": True, "intervalMinutes": 30, "startAt": due_at, "createdBy": "admin@example.com"},
    )
    assert schedule.status_code == 200

    scheduler = persistent_client.post("/v1/sync-scheduler/run", json={"maxSchedules": 10})
    assert scheduler.status_code == 200
    job_id = scheduler.json()["jobs"][0]["id"]

    reloaded_client = TestClient(create_app(store=SqlAlchemyStore(sqlite_persistent_store.engine)))
    persisted_schedule = reloaded_client.get(f"/v1/sources/{source_id}/sync-schedule")
    persisted_job = reloaded_client.get(f"/v1/sync-jobs/{job_id}")

    assert persisted_schedule.status_code == 200
    assert persisted_schedule.json()["lastEnqueuedSyncJobId"] == job_id
    assert persisted_job.status_code == 200
    assert persisted_job.json()["job"]["trigger"] == "scheduled"


def test_sqlalchemy_store_persists_concept_relation_officialization(
    persistent_client: TestClient,
    sqlite_persistent_store: SqlAlchemyStore,
) -> None:
    ids = _create_reviewed_evidence(persistent_client)
    concept_ids: list[str] = []
    for name in ["Persistence Source", "Persistence Target"]:
        created = persistent_client.post(
            "/v1/concepts",
            json={
                "name": name,
                "shortDefinition": f"{name} definition.",
                "evidenceFragmentIds": [ids["evidence_id"]],
                "createdBy": "reviewer@example.com",
            },
        )
        assert created.status_code == 201
        concept_id = created.json()["id"]
        official = persistent_client.post(
            f"/v1/concepts/{concept_id}/officialize",
            json={"reviewedBy": "reviewer@example.com"},
        )
        assert official.status_code == 200
        concept_ids.append(concept_id)

    relation = persistent_client.post(
        "/v1/concept-relations",
        json={
            "sourceConceptId": concept_ids[0],
            "targetConceptId": concept_ids[1],
            "relationType": "depends_on",
            "evidenceFragmentIds": [ids["evidence_id"]],
            "createdBy": "reviewer@example.com",
        },
    )
    assert relation.status_code == 201
    relation_id = relation.json()["id"]
    official_relation = persistent_client.post(
        f"/v1/concept-relations/{relation_id}/officialize",
        json={"reviewedBy": "reviewer@example.com"},
    )
    assert official_relation.status_code == 200

    reloaded_client = TestClient(create_app(store=SqlAlchemyStore(sqlite_persistent_store.engine)))
    relation_read = reloaded_client.get(f"/v1/concept-relations/{relation_id}")
    relation_list = reloaded_client.get(f"/v1/concept-relations?conceptId={concept_ids[0]}")

    assert relation_read.status_code == 200
    assert relation_read.json()["status"] == "official"
    assert relation_read.json()["evidenceFragmentIds"] == [ids["evidence_id"]]
    assert relation_list.status_code == 200
    assert relation_list.json()[0]["id"] == relation_id
