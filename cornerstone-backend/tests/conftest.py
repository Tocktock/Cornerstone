from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

os.environ.setdefault("CORNERSTONE_SKIP_GLOBAL_APP", "1")
from cornerstone.config import Settings
from cornerstone.store import InMemoryStore

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def test_settings() -> Settings:
    return Settings(production_mode=True)


@pytest.fixture
def client(store: InMemoryStore, test_settings: Settings):
    from fastapi.testclient import TestClient

    from cornerstone.main import create_app

    return TestClient(create_app(store=store, settings=test_settings))


@pytest.fixture
def sqlite_persistent_store():
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    from cornerstone.persistence.models import Base
    from cornerstone.persistence.store import SqlAlchemyStore

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    return SqlAlchemyStore(engine)


@pytest.fixture
def persistent_client(sqlite_persistent_store, test_settings: Settings):
    from fastapi.testclient import TestClient

    from cornerstone.main import create_app

    return TestClient(create_app(store=sqlite_persistent_store, settings=test_settings))


@pytest.fixture
def synced_evidence(client: TestClient) -> dict[str, str]:
    source_response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Pilot Domain", "productionEnabled": True},
    )
    assert source_response.status_code == 201
    source_id = source_response.json()["id"]

    sync_response = client.post(
        f"/v1/manual-sources/{source_id}/sync",
        json={
            "objects": [
                {
                    "sourceExternalId": "doc-1",
                    "title": "Cornerstone Overview",
                    "content": "Cornerstone is a shared organizational context layer. Official context must preserve provenance.",
                    "sourceUrl": "https://example.internal/doc-1",
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
    return {
        "source_id": source_id,
        "artifact_id": body["artifacts"][0]["id"],
        "evidence_id": evidence_id,
    }
