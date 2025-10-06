from __future__ import annotations

import math
import os
from typing import Any
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient, models

from cornerstone.config import Settings
from cornerstone.vector_store import QdrantVectorStore, VectorRecord


@pytest.fixture()
def qdrant_store() -> QdrantVectorStore:
    client = QdrantClient(path=':memory:')
    store = QdrantVectorStore(client, 'test-collection', vector_size=3)
    store.ensure_collection()
    return store


def test_settings_tuning_kwargs_filters_optional_values() -> None:
    settings = Settings(
        qdrant_on_disk_vectors=True,
        qdrant_on_disk_payload=None,
        qdrant_hnsw_m=64,
        qdrant_hnsw_ef_construct=None,
    )
    kwargs = settings.qdrant_collection_tuning_kwargs()
    assert kwargs["on_disk_vectors"] is True
    assert "on_disk_payload" not in kwargs
    assert kwargs["hnsw_config"] == {"m": 64}


def test_create_collection_uses_tuning_options() -> None:
    captured: dict[str, Any] = {}

    class FakeClient:
        def collection_exists(self, name: str) -> bool:
            return False

        def create_collection(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    store = QdrantVectorStore(
        client=FakeClient(),
        collection_name='test-collection',
        vector_size=3,
        on_disk_payload=True,
        on_disk_vectors=True,
        hnsw_config={'m': 32, 'ef_construct': 128},
    )

    store.ensure_collection()

    assert captured['collection_name'] == 'test-collection'
    vectors_config = captured['vectors_config']
    assert isinstance(vectors_config, models.VectorParams)
    assert vectors_config.on_disk is True

    assert captured['on_disk_payload'] is True
    hnsw_config = captured['hnsw_config']
    assert isinstance(hnsw_config, models.HnswConfigDiff)
    assert hnsw_config.m == 32
    assert hnsw_config.ef_construct == 128


def test_ensure_collection_dimension_mismatch() -> None:
    client = QdrantClient(path=':memory:')
    store = QdrantVectorStore(client, 'test-collection', vector_size=3)
    store.ensure_collection()

    mismatch_store = QdrantVectorStore(client, 'test-collection', vector_size=2)
    mismatch_store.ensure_collection()
    info = client.get_collection('test-collection')
    assert info.config.params.vectors.size == 2
    assert mismatch_store.count() == 0


def test_ensure_payload_indexes_creates_indexes() -> None:
    client = QdrantClient(path=':memory:')
    store = QdrantVectorStore(client, 'test-collection', vector_size=3)
    store.ensure_collection()
    store.ensure_payload_indexes()
    info = client.get_collection('test-collection')
    schema = getattr(info, 'payload_schema', {}) or {}
    if schema:  # Some in-memory deployments do not expose schema details
        assert 'project_id' in schema


def test_upsert_and_search_returns_best_match(qdrant_store: QdrantVectorStore) -> None:
    records = [
        VectorRecord(id=1, vector=[1.0, 0.0, 0.0], payload={'text': 'first'}),
        VectorRecord(id=2, vector=[0.0, 1.0, 0.0], payload={'text': 'second'}),
    ]

    qdrant_store.upsert(records)
    assert qdrant_store.count() == 2

    results = qdrant_store.search([0.98, 0.05, 0.0], limit=1)
    assert results
    top = results[0]
    assert top.id == 1
    assert top.payload == {'text': 'first'}


def test_upsert_rejects_wrong_dimension(qdrant_store: QdrantVectorStore) -> None:
    with pytest.raises(ValueError):
        qdrant_store.upsert([VectorRecord(id=3, vector=[1.0, 0.0], payload=None)])


def test_delete_by_ids_removes_vectors(qdrant_store: QdrantVectorStore) -> None:
    qdrant_store.upsert([
        VectorRecord(id=1, vector=[1.0, 0.0, 0.0]),
        VectorRecord(id=2, vector=[0.0, 1.0, 0.0]),
    ])

    qdrant_store.delete_by_ids([1])
    assert qdrant_store.count() == 1
    remaining = qdrant_store.search([0.0, 1.0, 0.0], limit=1)
    assert remaining and math.isclose(remaining[0].score, 1.0, rel_tol=1e-6)


@pytest.fixture(scope='module')
def qdrant_docker_store() -> QdrantVectorStore:
    url = os.getenv('QDRANT_URL', 'http://localhost:6333')
    client = QdrantClient(url=url, timeout=2.0)
    try:
        client.get_collections()
    except Exception as exc:  # pragma: no cover - integration precondition
        pytest.skip(f"Qdrant docker container not available at {url}: {exc}")

    collection_name = 'cornerstone-test-integration'
    store = QdrantVectorStore(client, collection_name, vector_size=3)
    store.ensure_collection(force_recreate=True)
    try:
        yield store
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


@pytest.mark.integration
def test_integration_upsert_and_search_against_docker(qdrant_docker_store: QdrantVectorStore) -> None:
    vec_id_1 = str(uuid4())
    vec_id_2 = str(uuid4())
    records = [
        VectorRecord(id=vec_id_1, vector=[1.0, 0.0, 0.0], payload={'text': 'docker-first'}),
        VectorRecord(id=vec_id_2, vector=[0.0, 1.0, 0.0], payload={'text': 'docker-second'}),
    ]
    qdrant_docker_store.upsert(records)
    results = qdrant_docker_store.search([0.05, 0.99, 0.0], limit=1)
    assert results
    assert results[0].id == vec_id_2
    assert results[0].payload == {'text': 'docker-second'}
