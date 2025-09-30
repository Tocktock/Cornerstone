from __future__ import annotations

import math
import os
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from cornerstone.vector_store import QdrantVectorStore, VectorRecord


@pytest.fixture()
def qdrant_store() -> QdrantVectorStore:
    client = QdrantClient(path=':memory:')
    store = QdrantVectorStore(client, 'test-collection', vector_size=3)
    store.ensure_collection()
    return store


def test_ensure_collection_dimension_mismatch() -> None:
    client = QdrantClient(path=':memory:')
    store = QdrantVectorStore(client, 'test-collection', vector_size=3)
    store.ensure_collection()

    mismatch_store = QdrantVectorStore(client, 'test-collection', vector_size=2)
    with pytest.raises(ValueError):
        mismatch_store.ensure_collection()


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
