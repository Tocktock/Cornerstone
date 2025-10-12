from __future__ import annotations

from types import SimpleNamespace

from cornerstone.vector_store import QdrantVectorStore


def _collection_info(size: int) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=size)
            )
        )
    )


def test_ensure_collection_creates_when_missing(monkeypatch):
    calls = []

    class DummyClient:
        def collection_exists(self, name):
            calls.append(("exists", name))
            return False

        def create_collection(self, collection_name, vectors_config):
            calls.append(("create", collection_name, vectors_config.size))

    store = QdrantVectorStore(DummyClient(), "test", vector_size=1024)
    store.ensure_collection()

    assert calls == [
        ("exists", "test"),
        ("create", "test", 1024),
    ]


def test_ensure_collection_recreates_on_dimension_mismatch(monkeypatch):
    calls = []

    class DummyClient:
        def collection_exists(self, name):
            calls.append(("exists", name))
            return True

        def get_collection(self, name):
            calls.append(("get", name))
            return _collection_info(size=3072)

        def delete_collection(self, name):
            calls.append(("delete", name))

        def create_collection(self, collection_name, vectors_config):
            calls.append(("create", collection_name, vectors_config.size))

    store = QdrantVectorStore(DummyClient(), "test", vector_size=4096)
    store.ensure_collection()

    assert calls == [
        ("exists", "test"),
        ("get", "test"),
        ("delete", "test"),
        ("create", "test", 4096),
    ]


def test_ensure_collection_noop_when_dimensions_match(monkeypatch):
    calls = []

    class DummyClient:
        def collection_exists(self, name):
            calls.append(("exists", name))
            return True

        def get_collection(self, name):
            calls.append(("get", name))
            return _collection_info(size=1024)

        def delete_collection(self, name):
            calls.append(("delete", name))

        def create_collection(self, collection_name, vectors_config):
            calls.append(("create", collection_name, vectors_config.size))

    store = QdrantVectorStore(DummyClient(), "test", vector_size=1024)
    store.ensure_collection()

    assert calls == [
        ("exists", "test"),
        ("get", "test"),
    ]
