from types import SimpleNamespace

from qdrant_client import models

from cornerstone.ingestion import ProjectVectorStoreManager


class FakeClient:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.calls: list[tuple] = []

    def collection_exists(self, name: str) -> bool:
        self.calls.append(("exists", name))
        return name in self.collections

    def create_collection(self, **kwargs):
        name = kwargs["collection_name"]
        self.collections.add(name)
        self.calls.append(("create", name))

    def delete_collection(self, name: str) -> None:
        self.calls.append(("delete", name))
        self.collections.discard(name)

    def create_payload_index(self, collection_name: str, field_name: str, field_schema) -> None:
        self.calls.append(("index", collection_name, field_name))

    def get_collection(self, name: str) -> SimpleNamespace:
        # Provide minimal info if queried; size value is irrelevant for this test.
        self.calls.append(("get", name))
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=3))))


def test_purge_project_recreates_collection_and_indexes() -> None:
    client = FakeClient()
    manager = ProjectVectorStoreManager(
        client_factory=lambda: client,
        vector_size=3,
        distance=models.Distance.COSINE,
        collection_name_fn=lambda project_id: f"collection-{project_id}",
    )

    manager.get_store("alpha")
    client.calls.clear()

    assert manager.purge_project("alpha") is True

    assert ("exists", "collection-alpha") in client.calls
    assert ("delete", "collection-alpha") in client.calls
    assert ("create", "collection-alpha") in client.calls

    index_calls = [call for call in client.calls if call[0] == "index"]
    assert index_calls, "Expected payload indexes to be recreated after purge"
