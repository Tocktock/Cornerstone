"""Qdrant vector store helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

from qdrant_client import QdrantClient, models

from .config import Settings


@dataclass(slots=True)
class VectorRecord:
    """Payload representing a vector to be stored in Qdrant."""

    id: int | str
    vector: Sequence[float]
    payload: dict[str, Any] | None = None


@dataclass(slots=True)
class SearchResult:
    """Result item returned from a similarity search."""

    id: int | str
    score: float
    payload: dict[str, Any] | None


class QdrantVectorStore:
    """High-level wrapper around the Qdrant client."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        *,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        if vector_size <= 0:
            msg = "vector_size must be a positive integer"
            raise ValueError(msg)

        self._client = client
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = distance

    @classmethod
    def from_settings(cls, settings: Settings, *, vector_size: int) -> "QdrantVectorStore":
        """Instantiate the store using application settings."""

        client = QdrantClient(**settings.qdrant_client_kwargs())
        return cls(client, settings.qdrant_collection, vector_size=vector_size)

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def vector_size(self) -> int:
        return self._vector_size

    def ensure_collection(self, *, force_recreate: bool = False) -> None:
        """Ensure the Qdrant collection exists with the expected vector size."""

        exists = self._client.collection_exists(self._collection_name)

        if force_recreate:
            if exists:
                self._client.delete_collection(self._collection_name)
            self._create_collection()
            return

        if not exists:
            self._create_collection()
            return

        info = self._client.get_collection(self._collection_name)
        existing_size = info.config.params.vectors.size
        if existing_size != self._vector_size:
            msg = (
                "Existing collection vector size "
                f"({existing_size}) does not match expected size {self._vector_size}."
            )
            raise ValueError(msg)

    def upsert(self, records: Sequence[VectorRecord], *, wait: bool = True) -> None:
        """Insert or update vectors in the collection."""

        if not records:
            return

        ids: list[int | str] = []
        vectors: list[list[float]] = []
        payloads: list[dict[str, Any]] = []

        for record in records:
            vector_list = list(record.vector)
            if len(vector_list) != self._vector_size:
                msg = (
                    f"Vector for id {record.id!r} has length {len(vector_list)}, "
                    f"expected {self._vector_size}."
                )
                raise ValueError(msg)

            ids.append(record.id)
            vectors.append(vector_list)
            payloads.append(record.payload or {})

        batch = models.Batch(ids=ids, vectors=vectors, payloads=payloads)
        self._client.upsert(
            collection_name=self._collection_name,
            points=batch,
            wait=wait,
        )

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        with_payload: bool = True,
    ) -> List[SearchResult]:
        """Search for similar vectors in the collection."""

        query_vector = list(vector)
        if len(query_vector) != self._vector_size:
            msg = f"Query vector has length {len(query_vector)}, expected {self._vector_size}."
            raise ValueError(msg)

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=with_payload,
        )

        results: list[SearchResult] = []
        for point in response.points:
            payload = dict(point.payload) if with_payload and point.payload is not None else None
            results.append(SearchResult(id=point.id, score=point.score, payload=payload))
        return results

    def _create_collection(self) -> None:
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(size=self._vector_size, distance=self._distance),
        )

    def count(self) -> int:
        """Return the number of stored vectors."""

        return self._client.count(self._collection_name).count

    def delete_by_ids(self, ids: Iterable[int | str]) -> None:
        """Remove vectors from the collection by their identifiers."""

        id_list = list(ids)
        if not id_list:
            return

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=id_list),
        )
