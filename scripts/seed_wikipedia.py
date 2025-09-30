"""Seed Qdrant with Cohere Wikipedia snippets using the active embedding backend."""

from __future__ import annotations

import argparse
from uuid import uuid4

from datasets import load_dataset

from cornerstone import EmbeddingService, QdrantVectorStore, Settings, VectorRecord


def chunked(iterable, size: int):
    batch: list[dict] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def main(limit: int, batch_size: int, collection_action: str, language: str) -> None:
    settings = Settings.from_env()
    embedding = EmbeddingService(settings)
    store = QdrantVectorStore.from_settings(settings, vector_size=embedding.dimension)

    if collection_action == "recreate":
        store.ensure_collection(force_recreate=True)
    else:
        store.ensure_collection()

    dataset = load_dataset(
        "Cohere/wikipedia-2023-11-embed-multilingual-v3",
        language,
        split=f"train[:{limit}]",
    )

    total = 0
    for batch in chunked(dataset, batch_size):
        texts = [row["text"] for row in batch]
        vectors = embedding.embed(texts)

        records = []
        for row, vector in zip(batch, vectors, strict=True):
            payload = {
                "title": row.get("title"),
                "url": row.get("url"),
                "text": row.get("text"),
            }
            records.append(VectorRecord(id=str(uuid4()), vector=vector, payload=payload))

        store.upsert(records)
        total += len(records)
        print(f"Indexed batch of {len(records)} (total {total})")

    print(f"Done seeding {total} Wikipedia excerpts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500, help="Number of rows to ingest")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Rows per embedding request",
    )
    parser.add_argument(
        "--collection-action",
        choices={"ensure", "recreate"},
        default="ensure",
        help="Whether to recreate the collection before ingesting",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code from the dataset configs (e.g. en, de, fr)",
    )
    args = parser.parse_args()
    main(
        limit=args.limit,
        batch_size=args.batch_size,
        collection_action=args.collection_action,
        language=args.language,
    )
