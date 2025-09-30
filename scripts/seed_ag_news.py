"""Seed Qdrant with a small slice of the AG News dataset using the active embedding backend."""

from __future__ import annotations

import argparse
from uuid import uuid4

from datasets import load_dataset

from cornerstone import EmbeddingService, QdrantVectorStore, Settings, VectorRecord


def main(limit: int, collection_action: str) -> None:
    settings = Settings.from_env()
    embedding = EmbeddingService(settings)
    store = QdrantVectorStore.from_settings(settings, vector_size=embedding.dimension)

    if collection_action == "recreate":
        store.ensure_collection(force_recreate=True)
    else:
        store.ensure_collection()

    dataset = load_dataset("ag_news", split=f"train[:{limit}]")

    records: list[VectorRecord] = []
    for row in dataset:
        title = row.get("title")
        description = row.get("description")
        text_field = row.get("text")

        if title and description:
            text = f"{title} - {description}"
        elif title and text_field:
            text = f"{title} - {text_field}"
        else:
            text = text_field or title or description or ""

        vector = embedding.embed_one(text)
        records.append(
            VectorRecord(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "text": text,
                    "label": int(row["label"]),
                },
            )
        )

    store.upsert(records)
    print(f"Seeded {len(records)} AG News entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200, help="Number of news items to ingest")
    parser.add_argument(
        "--collection-action",
        choices={"ensure", "recreate"},
        default="ensure",
        help="Whether to recreate the collection before ingesting",
    )
    args = parser.parse_args()
    main(limit=args.limit, collection_action=args.collection_action)
