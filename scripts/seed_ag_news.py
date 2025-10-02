"""Seed Qdrant with a small slice of the AG News dataset using the active embedding backend."""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from datasets import load_dataset
from qdrant_client import QdrantClient

from cornerstone import EmbeddingService, Settings, VectorRecord
from cornerstone.vector_store import QdrantVectorStore
from cornerstone.projects import ProjectStore


def main(limit: int, collection_action: str, project_name: str | None) -> None:
    settings = Settings.from_env()
    print(
        "[seed] Using embedding backend '",
        settings.embedding_model,
        "' and chat backend '",
        settings.chat_backend,
        "'.",
        sep="",
    )
    embedding = EmbeddingService(settings)
    print(f"[seed] Embedding dimension resolved to {embedding.dimension}.")

    project_store = ProjectStore(Path(settings.data_dir).resolve(), default_project_name=settings.default_project_name)
    project = resolve_project(project_store, project_name)
    print(f"[seed] Target project: {project.name} (id={project.id}).")

    client = QdrantClient(**settings.qdrant_client_kwargs())
    collection_name = settings.project_collection_name(project.id)
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        vector_size=embedding.dimension,
    )

    if collection_action == "recreate":
        print("[seed] Recreating collection before ingesting data.")
        store.ensure_collection(force_recreate=True)
    else:
        print("[seed] Ensuring collection exists before ingesting data.")
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
        if len(records) % 25 == 0:
            print(f"[seed] Embedded {len(records) + 1} records...")
        records.append(
            VectorRecord(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "text": text,
                    "label": int(row["label"]),
                    "project_id": project.id,
                },
            )
        )

    print(f"[seed] Upserting {len(records)} vectors to Qdrant...")
    store.upsert(records)
    print(f"[seed] Seeded {len(records)} AG News entries.")


def resolve_project(project_store: ProjectStore, project_name: str | None):
    if project_name:
        project = project_store.get_project(project_name) or project_store.find_by_name(project_name)
        if project:
            return project
        return project_store.create_project(project_name, "Created via seed script")
    projects = project_store.list_projects()
    if not projects:
        raise RuntimeError("No projects configured")
    return projects[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200, help="Number of news items to ingest")
    parser.add_argument(
        "--collection-action",
        choices={"ensure", "recreate"},
        default="ensure",
        help="Whether to recreate the collection before ingesting",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project name or identifier to ingest into (default: first project)",
    )
    args = parser.parse_args()
    main(limit=args.limit, collection_action=args.collection_action, project_name=args.project)
