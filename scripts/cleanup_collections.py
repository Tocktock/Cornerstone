"""Remove Qdrant collections matching the configured base name."""

from __future__ import annotations

import argparse
from pathlib import Path

from cornerstone import Settings
from qdrant_client import QdrantClient


def main(prefix: str | None, *, dry_run: bool) -> None:
    settings = Settings.from_env()
    client = QdrantClient(**settings.qdrant_client_kwargs())

    base_prefix = prefix or settings.qdrant_collection
    collections = client.get_collections().collections

    targets = [item.name for item in collections if item.name.startswith(base_prefix)]

    if not targets:
        print("No collections matched the prefix.")
        return

    for name in targets:
        if dry_run:
            print(f"Would delete collection: {name}")
        else:
            client.delete_collection(name)
            print(f"Deleted collection: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Collection name prefix to match (default: current Qdrant collection from settings)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching collections without deleting them",
    )
    args = parser.parse_args()
    main(prefix=args.prefix, dry_run=args.dry_run)
