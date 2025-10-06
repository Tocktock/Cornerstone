"""CLI for ingesting local data directories into Cornerstone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cornerstone.app import create_app
from cornerstone.local_ingest import ingest_directory, resolve_local_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest local data directories into Cornerstone")
    parser.add_argument(
        "--project",
        dest="project_id",
        help="Project identifier to ingest into (defaults to the first project)",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="",
        help="Relative path under data/local to ingest (default: entire local directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    app = create_app()
    state = app.state.services

    project_id = args.project_id or state.project_store.list_projects()[0].id
    base_dir = Path(state.settings.local_data_dir).resolve()
    try:
        target_dir = resolve_local_path(base_dir, args.path)
    except ValueError as exc:  # pragma: no cover - CLI validation
        parser.error(str(exc))
        return 1

    if not target_dir.exists() or not target_dir.is_dir():  # pragma: no cover - CLI validation
        parser.error(f"Directory '{args.path}' does not exist under {base_dir}")
        return 1

    manifest_path = Path(state.settings.data_dir).resolve() / "manifests" / f"{project_id}.json"
    metadata = ingest_directory(
        project_id=project_id,
        target_dir=target_dir,
        base_dir=base_dir,
        ingestion_service=state.ingestion_service,
        manifest_path=manifest_path,
    )
    chunks = metadata.chunk_count
    print(f"Ingested {chunks} chunk{'s' if chunks != 1 else ''} from {metadata.filename}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
