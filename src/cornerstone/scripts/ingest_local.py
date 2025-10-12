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
    job_manager = getattr(state, "ingestion_jobs", None)
    job = None
    if job_manager is not None:
        try:
            job_filename = str(target_dir.relative_to(base_dir))
        except ValueError:  # pragma: no cover - fallback guard
            job_filename = target_dir.name or str(target_dir)
        label = job_filename or target_dir.name or str(target_dir)
        job = job_manager.create_job(project_id, label)

    metadata = ingest_directory(
        project_id=project_id,
        target_dir=target_dir,
        base_dir=base_dir,
        ingestion_service=state.ingestion_service,
        manifest_path=manifest_path,
        job_manager=job_manager if job else None,
        job_id=job.id if job else None,
    )
    chunks = metadata.chunk_count
    file_suffix = ""
    if job_manager is not None and job is not None:
        job_state = job_manager.get(job.id)
        if job_state and job_state.processed_files is not None and job_state.total_files is not None:
            file_suffix = f" ({job_state.processed_files}/{job_state.total_files} files)"
    print(f"Ingested {chunks} chunk{'s' if chunks != 1 else ''} from {metadata.filename}{file_suffix}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
