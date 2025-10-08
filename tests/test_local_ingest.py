from __future__ import annotations

import json
from pathlib import Path

from cornerstone.ingestion import DocumentIngestor, IngestionResult
from cornerstone.local_ingest import ingest_directory, load_manifest
from cornerstone.projects import DocumentMetadata


class DummyIngestor:
    def __init__(self, model_id: str | None) -> None:
        self.embedding_model_id = model_id
        self.calls: list[str] = []

    def ingest_bytes(self, project_id: str, *, filename: str, data: bytes, content_type: str | None = None) -> IngestionResult:
        self.calls.append(filename)
        metadata = DocumentMetadata(
            id=f"doc-{len(self.calls)}",
            filename=filename,
            chunk_count=1,
            created_at=DocumentIngestor._now(),
            size_bytes=len(data),
            title=None,
            content_type=content_type,
        )
        return IngestionResult(document=metadata, chunks_ingested=1)


def _setup_directory(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    base_dir = tmp_path / "data"
    target_dir = base_dir / "docs"
    target_dir.mkdir(parents=True, exist_ok=True)
    source_file = target_dir / "sample.txt"
    source_file.write_text("hello world", encoding="utf-8")
    manifest_path = tmp_path / "manifests" / "proj.json"
    return base_dir, target_dir, source_file, manifest_path


def test_ingest_directory_skips_when_embedding_model_matches(tmp_path: Path) -> None:
    base_dir, target_dir, _, manifest_path = _setup_directory(tmp_path)

    ingestor_a = DummyIngestor("model-a")
    ingest_directory(
        project_id="proj",
        target_dir=target_dir,
        base_dir=base_dir,
        ingestion_service=ingestor_a,
        manifest_path=manifest_path,
    )
    assert ingestor_a.calls == ["docs/sample.txt"]
    manifest = load_manifest(manifest_path)
    entry = manifest.get("docs/sample.txt")
    assert entry and entry.get("embedding_model") == "model-a"

    ingestor_again = DummyIngestor("model-a")
    ingest_directory(
        project_id="proj",
        target_dir=target_dir,
        base_dir=base_dir,
        ingestion_service=ingestor_again,
        manifest_path=manifest_path,
    )
    assert ingestor_again.calls == []


def test_ingest_directory_reprocesses_when_model_changes(tmp_path: Path) -> None:
    base_dir, target_dir, _, manifest_path = _setup_directory(tmp_path)

    ingestor_a = DummyIngestor("model-a")
    ingest_directory(
        project_id="proj",
        target_dir=target_dir,
        base_dir=base_dir,
        ingestion_service=ingestor_a,
        manifest_path=manifest_path,
    )

    ingestor_b = DummyIngestor("model-b")
    ingest_directory(
        project_id="proj",
        target_dir=target_dir,
        base_dir=base_dir,
        ingestion_service=ingestor_b,
        manifest_path=manifest_path,
    )
    assert ingestor_b.calls == ["docs/sample.txt"]
    manifest = load_manifest(manifest_path)
    entry = manifest.get("docs/sample.txt")
    assert entry and entry.get("embedding_model") == "model-b"


def test_missing_embedding_model_is_backfilled_without_reingesting(tmp_path: Path) -> None:
    base_dir, target_dir, _, manifest_path = _setup_directory(tmp_path)

    ingestor_a = DummyIngestor("model-a")
    ingest_directory(
        project_id="proj",
        target_dir=target_dir,
        base_dir=base_dir,
        ingestion_service=ingestor_a,
        manifest_path=manifest_path,
    )

    manifest = load_manifest(manifest_path)
    entry = manifest["docs/sample.txt"]
    entry.pop("embedding_model", None)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    backfill = DummyIngestor("model-a")
    ingest_directory(
        project_id="proj",
        target_dir=target_dir,
        base_dir=base_dir,
        ingestion_service=backfill,
        manifest_path=manifest_path,
    )
    assert backfill.calls == []
    updated_manifest = load_manifest(manifest_path)
    assert updated_manifest["docs/sample.txt"].get("embedding_model") == "model-a"


def test_reingest_after_manifest_removed_processes_again(tmp_path: Path) -> None:
    base_dir, target_dir, _, manifest_path = _setup_directory(tmp_path)

    ingestor_a = DummyIngestor("model-a")
    ingest_directory(
        project_id="proj",
        target_dir=target_dir,
        base_dir=base_dir,
        ingestion_service=ingestor_a,
        manifest_path=manifest_path,
    )
    assert manifest_path.exists()

    manifest_path.unlink()

    ingestor_again = DummyIngestor("model-a")
    ingest_directory(
        project_id="proj",
        target_dir=target_dir,
        base_dir=base_dir,
        ingestion_service=ingestor_again,
        manifest_path=manifest_path,
    )
    assert ingestor_again.calls == ["docs/sample.txt"]
