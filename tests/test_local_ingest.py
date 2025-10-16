from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cornerstone.config import Settings
from cornerstone.ingestion import DocumentIngestor, IngestionJobManager, IngestionResult
from cornerstone.local_ingest import ingest_directory, list_directories, load_manifest, resolve_local_path
from cornerstone.projects import DocumentMetadata
from cornerstone.scripts import ingest_local


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


def test_resolve_local_path_accepts_descendant(tmp_path: Path) -> None:
    base_dir = tmp_path / "data"
    allowed_dir = base_dir / "allowed"
    allowed_dir.mkdir(parents=True, exist_ok=True)

    result = resolve_local_path(base_dir, "allowed/file.txt")

    assert result == (allowed_dir / "file.txt").resolve()


def test_resolve_local_path_rejects_escape(tmp_path: Path) -> None:
    base_dir = tmp_path / "data"
    base_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        resolve_local_path(base_dir, "../base-evil")


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


def test_missing_embedding_model_triggers_reingest(tmp_path: Path) -> None:
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
    assert backfill.calls == ["docs/sample.txt"]
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


def test_list_directories_treats_dot_as_root(tmp_path: Path) -> None:
    base_dir = tmp_path / "data"
    (base_dir / "alpha").mkdir(parents=True, exist_ok=True)
    (base_dir / "beta").mkdir()

    root_entries = list_directories(base_dir)
    dot_entries = list_directories(base_dir, ".")

    assert dot_entries == root_entries


def test_cli_ingest_waits_for_throttled_slot(monkeypatch, tmp_path: Path, capsys) -> None:
    job_manager = IngestionJobManager(
        max_active_per_project=1,
        throttle_poll_interval=0.01,
    )
    blocking_job = job_manager.create_job("proj", "blocking")
    job_manager.mark_processing(blocking_job.id)

    data_dir = tmp_path / "data"
    local_dir = data_dir / "local"
    docs_dir = local_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "sample.txt").write_text("hello", encoding="utf-8")

    settings = Settings()
    settings.data_dir = str(data_dir)
    settings.local_data_dir = str(local_dir)

    ingestor = DummyIngestor("model-x")

    class StubProjectStore:
        def __init__(self, project_id: str) -> None:
            self._projects = [SimpleNamespace(id=project_id)]

        def list_projects(self) -> list[SimpleNamespace]:
            return self._projects

    services = SimpleNamespace(
        project_store=StubProjectStore("proj"),
        ingestion_service=ingestor,
        settings=settings,
        ingestion_jobs=job_manager,
    )
    app = SimpleNamespace(state=SimpleNamespace(services=services))
    monkeypatch.setattr(ingest_local, "create_app", lambda: app)

    results: list[int] = []

    def run_cli() -> None:
        results.append(ingest_local.main(["docs"]))

    thread = threading.Thread(target=run_cli, daemon=True)
    thread.start()

    cli_job_id = None
    deadline = time.time() + 2.0
    while time.time() < deadline:
        jobs = job_manager.list_for_project("proj")
        if len(jobs) >= 2:
            for job in jobs:
                if job.id != blocking_job.id:
                    cli_job_id = job.id
            if cli_job_id and job_manager.get(cli_job_id).status == "throttled":
                break
        time.sleep(0.01)

    assert cli_job_id is not None
    assert job_manager.get(cli_job_id).status == "throttled"

    job_manager.mark_completed(
        blocking_job.id,
        DocumentMetadata(
            id="blocking",
            filename="blocking",
            chunk_count=1,
            created_at=DocumentIngestor._now(),
            size_bytes=0,
            title="blocking",
            content_type="application/x-directory",
        ),
    )

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert results == [0]

    job_record = job_manager.get(cli_job_id)
    assert job_record is not None
    assert job_record.status == "completed"
    assert job_record.processed_files == job_record.total_files == 1

    output = capsys.readouterr().out
    assert "Ingested" in output
    assert "docs" in output
