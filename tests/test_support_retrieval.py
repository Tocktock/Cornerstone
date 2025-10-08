from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from qdrant_client import QdrantClient, models

from cornerstone.chat import SupportAgentService
from cornerstone.config import Settings
from cornerstone.glossary import Glossary
from cornerstone.ingestion import DocumentIngestor, ProjectVectorStoreManager
from cornerstone.fts import FTSIndex
from cornerstone.personas import PersonaOverrides
from cornerstone.projects import ProjectStore


class FakeEmbedding:
    dimension = 3

    def embed(self, texts):
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text):
        length = float(len(text)) or 1.0
        return [length, length / 2.0, 1.0]


def build_service(tmpdir: Path, *, reranker=None):
    data_dir = tmpdir.resolve()
    settings = Settings(
        data_dir=str(data_dir),
        default_project_name="Retrieval Test",
        local_data_dir=str(data_dir / "local"),
    )
    Path(settings.local_data_dir).mkdir(parents=True, exist_ok=True)
    project_store = ProjectStore(data_dir, default_project_name=settings.default_project_name)
    project = project_store.list_projects()[0]

    client = QdrantClient(path=":memory:")
    store_manager = ProjectVectorStoreManager(
        client_factory=lambda: client,
        vector_size=3,
        distance=models.Distance.COSINE,
        collection_name_fn=lambda pid: f"retrieval-{pid}",
    )
    store_manager.get_store(project.id)

    embedding = FakeEmbedding()
    fts_index = FTSIndex(data_dir / "fts" / "fts.sqlite")
    ingestion = DocumentIngestor(
        embedding_service=embedding,
        store_manager=store_manager,
        project_store=project_store,
        fts_index=fts_index,
    )

    service = SupportAgentService(
        settings=settings,
        embedding_service=embedding,
        store_manager=store_manager,
        glossary=Glossary(),
        project_store=project_store,
        persona_store=None,
        fts_index=fts_index,
        reranker=reranker,
    )

    return service, ingestion, project, project_store


def test_support_service_fuses_keyword_and_semantic_results():
    tmpdir = Path(tempfile.mkdtemp(prefix="cornerstone-retrieval-"))
    try:
        service, ingestion, project, project_store = build_service(tmpdir)

        semantic_doc = (
            "# Troubleshooting Guide\n\n"
            "## Error 42\n"
            "If you encounter error 42, restart the widget service and clear cached data."
        )
        ingestion.ingest_bytes(
            project.id,
            filename="semantic.md",
            data=semantic_doc.encode("utf-8"),
            content_type="text/markdown",
        )

        service._fts.upsert_chunks(  # type: ignore[attr-defined]
            project_id=project.id,
            doc_id="kw-doc",
            entries=[
                {
                    "chunk_id": "KW-001",
                    "text": "Legacy runbook for widget outage error 42",
                    "title": "Legacy Runbook",
                    "metadata": {"source": "runbook.txt"},
                }
            ],
        )

        persona = service._resolve_persona(project)
        options = service._persona_options(persona)

        _, keyword_chunks = service._build_context(
            project,
            persona,
            "legacy outage runbook",
            [],
            options,
        )
        assert any(chunk["title"] == "Legacy Runbook" for chunk in keyword_chunks)

        _, semantic_chunks = service._build_context(
            project,
            persona,
            "restart widget error 42",
            [],
            options,
        )
        assert any("Error 42" in chunk.get("title", "") for chunk in semantic_chunks)

        _, fused_chunks = service._build_context(
            project,
            persona,
            "widget outage error 42 runbook",
            [],
            options,
        )
        fused_titles = {chunk["title"] for chunk in fused_chunks}
        assert "Legacy Runbook" in fused_titles
        assert any("Error 42" in title for title in fused_titles)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_support_service_applies_reranker_when_present():
    class DummyReranker:
        name = "dummy"

        def __init__(self) -> None:
            self.called_with: list[str] = []

        def rerank(self, query, *, query_embedding, chunks, top_k=None):
            self.called_with.append(query)
            return list(reversed(list(chunks)))

    tmpdir = Path(tempfile.mkdtemp(prefix="cornerstone-retrieval-"))
    try:
        dummy = DummyReranker()
        service, ingestion, project, project_store = build_service(tmpdir, reranker=dummy)
        persona = service._resolve_persona(project)
        fused = [
            {
                "chunk_id": "c1",
                "title": "First",
                "text": "Alpha",
                "origin": ["vector"],
            },
            {
                "chunk_id": "c2",
                "title": "Second",
                "text": "Bravo",
                "origin": ["keyword"],
            },
        ]

        reranked = service._apply_reranker(
            project,
            "follow",
            query_vector=service._embedding.embed_one("follow"),
            fused_chunks=fused,
            limit=len(fused),
        )

        assert dummy.called_with == ["follow"]
        assert reranked == list(reversed(fused))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_support_service_uses_project_glossary_entries():
    tmpdir = Path(tempfile.mkdtemp(prefix="cornerstone-retrieval-"))
    try:
        service, ingestion, project, project_store = build_service(tmpdir)
        project_store.create_glossary_entry(
            project.id,
            term="SLA",
            definition="Service level agreement defining support timelines.",
            keywords=["uptime", "contract"],
        )
        persona = service._resolve_persona(project)
        options = service._persona_options(persona)
        context, _ = service._build_context(project, persona, "uptime SLA", [], options)
        assert any(definition.startswith("SLA:") for definition in context.definitions)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_persona_options_respect_project_overrides():
    tmpdir = Path(tempfile.mkdtemp(prefix="cornerstone-retrieval-"))
    try:
        service, ingestion, project, project_store = build_service(tmpdir)
        updated = project_store.configure_persona(
            project.id,
            persona_id=None,
            overrides=PersonaOverrides(
                retrieval_top_k=5,
                glossary_top_k=1,
                chat_temperature=0.65,
                chat_max_tokens=240,
            ),
        )
        persona = service._resolve_persona(updated)
        options = service._persona_options(persona)
        assert options.retrieval_top_k == 5
        assert options.glossary_top_k == 1
        assert abs(options.chat_temperature - 0.65) < 1e-6
        assert options.chat_max_tokens == 240
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
