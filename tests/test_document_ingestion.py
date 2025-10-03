from __future__ import annotations

import io
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

try:  # pragma: no cover - optional dependency
    from pypdf import PdfReader  # noqa: F401
    HAS_PYPDF = True
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore
    HAS_PYPDF = False

from cornerstone.app import create_app
from cornerstone.chat import SupportAgentService
from cornerstone.config import Settings
from cornerstone.glossary import Glossary
from cornerstone.ingestion import DocumentIngestor, ProjectVectorStoreManager
from cornerstone.personas import PersonaStore
from cornerstone.projects import ProjectStore


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.dimension = 3

    def embed(self, texts):
        return [[float(len(text)), 0.0, 0.0] for text in texts]

    def embed_one(self, text: str):
        return [float(len(text)), 0.0, 0.0]


# Minimal PDF with extractable text ("PDF sample text")
SAMPLE_PDF_BYTES = b"%PDF-1.1\n1 0 obj<<>>endobj\n2 0 obj<< /Length 56 >>stream\nBT /F1 12 Tf 72 720 Td (PDF sample text) Tj ET\nendstream\nendobj\n3 0 obj<< /Type /Page /Parent 4 0 R /MediaBox [0 0 612 792] /Contents 2 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj\n4 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n6 0 obj<< /Type /Catalog /Pages 4 0 R >>endobj\nxref\n0 7\n0000000000 65535 f \n0000000010 00000 n \n0000000056 00000 n \n0000000125 00000 n \n0000000230 00000 n \n0000000302 00000 n \n0000000373 00000 n \ntrailer<< /Root 6 0 R /Size 7 >>\nstartxref\n430\n%%EOF"


def build_app():
    tmpdir = Path(tempfile.mkdtemp(prefix="cornerstone-ingest-"))
    settings = Settings(data_dir=str(tmpdir), default_project_name="Project One")
    embedding = FakeEmbeddingService()
    project_store = ProjectStore(tmpdir, default_project_name=settings.default_project_name)
    persona_store = PersonaStore(tmpdir)
    default_project = project_store.list_projects()[0]

    client = QdrantClient(path=":memory:")
    store_manager = ProjectVectorStoreManager(
        client_factory=lambda: client,
        vector_size=embedding.dimension,
        distance=models.Distance.COSINE,
        collection_name_fn=settings.project_collection_name,
    )
    store_manager.get_store(default_project.id)

    glossary = Glossary()
    chat_service = SupportAgentService(
        settings=settings,
        embedding_service=embedding,  # type: ignore[arg-type]
        store_manager=store_manager,
        glossary=glossary,
        persona_store=persona_store,
    )
    ingestion_service = DocumentIngestor(embedding, store_manager, project_store)

    app = create_app(
        settings=settings,
        embedding_service=embedding,  # type: ignore[arg-type]
        glossary=glossary,
        project_store=project_store,
        persona_store=persona_store,
        store_manager=store_manager,
        chat_service=chat_service,
        ingestion_service=ingestion_service,
    )

    return TestClient(app)


def test_uploads_across_multiple_projects():
    client = build_app()
    state = client.app.state.services
    project_store: ProjectStore = state.project_store

    # Create two additional projects via API
    resp = client.post(
        "/knowledge/projects",
        data={"name": "Project Two", "description": ""},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    resp = client.post(
        "/knowledge/projects",
        data={"name": "Project Three", "description": "Extra"},
        follow_redirects=False,
    )
    assert resp.status_code == 303

    projects = project_store.list_projects()
    assert len(projects) >= 3

    uploads = [
        (projects[0], ("notes.txt", b"Alpha troubleshooting steps", "text/plain")),
        (projects[1], ("guide.md", b"# Beta issue\n\n## Fix", "text/markdown")),
        (
            projects[2],
            (
                "manual.pdf" if HAS_PYPDF else "manual.txt",
                SAMPLE_PDF_BYTES if HAS_PYPDF else b"PDF fallback text",
                "application/pdf" if HAS_PYPDF else "text/plain",
            ),
        ),
    ]

    for project, file_tuple in uploads:
        response = client.post(
            "/knowledge/upload",
            data={"project_id": project.id},
            files={"file": file_tuple},
            follow_redirects=False,
        )
        assert response.status_code == 303, response.text

    for project, file_tuple in uploads:
        docs = project_store.list_documents(project.id)
        assert len(docs) == 1
        metadata = docs[0]
        assert metadata.filename == file_tuple[0]
        assert metadata.chunk_count > 0

        store = state.store_manager.get_store(project.id)
        assert store.count() >= metadata.chunk_count
        results = store.search(state.embedding_service.embed_one("sanity"), limit=1)
        assert results
        assert all(result.payload.get("project_id") == project.id for result in results if result.payload)

    # Delete the second project's document and ensure cleanup
    target_project, _ = uploads[1]
    target_doc = project_store.list_documents(target_project.id)[0]
    response = client.post(
        "/knowledge/delete",
        data={"project_id": target_project.id, "doc_id": target_doc.id},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert project_store.list_documents(target_project.id) == []
    store = state.store_manager.get_store(target_project.id)
    assert store.count() == 0


def test_cleanup_endpoint_purges_project_vectors():
    client = build_app()
    state = client.app.state.services
    project_store: ProjectStore = state.project_store
    project = project_store.list_projects()[0]

    response = client.post(
        "/knowledge/upload",
        data={"project_id": project.id},
        files={"file": ("guide.txt", b"Troubleshooting steps", "text/plain")},
        follow_redirects=False,
    )
    assert response.status_code == 303

    documents = project_store.list_documents(project.id)
    assert documents
    store = state.store_manager.get_store(project.id)
    assert store.count() > 0

    response = client.post(
        "/knowledge/cleanup",
        data={"project_id": project.id},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert project_store.list_documents(project.id) == []
    assert store.count() == 0
