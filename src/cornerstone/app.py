"""FastAPI application setup for the Cornerstone prototype."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient, models

from .chat import SupportAgentService
from .config import Settings
from .embeddings import EmbeddingService
from .glossary import Glossary, load_glossary
from .ingestion import DocumentIngestor, ProjectVectorStoreManager
from .projects import Project, ProjectStore
from .vector_store import QdrantVectorStore, SearchResult

_MIN_RESULT_SCORE = 1e-6
_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"

logger = logging.getLogger(__name__)


class ApplicationState:
    """Container for runtime dependencies used by the FastAPI app."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedding_service: EmbeddingService,
        glossary: Glossary,
        project_store: ProjectStore,
        store_manager: ProjectVectorStoreManager,
        chat_service: SupportAgentService,
        ingestion_service: DocumentIngestor,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.glossary = glossary
        self.project_store = project_store
        self.store_manager = store_manager
        self.chat_service = chat_service
        self.ingestion_service = ingestion_service


def create_app(
    *,
    settings: Settings | None = None,
    embedding_service: EmbeddingService | None = None,
    glossary: Glossary | None = None,
    project_store: ProjectStore | None = None,
    store_manager: ProjectVectorStoreManager | None = None,
    chat_service: SupportAgentService | None = None,
    ingestion_service: DocumentIngestor | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = settings or Settings.from_env()
    embedding_service = embedding_service or EmbeddingService(settings)

    project_root = Path(settings.data_dir).resolve()
    project_store = project_store or ProjectStore(project_root, default_project_name=settings.default_project_name)
    logger.info("app.start settings_loaded data_dir=%s default_project=%s", project_root, settings.default_project_name)

    if store_manager is None:
        client = QdrantClient(**settings.qdrant_client_kwargs())
        base_collection_name = settings.project_collection_name(_default_project_id(project_store))
        base_store = QdrantVectorStore(
            client=client,
            collection_name=base_collection_name,
            vector_size=embedding_service.dimension,
            distance=models.Distance.COSINE,
        )
        base_store.ensure_collection()
        store_manager = ProjectVectorStoreManager(
            client_factory=lambda: client,
            vector_size=embedding_service.dimension,
            distance=models.Distance.COSINE,
            collection_name_fn=settings.project_collection_name,
        )

    store_manager.get_store(_default_project_id(project_store))

    glossary = glossary or load_glossary(settings.glossary_path)

    chat_service = chat_service or SupportAgentService(
        settings=settings,
        embedding_service=embedding_service,
        store_manager=store_manager,
        glossary=glossary,
    )

    ingestion_service = ingestion_service or DocumentIngestor(
        embedding_service=embedding_service,
        store_manager=store_manager,
        project_store=project_store,
    )

    app = FastAPI()
    app.state.services = ApplicationState(
        settings=settings,
        embedding_service=embedding_service,
        glossary=glossary,
        project_store=project_store,
        store_manager=store_manager,
        chat_service=chat_service,
        ingestion_service=ingestion_service,
    )

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def get_state(request: Request) -> ApplicationState:
        return request.app.state.services

    def get_embedding_service(request: Request) -> EmbeddingService:
        return get_state(request).embedding_service

    def get_project_store(request: Request) -> ProjectStore:
        return get_state(request).project_store

    def get_store_manager(request: Request) -> ProjectVectorStoreManager:
        return get_state(request).store_manager

    def get_chat_service(request: Request) -> SupportAgentService:
        return get_state(request).chat_service

    def get_ingestion_service(request: Request) -> DocumentIngestor:
        return get_state(request).ingestion_service

    def get_glossary(request: Request) -> Glossary:
        return get_state(request).glossary

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:  # pragma: no cover - template rendering
        project_store = get_project_store(request)
        projects = project_store.list_projects()
        selected_project = request.query_params.get("project_id") or _default_project_id(project_store)
        context = {
            "request": request,
            "projects": projects,
            "selected_project": selected_project,
        }
        return templates.TemplateResponse("index.html", context)

    @app.post("/search", response_class=HTMLResponse)
    async def search(  # pragma: no cover - template rendering
        request: Request,
        query: str = Form(...),
        project_id: str = Form(...),
        embedding: EmbeddingService = Depends(get_embedding_service),
        store_manager: ProjectVectorStoreManager = Depends(get_store_manager),
        project_store: ProjectStore = Depends(get_project_store),
        ) -> HTMLResponse:
        project = _resolve_project(project_store, project_id)
        logger.info("search.request project=%s query=%s", project.id, query)
        vector = embedding.embed_one(query)
        store = store_manager.get_store(project.id)
        hits = store.search(vector)
        logger.info("search.completed project=%s matches=%s", project.id, len(hits))
        context = {
            "request": request,
            "query": query,
            "projects": project_store.list_projects(),
            "selected_project": project.id,
            "results": _format_results(hits),
        }
        return templates.TemplateResponse("index.html", context)

    @app.get("/support", response_class=HTMLResponse)
    async def support_page(request: Request) -> HTMLResponse:  # pragma: no cover - template rendering
        glossary_count = len(get_glossary(request))
        project_store = get_project_store(request)
        selected_project = request.query_params.get("project_id") or _default_project_id(project_store)
        context = {
            "request": request,
            "glossary_count": glossary_count,
            "projects": project_store.list_projects(),
            "selected_project": selected_project,
        }
        return templates.TemplateResponse("support.html", context)

    @app.post("/support/chat", response_class=JSONResponse)
    async def support_chat(
        request: Request,
        chat_service: SupportAgentService = Depends(get_chat_service),
        project_store: ProjectStore = Depends(get_project_store),
    ) -> JSONResponse:
        payload = await request.json()
        query = str(payload.get("query", "")).strip()
        history = payload.get("history") or []
        project_id = payload.get("projectId")
        project = _resolve_project(project_store, project_id)
        if not query:
            return JSONResponse({"error": "Query is required."}, status_code=400)
        logger.info("support.endpoint request project=%s history_turns=%s", project.id, len(history))
        response = chat_service.generate(project.id, query, conversation=history)
        logger.info("support.endpoint completed project=%s message_chars=%s", project.id, len(response.message))
        return JSONResponse(
            {
                "message": response.message,
                "sources": response.sources,
                "definitions": response.definitions,
            }
        )

    @app.post("/support/chat/stream", response_class=StreamingResponse)
    async def support_chat_stream(
        request: Request,
        chat_service: SupportAgentService = Depends(get_chat_service),
        project_store: ProjectStore = Depends(get_project_store),
    ) -> StreamingResponse:
        payload = await request.json()
        query = str(payload.get("query", "")).strip()
        history = payload.get("history") or []
        project_id = payload.get("projectId")
        project = _resolve_project(project_store, project_id)
        if not query:
            raise HTTPException(status_code=400, detail="Query is required.")

        try:
            context, stream = chat_service.stream_generate(project.id, query, conversation=history)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("support.stream.setup_failed project=%s error=%s", project.id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        logger.info(
            "support.stream.started project=%s sources=%s definitions=%s",
            project.id,
            len(context.sources),
            len(context.definitions),
        )

        def _encode_event(data: dict) -> bytes:
            return (json.dumps(data) + "\n").encode("utf-8")

        def _event_iterator() -> Iterable[bytes]:
            yield _encode_event(
                {
                    "event": "metadata",
                    "sources": context.sources,
                    "definitions": context.definitions,
                }
            )
            chunks: list[str] = []
            stream_iter = iter(stream)
            try:
                for delta in stream_iter:
                    if delta is None:
                        continue
                    text = str(delta)
                    if not text:
                        continue
                    chunks.append(text)
                    yield _encode_event({"event": "delta", "data": text})
            except Exception as exc:  # pragma: no cover - runtime streaming errors
                logger.exception("support.stream.runtime_error project=%s error=%s", project.id, exc)
                yield _encode_event({"event": "error", "message": str(exc)})
                return

            final_message = "".join(chunks)
            yield _encode_event({"event": "done", "message": final_message})

        return StreamingResponse(_event_iterator(), media_type="application/x-ndjson")

    @app.get("/knowledge", response_class=HTMLResponse)
    async def knowledge_dashboard(request: Request) -> HTMLResponse:  # pragma: no cover - template rendering
        project_store = get_project_store(request)
        projects = project_store.list_projects()
        selected_project = request.query_params.get("project_id") or _default_project_id(project_store)
        logger.info("knowledge.dashboard load project=%s", selected_project)
        project = _resolve_project(project_store, selected_project)
        documents = project_store.list_documents(project.id)
        context = {
            "request": request,
            "projects": projects,
            "selected_project": project.id,
            "project_name": project.name,
            "documents": documents,
        }
        return templates.TemplateResponse("knowledge.html", context)

    @app.post("/knowledge/projects", response_class=RedirectResponse)
    async def create_project(
        request: Request,
        name: str = Form(...),
        description: str | None = Form(None),
        project_store: ProjectStore = Depends(get_project_store),
        store_manager: ProjectVectorStoreManager = Depends(get_store_manager),
    ) -> RedirectResponse:
        project = project_store.create_project(name, description)
        store_manager.get_store(project.id)
        logger.info("knowledge.project.created id=%s name=%s", project.id, project.name)
        url = request.url_for("knowledge_dashboard").include_query_params(project_id=project.id)
        return RedirectResponse(url=str(url), status_code=303)

    @app.post("/knowledge/upload", response_class=RedirectResponse)
    async def upload_document(
        request: Request,
        project_id: str = Form(...),
        file: UploadFile = File(...),
        project_store: ProjectStore = Depends(get_project_store),
        ingestion: DocumentIngestor = Depends(get_ingestion_service),
    ) -> RedirectResponse:
        project = _resolve_project(project_store, project_id)
        if not file.filename:
            raise HTTPException(status_code=400, detail="File name is required")
        try:
            await ingestion.ingest_upload(project.id, file)
        except ValueError as exc:
            logger.warning("knowledge.upload.failed project=%s file=%s error=%s", project.id, file.filename, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info("knowledge.upload.completed project=%s file=%s", project.id, file.filename)
        url = request.url_for("knowledge_dashboard").include_query_params(project_id=project.id)
        return RedirectResponse(url=str(url), status_code=303)

    @app.post("/knowledge/cleanup", response_class=RedirectResponse)
    async def cleanup_project(
        request: Request,
        project_id: str = Form(...),
        project_store: ProjectStore = Depends(get_project_store),
        store_manager: ProjectVectorStoreManager = Depends(get_store_manager),
    ) -> RedirectResponse:
        project = _resolve_project(project_store, project_id)
        cleared = store_manager.purge_project(project.id)
        project_store.clear_documents(project.id)
        logger.info(
            "knowledge.cleanup.completed project=%s cleared=%s",
            project.id,
            cleared,
        )
        url = request.url_for("knowledge_dashboard").include_query_params(project_id=project.id)
        return RedirectResponse(url=str(url), status_code=303)

    @app.post("/knowledge/delete", response_class=RedirectResponse)
    async def delete_document(
        request: Request,
        project_id: str = Form(...),
        doc_id: str = Form(...),
        project_store: ProjectStore = Depends(get_project_store),
        store_manager: ProjectVectorStoreManager = Depends(get_store_manager),
    ) -> RedirectResponse:
        project = _resolve_project(project_store, project_id)
        if not project_store.remove_document(project.id, doc_id):
            raise HTTPException(status_code=404, detail="Document not found")
        removed_vectors = store_manager.delete_document(project.id, doc_id)
        logger.info(
            "knowledge.delete.completed project=%s doc_id=%s success=%s",
            project.id,
            doc_id,
            removed_vectors,
        )
        url = request.url_for("knowledge_dashboard").include_query_params(project_id=project.id)
        return RedirectResponse(url=str(url), status_code=303)

    return app


def _default_project_id(project_store: ProjectStore) -> str:
    projects = project_store.list_projects()
    if not projects:
        raise RuntimeError("No projects configured")
    return projects[0].id


def _resolve_project(project_store: ProjectStore, project_id: Optional[str]) -> Project:
    if project_id:
        project = project_store.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return project
    default_id = _default_project_id(project_store)
    project = project_store.get_project(default_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _format_results(results: Iterable[SearchResult]) -> list[dict[str, object]]:
    formatted: list[dict[str, object]] = []
    for result in results:
        if result.score <= _MIN_RESULT_SCORE:
            continue
        payload = result.payload or {}
        formatted.append(
            {
                "id": result.id,
                "score": f"{result.score:.3f}",
                "payload": payload,
            }
        )
    return formatted


__all__ = ["create_app", "ApplicationState"]
