"""FastAPI application setup for the Cornerstone prototype."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Iterable, Optional, AsyncGenerator

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse, Response
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient, models

from .chat import SupportAgentService
from .config import Settings
from .embeddings import EmbeddingService
from .glossary import Glossary, GlossaryEntry, load_glossary, load_query_hints
from .ingestion import (
    DocumentIngestor,
    IngestionJobManager,
    ProjectVectorStoreManager,
)
from .fts import FTSIndex
from . import local_ingest
from .keywords import KeywordLLMFilter, build_excerpt, extract_keyword_candidates
from .personas import PersonaOverrides, PersonaSnapshot, PersonaStore
from .projects import Project, ProjectStore
from .vector_store import QdrantVectorStore, SearchResult
from .observability import MetricsRecorder
from .reranker import Reranker, build_reranker
from .query_hints import QueryHintGenerator, merge_hint_sources
from .query_hint_scheduler import QueryHintScheduler
from .conversations import ConversationLogStore, ConversationLogger, AnalyticsService

_MIN_RESULT_SCORE = 1e-6
_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"

logger = logging.getLogger(__name__)


_LOGGING_CONFIGURED = False


def _ensure_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    cornerstone_logger = logging.getLogger("cornerstone")
    uvicorn_logger = logging.getLogger("uvicorn.error")

    handlers = list(uvicorn_logger.handlers)
    if handlers:
        cornerstone_logger.handlers = []
        for handler in handlers:
            cornerstone_logger.addHandler(handler)
    else:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        cornerstone_logger.addHandler(handler)

    if cornerstone_logger.level == logging.NOTSET or cornerstone_logger.level > logging.INFO:
        cornerstone_logger.setLevel(logging.INFO)
    cornerstone_logger.propagate = False
    _LOGGING_CONFIGURED = True


class ApplicationState:
    """Container for runtime dependencies used by the FastAPI app."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedding_service: EmbeddingService,
        glossary: Glossary,
        project_store: ProjectStore,
        persona_store: PersonaStore,
        store_manager: ProjectVectorStoreManager,
        chat_service: SupportAgentService,
        ingestion_service: DocumentIngestor,
        ingestion_jobs: IngestionJobManager,
        fts_index,
        metrics: MetricsRecorder | None,
        reranker: Reranker | None,
        query_hints: dict[str, list[str]] | None,
        query_hint_generator: QueryHintGenerator,
        conversation_logger: ConversationLogger,
        analytics_service: AnalyticsService,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.glossary = glossary
        self.project_store = project_store
        self.persona_store = persona_store
        self.store_manager = store_manager
        self.chat_service = chat_service
        self.ingestion_service = ingestion_service
        self.ingestion_jobs = ingestion_jobs
        self.fts_index = fts_index
        self.metrics = metrics
        self.reranker = reranker
        self.query_hints = query_hints or {}
        self.query_hint_generator = query_hint_generator
        self.hint_scheduler = QueryHintScheduler(settings.query_hint_cron)
        self.conversation_logger = conversation_logger
        self.analytics = analytics_service


def create_app(
    *,
    settings: Settings | None = None,
    embedding_service: EmbeddingService | None = None,
    glossary: Glossary | None = None,
    project_store: ProjectStore | None = None,
    persona_store: PersonaStore | None = None,
    store_manager: ProjectVectorStoreManager | None = None,
    chat_service: SupportAgentService | None = None,
    ingestion_service: DocumentIngestor | None = None,
    ingestion_jobs: IngestionJobManager | None = None,
    metrics: MetricsRecorder | None = None,
    reranker: Reranker | None = None,
    conversation_logger: ConversationLogger | None = None,
    analytics_service: AnalyticsService | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    _ensure_logging()

    settings = settings or Settings.from_env()
    embedding_service = embedding_service or EmbeddingService(settings)
    metrics = metrics or settings.build_metrics_recorder()

    project_root = Path(settings.data_dir).resolve()
    project_store = project_store or ProjectStore(project_root, default_project_name=settings.default_project_name)
    persona_store = persona_store or PersonaStore(project_root)
    conversation_logger = conversation_logger or ConversationLogger(
        ConversationLogStore(settings.conversation_log_path()),
        enabled=settings.conversation_logging_enabled,
        retention_days=settings.conversation_retention_days,
        metrics=metrics,
    )
    analytics_service = analytics_service or AnalyticsService(conversation_logger)
    logger.info(
        "app.start settings_loaded data_dir=%s default_project=%s",
        project_root,
        settings.default_project_name,
    )

    if store_manager is None:
        client = QdrantClient(**settings.qdrant_client_kwargs())
        base_collection_name = settings.project_collection_name(_default_project_id(project_store))
        collection_kwargs = settings.qdrant_collection_tuning_kwargs()
        base_store = QdrantVectorStore(
            client=client,
            collection_name=base_collection_name,
            vector_size=embedding_service.dimension,
            distance=models.Distance.COSINE,
            **collection_kwargs,
        )
        base_store.ensure_collection()
        store_manager = ProjectVectorStoreManager(
            client_factory=lambda: client,
            vector_size=embedding_service.dimension,
            distance=models.Distance.COSINE,
            collection_name_fn=settings.project_collection_name,
            collection_kwargs=collection_kwargs,
        )

    store_manager.get_store(_default_project_id(project_store))

    glossary = glossary or load_glossary(settings.glossary_path)
    query_hints = load_query_hints(settings.query_hint_path)
    query_hint_generator = QueryHintGenerator(
        settings,
        max_terms_per_prompt=max(1, settings.query_hint_batch_size),
    )

    fts_index = FTSIndex(Path(settings.fts_db_path).resolve())

    if reranker is None:
        if chat_service is None:
            reranker = build_reranker(settings, embedding_service)
        else:
            reranker = getattr(chat_service, "_reranker", None)

    if chat_service is None:
        chat_service = SupportAgentService(
            settings=settings,
            embedding_service=embedding_service,
            store_manager=store_manager,
            glossary=glossary,
            project_store=project_store,
            persona_store=persona_store,
            fts_index=fts_index,
            metrics=metrics,
            retrieval_top_k=settings.retrieval_top_k,
            reranker=reranker,
            query_hints=query_hints,
        )
    else:
        query_hints = getattr(chat_service, "_query_hints", query_hints)

    ingestion_service = ingestion_service or DocumentIngestor(
        embedding_service=embedding_service,
        store_manager=store_manager,
        project_store=project_store,
        fts_index=fts_index,
        metrics=metrics,
    )

    ingestion_jobs = ingestion_jobs or IngestionJobManager(
        max_active_per_project=settings.ingestion_project_concurrency_limit,
        max_files_per_minute=settings.ingestion_files_per_minute,
    )

    app = FastAPI()
    app.state.services = ApplicationState(
        settings=settings,
        embedding_service=embedding_service,
        glossary=glossary,
        project_store=project_store,
        persona_store=persona_store,
        store_manager=store_manager,
        chat_service=chat_service,
        ingestion_service=ingestion_service,
        ingestion_jobs=ingestion_jobs,
        fts_index=fts_index,
        metrics=metrics,
        reranker=reranker,
        query_hints=query_hints,
        query_hint_generator=query_hint_generator,
        conversation_logger=conversation_logger,
        analytics_service=analytics_service,
    )

    scheduler = app.state.services.hint_scheduler
    for project in project_store.list_projects():
        metadata = project_store.get_query_hint_metadata(project.id)
        schedule = metadata.get("schedule") if isinstance(metadata, dict) else None
        start = None
        last_generated = metadata.get("last_generated") if isinstance(metadata, dict) else None
        if isinstance(last_generated, str):
            try:
                start = datetime.fromisoformat(last_generated)
            except ValueError:
                start = None
        scheduler.update_job(project.id, schedule, start=start)

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def get_state(request: Request) -> ApplicationState:
        return request.app.state.services

    def get_embedding_service(request: Request) -> EmbeddingService:
        return get_state(request).embedding_service

    def get_settings_dependency(request: Request) -> Settings:
        return get_state(request).settings

    def get_project_store(request: Request) -> ProjectStore:
        return get_state(request).project_store

    def get_persona_store(request: Request) -> PersonaStore:
        return get_state(request).persona_store

    def get_store_manager(request: Request) -> ProjectVectorStoreManager:
        return get_state(request).store_manager

    def get_chat_service(request: Request) -> SupportAgentService:
        return get_state(request).chat_service

    def get_ingestion_service(request: Request) -> DocumentIngestor:
        return get_state(request).ingestion_service

    def get_glossary(request: Request) -> Glossary:
        return get_state(request).glossary

    def get_ingestion_jobs(request: Request) -> IngestionJobManager:
        return get_state(request).ingestion_jobs

    def get_fts_index(request: Request) -> FTSIndex:
        return get_state(request).fts_index

    def get_metrics(request: Request) -> MetricsRecorder | None:
        return get_state(request).metrics

    def get_query_hints(request: Request) -> dict[str, list[str]]:
        return get_state(request).query_hints

    def get_query_hint_generator(request: Request) -> QueryHintGenerator:
        return get_state(request).query_hint_generator

    def get_conversation_logger(request: Request) -> ConversationLogger:
        return get_state(request).conversation_logger

    def get_analytics_service(request: Request) -> AnalyticsService:
        return get_state(request).analytics

    def _parse_string_list(value) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            candidates = value.split(',')
        else:
            try:
                candidates = list(value)
            except TypeError:
                candidates = [value]
        items: list[str] = []
        for candidate in candidates:
            cleaned = str(candidate).strip()
            if cleaned and cleaned not in items:
                items.append(cleaned)
        return items

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
        persona_store = get_persona_store(request)
        selected_project = request.query_params.get("project_id") or _default_project_id(project_store)
        active_project = project_store.get_project(selected_project) if selected_project else None
        persona_snapshot: PersonaSnapshot | None = None
        if active_project is not None:
            persona_snapshot = persona_store.resolve_persona(active_project.persona_id, active_project.persona_overrides)
        context = {
            "request": request,
            "glossary_count": glossary_count,
            "projects": project_store.list_projects(),
            "selected_project": selected_project,
            "persona": persona_snapshot,
            "persona_base": persona_snapshot.base_persona if persona_snapshot else None,
        }
        return templates.TemplateResponse("support.html", context)

    @app.post("/support/chat", response_class=JSONResponse)
    async def support_chat(
        request: Request,
        chat_service: SupportAgentService = Depends(get_chat_service),
        project_store: ProjectStore = Depends(get_project_store),
        persona_store: PersonaStore = Depends(get_persona_store),
        conversation_logger: ConversationLogger = Depends(get_conversation_logger),
        settings_inst: Settings = Depends(get_settings_dependency),
    ) -> JSONResponse:
        payload = await request.json()
        query = str(payload.get("query", "")).strip()
        history = payload.get("history") or []
        project_id = payload.get("projectId")
        project = _resolve_project(project_store, project_id)
        if not query:
            return JSONResponse({"error": "Query is required."}, status_code=400)
        logger.info("support.endpoint request project=%s history_turns=%s", project.id, len(history))
        persona = persona_store.resolve_persona(project.persona_id, project.persona_overrides)
        start_time = time.perf_counter()
        response = chat_service.generate(project, query, conversation=history)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        try:
            conversation_logger.log_chat(
                project=project,
                persona=persona,
                query=query,
                response=response.message,
                history=history,
                sources=response.sources,
                definitions=response.definitions,
                backend=settings_inst.chat_backend,
                duration_ms=duration_ms,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("conversation.log.failed project=%s error=%s", project.id, exc)
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
        persona_store: PersonaStore = Depends(get_persona_store),
        conversation_logger: ConversationLogger = Depends(get_conversation_logger),
        settings_inst: Settings = Depends(get_settings_dependency),
    ) -> StreamingResponse:
        payload = await request.json()
        query = str(payload.get("query", "")).strip()
        history = payload.get("history") or []
        project_id = payload.get("projectId")
        project = _resolve_project(project_store, project_id)
        if not query:
            raise HTTPException(status_code=400, detail="Query is required.")

        persona = persona_store.resolve_persona(project.persona_id, project.persona_overrides)
        start_time = time.perf_counter()
        try:
            context, stream = chat_service.stream_generate(project, query, conversation=history)
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
            try:
                conversation_logger.log_chat(
                    project=project,
                    persona=persona,
                    query=query,
                    response=final_message,
                    history=history,
                    sources=context.sources,
                    definitions=context.definitions,
                    backend=settings_inst.chat_backend,
                    duration_ms=(time.perf_counter() - start_time) * 1000.0,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("conversation.log.failed project=%s error=%s", project.id, exc)
            yield _encode_event({"event": "done", "message": final_message})

        return StreamingResponse(_event_iterator(), media_type="application/x-ndjson")

    def _persona_to_dict(persona) -> dict:
        return {
            "id": persona.id,
            "name": persona.name,
            "description": persona.description,
            "tone": persona.tone,
            "system_prompt": persona.system_prompt,
            "avatar_url": persona.avatar_url,
            "tags": persona.tags,
            "created_at": persona.created_at,
            "glossary_top_k": persona.glossary_top_k,
            "retrieval_top_k": persona.retrieval_top_k,
            "chat_temperature": persona.chat_temperature,
            "chat_max_tokens": persona.chat_max_tokens,
        }

    def _parse_tags(raw: str | None) -> list[str]:
        if not raw:
            return []
        return [tag.strip() for tag in raw.split(",") if tag.strip()]

    def _parse_optional_int(raw: str | None, *, min_value: int | None = None) -> int | None:
        if raw is None:
            return None
        value = raw.strip()
        if not value:
            return None
        try:
            parsed = int(value)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=400, detail="Expected integer value") from exc
        if min_value is not None and parsed < min_value:
            raise HTTPException(status_code=400, detail=f"Value must be ≥ {min_value}")
        return parsed

    def _parse_optional_float(raw: str | None, *, min_value: float | None = None) -> float | None:
        if raw is None:
            return None
        value = raw.strip()
        if not value:
            return None
        try:
            parsed = float(value)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=400, detail="Expected numeric value") from exc
        if min_value is not None and parsed < min_value:
            raise HTTPException(status_code=400, detail=f"Value must be ≥ {min_value}")
        return parsed

    def _coerce_optional_int(value, *, min_value: int | None = None) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            return _parse_optional_int(value, min_value=min_value)
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=400, detail="Expected integer value") from exc
        if min_value is not None and parsed < min_value:
            raise HTTPException(status_code=400, detail=f"Value must be ≥ {min_value}")
        return parsed

    def _coerce_optional_float(value, *, min_value: float | None = None) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            return _parse_optional_float(value, min_value=min_value)
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=400, detail="Expected numeric value") from exc
        if min_value is not None and parsed < min_value:
            raise HTTPException(status_code=400, detail=f"Value must be ≥ {min_value}")
        return parsed

    @app.get("/knowledge", response_class=HTMLResponse)
    async def knowledge_dashboard(request: Request) -> HTMLResponse:  # pragma: no cover - template rendering
        project_store = get_project_store(request)
        persona_store = get_persona_store(request)
        settings_inst = get_settings_dependency(request)
        projects = project_store.list_projects()
        selected_project = request.query_params.get("project_id") or _default_project_id(project_store)
        logger.info("knowledge.dashboard load project=%s", selected_project)
        project = _resolve_project(project_store, selected_project)
        documents_all = project_store.list_documents(project.id)
        total_documents = len(documents_all)
        per_page = 25
        try:
            page = int(request.query_params.get("page", "1"))
        except ValueError:
            page = 1
        page = max(page, 1)

        if total_documents:
            total_pages = max(1, math.ceil(total_documents / per_page))
            if page > total_pages:
                page = total_pages
            start_index = (page - 1) * per_page
            end_index = min(start_index + per_page, total_documents)
            documents = documents_all[start_index:end_index]
            pagination = {
                "page": page,
                "per_page": per_page,
                "total_pages": total_pages,
                "total_documents": total_documents,
                "start_index": start_index,
                "end_index": end_index,
                "has_prev": page > 1,
                "has_next": page < total_pages,
            }
        else:
            documents = []
            pagination = None
        personas = persona_store.list_personas()
        persona_snapshot = persona_store.resolve_persona(project.persona_id, project.persona_overrides)
        context = {
            "request": request,
            "projects": projects,
            "selected_project": project.id,
            "project_name": project.name,
            "documents": documents,
            "documents_total": total_documents,
            "pagination": pagination,
            "persona": persona_snapshot,
            "persona_base": persona_snapshot.base_persona if persona_snapshot else None,
            "personas": personas,
            "project_persona_id": project.persona_id,
            "persona_overrides": project.persona_overrides,
            "defaults": {
                "glossary_top_k": settings_inst.glossary_top_k,
                "retrieval_top_k": settings_inst.retrieval_top_k,
                "chat_temperature": settings_inst.chat_temperature,
                "chat_max_tokens": settings_inst.chat_max_tokens,
            },
        }
        return templates.TemplateResponse("knowledge.html", context)

    @app.get("/admin/analytics", response_class=HTMLResponse)
    async def analytics_dashboard(request: Request) -> HTMLResponse:  # pragma: no cover - template rendering
        analytics_service = get_analytics_service(request)
        project_store = get_project_store(request)
        project_param = request.query_params.get("project_id") or "all"
        days = _parse_optional_int(request.query_params.get("days"), min_value=1) or 30

        project = None
        project_id: str | None
        if project_param == "all":
            project_id = None
        else:
            project = _resolve_project(project_store, project_param)
            project_id = project.id

        summary = analytics_service.build_summary(project_id=project_id, days=days)

        context = {
            "request": request,
            "projects": project_store.list_projects(),
            "selected_project": project_param,
            "summary": summary,
            "project_label": project.name if project else "All Projects",
            "days": days,
        }
        return templates.TemplateResponse("analytics.html", context)

    @app.get("/api/analytics/summary", response_class=JSONResponse)
    async def analytics_summary(
        project_id: str | None = Query(None),
        days: int = Query(30, ge=1),
        analytics_service: AnalyticsService = Depends(get_analytics_service),
    ) -> JSONResponse:
        normalized = (project_id or "").strip() or None
        if normalized and normalized.lower() == "all":
            normalized = None
        summary = analytics_service.build_summary(project_id=normalized, days=days)
        return JSONResponse(summary)

    @app.get("/metrics")
    async def metrics_endpoint(metrics: MetricsRecorder | None = Depends(get_metrics)) -> Response:
        if metrics is None or not metrics.prometheus_enabled:
            raise HTTPException(status_code=404, detail="Metrics export disabled")
        try:
            payload = metrics.render_prometheus()
        except RuntimeError as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=payload, media_type=metrics.prometheus_content_type)

    @app.post("/knowledge/persona", response_class=RedirectResponse)
    async def update_persona_settings(
        request: Request,
        project_id: str = Form(...),
        persona_id: str | None = Form(None),
        persona_name: str | None = Form(None),
        persona_tone: str | None = Form(None),
        persona_system_prompt: str | None = Form(None),
        persona_avatar_url: str | None = Form(None),
        persona_glossary_top_k: str | None = Form(None),
        persona_retrieval_top_k: str | None = Form(None),
        persona_chat_temperature: str | None = Form(None),
        persona_chat_max_tokens: str | None = Form(None),
        project_store: ProjectStore = Depends(get_project_store),
        persona_store: PersonaStore = Depends(get_persona_store),
    ) -> RedirectResponse:
        project = _resolve_project(project_store, project_id)
        glossary_top_k = _parse_optional_int(persona_glossary_top_k, min_value=0)
        retrieval_top_k = _parse_optional_int(persona_retrieval_top_k, min_value=1)
        chat_temperature = _parse_optional_float(persona_chat_temperature, min_value=0.0)
        chat_max_tokens = _parse_optional_int(persona_chat_max_tokens, min_value=0)
        overrides = PersonaOverrides(
            name=persona_name,
            tone=persona_tone,
            system_prompt=persona_system_prompt,
            avatar_url=persona_avatar_url,
            glossary_top_k=glossary_top_k,
            retrieval_top_k=retrieval_top_k,
            chat_temperature=chat_temperature,
            chat_max_tokens=chat_max_tokens,
        )
        updated = project_store.configure_persona(
            project.id,
            persona_id=persona_id,
            overrides=overrides,
        )
        resolved = persona_store.resolve_persona(updated.persona_id, updated.persona_overrides)
        has_overrides = any(
            getattr(updated.persona_overrides, field) is not None
            for field in (
                "name",
                "tone",
                "system_prompt",
                "avatar_url",
                "glossary_top_k",
                "retrieval_top_k",
                "chat_temperature",
                "chat_max_tokens",
            )
        )
        logger.info(
            "knowledge.persona.updated project=%s persona=%s overrides=%s",
            updated.id,
            resolved.id,
            has_overrides,
        )
        url = request.url_for("knowledge_dashboard").include_query_params(project_id=updated.id)
        return RedirectResponse(url=str(url), status_code=303)

    @app.get("/personas", response_class=HTMLResponse)
    async def personas_dashboard(request: Request) -> HTMLResponse:  # pragma: no cover - template rendering
        persona_store = get_persona_store(request)
        project_store = get_project_store(request)
        personas = persona_store.list_personas()
        assignments: dict[str, list[str]] = {}
        for project in project_store.list_projects():
            if project.persona_id:
                assignments.setdefault(project.persona_id, []).append(project.name)
        context = {
            "request": request,
            "personas": personas,
            "assignments": assignments,
        }
        return templates.TemplateResponse("personas.html", context)

    @app.post("/personas", response_class=RedirectResponse)
    async def create_persona(
        request: Request,
        name: str = Form(...),
        description: str | None = Form(None),
        tone: str | None = Form(None),
        system_prompt: str = Form(...),
        avatar_url: str | None = Form(None),
        tags: str | None = Form(None),
        glossary_top_k: str | None = Form(None),
        retrieval_top_k: str | None = Form(None),
        chat_temperature: str | None = Form(None),
        chat_max_tokens: str | None = Form(None),
        persona_store: PersonaStore = Depends(get_persona_store),
    ) -> RedirectResponse:
        new_persona = persona_store.create_persona(
            name=name,
            description=description,
            tone=tone,
            system_prompt=system_prompt,
            avatar_url=avatar_url,
            tags=_parse_tags(tags),
            glossary_top_k=_parse_optional_int(glossary_top_k, min_value=0),
            retrieval_top_k=_parse_optional_int(retrieval_top_k, min_value=1),
            chat_temperature=_parse_optional_float(chat_temperature, min_value=0.0),
            chat_max_tokens=_parse_optional_int(chat_max_tokens, min_value=0),
        )
        logger.info("persona.form.created id=%s name=%s", new_persona.id, new_persona.name)
        return RedirectResponse(url="/personas", status_code=303)

    @app.post("/personas/{persona_id}", response_class=RedirectResponse)
    async def edit_persona(
        persona_id: str,
        delete: str | None = Form(None),
        name: str = Form(...),
        description: str | None = Form(None),
        tone: str | None = Form(None),
        system_prompt: str = Form(...),
        avatar_url: str | None = Form(None),
        tags: str | None = Form(None),
        glossary_top_k: str | None = Form(None),
        retrieval_top_k: str | None = Form(None),
        chat_temperature: str | None = Form(None),
        chat_max_tokens: str | None = Form(None),
        persona_store: PersonaStore = Depends(get_persona_store),
        project_store: ProjectStore = Depends(get_project_store),
    ) -> RedirectResponse:
        if delete:
            assigned = [project.name for project in project_store.list_projects() if project.persona_id == persona_id]
            if assigned:
                raise HTTPException(
                    status_code=409,
                    detail=f"Persona in use by: {', '.join(assigned)}",
                )
            persona_store.delete_persona(persona_id)
            logger.info("persona.form.deleted id=%s", persona_id)
            return RedirectResponse(url="/personas", status_code=303)

        try:
            persona_store.update_persona(
                persona_id,
                name=name,
                description=description,
                tone=tone,
                system_prompt=system_prompt,
                avatar_url=avatar_url,
                tags=_parse_tags(tags),
                glossary_top_k=_parse_optional_int(glossary_top_k, min_value=0),
                retrieval_top_k=_parse_optional_int(retrieval_top_k, min_value=1),
                chat_temperature=_parse_optional_float(chat_temperature, min_value=0.0),
                chat_max_tokens=_parse_optional_int(chat_max_tokens, min_value=0),
            )
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        logger.info("persona.form.updated id=%s", persona_id)
        return RedirectResponse(url="/personas", status_code=303)

    @app.get("/api/personas", response_class=JSONResponse)
    async def list_personas_api(
        persona_store: PersonaStore = Depends(get_persona_store),
        project_store: ProjectStore = Depends(get_project_store),
    ) -> JSONResponse:
        projects = project_store.list_projects()
        assignments: dict[str, list[str]] = {}
        for project in projects:
            if project.persona_id:
                assignments.setdefault(project.persona_id, []).append(project.name)
        personas = [
            {
                **_persona_to_dict(persona),
                "assigned_projects": assignments.get(persona.id, []),
            }
            for persona in persona_store.list_personas()
        ]
        return JSONResponse(personas)

    @app.post("/api/personas", response_class=JSONResponse)
    async def create_persona_api(
        request: Request,
        persona_store: PersonaStore = Depends(get_persona_store),
    ) -> JSONResponse:
        payload = await request.json()
        name = (payload.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        persona = persona_store.create_persona(
            name=name,
            description=payload.get("description"),
            tone=payload.get("tone"),
            system_prompt=payload.get("system_prompt"),
            avatar_url=payload.get("avatar_url"),
            tags=payload.get("tags") or [],
            glossary_top_k=_coerce_optional_int(payload.get("glossary_top_k"), min_value=0),
            retrieval_top_k=_coerce_optional_int(payload.get("retrieval_top_k"), min_value=1),
            chat_temperature=_coerce_optional_float(payload.get("chat_temperature"), min_value=0.0),
            chat_max_tokens=_coerce_optional_int(payload.get("chat_max_tokens"), min_value=0),
        )
        return JSONResponse(_persona_to_dict(persona), status_code=201)

    @app.post("/api/personas/{persona_id}", response_class=JSONResponse)
    async def update_persona_api(
        persona_id: str,
        request: Request,
        persona_store: PersonaStore = Depends(get_persona_store),
    ) -> JSONResponse:
        payload = await request.json()
        try:
            persona = persona_store.update_persona(
                persona_id,
                name=payload.get("name"),
                description=payload.get("description"),
                tone=payload.get("tone"),
                system_prompt=payload.get("system_prompt"),
                avatar_url=payload.get("avatar_url"),
                tags=payload.get("tags"),
                glossary_top_k=_coerce_optional_int(payload.get("glossary_top_k"), min_value=0),
                retrieval_top_k=_coerce_optional_int(payload.get("retrieval_top_k"), min_value=1),
                chat_temperature=_coerce_optional_float(payload.get("chat_temperature"), min_value=0.0),
                chat_max_tokens=_coerce_optional_int(payload.get("chat_max_tokens"), min_value=0),
            )
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(_persona_to_dict(persona))

    @app.delete("/api/personas/{persona_id}", response_class=JSONResponse)
    async def delete_persona_api(
        persona_id: str,
        persona_store: PersonaStore = Depends(get_persona_store),
        project_store: ProjectStore = Depends(get_project_store),
    ) -> JSONResponse:
        assigned = [project.name for project in project_store.list_projects() if project.persona_id == persona_id]
        if assigned:
            raise HTTPException(
                status_code=409,
                detail={"message": "Persona is assigned to active projects.", "projects": assigned},
            )
        deleted = persona_store.delete_persona(persona_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Persona not found")
        return JSONResponse({"status": "deleted", "persona_id": persona_id})

    @app.get("/keywords", response_class=HTMLResponse)
    async def keywords_dashboard(request: Request) -> HTMLResponse:  # pragma: no cover - template rendering
        project_store = get_project_store(request)
        projects = project_store.list_projects()
        selected_project = request.query_params.get("project_id") or _default_project_id(project_store)
        context = {
            "request": request,
            "projects": projects,
            "selected_project": selected_project,
        }
        return templates.TemplateResponse("keywords.html", context)


    @app.get("/keywords/{project_id}/candidates", response_class=JSONResponse)
    async def project_keywords(
        project_id: str,
        project_store: ProjectStore = Depends(get_project_store),
        store_manager: ProjectVectorStoreManager = Depends(get_store_manager),
        settings: Settings = Depends(get_settings_dependency),
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=500),
    ) -> JSONResponse:
        project = _resolve_project(project_store, project_id)
        payloads = list(store_manager.iter_project_payloads(project.id))
        texts = [payload.get("text", "") for payload in payloads if payload.get("text")]
        keywords = extract_keyword_candidates(texts)

        context_snippets: list[str] = []
        for payload in payloads:
            snippet = str(payload.get("text", "")).strip()
            if not snippet:
                continue
            context_snippets.append(snippet[:400])
            if len(context_snippets) >= 5:
                break

        llm_filter = KeywordLLMFilter(settings)
        original_count = len(keywords)
        debug_payload: dict[str, object] = {}
        if llm_filter.enabled:
            keywords = llm_filter.filter_keywords(keywords, context_snippets)
            logger.info(
                "keyword.llm.apply backend=%s project=%s before=%s after=%s",
                llm_filter.backend,
                project.id,
                original_count,
                len(keywords),
            )
            debug_payload = llm_filter.debug_payload()
        else:
            logger.info(
                "keyword.llm.bypass backend=%s project=%s candidate_count=%s",
                llm_filter.backend,
                project.id,
                original_count,
            )
            debug_payload = llm_filter.debug_payload()
            debug_payload.setdefault("candidate_count", original_count)

        logger.info(
            "keyword.llm.summary project=%s backend=%s details=%s",
            project.id,
            debug_payload.get("backend"),
            debug_payload,
        )

        keyword_dicts = [
            {
                "term": item.term,
                "count": item.count,
                "core": item.is_core,
                "generated": item.generated,
                "reason": item.reason,
                "source": item.source,
            }
            for item in keywords
        ]

        max_page_size = max(1, min(page_size, 500))
        total = len(keyword_dicts)
        if total:
            total_pages = math.ceil(total / max_page_size)
            current_page = min(max(1, page), total_pages)
            start_index = (current_page - 1) * max_page_size
            end_index = min(start_index + max_page_size, total)
            range_start = start_index + 1
            range_end = end_index
        else:
            total_pages = 0
            current_page = 1
            start_index = 0
            end_index = 0
            range_start = 0
            range_end = 0

        page_items = keyword_dicts[start_index:end_index]
        pagination = {
            "page": current_page,
            "page_size": max_page_size,
            "total": total,
            "pages": total_pages,
            "range_start": range_start,
            "range_end": range_end,
            "has_next": total_pages > 0 and current_page < total_pages,
            "has_prev": total_pages > 0 and current_page > 1,
        }

        data = {
            "projectId": project.id,
            "keywords": page_items,
            "filter": debug_payload,
            "pagination": pagination,
        }
        return JSONResponse(data)

    @app.get("/keywords/{project_id}/definition", response_class=JSONResponse)
    async def keyword_definition(
        project_id: str,
        term: str,
        project_store: ProjectStore = Depends(get_project_store),
        store_manager: ProjectVectorStoreManager = Depends(get_store_manager),
        embedding: EmbeddingService = Depends(get_embedding_service),
        glossary: Glossary = Depends(get_glossary),
        settings: Settings = Depends(get_settings_dependency),
    ) -> JSONResponse:
        cleaned = term.strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="Term is required")
        project = _resolve_project(project_store, project_id)
        vector = embedding.embed_one(cleaned)
        store = store_manager.get_store(project.id)
        hits = store.search(vector, limit=5, with_payload=True)
        candidates: list[dict[str, object]] = []
        context_snippets: list[str] = []
        for hit in hits:
            payload = hit.payload or {}
            snippet = payload.get("text")
            if not snippet:
                continue
            excerpt = build_excerpt(snippet)
            context_snippets.append(snippet)
            candidates.append(
                {
                    "snippet": snippet,
                    "excerpt": excerpt,
                    "score": float(hit.score),
                    "doc_id": payload.get("doc_id"),
                    "chunk_index": payload.get("chunk_index"),
                    "source": payload.get("source"),
                }
            )
        definitions = [
            f"{entry.term}: {entry.definition}"
            for entry in glossary.top_matches(cleaned, settings.glossary_top_k)
        ]

        llm_filter = KeywordLLMFilter(settings)
        if not definitions and llm_filter.enabled:
            definitions.extend(llm_filter.generate_definitions(cleaned, context_snippets[:3]))

        if not definitions and context_snippets:
            definitions.append(build_excerpt(context_snippets[0], max_chars=200))

        return JSONResponse(
            {
                "projectId": project.id,
                "term": cleaned,
                "candidates": candidates,
                "definitions": definitions,
            }
        )

    @app.post("/keywords/{project_id}/insights", response_class=JSONResponse)
    async def save_keyword_insight(
        project_id: str,
        request: Request,
        project_store: ProjectStore = Depends(get_project_store),
    ) -> JSONResponse:
        project = _resolve_project(project_store, project_id)
        payload = await request.json()
        term = str(payload.get("term", "")).strip()
        if not term:
            raise HTTPException(status_code=400, detail="term is required")
        insight = {
            "term": term,
            "candidates": payload.get("candidates") or [],
            "definitions": payload.get("definitions") or [],
            "filter": payload.get("filter") or {},
        }
        saved = project_store.save_keyword_insight(project.id, insight)
        return JSONResponse(saved, status_code=201)

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

    @app.post("/knowledge/uploads", response_class=JSONResponse)
    async def upload_documents_async(
        background_tasks: BackgroundTasks,
        project_id: str = Form(...),
        files: list[UploadFile] = File(...),
        project_store: ProjectStore = Depends(get_project_store),
        ingestion: DocumentIngestor = Depends(get_ingestion_service),
        job_manager: IngestionJobManager = Depends(get_ingestion_jobs),
    ) -> JSONResponse:
        project = _resolve_project(project_store, project_id)
        if not files:
            raise HTTPException(status_code=400, detail="At least one file is required")

        job_ids: list[str] = []
        for upload in files:
            filename = upload.filename or "document"
            try:
                raw_bytes = await upload.read()
            except Exception as exc:  # pragma: no cover - unexpected IO errors
                job = job_manager.create_job(project.id, filename)
                job_manager.mark_failed(job.id, f"Failed to read file: {exc}")
                job_ids.append(job.id)
                continue

            job = job_manager.create_job(project.id, filename)
            job_ids.append(job.id)

            if not raw_bytes:
                job_manager.mark_failed(job.id, "File is empty")
                continue

            background_tasks.add_task(
                _process_ingestion_job,
                job_manager,
                ingestion,
                job.id,
                project.id,
                filename,
                upload.content_type,
                raw_bytes,
            )

        jobs_payload = [job_manager.get(job_id).to_dict() for job_id in job_ids if job_manager.get(job_id)]
        return JSONResponse({"jobs": jobs_payload}, status_code=202)

    @app.get("/knowledge/local-directories", response_class=JSONResponse)
    async def get_local_directories(
        path: str | None = Query(None),
        settings: Settings = Depends(get_settings_dependency),
    ) -> JSONResponse:
        base_dir = Path(settings.local_data_dir).resolve()
        try:
            directories = local_ingest.list_directories(base_dir, path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"directories": directories})

    @app.post("/knowledge/local-import", response_class=JSONResponse)
    async def import_local_directory(
        background_tasks: BackgroundTasks,
        project_id: str = Form(...),
        path: str = Form(...),
        settings: Settings = Depends(get_settings_dependency),
        project_store: ProjectStore = Depends(get_project_store),
        ingestion: DocumentIngestor = Depends(get_ingestion_service),
        job_manager: IngestionJobManager = Depends(get_ingestion_jobs),
    ) -> JSONResponse:
        project = _resolve_project(project_store, project_id)
        base_dir = Path(settings.local_data_dir).resolve()
        try:
            target_dir = local_ingest.resolve_local_path(base_dir, path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not target_dir.exists() or not target_dir.is_dir():
            raise HTTPException(status_code=404, detail="Directory not found")

        display_name = str(target_dir.relative_to(base_dir)) or "."
        manifest_path = Path(settings.data_dir).resolve() / "manifests" / f"{project.id}.json"
        job = job_manager.create_job(project.id, f"{display_name}/")
        background_tasks.add_task(
            local_ingest.ingest_directory,
            project_id=project.id,
            target_dir=target_dir,
            base_dir=base_dir,
            ingestion_service=ingestion,
            manifest_path=manifest_path,
            job_manager=job_manager,
            job_id=job.id,
        )
        return JSONResponse({"job": job.to_dict()}, status_code=202)

    @app.post("/knowledge/upload-url", response_class=JSONResponse)
    async def upload_document_url(
        background_tasks: BackgroundTasks,
        project_id: str = Form(...),
        url: str = Form(...),
        project_store: ProjectStore = Depends(get_project_store),
        ingestion: DocumentIngestor = Depends(get_ingestion_service),
        job_manager: IngestionJobManager = Depends(get_ingestion_jobs),
    ) -> JSONResponse:
        project = _resolve_project(project_store, project_id)
        sanitized_url = url.strip()
        if not sanitized_url:
            raise HTTPException(status_code=400, detail="URL is required")

        job = job_manager.create_job(project.id, sanitized_url)
        background_tasks.add_task(
            _process_ingestion_job,
            job_manager,
            ingestion,
            job.id,
            project.id,
            sanitized_url,
            None,
            None,
            sanitized_url,
        )

        return JSONResponse({"job": job.to_dict()}, status_code=202)

    @app.get("/knowledge/uploads", response_class=JSONResponse)
    async def list_upload_jobs(
        project_id: str = Query(...),
        job_manager: IngestionJobManager = Depends(get_ingestion_jobs),
    ) -> JSONResponse:
        jobs = [job.to_dict() for job in job_manager.list_for_project(project_id)]
        return JSONResponse({"jobs": jobs})

    @app.get("/knowledge/query-hints", response_class=JSONResponse)
    async def list_query_hints_endpoint(
        project_id: str = Query(...),
        project_store: ProjectStore = Depends(get_project_store),
        base_hints: dict[str, list[str]] = Depends(get_query_hints),
    ) -> JSONResponse:
        project = _resolve_project(project_store, project_id)
        project_hints = project_store.get_query_hints(project.id)
        metadata = project_store.get_query_hint_metadata(project.id)
        combined = merge_hint_sources(base_hints, project_hints)
        sorted_hints = {key: combined[key] for key in sorted(combined)}
        return JSONResponse({"projectId": project.id, "hints": sorted_hints, "metadata": metadata})

    @app.post("/knowledge/query-hints/generate", response_class=StreamingResponse)
    async def generate_query_hints_endpoint(
        request: Request,
        project_store: ProjectStore = Depends(get_project_store),
        base_hints: dict[str, list[str]] = Depends(get_query_hints),
        generator: QueryHintGenerator = Depends(get_query_hint_generator),
    ) -> StreamingResponse:
        payload = await request.json()
        project_id = str(payload.get("project_id", "")).strip()
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id is required")

        project = _resolve_project(project_store, project_id)

        schedule = str(payload.get("schedule", "")).strip().lower()
        if schedule not in {"", "daily", "weekly"}:
            raise HTTPException(status_code=400, detail="schedule must be '', 'daily', or 'weekly'")

        if generator is None or not generator.enabled:
            raise HTTPException(status_code=503, detail="Query hint generator is not configured")

        raw_entries = project_store.list_glossary_entries(project.id)
        entries: list[GlossaryEntry] = []
        for item in raw_entries:
            term = str(item.get("term", "")).strip()
            definition = str(item.get("definition", "")).strip()
            if not term or not definition:
                continue
            synonyms = item.get("synonyms") or []
            keywords = item.get("keywords") or []
            entries.append(
                GlossaryEntry(
                    term=term,
                    definition=definition,
                    synonyms=[str(value).strip() for value in synonyms if str(value).strip()],
                    keywords=[str(value).strip() for value in keywords if str(value).strip()],
                )
            )

        if not entries:
            raise HTTPException(status_code=400, detail="No glossary entries available for this project")

        existing_hints = project_store.get_query_hints(project.id)
        merged_project = dict(existing_hints)
        metadata = project_store.get_query_hint_metadata(project.id) or {}

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()
        request.app.state.services.hint_scheduler.update_job(project.id, schedule)

        def progress_callback(batch_index: int, batch_hints: dict[str, list[str]]) -> None:
            nonlocal merged_project
            merged_project = merge_hint_sources(merged_project, batch_hints)
            chunk = json.dumps(
                {
                    "event": "progress",
                    "batch": batch_index,
                    "hints": batch_hints,
                }
            )
            loop.call_soon_threadsafe(queue.put_nowait, chunk)

        payload_batch_size = max(1, min(20, int(payload.get("batch_size", settings.query_hint_batch_size))))
        generator._max_terms = payload_batch_size

        async def run_generation() -> tuple[dict[str, list[str]], str | None, int]:
            try:
                report = await loop.run_in_executor(
                    None,
                    lambda: generator.generate(entries, progress_callback=progress_callback),
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("query_hints.generate.failed project=%s", project.id)
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return report.hints, report.backend, report.prompts_sent

        async def event_stream() -> AsyncGenerator[bytes, None]:
            nonlocal merged_project, metadata
            task = asyncio.create_task(run_generation())
            try:
                while True:
                    if task.done() and queue.empty():
                        break
                    try:
                        chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    if chunk is None:
                        continue
                    yield (chunk + "\n").encode("utf-8")

                hints, backend, prompts_sent = await task
                merged_project = merge_hint_sources(merged_project, hints)
                metadata.update(
                    {
                        "schedule": schedule,
                        "backend": backend,
                        "prompts": prompts_sent,
                        "batch_size": payload_batch_size,
                        "last_generated": datetime.now(timezone.utc).isoformat(),
                    }
                )
                project_store.set_query_hints(project.id, merged_project, metadata=metadata)
                request.app.state.services.hint_scheduler.update_job(
                    project.id,
                    schedule,
                    start=datetime.now(timezone.utc),
                )
                effective = merge_hint_sources(base_hints, merged_project)
                sorted_hints = {key: effective[key] for key in sorted(effective)}
                final_payload = json.dumps(
                    {
                        "event": "completed",
                        "projectId": project.id,
                        "hints": sorted_hints,
                        "backend": backend,
                        "prompts": prompts_sent,
                        "batch_size": payload_batch_size,
                        "schedule": schedule,
                        "metadata": metadata,
                    }
                )
                yield (final_payload + "\n").encode("utf-8")
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    @app.get("/knowledge/glossary", response_class=JSONResponse)
    async def list_glossary_entries_endpoint(
        project_id: str = Query(...),
        project_store: ProjectStore = Depends(get_project_store),
    ) -> JSONResponse:
        project = _resolve_project(project_store, project_id)
        entries = project_store.list_glossary_entries(project.id)
        entries.sort(key=lambda item: str(item.get("term", "")).lower())
        return JSONResponse({"projectId": project.id, "entries": entries})

    @app.post("/knowledge/glossary", response_class=JSONResponse)
    async def create_glossary_entry_endpoint(
        request: Request,
        project_store: ProjectStore = Depends(get_project_store),
    ) -> JSONResponse:
        payload = await request.json()
        project_id = (payload.get("project_id") or payload.get("projectId") or "").strip()
        project = _resolve_project(project_store, project_id)
        try:
            entry = project_store.create_glossary_entry(
                project.id,
                term=str(payload.get("term", "")),
                definition=str(payload.get("definition", "")),
                synonyms=_parse_string_list(payload.get("synonyms")),
                keywords=_parse_string_list(payload.get("keywords")),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(entry, status_code=201)

    @app.put("/knowledge/glossary/{entry_id}", response_class=JSONResponse)
    async def update_glossary_entry_endpoint(
        entry_id: str,
        request: Request,
        project_store: ProjectStore = Depends(get_project_store),
    ) -> JSONResponse:
        payload = await request.json()
        project_id = (payload.get("project_id") or payload.get("projectId") or "").strip()
        project = _resolve_project(project_store, project_id)
        try:
            entry = project_store.update_glossary_entry(
                project.id,
                entry_id,
                term=payload.get("term"),
                definition=payload.get("definition"),
                synonyms=_parse_string_list(payload.get("synonyms")) if "synonyms" in payload else None,
                keywords=_parse_string_list(payload.get("keywords")) if "keywords" in payload else None,
            )
        except ValueError as exc:
            message = str(exc)
            status = 404 if "not found" in message.lower() else 400
            raise HTTPException(status_code=status, detail=message) from exc
        return JSONResponse(entry)

    @app.delete("/knowledge/glossary/{entry_id}", response_class=JSONResponse)
    async def delete_glossary_entry_endpoint(
        entry_id: str,
        project_id: str = Query(...),
        project_store: ProjectStore = Depends(get_project_store),
    ) -> JSONResponse:
        project = _resolve_project(project_store, project_id)
        deleted = project_store.delete_glossary_entry(project.id, entry_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Glossary entry not found")
        return JSONResponse({"status": "deleted"})

    @app.post("/knowledge/cleanup", response_class=RedirectResponse)
    async def cleanup_project(
        request: Request,
        project_id: str = Form(...),
        project_store: ProjectStore = Depends(get_project_store),
        store_manager: ProjectVectorStoreManager = Depends(get_store_manager),
        fts_index: FTSIndex = Depends(get_fts_index),
        settings: Settings = Depends(get_settings_dependency),
    ) -> RedirectResponse:
        project = _resolve_project(project_store, project_id)
        documents = project_store.list_documents(project.id)
        cleared = store_manager.purge_project(project.id)
        for doc in documents:
            fts_index.delete_document(project.id, doc.id)
        project_store.clear_documents(project.id)
        manifest_path = Path(settings.data_dir).resolve() / "manifests" / f"{project.id}.json"
        if manifest_path.exists():
            try:
                manifest_path.unlink()
                logger.info(
                    "knowledge.cleanup.manifest_removed project=%s path=%s",
                    project.id,
                    manifest_path,
                )
            except OSError as exc:  # pragma: no cover - filesystem failure
                logger.warning(
                    "knowledge.cleanup.manifest_remove_failed project=%s path=%s error=%s",
                    project.id,
                    manifest_path,
                    exc,
                )
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
        fts_index: FTSIndex = Depends(get_fts_index),
    ) -> RedirectResponse:
        project = _resolve_project(project_store, project_id)
        if not project_store.remove_document(project.id, doc_id):
            raise HTTPException(status_code=404, detail="Document not found")
        removed_vectors = store_manager.delete_document(project.id, doc_id)
        fts_index.delete_document(project.id, doc_id)
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


def _process_ingestion_job(
    job_manager: IngestionJobManager,
    ingestion_service: DocumentIngestor,
    job_id: str,
    project_id: str,
    filename: str,
    content_type: str | None,
    data: bytes | None = None,
    source_url: str | None = None,
) -> None:
    total_bytes = len(data) if data is not None else None
    logger.info(
        "ingest.job.start job_id=%s project=%s filename=%s source=%s size_bytes=%s",
        job_id,
        project_id,
        filename,
        "url" if source_url else "upload",
        total_bytes,
    )
    job_manager.mark_processing(
        job_id,
        total_files=1,
        processed_files=0,
        total_bytes=total_bytes,
        processed_bytes=0 if total_bytes is not None else None,
    )
    try:
        if source_url:
            result = ingestion_service.ingest_url(
                project_id,
                url=source_url,
            )
        else:
            if data is None:
                raise ValueError("No data provided for ingestion job")
            result = ingestion_service.ingest_bytes(
                project_id,
                filename=filename,
                data=data,
                content_type=content_type,
            )
        job_manager.mark_completed(
            job_id,
            result.document,
            processed_files=1,
            processed_bytes=result.document.size_bytes,
        )
        logger.info("ingest.job.completed job_id=%s project=%s", job_id, project_id)
    except Exception as exc:  # pragma: no cover - background task failure
        logger.exception("ingest.job.failed job_id=%s project=%s error=%s", job_id, project_id, exc)
        job_manager.mark_failed(
            job_id,
            str(exc),
            processed_files=0,
            total_files=1,
        )
