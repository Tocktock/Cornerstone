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
from .keywords import KeywordLLMFilter, build_excerpt, extract_keyword_candidates
from .personas import PersonaOverrides, PersonaSnapshot, PersonaStore
from .projects import Project, ProjectStore
from .vector_store import QdrantVectorStore, SearchResult

_MIN_RESULT_SCORE = 1e-6
_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"

logger = logging.getLogger(__name__)
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False


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
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.glossary = glossary
        self.project_store = project_store
        self.persona_store = persona_store
        self.store_manager = store_manager
        self.chat_service = chat_service
        self.ingestion_service = ingestion_service


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
) -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = settings or Settings.from_env()
    embedding_service = embedding_service or EmbeddingService(settings)

    project_root = Path(settings.data_dir).resolve()
    project_store = project_store or ProjectStore(project_root, default_project_name=settings.default_project_name)
    persona_store = persona_store or PersonaStore(project_root)
    logger.info(
        "app.start settings_loaded data_dir=%s default_project=%s",
        project_root,
        settings.default_project_name,
    )

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
        persona_store=persona_store,
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
        persona_store=persona_store,
        store_manager=store_manager,
        chat_service=chat_service,
        ingestion_service=ingestion_service,
    )

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
    ) -> JSONResponse:
        payload = await request.json()
        query = str(payload.get("query", "")).strip()
        history = payload.get("history") or []
        project_id = payload.get("projectId")
        project = _resolve_project(project_store, project_id)
        if not query:
            return JSONResponse({"error": "Query is required."}, status_code=400)
        logger.info("support.endpoint request project=%s history_turns=%s", project.id, len(history))
        response = chat_service.generate(project, query, conversation=history)
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
        }

    def _parse_tags(raw: str | None) -> list[str]:
        if not raw:
            return []
        return [tag.strip() for tag in raw.split(",") if tag.strip()]

    @app.get("/knowledge", response_class=HTMLResponse)
    async def knowledge_dashboard(request: Request) -> HTMLResponse:  # pragma: no cover - template rendering
        project_store = get_project_store(request)
        persona_store = get_persona_store(request)
        projects = project_store.list_projects()
        selected_project = request.query_params.get("project_id") or _default_project_id(project_store)
        logger.info("knowledge.dashboard load project=%s", selected_project)
        project = _resolve_project(project_store, selected_project)
        documents = project_store.list_documents(project.id)
        personas = persona_store.list_personas()
        persona_snapshot = persona_store.resolve_persona(project.persona_id, project.persona_overrides)
        context = {
            "request": request,
            "projects": projects,
            "selected_project": project.id,
            "project_name": project.name,
            "documents": documents,
            "persona": persona_snapshot,
            "persona_base": persona_snapshot.base_persona if persona_snapshot else None,
            "personas": personas,
            "project_persona_id": project.persona_id,
            "persona_overrides": project.persona_overrides,
        }
        return templates.TemplateResponse("knowledge.html", context)

    @app.post("/knowledge/persona", response_class=RedirectResponse)
    async def update_persona_settings(
        request: Request,
        project_id: str = Form(...),
        persona_id: str | None = Form(None),
        persona_name: str | None = Form(None),
        persona_tone: str | None = Form(None),
        persona_system_prompt: str | None = Form(None),
        persona_avatar_url: str | None = Form(None),
        project_store: ProjectStore = Depends(get_project_store),
        persona_store: PersonaStore = Depends(get_persona_store),
    ) -> RedirectResponse:
        project = _resolve_project(project_store, project_id)
        overrides = PersonaOverrides(
            name=persona_name,
            tone=persona_tone,
            system_prompt=persona_system_prompt,
            avatar_url=persona_avatar_url,
        )
        updated = project_store.configure_persona(
            project.id,
            persona_id=persona_id,
            overrides=overrides,
        )
        resolved = persona_store.resolve_persona(updated.persona_id, updated.persona_overrides)
        has_overrides = any(
            getattr(updated.persona_overrides, field)
            for field in ("name", "tone", "system_prompt", "avatar_url")
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
        persona_store: PersonaStore = Depends(get_persona_store),
    ) -> RedirectResponse:
        new_persona = persona_store.create_persona(
            name=name,
            description=description,
            tone=tone,
            system_prompt=system_prompt,
            avatar_url=avatar_url,
            tags=_parse_tags(tags),
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

        data = {
            "projectId": project.id,
            "keywords": [
                {
                    "term": item.term,
                    "count": item.count,
                    "core": item.is_core,
                    "generated": item.generated,
                    "reason": item.reason,
                    "source": item.source,
                }
                for item in keywords
            ],
            "filter": debug_payload,
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
