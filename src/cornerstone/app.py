"""FastAPI application setup for the Cornerstone prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from .config import Settings
from .embeddings import EmbeddingService
from .glossary import Glossary, load_glossary
from .chat import SupportAgentService
from .vector_store import QdrantVectorStore, SearchResult

_MIN_RESULT_SCORE = 1e-6

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"


class ApplicationState:
    """Container for runtime dependencies used by the FastAPI app."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        glossary: Glossary,
        chat_service: SupportAgentService,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.glossary = glossary
        self.chat_service = chat_service


def create_app(
    *,
    settings: Settings | None = None,
    embedding_service: EmbeddingService | None = None,
    vector_store: QdrantVectorStore | None = None,
    glossary: Glossary | None = None,
    chat_service: SupportAgentService | None = None,
    ensure_collection: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = settings or Settings.from_env()
    embedding_service = embedding_service or EmbeddingService(settings)
    vector_store = vector_store or QdrantVectorStore.from_settings(
        settings, vector_size=embedding_service.dimension
    )

    if ensure_collection:
        vector_store.ensure_collection()

    glossary = glossary or load_glossary(settings.glossary_path)

    chat_service = chat_service or SupportAgentService(
        settings=settings,
        embedding_service=embedding_service,
        vector_store=vector_store,
        glossary=glossary,
    )

    app = FastAPI()
    app.state.services = ApplicationState(embedding_service, vector_store, glossary, chat_service)

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def get_embedding_service(request: Request) -> EmbeddingService:
        return request.app.state.services.embedding_service

    def get_vector_store(request: Request) -> QdrantVectorStore:
        return request.app.state.services.vector_store

    def get_chat_service(request: Request) -> SupportAgentService:
        return request.app.state.services.chat_service

    def get_glossary(request: Request) -> Glossary:
        return request.app.state.services.glossary

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:  # pragma: no cover - template rendering
        return templates.TemplateResponse(request, "index.html", {})

    @app.post("/search", response_class=HTMLResponse)
    async def search(  # pragma: no cover - template rendering
        request: Request,
        query: str = Form(...),
        embedding: EmbeddingService = Depends(get_embedding_service),
        store: QdrantVectorStore = Depends(get_vector_store),
    ) -> HTMLResponse:
        vector = embedding.embed_one(query)
        hits = store.search(vector)
        context = {
            "query": query,
            "results": _format_results(hits),
        }
        return templates.TemplateResponse(request, "index.html", context)

    @app.get("/support", response_class=HTMLResponse)
    async def support_page(request: Request) -> HTMLResponse:  # pragma: no cover - template rendering
        glossary_count = len(get_glossary(request))
        context = {"request": request, "glossary_count": glossary_count}
        return templates.TemplateResponse("support.html", context)

    @app.post("/support/chat", response_class=JSONResponse)
    async def support_chat(
        request: Request,
        chat_service: SupportAgentService = Depends(get_chat_service),
    ) -> JSONResponse:
        payload = await request.json()
        query = str(payload.get("query", "")).strip()
        history = payload.get("history") or []
        if not query:
            return JSONResponse({"error": "Query is required."}, status_code=400)
        response = chat_service.generate(query, conversation=history)
        return JSONResponse(
            {
                "message": response.message,
                "sources": response.sources,
                "definitions": response.definitions,
            }
        )

    return app


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
