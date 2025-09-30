"""FastAPI application setup for the Cornerstone prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .config import Settings
from .embeddings import EmbeddingService
from .vector_store import QdrantVectorStore, SearchResult

_MIN_RESULT_SCORE = 1e-6

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"


class ApplicationState:
    """Container for runtime dependencies used by the FastAPI app."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: QdrantVectorStore) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store


def create_app(
    *,
    settings: Settings | None = None,
    embedding_service: EmbeddingService | None = None,
    vector_store: QdrantVectorStore | None = None,
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

    app = FastAPI()
    app.state.services = ApplicationState(embedding_service, vector_store)

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def get_embedding_service(request: Request) -> EmbeddingService:
        return request.app.state.services.embedding_service

    def get_vector_store(request: Request) -> QdrantVectorStore:
        return request.app.state.services.vector_store

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
