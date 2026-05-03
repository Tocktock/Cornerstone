from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from cornerstone.api.dependencies import get_store
from cornerstone.config import Settings, get_settings
from cornerstone.observability import log_event
from cornerstone.schemas import GroundedContextResponse
from cornerstone.services.grounded_context import GroundedContextService
from cornerstone.store import InMemoryStore

router = APIRouter(prefix="/context", tags=["context"])


@router.get("/query", response_model=GroundedContextResponse)
def query_context(
    q: str = Query(min_length=1),
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> GroundedContextResponse:
    response = GroundedContextService(store, production_mode=settings.production_mode).query(q)
    log_event(
        "context.query_completed",
        query=q,
        trustLabel=response.trust_label,
        conceptCount=len(response.concepts),
        evidenceCount=len(response.evidence),
        freshnessState=response.freshness.state,
        limitationCount=len(response.limitations),
    )
    return response
