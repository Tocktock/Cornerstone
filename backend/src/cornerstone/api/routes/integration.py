from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from cornerstone.api.dependencies import get_store
from cornerstone.config import Settings, get_settings
from cornerstone.integration import build_integration_manifest, build_integration_ontology_response
from cornerstone.schemas import IntegrationOntologyResponse, IntegrationPackageManifest

router = APIRouter(prefix="/integration", tags=["integration"])


@router.get("/package/manifest", response_model=IntegrationPackageManifest)
def get_integration_package_manifest() -> IntegrationPackageManifest:
    return build_integration_manifest()


@router.get("/ontology/{concept}", response_model=IntegrationOntologyResponse)
def get_integration_ontology(
    concept: str,
    include_candidates: bool = Query(default=False, alias="includeCandidates"),
    store: object = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> IntegrationOntologyResponse:
    if include_candidates:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Integration package exposes candidates only as summaries; review gates cannot be bypassed.",
        )
    try:
        return build_integration_ontology_response(
            store=store,
            concept=concept,
            production_mode=settings.production_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
