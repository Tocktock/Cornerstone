from __future__ import annotations

from fastapi import APIRouter

from cornerstone import __version__
from cornerstone.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok", service="cornerstone-backend", version=__version__)
