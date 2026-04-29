from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from cornerstone.api.dependencies import get_store
from cornerstone.schemas import Artifact
from cornerstone.store import InMemoryStore

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.get("", response_model=list[Artifact])
def list_artifacts(
    data_source_id: str | None = Query(default=None, alias="dataSourceId"),
    store: InMemoryStore = Depends(get_store),
) -> list[Artifact]:
    return store.list_artifacts(datasource_id=data_source_id)
