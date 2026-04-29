from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from cornerstone.config import Settings, get_settings
from cornerstone.connectors.catalog import get_connector_definition, list_connector_definitions
from cornerstone.observability import log_event
from cornerstone.schemas import ConnectorDefinition, DataSourceType

router = APIRouter(tags=["connectors"])


@router.get("/connectors", response_model=list[ConnectorDefinition])
def list_connectors(settings: Settings = Depends(get_settings)) -> list[ConnectorDefinition]:
    definitions = list_connector_definitions(settings)
    log_event("connector.catalog_listed", providerCount=len(definitions))
    return definitions


@router.get("/connectors/{provider}", response_model=ConnectorDefinition)
def get_connector_catalog_entry(
    provider: DataSourceType,
    settings: Settings = Depends(get_settings),
) -> ConnectorDefinition:
    try:
        definition = get_connector_definition(provider, settings)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found.") from exc
    log_event("connector.catalog_entry_read", provider=provider)
    return definition
