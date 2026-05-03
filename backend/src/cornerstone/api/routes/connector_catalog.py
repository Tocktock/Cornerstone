from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from cornerstone.config import Settings, get_settings
from cornerstone.connectors.catalog import get_connector_definition, list_connector_definitions
from cornerstone.observability import log_event
from cornerstone.schemas import ConnectorDefinition, DataSourceType
from cornerstone.schemas import ConnectorSupportMatrixItem, ConnectorSupportMatrixResponse

router = APIRouter(tags=["connectors"])


@router.get("/connectors", response_model=list[ConnectorDefinition])
def list_connectors(settings: Settings = Depends(get_settings)) -> list[ConnectorDefinition]:
    definitions = list_connector_definitions(settings)
    log_event("connector.catalog_listed", providerCount=len(definitions))
    return definitions


@router.get("/connectors/support-matrix", response_model=ConnectorSupportMatrixResponse)
def get_connector_support_matrix(settings: Settings = Depends(get_settings)) -> ConnectorSupportMatrixResponse:
    items: list[ConnectorSupportMatrixItem] = []
    for definition in list_connector_definitions(settings):
        for object_type in definition.supported_objects:
            state = "supported" if object_type in definition.ingestible_objects else "discoverable_only"
            items.append(
                ConnectorSupportMatrixItem(
                    provider=definition.provider,
                    object_type=object_type,
                    support_state=state,
                    proof_state="live_proof_gated" if definition.provider != DataSourceType.MANUAL else "local_regression",
                    creates_evidence=object_type in definition.ingestible_objects,
                    queues_candidate_reextraction=object_type in definition.ingestible_objects,
                    mutates_official_graph=False,
                    limitations=definition.limitations,
                )
            )
        for object_type in definition.discoverable_objects:
            if object_type in definition.supported_objects:
                continue
            items.append(
                ConnectorSupportMatrixItem(
                    provider=definition.provider,
                    object_type=object_type,
                    support_state="discoverable_only",
                    proof_state="not_ingested",
                    creates_evidence=False,
                    queues_candidate_reextraction=False,
                    mutates_official_graph=False,
                    limitations=definition.limitations,
                )
            )
    return ConnectorSupportMatrixResponse(
        items=items,
        secret_redaction_policy="Live proof artifacts must record provider/object ids and counts only; OAuth tokens, API keys, refresh tokens, and raw Authorization headers are never persisted.",
        live_proof_env_guards=[
            "NOTION_E2E_ACCESS_TOKEN",
            "NOTION_E2E_PAGE_ID",
            "GOOGLE_DRIVE_E2E_ACCESS_TOKEN",
            "GOOGLE_DRIVE_E2E_FILE_ID",
            "ONTOLOGY_LIVE_LLM_ENABLED",
        ],
    )


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
