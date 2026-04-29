from __future__ import annotations

from cornerstone.config import Settings
from cornerstone.schemas import (
    ConnectorAuthType,
    ConnectorAvailability,
    ConnectorCapabilities,
    ConnectorDefinition,
    ConnectorScope,
    ConnectorSetupStep,
    DataSourceType,
)


def list_connector_definitions(settings: Settings) -> list[ConnectorDefinition]:
    return [
        ConnectorDefinition(
            provider=DataSourceType.NOTION,
            display_name="Notion",
            description="Connect selected Notion pages as organizational evidence sources while discovering databases/data_sources for future ingestion support.",
            auth_type=ConnectorAuthType.OAUTH2,
            availability=ConnectorAvailability.AVAILABLE,
            production_ready=False,
            required_scopes=[
                ConnectorScope(
                    key="read_content",
                    label="Read selected content",
                    description="Read content explicitly granted to the Cornerstone Notion integration.",
                )
            ],
            optional_scopes=[
                ConnectorScope(
                    key="read_comments",
                    label="Read comments",
                    description="Reserved for future comment and discussion evidence extraction.",
                    required=False,
                )
            ],
            supported_objects=["page"],
            discoverable_objects=["page", "database", "data_source"],
            ingestible_objects=["page"],
            setup_steps=[
                ConnectorSetupStep(
                    key="authorize",
                    title="Authorize Notion",
                    description="Open Notion OAuth and grant Cornerstone access to selected pages.",
                    order=1,
                ),
                ConnectorSetupStep(
                    key="test_connection",
                    title="Test connection",
                    description="Verify that Cornerstone can read the workspace metadata.",
                    order=2,
                ),
                ConnectorSetupStep(
                    key="select_sources",
                    title="Select sources",
                    description="Choose supported pages to sync before the first ingestion job.",
                    order=3,
                ),
            ],
            limitations=[
                "v0.8.2 supports selected Notion page ingestion into normalized Artifacts.",
                "Databases/data_sources are discoverable but not selectable for sync until their ingestion semantics are implemented.",
            ],
            docs_url="https://developers.notion.com/docs/authorization",
            capabilities=ConnectorCapabilities(
                supports_oauth=True,
                supports_discovery=True,
                supports_selection=True,
                supports_incremental_sync=False,
                supports_webhooks=False,
                supports_source_updated_at=True,
                supports_source_urls=True,
                supports_permission_snapshots=False,
            ),
        ),
        ConnectorDefinition(
            provider=DataSourceType.SLACK,
            display_name="Slack",
            description="Extract evidence from selected channels and threads.",
            auth_type=ConnectorAuthType.OAUTH2,
            availability=ConnectorAvailability.COMING_SOON,
            production_ready=False,
            required_scopes=[],
            optional_scopes=[],
            supported_objects=["channel", "thread", "message"],
            discoverable_objects=["channel", "thread", "message"],
            ingestible_objects=["thread", "message"],
            setup_steps=[],
            limitations=["Coming after Notion because conversation ingestion requires stricter selection and privacy UX."],
            capabilities=ConnectorCapabilities(
                supports_oauth=True,
                supports_discovery=True,
                supports_selection=True,
                supports_incremental_sync=True,
                supports_source_urls=True,
                supports_author_metadata=True,
                supports_permission_snapshots=True,
            ),
        ),
        ConnectorDefinition(
            provider=DataSourceType.GOOGLE_DOCS,
            display_name="Google Docs",
            description="Extract evidence from selected Google Drive documents.",
            auth_type=ConnectorAuthType.OAUTH2,
            availability=ConnectorAvailability.COMING_SOON,
            production_ready=False,
            required_scopes=[],
            optional_scopes=[],
            supported_objects=["document"],
            discoverable_objects=["document", "folder", "shared_drive_file"],
            ingestible_objects=["document"],
            setup_steps=[],
            limitations=["Coming after Notion."],
            capabilities=ConnectorCapabilities(
                supports_oauth=True,
                supports_discovery=True,
                supports_selection=True,
                supports_incremental_sync=True,
                supports_source_updated_at=True,
                supports_source_urls=True,
                supports_author_metadata=True,
            ),
        ),
        ConnectorDefinition(
            provider=DataSourceType.GITHUB,
            display_name="GitHub",
            description="Extract implementation evidence from selected repositories and files.",
            auth_type=ConnectorAuthType.GITHUB_APP,
            availability=ConnectorAvailability.COMING_SOON,
            production_ready=False,
            required_scopes=[],
            optional_scopes=[],
            supported_objects=["repository", "file", "pull_request", "issue"],
            discoverable_objects=["repository", "file", "pull_request", "issue"],
            ingestible_objects=["file", "pull_request", "issue"],
            setup_steps=[],
            limitations=["Coming after document connectors."],
            capabilities=ConnectorCapabilities(
                supports_discovery=True,
                supports_selection=True,
                supports_incremental_sync=True,
                supports_webhooks=True,
                supports_source_updated_at=True,
                supports_source_urls=True,
                supports_author_metadata=True,
            ),
        ),
        ConnectorDefinition(
            provider=DataSourceType.MANUAL,
            display_name="Manual source",
            description="Capture manually entered organizational context as explicitly labeled evidence.",
            auth_type=ConnectorAuthType.MANUAL,
            availability=ConnectorAvailability.AVAILABLE,
            production_ready=True,
            required_scopes=[],
            optional_scopes=[],
            supported_objects=["manual_note"],
            discoverable_objects=["manual_note"],
            ingestible_objects=["manual_note"],
            setup_steps=[],
            limitations=["Manual evidence must still pass review before becoming official."],
            capabilities=ConnectorCapabilities(
                supports_discovery=True,
                supports_selection=True,
                supports_source_updated_at=False,
                supports_source_urls=True,
                supports_author_metadata=True,
            ),
        ),
    ]


def get_connector_definition(provider: DataSourceType, settings: Settings) -> ConnectorDefinition:
    for definition in list_connector_definitions(settings):
        if definition.provider == provider:
            return definition
    raise KeyError(provider)
