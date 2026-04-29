from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cornerstone.config import Settings
from cornerstone.schemas import (
    ConnectionTestResult,
    ConnectionTestStatus,
    DataSourceType,
    ProviderObjectAccessState,
    ProviderObjectSnapshot,
    ProviderObjectType,
    SourceObject,
)
from cornerstone.store import InMemoryStore
from cornerstone.verification import notion_e2e as e2e_module
from cornerstone.verification.notion_e2e import (
    NotionE2EConfig,
    notion_e2e_config_from_env,
    notion_e2e_config_issues,
    run_live_notion_page_e2e,
)

pytestmark = pytest.mark.unit


def _safe_live_settings() -> Settings:
    return Settings(
        notion_mock_external_api=False,
        connector_encryption_secret="live-notion-e2e-secret-value-32+",
    )


def test_notion_e2e_config_from_env_requires_explicit_opt_in_and_page() -> None:
    config = notion_e2e_config_from_env({})

    issues = notion_e2e_config_issues(config, Settings())

    assert config.enabled is False
    assert {issue.code for issue in issues} >= {
        "notion_e2e_not_enabled",
        "notion_e2e_access_token_required",
        "notion_e2e_page_id_required",
        "notion_e2e_requires_live_notion_api",
        "notion_e2e_requires_non_default_connector_secret",
    }


def test_notion_e2e_config_accepts_safe_live_inputs() -> None:
    config = notion_e2e_config_from_env(
        {
            "RUN_NOTION_E2E": "1",
            "NOTION_E2E_ACCESS_TOKEN": "secret-token",
            "NOTION_E2E_PAGE_ID": "page-1",
            "NOTION_E2E_REQUIRE_EVIDENCE": "true",
        }
    )

    issues = notion_e2e_config_issues(config, _safe_live_settings())

    assert issues == []
    assert config.enabled is True
    assert config.page_id == "page-1"
    assert config.require_evidence is True


def test_live_notion_e2e_runner_uses_same_worker_ingestion_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLiveNotionConnector:
        provider = DataSourceType.NOTION

        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        async def test_connection(self, *, credential: Any, access_token: str) -> ConnectionTestResult:
            assert access_token == "live-token"
            return ConnectionTestResult(
                status=ConnectionTestStatus.PASSED,
                datasource_id=credential.datasource_id,
                provider=DataSourceType.NOTION,
                external_workspace_name="Live Workspace",
                granted_scopes=["read_content"],
                can_read_objects=True,
                sample_object_count=1,
            )

        async def retrieve_page_snapshot(
            self,
            *,
            credential: Any,
            access_token: str,
            page_id: str,
        ) -> ProviderObjectSnapshot:
            assert access_token == "live-token"
            return ProviderObjectSnapshot(
                datasource_id=credential.datasource_id,
                provider=DataSourceType.NOTION,
                external_id=page_id,
                external_url="https://www.notion.so/live-page",
                object_type=ProviderObjectType.PAGE,
                title="Live Pilot Page",
                access_state=ProviderObjectAccessState.ACCESSIBLE,
                ingestion_supported=True,
                raw_metadata_hash="live-page-metadata-hash",
                provider_metadata={"e2e": True},
            )

        async def list_objects(
            self,
            *,
            credential: Any,
            access_token: str,
            selection: Any,
            selected_objects: list[ProviderObjectSnapshot] | None = None,
        ) -> list[SourceObject]:
            assert access_token == "live-token"
            assert selected_objects is not None
            return [
                SourceObject(
                    source_external_id=selected_objects[0].external_id,
                    title="Live Pilot Page",
                    content="Cornerstone live Notion E2E evidence requires provenance and freshness.",
                    source_url="https://www.notion.so/live-page",
                    source_object_type="page",
                    provider_metadata={"provider": "notion", "contentFormat": "test"},
                )
            ]

    monkeypatch.setattr(e2e_module, "NotionConnector", FakeLiveNotionConnector)
    import cornerstone.services.sync_worker as sync_worker_module

    monkeypatch.setattr(sync_worker_module, "get_connector", lambda provider, settings: FakeLiveNotionConnector(settings))
    result = asyncio.run(
        run_live_notion_page_e2e(
            store=InMemoryStore(),
            settings=_safe_live_settings(),
            config=NotionE2EConfig(enabled=True, access_token="live-token", page_id="page-1"),
        )
    )

    assert result.sync_job_status == "succeeded"
    assert result.artifact_count == 1
    assert result.evidence_fragment_count > 0
    assert result.source_next_action == "review_evidence"
