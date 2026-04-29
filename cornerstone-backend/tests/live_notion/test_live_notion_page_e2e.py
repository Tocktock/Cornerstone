from __future__ import annotations

import asyncio
import os

import pytest

from cornerstone.config import Settings
from cornerstone.persistence.database import create_persistent_store
from cornerstone.verification.notion_e2e import (
    NotionE2EConfigError,
    notion_e2e_config_from_env,
    run_live_notion_page_e2e,
)

pytestmark = pytest.mark.live_notion


def test_live_notion_page_to_artifact_and_evidence_e2e() -> None:
    settings = Settings.from_env()
    config = notion_e2e_config_from_env(os.environ)
    try:
        # Validate explicitly so default local runs never touch real Notion accidentally.
        from cornerstone.verification.notion_e2e import assert_notion_e2e_config_safe

        assert_notion_e2e_config_safe(config, settings)
    except NotionE2EConfigError as exc:
        pytest.skip(str(exc))

    store = create_persistent_store(settings) if settings.persistence_backend == "postgres" else None
    if store is None:
        pytest.skip("Live Notion E2E requires PERSISTENCE_BACKEND=postgres for the CI path.")
    store.reset()
    result = asyncio.run(run_live_notion_page_e2e(store=store, settings=settings, config=config))

    assert result.sync_job_status == "succeeded"
    assert result.artifact_count >= 1
    if config.require_evidence:
        assert result.evidence_fragment_count >= 1
    assert result.source_next_action == "review_evidence"
