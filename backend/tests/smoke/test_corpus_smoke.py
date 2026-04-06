from __future__ import annotations

import os
from pathlib import Path

import pytest

from cornerstone.config import Settings
from cornerstone.database import Database
from cornerstone.domain.enums import ContextSpaceKind, SyncMode, VisibilityClass
from cornerstone.domain.models import ContextSpace, SourceConnection
from cornerstone.services.bootstrap import initialize_database
from cornerstone.services.normalization import stable_id
from cornerstone.services.sync import run_sync


@pytest.mark.smoke
def test_curated_fixture_slice_syncs_with_markdown_and_csv(test_database_url: str):
    curated_root = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "curated"
        / "workspace"
        / "member-visible"
    )
    settings = Settings(database_url=test_database_url, fixed_now="2026-04-06T09:00:00+09:00")
    database = Database(test_database_url)
    initialize_database(database.engine, reset=True)

    with database.session_factory() as session:
        context_space = ContextSpace(
            id=stable_id("ctx", "curated-smoke"),
            kind=ContextSpaceKind.WORKSPACE,
            name="Curated smoke",
            slug="curated-smoke",
            membership_boundary="workspace:curated-smoke",
            default_visibility_class=VisibilityClass.MEMBER_VISIBLE,
            visibility_defaults={},
            is_default=False,
        )
        session.add(context_space)
        session.flush()
        connection = SourceConnection(
            id=stable_id("source", "curated-smoke"),
            context_space_id=context_space.id,
            provider="filesystem",
            source_label="Curated fixture slice",
            source_boundary_locator=str(curated_root),
            template_key="curated",
            visibility_class=VisibilityClass.MEMBER_VISIBLE,
            sync_mode=SyncMode.POLLING,
            sync_interval_seconds=300,
            effective_sync_policy={"suite": "smoke"},
        )
        session.add(connection)
        session.flush()

        result = run_sync(session, connection, settings)

        assert result.artifact_count == 2
        assert result.support_item_count == 3
        assert result.source_connection_state.value == "active"
        assert result.freshness_state.value == "current"


@pytest.mark.corpus
def test_full_sample_corpus_smoke_is_available_when_enabled():
    if os.getenv("CORNERSTONE_RUN_CORPUS_SMOKE") != "1":
        pytest.skip("Full sample-data smoke is opt-in.")

    corpus_root = Path(__file__).resolve().parents[3] / "sample-data" / "sendy-knowledge"
    assert corpus_root.exists()
    assert any(corpus_root.rglob("*.md"))
