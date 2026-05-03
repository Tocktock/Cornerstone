from __future__ import annotations

import ast
import textwrap
from datetime import timedelta
from pathlib import Path

import pytest

from cornerstone.connectors.providers.common import parse_datetime, stable_hash
from cornerstone.domain.ontology import ONTOLOGY_GRAPH_MAX_DEPTH, ensure_supported_graph_depth
from cornerstone.schemas import DataSourceType, SyncJob, SyncJobStatus, utc_now
from cornerstone.services.sync_jobs import sync_job_is_claimable


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src" / "cornerstone"


def test_ontology_depth_policy_is_centralized() -> None:
    assert ONTOLOGY_GRAPH_MAX_DEPTH == 1
    ensure_supported_graph_depth(0)
    ensure_supported_graph_depth(1)

    with pytest.raises(ValueError, match="maximum depth 1"):
        ensure_supported_graph_depth(2)


def test_sync_job_claimability_rule_is_shared_and_deterministic() -> None:
    now = utc_now()
    queued = SyncJob(datasource_id="src_1", provider=DataSourceType.MANUAL)
    assert sync_job_is_claimable(queued, now=now, include_not_ready=False)

    waiting = SyncJob(
        datasource_id="src_1",
        provider=DataSourceType.MANUAL,
        status=SyncJobStatus.RETRY_WAITING,
        next_attempt_at=now + timedelta(minutes=5),
    )
    assert not sync_job_is_claimable(waiting, now=now, include_not_ready=False)
    assert sync_job_is_claimable(waiting, now=now, include_not_ready=True)

    expired_running = SyncJob(
        datasource_id="src_1",
        provider=DataSourceType.MANUAL,
        status=SyncJobStatus.RUNNING,
        lease_expires_at=now - timedelta(seconds=1),
    )
    assert sync_job_is_claimable(expired_running, now=now, include_not_ready=False)


def test_connector_common_helpers_are_provider_neutral() -> None:
    assert parse_datetime("2026-05-02T12:00:00Z") is not None
    assert parse_datetime(None) is None

    left = {"id": "a", "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "id": "a"}
    assert stable_hash(left) == stable_hash(right)


def test_no_duplicate_function_bodies_after_refactor() -> None:
    bodies: dict[str, list[str]] = {}
    ignored_names = {"__init__"}
    for path in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in ignored_names:
                continue
            body_dump = ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)
            line_count = max((getattr(stmt, "end_lineno", stmt.lineno) - stmt.lineno + 1 for stmt in node.body), default=0)
            if line_count < 5:
                continue
            bodies.setdefault(body_dump, []).append(f"{path.relative_to(ROOT)}::{node.name}")

    duplicates = {body: refs for body, refs in bodies.items() if len(refs) > 1}
    formatted = "\n".join(textwrap.shorten(", ".join(refs), width=240) for refs in duplicates.values())
    assert not duplicates, formatted
