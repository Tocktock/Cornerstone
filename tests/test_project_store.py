from __future__ import annotations

from pathlib import Path

import pytest

from cornerstone.projects import KeywordRunRecord, ProjectStore


def _make_store(tmp_path: Path) -> ProjectStore:
    root = tmp_path / "data"
    return ProjectStore(root, default_project_name="Demo Project")


def test_keyword_run_persistence_roundtrip(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    project = store.find_by_name("Demo Project")
    assert project is not None

    run = store.create_keyword_run(project.id, requested_by="alice@example.com")
    assert run.status == "pending"

    updated = store.update_keyword_run(
        project.id,
        run.id,
        status="running",
        started_at="2025-01-17T10:00:00+00:00",
        stats={"chunk_total": 10, "token_total": 5000},
    )
    assert updated.status == "running"
    assert updated.stats["chunk_total"] == 10

    final = store.update_keyword_run(
        project.id,
        run.id,
        status="success",
        completed_at="2025-01-17T10:05:00+00:00",
        keywords=[{"term": "alpha", "count": 5}],
        insights=[{"term": "alpha", "summary": "test"}],
        debug={"candidate_count": 123},
        stats={"token_total": 750000},
        insight_job={"id": "job-1", "status": "success"},
    )
    assert final.status == "success"
    assert final.stats["token_total"] == 750000
    assert final.insight_job == {"id": "job-1", "status": "success"}
    latest = store.get_latest_keyword_run(project.id)
    assert latest is not None
    assert latest.id == final.id
    assert latest.keywords and latest.keywords[0]["term"] == "alpha"


def test_keyword_run_pointer_removed_on_failure(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    project = store.find_by_name("Demo Project")
    assert project is not None

    first = store.create_keyword_run(project.id)
    store.update_keyword_run(project.id, first.id, status="success")
    assert store.get_latest_keyword_run(project.id).id == first.id

    second = store.create_keyword_run(project.id)
    store.update_keyword_run(project.id, second.id, status="running")
    store.update_keyword_run(project.id, second.id, status="error", error="timeout")

    latest = store.get_latest_keyword_run(project.id)
    assert latest is not None
    assert latest.id == first.id


def test_keyword_run_listing_and_limit(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    project = store.find_by_name("Demo Project")
    assert project is not None

    run_ids = []
    for idx in range(5):
        run = store.create_keyword_run(project.id)
        store.update_keyword_run(project.id, run.id, status="success")
        run_ids.append(run.id)

    all_runs = store.list_keyword_runs(project.id)
    assert len(all_runs) == 5
    assert all_runs[0].requested_at >= all_runs[-1].requested_at

    limited = store.list_keyword_runs(project.id, limit=2)
    assert len(limited) == 2
    assert {record.id for record in limited} <= set(run_ids)
