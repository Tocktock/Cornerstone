from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cornerstone.keyword_jobs import KeywordRunExecutor, KeywordRunJob, KeywordRunQueue
from cornerstone.projects import ProjectStore


def _make_store(tmp_path: Path) -> ProjectStore:
    root = tmp_path / "data"
    return ProjectStore(root, default_project_name="Demo Project")


@pytest.mark.asyncio
async def test_keyword_run_queue_executes_job(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    project = store.find_by_name("Demo Project")
    assert project is not None

    async def executor(job: KeywordRunJob):
        await asyncio.sleep(0.01)
        return store.update_keyword_run(
            job.project_id,
            job.id,
            status="success",
            completed_at="2025-01-17T11:00:00+00:00",
            keywords=[{"term": "beta", "count": 42}],
            stats={"token_total": 1_000_000},
        )

    queue = KeywordRunQueue(store, max_queue=4, max_concurrency=1, executor=executor)
    queue.start()
    job = await queue.enqueue(project.id, requested_by="ops@example.com")

    completed = await job.wait(timeout=1.0)
    assert completed
    assert job.status == "success"
    assert job.record.keywords[0]["term"] == "beta"
    latest = store.get_latest_keyword_run(project.id)
    assert latest is not None and latest.status == "success"

    await queue.shutdown()


@pytest.mark.asyncio
async def test_keyword_run_queue_reports_configuration_error(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    project = store.find_by_name("Demo Project")
    assert project is not None

    queue = KeywordRunQueue(store)
    queue.start()

    job = await queue.enqueue(project.id)
    completed = await job.wait(timeout=1.0)
    assert completed
    assert job.status == "error"
    assert "executor" in (job.error or "")

    await queue.shutdown()


@pytest.mark.asyncio
async def test_keyword_run_queue_limits_capacity(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    project = store.find_by_name("Demo Project")
    assert project is not None

    async def blocking_executor(job: KeywordRunJob):
        await asyncio.sleep(0.2)
        return store.update_keyword_run(job.project_id, job.id, status="success")

    queue = KeywordRunQueue(store, max_queue=1, max_concurrency=1, executor=blocking_executor)

    await queue.enqueue(project.id)

    with pytest.raises(RuntimeError):
        await queue.enqueue(project.id)

    await queue.shutdown()
