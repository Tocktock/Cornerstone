from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cornerstone.config import Settings
from cornerstone.keyword_jobs import KeywordRunExecutor, KeywordRunJob, KeywordRunQueue
from cornerstone.keyword_refresh import KeywordRunAutoRefresher
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


@pytest.mark.asyncio
async def test_keyword_auto_refresher_enqueue_and_rerun(tmp_path: Path) -> None:
    data_root = (tmp_path / "data").resolve()
    store = ProjectStore(data_root, default_project_name="Demo Project")
    project = store.find_by_name("Demo Project")
    assert project is not None

    settings = Settings(
        data_dir=str(data_root),
        default_project_name="Demo Project",
        keyword_run_sync_mode=False,
        keyword_run_auto_refresh=True,
    )

    execution_count = 0

    async def executor(job: KeywordRunJob):
        nonlocal execution_count
        execution_count += 1
        await asyncio.sleep(0.01)
        return store.update_keyword_run(
            job.project_id,
            job.id,
            status="success",
            completed_at="2025-01-17T12:00:00+00:00",
            keywords=[{"term": f"run-{execution_count}", "count": 1}],
        )

    queue = KeywordRunQueue(store, max_queue=4, max_concurrency=1, executor=executor)
    refresher = KeywordRunAutoRefresher(settings=settings, queue=queue)
    refresher.attach_loop(asyncio.get_running_loop())
    queue.start()

    refresher.mark_project_dirty(project.id)

    async def wait_for_runs(expected: int, timeout: float = 1.5) -> None:
        remaining = timeout
        interval = 0.02
        while remaining > 0:
            runs = store.list_keyword_runs(project.id)
            if len(runs) >= expected and runs[-1].status in {"success", "error"}:
                return
            await asyncio.sleep(interval)
            remaining -= interval
        raise AssertionError(f"Timed out waiting for {expected} keyword runs (have {len(store.list_keyword_runs(project.id))})")

    await wait_for_runs(1)
    assert execution_count >= 1

    # Trigger additional refreshes while the first run is in-flight or immediately after completion.
    refresher.mark_project_dirty(project.id)
    await asyncio.sleep(0.002)
    refresher.mark_project_dirty(project.id)

    await wait_for_runs(2)
    assert execution_count >= 2

    await queue.shutdown()
