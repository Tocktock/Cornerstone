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


@pytest.mark.asyncio
async def test_keyword_run_queue_parallel_projects_with_per_project_limit(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    project_a = store.find_by_name("Demo Project")
    assert project_a is not None
    project_b = store.create_project("Parallel Project")

    start_events: dict[str, asyncio.Event] = {}
    finish_events: dict[str, asyncio.Event] = {}

    def ensure_events(job_id: str) -> tuple[asyncio.Event, asyncio.Event]:
        start_event = start_events.setdefault(job_id, asyncio.Event())
        finish_event = finish_events.setdefault(job_id, asyncio.Event())
        return start_event, finish_event

    async def executor(job: KeywordRunJob):
        start_event, finish_event = ensure_events(job.id)
        start_event.set()
        await finish_event.wait()
        return store.update_keyword_run(job.project_id, job.id, status="success")

    queue = KeywordRunQueue(
        store,
        max_queue=6,
        max_concurrency=3,
        max_concurrency_per_project=1,
        executor=executor,
    )
    queue.start()

    job1 = await queue.enqueue(project_a.id)
    start1, finish1 = ensure_events(job1.id)
    job2 = await queue.enqueue(project_b.id)
    start2, finish2 = ensure_events(job2.id)
    job3 = await queue.enqueue(project_a.id)
    start3, finish3 = ensure_events(job3.id)

    await asyncio.sleep(0.05)
    assert start1.is_set(), "First project job should start immediately"
    assert start2.is_set(), "Second project job should run in parallel"
    assert not start3.is_set(), "Second job for same project must wait"

    finish1.set()
    await asyncio.wait_for(start3.wait(), timeout=1.0)
    assert job3.status in {"running", "success"}

    finish2.set()
    finish3.set()

    await asyncio.wait_for(job1.wait(1.0), timeout=1.0)
    await asyncio.wait_for(job2.wait(1.0), timeout=1.0)
    await asyncio.wait_for(job3.wait(1.0), timeout=1.0)

    assert job1.status == "success"
    assert job2.status == "success"
    assert job3.status == "success"

    await queue.shutdown()
