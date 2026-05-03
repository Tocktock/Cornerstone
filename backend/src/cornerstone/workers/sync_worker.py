from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from cornerstone.config import Settings
from cornerstone.observability import configure_logging, log_event
from cornerstone.persistence.database import create_persistent_store
from cornerstone.schemas import SyncSchedulerRunResult, SyncWorkerRunResult
from cornerstone.services.sync_scheduler import run_sync_scheduler
from cornerstone.services.sync_worker import run_sync_worker
from cornerstone.store import InMemoryStore


@dataclass(frozen=True)
class WorkerIterationResult:
    scheduler: SyncSchedulerRunResult | None
    worker: SyncWorkerRunResult

    def model_dump_json(self) -> str:
        payload: dict[str, Any] = {
            "scheduler": None if self.scheduler is None else self.scheduler.model_dump(mode="json", by_alias=True),
            "worker": self.worker.model_dump(mode="json", by_alias=True),
        }
        return json.dumps(payload, sort_keys=True)


def create_worker_store(settings: Settings) -> Any:
    if settings.persistence_backend == "postgres":
        return create_persistent_store(settings)
    return InMemoryStore()


async def run_worker_iteration(
    *,
    store: Any,
    settings: Settings,
    max_jobs: int,
    run_scheduler_first: bool,
    include_not_ready: bool = False,
    include_not_due: bool = False,
    worker_id: str = "sync-worker-cli",
    lease_seconds: int = 300,
) -> WorkerIterationResult:
    scheduler_result = None
    if run_scheduler_first:
        scheduler_result = run_sync_scheduler(
            store=store,
            max_schedules=max_jobs,
            include_not_due=include_not_due,
        )
    worker_result = await run_sync_worker(
        store=store,
        settings=settings,
        max_jobs=max_jobs,
        include_not_ready=include_not_ready,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
    )
    log_event(
        "sync.external_worker_iteration_completed",
        scheduledJobCount=0 if scheduler_result is None else scheduler_result.enqueued_job_count,
        processedJobCount=worker_result.processed_job_count,
        skippedJobCount=worker_result.skipped_job_count,
    )
    return WorkerIterationResult(scheduler=scheduler_result, worker=worker_result)


async def _run_cli(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    settings.assert_runtime_config_safe()
    configure_logging(settings.log_level)
    store = create_worker_store(settings)
    once = bool(getattr(args, "once", False))
    iterations = 1 if once else getattr(args, "iterations", None)
    completed = 0
    while iterations is None or completed < iterations:
        result = await run_worker_iteration(
            store=store,
            settings=settings,
            max_jobs=int(getattr(args, "max_jobs", 10)),
            run_scheduler_first=bool(getattr(args, "run_scheduler", False)),
            include_not_ready=bool(getattr(args, "include_not_ready", False)),
            include_not_due=bool(getattr(args, "include_not_due", False)),
            worker_id=str(getattr(args, "worker_id", "sync-worker-cli")),
            lease_seconds=int(getattr(args, "lease_seconds", 300)),
        )
        print(result.model_dump_json(), flush=True)
        completed += 1
        if once:
            break
        if iterations is not None and completed >= iterations:
            break
        time.sleep(float(getattr(args, "sleep_seconds", 5.0)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Cornerstone durable sync worker.")
    parser.add_argument("--once", action="store_true", help="Run one scheduler/worker iteration and exit.")
    parser.add_argument("--iterations", type=int, default=None, help="Number of iterations to run before exiting.")
    parser.add_argument("--max-jobs", type=int, default=10, help="Maximum queued jobs to process per iteration.")
    parser.add_argument("--run-scheduler", action="store_true", help="Enqueue due scheduled sync jobs before draining jobs.")
    parser.add_argument("--include-not-ready", action="store_true", help="Run retry-waiting jobs even before nextAttemptAt.")
    parser.add_argument("--include-not-due", action="store_true", help="Enqueue schedules even before nextRunAt.")
    parser.add_argument("--worker-id", default="sync-worker-cli", help="Stable worker identity used for job leases.")
    parser.add_argument("--lease-seconds", type=int, default=300, help="Seconds before a claimed job lease expires.")
    parser.add_argument("--sleep-seconds", type=float, default=5.0, help="Sleep interval between iterations.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.once and args.iterations is None:
        args.iterations = None
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
