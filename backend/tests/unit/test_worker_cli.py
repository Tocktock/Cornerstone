from __future__ import annotations

import argparse
import asyncio

import pytest

from cornerstone.config import Settings
from cornerstone.store import InMemoryStore
from cornerstone.workers.sync_worker import _run_cli, build_parser, create_worker_store

pytestmark = pytest.mark.unit


def test_worker_parser_supports_once_scheduler_and_max_jobs() -> None:
    parser = build_parser()
    args = parser.parse_args(["--once", "--run-scheduler", "--max-jobs", "5", "--include-not-due"])

    assert args.once is True
    assert args.run_scheduler is True
    assert args.max_jobs == 5
    assert args.include_not_due is True
    assert args.worker_id == "sync-worker-cli"
    assert args.lease_seconds == 300


def test_create_worker_store_uses_memory_backend_by_default() -> None:
    store = create_worker_store(Settings(persistence_backend="memory"))

    assert isinstance(store, InMemoryStore)


def test_worker_cli_once_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        once=True,
        iterations=None,
        max_jobs=1,
        run_scheduler=False,
        include_not_ready=False,
        include_not_due=False,
        worker_id="test-worker",
        lease_seconds=120,
        sleep_seconds=0.0,
    )

    exit_code = asyncio.run(_run_cli(args))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"worker"' in captured.out
    assert '"processedJobCount": 0' in captured.out
