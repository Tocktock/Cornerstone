from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import asdict

from cornerstone import __version__

from ..support import (
    _api_url, _doctor_checks, _http_json, _os_family, _print_json, _print_next_action,
    _print_table, _project_root, _rows_from_sources, _run, _safe_setup_fixes,
    _setup_plan, _source_columns, _write_local_env,
)
from ..config import resolve_base_url


def command_doctor(args: argparse.Namespace) -> int:
    root = _project_root()
    checks = _doctor_checks(root)
    fixes: list[str] = []
    if getattr(args, "fix", False):
        fixes = _safe_setup_fixes(root, dry_run=getattr(args, "dry_run", False))
        checks = _doctor_checks(root)
    payload = {"projectRoot": str(root), "os": _os_family(), "checks": [asdict(check) for check in checks], "fixes": fixes}
    if args.json:
        _print_json(payload)
    else:
        print(f"Cornerstone CLI {__version__}")
        print(f"OS: {_os_family()} ({platform.platform()})")
        print(f"Project root: {root}")
        for check in checks:
            marker = "OK" if check.ok else ("WARN" if not check.required else "FAIL")
            print(f"[{marker}] {check.name}: {check.detail}")
        if fixes:
            print()
            print("Safe fixes:")
            for fix in fixes:
                print(f"- {fix}")
        failures = [check for check in checks if check.required and not check.ok]
        if failures:
            _print_next_action("Install missing required prerequisites, then rerun `cornerstone doctor`.")
    required_failures = [check for check in checks if check.required and not check.ok]
    return 1 if required_failures else 0

def command_version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0

def command_env_init(args: argparse.Namespace) -> int:
    root = _project_root()
    try:
        target = _write_local_env(root, force=args.force)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Created {target}")
    print("Review it before running live Notion proof. Do not commit .env.")
    return 0

def command_setup(args: argparse.Namespace) -> int:
    root = _project_root()
    plan = _setup_plan(args.target, root)
    if args.json:
        payload = {
            "plan": plan,
            "fixesApplied": [] if not args.fix else _safe_setup_fixes(root, dry_run=args.dry_run),
            "dryRun": args.dry_run,
        }
        _print_json(payload)
        return 0
    print(f"Cornerstone local setup ({plan['targetOs']})")
    print(f"Project root: {root}")
    print(f"Recommended shell: {plan['recommendedShell']}")
    print(f"Quickstart: {plan['quickstart']}")
    print()
    print("Prerequisites:")
    for item in plan["manualPrerequisites"]:
        print(f"- {item}")
    print()
    print("Recommended commands:")
    for command in plan["nextCommands"]:
        print(f"- {command}")
    if args.fix:
        print()
        print("Applying safe local fixes:")
        for action in _safe_setup_fixes(root, dry_run=args.dry_run):
            print(f"- {action}")
    if args.start_stack:
        print()
        code = _run(["docker", "compose", "up", "-d", "postgres"], cwd=root, dry_run=args.dry_run)
        if code != 0:
            return code
    if args.migrate:
        print()
        code = _run(["alembic", "upgrade", "head"], cwd=root, dry_run=args.dry_run)
        if code != 0:
            return code
    return 0

def command_local(args: argparse.Namespace) -> int:
    root = _project_root()
    if args.action == "reset":
        if not args.yes:
            print("ERROR: local reset is destructive. Re-run with --yes to confirm.", file=sys.stderr)
            return 1
        commands: list[list[str]] = [["docker", "compose", "down", "-v"]]
        if args.start_after:
            commands.append(["docker", "compose", "up", "-d", "postgres"])
        if args.migrate:
            commands.append(["alembic", "upgrade", "head"])
        for command in commands:
            code = _run(command, cwd=root, dry_run=args.dry_run)
            if code != 0:
                return code
        return 0
    print(f"ERROR: unknown local action {args.action}", file=sys.stderr)
    return 1

def command_stack(args: argparse.Namespace) -> int:
    root = _project_root()
    if args.action == "up":
        code = _run(["docker", "compose", "up", "-d", "postgres"], cwd=root, dry_run=args.dry_run)
        if code == 0 and args.migrate:
            code = _run(["alembic", "upgrade", "head"], cwd=root, dry_run=args.dry_run)
        return code
    if args.action == "down":
        return _run(["docker", "compose", "down"], cwd=root, dry_run=args.dry_run)
    print(f"ERROR: unknown stack action {args.action}", file=sys.stderr)
    return 1

def command_db(args: argparse.Namespace) -> int:
    root = _project_root()
    if args.action == "migrate":
        return _run(["alembic", "upgrade", "head"], cwd=root, dry_run=args.dry_run)
    if args.action == "check-extensions":
        return _run(["python", "scripts/check_postgres_extensions.py"], cwd=root, dry_run=args.dry_run)
    print(f"ERROR: unknown db action {args.action}", file=sys.stderr)
    return 1

def command_api(args: argparse.Namespace) -> int:
    root = _project_root()
    return _run(
        ["uvicorn", "cornerstone.main:app", "--host", args.host, "--port", str(args.port), *( ["--reload"] if args.reload else [] )],
        cwd=root,
        dry_run=args.dry_run,
    )

def command_worker(args: argparse.Namespace) -> int:
    root = _project_root()
    command = ["python", "scripts/run_sync_worker.py"]
    if args.once:
        command.append("--once")
    if args.run_scheduler:
        command.append("--run-scheduler")
    if args.max_jobs is not None:
        command.extend(["--max-jobs", str(args.max_jobs)])
    if args.iterations is not None:
        command.extend(["--iterations", str(args.iterations)])
    if args.sleep_seconds is not None:
        command.extend(["--sleep-seconds", str(args.sleep_seconds)])
    return _run(command, cwd=root, dry_run=args.dry_run)

def command_live(args: argparse.Namespace) -> int:
    root = _project_root()
    if args.target == "postgres":
        return _run(["python", "scripts/run_live_postgres_tests.py", "--min-passed", str(args.min_passed)], cwd=root, dry_run=args.dry_run)
    if args.target == "notion":
        return _run(["python", "scripts/run_live_notion_e2e.py"], cwd=root, dry_run=args.dry_run)
    if args.target == "google-drive":
        return _run(["python", "scripts/run_live_google_drive_e2e.py"], cwd=root, dry_run=args.dry_run)
    print(f"ERROR: unknown live target {args.target}", file=sys.stderr)
    return 1

def command_status(args: argparse.Namespace) -> int:
    base_url = resolve_base_url(args.base_url).rstrip("/")
    health = _http_json(f"{base_url}/healthz", timeout=args.timeout)
    sources = _http_json(f"{base_url}/v1/sources", timeout=args.timeout)
    payload = {"health": health, "sources": sources}
    if args.json:
        _print_json(payload)
        return 0
    print(f"Cornerstone API: {health.get('status')} — version {health.get('version')}")
    print(sources.get("message", ""))
    print()
    _print_table(_rows_from_sources(sources), _source_columns())
    return 0
