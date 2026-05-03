#!/usr/bin/env python3
"""Run or render the v2.0.3 dependency-complete verification plan.

The default mode is plan-only so the script is safe in incomplete sandboxes. CI and
local dependency-complete environments should pass `--strict --confirm-live-db`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cornerstone.verification.dependency_complete import (  # noqa: E402
    VerificationCommand,
    dependency_complete_command_plan,
    render_command_plan_markdown,
    write_command_plan_reports,
)


@dataclass(frozen=True)
class CommandResult:
    command: VerificationCommand
    exit_code: int
    report_path: Path


class VerificationConfigurationError(RuntimeError):
    """Raised when a strict run is not explicitly configured."""


def _validate_python_version() -> None:
    version = sys.version_info
    if version < (3, 13) or version >= (3, 15):
        raise VerificationConfigurationError(
            f"Cornerstone v2.0.3 verification requires Python >=3.13,<3.15; got {sys.version.split()[0]}."
        )


def _validate_live_db_confirmation(args: argparse.Namespace, commands: list[VerificationCommand]) -> None:
    if not args.strict:
        return
    if any(command.destructive for command in commands) and not args.confirm_live_db:
        raise VerificationConfigurationError(
            "Strict dependency-complete verification includes live PostgreSQL migration/test commands. "
            "Pass --confirm-live-db after pointing DATABASE_URL at a disposable verification database."
        )


def _validate_required_env(commands: list[VerificationCommand], env: dict[str, str]) -> None:
    missing: list[str] = []
    for command in commands:
        for name in command.required_env:
            if not env.get(name):
                missing.append(f"{command.id}:{name}")
    if missing:
        joined = ", ".join(missing)
        raise VerificationConfigurationError(f"Missing required environment for strict verification: {joined}")


def _write_plan(reports_dir: Path, commands: list[VerificationCommand]) -> None:
    json_path, markdown_path = write_command_plan_reports(commands=commands, reports_dir=reports_dir)
    print(f"wrote {json_path.relative_to(ROOT)}")
    print(f"wrote {markdown_path.relative_to(ROOT)}")


def _run_command(command: VerificationCommand, *, env: dict[str, str]) -> CommandResult:
    report_path = ROOT / command.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command.command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    header = [
        f"id={command.id}",
        f"goal={command.goal}",
        f"command={command.shell_command}",
        f"exit_code={completed.returncode}",
        "",
    ]
    report_path.write_text("\n".join(header) + completed.stdout, encoding="utf-8")
    return CommandResult(command=command, exit_code=completed.returncode, report_path=report_path)


def _render_summary(results: list[CommandResult]) -> str:
    lines = [
        "# v2.0.3 Dependency-Complete Verification Run Summary",
        "",
        "| ID | Status | Report |",
        "|---|---:|---|",
    ]
    for result in results:
        status = "passed" if result.exit_code == 0 else f"failed:{result.exit_code}"
        lines.append(
            f"| {result.command.id} | {status} | `{result.report_path.relative_to(ROOT).as_posix()}` |"
        )
    passed = sum(1 for result in results if result.exit_code == 0)
    failed = len(results) - passed
    lines.extend(["", f"passed={passed}", f"failed={failed}"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or render v2.0.3 dependency-complete verification.")
    parser.add_argument("--strict", action="store_true", help="Run the verification commands instead of only writing the plan.")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the command plan and exit without executing commands. This is the default unless --strict is set.",
    )
    parser.add_argument(
        "--confirm-live-db",
        action="store_true",
        help="Required with --strict because live PostgreSQL migrations/tests can mutate the configured database.",
    )
    parser.add_argument("--reports-dir", default="reports", help="Directory for plan and run reports.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to place in generated commands.")
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the rendered markdown command plan to stdout.",
    )
    args = parser.parse_args()

    reports_dir = (ROOT / args.reports_dir).resolve()
    commands = dependency_complete_command_plan(args.python)
    _validate_python_version()
    _write_plan(reports_dir, commands)

    if args.print_plan:
        print(render_command_plan_markdown(commands))

    if args.plan_only or not args.strict:
        return 0

    try:
        _validate_live_db_confirmation(args, commands)
        env = os.environ.copy()
        env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        _validate_required_env(commands, env)
    except VerificationConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    results: list[CommandResult] = []
    for command in commands:
        result = _run_command(command, env=env)
        results.append(result)
        print(f"{command.id}: exit_code={result.exit_code} report={result.report_path.relative_to(ROOT)}")
        if result.exit_code != 0:
            break

    summary_path = reports_dir / "dependency-complete-run-summary-v2.0.3.md"
    summary_path.write_text(_render_summary(results), encoding="utf-8")
    return 0 if all(result.exit_code == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
