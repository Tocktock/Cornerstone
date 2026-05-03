from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VerificationCommand:
    """One measurable command in the dependency-complete verification contract."""

    id: str
    goal: str
    command: tuple[str, ...]
    report_path: str
    required_env: tuple[str, ...] = ()
    expected_result: str = "exit_code=0"
    destructive: bool = False

    @property
    def shell_command(self) -> str:
        return " ".join(shlex.quote(part) for part in self.command)

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        payload["required_env"] = list(self.required_env)
        payload["shellCommand"] = self.shell_command
        return payload


DEPENDENCY_IMPORT_SNIPPET = (
    "import importlib.util; "
    "required=['fastapi','uvicorn','pydantic','httpx','sqlalchemy','alembic','psycopg',"
    "'cryptography','notion_client','googleapiclient','google.auth','multipart','pytest',"
    "'coverage','ruff','mypy']; "
    "missing=[name for name in required if importlib.util.find_spec(name) is None]; "
    "assert not missing, 'missing dependency modules: '+', '.join(missing); "
    "print('dependency imports ok: '+', '.join(required))"
)


def dependency_complete_command_plan(python_executable: str = "python") -> list[VerificationCommand]:
    """Return the canonical v2.0.3 dependency-complete verification command plan.

    The plan is intentionally product-behavior-neutral. It verifies packaging, dependency
    availability, lint/type/test contracts, migrations, PostgreSQL persistence, proof-loop
    coverage, and SSOT readiness without adding new ontology runtime behavior.
    """

    return [
        VerificationCommand(
            id="V203-CMD-01-python-version",
            goal="Confirm the active interpreter satisfies the Python 3.13+ release target.",
            command=(python_executable, "--version"),
            report_path="reports/v2.0.3-python-version.txt",
            expected_result="Python version is >=3.13 and <3.15.",
        ),
        VerificationCommand(
            id="V203-CMD-02-dependency-imports",
            goal="Confirm runtime and dev dependency modules are importable after editable install.",
            command=(python_executable, "-c", DEPENDENCY_IMPORT_SNIPPET),
            report_path="reports/v2.0.3-dependency-imports.txt",
            expected_result="All required runtime/dev modules import successfully.",
        ),
        VerificationCommand(
            id="V203-CMD-03-compileall",
            goal="Compile application, test, script, and migration Python files.",
            command=(python_executable, "-m", "compileall", "-q", "src", "tests", "scripts", "migrations"),
            report_path="reports/v2.0.3-compileall.txt",
        ),
        VerificationCommand(
            id="V203-CMD-04-release-candidate-check",
            goal="Run static release-candidate hygiene, version, document, and API-freeze checks.",
            command=(python_executable, "scripts/check_release_candidate.py"),
            report_path="reports/v2.0.3-release-candidate.txt",
        ),
        VerificationCommand(
            id="V203-CMD-05-ruff",
            goal="Run lint and import-order checks across application, tests, migrations, and scripts.",
            command=(python_executable, "-m", "ruff", "check", "src", "tests", "migrations", "scripts"),
            report_path="reports/v2.0.3-ruff.txt",
        ),
        VerificationCommand(
            id="V203-CMD-06-mypy",
            goal="Run strict static typing checks for application source.",
            command=(
                python_executable,
                "-m",
                "mypy",
                "src",
                "--show-error-codes",
                "--no-color-output",
                "--no-incremental",
                "--cache-dir",
                "reports/.mypy_cache_v2_0_3",
            ),
            report_path="reports/v2.0.3-mypy.txt",
        ),
        VerificationCommand(
            id="V203-CMD-07-alembic-offline-sql",
            goal="Render Alembic upgrade SQL to prove migration graph is syntactically reachable.",
            command=(python_executable, "-m", "alembic", "upgrade", "head", "--sql"),
            report_path="reports/v2.0.3-alembic-offline.sql",
        ),
        VerificationCommand(
            id="V203-CMD-08-alembic-live-upgrade",
            goal="Apply Alembic migrations to a live PostgreSQL database.",
            command=(python_executable, "-m", "alembic", "upgrade", "head"),
            report_path="reports/v2.0.3-alembic-live-upgrade.txt",
            required_env=("DATABASE_URL", "PERSISTENCE_BACKEND", "RUN_POSTGRES_TESTS"),
            destructive=True,
        ),
        VerificationCommand(
            id="V203-CMD-09-postgres-live-tests",
            goal="Run strict PostgreSQL tests with no skips when explicitly enabled.",
            command=(python_executable, "scripts/run_live_postgres_tests.py"),
            report_path="reports/v2.0.3-live-postgres.txt",
            required_env=("DATABASE_URL", "RUN_POSTGRES_TESTS"),
            destructive=True,
        ),
        VerificationCommand(
            id="V203-CMD-10-full-pytest",
            goal="Run the full test suite after dependency installation.",
            command=(python_executable, "-m", "pytest", "tests", "-q", "--color=no"),
            report_path="reports/v2.0.3-pytest-full.txt",
            required_env=("RUN_POSTGRES_TESTS",),
        ),
        VerificationCommand(
            id="V203-CMD-11-openapi-contract",
            goal="Verify OpenAPI snapshots and public contract stability.",
            command=(python_executable, "-m", "pytest", "tests/integration/test_openapi_contract.py", "-q", "--color=no"),
            report_path="reports/v2.0.3-openapi-contract.txt",
        ),
        VerificationCommand(
            id="V203-CMD-12-proof-and-readiness",
            goal="Verify ontology proof-loop and SSOT readiness tests through the API layer.",
            command=(
                python_executable,
                "-m",
                "pytest",
                "tests/integration/test_ontology_proof_api.py",
                "tests/integration/test_ontology_ssot_readiness_api.py",
                "-q",
                "--color=no",
            ),
            report_path="reports/v2.0.3-proof-readiness.txt",
        ),
        VerificationCommand(
            id="V203-CMD-13-duplicate-audit",
            goal="Run the duplicate-code boundary audit preserved from v2.0.1.",
            command=(python_executable, "-m", "pytest", "tests/unit/test_refactor_boundaries_v2_0_1.py", "-q", "--color=no"),
            report_path="reports/v2.0.3-duplicate-audit.txt",
        ),
    ]


def command_plan_to_json(commands: list[VerificationCommand]) -> str:
    return json.dumps([command.to_jsonable() for command in commands], indent=2, sort_keys=True)


def render_command_plan_markdown(commands: list[VerificationCommand]) -> str:
    lines = [
        "# v2.0.3 Dependency-Complete Verification Command Plan",
        "",
        "This report records the measurable command plan added for `v2.0.3`. It is generated by `scripts/run_dependency_complete_verification.py --plan-only` and is safe to produce in minimal environments.",
        "",
        "| ID | Goal | Command | Report | Required env | Expected result |",
        "|---|---|---|---|---|---|",
    ]
    for command in commands:
        env = ", ".join(command.required_env) if command.required_env else "none"
        lines.append(
            f"| {command.id} | {command.goal} | `{command.shell_command}` | `{command.report_path}` | {env} | {command.expected_result} |"
        )
    lines.extend(
        [
            "",
            "## Trust boundary",
            "",
            "This verification plan does not add product behavior. It checks that the existing ontology SSOT path remains installable, testable, migratable, and auditable in a dependency-complete environment.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_command_plan_reports(
    *,
    commands: list[VerificationCommand],
    reports_dir: Path,
    json_name: str = "dependency-complete-command-plan-v2.0.3.json",
    markdown_name: str = "dependency-complete-command-plan-v2.0.3.md",
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / json_name
    markdown_path = reports_dir / markdown_name
    json_path.write_text(command_plan_to_json(commands) + "\n", encoding="utf-8")
    markdown_path.write_text(render_command_plan_markdown(commands), encoding="utf-8")
    return json_path, markdown_path
