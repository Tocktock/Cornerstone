#!/usr/bin/env python3
"""Static release-candidate readiness checks.

This script intentionally avoids importing FastAPI, SQLAlchemy, Pydantic, or other
runtime dependencies so it can run even in minimal packaging environments.

v1.0.0 note:
- The local checkout may contain generated runtime artifacts after the documented
  verification workflow runs (for example .venv, pytest cache, mypy cache,
  Ruff cache, coverage output, and __pycache__). Those are allowed locally.
- Sensitive files and build/package artifacts are still rejected.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "1.0.0"
RC_TAG = "v1.0.0"

REQUIRED_DOCS = [
    "docs/27-backend-release-candidate-v0.13.0.md",
    "docs/28-backend-v1.0.0-rc.1.md",
    "docs/29-backend-v1.0.0.md",
    "docs/release/backend-operator-runbook.md",
    "docs/release/backend-release-checklist.md",
    "docs/release/known-limitations.md",
    "docs/release/production-deployment-checklist.md",
    "docs/release/secrets-and-credential-handling.md",
    "docs/release/api-freeze-review.md",
    "docs/release/live-proof-artifact-template.md",
    "docs/release/v1.0.0-readiness.md",
    "docs/release/v1.0.0-release-notes.md",
    "docs/release/v1.0.0-rc.1-verification-checklist.md",
    "docs/release/v1.0.0-rc.1-human-signoff.md",
    "docs/live-proof-records/2026-04-27-change-log.md",
    "docs/live-proof-records/2026-04-28-v0.13.1-blocker-fix.md",
]

LOCAL_RUNTIME_ALLOWLIST_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
}
LOCAL_RUNTIME_ALLOWLIST_FILES = {".coverage"}

FORBIDDEN_FILE_NAMES = {".env"}
FORBIDDEN_FILE_SUFFIXES_OUTSIDE_LOCAL_RUNTIME = {".pyc", ".pyo"}

REQUIRED_README_PHRASES = [
    "v1.0.0",
    "Backend MVP release",
    "grounded_context_task_success_rate",
    "docs/release/backend-operator-runbook.md",
    "docs/release/backend-release-checklist.md",
    "docs/release/known-limitations.md",
]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _has_allowed_runtime_part(path: Path) -> bool:
    return any(part in LOCAL_RUNTIME_ALLOWLIST_PARTS for part in path.parts)


def check_versions(errors: list[str]) -> None:
    pyproject = _read("pyproject.toml")
    init_py = _read("src/cornerstone/__init__.py")
    if f'version = "{EXPECTED_VERSION}"' not in pyproject:
        errors.append("pyproject.toml version does not match expected release-candidate base version")
    if f'__version__ = "{EXPECTED_VERSION}"' not in init_py:
        errors.append("src/cornerstone/__init__.py version does not match expected release-candidate base version")


def check_required_docs(errors: list[str]) -> None:
    for doc in REQUIRED_DOCS:
        path = ROOT / doc
        if not path.exists():
            errors.append(f"missing required release doc: {doc}")
        elif path.stat().st_size < 200:
            errors.append(f"release doc is unexpectedly small: {doc}")


def check_readme(errors: list[str]) -> None:
    readme = _read("README.md")
    for phrase in REQUIRED_README_PHRASES:
        if phrase not in readme:
            errors.append(f"README.md missing required phrase: {phrase}")


def check_api_freeze(errors: list[str]) -> None:
    api_freeze = _read("docs/release/api-freeze-review.md")
    required = [
        "POST /v1/sources/{sourceId}/oauth/complete",
        "POST /v1/sources/{sourceId}/sync",
        "POST /v1/manual-sources/{notionSourceId}/sync",
        "GET  /v1/context/query",
        "POST /v1/evaluations/tasks/{taskId}/run",
    ]
    for phrase in required:
        if phrase not in api_freeze:
            errors.append(f"API freeze review missing: {phrase}")


def check_package_hygiene(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        is_allowed_runtime = _has_allowed_runtime_part(path) or path.name in LOCAL_RUNTIME_ALLOWLIST_FILES
        if path.is_file():
            if path.name in FORBIDDEN_FILE_NAMES:
                errors.append(f"forbidden local secret/config file present: {rel}")
            if path.suffix in FORBIDDEN_FILE_SUFFIXES_OUTSIDE_LOCAL_RUNTIME and not is_allowed_runtime:
                errors.append(f"forbidden compiled Python file outside local runtime cache: {rel}")


def check_live_proof_record(errors: list[str]) -> None:
    proof = _read("docs/live-proof-records/2026-04-28-v0.13.1-blocker-fix.md")
    required = [
        "230 passed",
        "Live PostgreSQL verification passed: 5 tests, 0 skipped.",
        "runnerArtifactCount: 1",
        "runnerEvidenceFragmentCount: 5",
        "groundedContextTaskSuccessRate: 1.0",
        "direct_notion_source_409",
        "fake_oauth_completion_404",
        "legacy_source_sync_404",
        "manual_sync_on_notion_409",
        "weak_evaluation_task_422",
    ]
    for phrase in required:
        if phrase not in proof:
            errors.append(f"v0.13.1 proof record missing: {phrase}")


def check_rc_docs(errors: list[str]) -> None:
    rc_doc = _read("docs/28-backend-v1.0.0-rc.1.md")
    required = [
        "v1.0.0-rc.1",
        "same verified commit as v0.13.1",
        "No new backend feature work",
    ]
    for phrase in required:
        if phrase not in rc_doc:
            errors.append(f"RC-1 doc missing: {phrase}")


def main() -> int:
    errors: list[str] = []
    check_versions(errors)
    check_required_docs(errors)
    check_readme(errors)
    check_api_freeze(errors)
    check_live_proof_record(errors)
    check_rc_docs(errors)
    check_package_hygiene(errors)

    if errors:
        print("release-candidate check: failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"release-candidate check: passed for v{EXPECTED_VERSION}")
    print(f"required_docs={len(REQUIRED_DOCS)}")
    print("package_hygiene=passed")
    print("api_freeze_review=present")
    print("live_proof_record=present")
    print("rc_tag_plan=present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
