from __future__ import annotations

import json
import os
import platform
import secrets
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from .models import Column, DEFAULT_BASE_URL, DEFAULT_TIMEOUT, DoctorCheck, PROJECT_ROOT_MARKERS

def _project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in PROJECT_ROOT_MARKERS):
            return candidate
    return current

def _run(command: Sequence[str], *, cwd: Path | None = None, dry_run: bool = False) -> int:
    display = " ".join(command)
    if dry_run:
        print(f"DRY RUN: {display}")
        return 0
    print(f"$ {display}")
    return subprocess.call(list(command), cwd=str(cwd) if cwd else None)

def _http_json(url: str, *, timeout: float = DEFAULT_TIMEOUT) -> Any:
    return _http_request("GET", url, timeout=timeout)

def _http_request(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    data: bytes | None = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            parsed = json.loads(raw)
            detail = parsed.get("detail", parsed)
        except json.JSONDecodeError:
            detail = raw
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc
    if not raw:
        return None
    return json.loads(raw)

def _api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"

def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))

def _value(payload: dict[str, Any], key: str, default: Any = "") -> Any:
    return payload.get(key, default)

def _truncate(value: Any, width: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"

def _print_table(rows: list[dict[str, Any]], columns: list[Column]) -> None:
    if not rows:
        print("No results.")
        return
    widths = [max(len(column.label), min(column.width, max(len(str(row.get(column.key, ""))) for row in rows))) for column in columns]
    header = "  ".join(column.label.ljust(width) for column, width in zip(columns, widths, strict=True))
    print(header)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(_truncate(row.get(column.key, ""), width).ljust(width) for column, width in zip(columns, widths, strict=True)))

def _print_payload(payload: Any, *, json_output: bool, table_rows: list[dict[str, Any]] | None = None, columns: list[Column] | None = None) -> None:
    if json_output:
        _print_json(payload)
        return
    if table_rows is not None and columns is not None:
        _print_table(table_rows, columns)
        return
    _print_json(payload)

def _print_next_action(message: str) -> None:
    print(f"Next action: {message}")

def _command_exists(command: str) -> bool:
    return shutil.which(command) is not None

def _os_family() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return system or "unknown"

def _script_ext_for_os(os_family: str) -> str:
    return ".ps1" if os_family == "windows" else ".sh"

def _recommended_shell(os_family: str) -> str:
    if os_family == "windows":
        return "PowerShell 7+ or Windows PowerShell"
    if os_family == "macos":
        return "zsh"
    if os_family == "linux":
        return "bash"

def _write_local_env(root: Path, *, force: bool = False) -> Path:
    source = root / ".env.example"
    target = root / ".env"
    if not source.exists():
        raise RuntimeError(".env.example not found")
    if target.exists() and not force:
        raise RuntimeError(".env already exists. Use --force to overwrite.")
    text = source.read_text(encoding="utf-8")
    secret = f"local-{secrets.token_urlsafe(32)}"
    replacements = {
        "PRODUCTION_MODE=true": "PRODUCTION_MODE=false",
        "PERSISTENCE_BACKEND=postgres": "PERSISTENCE_BACKEND=memory",
        "CONNECTOR_ENCRYPTION_SECRET=local-dev-only-change-me-secret": f"CONNECTOR_ENCRYPTION_SECRET={secret}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if "CONNECTOR_ENCRYPTION_SECRET=" not in text:
        text += f"\nCONNECTOR_ENCRYPTION_SECRET={secret}\n"
    target.write_text(text, encoding="utf-8")
    return target

def _chmod_scripts(root: Path) -> list[str]:
    changed: list[str] = []
    if _os_family() == "windows":
        return changed
    scripts_dir = root / "scripts"
    for script in scripts_dir.glob("*.sh"):
        current = script.stat().st_mode
        script.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        changed.append(script.as_posix())
    return changed

def _safe_setup_fixes(root: Path, *, dry_run: bool = False) -> list[str]:
    actions: list[str] = []
    env_path = root / ".env"
    reports_path = root / "reports"
    if not env_path.exists():
        actions.append("create .env from .env.example with local-safe defaults")
        if not dry_run:
            _write_local_env(root, force=False)
    else:
        actions.append(".env already exists; leave unchanged")
    actions.append("ensure reports directory exists")
    if not dry_run:
        reports_path.mkdir(exist_ok=True)
    if _os_family() != "windows":
        actions.append("mark shell scripts executable")
        if not dry_run:
            _chmod_scripts(root)
    return actions

def _setup_plan(target: str, root: Path) -> dict[str, Any]:
    detected = _os_family()
    target_os = detected if target in {"auto", "local"} else target
    script_ext = _script_ext_for_os(target_os)
    quickstart = {
        "macos": "docs/integration-starter-kit/macos-quickstart.md",
        "linux": "docs/integration-starter-kit/linux-quickstart.md",
        "windows": "docs/integration-starter-kit/windows-quickstart.md",
    }.get(target_os, "docs/integration-starter-kit/local-quickstart.md")
    setup_script = {
        "macos": "scripts/setup_local.sh",
        "linux": "scripts/setup_local.sh",
        "windows": "scripts/windows_setup.ps1",
    }.get(target_os, f"scripts/setup_local{script_ext}")
    start_script = {
        "macos": "scripts/start_local.sh",
        "linux": "scripts/start_local.sh",
        "windows": "scripts/windows_start_local.ps1",
    }.get(target_os, f"scripts/start_local{script_ext}")
    proof_script = {
        "macos": "scripts/run_live_proof.sh",
        "linux": "scripts/run_live_proof.sh",
        "windows": "scripts/windows_run_live_proof.ps1",
    }.get(target_os, f"scripts/run_live_proof{script_ext}")
    return {
        "detectedOs": detected,
        "targetOs": target_os,
        "projectRoot": str(root),
        "recommendedShell": _recommended_shell(target_os),
        "quickstart": quickstart,
        "setupScript": setup_script,
        "startScript": start_script,
        "proofScript": proof_script,
        "safeFixes": [
            "create .env if missing",
            "create reports directory",
            "chmod shell scripts on Unix-like systems",
        ],
        "manualPrerequisites": [
            "Python 3.13+",
            "Docker Desktop or Docker Engine with Compose plugin",
            "Git",
            "Notion integration token and shared page for live Notion proof",
        ],
        "nextCommands": [
            "cornerstone doctor",
            "cornerstone setup --fix",
            "cornerstone stack up --migrate",
            "cornerstone api --reload",
            "cornerstone proof run --all --continue-on-failure --markdown",
        ],
    }

def _doctor_checks(root: Path) -> list[DoctorCheck]:
    python_ok = sys.version_info >= (3, 13)
    checks = [
        DoctorCheck("platform", True, f"{_os_family()} ({platform.platform()})", required=False),
        DoctorCheck(
            name="python",
            ok=python_ok,
            detail=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        ),
        DoctorCheck("docker", _command_exists("docker"), shutil.which("docker") or "not found"),
        DoctorCheck("git", _command_exists("git"), shutil.which("git") or "not found"),
        DoctorCheck("jq", _command_exists("jq"), shutil.which("jq") or "not found", required=False),
        DoctorCheck("pyproject", (root / "pyproject.toml").exists(), str(root / "pyproject.toml")),
        DoctorCheck("alembic", (root / "alembic.ini").exists(), str(root / "alembic.ini")),
        DoctorCheck("docker_compose", (root / "docker-compose.yml").exists(), str(root / "docker-compose.yml")),
        DoctorCheck("env_file", (root / ".env").exists(), str(root / ".env"), required=False),
    ]
    if _os_family() == "windows":
        checks.append(DoctorCheck("powershell", _command_exists("pwsh") or _command_exists("powershell"), shutil.which("pwsh") or shutil.which("powershell") or "not found", required=False))
    return checks

def _rows_from_sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in payload.get("sources", []):
        rows.append(
            {
                "id": source.get("id", ""),
                "type": source.get("type", ""),
                "auth": source.get("authStatus", ""),
                "connection": source.get("connectionStatus", ""),
                "sync": source.get("syncStatus", ""),
                "freshness": source.get("freshnessState", ""),
                "artifacts": source.get("artifactCount", 0),
                "evidence": source.get("evidenceFragmentCount", 0),
                "next": source.get("nextAction", ""),
            }
        )
    return rows

def _source_columns() -> list[Column]:
    return [
        Column("id", "ID", 12),
        Column("type", "TYPE", 12),
        Column("auth", "AUTH", 14),
        Column("connection", "CONNECTION", 16),
        Column("sync", "SYNC", 14),
        Column("freshness", "FRESHNESS", 12),
        Column("artifacts", "ARTIFACTS", 9),
        Column("evidence", "EVIDENCE", 8),
        Column("next", "NEXT ACTION", 18),
    ]

def _evidence_rows(queue_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in queue_payload.get("items", []):
        evidence = item.get("evidenceFragment", {})
        artifact = item.get("artifact", {})
        source = item.get("dataSource", {})
        rows.append(
            {
                "id": evidence.get("id", ""),
                "type": evidence.get("fragmentType", ""),
                "trust": evidence.get("trustState", ""),
                "freshness": evidence.get("freshnessState", ""),
                "source": source.get("name", source.get("type", "")),
                "artifact": artifact.get("title", ""),
                "text": evidence.get("text", ""),
            }
        )
    return rows

def _evidence_columns() -> list[Column]:
    return [
        Column("id", "ID", 12),
        Column("type", "TYPE", 14),
        Column("trust", "TRUST", 11),
        Column("freshness", "FRESHNESS", 10),
        Column("source", "SOURCE", 16),
        Column("artifact", "ARTIFACT", 22),
        Column("text", "TEXT", 44),
    ]

def _concept_rows(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": concept.get("id", ""),
            "name": concept.get("name", ""),
            "status": concept.get("status", ""),
            "evidence": len(concept.get("evidenceFragmentIds", [])),
            "definition": concept.get("shortDefinition", ""),
        }
        for concept in concepts
    ]

def _concept_columns() -> list[Column]:
    return [
        Column("id", "ID", 12),
        Column("name", "NAME", 24),
        Column("status", "STATUS", 12),
        Column("evidence", "EVIDENCE", 8),
        Column("definition", "DEFINITION", 56),
    ]

def _job_rows(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": job.get("id", ""),
            "status": job.get("status", ""),
            "trigger": job.get("trigger", ""),
            "attempts": f"{job.get('attemptCount', 0)}/{job.get('maxAttempts', 0)}",
            "artifacts": job.get("artifactCreatedCount", 0),
            "evidence": job.get("evidenceCreatedCount", 0),
            "created": job.get("createdAt", ""),
        }
        for job in jobs
    ]

def _job_columns() -> list[Column]:
    return [
        Column("id", "ID", 12),
        Column("status", "STATUS", 14),
        Column("trigger", "TRIGGER", 10),
        Column("attempts", "ATTEMPTS", 8),
        Column("artifacts", "ARTIFACTS", 9),
        Column("evidence", "EVIDENCE", 8),
        Column("created", "CREATED", 24),
    ]

def _http_status(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[int, Any]:
    data: bytes | None = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        status = exc.code
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc.reason)}
    if not raw:
        return status, None
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw
