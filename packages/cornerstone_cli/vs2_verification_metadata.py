from __future__ import annotations

import fnmatch
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POSTGRES_IMAGE = "postgres:16-alpine"
OPA_IMAGE = "openpolicyagent/opa@sha256:dc009236137bb225a1ef09293bb32f2ee1861cc428870d297bf71412d50221c3"
PYTHON_IMAGE = "python:3.12-bookworm"
VERIFICATION_IMAGES = (
    POSTGRES_IMAGE,
    OPA_IMAGE,
    PYTHON_IMAGE,
)

DEPENDENCY_FAMILIES: dict[str, list[str]] = {
    "cli_runtime": [
        "cornerstone",
        "Makefile",
        "packages/cornerstone_cli/*.py",
        "packages/cornerstone_cli/**/*.py",
    ],
    "verification_scripts": [
        "scripts/*",
        "scripts/**/*",
    ],
    "fixtures_and_tests": [
        "fixtures/*",
        "fixtures/**/*",
        "tests/*.py",
        "tests/**/*.py",
    ],
    "runtime_topology": [
        "compose*.yml",
        "docker-compose*.yml",
        "Dockerfile*",
        "docker/*",
        "docker/**/*",
        ".docker/*",
        ".docker/**/*",
    ],
    "vs2_configuration": [
        "config/vs2/*",
        "config/vs2/**/*",
        "migrations/vs2/*",
        "migrations/vs2/**/*",
        "policies/vs2/*",
        "policies/vs2/**/*",
    ],
    "scenario_authority": [
        "docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md",
        "docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_MATRIX.csv",
        "docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md",
        "docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md",
    ],
    "reviewed_documents": [
        "README.md",
        "docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_STATE_2026-06-19.md",
        "docs/verification-reports/VS2_LOCAL_RANGE_FIRST_SLICE_REPORT_2026-06-21.md",
    ],
    "dependency_manifests": [
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "requirements*.txt",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
    ],
}

LOCAL_RANGE_FAMILIES = [
    "cli_runtime",
    "verification_scripts",
    "fixtures_and_tests",
    "runtime_topology",
    "vs2_configuration",
    "scenario_authority",
    "reviewed_documents",
    "dependency_manifests",
]

LOCAL_PROOF_INPUT_PATTERNS = [
    pattern
    for family in LOCAL_RANGE_FAMILIES
    for pattern in DEPENDENCY_FAMILIES[family]
]

FINGERPRINT_EXCLUDED_PATH_PARTS = {"__pycache__"}
FINGERPRINT_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _git_value(root: Path, args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _command_value(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or result.stderr).strip() or None


def _command_json(command: list[str]) -> Any:
    value = _command_value(command)
    if not value:
        return None
    try:
        return json.loads(value)
    except ValueError:
        return value


def _matching_files(root: Path, patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            paths.update(path for path in root.glob(pattern) if _is_fingerprint_input(path))
            continue
        path = root / pattern
        if _is_fingerprint_input(path):
            paths.add(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _is_fingerprint_input(path: Path) -> bool:
    return (
        path.is_file()
        and ".git" not in path.parts
        and not (FINGERPRINT_EXCLUDED_PATH_PARTS & set(path.parts))
        and path.suffix not in FINGERPRINT_EXCLUDED_SUFFIXES
    )


def _patterns_for_family(family: str) -> list[str]:
    if family == "vs2_local_range":
        return [pattern for name in LOCAL_RANGE_FAMILIES for pattern in DEPENDENCY_FAMILIES[name]]
    if family == "vs2_local_proof":
        return LOCAL_PROOF_INPUT_PATTERNS
    return LOCAL_PROOF_INPUT_PATTERNS


def _path_matches_patterns(path: str, patterns: list[str]) -> bool:
    return any(path == pattern or fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _git_status_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        value = line[3:].strip()
        if " -> " in value:
            paths.extend(part.strip() for part in value.split(" -> ", 1))
        else:
            paths.append(value)
    return sorted(set(path for path in paths if path))


def _runtime_fingerprint() -> dict[str, Any]:
    docker_info = _command_json(["docker", "info", "--format", "{{json .}}"])
    docker_info_summary = None
    if isinstance(docker_info, dict):
        docker_info_summary = {
            "id": docker_info.get("ID"),
            "server_version": docker_info.get("ServerVersion"),
            "operating_system": docker_info.get("OperatingSystem"),
            "default_runtime": docker_info.get("DefaultRuntime"),
            "runtimes": sorted((docker_info.get("Runtimes") or {}).keys()),
            "cgroup_driver": docker_info.get("CgroupDriver"),
            "security_options": docker_info.get("SecurityOptions"),
        }
    image_identities: dict[str, Any] = {}
    for image in VERIFICATION_IMAGES:
        inspected = _command_json(["docker", "image", "inspect", image, "--format", "{{json .}}"])
        if isinstance(inspected, dict):
            image_identities[image] = {
                "id": inspected.get("Id"),
                "repo_digests": inspected.get("RepoDigests", []),
                "repo_tags": inspected.get("RepoTags", []),
                "created": inspected.get("Created"),
            }
        else:
            image_identities[image] = {"available": False, "inspect_result": inspected}
    return {
        "schema_version": "cs.vs2_runtime_fingerprint.v1",
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "docker_version": _command_value(["docker", "--version"]),
        "docker_context": _command_value(["docker", "context", "show"]),
        "docker_version_json": _command_json(["docker", "version", "--format", "{{json .}}"]),
        "docker_info": docker_info_summary,
        "container_runtime": docker_info_summary,
        "postgres_image": POSTGRES_IMAGE,
        "opa_image": OPA_IMAGE,
        "python_image": PYTHON_IMAGE,
        "verification_images": list(VERIFICATION_IMAGES),
        "image_identities": image_identities,
    }


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_source_fingerprint(root: Path, *, family: str) -> dict[str, Any]:
    root = root.resolve()
    patterns = _patterns_for_family(family)
    files = _matching_files(root, patterns)
    digest = hashlib.sha256()
    working_tree_digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(content_hash.encode())
        digest.update(b"\0")
        working_tree_digest.update(relative.encode())
        working_tree_digest.update(b"\0")
        working_tree_digest.update(content_hash.encode())
        working_tree_digest.update(b"\0")
        entries.append({"path": relative, "sha256": content_hash, "bytes": len(content)})
    dirty_paths = sorted(path for path in _git_status_paths(root) if _path_matches_patterns(path, patterns))
    for path in dirty_paths:
        working_tree_digest.update(b"dirty\0")
        working_tree_digest.update(path.encode())
        working_tree_digest.update(b"\0")
    runtime = _runtime_fingerprint()
    dependency_manifest = [
        {"family": name, "patterns": DEPENDENCY_FAMILIES[name]}
        for name in DEPENDENCY_FAMILIES
        if any(pattern in patterns for pattern in DEPENDENCY_FAMILIES[name])
    ]
    return {
        "schema_version": "cs.vs2_source_fingerprint.v1",
        "family": family,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value(root, ["rev-parse", "HEAD"]),
        "git_tree": _git_value(root, ["rev-parse", "HEAD^{tree}"]),
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
        "working_tree_digest": working_tree_digest.hexdigest(),
        "input_digest": digest.hexdigest(),
        "input_count": len(entries),
        "inputs": entries,
        "dependency_manifest": dependency_manifest,
        "runtime": runtime,
        "environment_digest": _sha256_json(runtime),
    }


def proof_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "proof_hash"}
    return _sha256_json(body)


APPROVED_EVIDENCE_ROOTS = (
    "reports/",
    "docs/scenario-contracts/",
    "docs/verification-reports/",
    "config/vs2/",
    "policies/vs2/",
    "tmp/",
)


def _resolve_evidence_path(root: Path, value: Any, errors: list[str], *, field: str) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field}_path_invalid")
        return None
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        errors.append(f"{field}_path_unsafe:{value}")
        return None
    normalized = path.as_posix()
    if not any(normalized.startswith(prefix) for prefix in APPROVED_EVIDENCE_ROOTS):
        errors.append(f"{field}_path_unapproved_root:{value}")
        return None
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        errors.append(f"{field}_path_outside_repo:{value}")
        return None
    return resolved


def _validate_artifact_ref(root: Path, artifact: dict[str, Any], errors: list[str], *, field: str) -> None:
    path_value = artifact.get("path")
    if path_value is None:
        return
    path = _resolve_evidence_path(root, path_value, errors, field=field)
    if path is None:
        return
    if not path.exists() or not path.is_file():
        errors.append(f"{field}_missing:{path_value}")
        return
    expected = artifact.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        errors.append(f"{field}_sha256_missing:{path_value}")
        return
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        errors.append(f"{field}_sha256_mismatch:{path_value}")


def _validate_evidence_manifest(root: Path, report: dict[str, Any], errors: list[str], scenario_rows: list[dict[str, Any]]) -> None:
    manifest_value = report.get("evidence_manifest")
    if not manifest_value:
        errors.append("evidence_manifest_missing")
        return
    manifest_path = _resolve_evidence_path(root, manifest_value, errors, field="evidence_manifest")
    if manifest_path is None:
        return
    if not manifest_path.exists() or not manifest_path.is_file():
        errors.append(f"evidence_manifest_missing:{manifest_value}")
        return
    try:
        manifest = json.loads(manifest_path.read_text())
    except ValueError as error:
        errors.append(f"evidence_manifest_invalid_json:{manifest_value}:{error}")
        return
    if not isinstance(manifest, dict):
        errors.append(f"evidence_manifest_not_object:{manifest_value}")
        return
    if manifest.get("schema_version") != "cs.vs2.evidence_manifest.v1":
        errors.append("evidence_manifest_schema_version_mismatch")
    raw_artifacts = manifest.get("raw_scenario_artifacts")
    foundational_artifacts = manifest.get("foundational_artifacts")
    if not isinstance(raw_artifacts, list):
        errors.append("evidence_manifest_raw_scenario_artifacts_invalid")
        raw_artifacts = []
    if not isinstance(foundational_artifacts, list):
        errors.append("evidence_manifest_foundational_artifacts_invalid")
        foundational_artifacts = []
    raw_ids: list[str] = []
    raw_status_by_id: dict[str, str] = {}
    for index, artifact in enumerate(raw_artifacts):
        if not isinstance(artifact, dict):
            errors.append(f"evidence_manifest_raw_artifact_not_object:{index}")
            continue
        scenario_id = artifact.get("scenario_id")
        if isinstance(scenario_id, str):
            raw_ids.append(scenario_id)
            if isinstance(artifact.get("status"), str):
                raw_status_by_id[scenario_id] = artifact["status"]
        _validate_artifact_ref(root, artifact, errors, field="evidence_manifest_raw_artifact")
    duplicate_raw_ids = sorted({scenario_id for scenario_id in raw_ids if raw_ids.count(scenario_id) > 1})
    if duplicate_raw_ids:
        errors.append("evidence_manifest_duplicate_scenarios:" + ",".join(duplicate_raw_ids))
    for index, artifact in enumerate(foundational_artifacts):
        if not isinstance(artifact, dict):
            errors.append(f"evidence_manifest_foundational_artifact_not_object:{index}")
            continue
        _validate_artifact_ref(root, artifact, errors, field="evidence_manifest_foundational_artifact")
    ai_status_by_id = {
        str(row.get("scenario_id") or row.get("id")): str(row.get("status"))
        for row in scenario_rows
        if row.get("owner") != "Human"
    }
    missing_raw = sorted(set(ai_status_by_id) - set(raw_status_by_id))
    extra_raw = sorted(set(raw_status_by_id) - set(ai_status_by_id))
    mismatched_status = sorted(
        scenario_id
        for scenario_id, status in raw_status_by_id.items()
        if scenario_id in ai_status_by_id and ai_status_by_id[scenario_id] != status
    )
    if missing_raw:
        errors.append("evidence_manifest_missing_ai_scenarios:" + ",".join(missing_raw[:10]))
    if extra_raw:
        errors.append("evidence_manifest_extra_scenarios:" + ",".join(extra_raw[:10]))
    if mismatched_status:
        errors.append("evidence_manifest_status_mismatch:" + ",".join(mismatched_status[:10]))


def _validate_scenario_inventory(
    root: Path,
    report: dict[str, Any],
    errors: list[str],
    *,
    expected_scenario_ids: set[str] | None,
    expected_scenario_owners: dict[str, str] | None,
    validate_evidence: bool,
) -> list[dict[str, Any]]:
    rows = report.get("scenario_results")
    if not isinstance(rows, list):
        errors.append("scenario_results_invalid")
        return []
    seen: list[str] = []
    valid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"scenario_result_not_object:{index}")
            continue
        scenario_id = row.get("scenario_id") or row.get("id")
        if not isinstance(scenario_id, str) or not scenario_id:
            errors.append(f"scenario_result_id_missing:{index}")
            continue
        seen.append(scenario_id)
        if row.get("id") and row.get("scenario_id") and row.get("id") != row.get("scenario_id"):
            errors.append(f"scenario_result_id_mismatch:{scenario_id}")
        if row.get("status") not in {"PASS", "FAIL", "NOT_VERIFIED", "NOT_RUN", "HUMAN_REQUIRED"}:
            errors.append(f"scenario_result_status_invalid:{scenario_id}")
        expected_owner = expected_scenario_owners.get(scenario_id) if expected_scenario_owners else None
        recorded_owner = row.get("owner")
        if expected_owner is not None:
            if recorded_owner != expected_owner:
                errors.append(f"scenario_result_owner_mismatch:{scenario_id}")
            if expected_owner == "AI" and row.get("status") == "HUMAN_REQUIRED":
                errors.append(f"scenario_result_ai_human_required_invalid:{scenario_id}")
            if expected_owner == "Human" and row.get("status") != "HUMAN_REQUIRED":
                errors.append(f"scenario_result_human_status_invalid:{scenario_id}")
        evidence_paths = row.get("evidence_paths")
        evidence_hashes = row.get("evidence_hashes")
        if evidence_paths is None:
            evidence_paths = row.get("evidence", [])
        if evidence_hashes is None:
            evidence_hashes = []
        if not isinstance(evidence_paths, list):
            errors.append(f"scenario_result_evidence_paths_invalid:{scenario_id}")
            evidence_paths = []
        if not isinstance(evidence_hashes, list) or not all(isinstance(value, str) for value in evidence_hashes):
            errors.append(f"scenario_result_evidence_hashes_invalid:{scenario_id}")
            evidence_hashes = []
        if row.get("status") == "PASS":
            if not row.get("validator"):
                errors.append(f"scenario_result_pass_validator_missing:{scenario_id}")
            if not evidence_paths:
                errors.append(f"scenario_result_pass_evidence_missing:{scenario_id}")
        if validate_evidence:
            for path_value in evidence_paths:
                path = _resolve_evidence_path(root, path_value, errors, field="scenario_result_evidence")
                if path is None:
                    continue
                if not path.exists() or not path.is_file():
                    errors.append(f"scenario_result_evidence_missing:{path_value}")
                    continue
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual not in evidence_hashes:
                    errors.append(f"scenario_result_evidence_sha256_mismatch:{path_value}")
        valid_rows.append(row)
    duplicate_ids = sorted({scenario_id for scenario_id in seen if seen.count(scenario_id) > 1})
    if duplicate_ids:
        errors.append("scenario_results_duplicate_ids:" + ",".join(duplicate_ids))
    if expected_scenario_ids is not None:
        actual = set(seen)
        missing = sorted(expected_scenario_ids - actual)
        extra = sorted(actual - expected_scenario_ids)
        if missing:
            errors.append("scenario_results_missing_ids:" + ",".join(missing[:10]))
        if extra:
            errors.append("scenario_results_extra_ids:" + ",".join(extra[:10]))
    return valid_rows


def _validate_local_range_shape(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("status") == "passed":
        checks = report.get("checks")
        if not isinstance(checks, dict) or not checks:
            errors.append("checks_missing")
        elif not all(value is True for value in checks.values()):
            errors.append("checks_not_all_passed")
    profile = report.get("profile")
    if isinstance(profile, dict):
        if profile.get("cleanup_success") is False:
            errors.append("cleanup_failed")
        if profile.get("cleanup_errors"):
            errors.append("cleanup_errors_present")
        for result in profile.get("cleanup_results", []):
            if isinstance(result, dict) and result.get("mandatory") is True and result.get("exit_code") != 0:
                errors.append(f"cleanup_result_failed:{result.get('label')}")


def validate_reusable_report(
    report: Any,
    *,
    root: Path,
    family: str,
    expected_schema: str,
    require_status: str | None = None,
    expected_scenario_ids: set[str] | None = None,
    expected_scenario_owners: dict[str, str] | None = None,
    validate_evidence: bool = True,
) -> tuple[bool, list[str], dict[str, Any]]:
    current = build_source_fingerprint(root, family=family)
    errors: list[str] = []
    if not isinstance(report, dict):
        return False, ["report_not_object"], current
    if report.get("schema_version") != expected_schema:
        errors.append("schema_version_mismatch")
    if require_status is not None and report.get("status") != require_status:
        errors.append("status_mismatch")
    if "status" not in report:
        errors.append("status_missing")
    recorded = report.get("source_fingerprint")
    if not isinstance(recorded, dict):
        errors.append("source_fingerprint_missing")
    else:
        for key in ["git_commit", "git_tree", "input_digest", "environment_digest", "dirty", "working_tree_digest"]:
            if recorded.get(key) != current.get(key):
                errors.append(f"source_fingerprint_{key}_mismatch")
        recorded_dirty_paths = recorded.get("dirty_paths")
        if not isinstance(recorded_dirty_paths, list) or sorted(recorded_dirty_paths) != current.get("dirty_paths"):
            errors.append("source_fingerprint_dirty_paths_mismatch")
    recorded_hash = report.get("proof_hash")
    if not isinstance(recorded_hash, str) or not recorded_hash:
        errors.append("proof_hash_missing")
    elif recorded_hash != proof_hash(report):
        errors.append("proof_hash_mismatch")
    if family == "vs2_local_range":
        _validate_local_range_shape(report, errors)
    if family == "vs2_local_proof":
        for field in ["summary", "negative_evidence", "scenario_set"]:
            if field not in report:
                errors.append(f"{field}_missing")
        scenario_rows = _validate_scenario_inventory(
            root,
            report,
            errors,
            expected_scenario_ids=expected_scenario_ids,
            expected_scenario_owners=expected_scenario_owners,
            validate_evidence=validate_evidence,
        )
        _validate_evidence_manifest(root, report, errors, scenario_rows)
    return not errors, errors, current
