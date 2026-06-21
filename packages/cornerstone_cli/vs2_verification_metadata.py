from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOCAL_RANGE_INPUT_PATTERNS = [
    "cornerstone",
    "packages/cornerstone_cli/main.py",
    "packages/cornerstone_cli/vs2_local_range.py",
    "packages/cornerstone_cli/vs2_verification_metadata.py",
    "config/vs2/*.json",
    "migrations/vs2/*.sql",
    "policies/vs2/*.rego",
]

LOCAL_PROOF_INPUT_PATTERNS = [
    *LOCAL_RANGE_INPUT_PATTERNS,
    "packages/cornerstone_cli/scenarios.py",
    "packages/cornerstone_cli/vs2_security.py",
    "docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md",
    "docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_MATRIX.csv",
    "docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_STATE_2026-06-19.md",
]


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


def _matching_files(root: Path, patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            paths.update(path for path in root.glob(pattern) if path.is_file())
            continue
        path = root / pattern
        if path.is_file():
            paths.add(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def build_source_fingerprint(root: Path, *, family: str) -> dict[str, Any]:
    root = root.resolve()
    patterns = LOCAL_PROOF_INPUT_PATTERNS if family == "vs2_local_proof" else LOCAL_RANGE_INPUT_PATTERNS
    files = _matching_files(root, patterns)
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(content_hash.encode())
        digest.update(b"\0")
        entries.append({"path": relative, "sha256": content_hash, "bytes": len(content)})
    return {
        "schema_version": "cs.vs2_source_fingerprint.v1",
        "family": family,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value(root, ["rev-parse", "HEAD"]),
        "git_tree": _git_value(root, ["rev-parse", "HEAD^{tree}"]),
        "input_digest": digest.hexdigest(),
        "input_count": len(entries),
        "inputs": entries,
    }


def proof_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "proof_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_reusable_report(
    report: dict[str, Any],
    *,
    root: Path,
    family: str,
    expected_schema: str,
    require_status: str | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    current = build_source_fingerprint(root, family=family)
    errors: list[str] = []
    if report.get("schema_version") != expected_schema:
        errors.append("schema_version_mismatch")
    if require_status is not None and report.get("status") != require_status:
        errors.append("status_mismatch")
    recorded = report.get("source_fingerprint")
    if not isinstance(recorded, dict):
        errors.append("source_fingerprint_missing")
    else:
        for key in ["git_commit", "git_tree", "input_digest"]:
            if recorded.get(key) != current.get(key):
                errors.append(f"source_fingerprint_{key}_mismatch")
    recorded_hash = report.get("proof_hash")
    if recorded_hash is not None and recorded_hash != proof_hash(report):
        errors.append("proof_hash_mismatch")
    return not errors, errors, current
