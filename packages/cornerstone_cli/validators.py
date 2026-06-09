from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{8,}"),
]


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_fixture_pack(pack_path: Path) -> tuple[dict[str, Any], list[ValidationIssue]]:
    pack = load_json(pack_path)
    issues: list[ValidationIssue] = []
    base = pack_path.parent

    if not pack.get("id"):
        issues.append(ValidationIssue("PACK_ID_MISSING", "Fixture pack id is required.", str(pack_path)))
    if not pack.get("scenario_ids"):
        issues.append(ValidationIssue("SCENARIOS_MISSING", "Fixture pack must cover at least one scenario.", str(pack_path)))
    if "negative_evidence" not in pack:
        issues.append(ValidationIssue("NEGATIVE_EVIDENCE_MISSING", "Fixture pack must declare negative evidence.", str(pack_path)))

    inputs = pack.get("inputs", [])
    if not inputs:
        issues.append(ValidationIssue("INPUTS_MISSING", "Fixture pack must declare input files.", str(pack_path)))

    for item in inputs:
        rel = item.get("path")
        input_path = base / rel if rel else base / "<missing>"
        if not rel or not input_path.exists():
            issues.append(ValidationIssue("INPUT_MISSING", "Fixture input file is missing.", str(input_path)))
            continue
        item["calculated_sha256"] = sha256_file(input_path)
        for required in ["owner_id", "namespace_id", "workspace_id"]:
            if not item.get(required):
                issues.append(ValidationIssue("SCOPE_MISSING", f"Fixture input missing {required}.", str(input_path)))

    return pack, issues


def redact_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def count_unredacted_secrets(text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in SECRET_PATTERNS)


def validate_redaction_pack(pack_path: Path) -> list[ValidationIssue]:
    pack = load_json(pack_path)
    issues: list[ValidationIssue] = []
    for item in pack.get("inputs", []):
        input_path = pack_path.parent / item["path"]
        raw = input_path.read_text()
        redacted = redact_text(raw)
        for forbidden in pack.get("forbidden_outputs", []):
            if forbidden in redacted:
                issues.append(ValidationIssue("SECRET_NOT_REDACTED", "Forbidden secret survived redaction.", str(input_path)))
        if count_unredacted_secrets(redacted) != 0:
            issues.append(ValidationIssue("SECRET_PATTERN_NOT_REDACTED", "A secret-like token survived redaction.", str(input_path)))
    return issues


def validate_prompt_injection_pack(pack_path: Path) -> list[ValidationIssue]:
    pack = load_json(pack_path)
    issues: list[ValidationIssue] = []
    negative = pack.get("negative_evidence", {})
    for key in ["tool_calls_created", "action_cards_created_from_untrusted_artifact", "external_http_calls"]:
        if negative.get(key) != 0:
            issues.append(ValidationIssue("NEGATIVE_EVIDENCE_NONZERO", f"{key} must be zero.", str(pack_path)))
    expected = pack.get("expected", {})
    if "prompt_injection_blocked" not in expected.get("required_policy_decisions", []):
        issues.append(ValidationIssue("POLICY_DECISION_MISSING", "Prompt-injection pack must require a blocked policy decision.", str(pack_path)))
    return issues


def validate_namespace_pack(pack_path: Path) -> list[ValidationIssue]:
    pack = load_json(pack_path)
    issues: list[ValidationIssue] = []
    namespaces = {item.get("namespace_id") for item in pack.get("inputs", [])}
    if len(namespaces) < 2:
        issues.append(ValidationIssue("NAMESPACE_VARIETY_MISSING", "Namespace fixture needs at least two namespaces.", str(pack_path)))
    if pack.get("negative_evidence", {}).get("cross_namespace_results") != 0:
        issues.append(ValidationIssue("CROSS_NAMESPACE_RESULTS_NONZERO", "cross_namespace_results must be zero.", str(pack_path)))
    return issues
