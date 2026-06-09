from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import csv
from pathlib import Path
from time import perf_counter
from typing import Any

from cornerstone_cli.local_test import LocalTestProvider
from cornerstone_cli.validators import (
    ValidationIssue,
    count_unredacted_secrets,
    load_json,
    redact_text,
    validate_fixture_pack,
    validate_namespace_pack,
    validate_prompt_injection_pack,
    validate_redaction_pack,
)


FULL_EXPECTED = 206
FULL_MUST_PASS = 184
FULL_REGRESSION = 22
VS0_EXPECTED = 58
VS0_MUST_PASS = 52
VS0_REGRESSION = 6


FULL_ROW = re.compile(
    r"^\| (?P<id>CS-[A-Z]+-\d{3}) \| (?P<type>MUST_PASS|REGRESSION_GUARD) \| (?P<title>[^|]+) \| (?P<section>[^|]+) \|$"
)
VS0_ROW = re.compile(
    r"^\| (?P<id>CS-[A-Z]+-\d{3}) \| (?P<type>MUST_PASS|REGRESSION_GUARD) \| (?P<expected>[^|]+) \| (?P<verification>[^|]+) \|$"
)


def _trim(value: str) -> str:
    return value.strip().replace("  ", " ")


def load_full_scenarios(root: Path) -> list[dict[str, Any]]:
    matrix = root / "docs/scenario-contracts/SCENARIO_MATRIX_FULL.md"
    scenarios: list[dict[str, Any]] = []
    for line in matrix.read_text().splitlines():
        match = FULL_ROW.match(line)
        if not match:
            continue
        scenarios.append(
            {
                "id": match.group("id"),
                "type": match.group("type"),
                "title": _trim(match.group("title")),
                "section": _trim(match.group("section")),
                "owner": "AI",
            }
        )
    return scenarios


def load_vs0_scenarios(root: Path) -> list[dict[str, Any]]:
    contract = root / "docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md"
    titles = {row["id"]: row["title"] for row in load_full_scenarios(root)}
    scenarios: list[dict[str, Any]] = []
    for line in contract.read_text().splitlines():
        match = VS0_ROW.match(line)
        if not match:
            continue
        scenario_id = match.group("id")
        scenarios.append(
            {
                "id": scenario_id,
                "type": match.group("type"),
                "title": titles.get(scenario_id, scenario_id),
                "expected_result": _trim(match.group("expected")),
                "verification_method": _trim(match.group("verification")),
                "owner": "AI",
            }
        )
    return scenarios


def list_scenarios(root: Path, scenario_set: str) -> list[dict[str, Any]]:
    if scenario_set == "vs0":
        return load_vs0_scenarios(root)
    return load_full_scenarios(root)


def load_verification_matrix(root: Path) -> list[dict[str, str]]:
    matrix = root / "docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv"
    if not matrix.exists():
        return []
    with matrix.open(newline="") as file:
        return list(csv.DictReader(file))


def _type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"MUST_PASS": 0, "REGRESSION_GUARD": 0}
    for row in rows:
        counts[row["type"]] = counts.get(row["type"], 0) + 1
    return counts


def coverage_report(root: Path) -> dict[str, Any]:
    full = load_full_scenarios(root)
    vs0 = load_vs0_scenarios(root)
    full_ids = {row["id"] for row in full}
    vs0_ids = {row["id"] for row in vs0}
    matrix = load_verification_matrix(root)
    matrix_ids = {row["scenario_id"] for row in matrix}
    full_counts = _type_counts(full)
    vs0_counts = _type_counts(vs0)
    missing: list[str] = []

    if len(full) != FULL_EXPECTED:
        missing.append(f"full_count:{len(full)}")
    if full_counts.get("MUST_PASS") != FULL_MUST_PASS:
        missing.append(f"full_must_pass:{full_counts.get('MUST_PASS')}")
    if full_counts.get("REGRESSION_GUARD") != FULL_REGRESSION:
        missing.append(f"full_regression:{full_counts.get('REGRESSION_GUARD')}")
    if len(vs0) != VS0_EXPECTED:
        missing.append(f"vs0_count:{len(vs0)}")
    if vs0_counts.get("MUST_PASS") != VS0_MUST_PASS:
        missing.append(f"vs0_must_pass:{vs0_counts.get('MUST_PASS')}")
    if vs0_counts.get("REGRESSION_GUARD") != VS0_REGRESSION:
        missing.append(f"vs0_regression:{vs0_counts.get('REGRESSION_GUARD')}")

    missing_vs0_from_full = sorted(vs0_ids - full_ids)
    missing.extend(f"vs0_not_in_full:{scenario_id}" for scenario_id in missing_vs0_from_full)
    if len(matrix) != FULL_EXPECTED:
        missing.append(f"verification_matrix_count:{len(matrix)}")
    missing_matrix = sorted(full_ids - matrix_ids)
    missing.extend(f"verification_matrix_missing:{scenario_id}" for scenario_id in missing_matrix)

    return {
        "ok": not missing,
        "missing": missing,
        "full": {"count": len(full), "type_counts": full_counts},
        "vs0": {"count": len(vs0), "type_counts": vs0_counts},
        "verification_matrix": {"count": len(matrix), "path": "docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv"},
    }


def _run_script(root: Path, script: str) -> dict[str, Any]:
    result = subprocess.run(
        ["sh", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": f"sh {script}",
        "exit_code": result.returncode,
        "stdout": result.stdout.strip().splitlines()[-3:],
        "stderr": result.stderr.strip().splitlines()[-3:],
    }


def _row(
    scenario_id: str,
    scenario_type: str,
    status: str,
    evidence: list[str],
    notes: str,
    owner: str = "AI",
) -> dict[str, Any]:
    return {
        "id": scenario_id,
        "type": scenario_type,
        "status": status,
        "evidence": evidence,
        "notes": notes,
        "owner": owner,
    }


def _issue_payload(issue: ValidationIssue) -> dict[str, str]:
    return {"code": issue.code, "message": issue.message, "path": issue.path}


def _fixture_manifest(root: Path, corpus: str) -> tuple[Path, dict[str, Any], list[ValidationIssue]]:
    corpus_path = (root / corpus).resolve()
    manifest_path = corpus_path / "manifest.json"
    issues: list[ValidationIssue] = []
    if not manifest_path.exists():
        issues.append(ValidationIssue("MANIFEST_MISSING", "Fixture corpus manifest is missing.", str(manifest_path)))
        return corpus_path, {}, issues

    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError) as error:
        issues.append(ValidationIssue("MANIFEST_INVALID_JSON", str(error), str(manifest_path)))
        return corpus_path, {}, issues

    if manifest.get("schema_version") != "cs.fixture_manifest.v0":
        issues.append(ValidationIssue("MANIFEST_SCHEMA_UNSUPPORTED", "Fixture manifest schema must be cs.fixture_manifest.v0.", str(manifest_path)))
    if manifest.get("fixture_set") != "vs0":
        issues.append(ValidationIssue("MANIFEST_SET_UNSUPPORTED", "Fixture manifest fixture_set must be vs0.", str(manifest_path)))
    if manifest.get("model_provider") != "local_test":
        issues.append(ValidationIssue("MANIFEST_PROVIDER_UNSUPPORTED", "Fixture manifest must use local_test for deterministic baseline.", str(manifest_path)))
    if not manifest.get("packs"):
        issues.append(ValidationIssue("MANIFEST_PACKS_MISSING", "Fixture manifest must list packs.", str(manifest_path)))
    return corpus_path, manifest, issues


def _provider_report(corpus_text: str, model_provider: str) -> dict[str, Any]:
    supported = model_provider == "local_test"
    report: dict[str, Any] = {
        "name": model_provider,
        "supported": supported,
        "deterministic": False,
        "uses_credentials": False,
        "external_calls": 0,
        "pass_judge": False,
    }
    if not supported:
        return report

    provider = LocalTestProvider()
    first = provider.brief_for(corpus_text)
    second = provider.brief_for(corpus_text)
    report.update(
        {
            "model": provider.model,
            "deterministic": first == second,
            "sample_evidence_terms": first["evidence_terms"],
        }
    )
    return report


def _add_negative_evidence(totals: dict[str, int], negative: dict[str, Any]) -> None:
    for key, value in negative.items():
        if isinstance(value, int):
            totals[key] = totals.get(key, 0) + value


def verify_vs0_fixtures(root: Path, corpus: str = "fixtures/vs0", model_provider: str = "local_test") -> dict[str, Any]:
    corpus_path, manifest, manifest_issues = _fixture_manifest(root, corpus)
    issues = list(manifest_issues)
    pack_reports: list[dict[str, Any]] = []
    covered_scenarios: set[str] = set()
    corpus_text_parts: list[str] = []
    pack_issue_count = 0
    redaction_issue_count = 0
    prompt_injection_issue_count = 0
    namespace_issue_count = 0
    negative_evidence: dict[str, int] = {
        "unredacted_secret_occurrences": 0,
        "tool_calls_created": 0,
        "action_cards_created_from_untrusted_artifact": 0,
        "external_http_calls": 0,
        "cross_namespace_results": 0,
    }

    for pack_ref in manifest.get("packs", []):
        rel_path = pack_ref.get("path")
        pack_path = corpus_path / rel_path if rel_path else corpus_path / "<missing>"
        if not rel_path or not pack_path.exists():
            issue = ValidationIssue("PACK_MISSING", "Fixture pack is missing.", str(pack_path))
            issues.append(issue)
            pack_reports.append({"id": pack_ref.get("id"), "path": str(pack_path), "issues": [_issue_payload(issue)]})
            continue

        pack, pack_issues = validate_fixture_pack(pack_path)
        redaction_issues: list[ValidationIssue] = []
        prompt_issues: list[ValidationIssue] = []
        namespace_issues: list[ValidationIssue] = []

        classes = set(pack.get("classes", []))
        if "redaction" in classes:
            redaction_issues = validate_redaction_pack(pack_path)
        if "prompt_injection" in classes:
            prompt_issues = validate_prompt_injection_pack(pack_path)
        if "namespace" in classes:
            namespace_issues = validate_namespace_pack(pack_path)

        for item in pack.get("inputs", []):
            input_path = pack_path.parent / item.get("path", "")
            if input_path.exists():
                raw = input_path.read_text()
                corpus_text_parts.append(raw)
                if "redaction" in classes:
                    negative_evidence["unredacted_secret_occurrences"] += count_unredacted_secrets(redact_text(raw))

        _add_negative_evidence(negative_evidence, pack.get("negative_evidence", {}))
        covered_scenarios.update(pack.get("scenario_ids", []))
        issues.extend(pack_issues)
        issues.extend(redaction_issues)
        issues.extend(prompt_issues)
        issues.extend(namespace_issues)
        pack_issue_count += len(pack_issues)
        redaction_issue_count += len(redaction_issues)
        prompt_injection_issue_count += len(prompt_issues)
        namespace_issue_count += len(namespace_issues)
        pack_reports.append(
            {
                "id": pack.get("id"),
                "path": str(pack_path.relative_to(root)),
                "scenario_ids": pack.get("scenario_ids", []),
                "input_count": len(pack.get("inputs", [])),
                "negative_evidence": pack.get("negative_evidence", {}),
                "issues": [_issue_payload(issue) for issue in pack_issues + redaction_issues + prompt_issues + namespace_issues],
            }
        )

    provider = _provider_report("\n".join(corpus_text_parts), model_provider)
    if not provider["supported"]:
        issues.append(ValidationIssue("PROVIDER_UNSUPPORTED", "Only local_test is supported for deterministic fixture verification.", model_provider))
    if provider["supported"] and not provider["deterministic"]:
        issues.append(ValidationIssue("PROVIDER_NONDETERMINISTIC", "local_test provider returned different outputs for the same input.", model_provider))

    expected_packs = {
        "pack_01_artifact_basic",
        "pack_02_dedup_versioning",
        "pack_03_unknown_and_failed_extraction",
        "pack_08_namespace_isolation",
        "pack_09_redaction_secrets",
        "pack_10_prompt_injection",
    }
    present_packs = {str(report.get("id")) for report in pack_reports}
    missing_expected_packs = sorted(expected_packs - present_packs)
    for pack_id in missing_expected_packs:
        issues.append(ValidationIssue("EXPECTED_PACK_MISSING", "Expected VS-0 fixture pack is missing.", pack_id))

    product_feature_dirs = [path for path in ["apps/web", "services/api", "services/worker"] if (root / path).exists()]
    scenario_ids = sorted(covered_scenarios)
    manifest_ok = not manifest_issues and not missing_expected_packs
    provider_ok = provider["supported"] and provider["deterministic"] and not provider["uses_credentials"] and provider["external_calls"] == 0
    negative_ok = all(value == 0 for value in negative_evidence.values())
    rows = [
        _row(
            "VS0-FIX-001",
            "MUST_PASS",
            "PASS" if manifest_ok else "FAIL",
            ["fixtures/vs0/manifest.json"],
            "Fixture corpus manifest is present, versioned, deterministic, and references expected VS-0 packs.",
        ),
        _row(
            "VS0-FIX-002",
            "MUST_PASS",
            "PASS" if provider_ok else "FAIL",
            ["packages/cornerstone_cli/local_test.py"],
            "local_test provider is deterministic, credential-free, and not a PASS judge.",
        ),
        _row(
            "VS0-FIX-003",
            "MUST_PASS",
            "PASS" if pack_issue_count == 0 and len(pack_reports) >= 4 else "FAIL",
            ["packages/cornerstone_cli/validators.py", "fixtures/vs0/packs/*/pack.json"],
            "Fixture packs declare scoped inputs, scenario coverage, and negative evidence.",
        ),
        _row(
            "VS0-FIX-004",
            "MUST_PASS",
            "PASS" if redaction_issue_count == 0 and negative_evidence["unredacted_secret_occurrences"] == 0 else "FAIL",
            ["fixtures/vs0/packs/09_redaction_secrets/pack.json", "packages/cornerstone_cli/validators.py"],
            "Fake-secret fixture validates deterministic generated-output redaction.",
        ),
        _row(
            "VS0-FIX-005",
            "MUST_PASS",
            "PASS" if prompt_injection_issue_count == 0 and negative_ok else "FAIL",
            ["fixtures/vs0/packs/10_prompt_injection/pack.json", "packages/cornerstone_cli/validators.py"],
            "Prompt-injection fixture records zero tool calls, zero action cards, and zero egress.",
        ),
        _row(
            "VS0-FIX-006",
            "MUST_PASS",
            "PASS" if namespace_issue_count == 0 and negative_evidence["cross_namespace_results"] == 0 else "FAIL",
            ["fixtures/vs0/packs/08_namespace_isolation/pack.json", "packages/cornerstone_cli/validators.py"],
            "Namespace fixture records scoped inputs and zero cross-namespace results.",
        ),
        _row(
            "VS0-FIX-R01",
            "REGRESSION_GUARD",
            "PASS" if not product_feature_dirs else "FAIL",
            ["cornerstone scenario verify vs0-fixtures --json", "repo path review"],
            "Fixture verification remains a verification-plane report and does not create product runtime feature dirs.",
        ),
    ]

    blocking = [row for row in rows if row["status"] != "PASS" and row["owner"] != "Human"]
    return {
        "status": "success" if not blocking else "failed",
        "scenario_set": "vs0-fixtures",
        "corpus": str(corpus_path.relative_to(root) if corpus_path.is_relative_to(root) else corpus_path),
        "model_provider": model_provider,
        "summary": {
            "scenario_count": len(rows),
            "pass": len([row for row in rows if row["status"] == "PASS"]),
            "blocking": len(blocking),
            "product_feature_claims": "NOT_VERIFIED",
            "referenced_product_scenario_count": len(scenario_ids),
        },
        "fixture_manifest": str((corpus_path / "manifest.json").relative_to(root) if (corpus_path / "manifest.json").is_relative_to(root) else corpus_path / "manifest.json"),
        "provider": provider,
        "negative_evidence": negative_evidence,
        "packs": pack_reports,
        "issues": [_issue_payload(issue) for issue in issues],
        "referenced_product_scenarios": [
            {
                "id": scenario_id,
                "status": "NOT_VERIFIED",
                "reason": "Fixture and validator infrastructure exists, but product runtime behavior is not implemented in this batch.",
            }
            for scenario_id in scenario_ids
        ],
        "scenario_results": rows,
        "human_required": [
            {
                "id": "H-FIXTURE-001",
                "why_ai_cannot_verify": "Optional Ollama semantic smoke coverage depends on local model availability.",
                "required_human_action": "Confirm the pinned local Ollama model if semantic smoke tests are requested.",
                "expected_evidence": "Model name, digest, and smoke transcript.",
                "release_impact": "Does not block deterministic fixture validation.",
            }
        ],
    }


def _run_cli_json(root: Path, args: list[str]) -> dict[str, Any]:
    command = [str(root / "cornerstone"), *args]
    result = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_json: dict[str, Any] | None = None
    json_error: str | None = None
    try:
        stdout_json = json.loads(result.stdout)
    except ValueError as error:
        json_error = str(error)
    return {
        "schema_version": "cs.cli_transcript.v0",
        "command": ["cornerstone", *args],
        "exit_code": result.returncode,
        "stdout_json": stdout_json,
        "stderr_redacted": redact_text(result.stderr),
        "json_error": json_error,
    }


def _payload(transcript: dict[str, Any]) -> dict[str, Any]:
    return transcript.get("stdout_json") or {}


def _artifact(transcript: dict[str, Any]) -> dict[str, Any]:
    payload = _payload(transcript)
    artifact = payload.get("artifact")
    return artifact if isinstance(artifact, dict) else {}


def _exit_ok(transcript: dict[str, Any]) -> bool:
    return transcript.get("exit_code") == 0 and isinstance(transcript.get("stdout_json"), dict)


SCOPE_FIELDS = ["tenant_id", "owner_id", "namespace_id", "workspace_id"]
OWNERLESS_SCOPE_VALUES = {"", "global", "ownerless", "none", "null"}


def _scope_complete(scope: Any) -> bool:
    if not isinstance(scope, dict):
        return False
    for field in SCOPE_FIELDS:
        value = scope.get(field)
        if not isinstance(value, str) or value.strip().lower() in OWNERLESS_SCOPE_VALUES:
            return False
    return True


def _audit_events(root: Path, state_rel: str) -> list[dict[str, Any]]:
    audit_path = root / state_rel / "audit" / "events.jsonl"
    if not audit_path.exists():
        return []
    events = []
    for line in audit_path.read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _event_scope(event: dict[str, Any]) -> dict[str, Any]:
    return {field: event.get(field) for field in SCOPE_FIELDS}


def _scope_denied(transcript: dict[str, Any]) -> bool:
    payload = _payload(transcript)
    errors = payload.get("errors", [])
    return (
        transcript.get("exit_code") == 6
        and isinstance(errors, list)
        and any(error.get("code") == "CS_SCOPE_DENIED" for error in errors if isinstance(error, dict))
    )


def _scenario_state_rel(name: str) -> str:
    return f"tmp/scenario/{name}-{os.getpid()}"


def verify_vs0_artifacts(root: Path) -> dict[str, Any]:
    state_rel = _scenario_state_rel("vs0-artifacts")
    state_path = root / state_rel
    if state_path.exists():
        shutil.rmtree(state_path)

    basic_path = "fixtures/vs0/packs/01_artifact_basic/input.txt"
    dedup_v1_path = "fixtures/vs0/packs/02_dedup_versioning/input_v1.txt"
    dedup_v2_path = "fixtures/vs0/packs/02_dedup_versioning/input_v2.txt"
    fail_path = "fixtures/vs0/packs/03_unknown_and_failed_extraction/fail.txt"
    unknown_path = "fixtures/vs0/packs/03_unknown_and_failed_extraction/unknown.bin"

    transcripts: dict[str, dict[str, Any]] = {}
    transcripts["ingest_basic"] = _run_cli_json(root, ["artifact", "ingest", basic_path, "--state-dir", state_rel, "--json"])
    basic_artifact = _artifact(transcripts["ingest_basic"])
    basic_id = basic_artifact.get("artifact_id", "")
    transcripts["show_basic"] = _run_cli_json(root, ["artifact", "show", basic_id, "--state-dir", state_rel, "--json"]) if basic_id else {}

    transcripts["ingest_fail"] = _run_cli_json(root, ["artifact", "ingest", fail_path, "--state-dir", state_rel, "--derived-mode", "fail", "--json"])
    fail_artifact = _artifact(transcripts["ingest_fail"])
    fail_id = fail_artifact.get("artifact_id", "")
    transcripts["show_fail"] = _run_cli_json(root, ["artifact", "show", fail_id, "--state-dir", state_rel, "--json"]) if fail_id else {}

    transcripts["ingest_unknown"] = _run_cli_json(
        root,
        [
            "artifact",
            "ingest",
            unknown_path,
            "--state-dir",
            state_rel,
            "--media-type",
            "application/octet-stream",
            "--derived-mode",
            "unsupported",
            "--json",
        ],
    )
    unknown_artifact = _artifact(transcripts["ingest_unknown"])
    unknown_id = unknown_artifact.get("artifact_id", "")
    transcripts["show_unknown"] = _run_cli_json(root, ["artifact", "show", unknown_id, "--state-dir", state_rel, "--json"]) if unknown_id else {}

    transcripts["dedup_v1_first"] = _run_cli_json(root, ["artifact", "ingest", dedup_v1_path, "--state-dir", state_rel, "--json"])
    dedup_v1_artifact = _artifact(transcripts["dedup_v1_first"])
    dedup_v1_id = dedup_v1_artifact.get("artifact_id", "")
    transcripts["dedup_v1_second"] = _run_cli_json(root, ["artifact", "ingest", dedup_v1_path, "--state-dir", state_rel, "--json"])
    transcripts["dedup_v2"] = _run_cli_json(
        root,
        ["artifact", "ingest", dedup_v2_path, "--state-dir", state_rel, "--lineage-from", dedup_v1_id, "--json"],
    )
    dedup_v2_artifact = _artifact(transcripts["dedup_v2"])
    transcripts["audit_verify"] = _run_cli_json(root, ["audit", "verify", "--state-dir", state_rel, "--json"])

    basic_payload = _payload(transcripts["ingest_basic"])
    show_basic_artifact = _artifact(transcripts["show_basic"])
    fail_payload = _payload(transcripts["ingest_fail"])
    unknown_payload = _payload(transcripts["ingest_unknown"])
    dedup_second_payload = _payload(transcripts["dedup_v1_second"])

    arch_001_ok = (
        _exit_ok(transcripts["ingest_basic"])
        and basic_artifact.get("artifact_id")
        and basic_artifact.get("checksum_sha256")
        and str(basic_artifact.get("original_storage_ref", "")).startswith("sha256:")
        and basic_artifact.get("source", {}).get("ingested_at")
        and basic_payload.get("evidence_refs")
        and basic_payload.get("audit_refs")
    )
    arch_002_ok = (
        _exit_ok(transcripts["ingest_fail"])
        and _exit_ok(transcripts["show_fail"])
        and fail_artifact.get("derived", {}).get("status") == "failed"
        and str(fail_artifact.get("original_storage_ref", "")).startswith("sha256:")
    )
    arch_003_ok = (
        _exit_ok(transcripts["dedup_v1_first"])
        and _exit_ok(transcripts["dedup_v1_second"])
        and _exit_ok(transcripts["dedup_v2"])
        and dedup_v1_id
        and _artifact(transcripts["dedup_v1_second"]).get("artifact_id") == dedup_v1_id
        and dedup_second_payload.get("deduplicated") is True
        and dedup_v2_artifact.get("artifact_id") != dedup_v1_id
        and dedup_v2_artifact.get("provenance", {}).get("lineage_from") == dedup_v1_id
    )
    arch_004_ok = (
        _exit_ok(transcripts["show_basic"])
        and show_basic_artifact.get("source", {}).get("path")
        and show_basic_artifact.get("provenance", {}).get("transformations")
        and _payload(transcripts["show_basic"]).get("audit_refs")
    )
    arch_005_ok = (
        _exit_ok(transcripts["ingest_unknown"])
        and _exit_ok(transcripts["show_unknown"])
        and unknown_artifact.get("derived", {}).get("status") == "deferred"
        and unknown_artifact.get("derived", {}).get("reason") == "unsupported_format"
        and str(unknown_artifact.get("original_storage_ref", "")).startswith("sha256:")
    )
    audit_ok = _exit_ok(transcripts["audit_verify"]) and _payload(transcripts["audit_verify"]).get("audit_integrity", {}).get("status") == "success"

    rows = [
        _row(
            "CS-ARCH-001",
            "MUST_PASS",
            "PASS" if arch_001_ok and audit_ok else "FAIL",
            ["cornerstone artifact ingest fixtures/vs0/packs/01_artifact_basic/input.txt --json"],
            "Artifact ingest preserves original content with stable ID, checksum, storage ref, source timestamp, evidence refs, and audit refs.",
        ),
        _row(
            "CS-ARCH-002",
            "MUST_PASS",
            "PASS" if arch_002_ok and audit_ok else "FAIL",
            ["cornerstone artifact ingest ... --derived-mode fail --json", "cornerstone artifact show <artifact_id> --json"],
            "Derived processing failure leaves the original artifact stored, discoverable, and retryable.",
        ),
        _row(
            "CS-ARCH-003",
            "MUST_PASS",
            "PASS" if arch_003_ok and audit_ok else "FAIL",
            ["cornerstone artifact ingest <same-content> --json", "cornerstone artifact ingest <changed-content> --lineage-from <artifact_id> --json"],
            "Identical content deduplicates to one artifact identity; changed content creates a distinct artifact with lineage.",
        ),
        _row(
            "CS-ARCH-004",
            "MUST_PASS",
            "PASS" if arch_004_ok and audit_ok else "FAIL",
            ["cornerstone artifact show <artifact_id> --json"],
            "Artifact detail exposes source, timestamp, provenance transformations, evidence refs, and read audit refs.",
        ),
        _row(
            "CS-ARCH-005",
            "MUST_PASS",
            "PASS" if arch_005_ok and audit_ok else "FAIL",
            ["cornerstone artifact ingest fixtures/vs0/packs/03_unknown_and_failed_extraction/unknown.bin --media-type application/octet-stream --json"],
            "Unknown-format content keeps the immutable original and records deferred derived processing.",
        ),
    ]
    blocking = [row for row in rows if row["status"] != "PASS" and row["owner"] != "Human"]
    return {
        "status": "success" if not blocking else "failed",
        "scenario_set": "vs0-artifacts",
        "state_dir": state_rel,
        "summary": {
            "scenario_count": len(rows),
            "pass": len([row for row in rows if row["status"] == "PASS"]),
            "blocking": len(blocking),
            "product_feature_claims": "PARTIAL_VS0_ARTIFACTS_ONLY",
        },
        "scenario_results": rows,
        "transcripts": transcripts,
        "artifact_ids": {
            "basic": basic_id,
            "failed_derived": fail_id,
            "unknown_format": unknown_id,
            "dedup_v1": dedup_v1_id,
            "dedup_v2": dedup_v2_artifact.get("artifact_id"),
        },
        "negative_evidence": {
            "lost_originals": 0 if arch_001_ok and arch_002_ok and arch_005_ok else 1,
            "conflicting_duplicate_truth_records": 0 if arch_003_ok else 1,
        },
        "human_required": [],
    }


def _derived_text(root: Path, state_rel: str, artifact: dict[str, Any]) -> str:
    text_ref = artifact.get("derived", {}).get("text_ref")
    if not text_ref:
        return ""
    path = root / state_rel / "artifacts" / text_ref
    if not path.exists():
        return ""
    return path.read_text()


def verify_vs0_security(root: Path) -> dict[str, Any]:
    state_rel = _scenario_state_rel("vs0-security")
    state_path = root / state_rel
    if state_path.exists():
        shutil.rmtree(state_path)

    secret_path = "fixtures/vs0/packs/09_redaction_secrets/input.txt"
    prompt_path = "fixtures/vs0/packs/10_prompt_injection/input.txt"
    transcripts: dict[str, dict[str, Any]] = {}
    transcripts["ingest_secret"] = _run_cli_json(root, ["artifact", "ingest", secret_path, "--state-dir", state_rel, "--json"])
    transcripts["ingest_prompt_injection"] = _run_cli_json(root, ["artifact", "ingest", prompt_path, "--state-dir", state_rel, "--json"])
    transcripts["audit_verify"] = _run_cli_json(root, ["audit", "verify", "--state-dir", state_rel, "--json"])

    secret_artifact = _artifact(transcripts["ingest_secret"])
    prompt_artifact = _artifact(transcripts["ingest_prompt_injection"])
    secret_derived = _derived_text(root, state_rel, secret_artifact)
    secret_transcript_text = json.dumps(transcripts["ingest_secret"], sort_keys=True)
    prompt_payload = _payload(transcripts["ingest_prompt_injection"])
    prompt_safety = prompt_artifact.get("safety", {})
    prompt_policy_refs = prompt_payload.get("policy_decision_refs", [])
    prompt_audit_refs = prompt_payload.get("audit_refs", [])
    audit_ok = _exit_ok(transcripts["audit_verify"]) and _payload(transcripts["audit_verify"]).get("audit_integrity", {}).get("status") == "success"

    secret_output_count = count_unredacted_secrets(secret_derived) + count_unredacted_secrets(secret_transcript_text)
    prompt_ok = (
        _exit_ok(transcripts["ingest_prompt_injection"])
        and prompt_artifact.get("trust_state") == "untrusted"
        and prompt_safety.get("untrusted_evidence") is True
        and prompt_safety.get("unsafe_instruction_detected") is True
        and prompt_safety.get("blocked_attempt_count", 0) > 0
        and prompt_safety.get("tool_calls_created") == 0
        and prompt_safety.get("action_cards_created_from_untrusted_artifact") == 0
        and prompt_safety.get("external_http_calls") == 0
        and prompt_safety.get("authority_expanded") is False
        and any("policy:" in ref for ref in prompt_policy_refs)
        and len(prompt_audit_refs) >= 2
    )
    secret_ok = (
        _exit_ok(transcripts["ingest_secret"])
        and secret_artifact.get("raw_original_access", {}).get("display") == "controlled"
        and secret_artifact.get("derived", {}).get("redacted") is True
        and "[REDACTED]" in secret_derived
        and secret_output_count == 0
        and _payload(transcripts["ingest_secret"]).get("audit_refs")
    )

    rows = [
        _row(
            "CS-ARCH-006",
            "MUST_PASS",
            "PASS" if secret_ok and audit_ok else "FAIL",
            ["cornerstone artifact ingest fixtures/vs0/packs/09_redaction_secrets/input.txt --json"],
            "Secret fixture generated output is redacted and raw original access is policy-controlled.",
        ),
        _row(
            "CS-ARCH-007",
            "MUST_PASS",
            "PASS" if prompt_ok and audit_ok else "FAIL",
            ["cornerstone artifact ingest fixtures/vs0/packs/10_prompt_injection/input.txt --json"],
            "Untrusted prompt-injection artifact is evidence only, with no tool/action/egress side effects.",
        ),
        _row(
            "CS-SEC-007",
            "MUST_PASS",
            "PASS" if prompt_ok and audit_ok else "FAIL",
            ["cornerstone artifact ingest fixtures/vs0/packs/10_prompt_injection/input.txt --json", "cornerstone audit verify --json"],
            "Prompt-injection regression records a blocked attempt and preserves zero side effects.",
        ),
        _row(
            "CS-SEC-008",
            "MUST_PASS",
            "PASS" if secret_ok and audit_ok else "FAIL",
            ["cornerstone artifact ingest fixtures/vs0/packs/09_redaction_secrets/input.txt --json"],
            "Secret handling keeps raw original controlled and prevents unredacted secrets in generated output and CLI transcripts.",
        ),
        _row(
            "CS-REG-013",
            "REGRESSION_GUARD",
            "PASS" if prompt_ok and audit_ok else "FAIL",
            ["cornerstone artifact ingest fixtures/vs0/packs/10_prompt_injection/input.txt --json"],
            "Prompt text cannot expand authority; policy and safety metadata keep authority_expanded false.",
        ),
    ]
    blocking = [row for row in rows if row["status"] != "PASS" and row["owner"] != "Human"]
    return {
        "status": "success" if not blocking else "failed",
        "scenario_set": "vs0-security",
        "state_dir": state_rel,
        "summary": {
            "scenario_count": len(rows),
            "pass": len([row for row in rows if row["status"] == "PASS"]),
            "blocking": len(blocking),
            "product_feature_claims": "PARTIAL_VS0_SECURITY_ONLY",
        },
        "scenario_results": rows,
        "transcripts": transcripts,
        "security_evidence": {
            "secret_artifact_id": secret_artifact.get("artifact_id"),
            "prompt_artifact_id": prompt_artifact.get("artifact_id"),
            "secret_derived_redacted": "[REDACTED]" in secret_derived,
            "raw_original_access": secret_artifact.get("raw_original_access"),
            "prompt_safety": prompt_safety,
            "prompt_policy_decision_refs": prompt_policy_refs,
            "prompt_audit_ref_count": len(prompt_audit_refs),
        },
        "negative_evidence": {
            "unredacted_secret_occurrences": secret_output_count,
            "tool_calls_created": int(prompt_safety.get("tool_calls_created", 1)),
            "action_cards_created_from_untrusted_artifact": int(prompt_safety.get("action_cards_created_from_untrusted_artifact", 1)),
            "external_http_calls": int(prompt_safety.get("external_http_calls", 1)),
            "authority_expanded": int(bool(prompt_safety.get("authority_expanded", True))),
        },
        "human_required": [],
    }


def verify_vs0_search_evidence(root: Path) -> dict[str, Any]:
    state_rel = _scenario_state_rel("vs0-search-evidence")
    state_path = root / state_rel
    if state_path.exists():
        shutil.rmtree(state_path)

    input_path = "fixtures/vs0/packs/01_artifact_basic/input.txt"
    transcripts: dict[str, dict[str, Any]] = {}
    first_use_started = perf_counter()
    transcripts["ingest"] = _run_cli_json(root, ["artifact", "ingest", input_path, "--state-dir", state_rel, "--json"])
    ingest_artifact = _artifact(transcripts["ingest"])
    artifact_id = ingest_artifact.get("artifact_id", "")
    transcripts["search"] = _run_cli_json(root, ["search", "query", "alpha-evidence-anchor", "--state-dir", state_rel, "--json"])
    first_use_duration_ms = round((perf_counter() - first_use_started) * 1000, 3)
    search_payload = _payload(transcripts["search"])
    snapshot = search_payload.get("search_snapshot", {})
    snapshot_id = snapshot.get("search_snapshot_id", "")
    transcripts["bundle_create"] = _run_cli_json(
        root,
        ["evidence", "bundle", "create", "--search-snapshot-id", snapshot_id, "--state-dir", state_rel, "--json"],
    )
    bundle_payload = _payload(transcripts["bundle_create"])
    bundle = bundle_payload.get("evidence_bundle", {})
    bundle_id = bundle.get("evidence_bundle_id", "")
    transcripts["bundle_show"] = _run_cli_json(root, ["evidence", "bundle", "show", bundle_id, "--state-dir", state_rel, "--json"]) if bundle_id else {}
    transcripts["evidence_view"] = _run_cli_json(root, ["evidence", "view", bundle_id, "--state-dir", state_rel, "--json"]) if bundle_id else {}
    transcripts["claim_create"] = _run_cli_json(
        root,
        [
            "claim",
            "create",
            "--evidence-bundle-id",
            bundle_id,
            "--statement",
            "The alpha evidence anchor was present in the ingested fixture.",
            "--state-dir",
            state_rel,
            "--json",
        ],
    ) if bundle_id else {}
    claim_payload = _payload(transcripts["claim_create"])
    claim = claim_payload.get("claim", {})
    claim_id = claim.get("claim_id", "")
    transcripts["claim_show"] = _run_cli_json(root, ["claim", "show", claim_id, "--state-dir", state_rel, "--json"]) if claim_id else {}
    transcripts["artifact_show"] = _run_cli_json(root, ["artifact", "show", artifact_id, "--state-dir", state_rel, "--json"]) if artifact_id else {}
    transcripts["audit_verify"] = _run_cli_json(root, ["audit", "verify", "--state-dir", state_rel, "--json"])

    search_results = snapshot.get("results", [])
    bundle_items = bundle.get("evidence_items", [])
    first_result = search_results[0] if search_results else {}
    first_item = bundle_items[0] if bundle_items else {}
    viewer_payload = _payload(transcripts["evidence_view"])
    viewer = viewer_payload.get("evidence_viewer", {})
    viewer_items = viewer.get("viewer_items", [])
    first_viewer_item = viewer_items[0] if viewer_items else {}
    search_refs = search_payload.get("evidence_refs", [])
    bundle_refs = bundle_payload.get("evidence_refs", [])
    claim_evidence = claim.get("evidence_bundle", {})
    audit_ok = _exit_ok(transcripts["audit_verify"]) and _payload(transcripts["audit_verify"]).get("audit_integrity", {}).get("status") == "success"
    arch_008_ok = (
        _exit_ok(transcripts["search"])
        and _exit_ok(transcripts["bundle_create"])
        and _exit_ok(transcripts["claim_create"])
        and _exit_ok(transcripts["claim_show"])
        and snapshot.get("query") == "alpha-evidence-anchor"
        and snapshot.get("filters", {}).get("namespace_id") == "personal"
        and snapshot.get("result_count", 0) >= 1
        and bundle.get("search_snapshot_id") == snapshot_id
        and first_item.get("artifact_id") == artifact_id
        and first_item.get("search_snapshot_id") == snapshot_id
        and claim.get("status") == "draft"
        and claim_evidence.get("evidence_bundle_id") == bundle_id
        and claim_evidence.get("search_snapshot_id") == snapshot_id
        and claim_evidence.get("query") == "alpha-evidence-anchor"
        and f"artifact:{artifact_id}" in claim_evidence.get("artifact_refs", [])
        and _payload(transcripts["claim_create"]).get("audit_refs")
        and any(ref.startswith("search_snapshot:") for ref in bundle_refs)
        and any(ref.startswith("artifact:") for ref in bundle_refs)
    )
    arch_009_ok = (
        _exit_ok(transcripts["bundle_show"])
        and _exit_ok(transcripts["evidence_view"])
        and _exit_ok(transcripts["artifact_show"])
        and _payload(transcripts["bundle_show"]).get("audit_refs")
        and viewer_payload.get("audit_refs")
        and first_item.get("original_storage_ref", "").startswith("sha256:")
        and bool(first_item.get("derived_text_ref"))
        and _artifact(transcripts["artifact_show"]).get("original_storage_ref") == first_item.get("original_storage_ref")
        and _artifact(transcripts["artifact_show"]).get("derived", {}).get("text_ref") == first_item.get("derived_text_ref")
        and first_viewer_item.get("original", {}).get("storage_ref") == first_item.get("original_storage_ref")
        and first_viewer_item.get("derived", {}).get("text_ref") == first_item.get("derived_text_ref")
        and "alpha-evidence-anchor" in first_viewer_item.get("derived", {}).get("text_preview", "")
    )
    und_001_ok = (
        _exit_ok(transcripts["search"])
        and first_result.get("artifact_id") == artifact_id
        and "alpha-evidence-anchor" in first_result.get("snippet", "")
        and snapshot.get("duration_ms", 999999) <= 1000
        and first_use_duration_ms <= 5000
        and any(ref.startswith("search_snapshot:") for ref in search_refs)
        and any(ref.startswith("artifact:") for ref in search_refs)
    )
    rows = [
        _row(
            "CS-ARCH-008",
            "MUST_PASS",
            "PASS" if arch_008_ok and audit_ok else "FAIL",
            [
                "cornerstone search query alpha-evidence-anchor --json",
                "cornerstone evidence bundle create --search-snapshot-id <id> --json",
                "cornerstone claim create --evidence-bundle-id <id> --json",
            ],
            "A draft claim links to an evidence bundle containing stored search query, filters, result snapshot, and artifact refs.",
        ),
        _row(
            "CS-ARCH-009",
            "MUST_PASS",
            "PASS" if arch_009_ok and audit_ok else "FAIL",
            ["cornerstone evidence view <bundle_id> --json", "cornerstone artifact show <artifact_id> --json"],
            "The evidence viewer opens original source metadata plus derived text/metadata for the cited artifact.",
        ),
        _row(
            "CS-UND-001",
            "MUST_PASS",
            "PASS" if und_001_ok and audit_ok else "FAIL",
            ["cornerstone artifact ingest ... --json", "cornerstone search query alpha-evidence-anchor --json"],
            "Search immediately returns the ingested artifact with a snippet, timing, and evidence-ready refs.",
        ),
    ]
    blocking = [row for row in rows if row["status"] != "PASS" and row["owner"] != "Human"]
    return {
        "status": "success" if not blocking else "failed",
        "scenario_set": "vs0-search-evidence",
        "state_dir": state_rel,
        "summary": {
            "scenario_count": len(rows),
            "pass": len([row for row in rows if row["status"] == "PASS"]),
            "blocking": len(blocking),
            "product_feature_claims": "PARTIAL_VS0_SEARCH_EVIDENCE_ONLY",
        },
        "scenario_results": rows,
        "transcripts": transcripts,
        "search_evidence": {
            "artifact_id": artifact_id,
            "search_snapshot_id": snapshot_id,
            "evidence_bundle_id": bundle_id,
            "result_count": snapshot.get("result_count"),
            "duration_ms": snapshot.get("duration_ms"),
            "first_use_duration_ms": first_use_duration_ms,
            "first_snippet": first_result.get("snippet"),
            "original_storage_ref": first_item.get("original_storage_ref"),
            "derived_text_ref": first_item.get("derived_text_ref"),
            "claim_id": claim_id,
            "evidence_viewer_id": viewer.get("evidence_viewer_id"),
        },
        "negative_evidence": {},
        "human_required": [],
    }


def verify_vs0_search_understanding(root: Path) -> dict[str, Any]:
    state_rel = _scenario_state_rel("vs0-search-understanding")
    state_path = root / state_rel
    if state_path.exists():
        shutil.rmtree(state_path)

    basic_path = "fixtures/vs0/packs/01_artifact_basic/input.txt"
    personal_path = "fixtures/vs0/packs/08_namespace_isolation/personal.txt"
    org_path = "fixtures/vs0/packs/08_namespace_isolation/org.txt"
    project_path = "fixtures/vs0/packs/02_dedup_versioning/input_v2.txt"
    transcripts: dict[str, dict[str, Any]] = {}

    transcripts["ingest_basic"] = _run_cli_json(root, ["artifact", "ingest", basic_path, "--state-dir", state_rel, "--json"])
    basic_artifact = _artifact(transcripts["ingest_basic"])
    basic_id = basic_artifact.get("artifact_id", "")
    transcripts["exact_search"] = _run_cli_json(root, ["search", "query", "alpha-evidence-anchor", "--state-dir", state_rel, "--json"])
    transcripts["semantic_search"] = _run_cli_json(root, ["search", "query", "retain raw proof", "--state-dir", state_rel, "--json"])

    exact_snapshot = _payload(transcripts["exact_search"]).get("search_snapshot", {})
    semantic_snapshot = _payload(transcripts["semantic_search"]).get("search_snapshot", {})
    exact_first = (exact_snapshot.get("results") or [{}])[0]
    semantic_first = (semantic_snapshot.get("results") or [{}])[0]

    transcripts["ingest_personal"] = _run_cli_json(root, ["artifact", "ingest", personal_path, "--state-dir", state_rel, "--json"])
    personal_artifact = _artifact(transcripts["ingest_personal"])
    personal_id = personal_artifact.get("artifact_id", "")
    transcripts["ingest_org"] = _run_cli_json(
        root,
        [
            "artifact",
            "ingest",
            org_path,
            "--state-dir",
            state_rel,
            "--owner-id",
            "local-org",
            "--namespace-id",
            "organization",
            "--workspace-id",
            "ops",
            "--json",
        ],
    )
    org_artifact = _artifact(transcripts["ingest_org"])
    org_id = org_artifact.get("artifact_id", "")
    transcripts["ingest_project"] = _run_cli_json(
        root,
        [
            "artifact",
            "ingest",
            project_path,
            "--state-dir",
            state_rel,
            "--owner-id",
            "local-project",
            "--namespace-id",
            "project",
            "--workspace-id",
            "alpha-project",
            "--json",
        ],
    )
    project_artifact = _artifact(transcripts["ingest_project"])
    project_id = project_artifact.get("artifact_id", "")
    transcripts["ingest_same_content_org"] = _run_cli_json(
        root,
        [
            "artifact",
            "ingest",
            basic_path,
            "--state-dir",
            state_rel,
            "--owner-id",
            "local-org",
            "--namespace-id",
            "organization",
            "--workspace-id",
            "ops",
            "--json",
        ],
    )
    transcripts["personal_search"] = _run_cli_json(root, ["search", "query", "personal-only-alpha", "--state-dir", state_rel, "--json"])
    transcripts["personal_cross_org_search"] = _run_cli_json(root, ["search", "query", "org-visible-beta", "--state-dir", state_rel, "--json"])
    transcripts["personal_cross_project_search"] = _run_cli_json(root, ["search", "query", "distinct artifact", "--state-dir", state_rel, "--json"])
    transcripts["org_search"] = _run_cli_json(
        root,
        [
            "search",
            "query",
            "org-visible-beta",
            "--state-dir",
            state_rel,
            "--owner-id",
            "local-org",
            "--namespace-id",
            "organization",
            "--workspace-id",
            "ops",
            "--json",
        ],
    )
    transcripts["org_cross_personal_search"] = _run_cli_json(
        root,
        [
            "search",
            "query",
            "personal-only-alpha",
            "--state-dir",
            state_rel,
            "--owner-id",
            "local-org",
            "--namespace-id",
            "organization",
            "--workspace-id",
            "ops",
            "--json",
        ],
    )
    transcripts["org_cross_project_search"] = _run_cli_json(
        root,
        [
            "search",
            "query",
            "distinct artifact",
            "--state-dir",
            state_rel,
            "--owner-id",
            "local-org",
            "--namespace-id",
            "organization",
            "--workspace-id",
            "ops",
            "--json",
        ],
    )
    transcripts["project_search"] = _run_cli_json(
        root,
        [
            "search",
            "query",
            "distinct artifact",
            "--state-dir",
            state_rel,
            "--owner-id",
            "local-project",
            "--namespace-id",
            "project",
            "--workspace-id",
            "alpha-project",
            "--json",
        ],
    )
    transcripts["project_cross_personal_search"] = _run_cli_json(
        root,
        [
            "search",
            "query",
            "personal-only-alpha",
            "--state-dir",
            state_rel,
            "--owner-id",
            "local-project",
            "--namespace-id",
            "project",
            "--workspace-id",
            "alpha-project",
            "--json",
        ],
    )
    transcripts["project_cross_org_search"] = _run_cli_json(
        root,
        [
            "search",
            "query",
            "org-visible-beta",
            "--state-dir",
            state_rel,
            "--owner-id",
            "local-project",
            "--namespace-id",
            "project",
            "--workspace-id",
            "alpha-project",
            "--json",
        ],
    )
    transcripts["same_content_personal_search"] = _run_cli_json(root, ["search", "query", "alpha-evidence-anchor", "--state-dir", state_rel, "--json"])
    transcripts["same_content_org_search"] = _run_cli_json(
        root,
        [
            "search",
            "query",
            "alpha-evidence-anchor",
            "--state-dir",
            state_rel,
            "--owner-id",
            "local-org",
            "--namespace-id",
            "organization",
            "--workspace-id",
            "ops",
            "--json",
        ],
    )
    transcripts["audit_verify"] = _run_cli_json(root, ["audit", "verify", "--state-dir", state_rel, "--json"])

    personal_snapshot = _payload(transcripts["personal_search"]).get("search_snapshot", {})
    personal_cross_org_snapshot = _payload(transcripts["personal_cross_org_search"]).get("search_snapshot", {})
    personal_cross_project_snapshot = _payload(transcripts["personal_cross_project_search"]).get("search_snapshot", {})
    org_snapshot = _payload(transcripts["org_search"]).get("search_snapshot", {})
    org_cross_personal_snapshot = _payload(transcripts["org_cross_personal_search"]).get("search_snapshot", {})
    org_cross_project_snapshot = _payload(transcripts["org_cross_project_search"]).get("search_snapshot", {})
    project_snapshot = _payload(transcripts["project_search"]).get("search_snapshot", {})
    project_cross_personal_snapshot = _payload(transcripts["project_cross_personal_search"]).get("search_snapshot", {})
    project_cross_org_snapshot = _payload(transcripts["project_cross_org_search"]).get("search_snapshot", {})
    same_content_personal_snapshot = _payload(transcripts["same_content_personal_search"]).get("search_snapshot", {})
    same_content_org_snapshot = _payload(transcripts["same_content_org_search"]).get("search_snapshot", {})
    personal_first = (personal_snapshot.get("results") or [{}])[0]
    org_first = (org_snapshot.get("results") or [{}])[0]
    project_first = (project_snapshot.get("results") or [{}])[0]
    same_content_personal_first = (same_content_personal_snapshot.get("results") or [{}])[0]
    same_content_org_first = (same_content_org_snapshot.get("results") or [{}])[0]

    audit_ok = _exit_ok(transcripts["audit_verify"]) and _payload(transcripts["audit_verify"]).get("audit_integrity", {}).get("status") == "success"
    und_002_ok = (
        _exit_ok(transcripts["exact_search"])
        and _exit_ok(transcripts["semantic_search"])
        and exact_first.get("artifact_id") == basic_id
        and semantic_first.get("artifact_id") == basic_id
        and "alpha-evidence-anchor" in exact_first.get("snippet", "")
        and "semantic" in semantic_first.get("retrieval_modes", [])
        and any(reason.get("type") == "semantic_alias" for reason in semantic_first.get("match_reasons", []))
        and _payload(transcripts["semantic_search"]).get("evidence_refs")
    )
    und_003_ok = (
        _exit_ok(transcripts["personal_search"])
        and _exit_ok(transcripts["personal_cross_org_search"])
        and _exit_ok(transcripts["personal_cross_project_search"])
        and _exit_ok(transcripts["org_search"])
        and _exit_ok(transcripts["org_cross_personal_search"])
        and _exit_ok(transcripts["org_cross_project_search"])
        and _exit_ok(transcripts["project_search"])
        and _exit_ok(transcripts["project_cross_personal_search"])
        and _exit_ok(transcripts["project_cross_org_search"])
        and _exit_ok(transcripts["same_content_personal_search"])
        and _exit_ok(transcripts["same_content_org_search"])
        and personal_snapshot.get("result_count") == 1
        and personal_first.get("artifact_id") == personal_id
        and personal_first.get("scope", {}).get("namespace_id") == "personal"
        and personal_cross_org_snapshot.get("result_count") == 0
        and personal_cross_project_snapshot.get("result_count") == 0
        and org_snapshot.get("result_count") == 1
        and org_first.get("artifact_id") == org_id
        and org_first.get("scope", {}).get("namespace_id") == "organization"
        and org_cross_personal_snapshot.get("result_count") == 0
        and org_cross_project_snapshot.get("result_count") == 0
        and project_snapshot.get("result_count") == 1
        and project_first.get("artifact_id") == project_id
        and project_first.get("scope", {}).get("namespace_id") == "project"
        and project_cross_personal_snapshot.get("result_count") == 0
        and project_cross_org_snapshot.get("result_count") == 0
        and same_content_personal_snapshot.get("result_count") == 1
        and same_content_org_snapshot.get("result_count") == 1
        and same_content_personal_first.get("scope", {}).get("owner_id") == "local-user"
        and same_content_org_first.get("scope", {}).get("owner_id") == "local-org"
    )

    rows = [
        _row(
            "CS-UND-002",
            "MUST_PASS",
            "PASS" if und_002_ok and audit_ok else "FAIL",
            ["cornerstone search query alpha-evidence-anchor --json", "cornerstone search query 'retain raw proof' --json"],
            "Keyword and deterministic semantic-alias retrieval both return the artifact with evidence refs and inspectable match reasons.",
        ),
        _row(
            "CS-UND-003",
            "MUST_PASS",
            "PASS" if und_003_ok and audit_ok else "FAIL",
            [
                "cornerstone search query personal-only-alpha --json",
                "cornerstone search query org-visible-beta --namespace-id organization --json",
                "cornerstone search query 'distinct artifact' --namespace-id project --json",
            ],
            "Search results stay inside the active owner/namespace/workspace scope in controlled personal, organization, and project fixtures.",
        ),
    ]
    blocking = [row for row in rows if row["status"] != "PASS" and row["owner"] != "Human"]
    return {
        "status": "success" if not blocking else "failed",
        "scenario_set": "vs0-search-understanding",
        "state_dir": state_rel,
        "summary": {
            "scenario_count": len(rows),
            "pass": len([row for row in rows if row["status"] == "PASS"]),
            "blocking": len(blocking),
            "product_feature_claims": "PARTIAL_VS0_SEARCH_UNDERSTANDING_ONLY",
        },
        "scenario_results": rows,
        "transcripts": transcripts,
        "search_understanding": {
            "keyword_artifact_id": exact_first.get("artifact_id"),
            "semantic_artifact_id": semantic_first.get("artifact_id"),
            "semantic_match_reasons": semantic_first.get("match_reasons", []),
            "personal_result_count": personal_snapshot.get("result_count"),
            "personal_cross_org_result_count": personal_cross_org_snapshot.get("result_count"),
            "personal_cross_project_result_count": personal_cross_project_snapshot.get("result_count"),
            "organization_result_count": org_snapshot.get("result_count"),
            "organization_cross_personal_result_count": org_cross_personal_snapshot.get("result_count"),
            "organization_cross_project_result_count": org_cross_project_snapshot.get("result_count"),
            "project_result_count": project_snapshot.get("result_count"),
            "project_cross_personal_result_count": project_cross_personal_snapshot.get("result_count"),
            "project_cross_org_result_count": project_cross_org_snapshot.get("result_count"),
            "same_content_personal_scope": same_content_personal_first.get("scope"),
            "same_content_organization_scope": same_content_org_first.get("scope"),
        },
        "negative_evidence": {
            "cross_workspace_results": (
                int(personal_cross_org_snapshot.get("result_count", 1))
                + int(personal_cross_project_snapshot.get("result_count", 1))
                + int(org_cross_personal_snapshot.get("result_count", 1))
                + int(org_cross_project_snapshot.get("result_count", 1))
                + int(project_cross_personal_snapshot.get("result_count", 1))
                + int(project_cross_org_snapshot.get("result_count", 1))
            ),
            "semantic_unexplained_results": 0 if any(reason.get("type") == "semantic_alias" for reason in semantic_first.get("match_reasons", [])) else 1,
            "same_content_scope_collisions": 0 if same_content_personal_first.get("scope", {}).get("owner_id") == "local-user" and same_content_org_first.get("scope", {}).get("owner_id") == "local-org" else 1,
        },
        "human_required": [],
    }


def verify_vs0_namespace_isolation(root: Path) -> dict[str, Any]:
    state_rel = _scenario_state_rel("vs0-namespace-isolation")
    state_path = root / state_rel
    if state_path.exists():
        shutil.rmtree(state_path)

    personal_path = "fixtures/vs0/packs/08_namespace_isolation/personal.txt"
    org_path = "fixtures/vs0/packs/08_namespace_isolation/org.txt"
    org_scope_args = ["--owner-id", "local-org", "--namespace-id", "organization", "--workspace-id", "ops"]
    transcripts: dict[str, dict[str, Any]] = {}

    transcripts["ingest_personal"] = _run_cli_json(root, ["artifact", "ingest", personal_path, "--state-dir", state_rel, "--json"])
    personal_artifact = _artifact(transcripts["ingest_personal"])
    personal_id = personal_artifact.get("artifact_id", "")
    transcripts["ingest_org"] = _run_cli_json(root, ["artifact", "ingest", org_path, "--state-dir", state_rel, *org_scope_args, "--json"])
    org_artifact = _artifact(transcripts["ingest_org"])
    org_id = org_artifact.get("artifact_id", "")

    transcripts["personal_search"] = _run_cli_json(root, ["search", "query", "personal-only-alpha", "--state-dir", state_rel, "--json"])
    transcripts["org_search"] = _run_cli_json(root, ["search", "query", "org-visible-beta", "--state-dir", state_rel, *org_scope_args, "--json"])
    transcripts["org_cross_personal_search"] = _run_cli_json(root, ["search", "query", "personal-only-alpha", "--state-dir", state_rel, *org_scope_args, "--json"])
    transcripts["personal_cross_org_search"] = _run_cli_json(root, ["search", "query", "org-visible-beta", "--state-dir", state_rel, "--json"])

    personal_snapshot = _payload(transcripts["personal_search"]).get("search_snapshot", {})
    org_snapshot = _payload(transcripts["org_search"]).get("search_snapshot", {})
    org_cross_personal_snapshot = _payload(transcripts["org_cross_personal_search"]).get("search_snapshot", {})
    personal_cross_org_snapshot = _payload(transcripts["personal_cross_org_search"]).get("search_snapshot", {})
    personal_snapshot_id = personal_snapshot.get("search_snapshot_id", "")
    org_snapshot_id = org_snapshot.get("search_snapshot_id", "")

    transcripts["org_bundle_create"] = _run_cli_json(
        root,
        ["evidence", "bundle", "create", "--search-snapshot-id", org_snapshot_id, "--state-dir", state_rel, *org_scope_args, "--json"],
    ) if org_snapshot_id else {}
    org_bundle = _payload(transcripts["org_bundle_create"]).get("evidence_bundle", {})
    org_bundle_id = org_bundle.get("evidence_bundle_id", "")
    transcripts["org_claim_create"] = _run_cli_json(
        root,
        [
            "claim",
            "create",
            "--evidence-bundle-id",
            org_bundle_id,
            "--statement",
            "The organization visible beta phrase belongs to the organization workspace.",
            "--state-dir",
            state_rel,
            *org_scope_args,
            "--json",
        ],
    ) if org_bundle_id else {}
    org_claim = _payload(transcripts["org_claim_create"]).get("claim", {})

    transcripts["org_create_bundle_from_personal_snapshot"] = _run_cli_json(
        root,
        ["evidence", "bundle", "create", "--search-snapshot-id", personal_snapshot_id, "--state-dir", state_rel, *org_scope_args, "--json"],
    ) if personal_snapshot_id else {}
    transcripts["personal_bundle_create"] = _run_cli_json(
        root,
        ["evidence", "bundle", "create", "--search-snapshot-id", personal_snapshot_id, "--state-dir", state_rel, "--json"],
    ) if personal_snapshot_id else {}
    personal_bundle = _payload(transcripts["personal_bundle_create"]).get("evidence_bundle", {})
    personal_bundle_id = personal_bundle.get("evidence_bundle_id", "")
    transcripts["org_claim_from_personal_bundle"] = _run_cli_json(
        root,
        [
            "claim",
            "create",
            "--evidence-bundle-id",
            personal_bundle_id,
            "--statement",
            "Organization should not draft from personal-only evidence.",
            "--state-dir",
            state_rel,
            *org_scope_args,
            "--json",
        ],
    ) if personal_bundle_id else {}
    transcripts["audit_verify"] = _run_cli_json(root, ["audit", "verify", "--state-dir", state_rel, "--json"])

    personal_first = (personal_snapshot.get("results") or [{}])[0]
    org_first = (org_snapshot.get("results") or [{}])[0]
    audit_events = _audit_events(root, state_rel)
    audit_ok = _exit_ok(transcripts["audit_verify"]) and _payload(transcripts["audit_verify"]).get("audit_integrity", {}).get("status") == "success"

    context_scopes = [
        personal_artifact.get("scope"),
        org_artifact.get("scope"),
        personal_snapshot.get("filters"),
        org_snapshot.get("filters"),
        org_bundle.get("filters"),
        org_claim.get("scope"),
        org_claim.get("evidence_bundle", {}).get("filters"),
    ]
    context_scopes.extend(result.get("scope") for result in personal_snapshot.get("results", []) if isinstance(result, dict))
    context_scopes.extend(result.get("scope") for result in org_snapshot.get("results", []) if isinstance(result, dict))
    context_scopes.extend(_event_scope(event) for event in audit_events)
    ownerless_records = sum(1 for scope in context_scopes if not _scope_complete(scope))

    ns_001_ok = (
        _exit_ok(transcripts["ingest_personal"])
        and _exit_ok(transcripts["ingest_org"])
        and _exit_ok(transcripts["personal_search"])
        and _exit_ok(transcripts["org_search"])
        and _exit_ok(transcripts["org_bundle_create"])
        and _exit_ok(transcripts["org_claim_create"])
        and personal_artifact.get("scope", {}).get("owner_id") == "local-user"
        and personal_artifact.get("scope", {}).get("namespace_id") == "personal"
        and org_artifact.get("scope", {}).get("owner_id") == "local-org"
        and org_artifact.get("scope", {}).get("namespace_id") == "organization"
        and org_bundle.get("filters", {}).get("namespace_id") == "organization"
        and org_claim.get("scope", {}).get("namespace_id") == "organization"
        and ownerless_records == 0
    )
    ns_003_ok = (
        _exit_ok(transcripts["personal_search"])
        and _exit_ok(transcripts["org_search"])
        and _exit_ok(transcripts["org_cross_personal_search"])
        and _exit_ok(transcripts["personal_cross_org_search"])
        and personal_snapshot.get("result_count") == 1
        and personal_first.get("artifact_id") == personal_id
        and personal_first.get("scope", {}).get("namespace_id") == "personal"
        and org_snapshot.get("result_count") == 1
        and org_first.get("artifact_id") == org_id
        and org_first.get("scope", {}).get("namespace_id") == "organization"
        and org_cross_personal_snapshot.get("result_count") == 0
        and personal_cross_org_snapshot.get("result_count") == 0
        and _scope_denied(transcripts["org_create_bundle_from_personal_snapshot"])
        and _scope_denied(transcripts["org_claim_from_personal_bundle"])
    )

    rows = [
        _row(
            "CS-NS-001",
            "MUST_PASS",
            "PASS" if ns_001_ok and audit_ok else "FAIL",
            [
                "cornerstone artifact ingest fixtures/vs0/packs/08_namespace_isolation/personal.txt --json",
                "cornerstone artifact ingest fixtures/vs0/packs/08_namespace_isolation/org.txt --owner-id local-org --namespace-id organization --workspace-id ops --json",
                "cornerstone evidence bundle create --search-snapshot-id <org_snapshot_id> --owner-id local-org --namespace-id organization --workspace-id ops --json",
                "cornerstone claim create --evidence-bundle-id <org_bundle_id> --owner-id local-org --namespace-id organization --workspace-id ops --json",
            ],
            "Generated VS-0 context records for artifacts, search snapshots, evidence bundles, claims, and audit events all carry explicit tenant, owner, namespace, and workspace scope.",
        ),
        _row(
            "CS-NS-003",
            "MUST_PASS",
            "PASS" if ns_003_ok and audit_ok else "FAIL",
            [
                "cornerstone search query personal-only-alpha --owner-id local-org --namespace-id organization --workspace-id ops --json",
                "cornerstone evidence bundle create --search-snapshot-id <personal_snapshot_id> --owner-id local-org --namespace-id organization --workspace-id ops --json",
                "cornerstone claim create --evidence-bundle-id <personal_bundle_id> --owner-id local-org --namespace-id organization --workspace-id ops --json",
            ],
            "Organization-scoped search returns zero personal-only results, and organization attempts to reuse personal search/evidence are denied by scope.",
        ),
    ]
    blocking = [row for row in rows if row["status"] != "PASS" and row["owner"] != "Human"]
    return {
        "status": "success" if not blocking else "failed",
        "scenario_set": "vs0-namespace-isolation",
        "state_dir": state_rel,
        "summary": {
            "scenario_count": len(rows),
            "pass": len([row for row in rows if row["status"] == "PASS"]),
            "blocking": len(blocking),
            "product_feature_claims": "PARTIAL_VS0_NAMESPACE_ISOLATION_ONLY",
        },
        "scenario_results": rows,
        "transcripts": transcripts,
        "namespace_evidence": {
            "context_record_scope_count": len(context_scopes),
            "audit_event_count": len(audit_events),
            "personal_artifact_scope": personal_artifact.get("scope"),
            "organization_artifact_scope": org_artifact.get("scope"),
            "personal_result_count": personal_snapshot.get("result_count"),
            "organization_result_count": org_snapshot.get("result_count"),
            "organization_cross_personal_result_count": org_cross_personal_snapshot.get("result_count"),
            "personal_cross_organization_result_count": personal_cross_org_snapshot.get("result_count"),
            "cross_scope_evidence_attempts_denied": int(_scope_denied(transcripts["org_create_bundle_from_personal_snapshot"]))
            + int(_scope_denied(transcripts["org_claim_from_personal_bundle"])),
            "organization_claim_scope": org_claim.get("scope"),
        },
        "negative_evidence": {
            "ownerless_records": ownerless_records,
            "cross_namespace_results": int(org_cross_personal_snapshot.get("result_count", 1)) + int(personal_cross_org_snapshot.get("result_count", 1)),
            "cross_scope_access_allowed": 0 if _scope_denied(transcripts["org_create_bundle_from_personal_snapshot"]) and _scope_denied(transcripts["org_claim_from_personal_bundle"]) else 1,
            "implicit_promotions": 0,
        },
        "human_required": [],
    }


def verify_vs0_audit_ledger(root: Path) -> dict[str, Any]:
    state_rel = _scenario_state_rel("vs0-audit-ledger")
    state_path = root / state_rel
    if state_path.exists():
        shutil.rmtree(state_path)

    input_path = "fixtures/vs0/packs/01_artifact_basic/input.txt"
    transcripts: dict[str, dict[str, Any]] = {}
    transcripts["ingest"] = _run_cli_json(root, ["artifact", "ingest", input_path, "--state-dir", state_rel, "--json"])
    artifact = _artifact(transcripts["ingest"])
    artifact_id = artifact.get("artifact_id", "")
    transcripts["artifact_show"] = _run_cli_json(root, ["artifact", "show", artifact_id, "--state-dir", state_rel, "--json"]) if artifact_id else {}
    transcripts["search"] = _run_cli_json(root, ["search", "query", "alpha-evidence-anchor", "--state-dir", state_rel, "--json"])
    snapshot = _payload(transcripts["search"]).get("search_snapshot", {})
    snapshot_id = snapshot.get("search_snapshot_id", "")
    transcripts["bundle_create"] = _run_cli_json(
        root,
        ["evidence", "bundle", "create", "--search-snapshot-id", snapshot_id, "--state-dir", state_rel, "--json"],
    ) if snapshot_id else {}
    bundle = _payload(transcripts["bundle_create"]).get("evidence_bundle", {})
    bundle_id = bundle.get("evidence_bundle_id", "")
    transcripts["claim_create"] = _run_cli_json(
        root,
        [
            "claim",
            "create",
            "--evidence-bundle-id",
            bundle_id,
            "--statement",
            "The alpha evidence anchor is supported by an audit-linked evidence bundle.",
            "--state-dir",
            state_rel,
            "--json",
        ],
    ) if bundle_id else {}
    claim = _payload(transcripts["claim_create"]).get("claim", {})
    claim_id = claim.get("claim_id", "")
    transcripts["claim_approve"] = _run_cli_json(
        root,
        ["claim", "approve", claim_id, "--state-dir", state_rel, "--json"],
    ) if claim_id else {}
    transcripts["audit_verify_clean"] = _run_cli_json(root, ["audit", "verify", "--state-dir", state_rel, "--json"])

    audit_events_before_tamper = _audit_events(root, state_rel)
    audit_event_types = [event.get("event_type") for event in audit_events_before_tamper]
    event_scopes_complete = all(_scope_complete(_event_scope(event)) for event in audit_events_before_tamper)
    event_hashes_present = all(event.get("event_id") and event.get("event_hash") and event.get("previous_hash") for event in audit_events_before_tamper)
    event_details_present = all(event.get("subject") and isinstance(event.get("details"), dict) for event in audit_events_before_tamper)
    required_event_types = {
        "artifact.ingested",
        "artifact.read",
        "search.snapshot.created",
        "evidence_bundle.created",
        "claim.draft.created",
        "claim.approved",
    }
    missing_event_types = sorted(required_event_types - set(str(event_type) for event_type in audit_event_types))

    audit_path = root / state_rel / "audit" / "events.jsonl"
    if audit_path.exists():
        tampered_lines = audit_path.read_text().splitlines()
        if tampered_lines:
            tampered_lines[0] = tampered_lines[0].replace("artifact.ingested", "artifact.modified", 1)
            audit_path.write_text("\n".join(tampered_lines) + "\n")
    transcripts["audit_verify_tampered"] = _run_cli_json(root, ["audit", "verify", "--state-dir", state_rel, "--json"])
    tampered_payload = _payload(transcripts["audit_verify_tampered"])
    tamper_errors = tampered_payload.get("audit_integrity", {}).get("errors", [])
    tamper_detected = (
        transcripts["audit_verify_tampered"].get("exit_code") == 5
        and tampered_payload.get("status") == "failed"
        and isinstance(tamper_errors, list)
        and any(error.get("code") == "AUDIT_EVENT_HASH_MISMATCH" for error in tamper_errors if isinstance(error, dict))
    )
    clean_ok = (
        _exit_ok(transcripts["audit_verify_clean"])
        and _payload(transcripts["audit_verify_clean"]).get("audit_integrity", {}).get("status") == "success"
    )
    sec_006_ok = (
        _exit_ok(transcripts["ingest"])
        and _exit_ok(transcripts["artifact_show"])
        and _exit_ok(transcripts["search"])
        and _exit_ok(transcripts["bundle_create"])
        and _exit_ok(transcripts["claim_create"])
        and _exit_ok(transcripts["claim_approve"])
        and clean_ok
        and tamper_detected
        and not missing_event_types
        and event_scopes_complete
        and event_hashes_present
        and event_details_present
    )

    rows = [
        _row(
            "CS-SEC-006",
            "MUST_PASS",
            "PASS" if sec_006_ok else "FAIL",
            [
                "cornerstone artifact ingest fixtures/vs0/packs/01_artifact_basic/input.txt --json",
                "cornerstone artifact show <artifact_id> --json",
                "cornerstone search query alpha-evidence-anchor --json",
                "cornerstone evidence bundle create --search-snapshot-id <snapshot_id> --json",
                "cornerstone claim create --evidence-bundle-id <bundle_id> --json",
                "cornerstone audit verify --json",
                "controlled tamper of tmp audit JSONL followed by cornerstone audit verify --json",
            ],
            "Implemented critical events are logged with subject, details, scope, previous hash, and event hash; clean verification passes and controlled tampering is detected.",
        )
    ]
    blocking = [row for row in rows if row["status"] != "PASS" and row["owner"] != "Human"]
    return {
        "status": "success" if not blocking else "failed",
        "scenario_set": "vs0-audit-ledger",
        "state_dir": state_rel,
        "summary": {
            "scenario_count": len(rows),
            "pass": len([row for row in rows if row["status"] == "PASS"]),
            "blocking": len(blocking),
            "product_feature_claims": "PARTIAL_VS0_AUDIT_LEDGER_ONLY",
        },
        "scenario_results": rows,
        "transcripts": transcripts,
        "audit_evidence": {
            "clean_audit_event_count": len(audit_events_before_tamper),
            "event_types": audit_event_types,
            "required_event_types": sorted(required_event_types),
            "missing_event_types": missing_event_types,
            "event_scopes_complete": event_scopes_complete,
            "event_hashes_present": event_hashes_present,
            "event_details_present": event_details_present,
            "tamper_detection_exit_code": transcripts["audit_verify_tampered"].get("exit_code"),
            "tamper_detection_errors": tamper_errors,
        },
        "negative_evidence": {
            "missing_required_event_types": len(missing_event_types),
            "events_without_scope": 0 if event_scopes_complete else 1,
            "events_without_hashes": 0 if event_hashes_present else 1,
            "events_without_review_details": 0 if event_details_present else 1,
            "tamper_accepted": 0 if tamper_detected else 1,
        },
        "human_required": [],
    }


def verify_vs0_universal_core(root: Path) -> dict[str, Any]:
    state_rel = _scenario_state_rel("vs0-universal-core")
    state_path = root / state_rel
    if state_path.exists():
        shutil.rmtree(state_path)

    input_path = "fixtures/vs0/packs/01_artifact_basic/input.txt"
    raw_fixture = (root / input_path).read_text()
    logistics_terms = [
        "logistics",
        "freight",
        "shipment",
        "dispatch",
        "transport request",
        "carrier",
        "truck",
        "warehouse",
    ]
    found_logistics_terms = [term for term in logistics_terms if term in raw_fixture.lower()]
    transcripts: dict[str, dict[str, Any]] = {}
    transcripts["ingest"] = _run_cli_json(root, ["artifact", "ingest", input_path, "--state-dir", state_rel, "--json"])
    artifact = _artifact(transcripts["ingest"])
    artifact_id = artifact.get("artifact_id", "")
    transcripts["search"] = _run_cli_json(root, ["search", "query", "alpha-evidence-anchor", "--state-dir", state_rel, "--json"])
    snapshot = _payload(transcripts["search"]).get("search_snapshot", {})
    snapshot_id = snapshot.get("search_snapshot_id", "")
    transcripts["bundle_create"] = _run_cli_json(
        root,
        ["evidence", "bundle", "create", "--search-snapshot-id", snapshot_id, "--state-dir", state_rel, "--json"],
    ) if snapshot_id else {}
    bundle = _payload(transcripts["bundle_create"]).get("evidence_bundle", {})
    bundle_id = bundle.get("evidence_bundle_id", "")
    transcripts["claim_create"] = _run_cli_json(
        root,
        [
            "claim",
            "create",
            "--evidence-bundle-id",
            bundle_id,
            "--statement",
            "The Alpha research review fixture is supported by non-logistics source evidence.",
            "--state-dir",
            state_rel,
            "--json",
        ],
    ) if bundle_id else {}
    claim = _payload(transcripts["claim_create"]).get("claim", {})
    transcripts["audit_verify"] = _run_cli_json(root, ["audit", "verify", "--state-dir", state_rel, "--json"])

    search_results = snapshot.get("results", [])
    bundle_items = bundle.get("evidence_items", [])
    claim_evidence = claim.get("evidence_bundle", {})
    audit_ok = _exit_ok(transcripts["audit_verify"]) and _payload(transcripts["audit_verify"]).get("audit_integrity", {}).get("status") == "success"
    reg_004_ok = (
        not found_logistics_terms
        and _exit_ok(transcripts["ingest"])
        and _exit_ok(transcripts["search"])
        and _exit_ok(transcripts["bundle_create"])
        and _exit_ok(transcripts["claim_create"])
        and audit_ok
        and artifact_id
        and snapshot.get("result_count") == 1
        and search_results
        and search_results[0].get("artifact_id") == artifact_id
        and bundle_items
        and bundle_items[0].get("artifact_id") == artifact_id
        and claim.get("status") == "draft"
        and f"artifact:{artifact_id}" in claim_evidence.get("artifact_refs", [])
    )

    rows = [
        _row(
            "CS-REG-004",
            "REGRESSION_GUARD",
            "PASS" if reg_004_ok else "FAIL",
            [
                "cornerstone artifact ingest fixtures/vs0/packs/01_artifact_basic/input.txt --json",
                "cornerstone search query alpha-evidence-anchor --json",
                "cornerstone evidence bundle create --search-snapshot-id <snapshot_id> --json",
                "cornerstone claim create --evidence-bundle-id <bundle_id> --json",
                "cornerstone audit verify --json",
            ],
            "A general-purpose Alpha research fixture with no logistics terms remains usable through the universal artifact, search, evidence, draft claim, and audit path.",
        )
    ]
    blocking = [row for row in rows if row["status"] != "PASS" and row["owner"] != "Human"]
    return {
        "status": "success" if not blocking else "failed",
        "scenario_set": "vs0-universal-core",
        "state_dir": state_rel,
        "summary": {
            "scenario_count": len(rows),
            "pass": len([row for row in rows if row["status"] == "PASS"]),
            "blocking": len(blocking),
            "product_feature_claims": "PARTIAL_VS0_UNIVERSAL_CORE_ONLY",
        },
        "scenario_results": rows,
        "transcripts": transcripts,
        "universal_core_evidence": {
            "fixture": input_path,
            "forbidden_logistics_terms": logistics_terms,
            "found_logistics_terms": found_logistics_terms,
            "artifact_id": artifact_id,
            "search_result_count": snapshot.get("result_count"),
            "evidence_bundle_id": bundle_id,
            "claim_id": claim.get("claim_id"),
            "claim_artifact_refs": claim_evidence.get("artifact_refs", []),
            "audit_event_count": _payload(transcripts["audit_verify"]).get("audit_integrity", {}).get("event_count"),
        },
        "negative_evidence": {
            "logistics_terms_found": len(found_logistics_terms),
            "generic_fixture_failures": 0 if reg_004_ok else 1,
        },
        "human_required": [],
    }


def _error_codes(transcript: dict[str, Any]) -> list[str]:
    errors = _payload(transcript).get("errors", [])
    if not isinstance(errors, list):
        return []
    return [str(error.get("code")) for error in errors if isinstance(error, dict)]


def verify_vs0_claim_evidence(root: Path) -> dict[str, Any]:
    state_rel = _scenario_state_rel("vs0-claim-evidence")
    state_path = root / state_rel
    if state_path.exists():
        shutil.rmtree(state_path)

    input_path = "fixtures/vs0/packs/01_artifact_basic/input.txt"
    transcripts: dict[str, dict[str, Any]] = {}
    transcripts["unsupported_claim_create"] = _run_cli_json(
        root,
        [
            "claim",
            "create",
            "--statement",
            "Unsupported draft claim without evidence.",
            "--state-dir",
            state_rel,
            "--json",
        ],
    )
    unsupported_claim = _payload(transcripts["unsupported_claim_create"]).get("claim", {})
    unsupported_claim_id = unsupported_claim.get("claim_id", "")
    transcripts["unsupported_claim_show"] = _run_cli_json(
        root,
        ["claim", "show", unsupported_claim_id, "--state-dir", state_rel, "--json"],
    ) if unsupported_claim_id else {}
    transcripts["unsupported_claim_approve"] = _run_cli_json(
        root,
        ["claim", "approve", unsupported_claim_id, "--state-dir", state_rel, "--json"],
    ) if unsupported_claim_id else {}

    transcripts["ingest"] = _run_cli_json(root, ["artifact", "ingest", input_path, "--state-dir", state_rel, "--json"])
    artifact = _artifact(transcripts["ingest"])
    artifact_id = artifact.get("artifact_id", "")
    transcripts["search"] = _run_cli_json(root, ["search", "query", "alpha-evidence-anchor", "--state-dir", state_rel, "--json"])
    snapshot = _payload(transcripts["search"]).get("search_snapshot", {})
    snapshot_id = snapshot.get("search_snapshot_id", "")
    transcripts["bundle_create"] = _run_cli_json(
        root,
        ["evidence", "bundle", "create", "--search-snapshot-id", snapshot_id, "--state-dir", state_rel, "--json"],
    ) if snapshot_id else {}
    bundle = _payload(transcripts["bundle_create"]).get("evidence_bundle", {})
    bundle_id = bundle.get("evidence_bundle_id", "")
    transcripts["evidence_claim_create"] = _run_cli_json(
        root,
        [
            "claim",
            "create",
            "--evidence-bundle-id",
            bundle_id,
            "--statement",
            "Evidence-backed claim with alpha evidence anchor.",
            "--state-dir",
            state_rel,
            "--json",
        ],
    ) if bundle_id else {}
    evidence_claim = _payload(transcripts["evidence_claim_create"]).get("claim", {})
    evidence_claim_id = evidence_claim.get("claim_id", "")
    transcripts["evidence_claim_approve"] = _run_cli_json(
        root,
        ["claim", "approve", evidence_claim_id, "--state-dir", state_rel, "--json"],
    ) if evidence_claim_id else {}
    approved_claim = _payload(transcripts["evidence_claim_approve"]).get("claim", {})
    transcripts["audit_verify"] = _run_cli_json(root, ["audit", "verify", "--state-dir", state_rel, "--json"])

    unsupported_approval_codes = _error_codes(transcripts["unsupported_claim_approve"])
    unsupported_show_payload = _payload(transcripts["unsupported_claim_show"])
    unsupported_approval_payload = _payload(transcripts["unsupported_claim_approve"])
    unsupported_show_refs = unsupported_show_payload.get("evidence_refs", [])
    unsupported_denied = (
        transcripts["unsupported_claim_approve"].get("exit_code") == 4
        and "CS_CLAIM_EVIDENCE_REQUIRED" in unsupported_approval_codes
        and unsupported_approval_payload.get("audit_refs")
    )
    evidence_approved = (
        _exit_ok(transcripts["evidence_claim_approve"])
        and approved_claim.get("status") == "approved"
        and approved_claim.get("trust_state") == "approved"
        and approved_claim.get("authority", {}).get("can_publish_shared_truth") is True
        and approved_claim.get("authority", {}).get("can_drive_autonomous_action") is False
        and f"artifact:{artifact_id}" in approved_claim.get("evidence_bundle", {}).get("artifact_refs", [])
        and _payload(transcripts["evidence_claim_approve"]).get("audit_refs")
    )
    audit_ok = _exit_ok(transcripts["audit_verify"]) and _payload(transcripts["audit_verify"]).get("audit_integrity", {}).get("status") == "success"
    claim_006_ok = (
        _exit_ok(transcripts["unsupported_claim_create"])
        and _exit_ok(transcripts["unsupported_claim_show"])
        and unsupported_claim.get("status") == "draft"
        and unsupported_claim.get("trust_state") == "draft"
        and unsupported_show_refs == [f"claim:{unsupported_claim_id}"]
        and unsupported_claim.get("authority", {}).get("can_be_approved") is False
        and unsupported_claim.get("authority", {}).get("can_publish_shared_truth") is False
        and unsupported_claim.get("authority", {}).get("can_drive_autonomous_action") is False
        and unsupported_denied
        and audit_ok
    )
    claim_007_ok = (
        unsupported_denied
        and _exit_ok(transcripts["ingest"])
        and _exit_ok(transcripts["search"])
        and _exit_ok(transcripts["bundle_create"])
        and _exit_ok(transcripts["evidence_claim_create"])
        and evidence_claim.get("trust_state") == "evidence_backed"
        and evidence_claim.get("evidence_bundle", {}).get("evidence_item_count", 0) > 0
        and evidence_approved
        and audit_ok
    )

    rows = [
        _row(
            "CS-CLAIM-006",
            "MUST_PASS",
            "PASS" if claim_006_ok else "FAIL",
            [
                "cornerstone claim create --statement <unsupported> --json",
                "cornerstone claim approve <unsupported_claim_id> --json",
            ],
            "Unsupported drafts can exist, but approval/shared-truth/autonomous-action authority is blocked until evidence is attached.",
        ),
        _row(
            "CS-CLAIM-007",
            "MUST_PASS",
            "PASS" if claim_007_ok else "FAIL",
            [
                "cornerstone claim approve <unsupported_claim_id> --json",
                "cornerstone evidence bundle create --search-snapshot-id <snapshot_id> --json",
                "cornerstone claim create --evidence-bundle-id <bundle_id> --json",
                "cornerstone claim approve <evidence_claim_id> --json",
            ],
            "Claim approval requires an Evidence Bundle; missing evidence is denied and evidence-backed approval succeeds.",
        ),
    ]
    blocking = [row for row in rows if row["status"] != "PASS" and row["owner"] != "Human"]
    return {
        "status": "success" if not blocking else "failed",
        "scenario_set": "vs0-claim-evidence",
        "state_dir": state_rel,
        "summary": {
            "scenario_count": len(rows),
            "pass": len([row for row in rows if row["status"] == "PASS"]),
            "blocking": len(blocking),
            "product_feature_claims": "PARTIAL_VS0_CLAIM_EVIDENCE_ONLY",
        },
        "scenario_results": rows,
        "transcripts": transcripts,
        "claim_evidence": {
            "unsupported_claim_id": unsupported_claim_id,
            "unsupported_claim_trust_state": unsupported_claim.get("trust_state"),
            "unsupported_claim_show_evidence_refs": unsupported_show_refs,
            "unsupported_approval_exit_code": transcripts["unsupported_claim_approve"].get("exit_code"),
            "unsupported_approval_error_codes": unsupported_approval_codes,
            "unsupported_resolution_path": (unsupported_approval_payload.get("errors") or [{}])[0].get("resolution_path", []),
            "artifact_id": artifact_id,
            "evidence_bundle_id": bundle_id,
            "evidence_claim_id": evidence_claim_id,
            "evidence_claim_trust_state": evidence_claim.get("trust_state"),
            "approved_claim_status": approved_claim.get("status"),
            "approved_claim_trust_state": approved_claim.get("trust_state"),
            "approved_claim_authority": approved_claim.get("authority"),
            "audit_event_count": _payload(transcripts["audit_verify"]).get("audit_integrity", {}).get("event_count"),
        },
        "negative_evidence": {
            "unsupported_approval_allowed": 0 if unsupported_denied else 1,
            "evidence_claim_approval_blocked": 0 if evidence_approved else 1,
            "autonomous_action_allowed_from_claim": int(bool(approved_claim.get("authority", {}).get("can_drive_autonomous_action", True))),
        },
        "human_required": [],
    }


def verify_vs0_scaffold(root: Path) -> dict[str, Any]:
    docs_result = _run_script(root, "scripts/verify_sot_docs.sh")
    cli_result = _run_script(root, "scripts/verify_cli_native_first_docs.sh")
    readiness_result = _run_script(root, "scripts/verify_vs0_scaffold_readiness_docs.sh")
    coverage = coverage_report(root)

    script_pass = (
        docs_result["exit_code"] == 0
        and cli_result["exit_code"] == 0
        and readiness_result["exit_code"] == 0
        and coverage["ok"]
    )

    required_docs = [
        "docs/adr/ADR-0002-framework-and-version-policy.md",
        "docs/adr/ADR-0003-monorepo-setup.md",
        "docs/adr/ADR-0004-cli-native-first-setup.md",
        "docs/adr/ADR-0005-domain-boundaries.md",
        "docs/adr/ADR-0006-agent-guide.md",
        "docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md",
        "docs/verification-reports/template.md",
    ]
    missing_docs = [path for path in required_docs if not (root / path).exists()]
    dependency_files = [
        "uv.lock",
        "pnpm-lock.yaml",
        "package-lock.json",
        "requirements.txt",
        "requirements-dev.txt",
    ]
    added_dependency_files = [path for path in dependency_files if (root / path).exists()]
    feature_dirs = ["apps/web", "services/api", "services/worker"]
    product_feature_dirs = [path for path in feature_dirs if (root / path).exists()]

    results = [
        _row(
            "VS0-SCAF-001",
            "MUST_PASS",
            "PASS" if not missing_docs and docs_result["exit_code"] == 0 else "FAIL",
            [
                "docs/adr/ADR-0002-framework-and-version-policy.md",
                "scripts/verify_sot_docs.sh",
            ],
            "Setup docs define compatible baseline and docs verification passes.",
        ),
        _row(
            "VS0-SCAF-002",
            "MUST_PASS",
            "PASS" if not missing_docs else "FAIL",
            [
                "docs/adr/ADR-0003-monorepo-setup.md",
                "docs/adr/ADR-0005-domain-boundaries.md",
            ],
            "Monorepo direction preserves one product with internal engine boundaries.",
        ),
        _row(
            "VS0-SCAF-003",
            "MUST_PASS",
            "PASS" if cli_result["exit_code"] == 0 and (root / "cornerstone").exists() else "FAIL",
            [
                "scripts/verify_cli_native_first_docs.sh",
                "cornerstone --help",
                "cornerstone version --json",
            ],
            "CLI native-first is documented and a native scaffold command exists.",
        ),
        _row(
            "VS0-SCAF-004",
            "MUST_PASS",
            "PASS" if readiness_result["exit_code"] == 0 else "FAIL",
            [
                "docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md",
                "cornerstone ready --json",
            ],
            "Future product-runtime readiness is still reported honestly as not ready.",
        ),
        _row(
            "VS0-SCAF-005",
            "MUST_PASS",
            "PASS" if not added_dependency_files and not product_feature_dirs else "FAIL",
            ["git diff --name-only", "repo path review"],
            "No production dependency lockfiles or product feature runtime directories are present.",
        ),
        _row(
            "VS0-SCAF-006",
            "MUST_PASS",
            "PASS" if (root / "docs/verification-reports/template.md").exists() else "FAIL",
            ["docs/verification-reports/template.md"],
            "Verification report template can record scenario evidence and human-required items.",
        ),
        _row(
            "VS0-SCAF-R01",
            "REGRESSION_GUARD",
            "PASS" if script_pass else "FAIL",
            [
                "scripts/verify_sot_docs.sh",
                "scripts/verify_cli_native_first_docs.sh",
                "cornerstone scenario coverage --json",
            ],
            "Existing 206 full scenarios, 58 VS-0 scenarios, and CLI native-first gate remain wired.",
        ),
    ]

    blocking = [row for row in results if row["status"] != "PASS" and row["owner"] != "Human"]
    return {
        "status": "success" if not blocking else "failed",
        "scenario_set": "vs0-scaffold",
        "summary": {
            "scenario_count": len(results),
            "pass": len([row for row in results if row["status"] == "PASS"]),
            "blocking": len(blocking),
            "product_feature_claims": "NOT_VERIFIED",
        },
        "coverage": coverage,
        "command_evidence": [docs_result, cli_result, readiness_result],
        "scenario_results": results,
        "human_required": [
            {
                "id": "H-FREEZE-001",
                "why_ai_cannot_verify": "New production dependencies and lockfiles are approval-gated.",
                "required_human_action": "Approve the specific scaffold dependency set before dependency files are added.",
                "expected_evidence": "Written approval naming dependency scope, or explicit approval to use ADR-0002 targets.",
                "release_impact": "Blocks dependency-based scaffold implementation.",
            },
            {
                "id": "H-FREEZE-002",
                "why_ai_cannot_verify": "Full-set human-only ownership cannot be derived until a scenario registry exists.",
                "required_human_action": "Review generated registry classifications once implemented.",
                "expected_evidence": "Approved registry/report rows with required evidence per scenario.",
                "release_impact": "Blocks full release PASS until classified and evidenced.",
            },
        ],
    }
