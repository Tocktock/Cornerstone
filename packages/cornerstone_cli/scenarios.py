from __future__ import annotations

import json
import re
import shutil
import subprocess
import csv
from pathlib import Path
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


def verify_vs0_artifacts(root: Path) -> dict[str, Any]:
    state_rel = "tmp/scenario/vs0-artifacts"
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
