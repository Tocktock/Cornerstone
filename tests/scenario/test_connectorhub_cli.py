from __future__ import annotations

import csv
import json
import hashlib
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI_TIMEOUT_SECONDS = float(os.environ.get("CORNERSTONE_TEST_CLI_TIMEOUT_SECONDS", "360"))
sys.path.insert(0, str(ROOT / "packages"))

from cornerstone_cli.connector import (
    CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES,
    CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
    CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_REPORT_SCHEMA,
    CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_SCHEMA,
    CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_REPORT_SCHEMA,
    CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_SCHEMA,
    CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_REPORT_SCHEMA,
    CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_SCHEMA,
    CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_RECORD_DRAFT_REPORT_SCHEMA,
    CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_VALIDATION_REPORT_SCHEMA,
    CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_SCHEMA,
    CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_REPORT_SCHEMA,
    CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_ITEMS,
    CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
    CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
    CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
    CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_ITEMS,
    CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
    CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
    CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
    CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_ITEMS,
    CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
    CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
    CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
    CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS,
    CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
    CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
    CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
    CONNECTOR_HUMAN_GATE_H04_FIELD_REF_ITEMS,
    CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_ITEMS,
    CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
    CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
    CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
    CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_ITEMS,
    CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
    CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
    CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
    CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_ITEMS,
    CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
    CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
    CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
    CONNECTOR_HUMAN_GATE_H04_BASELINE_CLAIM_BOUNDARY,
    CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_BUNDLE_CLAIM_BOUNDARY,
    CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_BUNDLE_SCHEMA,
    CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_CLAIM_BOUNDARY,
    CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA,
    CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS,
    CONNECTOR_HUMAN_GATE_PREFLIGHT_BUNDLE_REPORT_SCHEMA,
    CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY,
    CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_SUMMARY_SCHEMA,
    CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY,
    CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_IDS,
    CONNECTOR_SENSITIVE_MARKER_FINDING_FIELDS,
    CONNECTOR_SENSITIVE_MARKER_POLICY_SCHEMA,
    CONNECTOR_SENSITIVE_MARKER_TYPES,
    connector_human_gate_packet_file_metadata,
    connector_human_gate_evidence_packet_scaffold_template_content,
)
from cornerstone_cli.vs2_verification_metadata import validate_reusable_report

SKIP_VS2_REGRESSION_TESTS = os.environ.get("CORNERSTONE_SKIP_VS2_REGRESSION_TESTS") == "1"
FORCE_VS2_REUSABLE_PROOF_REFRESH = os.environ.get("CORNERSTONE_FORCE_VS2_REUSABLE_PROOF_REFRESH") == "1"
VS2_REUSABLE_PROOF_REFRESHED = False
VS2_LOCAL_PROOF_REPORT = ROOT / "reports/security/vs2-local-security-proof.json"
EXPECTED_H04_PREFLIGHT_COMMAND_PLAN = [
    {
        "schema_version": CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA,
        "step_order": 1,
        "operator_phase": "refresh_local_vs2_baseline_inputs",
        "command": "cornerstone security vs2-local-proof --json",
        "purpose": (
            "Refresh the current local VS2 proof inputs before H04 review without treating "
            "local proof as production-like acceptance."
        ),
        "expected_report_paths": [
            "reports/security/vs2-local-security-proof.json",
            "reports/network/vs2-egress-proof.json",
            "reports/security/vs2-local-range.json",
        ],
        "expected_report_count": 3,
        "review_input_only": True,
        "acceptance_sufficient": False,
        "product_claim_allowed": False,
        "pass_claim_allowed": False,
        "claim_boundary": CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_CLAIM_BOUNDARY,
    },
    {
        "schema_version": CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA,
        "step_order": 2,
        "operator_phase": "refresh_vs2_scenario_report",
        "command": (
            "cornerstone scenario verify vs2-policy-tenancy-egress "
            "--reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json"
        ),
        "purpose": (
            "Refresh the local VS2 scenario report that H04 reviewers compare against the "
            "production-like environment transcript."
        ),
        "expected_report_paths": [
            "reports/security/vs2-local-security-proof.json",
            "reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json",
        ],
        "expected_report_count": 2,
        "review_input_only": True,
        "acceptance_sufficient": False,
        "product_claim_allowed": False,
        "pass_claim_allowed": False,
        "claim_boundary": CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_CLAIM_BOUNDARY,
    },
    {
        "schema_version": CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA,
        "step_order": 3,
        "operator_phase": "refresh_connectorhub_dependency_report",
        "command": "cornerstone scenario verify connector-contract-adapter --scenario CS-CH-036 --json",
        "purpose": (
            "Refresh the ConnectorHub CS-CH-036 dependency report that remains local fixture "
            "evidence until H04/H07 human proof exists."
        ),
        "expected_report_paths": [
            "reports/scenario/connector-contract-adapter-cs-ch-036-2026-06-23.json",
        ],
        "expected_report_count": 1,
        "review_input_only": True,
        "acceptance_sufficient": False,
        "product_claim_allowed": False,
        "pass_claim_allowed": False,
        "claim_boundary": CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_CLAIM_BOUNDARY,
    },
]
VS2_SCENARIO_REPORT = ROOT / "reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json"
VS2_MATRIX = ROOT / "docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_MATRIX.csv"
CONTRACT_FIXTURE = "fixtures/connectorhub/contracts/github_readonly_contract.json"
RAW_ACCESS_CONTRACT_FIXTURE = "fixtures/connectorhub/contracts/github_raw_access_contract.json"
SELECTED_REPOSITORIES_CONTRACT_FIXTURE = "fixtures/connectorhub/contracts/github_selected_repositories_contract.json"
GITHUB_WRITE_ACTION_CONTRACT_FIXTURE = "fixtures/connectorhub/contracts/github_write_action_contract.json"
MISSING_REQUIRED_CONTRACT_FIXTURE = "fixtures/connectorhub/contracts/github_required_missing_contract.json"
OPTIONAL_MISSING_CONTRACT_FIXTURE = "fixtures/connectorhub/contracts/github_optional_missing_contract.json"
DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_issue_projection_delivery.json"
PROMPT_INJECTION_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_issue_projection_delivery_prompt_injection.json"
REPOSITORY_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_repository_projection_delivery.json"
COMMIT_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_commit_projection_delivery.json"
CHANGE_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_change_projection_delivery.json"
FILE_SNAPSHOT_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery.json"
SECRET_MARKER_FILE_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery_secret_marker.json"
PRIVATE_KEY_FILE_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery_private_key.json"
BINARY_FILE_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery_binary.json"
LARGE_FILE_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery_large.json"
FORBIDDEN_PATH_FILE_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery_forbidden_path.json"
GENERATED_FILE_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_file_snapshot_projection_delivery_generated.json"
SELECTED_REPO_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_selected_repo_issue_projection_delivery.json"
UNSELECTED_REPO_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_unselected_repo_issue_projection_delivery.json"
ACTIVITY_SAMPLES_FIXTURE = "fixtures/connectorhub/activity_samples/macos_activity_samples_cs_ch_022.json"
WATCH_RULE_FIXTURE = "fixtures/connectorhub/watch_rules/project_alpha_watch_rule_cs_ch_023.json"
WATCH_RULE_EDIT_FIXTURE = "fixtures/connectorhub/watch_rules/project_alpha_watch_rule_edit_cs_ch_023.json"
CHROME_ACTIVE_TAB_ALLOWED_FIXTURE = "fixtures/connectorhub/chrome/active_tab_capture_allowed_cs_ch_024.json"
CHROME_ACTIVE_TAB_POPUP_BLOCKED_FIXTURE = "fixtures/connectorhub/chrome/active_tab_capture_popup_blocked_cs_ch_024.json"
CHROME_AUTO_CAPTURE_CONFIG_FIXTURE = "fixtures/connectorhub/chrome/auto_capture_config_cs_ch_025.json"
CHROME_AUTO_CAPTURE_ALLOWED_FIXTURE = "fixtures/connectorhub/chrome/auto_capture_allowed_cs_ch_025.json"
CHROME_AUTO_CAPTURE_BLOCKED_FIXTURE = "fixtures/connectorhub/chrome/auto_capture_blocked_cs_ch_025.json"
CHROME_SENSITIVE_PAGE_FIXTURE = "fixtures/connectorhub/chrome/sensitive_pages_cs_ch_026.json"
CAPTURE_LIFECYCLE_FIXTURE = "fixtures/connectorhub/capture/lifecycle_state_cs_ch_027.json"
WATCH_RESULT_FIXTURE = "fixtures/connectorhub/watch_results/project_alpha_watch_result_cs_ch_028.json"
ACTION_PREFLIGHT_FIXTURE = "fixtures/connectorhub/actions/non_github_action_preflight_cs_ch_029.json"
DIRECT_PROVIDER_PACK_FIXTURE = "fixtures/vs0/packs/14_extension_ecosystem/direct_provider_agent_pack_manifest.json"
POISON_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_issue_poison_projection_delivery.json"
DUPLICATE_EVENT_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_issue_projection_delivery_duplicate_event.json"
BAD_WEBHOOK_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_issue_projection_delivery_bad_webhook_signature.json"
UNCHANGED_EVENT_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_issue_projection_delivery_unchanged_event.json"
CHANGED_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_issue_projection_delivery_changed.json"
FORBIDDEN_BODY_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_issue_projection_delivery_forbidden_body.json"
OVERSIZED_EXCERPT_DELIVERY_FIXTURE = "fixtures/connectorhub/deliveries/github_issue_projection_delivery_oversized_excerpt.json"
ISSUE_SOURCE_EXTERNAL_ID = "github:repo:owner/project-alpha:issue:1001"
CONNECTORHUB_LOCAL_FIXTURE_PRODUCT_CLAIM = "LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING"
HUMAN_GATE_PERSPECTIVE_FINDINGS = {
    "product_value": "redacted product-value finding",
    "domain_architecture": "redacted domain-architecture finding",
    "data_contract": "redacted data-contract finding",
    "reliability_observability": "redacted reliability-observability finding",
    "security_privacy": "redacted security-privacy finding",
    "testability_migration": "redacted testability-migration finding",
}
HUMAN_GATE_SOURCE_REQUIREMENT_HUMAN_PENDING_IDS = [
    "ER-05",
    "ER-06",
    "IR-01",
    "IR-04",
    "IR-07",
    "IR-11",
    "IR-13",
    "IR-14",
    "IR-16",
    "IR-17",
    "IR-18",
]
HUMAN_GATE_REQUIRED_FIELD_TOTAL = sum(
    len(definition["required_human_record"]["required_fields"])
    for definition in CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS.values()
)
HUMAN_GATE_REQUIRED_EVIDENCE_TOTAL = sum(
    len(definition["required_human_record"]["required_evidence"])
    for definition in CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS.values()
)


def human_gate_evidence_packet_manifest(scenario_id: str) -> list[dict[str, object]]:
    required_evidence = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS[scenario_id]["required_human_record"][
        "required_evidence"
    ]
    return [
        {
            "required_evidence_index": index,
            "required_evidence": str(evidence),
            "evidence_ref": f"evidence:redacted-human-gate-{index}",
            "redaction_status": CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES[0],
        }
        for index, evidence in enumerate(required_evidence, start=1)
    ]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = f"{ROOT}{os.pathsep}{env.get('PATH', '')}"
    command = ["cornerstone", *args]
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=CLI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        timeout_stderr = (
            f"command timed out after {CLI_TIMEOUT_SECONDS:g}s: "
            f"{' '.join(command)}\n{stderr}"
        )
        return subprocess.CompletedProcess(command, 124, stdout, timeout_stderr)


def run_json(*args: str) -> dict:
    result = run_cli(*args, "--json")
    if result.returncode != 0:
        raise AssertionError(f"command failed: {result.args}\nstdout={result.stdout}\nstderr={result.stderr}")
    return json.loads(result.stdout)


def _load_report_if_valid(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _summary_matches_vs2_reusable_proof(payload: dict) -> bool:
    summary = payload.get("summary")
    return (
        isinstance(summary, dict)
        and payload.get("status") == "success"
        and payload.get("scenario_set") == "vs2-policy-tenancy-egress"
        and summary.get("scenario_count") == 93
        and summary.get("pass") == 86
        and summary.get("blocking") == 0
        and isinstance(payload.get("scenario_results"), list)
        and len(payload["scenario_results"]) == 93
    )


def _vs2_expected_ids_and_owners() -> tuple[set[str], dict[str, str]]:
    with VS2_MATRIX.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = {row["scenario_id"] for row in rows if row.get("scenario_id")}
    expected_owners = {
        row["scenario_id"]: "Human" if row.get("priority") == "HUMAN_REQUIRED" else "AI"
        for row in rows
        if row.get("scenario_id")
    }
    return expected_ids, expected_owners


def existing_vs2_reusable_proof_current() -> bool:
    if FORCE_VS2_REUSABLE_PROOF_REFRESH:
        return False
    proof = _load_report_if_valid(VS2_LOCAL_PROOF_REPORT)
    scenario_report = _load_report_if_valid(VS2_SCENARIO_REPORT)
    if proof is None or scenario_report is None:
        return False
    expected_ids, expected_owners = _vs2_expected_ids_and_owners()
    proof_reusable, proof_errors, _ = validate_reusable_report(
        proof,
        root=ROOT,
        family="vs2_local_proof",
        expected_schema="cs.vs2_local_security_proof.v0",
        require_status=None,
        expected_scenario_ids=expected_ids,
        expected_scenario_owners=expected_owners,
        validate_evidence=True,
    )
    return (
        proof_reusable
        and not proof_errors
        and _summary_matches_vs2_reusable_proof(proof)
        and scenario_report.get("schema_version") == "cs.cli.v0"
        and scenario_report.get("command") == "cornerstone scenario verify vs2-policy-tenancy-egress"
        and _summary_matches_vs2_reusable_proof(scenario_report)
        and scenario_report.get("local_security_proof", {}).get("proof_hash") == proof.get("proof_hash")
    )


def ensure_vs2_reusable_proof_current(test_case: unittest.TestCase) -> None:
    global VS2_REUSABLE_PROOF_REFRESHED
    if VS2_REUSABLE_PROOF_REFRESHED:
        return
    if existing_vs2_reusable_proof_current():
        VS2_REUSABLE_PROOF_REFRESHED = True
        return
    proof = run_cli("security", "vs2-local-proof", "--json")
    test_case.assertEqual(proof.returncode, 0, proof.stdout + proof.stderr)
    scenario_report = run_cli(
        "scenario",
        "verify",
        "vs2-policy-tenancy-egress",
        "--reuse-vs2-local-proof-report",
        "reports/security/vs2-local-security-proof.json",
        "--json",
        "--output",
        "reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json",
    )
    test_case.assertEqual(scenario_report.returncode, 0, scenario_report.stdout + scenario_report.stderr)
    VS2_REUSABLE_PROOF_REFRESHED = True


def state_file_texts(state_dir: Path) -> str:
    parts: list[str] = []
    if not state_dir.exists():
        return ""
    for path in sorted(state_dir.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".txt", ".md"}:
            parts.append(path.read_text(errors="ignore"))
    return "\n".join(parts)


class ConnectorHubCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state_rel = f"tmp/test-connectorhub-cli-{os.getpid()}-{self._testMethodName}"
        self.state_dir = ROOT / self.state_rel
        self.record_dir = ROOT / f"tmp/test-connectorhub-records-{os.getpid()}-{self._testMethodName}"
        shutil.rmtree(self.state_dir, ignore_errors=True)
        shutil.rmtree(self.record_dir, ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.state_dir, ignore_errors=True)
        shutil.rmtree(self.record_dir, ignore_errors=True)

    def assert_human_gate_redaction_guidance(
        self,
        guidance: dict,
        scenario_id: str,
        *,
        required_fields: list[str],
        required_evidence: list[str],
        dependencies: list[str],
    ) -> None:
        self.assertEqual(guidance["schema_version"], "cs.connector_human_gate_redaction_guidance.v1")
        self.assertEqual(guidance["scenario_id"], scenario_id)
        self.assertEqual(guidance["status"], "operator_guidance_only")
        self.assertFalse(guidance["raw_secret_values_allowed"])
        self.assertFalse(guidance["raw_provider_payloads_allowed"])
        self.assertFalse(guidance["raw_evidence_values_allowed"])
        self.assertFalse(guidance["raw_record_body_persisted_by_validator"])
        self.assertFalse(guidance["raw_record_path_persisted_by_validator"])
        self.assertTrue(guidance["sensitive_marker_scan_required_by_validator"])
        self.assertEqual(
            guidance["allowed_redaction_statuses"],
            CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES,
        )
        sensitive_marker_policy = guidance["sensitive_marker_policy"]
        self.assertEqual(sensitive_marker_policy["schema_version"], CONNECTOR_SENSITIVE_MARKER_POLICY_SCHEMA)
        self.assertEqual(sensitive_marker_policy["marker_types"], CONNECTOR_SENSITIVE_MARKER_TYPES)
        self.assertEqual(sensitive_marker_policy["finding_fields"], CONNECTOR_SENSITIVE_MARKER_FINDING_FIELDS)
        self.assertFalse(sensitive_marker_policy["raw_match_values_returned"])
        self.assertFalse(sensitive_marker_policy["raw_match_values_persisted"])
        self.assertTrue(sensitive_marker_policy["fingerprints_only"])
        self.assertEqual(sensitive_marker_policy["validator_structural_error"], "sensitive_marker_detected")
        self.assertIn("redacted evidence reference", sensitive_marker_policy["operator_action"])
        self.assertEqual(set(guidance["field_guidance"]), set(required_fields))
        for field in required_fields:
            self.assertTrue(guidance["field_guidance"][field]["required"])
            self.assertFalse(guidance["field_guidance"][field]["raw_secret_values_allowed"])
            self.assertFalse(guidance["field_guidance"][field]["raw_provider_payloads_allowed"])
            if scenario_id == "CS-CH-H04" and field in {
                item["field"] for item in CONNECTOR_HUMAN_GATE_H04_FIELD_REF_ITEMS
            }:
                contract_item = next(
                    item for item in CONNECTOR_HUMAN_GATE_H04_FIELD_REF_ITEMS if item["field"] == field
                )
                self.assertEqual(
                    guidance["field_guidance"][field]["accepted_value_shape"],
                    "typed redacted evidence reference matching field_ref_contract",
                )
                self.assertEqual(
                    guidance["field_guidance"][field]["accepted_container"],
                    contract_item["accepted_container"],
                )
                self.assertEqual(
                    guidance["field_guidance"][field]["accepted_ref_prefixes"],
                    contract_item["accepted_ref_prefixes"],
                )
                self.assertFalse(guidance["field_guidance"][field]["raw_value_persisted_by_validator"])
        if scenario_id == "CS-CH-H04":
            self.assert_h04_field_ref_contract(guidance["field_ref_contract"])
        else:
            self.assertNotIn("field_ref_contract", guidance)
        self.assertEqual(
            set(guidance["senior_review_perspective_findings"]["required_roles"]),
            set(HUMAN_GATE_PERSPECTIVE_FINDINGS),
        )
        self.assertFalse(guidance["senior_review_perspective_findings"]["raw_secret_values_allowed"])
        self.assertFalse(guidance["senior_review_perspective_findings"]["persisted_by_validator"])
        self.assertEqual(
            guidance["evidence_packet_manifest"]["required_evidence_count"],
            len(required_evidence),
        )
        self.assertEqual(guidance["evidence_packet_manifest"]["required_evidence_labels"], required_evidence)
        self.assertTrue(guidance["evidence_packet_manifest"]["redaction_status_required"])
        self.assertTrue(guidance["evidence_packet_manifest"]["evidence_ref_uniqueness_required"])
        self.assertEqual(
            guidance["evidence_packet_manifest"]["allowed_redaction_statuses"],
            CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES,
        )
        self.assertFalse(guidance["evidence_packet_manifest"]["values_persisted_by_validator"])
        self.assertEqual(guidance["dependency_human_gate_refs"]["required_gates"], dependencies)
        self.assertEqual(
            guidance["dependency_human_gate_refs"]["accepted_ref_prefix"],
            "connector_human_gate_record_validation:",
        )
        self.assertTrue(
            guidance["dependency_human_gate_refs"]["must_reference_structurally_valid_accept_validation"]
        )
        self.assertIn("Do not paste raw secrets", guidance["operator_warning"])

    def assert_human_gate_reviewer_checklist(
        self,
        checklist: dict,
        scenario_id: str,
        *,
        required_fields: list[str],
        required_evidence: list[str],
        dependencies: list[str],
        record_template_output_command: str,
        validation_output_command: str,
    ) -> None:
        self.assertEqual(checklist["schema_version"], "cs.connector_human_gate_reviewer_checklist.v1")
        self.assertEqual(checklist["scenario_id"], scenario_id)
        self.assertEqual(checklist["status"], "operator_preparation_only")
        self.assertFalse(checklist["product_claim_allowed_by_checklist"])
        self.assertFalse(checklist["pass_claim_allowed_by_checklist"])
        self.assertTrue(checklist["reviewer_record_validation_required"])
        self.assertEqual(checklist["record_template_output_command"], record_template_output_command)
        self.assertEqual(checklist["validation_output_command"], validation_output_command)
        self.assertEqual(
            [item["field"] for item in checklist["required_field_items"]],
            required_fields,
        )
        self.assertTrue(all(item["required"] for item in checklist["required_field_items"]))
        self.assertTrue(
            all(not item["raw_secret_values_allowed"] for item in checklist["required_field_items"])
        )
        field_items = {item["field"]: item for item in checklist["required_field_items"]}
        if scenario_id == "CS-CH-H04":
            self.assert_h04_field_ref_contract(checklist["field_ref_contract"])
            for contract_item in CONNECTOR_HUMAN_GATE_H04_FIELD_REF_ITEMS:
                field_item = field_items[contract_item["field"]]
                self.assertEqual(
                    field_item["accepted_value_shape"],
                    "typed redacted evidence reference matching field_ref_contract",
                )
                self.assertEqual(field_item["accepted_container"], contract_item["accepted_container"])
                self.assertEqual(field_item["accepted_ref_prefixes"], contract_item["accepted_ref_prefixes"])
                self.assertFalse(field_item["raw_value_persisted_by_validator"])
        else:
            self.assertNotIn("field_ref_contract", checklist)
        if scenario_id in {"CS-CH-H01", "CS-CH-H02", "CS-CH-H03", "CS-CH-H04", "CS-CH-H05", "CS-CH-H06", "CS-CH-H07"}:
            self.assert_evidence_packet_workflow(checklist["evidence_packet_workflow"], scenario_id)
            self.assertEqual(checklist["evidence_packet_workflow_command_count"], 6)
            expected_claim_boundary = {
                "CS-CH-H01": CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
                "CS-CH-H02": CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
                "CS-CH-H03": CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
                "CS-CH-H04": CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
                "CS-CH-H05": CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
                "CS-CH-H06": CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
                "CS-CH-H07": CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
            }[scenario_id]
            self.assertEqual(
                checklist["evidence_packet_workflow_claim_boundary"],
                expected_claim_boundary,
            )
        else:
            self.assertNotIn("evidence_packet_workflow", checklist)
            self.assertNotIn("evidence_packet_workflow_command_count", checklist)
            self.assertNotIn("evidence_packet_workflow_claim_boundary", checklist)
        self.assertEqual(
            {item["role"] for item in checklist["senior_review_perspective_items"]},
            set(HUMAN_GATE_PERSPECTIVE_FINDINGS),
        )
        self.assertTrue(all(item["required"] for item in checklist["senior_review_perspective_items"]))
        self.assertTrue(
            all(not item["persisted_by_validator"] for item in checklist["senior_review_perspective_items"])
        )
        self.assertEqual(
            [item["required_evidence"] for item in checklist["evidence_packet_manifest_items"]],
            required_evidence,
        )
        self.assertEqual(
            [item["required_evidence_index"] for item in checklist["evidence_packet_manifest_items"]],
            list(range(1, len(required_evidence) + 1)),
        )
        self.assertTrue(
            all(item["evidence_ref_required"] for item in checklist["evidence_packet_manifest_items"])
        )
        self.assertTrue(
            all(item["evidence_ref_uniqueness_required"] for item in checklist["evidence_packet_manifest_items"])
        )
        self.assertEqual(
            {tuple(item["allowed_redaction_statuses"]) for item in checklist["evidence_packet_manifest_items"]},
            {tuple(CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES)},
        )
        self.assertEqual(
            [item["scenario_id"] for item in checklist["dependency_human_gate_ref_items"]],
            dependencies,
        )
        self.assertTrue(
            all(
                item["accepted_ref_prefix"] == "connector_human_gate_record_validation:"
                for item in checklist["dependency_human_gate_ref_items"]
            )
        )
        self.assertIn("does not collect approval", checklist["completion_rule"])

    def assert_human_gate_delivery_unit_plan(
        self,
        plan: dict,
        scenario_id: str,
        *,
        dependencies: list[str],
        required_fields: list[str],
        required_evidence: list[str],
        package_command: str,
        record_template_output_command: str,
        validation_command: str,
        validation_output_command: str,
    ) -> None:
        self.assertEqual(plan["schema_version"], "cs.connector_human_gate_delivery_unit_plan.v1")
        self.assertEqual(plan["scenario_id"], scenario_id)
        self.assertEqual(plan["status"], "operator_preparation_only")
        self.assertEqual(plan["current_verdict"], "HUMAN_REQUIRED")
        self.assertTrue(plan["scenario_first_independent_delivery_unit"])
        self.assertFalse(plan["product_claim_allowed_by_plan"])
        self.assertFalse(plan["pass_claim_allowed_by_plan"])
        self.assertFalse(plan["approval_collected_by_plan"])
        self.assertFalse(plan["dependency_unlock_allowed_by_plan"])
        self.assertEqual(plan["depends_on_human_gates"], dependencies)
        self.assertEqual(
            {item["role"] for item in plan["senior_review_perspective_sequence"]},
            set(HUMAN_GATE_PERSPECTIVE_FINDINGS),
        )
        self.assertTrue(
            all(item["required_before_human_acceptance"] for item in plan["senior_review_perspective_sequence"])
        )
        self.assertEqual(
            [item["phase"] for item in plan["lifecycle_steps"]],
            [
                "research_from_senior_perspectives",
                "define_implementation_approach",
                "execute_smallest_complete_rehearsal",
                "refactor_or_remediate_before_acceptance",
                "verify_record_structure",
                "document_scenario_result",
                "move_to_next_gate_only_after_dependency_rule",
            ],
        )
        self.assertEqual([item["step_order"] for item in plan["lifecycle_steps"]], list(range(1, 8)))
        self.assertEqual(plan["command_sequence"]["package"], package_command)
        self.assertEqual(plan["command_sequence"]["record_template_output"], record_template_output_command)
        self.assertEqual(plan["command_sequence"]["validate_record"], validation_command)
        self.assertEqual(plan["command_sequence"]["validate_record_output"], validation_output_command)
        self.assertEqual(plan["command_sequence"]["readiness_report"], "cornerstone connector human-gate report --json")
        self.assertEqual(plan["command_sequence"]["next_selector"], "cornerstone connector human-gate next --json")
        if scenario_id in {"CS-CH-H01", "CS-CH-H02", "CS-CH-H03", "CS-CH-H04", "CS-CH-H05", "CS-CH-H06", "CS-CH-H07"}:
            self.assert_evidence_packet_workflow(plan["evidence_packet_workflow"], scenario_id)
            self.assertEqual(plan["evidence_packet_workflow_command_count"], 6)
            expected_claim_boundary = {
                "CS-CH-H01": CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
                "CS-CH-H02": CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
                "CS-CH-H03": CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
                "CS-CH-H04": CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
                "CS-CH-H05": CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
                "CS-CH-H06": CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
                "CS-CH-H07": CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
            }[scenario_id]
            self.assertEqual(
                plan["evidence_packet_workflow_claim_boundary"],
                expected_claim_boundary,
            )
            self.assertEqual(
                plan["command_sequence"]["evidence_packet_workflow"],
                plan["evidence_packet_workflow"]["commands"],
            )
        else:
            self.assertNotIn("evidence_packet_workflow", plan)
            self.assertNotIn("evidence_packet_workflow_command_count", plan)
            self.assertNotIn("evidence_packet_workflow_claim_boundary", plan)
            self.assertNotIn("evidence_packet_workflow", plan["command_sequence"])
        self.assertEqual(plan["required_human_record_summary"]["required_fields"], required_fields)
        self.assertEqual(plan["required_human_record_summary"]["required_evidence"], required_evidence)
        self.assertEqual(plan["required_human_record_summary"]["allowed_decision_values"], ["ACCEPT", "REJECT"])
        self.assertEqual(plan["dependency_rule"]["required_dependency_human_gates"], dependencies)
        self.assertEqual(
            plan["dependency_rule"]["accepted_ref_prefix"],
            "connector_human_gate_record_validation:",
        )
        self.assertTrue(plan["dependency_rule"]["only_structurally_valid_accept_records_unlock_dependencies"])
        self.assertTrue(plan["dependency_rule"]["structurally_valid_reject_records_do_not_unlock_dependencies"])
        self.assertIn("preparation metadata only", plan["documentation_rule"])

    def assert_human_gate_delivery_unit_plan_summary(self, summary: dict) -> None:
        self.assertEqual(summary["schema_version"], "cs.connector_human_gate_delivery_unit_plan_summary.v1")
        self.assertEqual(
            summary["scenario_delivery_unit_plan_schema_version"],
            "cs.connector_human_gate_delivery_unit_plan.v1",
        )
        self.assertTrue(summary["scenario_delivery_unit_plan_ready"])
        self.assertEqual(summary["scenario_delivery_unit_plan_lifecycle_step_count"], 7)
        self.assertGreaterEqual(summary["scenario_delivery_unit_plan_senior_review_perspective_count"], 6)
        self.assertFalse(summary["scenario_delivery_unit_plan_product_claim_allowed"])
        self.assertFalse(summary["scenario_delivery_unit_plan_pass_claim_allowed"])
        self.assertFalse(summary["scenario_delivery_unit_plan_approval_collected"])
        self.assertFalse(summary["scenario_delivery_unit_plan_dependency_unlock_allowed"])

    def assert_human_gate_completion_boundary(self, payload: dict) -> None:
        self.assertTrue(payload["goal_completion_claim_blocked"])
        self.assertFalse(payload["full_goal_completion_allowed"])
        self.assertEqual(
            payload["completion_claim_boundary"],
            CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
        )

    def assert_h04_field_ref_contract(self, contract: dict) -> None:
        self.assertEqual(contract["schema_version"], CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_SCHEMA)
        self.assertEqual(contract["scenario_id"], "CS-CH-H04")
        self.assertEqual(contract["status"], "operator_preparation_only")
        self.assertEqual(contract["validation_scope"], "field_reference_shape_only")
        self.assertFalse(contract["raw_field_values_persisted_by_validator"])
        self.assertEqual(contract["invalid_value_report_shape"], "field_names_only")
        self.assertEqual(
            [
                {
                    "field": item["field"],
                    "accepted_container": item["accepted_container"],
                    "accepted_ref_prefixes": item["accepted_ref_prefixes"],
                }
                for item in contract["required_field_ref_items"]
            ],
            CONNECTOR_HUMAN_GATE_H04_FIELD_REF_ITEMS,
        )
        self.assertTrue(all(item["required"] for item in contract["required_field_ref_items"]))
        self.assertTrue(
            all(
                item["raw_value_persisted_by_validator"] is False
                for item in contract["required_field_ref_items"]
            )
        )

    def assert_evidence_packet_workflow(self, workflow: dict, scenario_id: str) -> None:
        if scenario_id == "CS-CH-H01":
            expected_schema = CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_WORKFLOW_SCHEMA
            expected_packet_dir = CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY
            expected_claim_boundary = CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY
        elif scenario_id == "CS-CH-H02":
            expected_schema = CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_WORKFLOW_SCHEMA
            expected_packet_dir = CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY
            expected_claim_boundary = CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY
        elif scenario_id == "CS-CH-H03":
            expected_schema = CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_WORKFLOW_SCHEMA
            expected_packet_dir = CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY
            expected_claim_boundary = CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY
        elif scenario_id == "CS-CH-H04":
            expected_schema = CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_WORKFLOW_SCHEMA
            expected_packet_dir = CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY
            expected_claim_boundary = CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY
        elif scenario_id == "CS-CH-H05":
            expected_schema = CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_WORKFLOW_SCHEMA
            expected_packet_dir = CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY
            expected_claim_boundary = CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY
        elif scenario_id == "CS-CH-H06":
            expected_schema = CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_WORKFLOW_SCHEMA
            expected_packet_dir = CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY
            expected_claim_boundary = CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY
        elif scenario_id == "CS-CH-H07":
            expected_schema = CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_WORKFLOW_SCHEMA
            expected_packet_dir = CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY
            expected_claim_boundary = CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY
        else:
            self.fail(f"Unsupported evidence-packet workflow scenario: {scenario_id}")
        self.assertEqual(
            workflow["schema_version"],
            expected_schema,
        )
        self.assertEqual(workflow["scenario_id"], scenario_id)
        self.assertEqual(workflow["status"], "operator_preparation_only")
        self.assertEqual(
            workflow["packet_dir_placeholder"],
            expected_packet_dir,
        )
        self.assertEqual(workflow["record_output_placeholder"], "<reviewer-record-draft.json>")
        self.assertEqual(workflow["validation_output_placeholder"], "<redacted-validation-envelope.json>")
        self.assertTrue(workflow["review_input_only"])
        self.assertFalse(workflow["acceptance_sufficient"])
        self.assertFalse(workflow["product_claim_allowed"])
        self.assertFalse(workflow["pass_claim_allowed"])
        self.assertFalse(workflow["dependency_unlock_allowed_by_workflow"])
        self.assertFalse(workflow["human_acceptance_collected_by_workflow"])
        self.assertFalse(workflow["raw_packet_file_contents_recorded_by_workflow"])
        self.assertFalse(workflow["packet_file_contents_persisted_by_workflow"])
        self.assertEqual(
            workflow["claim_boundary"],
            expected_claim_boundary,
        )
        expected_commands = [
            f"cornerstone connector human-gate evidence-packet-contract --scenario {scenario_id} --json",
            f"cornerstone connector human-gate evidence-packet-file-contract --scenario {scenario_id} --json",
            (
                f"cornerstone connector human-gate evidence-packet-scaffold --scenario {scenario_id} "
                f"--packet-dir {expected_packet_dir} --json --write"
            ),
            (
                f"cornerstone connector human-gate evidence-packet-validate --scenario {scenario_id} "
                f"--packet-dir {expected_packet_dir} --json"
            ),
            (
                f"cornerstone connector human-gate evidence-packet-record-draft --scenario {scenario_id} "
                f"--packet-dir {expected_packet_dir} --json --record-output <reviewer-record-draft.json>"
            ),
            (
                f"cornerstone connector human-gate validate-record --scenario {scenario_id} "
                "--record-file <filled-reviewer-record.json> --json "
                "--output <redacted-validation-envelope.json>"
            ),
        ]
        self.assertEqual(workflow["commands"], expected_commands)
        self.assertEqual(workflow["command_count"], len(expected_commands))
        self.assertEqual(
            [row["command"] for row in workflow["command_sequence"]],
            expected_commands,
        )
        self.assertEqual(
            [row["step_order"] for row in workflow["command_sequence"]],
            list(range(1, len(expected_commands) + 1)),
        )
        self.assertEqual(
            [row["phase"] for row in workflow["command_sequence"]],
            [
                "inspect_evidence_packet_contract",
                "inspect_evidence_packet_file_contract",
                "scaffold_packet_templates",
                "validate_packet_hashes",
                "draft_record_from_packet_hashes",
                "validate_completed_reviewer_record",
            ],
        )

    def assert_h04_evidence_packet_workflow(self, workflow: dict) -> None:
        self.assert_evidence_packet_workflow(workflow, "CS-CH-H04")

    def assert_h01_evidence_packet_workflow(self, workflow: dict) -> None:
        self.assert_evidence_packet_workflow(workflow, "CS-CH-H01")

    def assert_h02_evidence_packet_workflow(self, workflow: dict) -> None:
        self.assert_evidence_packet_workflow(workflow, "CS-CH-H02")

    def assert_h03_evidence_packet_workflow(self, workflow: dict) -> None:
        self.assert_evidence_packet_workflow(workflow, "CS-CH-H03")

    def assert_h05_evidence_packet_workflow(self, workflow: dict) -> None:
        self.assert_evidence_packet_workflow(workflow, "CS-CH-H05")

    def assert_h06_evidence_packet_workflow(self, workflow: dict) -> None:
        self.assert_evidence_packet_workflow(workflow, "CS-CH-H06")

    def assert_h07_evidence_packet_workflow(self, workflow: dict) -> None:
        self.assert_evidence_packet_workflow(workflow, "CS-CH-H07")

    def assert_h04_remaining_evidence_summary_workflow(self, summary: dict) -> None:
        self.assert_evidence_packet_workflow(
            summary["evidence_packet_workflow"],
            summary["scenario_id"],
        )
        self.assertEqual(summary["evidence_packet_workflow_command_count"], 6)
        self.assertEqual(
            summary["evidence_packet_workflow_commands"],
            summary["evidence_packet_workflow"]["commands"],
        )
        expected_claim_boundary = {
            "CS-CH-H01": CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
            "CS-CH-H02": CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
            "CS-CH-H03": CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
            "CS-CH-H04": CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
            "CS-CH-H05": CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
            "CS-CH-H06": CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
            "CS-CH-H07": CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
        }[summary["scenario_id"]]
        self.assertEqual(summary["evidence_packet_workflow_claim_boundary"], expected_claim_boundary)
        self.assertFalse(summary["dependency_unlock_allowed_by_evidence_packet_workflow"])
        self.assertFalse(summary["human_acceptance_collected_by_evidence_packet_workflow"])
        self.assertFalse(summary["raw_packet_file_contents_recorded_by_evidence_packet_workflow"])
        self.assertFalse(summary["packet_file_contents_persisted_by_evidence_packet_workflow"])

    def assert_h04_local_baseline_preflight_bundle(
        self,
        bundle: dict,
        report_paths: list[str],
        report_rows: dict[str, dict],
    ) -> None:
        self.assertEqual(
            bundle["schema_version"],
            CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_BUNDLE_SCHEMA,
        )
        self.assertEqual(bundle["scenario_id"], "CS-CH-H04")
        self.assertEqual(bundle["status"], "operator_preparation_only")
        self.assertEqual(bundle["baseline_scope"], "local_ai_verifiable_vs2_and_connectorhub_dependency_proof")
        self.assertTrue(bundle["review_input_only"])
        self.assertFalse(bundle["acceptance_sufficient"])
        self.assertFalse(bundle["product_claim_allowed"])
        self.assertFalse(bundle["pass_claim_allowed"])
        self.assertEqual(bundle["claim_boundary"], CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_BUNDLE_CLAIM_BOUNDARY)
        self.assertEqual(bundle["required_human_delta_count"], 5)
        self.assertEqual(bundle["command_plan_schema_version"], CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA)
        self.assertEqual(bundle["command_plan_count"], 3)
        self.assertEqual(bundle["command_plan"], EXPECTED_H04_PREFLIGHT_COMMAND_PLAN)
        self.assertEqual(
            bundle["recommended_preflight_command_plan_schema_version"],
            CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA,
        )
        self.assertEqual(bundle["recommended_preflight_command_plan_count"], 3)
        self.assertEqual(bundle["recommended_preflight_command_plan"], EXPECTED_H04_PREFLIGHT_COMMAND_PLAN)
        self.assertEqual(bundle["recommended_preflight_commands"], [row["command"] for row in EXPECTED_H04_PREFLIGHT_COMMAND_PLAN])
        self.assertEqual(bundle["current_report_count"], 5)
        self.assertEqual(bundle["current_report_paths"], report_paths)
        self.assertEqual(bundle["ready_report_count"], 5)
        self.assertEqual(bundle["ready_report_paths"], report_paths)
        self.assertEqual(bundle["missing_reports"], [])
        self.assertEqual(bundle["invalid_json_reports"], [])
        self.assertTrue(bundle["all_reports_present"])
        self.assertTrue(bundle["all_reports_json_valid"])
        self.assertEqual(
            bundle["command_plan_expected_report_paths"],
            sorted(report_paths),
        )
        self.assertEqual(bundle["command_plan_expected_report_path_count"], 5)
        self.assertEqual(bundle["command_plan_paths_covered_by_current_reports"], sorted(report_paths))
        self.assertEqual(bundle["command_plan_paths_missing_from_current_reports"], [])
        self.assertEqual(bundle["commands_executed_by_bundle"], 0)
        self.assertEqual(bundle["live_provider_calls_executed_by_bundle"], 0)
        self.assertEqual(bundle["provider_mutations_executed_by_bundle"], 0)
        self.assertEqual(bundle["external_mutations_executed_by_bundle"], 0)
        self.assertFalse(bundle["human_acceptance_collected_by_bundle"])
        self.assertIn("local comparison input", bundle["operator_next_step"])
        fingerprints = {row["path"]: row for row in bundle["current_report_fingerprints"]}
        self.assertEqual(set(fingerprints), set(report_paths))
        for path in report_paths:
            self.assertTrue(fingerprints[path]["present"])
            self.assertTrue(fingerprints[path]["json_valid"])
            self.assertEqual(fingerprints[path]["sha256"], report_rows[path]["sha256"])
            self.assertTrue(fingerprints[path]["review_input_only"])
            self.assertFalse(fingerprints[path]["acceptance_sufficient"])
            self.assertFalse(fingerprints[path]["product_claim_allowed"])
            self.assertFalse(fingerprints[path]["pass_claim_allowed"])
            self.assertEqual(
                fingerprints[path]["claim_boundary"],
                CONNECTOR_HUMAN_GATE_H04_BASELINE_CLAIM_BOUNDARY,
            )

    def test_connector_contract_validate_and_setup_plan_cs_ch_001(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        contract = validate["connector_capability_contract"]
        self.assertEqual(contract["schema_version"], "cs.connector_capability_contract.v1")
        self.assertEqual(contract["status"], "draft_validated")
        self.assertEqual(contract["contract_id"], "ccon_project_alpha_github")
        self.assertFalse(contract["connector_port"]["product_depends_on_provider_sdk"])
        self.assertEqual(validate["provider_internal_findings"], [])
        self.assertTrue(validate["audit_refs"])

        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")
        setup = plan["connector_setup_result"]
        policy = plan["connector_source_policy"]
        self.assertEqual(setup["schema_version"], "cs.connector_setup_result.v1")
        self.assertEqual(setup["readiness"], "ready")
        self.assertTrue(setup["activation_allowed"])
        self.assertTrue(setup["required_capabilities_available"])
        self.assertEqual(setup["blocked_reason_code"], None)
        self.assertTrue(setup["delivery_streams"])
        self.assertTrue(all(item["enabled"] for item in setup["feature_availability"]))
        self.assertEqual(setup["disabled_surfaces"], [])
        self.assertEqual(setup["provider_call_ledger"]["before_activation"], 0)
        self.assertEqual(setup["provider_call_ledger"]["during_plan"], 0)
        self.assertIn(f"fixture:{CONTRACT_FIXTURE}", setup["verification_refs"])
        self.assertEqual(policy["raw_access"], "denied")
        self.assertEqual(plan["provider_internal_findings"], [])
        self.assertTrue(plan["evidence_refs"])
        self.assertTrue(plan["audit_refs"])

        setup_path = self.state_dir / "connector" / "setup_results" / f"{setup['setup_result_id']}.json"
        policy_path = self.state_dir / "connector" / "source_policies" / f"{policy['source_policy_id']}.json"
        self.assertTrue(setup_path.exists())
        self.assertTrue(policy_path.exists())

    def test_connector_setup_plan_blocks_missing_required_capability_cs_ch_002(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            MISSING_REQUIRED_CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        self.assertEqual(validate["ids"]["contract_id"], "ccon_project_alpha_missing_required")

        result = run_cli(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_missing_required",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["status"], "blocked")
        self.assertIn("CS_CONNECTOR_REQUIRED_CAPABILITY_MISSING", {error["code"] for error in plan["errors"]})
        setup = plan["connector_setup_result"]
        self.assertEqual(setup["readiness"], "blocked")
        self.assertEqual(setup["activation_state"], "blocked_required_capability_missing")
        self.assertFalse(setup["activation_allowed"])
        self.assertFalse(setup["required_capabilities_available"])
        self.assertEqual(setup["blocked_reason_code"], "CS_CONNECTOR_REQUIRED_CAPABILITY_MISSING")
        self.assertEqual(setup["delivery_streams"], [])
        self.assertEqual(setup["provider_call_ledger"]["before_activation"], 0)
        self.assertEqual(setup["provider_call_ledger"]["during_plan"], 0)
        self.assertIn(f"fixture:{MISSING_REQUIRED_CONTRACT_FIXTURE}", setup["verification_refs"])
        required_gaps = [gap for gap in setup["gaps"] if gap["required"]]
        self.assertEqual(len(required_gaps), 1)
        self.assertEqual(required_gaps[0]["common_capability"], "source_control.pull_request.read")
        self.assertEqual(required_gaps[0]["reason_code"], "CS_CONNECTOR_CAPABILITY_UNAVAILABLE")
        self.assertIn("compatible Provider Pack", required_gaps[0]["resolution"])
        self.assertIn("compatible Provider Pack", setup["activation_guidance"])
        self.assertEqual(plan["provider_internal_findings"], [])
        self.assertTrue(plan["evidence_refs"])
        self.assertTrue(plan["audit_refs"])

        setup_path = self.state_dir / "connector" / "setup_results" / f"{setup['setup_result_id']}.json"
        self.assertTrue(setup_path.exists())

    def test_connector_delivery_ingest_archives_projection_as_artifact_cs_ch_007(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        source_policy = plan["connector_source_policy"]
        ingest = run_json(
            "connector",
            "delivery",
            "ingest",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(ingest["status"], "success")
        artifact = ingest["artifact"]
        receipt = ingest["connector_delivery_receipt"]
        snapshot = ingest["connector_projection_snapshot"]
        link = ingest["connector_evidence_link"]
        delivery_bytes = (ROOT / DELIVERY_FIXTURE).read_bytes()
        delivery_checksum = hashlib.sha256(delivery_bytes).hexdigest()

        self.assertEqual(receipt["schema_version"], "cs.connector_delivery_receipt.v1")
        self.assertEqual(receipt["commit_state"], "artifact_committed")
        self.assertEqual(receipt["acknowledgement_state"], "not_acknowledged_by_cs_ch_007")
        self.assertFalse(receipt["product_interpretation"]["before_archive_commit"])
        self.assertTrue(receipt["product_interpretation"]["handlers_receive_committed_artifact_only"])
        self.assertEqual(receipt["source_policy_id"], source_policy["source_policy_id"])
        self.assertEqual(receipt["artifact_id"], artifact["artifact_id"])
        self.assertEqual(receipt["envelope_sha256"], delivery_checksum)
        self.assertEqual(artifact["checksum_sha256"], delivery_checksum)
        self.assertEqual(artifact["source"]["type"], "connector_projection_delivery")
        self.assertTrue(artifact["connector_delivery"]["exact_envelope_preserved"])
        self.assertFalse(artifact["connector_delivery"]["raw_provider_payload_stored_in_product_state"])
        self.assertEqual(artifact["connector_delivery"]["source_policy_id"], source_policy["source_policy_id"])
        self.assertEqual(snapshot["artifact_id"], artifact["artifact_id"])
        self.assertEqual(snapshot["envelope_sha256"], delivery_checksum)
        self.assertEqual(link["artifact_id"], artifact["artifact_id"])
        self.assertEqual(link["delivery_receipt_id"], receipt["delivery_receipt_id"])
        self.assertEqual(link["evidence_ref"]["evidence_ref_id"], "eref_project_alpha_issue_1001")
        self.assertEqual(ingest["provider_internal_findings"], [])
        self.assertTrue(ingest["evidence_refs"])
        self.assertTrue(ingest["audit_refs"])

        original_path = self.state_dir / "artifacts" / "originals" / delivery_checksum
        receipt_path = self.state_dir / "connector" / "delivery_receipts" / f"{receipt['delivery_receipt_id']}.json"
        snapshot_path = self.state_dir / "connector" / "projection_snapshots" / f"{snapshot['projection_snapshot_id']}.json"
        link_path = self.state_dir / "connector" / "evidence_links" / f"{link['evidence_link_id']}.json"
        self.assertTrue(original_path.exists())
        self.assertEqual(original_path.read_bytes(), delivery_bytes)
        self.assertTrue(receipt_path.exists())
        self.assertTrue(snapshot_path.exists())
        self.assertTrue(link_path.exists())

        shown = run_json(
            "artifact",
            "show",
            artifact["artifact_id"],
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(shown["status"], "success")
        self.assertEqual(shown["artifact"]["connector_delivery"]["delivery_id"], receipt["delivery_id"])

    def test_connector_delivery_process_acks_only_after_durable_commit_cs_ch_008(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")

        before_commit = run_cli(
            "connector",
            "delivery",
            "process",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--fault-mode",
            "before_commit",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(before_commit.returncode, 5, before_commit.stdout + before_commit.stderr)
        before_payload = json.loads(before_commit.stdout)
        self.assertEqual(before_payload["status"], "interrupted")
        self.assertEqual(before_payload["crash_point"], "before_commit")
        self.assertFalse(before_payload["acknowledgement"]["ack_sent"])
        self.assertFalse((self.state_dir / "connector" / "delivery_receipts").exists())
        self.assertFalse((self.state_dir / "connector" / "ack_outbox").exists())

        after_commit = run_cli(
            "connector",
            "delivery",
            "process",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--fault-mode",
            "after_commit_before_ack",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(after_commit.returncode, 5, after_commit.stdout + after_commit.stderr)
        after_payload = json.loads(after_commit.stdout)
        self.assertEqual(after_payload["status"], "interrupted")
        self.assertEqual(after_payload["crash_point"], "after_commit_before_ack")
        self.assertFalse(after_payload["acknowledgement"]["ack_sent"])
        self.assertTrue(after_payload["acknowledgement"]["durable_commit_completed"])
        pending_receipt = after_payload["connector_delivery_receipt"]
        pending_outbox = after_payload["connector_ack_outbox"]
        self.assertEqual(pending_receipt["acknowledgement_state"], "pending_after_commit")
        self.assertEqual(pending_outbox["status"], "pending")
        self.assertFalse(pending_outbox["ack_sent"])

        redelivery = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(redelivery["status"], "success")
        self.assertTrue(redelivery["acknowledgement"]["ack_sent"])
        self.assertTrue(redelivery["acknowledgement"]["durable_commit_completed"])
        self.assertFalse(redelivery["acknowledgement"]["acknowledged_without_artifact"])
        self.assertFalse(redelivery["acknowledgement"]["duplicate_downstream_effect"])
        acked_receipt = redelivery["connector_delivery_receipt"]
        acked_outbox = redelivery["connector_ack_outbox"]
        self.assertEqual(acked_receipt["delivery_receipt_id"], pending_receipt["delivery_receipt_id"])
        self.assertEqual(acked_receipt["artifact_id"], pending_receipt["artifact_id"])
        self.assertEqual(acked_receipt["acknowledgement_state"], "acknowledged_after_commit")
        self.assertEqual(acked_outbox["status"], "acknowledged")
        self.assertTrue(acked_outbox["ack_sent"])
        self.assertEqual(len(acked_outbox["attempts"]), 1)

        duplicate = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(duplicate["status"], "success")
        self.assertTrue(duplicate["acknowledgement"]["replayed"])
        self.assertFalse(duplicate["acknowledgement"]["duplicate_downstream_effect"])
        self.assertEqual(duplicate["connector_delivery_receipt"]["delivery_receipt_id"], pending_receipt["delivery_receipt_id"])
        self.assertEqual(duplicate["connector_delivery_receipt"]["artifact_id"], pending_receipt["artifact_id"])

        receipts = list((self.state_dir / "connector" / "delivery_receipts").glob("*.json"))
        outboxes = list((self.state_dir / "connector" / "ack_outbox").glob("*.json"))
        artifact_records = list((self.state_dir / "artifacts" / "records").glob(f"**/{pending_receipt['artifact_id']}.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(len(outboxes), 1)
        self.assertEqual(len(artifact_records), 1)

        reconcile = run_json(
            "connector",
            "delivery",
            "reconcile",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(reconcile["status"], "success")
        ack_reconciliation = reconcile["connector_ack_reconciliation"]
        self.assertEqual(ack_reconciliation["receipt_count"], 1)
        self.assertEqual(ack_reconciliation["ack_outbox_count"], 1)
        self.assertEqual(ack_reconciliation["artifact_count"], 1)
        self.assertEqual(ack_reconciliation["pending_ack_count"], 0)
        self.assertEqual(ack_reconciliation["acknowledged_without_artifact_count"], 0)
        self.assertEqual(ack_reconciliation["duplicate_logical_artifact_count"], 0)

    def test_connector_delivery_retry_and_quarantine_cs_ch_009(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")

        transient_1 = run_cli(
            "connector",
            "delivery",
            "process",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--failure-mode",
            "transient",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(transient_1.returncode, 5, transient_1.stdout + transient_1.stderr)
        transient_payload_1 = json.loads(transient_1.stdout)
        self.assertEqual(transient_payload_1["status"], "retry_scheduled")
        retry_state_1 = transient_payload_1["connector_delivery_retry_state"]
        self.assertEqual(retry_state_1["attempt_count"], 1)
        self.assertEqual(retry_state_1["retry_schedule"][0]["delay_seconds"], 60)
        self.assertFalse(retry_state_1["raw_provider_payload_persisted"])
        self.assertFalse(retry_state_1["unrelated_streams_blocked"])
        self.assertFalse((self.state_dir / "connector" / "delivery_receipts").exists())
        self.assertFalse((self.state_dir / "connector" / "ack_outbox").exists())

        transient_2 = run_cli(
            "connector",
            "delivery",
            "process",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--failure-mode",
            "transient",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(transient_2.returncode, 5, transient_2.stdout + transient_2.stderr)
        retry_state_2 = json.loads(transient_2.stdout)["connector_delivery_retry_state"]
        self.assertEqual(retry_state_2["attempt_count"], 2)
        self.assertEqual([item["delay_seconds"] for item in retry_state_2["retry_schedule"]], [60, 120])
        self.assertNotIn("connector_delivery_quarantine", json.loads(transient_2.stdout))

        recovered = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(recovered["status"], "success")
        self.assertTrue(recovered["acknowledgement"]["ack_sent"])
        self.assertEqual(recovered["connector_delivery_retry_state"]["status"], "resolved")

        poison_payloads = []
        for expected_returncode in [5, 5, 7]:
            poison = run_cli(
                "connector",
                "delivery",
                "process",
                "--file",
                POISON_DELIVERY_FIXTURE,
                "--contract-id",
                "ccon_project_alpha_github",
                "--state-dir",
                self.state_rel,
                "--json",
            )
            self.assertEqual(poison.returncode, expected_returncode, poison.stdout + poison.stderr)
            poison_payloads.append(json.loads(poison.stdout))

        self.assertEqual([payload["connector_delivery_retry_state"]["attempt_count"] for payload in poison_payloads], [1, 2, 3])
        self.assertEqual(poison_payloads[0]["status"], "retry_scheduled")
        self.assertEqual(poison_payloads[1]["status"], "retry_scheduled")
        self.assertEqual(poison_payloads[2]["status"], "quarantined")
        quarantine = poison_payloads[2]["connector_delivery_quarantine"]
        self.assertEqual(quarantine["status"], "quarantined")
        self.assertEqual(quarantine["attempt_count"], 3)
        self.assertEqual(quarantine["reason_code"], "CS_CONNECTOR_DELIVERY_PROJECTION_UNSUPPORTED")
        self.assertFalse(quarantine["safe_diagnostics"]["raw_provider_payload_persisted"])
        self.assertFalse(quarantine["safe_diagnostics"]["raw_provider_payload_in_operator_output"])
        self.assertTrue(quarantine["source_health_impact"]["unrelated_streams_continue"])
        self.assertFalse(quarantine["source_health_impact"]["unrelated_streams_blocked"])

        receipts = list((self.state_dir / "connector" / "delivery_receipts").glob("*.json"))
        outboxes = list((self.state_dir / "connector" / "ack_outbox").glob("*.json"))
        quarantines = list((self.state_dir / "connector" / "quarantine").glob("*.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(len(outboxes), 1)
        self.assertEqual(len(quarantines), 1)

        listed = run_json(
            "connector",
            "quarantine",
            "list",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(listed["status"], "success")
        quarantine_list = listed["connector_quarantine_list"]
        self.assertEqual(quarantine_list["quarantine_count"], 1)
        self.assertEqual(quarantine_list["open_quarantine_count"], 1)
        self.assertEqual(quarantine_list["items"][0]["quarantine_id"], quarantine["quarantine_id"])

        replay = run_json(
            "connector",
            "quarantine",
            "replay",
            "--quarantine-id",
            quarantine["quarantine_id"],
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(replay["status"], "success")
        self.assertEqual(replay["connector_delivery_quarantine"]["status"], "replay_requested")
        self.assertTrue(replay["quarantine_replay_attempt"]["failure_evidence_preserved"])
        self.assertEqual(replay["provider_internal_findings"], [])
        self.assertTrue(replay["audit_refs"])

    def test_connector_delivery_dedupes_and_versions_changed_content_cs_ch_010(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")

        first = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(first["status"], "success")
        first_receipt = first["connector_delivery_receipt"]
        first_artifact = first["artifact"]
        first_dedupe = first["connector_delivery_dedupe_state"]
        first_version = first["connector_content_version"]
        self.assertFalse(first["deduplicated"])
        self.assertEqual(first_dedupe["status"], "canonical")
        self.assertEqual(first_dedupe["delivery_count"], 1)
        self.assertEqual(first_dedupe["source_external_id"], ISSUE_SOURCE_EXTERNAL_ID)
        self.assertEqual(first_version["version_ordinal"], 1)
        self.assertEqual(first_version["predecessor_content_version_id"], None)

        duplicate_same_event = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            DUPLICATE_EVENT_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(duplicate_same_event["status"], "success")
        self.assertTrue(duplicate_same_event["deduplicated"])
        self.assertEqual(duplicate_same_event["connector_delivery_receipt"]["delivery_receipt_id"], first_receipt["delivery_receipt_id"])
        self.assertEqual(duplicate_same_event["artifact"]["artifact_id"], first_artifact["artifact_id"])
        duplicate_state = duplicate_same_event["connector_delivery_dedupe_state"]
        self.assertEqual(duplicate_state["delivery_count"], 2)
        self.assertEqual(duplicate_state["duplicate_delivery_count"], 1)
        self.assertTrue(duplicate_state["no_new_artifact_created"])
        self.assertTrue(duplicate_state["no_duplicate_active_truth_created"])

        duplicate_unchanged_content = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            UNCHANGED_EVENT_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(duplicate_unchanged_content["status"], "success")
        self.assertTrue(duplicate_unchanged_content["deduplicated"])
        self.assertEqual(duplicate_unchanged_content["connector_delivery_receipt"]["delivery_receipt_id"], first_receipt["delivery_receipt_id"])
        self.assertEqual(duplicate_unchanged_content["artifact"]["artifact_id"], first_artifact["artifact_id"])
        unchanged_state = duplicate_unchanged_content["connector_delivery_dedupe_state"]
        self.assertEqual(unchanged_state["delivery_count"], 3)
        self.assertEqual(len(unchanged_state["provider_event_ids"]), 2)
        self.assertEqual(len(unchanged_state["source_revisions"]), 2)

        changed = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            CHANGED_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(changed["status"], "success")
        changed_receipt = changed["connector_delivery_receipt"]
        changed_artifact = changed["artifact"]
        changed_version = changed["connector_content_version"]
        self.assertFalse(changed["deduplicated"])
        self.assertNotEqual(changed_receipt["delivery_receipt_id"], first_receipt["delivery_receipt_id"])
        self.assertNotEqual(changed_artifact["artifact_id"], first_artifact["artifact_id"])
        self.assertEqual(changed_version["version_ordinal"], 2)
        self.assertEqual(changed_version["predecessor_content_version_id"], first_version["content_version_id"])
        self.assertEqual(changed_version["predecessor_artifact_id"], first_artifact["artifact_id"])
        self.assertEqual(changed_artifact["provenance"]["lineage_from"], f"artifact:{first_artifact['artifact_id']}")
        self.assertFalse(changed_version["historical_evidence_mutated"])

        lineage = run_json(
            "connector",
            "lineage",
            "show",
            "--contract-id",
            "ccon_project_alpha_github",
            "--source-external-id",
            ISSUE_SOURCE_EXTERNAL_ID,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(lineage["status"], "success")
        content_lineage = lineage["connector_content_lineage"]
        self.assertEqual(content_lineage["schema_version"], "cs.connector_content_lineage.v1")
        self.assertEqual(content_lineage["version_count"], 2)
        self.assertTrue(content_lineage["one_current_logical_truth"])
        self.assertEqual(content_lineage["duplicate_active_truth_count"], 0)
        self.assertEqual(content_lineage["historical_evidence_mutation_count"], 0)
        self.assertEqual(content_lineage["current_content_version_id"], changed_version["content_version_id"])
        self.assertEqual(
            [version["content_version_id"] for version in content_lineage["versions"]],
            [first_version["content_version_id"], changed_version["content_version_id"]],
        )
        self.assertEqual(lineage["provider_internal_findings"], [])
        self.assertTrue(lineage["evidence_refs"])
        self.assertTrue(lineage["audit_refs"])

        receipts = list((self.state_dir / "connector" / "delivery_receipts").glob("*.json"))
        outboxes = list((self.state_dir / "connector" / "ack_outbox").glob("*.json"))
        artifact_records = list((self.state_dir / "artifacts" / "records").glob("**/*.json"))
        content_versions = list((self.state_dir / "connector" / "content_versions").glob("*.json"))
        dedupe_states = list((self.state_dir / "connector" / "delivery_dedupe").glob("*.json"))
        self.assertEqual(len(receipts), 2)
        self.assertEqual(len(outboxes), 2)
        self.assertEqual(len(artifact_records), 2)
        self.assertEqual(len(content_versions), 2)
        self.assertEqual(len(dedupe_states), 2)

    def test_connector_delivery_enforces_source_policy_restrictions_cs_ch_011(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")
        base_policy = plan["connector_source_policy"]

        allowed = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(allowed["status"], "success")
        allowed_decision = allowed["connector_projection_policy_decision"]
        self.assertEqual(allowed_decision["schema_version"], "cs.connector_projection_policy_decision.v1")
        self.assertEqual(allowed_decision["decision"], "allow")
        self.assertEqual(allowed_decision["enforcement_action"], "normalize")
        self.assertEqual(allowed_decision["source_policy_id"], base_policy["source_policy_id"])
        self.assertIn("body_markdown_excerpt", allowed_decision["included_fields"])
        self.assertEqual(allowed_decision["excluded_fields"], [])
        self.assertFalse(allowed_decision["body_restriction"]["full_body_allowed"])
        self.assertFalse(allowed_decision["raw_content_persisted"])
        self.assertFalse(allowed_decision["raw_provider_payload_persisted"])
        self.assertIn("body_markdown_excerpt", allowed_decision["normalized_projection"]["payload"])
        self.assertNotIn("body_markdown", allowed_decision["normalized_projection"]["payload"])
        self.assertIn("Source Policy allows metadata", allowed["artifact"]["connector_delivery"]["source_policy_enforcement"]["restriction_summary"])
        self.assertTrue(allowed["policy_decision_refs"])
        self.assertTrue(allowed["audit_refs"])

        forbidden = run_cli(
            "connector",
            "delivery",
            "ingest",
            "--file",
            FORBIDDEN_BODY_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(forbidden.returncode, 1, forbidden.stdout + forbidden.stderr)
        forbidden_payload = json.loads(forbidden.stdout)
        self.assertEqual(forbidden_payload["status"], "failed")
        self.assertIn("CS_CONNECTOR_SOURCE_POLICY_FIELD_FORBIDDEN", {error["code"] for error in forbidden_payload["errors"]})
        forbidden_decision = forbidden_payload["connector_projection_policy_decision"]
        self.assertEqual(forbidden_decision["decision"], "deny")
        self.assertEqual(forbidden_decision["enforcement_action"], "block")
        self.assertIn("body_markdown", forbidden_decision["excluded_fields"])
        self.assertIn("body_markdown", forbidden_decision["forbidden_body_fields"])
        self.assertEqual(forbidden_decision["normalized_projection"]["payload"], {})
        self.assertFalse(forbidden_decision["product_state_safe_to_use"])
        self.assertFalse(forbidden_decision["raw_content_persisted"])
        self.assertFalse(forbidden_decision["raw_provider_payload_persisted"])
        self.assertTrue(forbidden_payload["policy_decision_refs"])
        self.assertTrue(forbidden_payload["audit_refs"])

        artifact_records_after_forbidden = list((self.state_dir / "artifacts" / "records").glob("**/*.json"))
        receipts_after_forbidden = list((self.state_dir / "connector" / "delivery_receipts").glob("*.json"))
        self.assertEqual(len(artifact_records_after_forbidden), 1)
        self.assertEqual(len(receipts_after_forbidden), 1)
        self.assertNotIn("CS_CH_011_FORBIDDEN_FULL_BODY_MUST_NOT_PERSIST", state_file_texts(self.state_dir))

        narrowed = run_json(
            "connector",
            "source-policy",
            "confirm",
            "--contract-id",
            "ccon_project_alpha_github",
            "--max-content-bytes",
            "120",
            "--state-dir",
            self.state_rel,
        )
        narrowed_policy = narrowed["connector_source_policy"]
        self.assertEqual(narrowed_policy["max_content_bytes"], 120)
        self.assertNotEqual(narrowed_policy["source_policy_id"], base_policy["source_policy_id"])

        oversized = run_cli(
            "connector",
            "delivery",
            "ingest",
            "--file",
            OVERSIZED_EXCERPT_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(oversized.returncode, 1, oversized.stdout + oversized.stderr)
        oversized_payload = json.loads(oversized.stdout)
        self.assertIn("CS_CONNECTOR_SOURCE_POLICY_CONTENT_TOO_LARGE", {error["code"] for error in oversized_payload["errors"]})
        oversized_decision = oversized_payload["connector_projection_policy_decision"]
        self.assertEqual(oversized_decision["decision"], "deny")
        self.assertEqual(oversized_decision["source_policy_id"], narrowed_policy["source_policy_id"])
        self.assertEqual(oversized_decision["max_content_bytes"], 120)
        self.assertGreater(oversized_decision["content_size_bytes"], 120)
        self.assertEqual(oversized_decision["normalized_projection"]["payload"], {})

        artifact_records_after_oversized = list((self.state_dir / "artifacts" / "records").glob("**/*.json"))
        receipts_after_oversized = list((self.state_dir / "connector" / "delivery_receipts").glob("*.json"))
        policy_decisions = list((self.state_dir / "connector" / "projection_policy_decisions").glob("*.json"))
        self.assertEqual(len(artifact_records_after_oversized), 1)
        self.assertEqual(len(receipts_after_oversized), 1)
        self.assertEqual(len(policy_decisions), 3)

    def test_connector_evidence_bundle_promotes_evidenceref_metadata_cs_ch_012(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")
        process = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(process["status"], "success")
        receipt = process["connector_delivery_receipt"]
        artifact = process["artifact"]
        evidence_ref_id = receipt["evidence_ref"]["evidence_ref_id"]

        bundle_payload = run_json(
            "connector",
            "evidence",
            "bundle",
            "create",
            "--delivery-receipt-id",
            receipt["delivery_receipt_id"],
            "--query",
            "project alpha connector evidence",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(bundle_payload["status"], "success")
        self.assertEqual(bundle_payload["provider_internal_findings"], [])
        bundle = bundle_payload["evidence_bundle"]
        connector_link = bundle_payload["connector_evidence_bundle_link"]
        coverage = bundle["coverage"]
        self.assertEqual(bundle["schema_version"], "cs.evidence_bundle.v0")
        self.assertEqual(bundle["origin"], "connector_projection_delivery")
        self.assertEqual(len(bundle["evidence_items"]), 1)
        self.assertEqual(bundle["evidence_items"][0]["artifact_id"], artifact["artifact_id"])
        self.assertEqual(bundle["evidence_items"][0]["artifact_checksum_sha256"], artifact["checksum_sha256"])
        self.assertEqual(bundle["evidence_items"][0]["connector_evidence_ref"]["evidence_ref_id"], evidence_ref_id)
        self.assertEqual(connector_link["schema_version"], "cs.connector_evidence_bundle_link.v1")
        self.assertEqual(connector_link["delivery_receipt_id"], receipt["delivery_receipt_id"])
        self.assertEqual(connector_link["artifact_id"], artifact["artifact_id"])
        self.assertTrue(coverage["artifact_ref_present"])
        self.assertTrue(coverage["artifact_checksum_matches"])
        self.assertTrue(coverage["delivery_receipt_ref_present"])
        self.assertTrue(coverage["setup_result_ref_present"])
        self.assertTrue(coverage["source_policy_ref_present"])
        self.assertTrue(coverage["evidence_ref_metadata_present"])
        self.assertTrue(coverage["search_snapshot_ref_present"])
        self.assertTrue(coverage["audit_refs_present"])
        self.assertFalse(coverage["evidence_ref_alone_is_original"])
        self.assertEqual(coverage["inaccessible_phantom_evidence_count"], 0)
        self.assertFalse(bundle["raw_provider_payload_available"])
        self.assertTrue(bundle["raw_access_default_denied"])
        self.assertIn(f"connector_delivery_receipt:{receipt['delivery_receipt_id']}", bundle_payload["evidence_refs"])
        self.assertIn(f"connector_evidence_ref:{evidence_ref_id}", bundle_payload["evidence_refs"])
        self.assertTrue(bundle_payload["audit_refs"])

        bundle_path = self.state_dir / "evidence" / "bundles" / f"{bundle['evidence_bundle_id']}.json"
        snapshot_path = self.state_dir / "search" / "snapshots" / f"{bundle['search_snapshot_id']}.json"
        self.assertTrue(bundle_path.exists())
        self.assertTrue(snapshot_path.exists())

        shown = run_json(
            "evidence",
            "bundle",
            "show",
            bundle["evidence_bundle_id"],
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(shown["status"], "success")
        self.assertEqual(shown["evidence_bundle"]["evidence_bundle_id"], bundle["evidence_bundle_id"])

        claim = run_json(
            "claim",
            "create",
            "--evidence-bundle-id",
            bundle["evidence_bundle_id"],
            "--statement",
            "Project Alpha issue evidence is available from a connected source.",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(claim["status"], "success")
        self.assertEqual(claim["claim"]["trust_state"], "evidence_backed")
        self.assertEqual(claim["claim"]["evidence_bundle"]["evidence_item_count"], 1)
        approved = run_json(
            "claim",
            "approve",
            claim["claim"]["claim_id"],
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(approved["status"], "success")
        self.assertEqual(approved["claim"]["trust_state"], "approved")

        evidence_ref_only = run_cli(
            "connector",
            "evidence",
            "bundle",
            "create",
            "--evidence-ref-id",
            evidence_ref_id,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(evidence_ref_only.returncode, 4, evidence_ref_only.stdout + evidence_ref_only.stderr)
        evidence_ref_only_payload = json.loads(evidence_ref_only.stdout)
        self.assertEqual(evidence_ref_only_payload["status"], "failed")
        self.assertIn(
            "CS_CONNECTOR_EVIDENCE_REF_ONLY_UNSUPPORTED",
            {error["code"] for error in evidence_ref_only_payload["errors"]},
        )

        unsupported_claim = run_json(
            "claim",
            "create",
            "--statement",
            "EvidenceRef metadata alone should not become approved truth.",
            "--state-dir",
            self.state_rel,
        )
        deny = run_cli(
            "claim",
            "approve",
            unsupported_claim["claim"]["claim_id"],
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(deny.returncode, 4, deny.stdout + deny.stderr)
        deny_payload = json.loads(deny.stdout)
        self.assertEqual(deny_payload["status"], "failed")
        self.assertIn("CS_CLAIM_EVIDENCE_REQUIRED", {error["code"] for error in deny_payload["errors"]})
        self.assertNotEqual(deny_payload["claim"]["trust_state"], "approved")

    def test_connector_raw_access_is_denied_and_tightly_bounded_cs_ch_013(self) -> None:
        validate_default = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate_default["status"], "success")
        plan_default = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan_default["connector_source_policy"]["raw_access"], "denied")

        denied = run_cli(
            "connector",
            "raw-access",
            "request",
            "--contract-id",
            "ccon_project_alpha_github",
            "--evidence-ref-id",
            "eref_project_alpha_issue_1001",
            "--source-external-id",
            ISSUE_SOURCE_EXTERNAL_ID,
            "--purpose",
            "diagnose_ingestion_gap",
            "--classification",
            "restricted",
            "--ttl-seconds",
            "60",
            "--max-reads",
            "1",
            "--human-approved",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(denied.returncode, 8, denied.stdout + denied.stderr)
        denied_payload = json.loads(denied.stdout)
        self.assertEqual(denied_payload["status"], "denied")
        self.assertIn("CS_CONNECTOR_RAW_ACCESS_DENIED_BY_DEFAULT", {error["code"] for error in denied_payload["errors"]})
        self.assertTrue(denied_payload["audit_refs"])
        self.assertNotIn("connector_raw_access_grant", denied_payload)

        validate_raw = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            RAW_ACCESS_CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate_raw["status"], "success")
        raw_contract_id = validate_raw["ids"]["contract_id"]
        plan_raw = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            raw_contract_id,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan_raw["connector_source_policy"]["raw_access"], "temporary_scoped")
        self.assertEqual(plan_raw["connector_source_policy"]["raw_access_policy"]["max_ttl_seconds"], 60)
        self.assertEqual(plan_raw["connector_source_policy"]["raw_access_policy"]["max_reads"], 1)

        ttl_denied = run_cli(
            "connector",
            "raw-access",
            "request",
            "--contract-id",
            raw_contract_id,
            "--evidence-ref-id",
            "eref_project_alpha_issue_1001",
            "--source-external-id",
            ISSUE_SOURCE_EXTERNAL_ID,
            "--purpose",
            "diagnose_ingestion_gap",
            "--classification",
            "restricted",
            "--ttl-seconds",
            "61",
            "--max-reads",
            "1",
            "--human-approved",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(ttl_denied.returncode, 8, ttl_denied.stdout + ttl_denied.stderr)
        ttl_denied_payload = json.loads(ttl_denied.stdout)
        self.assertIn("CS_CONNECTOR_RAW_ACCESS_TTL_DENIED", {error["code"] for error in ttl_denied_payload["errors"]})

        read_limit_denied = run_cli(
            "connector",
            "raw-access",
            "request",
            "--contract-id",
            raw_contract_id,
            "--evidence-ref-id",
            "eref_project_alpha_issue_1001",
            "--source-external-id",
            ISSUE_SOURCE_EXTERNAL_ID,
            "--purpose",
            "diagnose_ingestion_gap",
            "--classification",
            "restricted",
            "--ttl-seconds",
            "60",
            "--max-reads",
            "2",
            "--human-approved",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(read_limit_denied.returncode, 8, read_limit_denied.stdout + read_limit_denied.stderr)
        read_limit_payload = json.loads(read_limit_denied.stdout)
        self.assertIn("CS_CONNECTOR_RAW_ACCESS_READ_LIMIT_DENIED", {error["code"] for error in read_limit_payload["errors"]})

        grant_payload = run_json(
            "connector",
            "raw-access",
            "request",
            "--contract-id",
            raw_contract_id,
            "--evidence-ref-id",
            "eref_project_alpha_issue_1001",
            "--source-external-id",
            ISSUE_SOURCE_EXTERNAL_ID,
            "--purpose",
            "diagnose_ingestion_gap",
            "--classification",
            "restricted",
            "--ttl-seconds",
            "60",
            "--max-reads",
            "1",
            "--human-approved",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(grant_payload["status"], "success")
        self.assertEqual(grant_payload["provider_internal_findings"], [])
        grant = grant_payload["connector_raw_access_grant"]
        grant_id = grant["raw_access_grant_id"]
        self.assertEqual(grant["status"], "active")
        self.assertEqual(grant["max_reads"], 1)
        self.assertEqual(grant["remaining_reads"], 1)
        self.assertFalse(grant["opaque_reference_exposed"])
        self.assertFalse(grant["reusable_raw_handle"])
        self.assertFalse(grant["raw_content_copied_to_product_records"])
        self.assertFalse(grant["raw_provider_payload_persisted"])
        self.assertTrue(grant["redaction"]["opaque_reference_redacted_in_output"])
        self.assertTrue(grant_payload["audit_refs"])

        exported = run_json(
            "connector",
            "raw-access",
            "export",
            "--grant-id",
            grant_id,
            "--state-dir",
            self.state_rel,
        )
        raw_export = exported["connector_raw_access_export"]
        self.assertFalse(raw_export["raw_content_included"])
        self.assertFalse(raw_export["raw_provider_payload_included"])
        self.assertFalse(raw_export["raw_access_handle_included"])
        self.assertIn("opaque_reference_fingerprint", raw_export)

        read = run_json(
            "connector",
            "raw-access",
            "read",
            "--grant-id",
            grant_id,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(read["status"], "success")
        self.assertFalse(read["connector_raw_access_result"]["raw_content_returned"])
        self.assertFalse(read["connector_raw_access_result"]["provider_payload_returned"])
        self.assertEqual(read["connector_raw_access_grant"]["read_count"], 1)
        self.assertEqual(read["connector_raw_access_grant"]["remaining_reads"], 0)

        exhausted = run_cli(
            "connector",
            "raw-access",
            "read",
            "--grant-id",
            grant_id,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(exhausted.returncode, 8, exhausted.stdout + exhausted.stderr)
        exhausted_payload = json.loads(exhausted.stdout)
        self.assertEqual(exhausted_payload["status"], "denied")
        self.assertIn("CS_CONNECTOR_RAW_ACCESS_READ_LIMIT_EXHAUSTED", {error["code"] for error in exhausted_payload["errors"]})

        expiry_grant = run_json(
            "connector",
            "raw-access",
            "request",
            "--contract-id",
            raw_contract_id,
            "--evidence-ref-id",
            "eref_project_alpha_issue_1001_expiry",
            "--source-external-id",
            ISSUE_SOURCE_EXTERNAL_ID,
            "--purpose",
            "diagnose_ingestion_gap",
            "--classification",
            "restricted",
            "--ttl-seconds",
            "60",
            "--max-reads",
            "1",
            "--human-approved",
            "--state-dir",
            self.state_rel,
        )["connector_raw_access_grant"]
        expired = run_cli(
            "connector",
            "raw-access",
            "read",
            "--grant-id",
            expiry_grant["raw_access_grant_id"],
            "--at",
            "2999-01-01T00:00:00Z",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(expired.returncode, 8, expired.stdout + expired.stderr)
        expired_payload = json.loads(expired.stdout)
        self.assertIn("CS_CONNECTOR_RAW_ACCESS_EXPIRED", {error["code"] for error in expired_payload["errors"]})

        revoke_grant = run_json(
            "connector",
            "raw-access",
            "request",
            "--contract-id",
            raw_contract_id,
            "--evidence-ref-id",
            "eref_project_alpha_issue_1001_revoke",
            "--source-external-id",
            ISSUE_SOURCE_EXTERNAL_ID,
            "--purpose",
            "diagnose_ingestion_gap",
            "--classification",
            "restricted",
            "--ttl-seconds",
            "60",
            "--max-reads",
            "1",
            "--human-approved",
            "--state-dir",
            self.state_rel,
        )["connector_raw_access_grant"]
        revoked = run_json(
            "connector",
            "raw-access",
            "revoke",
            "--grant-id",
            revoke_grant["raw_access_grant_id"],
            "--reason",
            "operator finished diagnostic review",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(revoked["connector_raw_access_grant"]["status"], "revoked")
        revoked_read = run_cli(
            "connector",
            "raw-access",
            "read",
            "--grant-id",
            revoke_grant["raw_access_grant_id"],
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(revoked_read.returncode, 8, revoked_read.stdout + revoked_read.stderr)
        revoked_payload = json.loads(revoked_read.stdout)
        self.assertIn("CS_CONNECTOR_RAW_ACCESS_REVOKED", {error["code"] for error in revoked_payload["errors"]})

        raw_access_state = state_file_texts(self.state_dir)
        self.assertNotIn("CS_CH_013_RAW_PROVIDER_PAYLOAD_MUST_NOT_PERSIST", raw_access_state)
        self.assertNotIn("raw_access_handle\":\"", raw_access_state)

    def test_connector_untrusted_content_cannot_direct_agents_or_actions_cs_ch_014(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")

        process = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            PROMPT_INJECTION_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(process["status"], "success")
        artifact = process["artifact"]
        receipt = process["connector_delivery_receipt"]
        review = process["connector_untrusted_content_review"]
        self.assertEqual(artifact["trust_state"], "untrusted")
        self.assertEqual(receipt["source_summary"]["trust_state"], "untrusted_connector_content")
        self.assertEqual(review["schema_version"], "cs.connector_untrusted_content_review.v1")
        self.assertEqual(review["source_trust_label"], "untrusted_connector_content")
        self.assertTrue(review["unsafe_instruction_detected"])
        self.assertGreater(review["blocked_attempt_count"], 0)
        self.assertFalse(review["content_handling"]["treated_as_system_instruction"])
        self.assertTrue(review["content_handling"]["quoted_or_summarized_as_evidence_only"])
        self.assertTrue(all(value == 0 for value in review["negative_evidence"].values()))
        self.assertFalse(artifact["safety"]["authority_expanded"])
        self.assertEqual(artifact["safety"]["external_http_calls"], 0)

        shown_review = run_json(
            "connector",
            "untrusted-content",
            "review",
            "--delivery-receipt-id",
            receipt["delivery_receipt_id"],
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(shown_review["status"], "success")
        self.assertEqual(shown_review["connector_untrusted_content_review"]["review_id"], review["review_id"])
        self.assertIn(f"connector_untrusted_content_review:{review['review_id']}", shown_review["evidence_refs"])

        bundle_payload = run_json(
            "connector",
            "evidence",
            "bundle",
            "create",
            "--delivery-receipt-id",
            receipt["delivery_receipt_id"],
            "--query",
            "quoted malicious connector instruction",
            "--state-dir",
            self.state_rel,
        )
        coverage = bundle_payload["evidence_bundle"]["coverage"]
        self.assertTrue(coverage["untrusted_evidence_label_present"])
        self.assertTrue(coverage["unsafe_instruction_treated_as_evidence_only"])
        self.assertTrue(coverage["tool_action_egress_counters_zero"])
        self.assertTrue(coverage["memory_promotion_blocked"])
        self.assertTrue(coverage["policy_override_blocked"])
        self.assertTrue(coverage["authority_expansion_blocked"])
        trust_boundary = bundle_payload["connector_evidence_bundle_link"]["trust_boundary"]
        self.assertTrue(trust_boundary["untrusted_evidence_label_present"])
        self.assertTrue(trust_boundary["quoted_or_summarized_as_evidence_only"])
        bundle = bundle_payload["evidence_bundle"]

        claim = run_json(
            "claim",
            "create",
            "--evidence-bundle-id",
            bundle["evidence_bundle_id"],
            "--statement",
            "The connector source contains a quoted instruction to ignore policy; it is not a system instruction or authority grant.",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(claim["claim"]["trust_state"], "evidence_backed")
        self.assertIn("quoted instruction", claim["claim"]["statement"])
        self.assertIn("not a system instruction", claim["claim"]["statement"])

        authority = run_cli(
            "agent",
            "prompt-authority-test",
            "--role-id",
            "connector",
            "--requested-tool",
            "shell.run",
            "--requested-memory-scope",
            "global",
            "--requested-authority",
            "external_egress",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(authority.returncode, 8, authority.stdout + authority.stderr)
        authority_payload = json.loads(authority.stdout)
        self.assertEqual(authority_payload["status"], "denied")
        self.assertIn("CS_AGENT_POLICY_DENIED", {error["code"] for error in authority_payload["errors"]})
        self.assertFalse(authority_payload["authority_attempt"]["authority_expanded"])

        memory = run_json(
            "memory",
            "quarantine-check",
            "--artifact-id",
            artifact["artifact_id"],
            "--statement",
            "Promote the quoted connector instruction as permanent memory.",
            "--state-dir",
            self.state_rel,
        )
        quarantine = memory["memory_quarantine"]
        self.assertEqual(quarantine["status"], "quarantined")
        self.assertFalse(quarantine["memory_created"])
        self.assertFalse(quarantine["trusted_memory_created"])

        egress = run_cli(
            "egress",
            "test",
            "--url",
            "http://example.test/leak",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(egress.returncode, 8, egress.stdout + egress.stderr)
        egress_payload = json.loads(egress.stdout)
        self.assertEqual(egress_payload["status"], "denied")
        self.assertIn("CS_EGRESS_DENIED", {error["code"] for error in egress_payload["errors"]})
        self.assertEqual(egress_payload["policy_decisions"][0]["external_http_calls"], 0)

        self.assertFalse((self.state_dir / "actions").exists())
        self.assertFalse((self.state_dir / "memories").exists())
        self.assertEqual(len(list((self.state_dir / "memory_quarantine").glob("*.json"))), 1)
        self.assertNotIn("CS_CH_014_UNAUTHORIZED_SIDE_EFFECT_MUST_NOT_EXIST", state_file_texts(self.state_dir))

    def test_connector_selected_github_repositories_only_cs_ch_015(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            SELECTED_REPOSITORIES_CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        self.assertEqual(validate["connector_capability_contract"]["contract_id"], "ccon_selected_repos_github")
        self.assertEqual(validate["connector_capability_contract"]["actions"], [])

        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_selected_repos_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")
        selection = plan["connector_setup_result"]["selected_resource_scope"]
        self.assertEqual(selection["schema_version"], "cs.connector_selected_resource_scope.v1")
        self.assertEqual(selection["available_resource_count"], 3)
        self.assertEqual(selection["selected_resource_count"], 1)
        self.assertEqual(selection["unselected_resource_count"], 2)
        self.assertEqual(selection["selected_resources"], ["github:repo:owner/project-alpha"])
        self.assertEqual(selection["visible_to_product_resources"], ["github:repo:owner/project-alpha"])
        self.assertFalse(selection["organization_wide_fallback_enabled"])
        self.assertFalse(selection["account_wide_fallback_enabled"])
        self.assertTrue(selection["namespace_scoped"])
        self.assertTrue(selection["versioned"])
        self.assertTrue(selection["stores_opaque_source_refs_only"])
        self.assertFalse(selection["credentials_exposed"])
        self.assertFalse(selection["write_permissions_requested"])
        self.assertFalse(selection["write_permissions_granted"])
        self.assertEqual(selection["provider_mutation_capabilities"], [])
        self.assertEqual(
            plan["connector_source_policy"]["selected_resource_scope"]["selection_version_id"],
            selection["selection_version_id"],
        )

        allowed = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            SELECTED_REPO_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_selected_repos_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(allowed["status"], "success")
        self.assertEqual(allowed["connector_projection_policy_decision"]["selected_resource_allowed"], True)
        self.assertEqual(
            allowed["connector_delivery_receipt"]["source_summary"]["source_ref"],
            "github:repo:owner/project-alpha",
        )

        denied = run_cli(
            "connector",
            "delivery",
            "process",
            "--file",
            UNSELECTED_REPO_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_selected_repos_github",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(denied.returncode, 1, denied.stdout + denied.stderr)
        denied_payload = json.loads(denied.stdout)
        self.assertEqual(denied_payload["status"], "failed")
        self.assertIn("CS_CONNECTOR_SOURCE_POLICY_RESOURCE_DENIED", [error["code"] for error in denied_payload["errors"]])
        denied_decision = denied_payload["connector_projection_policy_decision"]
        self.assertEqual(denied_decision["decision"], "deny")
        self.assertEqual(denied_decision["enforcement_action"], "block")
        self.assertFalse(denied_decision["selected_resource_allowed"])
        self.assertEqual(denied_decision["normalized_projection"]["payload"], {})
        self.assertNotIn("artifact", denied_payload)
        self.assertNotIn("connector_delivery_receipt", denied_payload)
        self.assertTrue(denied_payload["audit_refs"])
        self.assertTrue(denied_payload["evidence_refs"])

        direct_write = run_cli(
            "connector",
            "direct-write-test",
            "--provider",
            "github",
            "--target",
            "github:repo:owner/project-alpha",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(direct_write.returncode, 8, direct_write.stdout + direct_write.stderr)
        direct_payload = json.loads(direct_write.stdout)
        self.assertEqual(direct_payload["status"], "denied")
        self.assertIn("CS_DIRECT_WRITE_DENIED", [error["code"] for error in direct_payload["errors"]])
        self.assertEqual(direct_payload["errors"][0]["external_http_calls"], 0)

        broaden = run_cli(
            "connector",
            "source-policy",
            "confirm",
            "--contract-id",
            "ccon_selected_repos_github",
            "--selected-resource",
            "github:repo:owner/project-alpha",
            "--selected-resource",
            "github:repo:owner/project-beta",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(broaden.returncode, 8, broaden.stdout + broaden.stderr)
        broaden_payload = json.loads(broaden.stdout)
        self.assertIn("CS_CONNECTOR_SOURCE_POLICY_BROADENING_DENIED", [error["code"] for error in broaden_payload["errors"]])

        receipts = list((self.state_dir / "connector" / "delivery_receipts").glob("*.json"))
        artifacts = list((self.state_dir / "artifacts" / "records").glob("**/*.json"))
        acks = list((self.state_dir / "connector" / "ack_outbox").glob("*.json"))
        policies = list((self.state_dir / "connector" / "source_policies").glob("*.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(len(acks), 1)
        self.assertEqual(len(policies), 1)
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn("CS_CH_015_UNSELECTED_REPO_ARTIFACT_MUST_NOT_EXIST", state_text)
        self.assertNotIn('"provider_token"', state_text)

    def test_connector_github_write_paths_denied_cs_ch_019(self) -> None:
        rejected_contract = run_cli(
            "connector",
            "contract",
            "validate",
            "--file",
            GITHUB_WRITE_ACTION_CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(rejected_contract.returncode, 1, rejected_contract.stdout + rejected_contract.stderr)
        rejected_payload = json.loads(rejected_contract.stdout)
        self.assertEqual(rejected_payload["status"], "failed")
        error_codes = [error["code"] for error in rejected_payload["errors"]]
        self.assertIn("CS_CONNECTOR_GITHUB_WRITE_ACTION_DENIED", error_codes)
        self.assertFalse((self.state_dir / "connector" / "contracts").exists())

        guard = run_json(
            "connector",
            "github-write-guard",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(guard["status"], "success")
        guard_report = guard["connector_github_write_guard"]
        self.assertEqual(guard_report["schema_version"], "cs.connector_github_write_guard.v1")
        self.assertEqual(guard_report["status"], "pass")
        self.assertTrue(guard["audit_refs"])
        self.assertTrue(guard["evidence_refs"])
        for counter, value in guard_report["negative_evidence"].items():
            self.assertEqual(value, 0, counter)
        self.assertEqual(guard_report["forbidden_cli_command_hits"], [])
        self.assertEqual(guard_report["forbidden_endpoint_literal_hits"], [])
        self.assertTrue(guard_report["controlled_egress_attempts"])
        self.assertTrue(
            all(
                attempt["status"] == "denied"
                and attempt["external_http_calls"] == 0
                and attempt["provider_mutations"] == 0
                and attempt["direct_provider_access"] is False
                for attempt in guard_report["controlled_egress_attempts"]
            )
        )
        self.assertTrue(all(pack["declared_action_count"] == 0 for pack in guard_report["provider_packs"]))
        self.assertTrue(all(contract.get("write_action_count", 0) == 0 for contract in guard_report["active_contracts"]))

        for operation in ["issue.comment", "issue.label", "pull_request.merge", "file.write", "repository.settings.update"]:
            direct_write = run_cli(
                "connector",
                "direct-write-test",
                "--provider",
                "github",
                "--target",
                f"github:repo:owner/project-alpha:{operation}",
                "--operation",
                operation,
                "--state-dir",
                self.state_rel,
                "--json",
            )
            self.assertEqual(direct_write.returncode, 8, direct_write.stdout + direct_write.stderr)
            direct_payload = json.loads(direct_write.stdout)
            self.assertEqual(direct_payload["status"], "denied")
            self.assertEqual(direct_payload["direct_write_denial"]["operation"], operation)
            self.assertEqual(direct_payload["direct_write_denial"]["external_http_calls"], 0)
            self.assertEqual(direct_payload["direct_write_denial"]["provider_mutations"], 0)
            self.assertFalse(direct_payload["direct_write_denial"]["direct_provider_access"])
            self.assertIn("CS_DIRECT_WRITE_DENIED", [error["code"] for error in direct_payload["errors"]])

        state_text = state_file_texts(self.state_dir)
        self.assertNotIn('"provider_token"', state_text)
        self.assertNotIn('"auth_header"', state_text)
        self.assertNotIn('"direct_api_handle"', state_text)

    def test_connector_github_provider_failure_states_cs_ch_020(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")
        baseline = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(baseline["status"], "success")
        artifact_count_before = len(list((self.state_dir / "artifacts" / "records").glob("**/*.json")))
        receipt_count_before = len(list((self.state_dir / "connector" / "delivery_receipts").glob("*.json")))
        self.assertEqual(artifact_count_before, 1)
        self.assertEqual(receipt_count_before, 1)

        failure_states = {}
        for failure_mode in ("rate_limit", "permission_revoked", "repository_removed", "transient_transport"):
            payload = run_json(
                "connector",
                "github-failure",
                "simulate",
                "--failure-mode",
                failure_mode,
                "--contract-id",
                "ccon_project_alpha_github",
                "--source-ref",
                "github:repo:owner/project-alpha",
                "--state-dir",
                self.state_rel,
            )
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["provider_internal_findings"], [])
            self.assertTrue(payload["audit_refs"])
            self.assertTrue(payload["evidence_refs"])
            state = payload["connector_provider_failure_state"]
            failure_states[failure_mode] = state
            self.assertEqual(state["schema_version"], "cs.connector_provider_failure_state.v1")
            self.assertEqual(state["provider"], "github")
            self.assertEqual(state["source_ref"], "github:repo:owner/project-alpha")
            self.assertEqual(state["existing_evidence"]["artifact_count_before"], 1)
            self.assertEqual(state["existing_evidence"]["delivery_receipt_count_before"], 1)
            self.assertTrue(state["existing_evidence"]["existing_artifacts_preserved"])
            self.assertFalse(state["existing_evidence"]["delete_existing_evidence"])
            self.assertFalse(state["freshness"]["current_data_claim_allowed"])
            self.assertFalse(state["freshness"]["fresh_sync_claim_allowed"])
            self.assertTrue(state["surface_warnings"]["search_result_warning"]["warning_required"])
            self.assertFalse(state["surface_warnings"]["claim_warning"]["current_data_claim_allowed"])
            self.assertTrue(state["surface_warnings"]["claim_warning"]["unsupported_fresh_claim_denied"])
            self.assertEqual(state["ingestion_control"]["external_http_calls"], 0)
            self.assertEqual(state["ingestion_control"]["provider_mutations"], 0)
            for counter, value in state["negative_evidence"].items():
                self.assertEqual(value, 0, counter)

        rate_limit = failure_states["rate_limit"]
        self.assertEqual(rate_limit["reason_code"], "CS_CONNECTOR_GITHUB_RATE_LIMITED")
        self.assertEqual(rate_limit["retry_policy"]["status"], "scheduled")
        self.assertGreaterEqual(rate_limit["retry_policy"]["retry_after_seconds"], 60)
        self.assertTrue(rate_limit["retry_policy"]["tight_retry_loop_prevented"])
        self.assertEqual(rate_limit["freshness"]["state"], "delayed")
        self.assertEqual(rate_limit["ingestion_control"]["stream_state"], "delayed")

        revoked = failure_states["permission_revoked"]
        self.assertEqual(revoked["reason_code"], "CS_CONNECTOR_GITHUB_PERMISSION_REVOKED")
        self.assertTrue(revoked["setup_gap"]["permanent"])
        self.assertEqual(revoked["ingestion_control"]["stream_state"], "suspended")
        self.assertFalse(revoked["ingestion_control"]["future_ingestion_allowed"])
        self.assertTrue(revoked["recovery_path"]["owner_action_required"])
        self.assertTrue(revoked["recovery_path"]["requires_new_verification"])

        removed = failure_states["repository_removed"]
        self.assertEqual(removed["reason_code"], "CS_CONNECTOR_GITHUB_REPOSITORY_REMOVED")
        self.assertEqual(removed["source_health"]["source_availability"], "repository_removed")
        self.assertEqual(removed["ingestion_control"]["stream_state"], "stopped")
        self.assertFalse(removed["ingestion_control"]["future_ingestion_allowed"])
        self.assertTrue(removed["ingestion_control"]["stop_removed_scope"])
        self.assertEqual(removed["freshness"]["state"], "unavailable")

        transient = failure_states["transient_transport"]
        self.assertEqual(transient["reason_code"], "CS_CONNECTOR_GITHUB_TRANSPORT_TRANSIENT")
        self.assertEqual(transient["retry_policy"]["status"], "scheduled")
        self.assertFalse(transient["setup_gap"]["permanent"])
        self.assertEqual(transient["ingestion_control"]["stream_state"], "retrying")

        artifact_count_after = len(list((self.state_dir / "artifacts" / "records").glob("**/*.json")))
        receipt_count_after = len(list((self.state_dir / "connector" / "delivery_receipts").glob("*.json")))
        failure_state_count = len(list((self.state_dir / "connector" / "provider_failure_states").glob("*.json")))
        self.assertEqual(artifact_count_after, artifact_count_before)
        self.assertEqual(receipt_count_after, receipt_count_before)
        self.assertEqual(failure_state_count, 4)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn('"provider_token"', state_text)
        self.assertNotIn('"auth_header"', state_text)
        self.assertNotIn('"direct_api_handle"', state_text)

    def test_connector_macos_capture_disabled_until_consent_and_permission_cs_ch_021(self) -> None:
        missing_probe = run_json(
            "connector",
            "capture",
            "permission",
            "probe",
            "--platform-permission-state",
            "not_granted",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(missing_probe["status"], "success")
        self.assertEqual(missing_probe["provider_internal_findings"], [])
        permission_probe = missing_probe["connector_capture_permission_probe"]
        self.assertEqual(permission_probe["schema_version"], "cs.connector_capture_permission_probe.v1")
        self.assertTrue(permission_probe["permission_probe_only"])
        self.assertFalse(permission_probe["capture_enabled"])
        self.assertFalse(permission_probe["capture_attempted"])
        self.assertEqual(permission_probe["capture_samples_created"], 0)
        self.assertEqual(permission_probe["screenshots_created"], 0)
        for counter, value in permission_probe["negative_evidence"].items():
            self.assertEqual(value, 0, counter)

        no_consent_no_permission = run_json(
            "connector",
            "capture",
            "guard",
            "evaluate",
            "--platform-permission-state",
            "not_granted",
            "--state-dir",
            self.state_rel,
        )
        guard = no_consent_no_permission["connector_capture_guard_decision"]
        self.assertEqual(guard["schema_version"], "cs.connector_capture_guard_decision.v1")
        self.assertEqual(guard["status"], "blocked")
        self.assertFalse(guard["capture_enabled"])
        self.assertFalse(guard["collection_started_by_this_command"])
        self.assertIn("CS_CONNECTOR_CAPTURE_CONSENT_MISSING", guard["reason_codes"])
        self.assertIn("CS_CONNECTOR_CAPTURE_PLATFORM_PERMISSION_MISSING", guard["reason_codes"])
        self.assertEqual(no_consent_no_permission["provider_internal_findings"], [])

        granted_probe = run_json(
            "connector",
            "capture",
            "permission",
            "probe",
            "--platform-permission-state",
            "granted",
            "--state-dir",
            self.state_rel,
        )
        self.assertTrue(granted_probe["connector_capture_permission_probe"]["permission_active_for_local_fixture"])
        permission_only = run_json(
            "connector",
            "capture",
            "guard",
            "evaluate",
            "--platform-permission-state",
            "granted",
            "--state-dir",
            self.state_rel,
        )
        permission_only_guard = permission_only["connector_capture_guard_decision"]
        self.assertEqual(permission_only_guard["status"], "blocked")
        self.assertFalse(permission_only_guard["capture_enabled"])
        self.assertIn("CS_CONNECTOR_CAPTURE_CONSENT_MISSING", permission_only_guard["reason_codes"])
        self.assertNotIn("CS_CONNECTOR_CAPTURE_PLATFORM_PERMISSION_MISSING", permission_only_guard["reason_codes"])

        consent = run_json(
            "connector",
            "capture",
            "consent",
            "granted",
            "--purpose",
            "local fixture WatchAgent capture readiness",
            "--state-dir",
            self.state_rel,
        )
        consent_record = consent["connector_watch_source_consent"]
        self.assertEqual(consent_record["schema_version"], "cs.connector_watch_source_consent.v1")
        self.assertTrue(consent_record["explicit_owner_consent"])
        self.assertFalse(consent_record["capture_enabled"])
        self.assertFalse(consent_record["collection_started"])

        consent_only = run_json(
            "connector",
            "capture",
            "guard",
            "evaluate",
            "--platform-permission-state",
            "not_granted",
            "--state-dir",
            self.state_rel,
        )
        consent_only_guard = consent_only["connector_capture_guard_decision"]
        self.assertEqual(consent_only_guard["status"], "blocked")
        self.assertFalse(consent_only_guard["capture_enabled"])
        self.assertTrue(consent_only_guard["consent"]["active"])
        self.assertIn("CS_CONNECTOR_CAPTURE_PLATFORM_PERMISSION_MISSING", consent_only_guard["reason_codes"])

        ready = run_json(
            "connector",
            "capture",
            "guard",
            "evaluate",
            "--platform-permission-state",
            "granted",
            "--state-dir",
            self.state_rel,
        )
        ready_guard = ready["connector_capture_guard_decision"]
        self.assertEqual(ready_guard["status"], "ready")
        self.assertTrue(ready_guard["capture_enabled"])
        self.assertTrue(ready_guard["capture_allowed_for_future_collection"])
        self.assertFalse(ready_guard["collection_started_by_this_command"])
        self.assertEqual(ready_guard["capture_samples_created"], 0)
        self.assertEqual(ready_guard["artifacts_created"], 0)
        self.assertIn("CS_CONNECTOR_CAPTURE_READY", ready_guard["reason_codes"])
        self.assertEqual(ready["provider_internal_findings"], [])
        for payload in [no_consent_no_permission, permission_only, consent, consent_only, ready]:
            self.assertTrue(payload["audit_refs"])
            self.assertTrue(payload["evidence_refs"])
            for counter, value in payload["negative_evidence"].items():
                self.assertEqual(value, 0, counter)

        self.assertEqual(len(list((self.state_dir / "connector" / "capture_permission_probes").glob("*.json"))), 2)
        self.assertEqual(len(list((self.state_dir / "connector" / "watch_source_consents").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "capture_guard_decisions").glob("*.json"))), 4)
        self.assertFalse((self.state_dir / "connector" / "capture_samples").exists())
        self.assertEqual(len(list((self.state_dir / "artifacts" / "records").glob("**/*.json"))), 0)
        state_text = state_file_texts(self.state_dir)
        for marker in [
            "CS_CH_021_HIDDEN_STARTUP_CAPTURE_MUST_NOT_EXIST",
            "CS_CH_021_CROSS_NAMESPACE_CAPTURE_MUST_NOT_EXIST",
            "CS_CH_021_SCREENSHOT_MUST_NOT_EXIST",
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_connector_activity_samples_sessionize_to_bounded_sessions_cs_ch_022(self) -> None:
        payload = run_json(
            "connector",
            "capture",
            "sessionize",
            "--file",
            ACTIVITY_SAMPLES_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["provider_internal_findings"], [])
        batch = payload["connector_activity_sample_batch"]
        sessionization = payload["connector_activity_sessionization"]
        sessions = payload["activity_session_projections"]
        self.assertEqual(batch["schema_version"], "cs.connector_activity_sample_batch.v1")
        self.assertEqual(sessionization["schema_version"], "cs.connector_activity_sessionization.v1")
        self.assertEqual(sessionization["session_projection_schema"], "cs.activity_session_projection.v1")
        self.assertEqual(len(sessions), 3)
        self.assertEqual(sessionization["input_metrics"]["sample_count"], 9)
        self.assertEqual(sessionization["input_metrics"]["unique_sample_count"], 8)
        self.assertEqual(sessionization["input_metrics"]["retained_active_sample_count"], 6)
        self.assertEqual(sessionization["input_metrics"]["duplicate_sample_count"], 1)
        self.assertEqual(sessionization["input_metrics"]["idle_gap_sample_count"], 1)
        self.assertEqual(sessionization["input_metrics"]["low_information_sample_count"], 1)
        self.assertEqual(sessionization["filtered_samples"]["duplicates"], ["sample-001-duplicate"])
        self.assertEqual(sessionization["filtered_samples"]["idle_gap_markers"], ["sample-004-idle"])
        self.assertEqual(sessionization["filtered_samples"]["low_information_noise"], ["sample-006-noise"])
        self.assertFalse(sessionization["unsupported_intent_claim_present"])
        self.assertFalse(sessionization["inference_stored_as_observed_fact"])
        for counter, value in sessionization["negative_evidence"].items():
            self.assertEqual(value, 0, counter)

        first = sessions[0]
        self.assertEqual(first["source_sample_ids"], ["sample-001", "sample-002", "sample-003"])
        self.assertEqual(first["observed_facts"]["app_switch_count"], 2)
        self.assertEqual(first["observed_facts"]["project_hints"], ["connectorhub"])
        self.assertIn("contains_app_switches", first["confidence"]["caveats"])
        self.assertFalse(first["inference"]["unsupported_intent_claim_present"])
        self.assertIsNone(first["inference"]["intent_claim"])

        second = sessions[1]
        self.assertEqual(second["source_sample_ids"], ["sample-005", "sample-007"])
        self.assertIn("idle_gap_boundary", second["confidence"]["caveats"])
        self.assertIn("low_information_samples_filtered_in_batch", second["confidence"]["caveats"])

        third = sessions[2]
        self.assertEqual(third["source_sample_ids"], ["sample-008"])
        self.assertIn("sparse_sample_count", third["confidence"]["caveats"])
        self.assertEqual(third["observed_facts"]["project_hints"], ["team_coordination"])

        for session in sessions:
            self.assertTrue(session["bounded"])
            self.assertGreater(session["duration_seconds"], 0)
            self.assertFalse(session["privacy"]["raw_titles_stored"])
            self.assertFalse(session["privacy"]["full_urls_stored"])
            self.assertFalse(session["privacy"]["keystrokes_collected"])
            self.assertFalse(session["privacy"]["clipboard_values_collected"])
            self.assertFalse(session["privacy"]["screenshots_collected"])
            self.assertTrue(session["evidence_refs"])
            self.assertTrue(session["audit_refs"])

        self.assertEqual(len(list((self.state_dir / "connector" / "activity_sample_batches").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "activity_sessionizations").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "activity_sessions").glob("*.json"))), 3)
        self.assertEqual(len(list((self.state_dir / "artifacts" / "records").glob("**/*.json"))), 0)
        state_text = state_file_texts(self.state_dir)
        for marker in [
            "CS_CH_022_UNSUPPORTED_INTENT_MUST_NOT_EXIST",
            "CS_CH_022_RAW_WINDOW_TITLE_MUST_NOT_EXIST",
            "CS_CH_022_BROWSER_HISTORY_MUST_NOT_EXIST",
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_watch_rule_lifecycle_is_scoped_versioned_and_audited_cs_ch_023(self) -> None:
        create = run_json(
            "watch",
            "rule",
            "create",
            "--file",
            WATCH_RULE_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(create["status"], "success")
        self.assertEqual(create["provider_internal_findings"], [])
        rule = create["watch_rule"]
        version = create["watch_rule_version"]
        create_policy = create["watch_rule_policy_decision"]
        watch_rule_id = rule["watch_rule_id"]
        first_version_id = version["watch_rule_version_id"]
        self.assertEqual(rule["schema_version"], "cs.watch_rule.v1")
        self.assertEqual(version["schema_version"], "cs.watch_rule_version.v1")
        self.assertEqual(create_policy["schema_version"], "cs.watch_rule_policy_decision.v1")
        self.assertEqual(rule["status"], "draft")
        self.assertIsNone(rule["active_version_id"])
        self.assertEqual(rule["scope"]["namespace_id"], "personal")
        self.assertFalse(rule["external_action_authority"])
        self.assertFalse(rule["can_authorize_external_action_execution"])
        self.assertTrue(rule["source_refs"])
        self.assertTrue(rule["connector_contract_refs"])
        self.assertTrue(rule["source_policy_refs"])
        self.assertEqual(create_policy["decision"], "allow")
        self.assertFalse(create_policy["activation_allowed"])
        for counter, value in create["negative_evidence"].items():
            self.assertEqual(value, 0, counter)

        missing = run_cli(
            "watch",
            "rule",
            "activate",
            "--watch-rule-id",
            watch_rule_id,
            "--source-readiness",
            "missing",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(missing.returncode, 8, missing.stdout + missing.stderr)
        missing_payload = json.loads(missing.stdout)
        self.assertEqual(missing_payload["status"], "denied")
        self.assertIn("CS_WATCH_RULE_SOURCE_NOT_READY", {error["code"] for error in missing_payload["errors"]})
        missing_policy = missing_payload["watch_rule_policy_decision"]
        self.assertEqual(missing_policy["decision"], "deny")
        self.assertFalse(missing_policy["activation_allowed"])
        self.assertFalse(missing_policy["checks"]["source_permissions_ready"])
        self.assertEqual(missing_payload["watch_rule"]["status"], "draft")
        self.assertIsNone(missing_payload["watch_rule"]["active_version_id"])

        active = run_json(
            "watch",
            "rule",
            "activate",
            "--watch-rule-id",
            watch_rule_id,
            "--source-readiness",
            "ready",
            "--state-dir",
            self.state_rel,
        )
        active_rule = active["watch_rule"]
        active_policy = active["watch_rule_policy_decision"]
        self.assertEqual(active_rule["status"], "active")
        self.assertEqual(active_rule["active_version_id"], first_version_id)
        self.assertEqual(active_policy["decision"], "allow")
        self.assertTrue(active_policy["activation_allowed"])
        self.assertTrue(active_policy["checks"]["source_permissions_ready"])

        trace_payload = run_json(
            "watch",
            "rule",
            "evaluate",
            "--watch-rule-id",
            watch_rule_id,
            "--source-evidence-ref",
            "connector_delivery_receipt:cdelrec_project_alpha_1001",
            "--state-dir",
            self.state_rel,
        )
        trace = trace_payload["watch_rule_evaluation_trace"]
        self.assertEqual(trace["schema_version"], "cs.watch_rule_evaluation_trace.v1")
        self.assertEqual(trace["watch_rule_version_id"], first_version_id)
        self.assertFalse(trace["external_action_authority"])
        self.assertFalse(trace["provider_mutation_authority"])
        self.assertFalse(trace["action_card_created"])
        self.assertEqual(trace_payload["provider_internal_findings"], [])

        paused = run_json("watch", "rule", "pause", "--watch-rule-id", watch_rule_id, "--state-dir", self.state_rel)
        self.assertEqual(paused["watch_rule"]["status"], "paused")
        resumed = run_json("watch", "rule", "resume", "--watch-rule-id", watch_rule_id, "--state-dir", self.state_rel)
        self.assertEqual(resumed["watch_rule"]["status"], "active")

        edited = run_json(
            "watch",
            "rule",
            "edit",
            "--watch-rule-id",
            watch_rule_id,
            "--file",
            WATCH_RULE_EDIT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        edited_rule = edited["watch_rule"]
        second_version = edited["watch_rule_version"]
        self.assertEqual(edited_rule["status"], "active_pending_review")
        self.assertEqual(edited_rule["version_count"], 2)
        self.assertEqual(edited_rule["active_version_id"], first_version_id)
        self.assertEqual(edited_rule["current_version_id"], second_version["watch_rule_version_id"])
        self.assertEqual(second_version["version_number"], 2)
        self.assertEqual(second_version["previous_version_id"], first_version_id)
        self.assertEqual(second_version["version_diff"]["broadened_fields"], [])
        self.assertIn("match_criteria", second_version["version_diff"]["changed_fields"])

        trace_path = self.state_dir / "connector" / "watch_rule_evaluation_traces" / f"{trace['evaluation_trace_id']}.json"
        retained_trace = json.loads(trace_path.read_text())
        self.assertEqual(retained_trace["watch_rule_version_id"], first_version_id)

        cross_scope = run_cli(
            "watch",
            "rule",
            "show",
            "--watch-rule-id",
            watch_rule_id,
            "--namespace-id",
            "other",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(cross_scope.returncode, 6, cross_scope.stdout + cross_scope.stderr)
        cross_scope_payload = json.loads(cross_scope.stdout)
        self.assertIn("CS_SCOPE_DENIED", {error["code"] for error in cross_scope_payload["errors"]})

        deleted = run_json("watch", "rule", "delete", "--watch-rule-id", watch_rule_id, "--state-dir", self.state_rel)
        self.assertEqual(deleted["watch_rule"]["status"], "deleted")
        self.assertFalse(deleted["watch_rule"]["physical_delete_performed"])
        self.assertTrue(deleted["watch_rule"]["retained_for_audit"])

        self.assertEqual(len(list((self.state_dir / "connector" / "watch_rules").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "watch_rule_versions").glob("*.json"))), 2)
        self.assertEqual(len(list((self.state_dir / "connector" / "watch_rule_policy_decisions").glob("*.json"))), 4)
        self.assertEqual(len(list((self.state_dir / "connector" / "watch_rule_evaluation_traces").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "artifacts" / "records").glob("**/*.json"))), 0)
        state_text = state_file_texts(self.state_dir)
        for marker in [
            '"external_action_authority": true',
            '"external_action_execution_allowed": true',
            '"provider_mutation_allowed": true',
            "CS_CH_023_OWNERLESS_GLOBAL_RULE_MUST_NOT_EXIST",
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_connector_chrome_active_tab_capture_is_bounded_and_backend_validated_cs_ch_024(self) -> None:
        no_consent = run_cli(
            "connector",
            "capture",
            "browser",
            "active-tab",
            "--file",
            CHROME_ACTIVE_TAB_ALLOWED_FIXTURE,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(no_consent.returncode, 8, no_consent.stdout + no_consent.stderr)
        no_consent_payload = json.loads(no_consent.stdout)
        self.assertEqual(no_consent_payload["status"], "denied")
        no_consent_policy = no_consent_payload["chrome_active_tab_policy_decision"]
        self.assertEqual(no_consent_policy["schema_version"], "cs.connector_chrome_active_tab_policy_decision.v1")
        self.assertEqual(no_consent_policy["decision"], "deny")
        self.assertIn("CS_CHROME_ACTIVE_TAB_CONSENT_MISSING", no_consent_policy["reason_codes"])
        self.assertNotIn("chrome_active_tab_capture_summary", no_consent_payload)
        self.assertEqual(no_consent_payload["provider_internal_findings"], [])

        consent = run_json(
            "connector",
            "capture",
            "consent",
            "granted",
            "--source-id",
            "chrome_active_tab",
            "--purpose",
            "explicit Chrome active-tab summary capture",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(consent["status"], "success")
        self.assertTrue(consent["connector_watch_source_consent"]["explicit_owner_consent"])

        popup_blocked = run_cli(
            "connector",
            "capture",
            "browser",
            "active-tab",
            "--file",
            CHROME_ACTIVE_TAB_POPUP_BLOCKED_FIXTURE,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(popup_blocked.returncode, 8, popup_blocked.stdout + popup_blocked.stderr)
        popup_payload = json.loads(popup_blocked.stdout)
        popup_policy = popup_payload["chrome_active_tab_policy_decision"]
        self.assertEqual(popup_payload["status"], "denied")
        self.assertEqual(popup_policy["decision"], "deny")
        self.assertIn("CS_CHROME_ACTIVE_TAB_USER_GESTURE_REQUIRED", popup_policy["reason_codes"])
        self.assertIn("CS_CHROME_ACTIVE_TAB_POPUP_ONLY_DENIED", popup_policy["reason_codes"])
        self.assertIn("CS_CHROME_ACTIVE_TAB_BROWSER_INTERNAL_BLOCKED", popup_policy["reason_codes"])
        self.assertNotIn("chrome_active_tab_capture_summary", popup_payload)
        self.assertNotIn("capture_inbox_item", popup_payload)
        self.assertFalse(popup_payload["chrome_active_tab_payload"]["bounded_payload"]["raw_text_stored"])
        self.assertFalse(popup_payload["chrome_active_tab_payload"]["bounded_payload"]["raw_html_stored"])

        allowed = run_json(
            "connector",
            "capture",
            "browser",
            "active-tab",
            "--file",
            CHROME_ACTIVE_TAB_ALLOWED_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(allowed["status"], "success")
        self.assertEqual(allowed["provider_internal_findings"], [])
        permission_event = allowed["chrome_active_tab_permission_event"]
        active_payload = allowed["chrome_active_tab_payload"]
        policy = allowed["chrome_active_tab_policy_decision"]
        summary = allowed["chrome_active_tab_capture_summary"]
        inbox_item = allowed["capture_inbox_item"]
        self.assertEqual(permission_event["schema_version"], "cs.connector_chrome_active_tab_permission_event.v1")
        self.assertEqual(active_payload["schema_version"], "cs.connector_chrome_active_tab_payload.v1")
        self.assertEqual(summary["schema_version"], "cs.connector_chrome_active_tab_capture_summary.v1")
        self.assertEqual(inbox_item["schema_version"], "cs.capture_inbox_item.v1")
        self.assertEqual(policy["decision"], "allow")
        self.assertEqual(policy["reason_codes"], ["CS_CHROME_ACTIVE_TAB_POLICY_ALLOW"])
        self.assertTrue(policy["server_revalidated"])
        self.assertTrue(policy["checks"]["consent_active"])
        self.assertTrue(policy["checks"]["active_tab_permission_present"])
        self.assertTrue(policy["checks"]["user_gesture_present"])
        self.assertTrue(policy["checks"]["explicit_confirmation_present"])
        self.assertTrue(policy["checks"]["no_broad_all_urls_permission"])
        self.assertTrue(policy["checks"]["bounded_payload"])
        self.assertTrue(policy["checks"]["raw_browser_data_absent"])
        self.assertEqual(permission_event["permissions"], ["activeTab", "storage"])
        self.assertFalse(permission_event["broad_all_urls_permission"])
        self.assertFalse(active_payload["bounded_payload"]["raw_text_stored"])
        self.assertFalse(active_payload["bounded_payload"]["raw_html_stored"])
        self.assertFalse(summary["raw_text_stored"])
        self.assertFalse(summary["raw_html_stored"])
        self.assertFalse(summary["cookies_collected"])
        self.assertFalse(summary["screenshots_collected"])
        self.assertEqual(inbox_item["status"], "pending_review")
        self.assertTrue(inbox_item["owner_review_required"])
        self.assertTrue(inbox_item["can_save_as_evidence"])
        for payload in [no_consent_payload, popup_payload, allowed]:
            self.assertTrue(payload["audit_refs"])
            self.assertTrue(payload["evidence_refs"])
            for counter, value in payload["negative_evidence"].items():
                self.assertEqual(value, 0, counter)

        self.assertEqual(len(list((self.state_dir / "connector" / "watch_source_consents").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "chrome_active_tab_permission_events").glob("*.json"))), 2)
        self.assertEqual(len(list((self.state_dir / "connector" / "chrome_active_tab_payloads").glob("*.json"))), 2)
        self.assertEqual(len(list((self.state_dir / "connector" / "chrome_active_tab_policy_decisions").glob("*.json"))), 3)
        self.assertEqual(len(list((self.state_dir / "connector" / "chrome_active_tab_summaries").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "capture_inbox_items").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "artifacts" / "records").glob("**/*.json"))), 0)
        state_text = state_file_texts(self.state_dir)
        for marker in [
            "CS_CH_024_RAW_TEXT_MUST_NOT_PERSIST",
            "chrome://settings/passwords",
            '"raw_html_stored": true',
            '"raw_text_stored": true',
            '"broad_all_urls_permission": true',
            "<all_urls>",
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_connector_chrome_auto_capture_requires_two_sided_consent_cs_ch_025(self) -> None:
        no_config = run_cli(
            "connector",
            "capture",
            "browser",
            "auto-capture",
            "--file",
            CHROME_AUTO_CAPTURE_ALLOWED_FIXTURE,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(no_config.returncode, 8, no_config.stdout + no_config.stderr)
        no_config_payload = json.loads(no_config.stdout)
        self.assertEqual(no_config_payload["status"], "denied")
        no_config_policy = no_config_payload["chrome_auto_capture_policy_decision"]
        self.assertEqual(no_config_policy["schema_version"], "cs.connector_chrome_auto_capture_policy_decision.v1")
        self.assertEqual(no_config_policy["decision"], "deny")
        self.assertIn("CS_CHROME_AUTO_CAPTURE_CONSENT_MISSING", no_config_policy["reason_codes"])
        self.assertIn("CS_CHROME_AUTO_CAPTURE_CONFIG_MISSING", no_config_policy["reason_codes"])
        self.assertNotIn("chrome_auto_capture_summary", no_config_payload)
        self.assertNotIn("capture_inbox_item", no_config_payload)

        consent = run_json(
            "connector",
            "capture",
            "consent",
            "granted",
            "--source-id",
            "chrome_auto_capture",
            "--purpose",
            "allowlist Chrome auto capture for Project Alpha docs",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(consent["status"], "success")
        self.assertTrue(consent["connector_watch_source_consent"]["explicit_owner_consent"])

        config = run_json(
            "connector",
            "capture",
            "browser",
            "auto-config",
            "--file",
            CHROME_AUTO_CAPTURE_CONFIG_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        auto_config = config["chrome_auto_capture_config"]
        self.assertEqual(auto_config["schema_version"], "cs.connector_chrome_auto_capture_config.v1")
        self.assertEqual(auto_config["status"], "ready")
        self.assertTrue(auto_config["auto_capture_enabled"])
        self.assertTrue(auto_config["two_sided_consent"]["owner_rule_confirmed"])
        self.assertTrue(auto_config["two_sided_consent"]["site_allowance_present"])
        self.assertTrue(auto_config["two_sided_consent"]["source_pack_allowance_present"])
        self.assertTrue(auto_config["two_sided_consent"]["browser_permission_granted"])
        self.assertEqual(config["provider_internal_findings"], [])

        blocked = run_cli(
            "connector",
            "capture",
            "browser",
            "auto-capture",
            "--file",
            CHROME_AUTO_CAPTURE_BLOCKED_FIXTURE,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(blocked.returncode, 8, blocked.stdout + blocked.stderr)
        blocked_payload = json.loads(blocked.stdout)
        blocked_policy = blocked_payload["chrome_auto_capture_policy_decision"]
        blocked_trigger = blocked_payload["chrome_auto_capture_trigger"]
        self.assertEqual(blocked_payload["status"], "denied")
        self.assertEqual(blocked_policy["decision"], "deny")
        self.assertIn("CS_CHROME_AUTO_CAPTURE_SITE_NOT_ALLOWED", blocked_policy["reason_codes"])
        self.assertIn("CS_CHROME_AUTO_CAPTURE_SOURCE_PACK_NOT_ALLOWED", blocked_policy["reason_codes"])
        self.assertIn("CS_CHROME_AUTO_CAPTURE_CONSENT_VERSION_MISMATCH", blocked_policy["reason_codes"])
        self.assertIn("CS_CHROME_AUTO_CAPTURE_CONFIG_VERSION_MISMATCH", blocked_policy["reason_codes"])
        self.assertIn("CS_CHROME_AUTO_CAPTURE_TRIGGER_NOT_ALLOWED", blocked_policy["reason_codes"])
        self.assertIn("CS_CHROME_AUTO_CAPTURE_THROTTLED", blocked_policy["reason_codes"])
        self.assertIn("CS_CHROME_AUTO_CAPTURE_SESSION_LIMIT_REACHED", blocked_policy["reason_codes"])
        self.assertFalse(blocked_policy["checks"]["site_allowed"])
        self.assertFalse(blocked_policy["checks"]["source_pack_allowed"])
        self.assertFalse(blocked_policy["checks"]["consent_version_matches"])
        self.assertFalse(blocked_policy["checks"]["config_version_matches"])
        self.assertFalse(blocked_policy["checks"]["throttle_passed"])
        self.assertFalse(blocked_policy["checks"]["session_limit_passed"])
        self.assertIsNone(blocked_trigger["page"]["origin"])
        self.assertNotIn("chrome_auto_capture_summary", blocked_payload)
        self.assertNotIn("capture_inbox_item", blocked_payload)

        allowed = run_json(
            "connector",
            "capture",
            "browser",
            "auto-capture",
            "--file",
            CHROME_AUTO_CAPTURE_ALLOWED_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(allowed["status"], "success")
        self.assertEqual(allowed["provider_internal_findings"], [])
        allowed_trigger = allowed["chrome_auto_capture_trigger"]
        allowed_policy = allowed["chrome_auto_capture_policy_decision"]
        summary = allowed["chrome_auto_capture_summary"]
        inbox_item = allowed["capture_inbox_item"]
        self.assertEqual(allowed_trigger["schema_version"], "cs.connector_chrome_auto_capture_trigger.v1")
        self.assertEqual(allowed_policy["schema_version"], "cs.connector_chrome_auto_capture_policy_decision.v1")
        self.assertEqual(summary["schema_version"], "cs.connector_chrome_auto_capture_summary.v1")
        self.assertEqual(inbox_item["schema_version"], "cs.capture_inbox_item.v1")
        self.assertEqual(allowed_policy["decision"], "allow")
        self.assertEqual(allowed_policy["reason_codes"], ["CS_CHROME_AUTO_CAPTURE_POLICY_ALLOW"])
        self.assertTrue(allowed_policy["server_revalidated"])
        self.assertTrue(allowed_policy["checks"]["consent_active"])
        self.assertTrue(allowed_policy["checks"]["config_ready"])
        self.assertTrue(allowed_policy["checks"]["owner_rule_confirmed"])
        self.assertTrue(allowed_policy["checks"]["site_allowed"])
        self.assertTrue(allowed_policy["checks"]["source_pack_allowed"])
        self.assertTrue(allowed_policy["checks"]["browser_permission_granted"])
        self.assertTrue(allowed_policy["checks"]["consent_version_matches"])
        self.assertTrue(allowed_policy["checks"]["config_version_matches"])
        self.assertTrue(allowed_policy["checks"]["trigger_type_allowed"])
        self.assertTrue(allowed_policy["checks"]["active_allowed_page"])
        self.assertTrue(allowed_policy["checks"]["throttle_passed"])
        self.assertTrue(allowed_policy["checks"]["session_limit_passed"])
        self.assertTrue(allowed_policy["checks"]["idempotency_unique"])
        self.assertTrue(allowed_policy["checks"]["raw_browser_data_absent"])
        self.assertEqual(summary["source_origin"], "https://docs.project-alpha.example")
        self.assertFalse(summary["raw_text_stored"])
        self.assertFalse(summary["raw_html_stored"])
        self.assertEqual(inbox_item["status"], "pending_review")
        self.assertTrue(inbox_item["owner_review_required"])
        self.assertTrue(inbox_item["can_save_as_evidence"])

        duplicate = run_cli(
            "connector",
            "capture",
            "browser",
            "auto-capture",
            "--file",
            CHROME_AUTO_CAPTURE_ALLOWED_FIXTURE,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(duplicate.returncode, 8, duplicate.stdout + duplicate.stderr)
        duplicate_payload = json.loads(duplicate.stdout)
        duplicate_policy = duplicate_payload["chrome_auto_capture_policy_decision"]
        self.assertEqual(duplicate_payload["status"], "denied")
        self.assertIn("CS_CHROME_AUTO_CAPTURE_IDEMPOTENCY_DUPLICATE", duplicate_policy["reason_codes"])
        self.assertFalse(duplicate_policy["checks"]["idempotency_unique"])
        self.assertNotIn("chrome_auto_capture_summary", duplicate_payload)
        self.assertNotIn("capture_inbox_item", duplicate_payload)

        for payload in [no_config_payload, config, blocked_payload, allowed, duplicate_payload]:
            self.assertTrue(payload["audit_refs"])
            self.assertTrue(payload["evidence_refs"])
            self.assertEqual(payload["provider_internal_findings"], [])
            for counter, value in payload["negative_evidence"].items():
                self.assertEqual(value, 0, counter)

        self.assertEqual(len(list((self.state_dir / "connector" / "watch_source_consents").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "chrome_auto_capture_configs").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "chrome_auto_capture_triggers").glob("*.json"))), 2)
        self.assertEqual(len(list((self.state_dir / "connector" / "chrome_auto_capture_policy_decisions").glob("*.json"))), 4)
        self.assertEqual(len(list((self.state_dir / "connector" / "chrome_auto_capture_summaries").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "capture_inbox_items").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "artifacts" / "records").glob("**/*.json"))), 0)
        state_text = state_file_texts(self.state_dir)
        for marker in [
            "CS_CH_025_RAW_TEXT_MUST_NOT_PERSIST",
            "https://unapproved.example",
            '"raw_html_stored": true',
            '"raw_text_stored": true',
            '"broad_all_urls_permission": true',
            "<all_urls>",
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_connector_chrome_sensitive_page_policy_blocks_or_degrades_cs_ch_026(self) -> None:
        result = run_json(
            "connector",
            "capture",
            "browser",
            "sensitive-policy",
            "--file",
            CHROME_SENSITIVE_PAGE_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["provider_internal_findings"], [])
        summary = result["chrome_sensitive_page_summary"]
        policies = result["chrome_sensitive_page_policy_decisions"]
        degraded_payloads = result["chrome_sensitive_page_degraded_payloads"]
        history_items = result["chrome_sensitive_page_history_items"]
        self.assertEqual(summary["case_count"], 8)
        self.assertEqual(summary["blocked_count"], 6)
        self.assertEqual(summary["degraded_count"], 2)
        self.assertEqual(summary["allowed_count"], 0)
        self.assertFalse(summary["content_sent_to_models"])
        self.assertEqual(summary["searchable_content_artifacts_created"], 0)
        self.assertEqual(summary["capture_inbox_items_created"], 0)
        self.assertEqual(len(policies), 8)
        self.assertEqual(len(degraded_payloads), 2)
        self.assertEqual(len(history_items), 8)
        decisions = {policy["case_id"]: policy for policy in policies}
        self.assertEqual(decisions["password-page-client-block"]["decision"], "block")
        self.assertIn(
            "CS_CHROME_SENSITIVE_PASSWORD_FIELD_BLOCKED",
            decisions["password-page-client-block"]["reason_codes"],
        )
        self.assertEqual(decisions["payment-page-client-block"]["decision"], "block")
        self.assertIn(
            "CS_CHROME_SENSITIVE_PAYMENT_FIELD_BLOCKED",
            decisions["payment-page-client-block"]["reason_codes"],
        )
        self.assertEqual(decisions["secret-token-false-safe"]["decision"], "block")
        self.assertIn(
            "CS_CHROME_SENSITIVE_BACKEND_RECHECK_BLOCKED_FALSE_SAFE",
            decisions["secret-token-false-safe"]["reason_codes"],
        )
        self.assertEqual(decisions["mail-compose-degraded"]["decision"], "degraded")
        self.assertEqual(decisions["private-account-block"]["decision"], "block")
        self.assertEqual(decisions["browser-internal-block"]["decision"], "block")
        self.assertEqual(decisions["unsupported-scheme-block"]["decision"], "block")
        self.assertEqual(decisions["oversized-page-degraded"]["decision"], "degraded")
        for policy in policies:
            self.assertEqual(policy["schema_version"], "cs.connector_chrome_sensitive_page_policy_decision.v1")
            self.assertTrue(policy["server_revalidated"])
            self.assertTrue(policy["backend_restriction_preserved_or_increased"])
            self.assertTrue(policy["checks"]["client_block_not_downgraded"])
            self.assertTrue(policy["checks"]["backend_revalidated_sensitive_signals"])
            self.assertFalse(policy["checks"]["content_sent_to_models"])
            self.assertFalse(policy["checks"]["searchable_content_artifact_created"])
            self.assertFalse(policy["checks"]["capture_inbox_item_created"])
            self.assertTrue(policy["evidence_refs"])
            self.assertTrue(policy["audit_refs"])
            self.assertNotIn("url", policy["page"])
            self.assertNotIn("origin", policy["page"])
            self.assertNotIn("title", policy["page"])
        for degraded_payload in degraded_payloads:
            self.assertEqual(
                degraded_payload["schema_version"],
                "cs.connector_chrome_sensitive_page_degraded_payload.v1",
            )
            self.assertEqual(degraded_payload["restriction"], "metadata_hash_only")
            self.assertFalse(degraded_payload["raw_text_stored"])
            self.assertFalse(degraded_payload["raw_html_stored"])
            self.assertFalse(degraded_payload["content_sent_to_models"])
            self.assertFalse(degraded_payload["searchable_content_artifact_created"])
            self.assertFalse(degraded_payload["capture_inbox_item_created"])
            self.assertNotIn("url", degraded_payload["page_metadata"])
            self.assertNotIn("origin", degraded_payload["page_metadata"])
            self.assertNotIn("title", degraded_payload["page_metadata"])
        for history_item in history_items:
            self.assertEqual(history_item["schema_version"], "cs.connector_chrome_sensitive_page_history_item.v1")
            self.assertEqual(history_item["capture_inbox_surface"], "history_only")
            self.assertFalse(history_item["can_save_as_evidence"])
            self.assertIn("Capture was", history_item["ui_explanation"])
            self.assertTrue(history_item["safe_manual_alternative"])
            self.assertFalse(history_item["raw_text_stored"])
            self.assertFalse(history_item["raw_html_stored"])
            self.assertFalse(history_item["content_sent_to_models"])
        for counter, value in result["negative_evidence"].items():
            self.assertEqual(value, 0, counter)
        self.assertEqual(
            len(list((self.state_dir / "connector" / "chrome_sensitive_page_policy_decisions").glob("*.json"))),
            8,
        )
        self.assertEqual(
            len(list((self.state_dir / "connector" / "chrome_sensitive_page_degraded_payloads").glob("*.json"))),
            2,
        )
        self.assertEqual(
            len(list((self.state_dir / "connector" / "chrome_sensitive_page_history_items").glob("*.json"))),
            8,
        )
        self.assertEqual(len(list((self.state_dir / "connector" / "capture_inbox_items").glob("*.json"))), 0)
        self.assertEqual(len(list((self.state_dir / "artifacts" / "records").glob("**/*.json"))), 0)
        state_text = state_file_texts(self.state_dir)
        for marker in [
            "CS_CH_026_PASSWORD_RAW_TEXT_MUST_NOT_PERSIST",
            "CS_CH_026_PAYMENT_RAW_TEXT_MUST_NOT_PERSIST",
            "CS_CH_026_TOKEN_RAW_TEXT_MUST_NOT_PERSIST",
            "CS_CH_026_COMPOSE_RAW_TEXT_MUST_NOT_PERSIST",
            "CS_CH_026_PRIVATE_ACCOUNT_RAW_TEXT_MUST_NOT_PERSIST",
            "CS_CH_026_BROWSER_INTERNAL_RAW_TEXT_MUST_NOT_PERSIST",
            "CS_CH_026_UNSUPPORTED_SCHEME_RAW_TEXT_MUST_NOT_PERSIST",
            "CS_CH_026_OVERSIZED_RAW_TEXT_MUST_NOT_PERSIST",
            "chrome://settings/passwords",
            "file:///Users/local/private-note.html",
            "https://accounts.example/login",
            '"raw_text_included": true',
            '"raw_html_included": true',
            '"raw_text_stored": true',
            '"raw_html_stored": true',
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_connector_capture_lifecycle_controls_cs_ch_027(self) -> None:
        seed = run_json(
            "connector",
            "capture",
            "lifecycle",
            "seed",
            "--file",
            CAPTURE_LIFECYCLE_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(seed["status"], "success")
        self.assertEqual(seed["provider_internal_findings"], [])
        self.assertEqual(len(seed["capture_lifecycle_source_states"]), 4)
        for state in seed["capture_lifecycle_source_states"]:
            self.assertEqual(state["schema_version"], "cs.connector_capture_lifecycle_source_state.v1")
            self.assertTrue(state["configuration_retained"])
            self.assertFalse(state["collected_state_summary"]["raw_content_retained"])
            self.assertFalse(state["collected_state_summary"]["raw_browser_payload_retained"])
            for counter, value in state["negative_evidence"].items():
                self.assertEqual(value, 0, counter)

        pause = run_json(
            "connector",
            "capture",
            "lifecycle",
            "pause",
            "--source-id",
            "macos_activity",
            "--target-kind",
            "source",
            "--reason",
            "Owner paused local activity collection.",
            "--state-dir",
            self.state_rel,
        )
        pause_decision = pause["capture_lifecycle_decision"]
        self.assertEqual(pause_decision["resulting_status"], "paused")
        self.assertFalse(pause_decision["collection_enabled"])
        self.assertTrue(pause_decision["configuration_retained"])
        self.assertTrue(pause_decision["checks"]["pause_or_revoke_stops_future_capture"])

        paused_attempt = run_json(
            "connector",
            "capture",
            "lifecycle",
            "sample-attempt",
            "--source-id",
            "macos_activity",
            "--event-id",
            "sample-while-paused",
            "--state-dir",
            self.state_rel,
        )
        paused_decision = paused_attempt["capture_lifecycle_decision"]
        self.assertEqual(paused_decision["decision"], "deny")
        self.assertEqual(paused_decision["capture_samples_created"], 0)
        self.assertFalse(paused_decision["sample_created"])
        self.assertIn("CS_CAPTURE_LIFECYCLE_PAUSED_BLOCKS_SAMPLE", paused_decision["reason_codes"])

        resume = run_json(
            "connector",
            "capture",
            "lifecycle",
            "resume",
            "--source-id",
            "macos_activity",
            "--target-kind",
            "source",
            "--reason",
            "Owner resumed local activity collection.",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(resume["capture_lifecycle_decision"]["resulting_status"], "active")
        self.assertTrue(resume["capture_lifecycle_decision"]["collection_enabled"])

        active_attempt = run_json(
            "connector",
            "capture",
            "lifecycle",
            "sample-attempt",
            "--source-id",
            "macos_activity",
            "--event-id",
            "sample-after-resume-check",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(active_attempt["capture_lifecycle_decision"]["decision"], "allow")
        self.assertFalse(active_attempt["capture_lifecycle_decision"]["sample_created"])

        watch_pause = run_json(
            "connector",
            "capture",
            "lifecycle",
            "pause",
            "--source-id",
            "macos_activity",
            "--target-kind",
            "watch_rule",
            "--target-id",
            "wrule_project_alpha_capture",
            "--reason",
            "Pause one Watch Rule without removing configuration.",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(watch_pause["capture_lifecycle_source_state"]["target_kind"], "watch_rule")
        self.assertTrue(watch_pause["capture_lifecycle_source_state"]["configuration_retained"])

        global_pause = run_json(
            "connector",
            "capture",
            "lifecycle",
            "pause",
            "--source-id",
            "all_collection",
            "--target-kind",
            "global",
            "--target-id",
            "global_collection",
            "--reason",
            "Pause all collection.",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(global_pause["capture_lifecycle_source_state"]["target_kind"], "global")
        self.assertFalse(global_pause["capture_lifecycle_decision"]["future_capture_allowed"])

        revoke = run_json(
            "connector",
            "capture",
            "lifecycle",
            "revoke",
            "--source-id",
            "chrome_auto_capture",
            "--target-kind",
            "source",
            "--reason",
            "Owner revoked Chrome capture authority.",
            "--state-dir",
            self.state_rel,
        )
        revoke_decision = revoke["capture_lifecycle_decision"]
        self.assertEqual(revoke_decision["resulting_status"], "revoked")
        self.assertFalse(revoke_decision["collection_enabled"])
        self.assertTrue(revoke_decision["requires_new_consent"])

        revoked_attempt = run_json(
            "connector",
            "capture",
            "lifecycle",
            "sample-attempt",
            "--source-id",
            "chrome_auto_capture",
            "--event-id",
            "chrome-sample-after-revoke",
            "--state-dir",
            self.state_rel,
        )
        revoked_decision = revoked_attempt["capture_lifecycle_decision"]
        self.assertEqual(revoked_decision["decision"], "deny")
        self.assertEqual(revoked_decision["capture_samples_created"], 0)
        self.assertIn("CS_CAPTURE_LIFECYCLE_REVOKED_BLOCKS_SAMPLE", revoked_decision["reason_codes"])

        retention = run_json(
            "connector",
            "capture",
            "lifecycle",
            "retention",
            "--source-id",
            "macos_activity",
            "--target-kind",
            "source",
            "--retention-days",
            "7",
            "--reason",
            "Reduce derived sample retention.",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(retention["capture_lifecycle_source_state"]["retention_days"], 7)
        self.assertTrue(retention["capture_lifecycle_decision"]["checks"]["retention_boundary_visible"])

        exported = run_json(
            "connector",
            "capture",
            "lifecycle",
            "export",
            "--source-id",
            "macos_activity",
            "--include-history",
            "--state-dir",
            self.state_rel,
        )
        bundle = exported["capture_lifecycle_export"]
        self.assertEqual(bundle["schema_version"], "cs.connector_capture_lifecycle_export.v1")
        self.assertTrue(bundle["scoped_to_requested_source"])
        self.assertTrue(bundle["redacted"])
        self.assertFalse(bundle["raw_content_included"])
        self.assertFalse(bundle["raw_browser_payload_included"])
        self.assertFalse(bundle["credential_values_included"])
        self.assertGreaterEqual(bundle["state_count"], 2)
        self.assertTrue(all(state["source_id"] == "macos_activity" for state in bundle["states"]))
        self.assertTrue(any(state["lifecycle_history"] for state in bundle["states"]))

        saved = run_json(
            "connector",
            "capture",
            "lifecycle",
            "review-result",
            "--result-id",
            "capres_project_alpha_activity_summary",
            "--decision",
            "save",
            "--note",
            "Save as evidence-backed activity summary.",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(saved["capture_result_review"]["status"], "saved")
        self.assertTrue(saved["capture_result_review"]["saved_as_evidence"])
        self.assertTrue(saved["capture_result_review"]["saved_evidence_ref"].startswith("evidence:capture_result:"))

        dismissed = run_json(
            "connector",
            "capture",
            "lifecycle",
            "review-result",
            "--result-id",
            "capres_chrome_project_page_hint",
            "--decision",
            "dismiss",
            "--note",
            "Dismiss noisy browser hint.",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(dismissed["capture_result_review"]["status"], "dismissed")
        self.assertTrue(dismissed["capture_result_review"]["dismissed_from_inbox"])
        self.assertFalse(dismissed["capture_result_review"]["raw_browser_payload_stored"])

        dry_run = run_json(
            "connector",
            "capture",
            "lifecycle",
            "delete",
            "--source-id",
            "macos_activity",
            "--dry-run",
            "--reason",
            "Owner requested local fixture cleanup.",
            "--state-dir",
            self.state_rel,
        )
        dry_receipt = dry_run["capture_lifecycle_deletion_receipt"]
        self.assertEqual(dry_receipt["status"], "dry_run")
        self.assertGreaterEqual(dry_receipt["will_delete"], 2)
        self.assertGreaterEqual(dry_receipt["will_retain"], 2)
        self.assertGreaterEqual(dry_receipt["will_anonymize"], 1)
        self.assertTrue(dry_receipt["retained_audit_explanation"])
        self.assertFalse(dry_receipt["misleading_delete_everything_promise"])
        self.assertEqual(dry_receipt["audit_records_deleted"], 0)

        executed = run_json(
            "connector",
            "capture",
            "lifecycle",
            "delete",
            "--source-id",
            "macos_activity",
            "--execute",
            "--authorized",
            "--reason",
            "Authorized local fixture deletion.",
            "--state-dir",
            self.state_rel,
        )
        receipt = executed["capture_lifecycle_deletion_receipt"]
        self.assertEqual(receipt["status"], "executed")
        self.assertGreaterEqual(receipt["eligible_deleted_count"], 2)
        self.assertEqual(receipt["eligible_remaining_after_execute"], 0)
        self.assertEqual(receipt["audit_records_deleted"], 0)
        self.assertTrue(receipt["retained_audit_explanation"])
        for state in executed["capture_lifecycle_source_states"]:
            self.assertEqual(state["status"], "disabled")
            self.assertFalse(state["collection_enabled"])
            self.assertTrue(state["configuration_retained"])

        for payload in [
            seed,
            pause,
            paused_attempt,
            resume,
            active_attempt,
            watch_pause,
            global_pause,
            revoke,
            revoked_attempt,
            retention,
            exported,
            saved,
            dismissed,
            dry_run,
            executed,
        ]:
            self.assertEqual(payload["provider_internal_findings"], [])
            self.assertTrue(payload["audit_refs"])
            self.assertTrue(payload["evidence_refs"])
            for counter, value in payload.get("negative_evidence", {}).items():
                self.assertEqual(value, 0, counter)

        self.assertEqual(len(list((self.state_dir / "connector" / "capture_lifecycle_source_states").glob("*.json"))), 4)
        self.assertGreaterEqual(len(list((self.state_dir / "connector" / "capture_lifecycle_decisions").glob("*.json"))), 8)
        self.assertEqual(len(list((self.state_dir / "connector" / "capture_lifecycle_exports").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "capture_lifecycle_deletion_receipts").glob("*.json"))), 2)
        self.assertEqual(len(list((self.state_dir / "connector" / "capture_result_reviews").glob("*.json"))), 2)
        self.assertEqual(len(list((self.state_dir / "connector" / "capture_samples").glob("*.json"))), 0)
        state_text = state_file_texts(self.state_dir)
        for marker in [
            '"raw_content_included": true',
            '"raw_browser_payload_included": true',
            '"credential_values_included": true',
            '"delete_everything_promised": true',
            '"misleading_delete_everything_promise": true',
            '"audit_records_deleted": 1',
            '"provider_token"',
            '"auth_header"',
            '"private_key"',
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_watch_result_separates_observation_inference_proposal_cs_ch_028(self) -> None:
        built = run_json(
            "watch",
            "result",
            "build",
            "--file",
            WATCH_RESULT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(built["status"], "success")
        self.assertEqual(built["provider_internal_findings"], [])
        observations = built["watch_observations"]
        inferences = built["watch_inferences"]
        watch_result = built["watch_result"]
        self.assertEqual(len(observations), 2)
        self.assertEqual(len(inferences), 2)
        self.assertEqual(watch_result["schema_version"], "cs.watch_result.v1")
        self.assertEqual(watch_result["section_order"], ["Observation", "Inference", "Evidence/Caveats", "Proposed"])
        self.assertTrue(watch_result["checks"]["sections_separated"])
        self.assertTrue(watch_result["checks"]["connector_delivered_evidence_only"])
        self.assertTrue(watch_result["checks"]["product_intelligence_created_inference"])
        self.assertTrue(watch_result["checks"]["observed_records_contain_no_hypothesis"])
        self.assertTrue(watch_result["checks"]["inferences_not_observed_facts"])
        self.assertTrue(watch_result["checks"]["unsupported_or_low_confidence_stays_draft"])
        self.assertTrue(watch_result["checks"]["proposal_non_executing"])
        self.assertEqual(watch_result["trust_state"], "draft_hypothesis")
        self.assertFalse(watch_result["proposal"]["executed"])
        self.assertFalse(watch_result["proposal"]["workflow_run_started"])
        self.assertFalse(watch_result["proposal"]["provider_mutation"])
        self.assertTrue(watch_result["evidence_caveats"])
        for observation in observations:
            self.assertEqual(observation["schema_version"], "cs.watch_observation.v1")
            self.assertEqual(observation["section"], "Observation")
            self.assertTrue(observation["observed_only"])
            self.assertFalse(observation["contains_hypothesis"])
            self.assertFalse(observation["contains_proposal"])
            self.assertTrue(observation["inference_fields_absent"])
            self.assertFalse(observation["inferred_intent_labeled_as_observed"])
            self.assertTrue(observation["evidence_refs"])
            self.assertTrue(observation["audit_refs"])
            self.assertNotIn("hypothesis", observation["observed_facts"])
            self.assertNotIn("intent", json.dumps(observation["observed_facts"]))
        low_confidence_ids = []
        unsupported_ids = []
        for inference in inferences:
            self.assertEqual(inference["schema_version"], "cs.watch_inference.v1")
            self.assertEqual(inference["section"], "Inference")
            self.assertFalse(inference["stored_as_observed_fact"])
            self.assertTrue(inference["requires_owner_review"])
            self.assertTrue(inference["caveats"])
            self.assertTrue(inference["alternatives"])
            self.assertTrue(inference["evidence_refs"])
            self.assertTrue(inference["audit_refs"])
            if inference["low_confidence"]:
                low_confidence_ids.append(inference["watch_inference_id"])
                self.assertEqual(inference["trust_state"], "draft_hypothesis")
                self.assertFalse(inference["eligible_for_approved_memory"])
            if inference["unsupported"]:
                unsupported_ids.append(inference["watch_inference_id"])
                self.assertEqual(inference["trust_state"], "draft_hypothesis")
                self.assertFalse(inference["eligible_for_approved_memory"])
        self.assertTrue(low_confidence_ids)
        self.assertTrue(unsupported_ids)
        for counter, value in built["negative_evidence"].items():
            self.assertEqual(value, 0, counter)

        watch_result_id = watch_result["watch_result_id"]
        corrected = run_json(
            "watch",
            "result",
            "correct",
            "--watch-result-id",
            watch_result_id,
            "--inference-id",
            low_confidence_ids[0],
            "--hypothesis",
            "Project Alpha evidence should be reviewed; launch impact remains uncertain.",
            "--reason",
            "Owner corrected an over-specific risk interpretation.",
            "--state-dir",
            self.state_rel,
        )
        correction = corrected["watch_result_correction"]
        self.assertEqual(correction["schema_version"], "cs.watch_result_correction.v1")
        self.assertEqual(correction["changed_section"], "Inference")
        self.assertTrue(correction["observation_immutable"])
        self.assertEqual(correction["observation_hash_before"], correction["observation_hash_after"])
        self.assertFalse(correction["observation_section_changed"])
        self.assertEqual(corrected["watch_inference"]["trust_state"], "draft_hypothesis")
        for counter, value in corrected["negative_evidence"].items():
            self.assertEqual(value, 0, counter)

        denied = run_cli(
            "watch",
            "result",
            "approve-memory",
            "--watch-result-id",
            watch_result_id,
            "--inference-id",
            low_confidence_ids[0],
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(denied.returncode, 8, denied.stdout + denied.stderr)
        denied_payload = json.loads(denied.stdout)
        self.assertEqual(denied_payload["status"], "denied")
        denied_review = denied_payload["watch_result_review"]
        self.assertEqual(denied_review["status"], "denied")
        self.assertIn(
            "CS_WATCH_RESULT_LOW_CONFIDENCE_MEMORY_APPROVAL_DENIED",
            denied_review["reason_codes"],
        )
        self.assertFalse(denied_review["approved_memory_created"])
        self.assertFalse(denied_review["action_card_created"])
        self.assertFalse(denied_review["proposal_executed"])

        reviewed = run_json(
            "watch",
            "result",
            "review",
            "--watch-result-id",
            watch_result_id,
            "--decision",
            "save_draft_memory",
            "--note",
            "Keep as draft memory candidate only.",
            "--state-dir",
            self.state_rel,
        )
        review = reviewed["watch_result_review"]
        self.assertEqual(review["schema_version"], "cs.watch_result_review.v1")
        self.assertEqual(review["status"], "draft_memory_saved")
        self.assertTrue(review["draft_memory_saved"])
        self.assertFalse(review["approved_memory_created"])
        self.assertFalse(review["claim_created"])
        self.assertFalse(review["mission_opened"])
        self.assertFalse(review["action_card_created"])
        self.assertFalse(review["proposal_executed"])
        for payload in [built, corrected, reviewed]:
            self.assertEqual(payload["provider_internal_findings"], [])
            self.assertTrue(payload["audit_refs"])
            self.assertTrue(payload["evidence_refs"])
            for counter, value in payload.get("negative_evidence", {}).items():
                self.assertEqual(value, 0, counter)

        self.assertEqual(len(list((self.state_dir / "connector" / "watch_observations").glob("*.json"))), 2)
        self.assertEqual(len(list((self.state_dir / "connector" / "watch_inferences").glob("*.json"))), 2)
        self.assertEqual(len(list((self.state_dir / "connector" / "watch_results").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "watch_result_corrections").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "watch_result_reviews").glob("*.json"))), 2)
        self.assertEqual(len(list((self.state_dir / "actions").glob("**/*.json"))), 0)
        state_text = state_file_texts(self.state_dir)
        for marker in [
            '"inferred_intent_labeled_as_observed": true',
            '"stored_as_observed_fact": true',
            '"eligible_for_approved_memory": true',
            '"approved_memory_created": true',
            '"proposal_executed": true',
            '"action_card_created": true',
            '"claim_created": true',
            '"mission_opened": true',
            '"workflow_run_started": true',
            '"provider_mutation": true',
            '"external_call": true',
            '"raw_content_stored": true',
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_connector_action_preflight_combines_actioncard_dry_run_cs_ch_029(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")
        ingest = run_json(
            "connector",
            "delivery",
            "ingest",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        bundle = run_json(
            "connector",
            "evidence",
            "bundle",
            "create",
            "--delivery-receipt-id",
            ingest["connector_delivery_receipt"]["delivery_receipt_id"],
            "--query",
            "project alpha support action preflight evidence",
            "--state-dir",
            self.state_rel,
        )
        claim = run_json(
            "claim",
            "create",
            "--evidence-bundle-id",
            bundle["evidence_bundle"]["evidence_bundle_id"],
            "--statement",
            "Project Alpha support ticket CS-1001 should be updated from archived connector evidence.",
            "--state-dir",
            self.state_rel,
        )
        approved = run_json("claim", "approve", claim["claim"]["claim_id"], "--state-dir", self.state_rel)
        mission = run_json(
            "mission",
            "create",
            "--goal",
            "Prepare a governed supportdesk update from evidence-backed Project Alpha context.",
            "--claim-id",
            approved["claim"]["claim_id"],
            "--state-dir",
            self.state_rel,
        )
        proposed = run_json(
            "action",
            "propose",
            "--mission-id",
            mission["mission"]["mission_id"],
            "--claim-id",
            approved["claim"]["claim_id"],
            "--goal",
            "Update support ticket CS-1001 with the evidence-backed follow-up status.",
            "--action-kind",
            "external_writeback",
            "--risk",
            "high",
            "--connector",
            "supportdesk",
            "--target",
            "supportdesk:ticket:CS-1001",
            "--state-dir",
            self.state_rel,
        )
        action_id = proposed["action_card"]["action_id"]
        dry_run = run_json("action", "dry-run", action_id, "--state-dir", self.state_rel)
        self.assertEqual(dry_run["status"], "success")
        self.assertEqual(dry_run["action_card"]["policy_decision"]["decision"], "requires_approval")
        self.assertTrue(dry_run["action_card"]["policy_decision"]["approval_required"])
        self.assertEqual(dry_run["dry_run"]["expected_impact"]["expected_connector_calls"], 1)
        self.assertEqual(dry_run["dry_run"]["expected_impact"]["real_external_http_calls"], 0)
        self.assertFalse(dry_run["action_card"]["connector_boundary"]["direct_provider_access"])

        allowed = run_json(
            "connector",
            "action-preflight",
            "run",
            "--action-id",
            action_id,
            "--file",
            ACTION_PREFLIGHT_FIXTURE,
            "--case-id",
            "allowed",
            "--state-dir",
            self.state_rel,
        )
        preflight = allowed["connector_action_preflight"]
        review = allowed["connector_action_preflight_review"]
        self.assertEqual(preflight["schema_version"], "cs.connector_action_preflight.v1")
        self.assertEqual(review["schema_version"], "cs.connector_action_preflight_review.v1")
        self.assertEqual(preflight["decision"], "allow")
        self.assertEqual(review["status"], "owner_review_required")
        self.assertEqual(
            review["section_order"],
            [
                "Product Impact",
                "Connector Feasibility",
                "Permissions",
                "Source Policy",
                "Risk",
                "Idempotency",
                "Expected Calls",
                "Evidence",
                "Approval",
            ],
        )
        self.assertEqual(preflight["call_ledger"]["expected_provider_call_count"], 1)
        self.assertEqual(preflight["call_ledger"]["real_provider_call_count"], 0)
        self.assertEqual(preflight["call_ledger"]["external_http_calls"], 0)
        self.assertEqual(preflight["call_ledger"]["provider_mutations"], 0)
        self.assertFalse(preflight["risk"]["preflight_counts_as_approval"])
        self.assertFalse(review["approval"]["preflight_is_approval"])
        self.assertFalse(review["approval"]["execution_allowed"])
        self.assertEqual(allowed["action_card"]["connector_preflight"]["preflight_ref"], f"connector_action_preflight:{preflight['connector_action_preflight_id']}")
        self.assertEqual(allowed["action_card"]["connector_preflight"]["review_ref"], f"connector_action_preflight_review:{review['connector_action_preflight_review_id']}")
        self.assertFalse(allowed["action_card"]["execution"]["can_execute_now"])
        for counter, value in preflight["negative_evidence"].items():
            self.assertEqual(value, 0, counter)
        for counter, value in review["negative_evidence"].items():
            self.assertEqual(value, 0, counter)

        execute = run_cli("action", "execute", action_id, "--state-dir", self.state_rel, "--json")
        self.assertEqual(execute.returncode, 8, execute.stdout + execute.stderr)
        execute_payload = json.loads(execute.stdout)
        self.assertEqual(execute_payload["status"], "denied")
        self.assertIn("CS_ACTION_POLICY_DENIED", {error["code"] for error in execute_payload["errors"]})
        self.assertNotIn("action_result", execute_payload)

        denial_cases = {
            "undeclared_action": "CS_CONNECTOR_ACTION_PREFLIGHT_UNDECLARED_ACTION",
            "unsupported_provider": "CS_CONNECTOR_ACTION_PREFLIGHT_PROVIDER_UNSUPPORTED",
            "missing_permission": "CS_CONNECTOR_ACTION_PREFLIGHT_PERMISSION_MISSING",
            "invalid_input": "CS_CONNECTOR_ACTION_PREFLIGHT_INPUT_INVALID",
            "missing_idempotency": "CS_CONNECTOR_ACTION_PREFLIGHT_IDEMPOTENCY_REQUIRED",
        }
        for case_id, reason_code in denial_cases.items():
            result = run_cli(
                "connector",
                "action-preflight",
                "run",
                "--action-id",
                action_id,
                "--file",
                ACTION_PREFLIGHT_FIXTURE,
                "--case-id",
                case_id,
                "--state-dir",
                self.state_rel,
                "--json",
            )
            self.assertEqual(result.returncode, 8, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            denied_preflight = payload["connector_action_preflight"]
            denied_review = payload["connector_action_preflight_review"]
            self.assertEqual(payload["status"], "denied")
            self.assertEqual(denied_preflight["decision"], "deny")
            self.assertEqual(denied_review["status"], "blocked")
            self.assertIn(reason_code, denied_preflight["reason_codes"])
            self.assertEqual(denied_preflight["call_ledger"]["real_provider_call_count"], 0)
            self.assertEqual(denied_preflight["call_ledger"]["external_http_calls"], 0)
            self.assertEqual(denied_preflight["call_ledger"]["provider_mutations"], 0)

        github_action = run_json(
            "action",
            "propose",
            "--mission-id",
            mission["mission"]["mission_id"],
            "--claim-id",
            approved["claim"]["claim_id"],
            "--goal",
            "Attempt a GitHub issue comment that must remain outside ConnectorHub action preflight.",
            "--action-kind",
            "external_writeback",
            "--risk",
            "high",
            "--connector",
            "github",
            "--target",
            "github:repo:owner/project-alpha:issue:1001",
            "--state-dir",
            self.state_rel,
        )
        github_denied = run_cli(
            "connector",
            "action-preflight",
            "run",
            "--action-id",
            github_action["action_card"]["action_id"],
            "--file",
            ACTION_PREFLIGHT_FIXTURE,
            "--case-id",
            "github_read_only",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(github_denied.returncode, 8, github_denied.stdout + github_denied.stderr)
        github_payload = json.loads(github_denied.stdout)
        self.assertIn(
            "CS_CONNECTOR_ACTION_PREFLIGHT_GITHUB_READ_ONLY_DENIED",
            github_payload["connector_action_preflight"]["reason_codes"],
        )
        self.assertEqual(github_payload["connector_action_preflight"]["negative_evidence"]["github_read_only_action_admitted"], 0)

        self.assertEqual(len(list((self.state_dir / "connector" / "connector_action_preflights").glob("*.json"))), 7)
        self.assertEqual(len(list((self.state_dir / "connector" / "connector_action_preflight_reviews").glob("*.json"))), 7)
        self.assertEqual(len(list((self.state_dir / "workflow_runs").glob("*.json"))), 0)
        state_text = state_file_texts(self.state_dir)
        for marker in [
            '"preflight_counts_as_approval": true',
            '"preflight_is_approval": true',
            '"execution_result_created": true',
            '"workflow_run_started": true',
            '"real_provider_call_count": 1',
            '"external_http_calls": 1',
            '"provider_mutations": 1',
            '"direct_provider_access": 1',
            '"credential_values_exposed": 1',
            '"github_read_only_action_admitted": 1',
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_action_execution_requires_safety_envelope_cs_ch_030(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")
        ingest = run_json(
            "connector",
            "delivery",
            "ingest",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        bundle = run_json(
            "connector",
            "evidence",
            "bundle",
            "create",
            "--delivery-receipt-id",
            ingest["connector_delivery_receipt"]["delivery_receipt_id"],
            "--query",
            "project alpha action safety envelope evidence",
            "--state-dir",
            self.state_rel,
        )
        claim = run_json(
            "claim",
            "create",
            "--evidence-bundle-id",
            bundle["evidence_bundle"]["evidence_bundle_id"],
            "--statement",
            "Project Alpha support action execution requires every safety envelope gate.",
            "--state-dir",
            self.state_rel,
        )
        approved_claim = run_json("claim", "approve", claim["claim"]["claim_id"], "--state-dir", self.state_rel)
        mission = run_json(
            "mission",
            "create",
            "--goal",
            "Prepare a governed supportdesk execution safety envelope.",
            "--claim-id",
            approved_claim["claim"]["claim_id"],
            "--state-dir",
            self.state_rel,
        )
        proposed = run_json(
            "action",
            "propose",
            "--mission-id",
            mission["mission"]["mission_id"],
            "--claim-id",
            approved_claim["claim"]["claim_id"],
            "--goal",
            "Update support ticket CS-1001 only if every execution gate is valid.",
            "--action-kind",
            "external_writeback",
            "--risk",
            "high",
            "--connector",
            "supportdesk",
            "--target",
            "supportdesk:ticket:CS-1001",
            "--state-dir",
            self.state_rel,
        )
        action_id = proposed["action_card"]["action_id"]
        allowed = run_json(
            "connector",
            "action-preflight",
            "run",
            "--action-id",
            action_id,
            "--file",
            ACTION_PREFLIGHT_FIXTURE,
            "--case-id",
            "allowed",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(allowed["connector_action_preflight"]["decision"], "allow")

        invalid_approval = run_cli(
            "action",
            "approve",
            action_id,
            "--approver",
            "unauthorized_delegate",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(invalid_approval.returncode, 8, invalid_approval.stdout + invalid_approval.stderr)
        invalid_approval_payload = json.loads(invalid_approval.stdout)
        self.assertEqual(invalid_approval_payload["status"], "denied")
        self.assertIn("CS_ACTION_APPROVAL_DENIED", {error["code"] for error in invalid_approval_payload["errors"]})
        self.assertIn(
            "CS_ACTION_APPROVER_UNAUTHORIZED",
            {error.get("reason_code") for error in invalid_approval_payload["errors"]},
        )

        approved_action = run_json(
            "action",
            "approve",
            action_id,
            "--approver",
            "owner",
            "--state-dir",
            self.state_rel,
        )
        valid_action = approved_action["action_card"]
        action_path = self.state_dir / "actions" / f"{action_id}.json"

        def clone(record: dict) -> dict:
            return json.loads(json.dumps(record))

        def write_action(record: dict) -> None:
            action_path.write_text(json.dumps(record, sort_keys=True) + "\n")

        def execute_denied(expected_reason_code: str) -> dict:
            result = run_cli("action", "execute", action_id, "--state-dir", self.state_rel, "--json")
            self.assertEqual(result.returncode, 8, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "denied")
            self.assertIn(expected_reason_code, {error.get("reason_code") for error in payload["errors"]})
            envelope = payload["action_safety_envelope"]
            self.assertEqual(envelope["schema_version"], "cs.action_safety_envelope.v0")
            self.assertEqual(envelope["status"], "denied")
            self.assertEqual(envelope["reason_code"], expected_reason_code)
            self.assertEqual(envelope["external_http_calls"], 0)
            self.assertEqual(envelope["provider_mutations"], 0)
            self.assertEqual(envelope["real_provider_calls"], 0)
            self.assertFalse(envelope["execution_result_created"])
            self.assertFalse(envelope["workflow_run_started"])
            return payload

        missing_evidence_action = clone(valid_action)
        missing_evidence_action["evidence"]["artifact_refs"] = []
        write_action(missing_evidence_action)
        execute_denied("CS_ACTION_EVIDENCE_REQUIRED")

        write_action(valid_action)
        run_json("workspace", "mode", "set", "locked", "--state-dir", self.state_rel)
        execute_denied("CS_ACTION_POLICY_DENIED")
        run_json("workspace", "mode", "set", "assist", "--state-dir", self.state_rel)

        unapproved_action = clone(valid_action)
        unapproved_action["approval"]["status"] = "pending"
        unapproved_action["approval"]["approver"] = None
        unapproved_action["execution"]["can_execute_now"] = False
        write_action(unapproved_action)
        execute_denied("CS_ACTION_AUTHORIZED_APPROVAL_REQUIRED")

        unauthorized_action = clone(valid_action)
        unauthorized_action["approval"]["approver"] = "unauthorized_delegate"
        write_action(unauthorized_action)
        execute_denied("CS_ACTION_APPROVER_UNAUTHORIZED")

        write_action(valid_action)
        missing_permission = run_cli(
            "connector",
            "action-preflight",
            "run",
            "--action-id",
            action_id,
            "--file",
            ACTION_PREFLIGHT_FIXTURE,
            "--case-id",
            "missing_permission",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(missing_permission.returncode, 8, missing_permission.stdout + missing_permission.stderr)
        execute_denied("CS_ACTION_CONNECTOR_PERMISSION_REQUIRED")

        write_action(valid_action)
        missing_idempotency = run_cli(
            "connector",
            "action-preflight",
            "run",
            "--action-id",
            action_id,
            "--file",
            ACTION_PREFLIGHT_FIXTURE,
            "--case-id",
            "missing_idempotency",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(missing_idempotency.returncode, 8, missing_idempotency.stdout + missing_idempotency.stderr)
        execute_denied("CS_ACTION_IDEMPOTENCY_REQUIRED")

        stale_preflight_action = clone(valid_action)
        stale_preflight_action["dry_run"]["expected_impact"]["target"] = "supportdesk:ticket:CS-2002"
        write_action(stale_preflight_action)
        execute_denied("CS_ACTION_PREFLIGHT_STALE")

        stale_connector_action = clone(valid_action)
        stale_connector_action["connector_boundary"]["connector"] = "supportdesk_v2"
        write_action(stale_connector_action)
        execute_denied("CS_ACTION_PREFLIGHT_STALE")

        scoped = run_cli(
            "action",
            "execute",
            action_id,
            "--owner-id",
            "other-user",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(scoped.returncode, 6, scoped.stdout + scoped.stderr)
        scoped_payload = json.loads(scoped.stdout)
        self.assertIn("CS_SCOPE_DENIED", {error["code"] for error in scoped_payload["errors"]})

        safety_envelopes = list((self.state_dir / "actions" / "safety_envelopes").glob("*.json"))
        self.assertGreaterEqual(len(safety_envelopes), 8)
        state_text = state_file_texts(self.state_dir)
        for marker in [
            '"schema_version": "cs.action_result.v0"',
            '"status": "executed"',
            '"external_http_calls": 1',
            '"provider_mutations": 1',
            '"real_provider_calls": 1',
            '"execution_result_created": true',
            '"workflow_run_started": true',
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_declared_action_execution_reingests_outcome_cs_ch_031(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")
        ingest = run_json(
            "connector",
            "delivery",
            "ingest",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        bundle = run_json(
            "connector",
            "evidence",
            "bundle",
            "create",
            "--delivery-receipt-id",
            ingest["connector_delivery_receipt"]["delivery_receipt_id"],
            "--query",
            "project alpha connector action execution outcome",
            "--state-dir",
            self.state_rel,
        )
        claim = run_json(
            "claim",
            "create",
            "--evidence-bundle-id",
            bundle["evidence_bundle"]["evidence_bundle_id"],
            "--statement",
            "Project Alpha support action execution has enough evidence for a governed fixture writeback.",
            "--state-dir",
            self.state_rel,
        )
        approved_claim = run_json("claim", "approve", claim["claim"]["claim_id"], "--state-dir", self.state_rel)
        mission = run_json(
            "mission",
            "create",
            "--goal",
            "Execute a declared supportdesk action and re-ingest its outcome.",
            "--claim-id",
            approved_claim["claim"]["claim_id"],
            "--state-dir",
            self.state_rel,
        )
        proposed = run_json(
            "action",
            "propose",
            "--mission-id",
            mission["mission"]["mission_id"],
            "--claim-id",
            approved_claim["claim"]["claim_id"],
            "--goal",
            "Update support ticket CS-1001 through ConnectorHub and capture the provider outcome.",
            "--action-kind",
            "external_writeback",
            "--risk",
            "high",
            "--connector",
            "supportdesk",
            "--target",
            "supportdesk:ticket:CS-1001",
            "--state-dir",
            self.state_rel,
        )
        action_id = proposed["action_card"]["action_id"]
        preflight = run_json(
            "connector",
            "action-preflight",
            "run",
            "--action-id",
            action_id,
            "--file",
            ACTION_PREFLIGHT_FIXTURE,
            "--case-id",
            "allowed",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(preflight["connector_action_preflight"]["decision"], "allow")
        approved_action = run_json(
            "action",
            "approve",
            action_id,
            "--approver",
            "owner",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(approved_action["action_card"]["approval"]["status"], "approved")

        executed = run_json("action", "execute", action_id, "--state-dir", self.state_rel)
        result = executed["action_result"]
        workflow_run = executed["workflow_run"]
        receipt = executed["provider_receipt"]
        idempotency = executed["idempotency"]
        outcome_artifact = executed["outcome_artifact"]
        outcome_bundle = executed["outcome_evidence_bundle"]
        connected_outcome = executed["connected_outcome"]

        self.assertEqual(result["schema_version"], "cs.action_result.v0")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["side_effect_boundary"], "connectorhub_provider_pack_fixture")
        self.assertTrue(result["connectorhub_mediated"])
        self.assertFalse(result["direct_provider_access"])
        self.assertEqual(result["external_http_calls"], 0)
        self.assertEqual(result["fixture_provider_effect_count"], 1)
        self.assertEqual(result["duplicate_side_effect_count"], 0)
        self.assertEqual(workflow_run["schema_version"], "cs.workflow_run.v0")
        self.assertEqual(workflow_run["status"], "succeeded")
        self.assertEqual(workflow_run["execution_adapter"], "ConnectorHub")
        self.assertEqual(workflow_run["action_result_id"], result["action_result_id"])
        self.assertEqual(receipt["schema_version"], "cs.connector_provider_receipt.v1")
        self.assertEqual(receipt["status"], "success")
        self.assertEqual(receipt["provider_receipt_id"], result["provider_receipt_id"])
        self.assertEqual(receipt["duplicate_side_effect_count"], 0)
        self.assertFalse(receipt["direct_provider_access"])
        self.assertFalse(receipt["raw_provider_payload_persisted"])
        self.assertFalse(receipt["credential_values_exposed"])
        self.assertEqual(idempotency["schema_version"], "cs.connector_action_idempotency.v1")
        self.assertEqual(idempotency["status"], "committed")
        self.assertEqual(idempotency["duplicate_request"]["side_effect_count"], 0)
        self.assertEqual(outcome_artifact["schema_version"], "cs.artifact.v0")
        self.assertEqual(outcome_artifact["source"]["type"], "connector_action_outcome")
        self.assertEqual(outcome_bundle["schema_version"], "cs.evidence_bundle.v0")
        self.assertEqual(outcome_bundle["outcome_artifact_id"], outcome_artifact["artifact_id"])
        self.assertIn(f"action_result:{result['action_result_id']}", outcome_bundle["action_execution_refs"])
        self.assertIn(f"connector_provider_receipt:{receipt['provider_receipt_id']}", outcome_bundle["action_execution_refs"])
        self.assertEqual(connected_outcome["schema_version"], "cs.connected_outcome.v0")
        self.assertTrue(connected_outcome["source"]["connectorhub_mediated"])
        self.assertTrue(connected_outcome["source"]["reingested_as_evidence"])
        self.assertEqual(connected_outcome["source"]["external_http_calls"], 0)
        self.assertTrue(executed["audit_refs"])

        replay = run_json("action", "execute", action_id, "--state-dir", self.state_rel)
        self.assertEqual(replay["action_result"]["action_result_id"], result["action_result_id"])
        self.assertEqual(replay["provider_receipt"]["provider_receipt_id"], receipt["provider_receipt_id"])
        self.assertEqual(replay["idempotency"]["duplicate_request"]["side_effect_count"], 0)
        self.assertEqual(len(list((self.state_dir / "workflow_runs").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "action_results").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "provider_receipts").glob("*.json"))), 1)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_undeclared_action_and_provider_bypass_denied_cs_ch_032(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")
        ingest = run_json(
            "connector",
            "delivery",
            "ingest",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        bundle = run_json(
            "connector",
            "evidence",
            "bundle",
            "create",
            "--delivery-receipt-id",
            ingest["connector_delivery_receipt"]["delivery_receipt_id"],
            "--query",
            "project alpha undeclared connector action bypass proof",
            "--state-dir",
            self.state_rel,
        )
        claim = run_json(
            "claim",
            "create",
            "--evidence-bundle-id",
            bundle["evidence_bundle"]["evidence_bundle_id"],
            "--statement",
            "Connector Hub bypass attempts must be denied by the backend boundary.",
            "--state-dir",
            self.state_rel,
        )
        approved_claim = run_json("claim", "approve", claim["claim"]["claim_id"], "--state-dir", self.state_rel)
        mission = run_json(
            "mission",
            "create",
            "--goal",
            "Deny undeclared ConnectorHub actions and direct provider bypasses.",
            "--claim-id",
            approved_claim["claim"]["claim_id"],
            "--state-dir",
            self.state_rel,
        )
        proposed = run_json(
            "action",
            "propose",
            "--mission-id",
            mission["mission"]["mission_id"],
            "--claim-id",
            approved_claim["claim"]["claim_id"],
            "--goal",
            "Attempt an undeclared supportdesk delete action through ConnectorHub.",
            "--action-kind",
            "external_writeback",
            "--risk",
            "high",
            "--connector",
            "supportdesk",
            "--target",
            "supportdesk:ticket:CS-1001",
            "--state-dir",
            self.state_rel,
        )
        action_id = proposed["action_card"]["action_id"]

        undeclared = run_cli(
            "connector",
            "action-preflight",
            "run",
            "--action-id",
            action_id,
            "--file",
            ACTION_PREFLIGHT_FIXTURE,
            "--case-id",
            "undeclared_action",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(undeclared.returncode, 8, undeclared.stdout + undeclared.stderr)
        undeclared_payload = json.loads(undeclared.stdout)
        preflight = undeclared_payload["connector_action_preflight"]
        review = undeclared_payload["connector_action_preflight_review"]
        self.assertEqual(undeclared_payload["status"], "denied")
        self.assertEqual(preflight["decision"], "deny")
        self.assertIn("CS_CONNECTOR_ACTION_PREFLIGHT_UNDECLARED_ACTION", preflight["reason_codes"])
        self.assertFalse(preflight["gate_results"]["declared_action"])
        self.assertEqual(preflight["call_ledger"]["expected_provider_call_count"], 0)
        self.assertEqual(preflight["call_ledger"]["real_provider_call_count"], 0)
        self.assertEqual(preflight["call_ledger"]["external_http_calls"], 0)
        self.assertEqual(preflight["call_ledger"]["provider_mutations"], 0)
        self.assertEqual(preflight["negative_evidence"]["undeclared_actions_executed"], 0)
        self.assertEqual(preflight["negative_evidence"]["direct_provider_access"], 0)
        self.assertEqual(preflight["negative_evidence"]["provider_clients_exposed"], 0)
        self.assertEqual(preflight["negative_evidence"]["credential_values_exposed"], 0)
        self.assertEqual(review["status"], "blocked")
        self.assertEqual(review["no_side_effects"]["external_http_calls"], 0)
        self.assertEqual(review["no_side_effects"]["provider_mutations"], 0)
        self.assertEqual(review["no_side_effects"]["real_provider_calls"], 0)
        self.assertTrue(undeclared_payload["audit_refs"])

        approved_action = run_json(
            "action",
            "approve",
            action_id,
            "--approver",
            "owner",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(approved_action["action_card"]["approval"]["status"], "approved")
        blocked_execute = run_cli("action", "execute", action_id, "--state-dir", self.state_rel, "--json")
        self.assertEqual(blocked_execute.returncode, 8, blocked_execute.stdout + blocked_execute.stderr)
        blocked_payload = json.loads(blocked_execute.stdout)
        self.assertEqual(blocked_payload["status"], "denied")
        self.assertIn("CS_ACTION_PREFLIGHT_NOT_ALLOWED", {error.get("reason_code") for error in blocked_payload["errors"]})
        envelope = blocked_payload["action_safety_envelope"]
        self.assertEqual(envelope["status"], "denied")
        self.assertEqual(envelope["reason_code"], "CS_ACTION_PREFLIGHT_NOT_ALLOWED")
        self.assertEqual(envelope["external_http_calls"], 0)
        self.assertEqual(envelope["provider_mutations"], 0)
        self.assertEqual(envelope["real_provider_calls"], 0)
        self.assertFalse(envelope["execution_result_created"])
        self.assertFalse(envelope["workflow_run_started"])

        direct_write = run_cli(
            "connector",
            "direct-write-test",
            "--provider",
            "supportdesk",
            "--target",
            "supportdesk:ticket:CS-1001",
            "--operation",
            "direct_provider_sdk_patch",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(direct_write.returncode, 8, direct_write.stdout + direct_write.stderr)
        direct_payload = json.loads(direct_write.stdout)
        self.assertEqual(direct_payload["status"], "denied")
        self.assertEqual(direct_payload["direct_write_denial"]["external_http_calls"], 0)
        self.assertEqual(direct_payload["direct_write_denial"]["provider_mutations"], 0)
        self.assertFalse(direct_payload["direct_write_denial"]["direct_provider_access"])
        self.assertFalse(direct_payload["direct_write_denial"]["provider_client_exposed"])
        self.assertFalse(direct_payload["direct_write_denial"]["credential_values_exposed"])

        credential_boundary = run_json(
            "connector",
            "credential-boundary-test",
            "--provider",
            "supportdesk",
            "--capability",
            "support.ticket.write",
            "--state-dir",
            self.state_rel,
        )
        boundary = credential_boundary["credential_boundary"]
        self.assertEqual(boundary["credential_custody"], "connectorhub")
        self.assertFalse(boundary["credential_secret_value_present"])
        self.assertFalse(boundary["credentials_exposed_to_agent"])
        self.assertFalse(boundary["credentials_exposed_to_product_output"])
        self.assertFalse(boundary["direct_provider_access"])
        self.assertEqual(boundary["external_http_calls"], 0)

        pack_import = run_cli(
            "pack",
            "import",
            "--manifest",
            DIRECT_PROVIDER_PACK_FIXTURE,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(pack_import.returncode, 8, pack_import.stdout + pack_import.stderr)
        pack_payload = json.loads(pack_import.stdout)
        quarantine = pack_payload["quarantine"]
        self.assertEqual(pack_payload["status"], "failed")
        self.assertEqual(quarantine["status"], "quarantined")
        self.assertTrue(quarantine["direct_provider_logic_detected"])
        self.assertTrue(quarantine["forbidden_runtime"]["provider_clients"])
        self.assertTrue(quarantine["forbidden_runtime"]["extension_owned_credentials"])
        self.assertTrue(quarantine["forbidden_runtime"]["direct_api_writeback"])
        self.assertTrue(quarantine["forbidden_runtime"]["raw_secret_access"])
        self.assertTrue(pack_payload["audit_refs"])

        state_text = state_file_texts(self.state_dir)
        for marker in [
            '"schema_version": "cs.action_result.v0"',
            '"schema_version": "cs.workflow_run.v0"',
            '"external_http_calls": 1',
            '"provider_mutations": 1',
            '"real_provider_call_count": 1',
            '"direct_provider_access": true',
            '"provider_client_exposed": true',
            '"credential_values_exposed": true',
            '"credentials_exposed_to_agent": true',
            '"credential_secret_value_present": true',
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_action_retry_idempotent_and_compensation_visible_cs_ch_033(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")
        ingest = run_json(
            "connector",
            "delivery",
            "ingest",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        bundle = run_json(
            "connector",
            "evidence",
            "bundle",
            "create",
            "--delivery-receipt-id",
            ingest["connector_delivery_receipt"]["delivery_receipt_id"],
            "--query",
            "project alpha connector retry idempotency compensation evidence",
            "--state-dir",
            self.state_rel,
        )
        claim = run_json(
            "claim",
            "create",
            "--evidence-bundle-id",
            bundle["evidence_bundle"]["evidence_bundle_id"],
            "--statement",
            "ConnectorHub retries must return the existing result and expose compensation expectations.",
            "--state-dir",
            self.state_rel,
        )
        approved_claim = run_json("claim", "approve", claim["claim"]["claim_id"], "--state-dir", self.state_rel)
        mission = run_json(
            "mission",
            "create",
            "--goal",
            "Verify ConnectorHub idempotent retry and conflict handling.",
            "--claim-id",
            approved_claim["claim"]["claim_id"],
            "--state-dir",
            self.state_rel,
        )
        proposed = run_json(
            "action",
            "propose",
            "--mission-id",
            mission["mission"]["mission_id"],
            "--claim-id",
            approved_claim["claim"]["claim_id"],
            "--goal",
            "Update support ticket CS-1001 through ConnectorHub with retry-safe semantics.",
            "--action-kind",
            "external_writeback",
            "--risk",
            "high",
            "--connector",
            "supportdesk",
            "--target",
            "supportdesk:ticket:CS-1001",
            "--state-dir",
            self.state_rel,
        )
        action_id = proposed["action_card"]["action_id"]
        preflight = run_json(
            "connector",
            "action-preflight",
            "run",
            "--action-id",
            action_id,
            "--file",
            ACTION_PREFLIGHT_FIXTURE,
            "--case-id",
            "allowed",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(preflight["connector_action_preflight"]["decision"], "allow")
        run_json("action", "approve", action_id, "--approver", "owner", "--state-dir", self.state_rel)

        first = run_json("action", "execute", action_id, "--state-dir", self.state_rel)
        first_result = first["action_result"]
        first_receipt = first["provider_receipt"]
        first_idempotency = first["idempotency"]
        first_retry_status = first_idempotency["retry_status"]
        self.assertEqual(first_result["status"], "success")
        self.assertEqual(first_result["fixture_provider_effect_count"], 1)
        self.assertEqual(first_result["duplicate_side_effect_count"], 0)
        self.assertEqual(first_retry_status["status"], "completed_replayable")
        self.assertEqual(first_retry_status["duplicate_retry_count"], 0)
        self.assertTrue(first_retry_status["association_stored_before_response"])
        self.assertTrue(first_retry_status["timeout_before_response_reconciled"])
        self.assertTrue(first_retry_status["ambiguous_provider_response_reconciled"])
        self.assertTrue(first_retry_status["provider_duplicate_response_reconciled"])
        self.assertTrue(first_retry_status["process_restart_replay_safe"])
        self.assertTrue(first_idempotency["compensation"]["expectation_visible"])
        self.assertFalse(first_idempotency["compensation"]["automatic_compensation_executed"])

        replay = run_json("action", "execute", action_id, "--state-dir", self.state_rel)
        replay_idempotency = replay["idempotency"]
        replay_retry_status = replay_idempotency["retry_status"]
        self.assertEqual(replay["action_result"]["action_result_id"], first_result["action_result_id"])
        self.assertEqual(replay["workflow_run"]["workflow_run_id"], first["workflow_run"]["workflow_run_id"])
        self.assertEqual(replay["provider_receipt"]["provider_receipt_id"], first_receipt["provider_receipt_id"])
        self.assertEqual(replay_idempotency["idempotency_id"], first_idempotency["idempotency_id"])
        self.assertEqual(replay_idempotency["request_digest"], first_idempotency["request_digest"])
        self.assertEqual(replay_idempotency["duplicate_request"]["side_effect_count"], 0)
        self.assertTrue(replay_idempotency["duplicate_request"]["returned_existing_result"])
        self.assertGreaterEqual(replay_retry_status["duplicate_retry_count"], 1)
        self.assertTrue(replay_retry_status["same_key_same_request_digest_returned_existing_result"])
        self.assertTrue(replay_retry_status["timeout_before_response_reconciled"])
        self.assertTrue(replay_retry_status["ambiguous_provider_response_reconciled"])
        self.assertTrue(replay_retry_status["provider_duplicate_response_reconciled"])
        self.assertTrue(replay_retry_status["process_restart_replay_safe"])

        conflicting = run_json(
            "action",
            "propose",
            "--mission-id",
            mission["mission"]["mission_id"],
            "--claim-id",
            approved_claim["claim"]["claim_id"],
            "--goal",
            "Reuse the same idempotency key for a different support ticket and prove it is denied.",
            "--action-kind",
            "external_writeback",
            "--risk",
            "high",
            "--connector",
            "supportdesk",
            "--target",
            "supportdesk:ticket:CS-2002",
            "--state-dir",
            self.state_rel,
        )
        conflicting_action_id = conflicting["action_card"]["action_id"]
        conflict_preflight = run_json(
            "connector",
            "action-preflight",
            "run",
            "--action-id",
            conflicting_action_id,
            "--file",
            ACTION_PREFLIGHT_FIXTURE,
            "--case-id",
            "allowed_conflicting_intent",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(conflict_preflight["connector_action_preflight"]["decision"], "allow")
        run_json("action", "approve", conflicting_action_id, "--approver", "owner", "--state-dir", self.state_rel)
        conflict_result = run_cli("action", "execute", conflicting_action_id, "--state-dir", self.state_rel, "--json")
        self.assertEqual(conflict_result.returncode, 8, conflict_result.stdout + conflict_result.stderr)
        conflict_payload = json.loads(conflict_result.stdout)
        self.assertEqual(conflict_payload["status"], "denied")
        self.assertIn("CS_ACTION_IDEMPOTENCY_CONFLICT", {error.get("reason_code") for error in conflict_payload["errors"]})
        conflict = conflict_payload["idempotency_conflict"]
        self.assertEqual(conflict["status"], "conflict_rejected")
        self.assertEqual(conflict["reason_code"], "CS_ACTION_IDEMPOTENCY_CONFLICT")
        self.assertEqual(conflict["existing_action_id"], action_id)
        self.assertEqual(conflict["incoming_action_id"], conflicting_action_id)
        self.assertNotEqual(conflict["existing_request_digest"], conflict["incoming_request_digest"])
        self.assertEqual(conflict["duplicate_side_effect_count"], 0)
        self.assertEqual(conflict["external_http_calls"], 0)
        self.assertEqual(conflict["provider_mutations"], 0)
        self.assertEqual(conflict["real_provider_calls"], 0)
        self.assertTrue(conflict["compensation"]["expectation_visible"])
        self.assertFalse(conflict["compensation"]["automatic_compensation_executed"])
        self.assertTrue(conflict["compensation"]["separate_governed_action_required"])
        self.assertEqual(conflict_payload["idempotency"]["idempotency_id"], first_idempotency["idempotency_id"])
        self.assertEqual(conflict_payload["idempotency"]["conflict_attempt_count"], 1)
        self.assertTrue(conflict_payload["idempotency"]["conflicting_intent_rejected"])
        self.assertEqual(conflict_payload["action_safety_envelope"]["reason_code"], "CS_ACTION_IDEMPOTENCY_CONFLICT")
        self.assertEqual(conflict_payload["action_safety_envelope"]["external_http_calls"], 0)
        self.assertEqual(conflict_payload["action_safety_envelope"]["provider_mutations"], 0)
        self.assertFalse(conflict_payload["action_safety_envelope"]["execution_result_created"])
        self.assertFalse(conflict_payload["action_safety_envelope"]["workflow_run_started"])

        self.assertEqual(len(list((self.state_dir / "workflow_runs").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "action_results").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "provider_receipts").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "security" / "idempotency").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connected_outcomes").glob("*.json"))), 1)
        outcome_artifacts = [
            path
            for path in (self.state_dir / "artifacts" / "records").glob("**/*.json")
            if json.loads(path.read_text()).get("source", {}).get("type") == "connector_action_outcome"
        ]
        self.assertEqual(len(outcome_artifacts), 1)
        state_text = state_file_texts(self.state_dir)
        for marker in [
            '"duplicate_side_effect_count": 1',
            '"external_http_calls": 1',
            '"provider_mutations": 1',
            '"real_provider_calls": 1',
            '"automatic_compensation_executed": true',
            '"credential_values_exposed": true',
            '"direct_provider_access": true',
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_connector_scope_isolation_for_delivery_watch_and_action_cs_ch_034(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        contract_scope = validate["connector_capability_contract"]["scope"]
        self.assertEqual(contract_scope["owner_id"], "local-user")
        self.assertEqual(contract_scope["namespace_id"], "personal")

        setup = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(setup["connector_setup_result"]["scope"], contract_scope)
        self.assertEqual(setup["connector_source_policy"]["scope"], contract_scope)
        other_setup = run_cli(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--owner-id",
            "other-user",
            "--namespace-id",
            "other",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(other_setup.returncode, 6, other_setup.stdout + other_setup.stderr)
        other_setup_payload = json.loads(other_setup.stdout)
        self.assertIn("CS_SCOPE_DENIED", {error["code"] for error in other_setup_payload["errors"]})
        self.assertNotIn("connector_setup_result", other_setup_payload)

        delivery = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        receipt = delivery["connector_delivery_receipt"]
        artifact = delivery["artifact"]
        self.assertEqual(receipt["scope"], contract_scope)
        self.assertEqual(artifact["scope"], contract_scope)
        other_delivery = run_cli(
            "connector",
            "delivery",
            "process",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--owner-id",
            "other-user",
            "--namespace-id",
            "other",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(other_delivery.returncode, 6, other_delivery.stdout + other_delivery.stderr)
        other_delivery_payload = json.loads(other_delivery.stdout)
        self.assertIn("CS_SCOPE_DENIED", {error["code"] for error in other_delivery_payload["errors"]})
        self.assertNotIn("connector_delivery_receipt", other_delivery_payload)
        self.assertNotIn("artifact", other_delivery_payload)

        bundle = run_json(
            "connector",
            "evidence",
            "bundle",
            "create",
            "--delivery-receipt-id",
            receipt["delivery_receipt_id"],
            "--query",
            "project alpha scoped connector evidence",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(bundle["evidence_bundle"]["filters"], contract_scope)
        other_bundle = run_cli(
            "connector",
            "evidence",
            "bundle",
            "create",
            "--delivery-receipt-id",
            receipt["delivery_receipt_id"],
            "--query",
            "project alpha scoped connector evidence",
            "--owner-id",
            "other-user",
            "--namespace-id",
            "other",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(other_bundle.returncode, 6, other_bundle.stdout + other_bundle.stderr)
        other_bundle_payload = json.loads(other_bundle.stdout)
        self.assertIn("CS_SCOPE_DENIED", {error["code"] for error in other_bundle_payload["errors"]})
        self.assertNotIn("evidence_bundle", other_bundle_payload)

        watch = run_json(
            "watch",
            "result",
            "build",
            "--file",
            WATCH_RESULT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        watch_result = watch["watch_result"]
        self.assertEqual(watch_result["scope"], contract_scope)
        other_watch = run_cli(
            "watch",
            "result",
            "review",
            "--watch-result-id",
            watch_result["watch_result_id"],
            "--decision",
            "save_draft_memory",
            "--owner-id",
            "other-user",
            "--namespace-id",
            "other",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(other_watch.returncode, 6, other_watch.stdout + other_watch.stderr)
        other_watch_payload = json.loads(other_watch.stdout)
        self.assertIn("CS_SCOPE_DENIED", {error["code"] for error in other_watch_payload["errors"]})
        self.assertNotIn("watch_result", other_watch_payload)
        self.assertNotIn("watch_result_review", other_watch_payload)

        claim = run_json(
            "claim",
            "create",
            "--evidence-bundle-id",
            bundle["evidence_bundle"]["evidence_bundle_id"],
            "--statement",
            "ConnectorHub scoped records must not leak across owner namespaces.",
            "--state-dir",
            self.state_rel,
        )
        approved_claim = run_json("claim", "approve", claim["claim"]["claim_id"], "--state-dir", self.state_rel)
        mission = run_json(
            "mission",
            "create",
            "--goal",
            "Verify scoped ConnectorHub delivery, evidence, watch, and action boundaries.",
            "--claim-id",
            approved_claim["claim"]["claim_id"],
            "--state-dir",
            self.state_rel,
        )
        action = run_json(
            "action",
            "propose",
            "--mission-id",
            mission["mission"]["mission_id"],
            "--claim-id",
            approved_claim["claim"]["claim_id"],
            "--goal",
            "Update support ticket CS-1001 through ConnectorHub after scope verification.",
            "--action-kind",
            "external_writeback",
            "--risk",
            "high",
            "--connector",
            "supportdesk",
            "--target",
            "supportdesk:ticket:CS-1001",
            "--state-dir",
            self.state_rel,
        )
        action_id = action["action_card"]["action_id"]
        preflight = run_json(
            "connector",
            "action-preflight",
            "run",
            "--action-id",
            action_id,
            "--file",
            ACTION_PREFLIGHT_FIXTURE,
            "--case-id",
            "allowed",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(preflight["connector_action_preflight"]["scope"], contract_scope)
        approved_action = run_json("action", "approve", action_id, "--approver", "owner", "--state-dir", self.state_rel)
        self.assertEqual(approved_action["action_card"]["scope"], contract_scope)
        other_execute = run_cli(
            "action",
            "execute",
            action_id,
            "--owner-id",
            "other-user",
            "--namespace-id",
            "other",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(other_execute.returncode, 6, other_execute.stdout + other_execute.stderr)
        other_execute_payload = json.loads(other_execute.stdout)
        self.assertIn("CS_SCOPE_DENIED", {error["code"] for error in other_execute_payload["errors"]})
        self.assertNotIn("action_result", other_execute_payload)
        self.assertNotIn("workflow_run", other_execute_payload)
        self.assertNotIn("provider_receipt", other_execute_payload)

        self.assertEqual(len(list((self.state_dir / "connector" / "delivery_receipts").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "watch_results").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "actions").glob("*.json"))), 1)
        self.assertFalse((self.state_dir / "workflow_runs").exists())
        self.assertFalse((self.state_dir / "action_results").exists())
        self.assertFalse((self.state_dir / "connector" / "provider_receipts").exists())

        state_text = state_file_texts(self.state_dir)
        for marker in [
            '"owner_id": "other-user"',
            '"namespace_id": "other"',
            '"schema_version": "cs.action_result.v0"',
            '"workflow_run_started": true',
            '"external_http_calls": 1',
            '"provider_mutations": 1',
            '"credential_values_exposed": true',
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_connector_credentials_remain_connectorhub_custody_cs_ch_035(self) -> None:
        canary_id = "cs-ch-035-canary"
        connection_id = "conn_project_alpha"
        raw_canary = f"connectorhub-private-secret::github::{connection_id}::{canary_id}"

        status = run_json(
            "connector",
            "credential",
            "status",
            "--provider",
            "github",
            "--connection-id",
            connection_id,
            "--canary-id",
            canary_id,
            "--state-dir",
            self.state_rel,
        )
        rotated = run_json(
            "connector",
            "credential",
            "rotate",
            "--provider",
            "github",
            "--connection-id",
            connection_id,
            "--canary-id",
            canary_id,
            "--state-dir",
            self.state_rel,
        )
        revoked = run_json(
            "connector",
            "credential",
            "revoke",
            "--provider",
            "github",
            "--connection-id",
            connection_id,
            "--canary-id",
            canary_id,
            "--state-dir",
            self.state_rel,
        )
        boundary_payload = run_json(
            "connector",
            "credential-boundary-test",
            "--provider",
            "github",
            "--capability",
            "source_control.issue.read",
            "--state-dir",
            self.state_rel,
        )

        lifecycle_records = [
            status["connector_credential_lifecycle"],
            rotated["connector_credential_lifecycle"],
            revoked["connector_credential_lifecycle"],
        ]
        for payload in [status, rotated, revoked, boundary_payload]:
            self.assertTrue(payload["evidence_refs"])
            self.assertTrue(payload["audit_refs"])
            self.assertNotIn(raw_canary, json.dumps(payload, sort_keys=True))

        for record in lifecycle_records:
            self.assertEqual(record["schema_version"], "cs.connector_credential_lifecycle.v1")
            self.assertEqual(record["credential_custody"], "connectorhub")
            self.assertEqual(record["secret_manager_boundary"], "ConnectorHub")
            self.assertEqual(record["credential_ref"], "connectorhub://credential/github/conn_project_alpha")
            self.assertEqual(record["secret_canary_id"], canary_id)
            self.assertTrue(record["credential_fingerprint"])
            self.assertFalse(record["raw_secret_value_present"])
            self.assertFalse(record["raw_handle_present"])
            self.assertFalse(record["auth_header_present"])
            self.assertFalse(record["credential_bearing_url_present"])
            self.assertFalse(record["credentials_exposed_to_agent"])
            self.assertFalse(record["credentials_exposed_to_product_output"])
            self.assertFalse(record["credentials_exposed_to_logs"])
            self.assertFalse(record["credentials_exposed_to_exports"])
            self.assertEqual(record["product_secret_writes"], 0)
            self.assertEqual(record["raw_secret_reads"], 0)
            self.assertEqual(record["external_http_calls"], 0)
            self.assertEqual(record["provider_mutations"], 0)

        self.assertEqual(status["connector_credential_lifecycle"]["connection_status"]["status"], "active")
        self.assertEqual(rotated["connector_credential_lifecycle"]["operation"], "rotate")
        self.assertEqual(rotated["connector_credential_lifecycle"]["connection_status"]["status"], "active")
        self.assertIsNotNone(rotated["connector_credential_lifecycle"]["connection_status"]["last_rotated_at"])
        self.assertEqual(revoked["connector_credential_lifecycle"]["operation"], "revoke")
        self.assertEqual(revoked["connector_credential_lifecycle"]["connection_status"]["status"], "revoked")
        self.assertTrue(revoked["connector_credential_lifecycle"]["connection_status"]["revocation_recorded"])

        boundary = boundary_payload["credential_boundary"]
        self.assertEqual(boundary["schema_version"], "cs.connector_credential_boundary_test.v0")
        self.assertEqual(boundary["credential_custody"], "connectorhub")
        self.assertFalse(boundary["credential_secret_value_present"])
        self.assertFalse(boundary["credentials_exposed_to_agent"])
        self.assertFalse(boundary["credentials_exposed_to_product_output"])
        self.assertFalse(boundary["direct_provider_access"])
        self.assertEqual(boundary["raw_secret_reads"], 0)
        self.assertEqual(boundary["external_http_calls"], 0)

        self.assertEqual(len(list((self.state_dir / "security" / "credential_lifecycle").glob("*.json"))), 3)
        self.assertEqual(len(list((self.state_dir / "security" / "credential_boundaries").glob("*.json"))), 1)
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn(raw_canary, state_text)
        for marker in [
            '"raw_secret_value_present": true',
            '"credential_secret_value_present": true',
            '"raw_handle_present": true',
            '"auth_header_present": true',
            '"credential_bearing_url_present": true',
            '"credentials_exposed_to_agent": true',
            '"credentials_exposed_to_product_output": true',
            '"credentials_exposed_to_logs": true',
            '"credentials_exposed_to_exports": true',
            '"product_secret_writes": 1',
            '"raw_secret_reads": 1',
            '"external_http_calls": 1',
            '"provider_mutations": 1',
        ]:
            self.assertNotIn(marker, state_text)
        audit = run_json("audit", "verify", "--state-dir", self.state_rel)
        self.assertEqual(audit["audit_integrity"]["status"], "success")

    def test_connector_source_control_projection_family_cs_ch_016(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")

        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        setup = plan["connector_setup_result"]
        projection_types = {
            projection_type
            for item in setup["product_projection_contract"]
            for projection_type in item["projection_types"]
        }
        expected = {
            "repository": (REPOSITORY_DELIVERY_FIXTURE, "source_control.repository.v1"),
            "commit": (COMMIT_DELIVERY_FIXTURE, "source_control.commit.v1"),
            "change": (CHANGE_DELIVERY_FIXTURE, "source_control.change.v1"),
            "issue": (DELIVERY_FIXTURE, "source_control.issue.v1"),
            "file_snapshot": (FILE_SNAPSHOT_DELIVERY_FIXTURE, "source_control.file_snapshot.v1"),
        }
        self.assertTrue({item[1] for item in expected.values()}.issubset(projection_types))
        self.assertFalse(setup["product_handler_contract"]["requires_provider_sdk"])

        receipts = {}
        artifacts = {}
        for key, (fixture, projection_type) in expected.items():
            processed = run_json(
                "connector",
                "delivery",
                "process",
                "--file",
                fixture,
                "--contract-id",
                "ccon_project_alpha_github",
                "--state-dir",
                self.state_rel,
            )
            self.assertEqual(processed["status"], "success")
            receipt = processed["connector_delivery_receipt"]
            artifact = processed["artifact"]
            snapshot = processed["connector_projection_snapshot"]
            policy = processed["connector_projection_policy_decision"]
            ack = processed["connector_ack_outbox"]
            content_version = processed["connector_content_version"]
            self.assertEqual(receipt["projection_type"], projection_type)
            self.assertEqual(snapshot["projection_type"], projection_type)
            self.assertEqual(policy["decision"], "allow")
            self.assertTrue(policy["selected_resource_allowed"])
            self.assertEqual(receipt["source_summary"]["source_ref"], "github:repo:owner/project-alpha")
            self.assertTrue(receipt["source_external_id"].startswith("github:repo:owner/project-alpha"))
            self.assertTrue(receipt["source_revision"])
            self.assertEqual(receipt["source_revision"], snapshot["source_revision"])
            self.assertEqual(receipt["source_revision"], content_version["source_revision"])
            self.assertEqual(receipt["acknowledgement_state"], "acknowledged_after_commit")
            self.assertEqual(ack["status"], "acknowledged")
            self.assertTrue(ack["ack_sent"])
            self.assertEqual(ack["artifact_id"], artifact["artifact_id"])
            self.assertFalse(receipt["raw_provider_payload"]["stored_in_product_state"])
            self.assertFalse(artifact["connector_delivery"]["raw_provider_payload_stored_in_product_state"])
            self.assertEqual(processed["provider_internal_findings"], [])
            receipts[key] = receipt
            artifacts[key] = artifact

            bundle_payload = run_json(
                "connector",
                "evidence",
                "bundle",
                "create",
                "--delivery-receipt-id",
                receipt["delivery_receipt_id"],
                "--query",
                f"source-control {key} project-alpha",
                "--state-dir",
                self.state_rel,
            )
            self.assertEqual(bundle_payload["status"], "success")
            bundle = bundle_payload["evidence_bundle"]
            search = bundle_payload["search_snapshot"]
            link = bundle_payload["connector_evidence_bundle_link"]
            self.assertEqual(search["result_count"], 1)
            self.assertEqual(search["results"][0]["artifact_id"], artifact["artifact_id"])
            self.assertEqual(search["results"][0]["source_revision"], receipt["source_revision"])
            self.assertEqual(link["artifact_id"], artifact["artifact_id"])
            self.assertEqual(link["delivery_receipt_id"], receipt["delivery_receipt_id"])
            self.assertFalse(bundle["raw_provider_payload_available"])
            self.assertFalse(link["coverage"]["raw_provider_payload_included"])
            self.assertEqual(bundle_payload["provider_internal_findings"], [])

        self.assertEqual(len(list((self.state_dir / "connector" / "delivery_receipts").glob("*.json"))), 5)
        self.assertEqual(len(list((self.state_dir / "artifacts" / "records").glob("**/*.json"))), 5)
        self.assertEqual(len(list((self.state_dir / "connector" / "ack_outbox").glob("*.json"))), 5)
        self.assertEqual(len(list((self.state_dir / "connector" / "content_versions").glob("*.json"))), 5)
        self.assertEqual(len(list((self.state_dir / "evidence" / "bundles").glob("*.json"))), 5)
        self.assertEqual(len(list((self.state_dir / "search" / "snapshots").glob("*.json"))), 5)
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn('"provider_token"', state_text)
        self.assertNotIn("CS_CH_016_RAW_OR_WRITE_MUST_NOT_EXIST", state_text)

    def test_connector_github_content_restrictions_and_sensitive_hygiene_cs_ch_018(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")

        redacted = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            SECRET_MARKER_FILE_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        redaction_decision = redacted["connector_content_restriction_decision"]
        redacted_payload = redaction_decision["normalized_payload"]
        self.assertEqual(redacted["status"], "success")
        self.assertEqual(redaction_decision["schema_version"], "cs.connector_content_restriction_decision.v1")
        self.assertEqual(redaction_decision["action"], "redact")
        self.assertTrue(redaction_decision["redaction_applied"])
        self.assertEqual(redaction_decision["sensitive_marker_scan"]["matches_detected"], 2)
        self.assertIn("payload.markdown_excerpt", redaction_decision["sensitive_marker_scan"]["redacted_fields"])
        self.assertIn("[REDACTED:", redacted_payload["markdown_excerpt"])
        self.assertNotIn("CS_CH_018_SECRET_CANARY_MUST_NOT_PERSIST", redacted_payload["markdown_excerpt"])
        self.assertTrue(redacted["artifact"]["connector_delivery"]["artifact_input_sanitized"])
        self.assertFalse(redacted["artifact"]["connector_delivery"]["exact_envelope_preserved"])
        self.assertEqual(redacted["connector_delivery_receipt"]["source_policy_enforcement"]["content_policy_action"], "redact")
        self.assertEqual(redacted["provider_internal_findings"], [])

        binary = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            BINARY_FILE_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        binary_decision = binary["connector_content_restriction_decision"]
        self.assertEqual(binary["status"], "success")
        self.assertEqual(binary_decision["action"], "metadata_only")
        self.assertTrue(binary_decision["binary_content"])
        self.assertTrue(binary_decision["metadata_only"])
        self.assertNotIn("markdown_excerpt", binary_decision["normalized_payload"])
        self.assertEqual(binary["connector_delivery_receipt"]["source_policy_enforcement"]["partial_status"], "metadata_only_binary")

        large = run_json(
            "connector",
            "delivery",
            "process",
            "--file",
            LARGE_FILE_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        large_decision = large["connector_content_restriction_decision"]
        self.assertEqual(large["status"], "success")
        self.assertEqual(large_decision["action"], "metadata_only")
        self.assertEqual(large_decision["partial_status"], "metadata_only_size_limit")
        self.assertGreater(large_decision["declared_size_bytes"], large_decision["max_content_bytes"])
        self.assertNotIn("markdown_excerpt", large_decision["normalized_payload"])

        forbidden = run_cli(
            "connector",
            "delivery",
            "process",
            "--file",
            FORBIDDEN_PATH_FILE_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(forbidden.returncode, 1, forbidden.stdout + forbidden.stderr)
        forbidden_payload = json.loads(forbidden.stdout)
        forbidden_decision = forbidden_payload["connector_content_restriction_decision"]
        self.assertEqual(forbidden_payload["status"], "failed")
        self.assertEqual(forbidden_decision["action"], "skip")
        self.assertEqual(forbidden_decision["partial_status"], "skipped_path_denied")
        self.assertIn("CS_CONNECTOR_SOURCE_POLICY_PATH_DENIED", [error["code"] for error in forbidden_payload["errors"]])
        self.assertNotIn("artifact", forbidden_payload)
        self.assertNotIn("connector_delivery_receipt", forbidden_payload)

        generated = run_cli(
            "connector",
            "delivery",
            "process",
            "--file",
            GENERATED_FILE_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(generated.returncode, 1, generated.stdout + generated.stderr)
        generated_payload = json.loads(generated.stdout)
        generated_decision = generated_payload["connector_content_restriction_decision"]
        self.assertEqual(generated_decision["action"], "skip")
        self.assertTrue(generated_decision["generated_content"])
        self.assertIn("CS_CONNECTOR_CONTENT_GENERATED_SKIPPED", [error["code"] for error in generated_payload["errors"]])
        self.assertNotIn("artifact", generated_payload)

        quarantined = run_cli(
            "connector",
            "delivery",
            "process",
            "--file",
            PRIVATE_KEY_FILE_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(quarantined.returncode, 7, quarantined.stdout + quarantined.stderr)
        quarantined_payload = json.loads(quarantined.stdout)
        quarantine_decision = quarantined_payload["connector_content_restriction_decision"]
        quarantine = quarantined_payload["connector_delivery_quarantine"]
        self.assertEqual(quarantined_payload["status"], "quarantined")
        self.assertEqual(quarantine_decision["action"], "quarantine")
        self.assertEqual(quarantine_decision["partial_status"], "quarantined_sensitive_material")
        self.assertIn("private_key_block", quarantine_decision["sensitive_marker_scan"]["marker_types"])
        self.assertEqual(quarantine["failure_class"], "content_restriction")
        self.assertFalse(quarantine["safe_diagnostics"]["raw_provider_payload_persisted"])
        self.assertNotIn("artifact", quarantined_payload)
        self.assertNotIn("connector_delivery_receipt", quarantined_payload)
        self.assertEqual(quarantined_payload["provider_internal_findings"], [])

        self.assertEqual(len(list((self.state_dir / "connector" / "content_restriction_decisions").glob("*.json"))), 6)
        self.assertEqual(len(list((self.state_dir / "connector" / "delivery_receipts").glob("*.json"))), 3)
        self.assertEqual(len(list((self.state_dir / "artifacts" / "records").glob("**/*.json"))), 3)
        self.assertEqual(len(list((self.state_dir / "connector" / "ack_outbox").glob("*.json"))), 3)
        self.assertEqual(len(list((self.state_dir / "connector" / "quarantine").glob("*.json"))), 1)
        persisted_decisions = [
            json.loads(path.read_text())
            for path in sorted((self.state_dir / "connector" / "content_restriction_decisions").glob("*.json"))
        ]
        self.assertTrue(all(decision["evidence_refs"] for decision in persisted_decisions))
        self.assertTrue(all(decision["audit_refs"] for decision in persisted_decisions))
        by_partial_status = {decision["partial_status"]: decision for decision in persisted_decisions}
        for partial_status, payload in {
            "redacted_sensitive_markers": redacted,
            "metadata_only_binary": binary,
            "metadata_only_size_limit": large,
        }.items():
            decision = by_partial_status[partial_status]
            linked = decision["linked_records"]
            self.assertEqual(linked["artifact_id"], payload["artifact"]["artifact_id"])
            self.assertEqual(linked["delivery_receipt_id"], payload["connector_delivery_receipt"]["delivery_receipt_id"])
            self.assertIn(f"artifact:{payload['artifact']['artifact_id']}", decision["evidence_refs"])
            self.assertIn(
                f"connector_delivery_receipt:{payload['connector_delivery_receipt']['delivery_receipt_id']}",
                decision["evidence_refs"],
            )
        self.assertEqual(
            by_partial_status["quarantined_sensitive_material"]["linked_records"]["quarantine_id"],
            quarantine["quarantine_id"],
        )
        self.assertIn(
            f"connector_delivery_quarantine:{quarantine['quarantine_id']}",
            by_partial_status["quarantined_sensitive_material"]["evidence_refs"],
        )
        for partial_status in ("skipped_path_denied", "skipped_generated_content"):
            linked = by_partial_status[partial_status]["linked_records"]
            self.assertEqual(linked["rejected_delivery_id"], by_partial_status[partial_status]["delivery_id"])
            self.assertNotIn("artifact_id", linked)
            self.assertNotIn("delivery_receipt_id", linked)

        state_text = state_file_texts(self.state_dir)
        self.assertNotIn("CS_CH_018_SECRET_CANARY_MUST_NOT_PERSIST", state_text)
        self.assertNotIn("ghp_CSCH018SECRET000000", state_text)
        self.assertNotIn("CS_CH_018_PRIVATE_KEY_MUST_NOT_PERSIST", state_text)
        self.assertNotIn("-----BEGIN PRIVATE KEY-----", state_text)
        self.assertNotIn("CS_CH_018_FORBIDDEN_PATH_MUST_NOT_IMPORT", state_text)
        self.assertNotIn('"provider_token"', state_text)

    def test_connector_incremental_sync_idempotent_cursor_replay_cs_ch_017(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")

        cursor_id = "github:repo:owner/project-alpha:issues"
        bad_webhook = run_cli(
            "connector",
            "sync",
            "incremental",
            "--file",
            BAD_WEBHOOK_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--signal",
            "webhook",
            "--cursor-id",
            cursor_id,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(bad_webhook.returncode, 1, bad_webhook.stdout + bad_webhook.stderr)
        bad_webhook_payload = json.loads(bad_webhook.stdout)
        self.assertEqual(bad_webhook_payload["status"], "failed")
        self.assertIn(
            "CS_CONNECTOR_WEBHOOK_SIGNATURE_INVALID",
            {error["code"] for error in bad_webhook_payload["errors"]},
        )
        self.assertFalse(
            bad_webhook_payload["connector_sync_signal_receipt"]["origin_verified_inside_connector_boundary"]
        )
        self.assertFalse(bad_webhook_payload["connector_sync_signal_receipt"]["webhook_signature_verified"])
        self.assertEqual(bad_webhook_payload["cursor_update"]["status"], "not_advanced")
        self.assertFalse(bad_webhook_payload["cursor_update"]["cursor_advanced_before_commit"])
        self.assertFalse((self.state_dir / "connector" / "delivery_receipts").exists())
        self.assertFalse((self.state_dir / "connector" / "sync_cursors").exists())

        webhook = run_json(
            "connector",
            "sync",
            "incremental",
            "--file",
            DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--signal",
            "webhook",
            "--cursor-id",
            cursor_id,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(webhook["status"], "success")
        self.assertFalse(webhook["deduplicated"])
        self.assertEqual(webhook["cursor_update"]["status"], "advanced")
        self.assertEqual(webhook["cursor_update"]["reason"], "advanced_after_durable_commit")
        self.assertFalse(webhook["cursor_update"]["cursor_advanced_before_commit"])
        self.assertTrue(webhook["connector_sync_signal_receipt"]["origin_verified_inside_connector_boundary"])
        self.assertTrue(webhook["connector_sync_signal_receipt"]["webhook_signature_verified"])
        event_key_parts = webhook["connector_sync_signal_receipt"]["provider_event_key_parts"]
        self.assertEqual(event_key_parts["provider_installation_id"], "github-installation:project-alpha-readonly")
        self.assertEqual(event_key_parts["repository_ref"], "github:repo:owner/project-alpha")
        self.assertEqual(event_key_parts["object_ref"], ISSUE_SOURCE_EXTERNAL_ID)
        self.assertEqual(event_key_parts["action"], "updated")
        self.assertEqual(event_key_parts["source_revision"], "issue:1001:2026-06-23T00:00:00Z")
        first_receipt_id = webhook["connector_delivery_receipt"]["delivery_receipt_id"]
        first_artifact_id = webhook["artifact"]["artifact_id"]
        first_version_id = webhook["connector_content_version"]["content_version_id"]

        poll_duplicate = run_json(
            "connector",
            "sync",
            "incremental",
            "--file",
            DUPLICATE_EVENT_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--signal",
            "poll",
            "--cursor-id",
            cursor_id,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(poll_duplicate["status"], "success")
        self.assertTrue(poll_duplicate["deduplicated"])
        self.assertEqual(poll_duplicate["connector_delivery_receipt"]["delivery_receipt_id"], first_receipt_id)
        self.assertEqual(poll_duplicate["artifact"]["artifact_id"], first_artifact_id)
        self.assertEqual(poll_duplicate["sync_provider_event_key"], webhook["sync_provider_event_key"])
        self.assertEqual(poll_duplicate["cursor_update"]["status"], "observed")
        self.assertEqual(poll_duplicate["cursor_update"]["reason"], "duplicate_or_replay_observed")

        crash_before_cursor = run_cli(
            "connector",
            "sync",
            "incremental",
            "--file",
            CHANGED_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--signal",
            "poll",
            "--cursor-id",
            cursor_id,
            "--fault-mode",
            "after_commit_before_cursor",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(crash_before_cursor.returncode, 5, crash_before_cursor.stdout + crash_before_cursor.stderr)
        crash_payload = json.loads(crash_before_cursor.stdout)
        self.assertEqual(crash_payload["status"], "interrupted")
        self.assertEqual(crash_payload["crash_point"], "after_commit_before_cursor")
        self.assertEqual(crash_payload["cursor_update"]["status"], "not_advanced")
        self.assertFalse(crash_payload["cursor_update"]["cursor_advanced_before_commit"])

        gap_reconcile = run_cli(
            "connector",
            "sync",
            "reconcile",
            "--cursor-id",
            cursor_id,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(gap_reconcile.returncode, 4, gap_reconcile.stdout + gap_reconcile.stderr)
        gap_reconciliation = json.loads(gap_reconcile.stdout)["connector_sync_reconciliation"]
        self.assertEqual(gap_reconciliation["status"], "failed")
        self.assertEqual(gap_reconciliation["unobserved_delivery_receipt_count"], 1)
        self.assertEqual(gap_reconciliation["sync_lag_metrics"]["unobserved_delivery_receipt_count"], 1)
        self.assertEqual(gap_reconciliation["cursor_advanced_before_commit_count"], 0)

        replay_changed = run_json(
            "connector",
            "sync",
            "incremental",
            "--file",
            CHANGED_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--signal",
            "poll",
            "--cursor-id",
            cursor_id,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(replay_changed["status"], "success")
        self.assertTrue(replay_changed["deduplicated"])
        self.assertEqual(replay_changed["cursor_update"]["status"], "advanced")
        self.assertEqual(replay_changed["cursor_update"]["reason"], "advanced_after_durable_commit")
        self.assertEqual(replay_changed["connector_content_version"]["version_ordinal"], 2)
        self.assertEqual(replay_changed["connector_content_version"]["predecessor_content_version_id"], first_version_id)
        changed_version_id = replay_changed["connector_content_version"]["content_version_id"]

        crash_after_cursor = run_cli(
            "connector",
            "sync",
            "incremental",
            "--file",
            CHANGED_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--signal",
            "poll",
            "--cursor-id",
            cursor_id,
            "--fault-mode",
            "after_cursor",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(crash_after_cursor.returncode, 5, crash_after_cursor.stdout + crash_after_cursor.stderr)
        after_cursor_payload = json.loads(crash_after_cursor.stdout)
        self.assertEqual(after_cursor_payload["status"], "interrupted")
        self.assertEqual(after_cursor_payload["crash_point"], "after_cursor")
        self.assertEqual(after_cursor_payload["cursor_update"]["status"], "observed")
        self.assertFalse(after_cursor_payload["cursor_update"]["cursor_advanced_before_commit"])

        replay_after_cursor = run_json(
            "connector",
            "sync",
            "incremental",
            "--file",
            CHANGED_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--signal",
            "poll",
            "--cursor-id",
            cursor_id,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(replay_after_cursor["status"], "success")
        self.assertTrue(replay_after_cursor["deduplicated"])
        self.assertEqual(replay_after_cursor["cursor_update"]["status"], "observed")
        self.assertEqual(replay_after_cursor["connector_content_version"]["content_version_id"], changed_version_id)

        out_of_order = run_json(
            "connector",
            "sync",
            "incremental",
            "--file",
            UNCHANGED_EVENT_DELIVERY_FIXTURE,
            "--contract-id",
            "ccon_project_alpha_github",
            "--signal",
            "webhook",
            "--cursor-id",
            cursor_id,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(out_of_order["status"], "success")
        self.assertTrue(out_of_order["deduplicated"])
        self.assertEqual(out_of_order["cursor_update"]["status"], "observed")
        self.assertEqual(out_of_order["cursor_update"]["reason"], "out_of_order_source_revision_observed")
        self.assertEqual(out_of_order["connector_sync_cursor"]["cursor_value"], "2026-06-23T00:02:00Z")

        reconcile = run_json(
            "connector",
            "sync",
            "reconcile",
            "--cursor-id",
            cursor_id,
            "--state-dir",
            self.state_rel,
        )
        reconciliation = reconcile["connector_sync_reconciliation"]
        self.assertEqual(reconcile["status"], "success")
        self.assertEqual(reconciliation["status"], "success")
        self.assertEqual(reconciliation["cursor_count"], 1)
        self.assertEqual(reconciliation["delivery_receipt_count"], 2)
        self.assertEqual(reconciliation["artifact_count"], 2)
        self.assertEqual(reconciliation["dedupe_record_count"], 2)
        self.assertEqual(reconciliation["duplicate_logical_artifact_count"], 0)
        self.assertEqual(reconciliation["cursor_advanced_before_commit_count"], 0)
        self.assertGreaterEqual(reconciliation["duplicate_or_replay_observation_count"], 3)
        self.assertEqual(reconciliation["out_of_order_observation_count"], 1)
        self.assertEqual(reconciliation["missing_cursor_receipt_count"], 0)
        self.assertEqual(reconciliation["missing_cursor_artifact_count"], 0)
        self.assertEqual(reconciliation["unobserved_delivery_receipt_count"], 0)
        self.assertEqual(reconciliation["sync_lag_metrics"]["unobserved_delivery_receipt_count"], 0)
        self.assertEqual(set(reconciliation["signals_seen"]), {"poll", "webhook"})

        lineage = run_json(
            "connector",
            "lineage",
            "show",
            "--contract-id",
            "ccon_project_alpha_github",
            "--source-external-id",
            ISSUE_SOURCE_EXTERNAL_ID,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(lineage["connector_content_lineage"]["version_count"], 2)
        self.assertTrue(lineage["connector_content_lineage"]["one_current_logical_truth"])
        self.assertEqual(lineage["connector_content_lineage"]["duplicate_active_truth_count"], 0)

        self.assertEqual(len(list((self.state_dir / "connector" / "sync_cursors").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "sync_signal_receipts").glob("*.json"))), 3)
        self.assertEqual(len(list((self.state_dir / "connector" / "sync_reconciliations").glob("*.json"))), 1)
        self.assertEqual(len(list((self.state_dir / "connector" / "delivery_receipts").glob("*.json"))), 2)
        self.assertEqual(len(list((self.state_dir / "artifacts" / "records").glob("**/*.json"))), 2)
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn('"provider_token"', state_text)
        self.assertNotIn("CS_CH_017_CURSOR_ADVANCED_BEFORE_COMMIT_MUST_NOT_EXIST", state_text)

    def test_connector_setup_plan_degrades_optional_missing_capability_cs_ch_003(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            OPTIONAL_MISSING_CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        self.assertEqual(validate["ids"]["contract_id"], "ccon_project_alpha_optional_missing")

        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_optional_missing",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(plan["status"], "success")
        setup = plan["connector_setup_result"]
        self.assertEqual(setup["readiness"], "ready_with_gaps")
        self.assertEqual(setup["activation_state"], "planned_ready")
        self.assertTrue(setup["activation_allowed"])
        self.assertTrue(setup["required_capabilities_available"])
        self.assertEqual(setup["blocked_reason_code"], None)
        self.assertEqual(setup["provider_call_ledger"]["before_activation"], 0)
        self.assertEqual(setup["provider_call_ledger"]["during_plan"], 0)
        self.assertIn(f"fixture:{OPTIONAL_MISSING_CONTRACT_FIXTURE}", setup["verification_refs"])
        self.assertEqual(len(setup["delivery_streams"]), 1)
        self.assertEqual(len(setup["gaps"]), 1)
        self.assertFalse(setup["gaps"][0]["required"])
        self.assertEqual(setup["gaps"][0]["common_capability"], "source_control.pull_request.read")

        availability = {item["common_capability"]: item for item in setup["feature_availability"]}
        self.assertTrue(availability["source_control.repository.read"]["enabled"])
        self.assertEqual(availability["source_control.repository.read"]["state"], "enabled")
        self.assertFalse(availability["source_control.pull_request.read"]["enabled"])
        self.assertEqual(availability["source_control.pull_request.read"]["state"], "disabled_optional_unavailable")
        self.assertEqual(
            availability["source_control.pull_request.read"]["reason_code"],
            "CS_CONNECTOR_OPTIONAL_CAPABILITY_UNAVAILABLE",
        )
        self.assertEqual(len(setup["disabled_surfaces"]), 1)
        self.assertEqual(setup["disabled_surfaces"][0]["surface"], "Pull request evidence")
        self.assertFalse(setup["disabled_surfaces"][0]["required"])
        self.assertEqual(plan["provider_internal_findings"], [])
        self.assertTrue(plan["evidence_refs"])
        self.assertTrue(plan["audit_refs"])

        setup_path = self.state_dir / "connector" / "setup_results" / f"{setup['setup_result_id']}.json"
        self.assertTrue(setup_path.exists())

    def test_connector_source_policy_confirm_or_override_cs_ch_004(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        base_policy = plan["connector_source_policy"]
        self.assertEqual(base_policy["allowed_paths"], ["docs/**", "README.md"])

        confirm = run_json(
            "connector",
            "source-policy",
            "confirm",
            "--contract-id",
            "ccon_project_alpha_github",
            "--allowed-path",
            "docs/**",
            "--max-content-bytes",
            "100000",
            "--retention-days",
            "30",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(confirm["status"], "success")
        policy = confirm["connector_source_policy"]
        diff = confirm["source_policy_diff"]
        self.assertEqual(policy["confirmation"]["kind"], "owner_override")
        self.assertTrue(policy["confirmation"]["owner_confirmed"])
        self.assertEqual(policy["confirmation"]["confirmed_by"], "local-user")
        self.assertFalse(policy["confirmation"]["silent_broadening"])
        self.assertTrue(policy["constraints_never_broadened_silently"])
        self.assertEqual(policy["allowed_paths"], ["docs/**"])
        self.assertEqual(policy["max_content_bytes"], 100000)
        self.assertEqual(policy["retention_days"], 30)
        self.assertEqual(policy["compatibility_decision"]["status"], "compatible")
        self.assertFalse(policy["compatibility_decision"]["broadened"])
        self.assertIn("allowed_paths", diff["narrowed_fields"])
        self.assertIn("max_content_bytes", diff["narrowed_fields"])
        self.assertIn("retention_days", diff["narrowed_fields"])
        self.assertNotEqual(policy["source_policy_id"], base_policy["source_policy_id"])
        self.assertEqual(confirm["provider_internal_findings"], [])
        self.assertTrue(confirm["evidence_refs"])
        self.assertTrue(confirm["audit_refs"])

        policy_path = self.state_dir / "connector" / "source_policies" / f"{policy['source_policy_id']}.json"
        self.assertTrue(policy_path.exists())

        broaden = run_cli(
            "connector",
            "source-policy",
            "confirm",
            "--contract-id",
            "ccon_project_alpha_github",
            "--allowed-path",
            "docs/**",
            "--allowed-path",
            "secrets/**",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(broaden.returncode, 8, broaden.stdout + broaden.stderr)
        denied = json.loads(broaden.stdout)
        self.assertEqual(denied["status"], "denied")
        self.assertIn("CS_CONNECTOR_SOURCE_POLICY_BROADENING_DENIED", {error["code"] for error in denied["errors"]})
        self.assertTrue(denied["audit_refs"])
        self.assertNotIn("connector_source_policy", denied)

    def test_connector_provider_swap_keeps_product_contract_cs_ch_005(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        default_plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--state-dir",
            self.state_rel,
        )
        alternate_plan = run_json(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--provider-pack-id",
            "local_source_control_readonly_alt.v1",
            "--state-dir",
            self.state_rel,
        )
        default_setup = default_plan["connector_setup_result"]
        alternate_setup = alternate_plan["connector_setup_result"]
        self.assertEqual(default_setup["readiness"], "ready")
        self.assertEqual(alternate_setup["readiness"], "ready")
        self.assertEqual(default_setup["product_handler_contract"], alternate_setup["product_handler_contract"])
        self.assertEqual(default_setup["product_projection_contract"], alternate_setup["product_projection_contract"])
        self.assertEqual(default_setup["product_object_preview"], alternate_setup["product_object_preview"])
        self.assertNotEqual(
            default_setup["source_policy_snapshot"]["selected_provider_pack_id"],
            alternate_setup["source_policy_snapshot"]["selected_provider_pack_id"],
        )
        self.assertNotEqual(
            default_setup["source_policy_snapshot"]["provider_pack_ids"],
            alternate_setup["source_policy_snapshot"]["provider_pack_ids"],
        )
        self.assertIn("provider_pack:local_source_control_readonly.v1", default_setup["verification_refs"])
        self.assertIn("provider_pack:local_source_control_readonly_alt.v1", alternate_setup["verification_refs"])
        self.assertEqual(default_setup["provider_call_ledger"]["during_plan"], 0)
        self.assertEqual(alternate_setup["provider_call_ledger"]["during_plan"], 0)
        self.assertEqual(default_plan["provider_internal_findings"], [])
        self.assertEqual(alternate_plan["provider_internal_findings"], [])

    def test_connector_permission_gap_status_is_safe_cs_ch_006(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        result = run_cli(
            "connector",
            "setup",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--provider-pack-id",
            "local_source_control_permission_gap.v1",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "blocked")
        setup = payload["connector_setup_result"]
        self.assertEqual(setup["readiness"], "blocked")
        self.assertEqual(setup["activation_state"], "blocked_permission_required")
        self.assertFalse(setup["activation_allowed"])
        self.assertEqual(setup["blocked_reason_code"], "CS_CONNECTOR_PERMISSION_REQUIRED")
        self.assertEqual(setup["provider_call_ledger"]["before_activation"], 0)
        self.assertEqual(setup["provider_call_ledger"]["during_plan"], 0)
        explanation = setup["status_explanation"]
        self.assertEqual(explanation["reason_code"], "CS_CONNECTOR_PERMISSION_REQUIRED")
        self.assertIn("read-only selected-repository grant", explanation["cause"])
        self.assertIn("streams stay unavailable", explanation["impact"])
        self.assertTrue(explanation["resolution_steps"])
        self.assertTrue(explanation["safe_to_show_to_owner"])
        self.assertFalse(any(explanation["redaction"].values()))
        self.assertEqual(payload["provider_internal_findings"], [])
        lowered = result.stdout.lower()
        self.assertNotIn("bearer ", lowered)
        self.assertNotIn("ghp_", lowered)
        self.assertNotIn("sk-", lowered)
        self.assertNotIn("raw provider response:", lowered)
        self.assertTrue(payload["audit_refs"])

    def test_connector_upgrade_plan_pins_versions_and_blocks_breaking_change_cs_ch_038(self) -> None:
        validate = run_json(
            "connector",
            "contract",
            "validate",
            "--file",
            CONTRACT_FIXTURE,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(validate["status"], "success")
        compatible = run_json(
            "connector",
            "upgrade",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--target-provider-pack-id",
            "local_source_control_readonly_alt.v1",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(compatible["status"], "success")
        compatible_plan = compatible["connector_upgrade_plan"]
        self.assertEqual(compatible_plan["compatibility"]["status"], "compatible")
        self.assertTrue(compatible_plan["pinned_versions_remain_active"])
        self.assertTrue(compatible_plan["activation_blocked_until_reviewed"])
        self.assertTrue(compatible_plan["rollback_available"])
        self.assertEqual(
            compatible_plan["migration_plan"]["rollback"]["provider_pack_id"],
            "local_source_control_readonly.v1",
        )
        self.assertEqual(compatible["provider_internal_findings"], [])
        self.assertTrue(compatible["audit_refs"])

        breaking = run_cli(
            "connector",
            "upgrade",
            "plan",
            "--contract-id",
            "ccon_project_alpha_github",
            "--target-provider-pack-id",
            "local_source_control_breaking_v2.v2",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(breaking.returncode, 7, breaking.stdout + breaking.stderr)
        blocked = json.loads(breaking.stdout)
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("CS_CONNECTOR_PROVIDER_PACK_INCOMPATIBLE", {error["code"] for error in blocked["errors"]})
        breaking_plan = blocked["connector_upgrade_plan"]
        self.assertEqual(breaking_plan["compatibility"]["status"], "incompatible")
        self.assertTrue(breaking_plan["compatibility"]["incompatible_items"])
        self.assertTrue(breaking_plan["pinned_versions_remain_active"])
        self.assertEqual(
            breaking_plan["migration_plan"]["rollback"]["provider_pack_id"],
            "local_source_control_readonly.v1",
        )
        self.assertEqual(blocked["provider_internal_findings"], [])

        compatible_path = self.state_dir / "connector" / "upgrade_plans" / f"{compatible_plan['upgrade_plan_id']}.json"
        breaking_path = self.state_dir / "connector" / "upgrade_plans" / f"{breaking_plan['upgrade_plan_id']}.json"
        self.assertTrue(compatible_path.exists())
        self.assertTrue(breaking_path.exists())

    def test_connector_product_surface_audit_keeps_one_cornerstone_product_cs_ch_039(self) -> None:
        walkthrough = run_json("product", "walkthrough")
        self.assertEqual(walkthrough["walkthrough"]["product_name"], "CornerStone")
        self.assertTrue(walkthrough["walkthrough"]["one_service"])
        self.assertFalse(walkthrough["walkthrough"]["daily_user_requires_subsystem_knowledge"])

        audit = run_json(
            "connector",
            "product-surface",
            "audit",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(audit["status"], "success")
        surface = audit["connector_product_surface_audit"]
        self.assertTrue(surface["one_product_experience"])
        self.assertEqual(surface["connected_source_surface"]["label"], "Connected Sources")
        self.assertFalse(surface["connected_source_surface"]["requires_connectorhub_sub_product"])
        self.assertEqual(surface["normal_user_forbidden_term_hits"], [])
        self.assertTrue(surface["native_cli"]["commands_begin_with_cornerstone"])
        self.assertTrue(all(value == 0 for value in surface["negative_counters"].values()))
        self.assertTrue(audit["audit_refs"])

    def test_connector_human_gate_package_prepares_github_readonly_rehearsal_h01(self) -> None:
        package_payload = run_json(
            "connector",
            "human-gate",
            "package",
            "--scenario",
            "CS-CH-H01",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(package_payload["status"], "success")
        self.assertEqual(package_payload["final_verdict"], "HUMAN_REQUIRED")
        package = package_payload["connector_human_gate_package"]
        self.assertEqual(package["schema_version"], "cs.connector_human_gate_package.v1")
        self.assertEqual(package["scenario_id"], "CS-CH-H01")
        self.assertEqual(package["status"], "human_review_required")
        self.assertEqual(package["approval_status"], "pending")
        self.assertEqual(package["review_order"], 3)
        self.assertEqual(package["gate_category"], "live_readonly_provider")
        self.assertEqual(package["depends_on_human_gates"], ["CS-CH-H04", "CS-CH-H07"])
        self.assertEqual(
            package["source_requirement_ids"],
            CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_IDS["CS-CH-H01"],
        )
        self.assertEqual(package["source_requirement_count"], 2)
        self.assertEqual(package["source_requirement_status"], "human_external_pending")
        self.assertEqual(
            package["source_requirement_claim_boundary"],
            CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY,
        )
        self.assertEqual(package["execution_queue_item"]["scenario_id"], "CS-CH-H01")
        self.assertIn("zero-write proof", package["execution_queue_item"]["required_proof_package"])
        self.assertIn("write-capable scope", package["stop_or_reject_when"])
        self.assertEqual(
            package["template_path"],
            "docs/verification-reports/CONNECTOR_HUB_CS_CH_H01_HUMAN_REVIEW_TEMPLATE_2026-06-24.md",
        )
        self.assertTrue(package["template_present"])
        self.assertTrue(package["template_structure_ready"])
        self.assertEqual(package["template_structure"]["missing_required_tokens"], [])
        self.assertTrue(package["template_structure"]["has_no_pass_boundary"])
        self.assertTrue(package["template_structure"]["has_acceptance_evidence_packet"])
        self.assertTrue(package["template_structure"]["has_senior_review_perspectives"])
        self.assertEqual(package["template_structure"]["missing_senior_review_perspectives"], [])
        self.assertTrue(package["template_structure"]["has_reject_conditions"])
        self.assertFalse(package["proof_boundary"]["product_claim_allowed"])
        self.assertFalse(package["proof_boundary"]["pass_claim_allowed_without_human_record"])
        self.assertEqual(package["proof_boundary"]["live_provider_read_verified"], "HUMAN_REQUIRED")
        self.assertEqual(package["proof_boundary"]["live_provider_write_verified"], "OUT_OF_SCOPE_READ_ONLY")
        required = package["required_human_record"]
        self.assertEqual(set(required["decision_values"]), {"ACCEPT", "REJECT"})
        for field in [
            "github_app_installation_id_redacted",
            "selected_repositories",
            "permission_snapshot",
            "call_ledger",
            "delivery_refs",
            "audit_refs",
            "zero_write_proof",
        ]:
            self.assertIn(field, required["required_fields"])
        proposed_record_template = package["proposed_record_template"]
        self.assertEqual(
            proposed_record_template["schema_version"],
            "cs.connector_human_gate_record_template.v1",
        )
        self.assertEqual(proposed_record_template["scenario_id"], "CS-CH-H01")
        self.assertEqual(proposed_record_template["status"], "blank_template_requires_human_evidence")
        self.assertFalse(proposed_record_template["product_claim_allowed_by_template"])
        self.assertFalse(proposed_record_template["pass_claim_allowed_by_template"])
        self.assertEqual(proposed_record_template["dependency_human_gates"], ["CS-CH-H04", "CS-CH-H07"])
        self.assertEqual(
            set(proposed_record_template["record_template"]["dependency_human_gate_refs"]),
            {"CS-CH-H04", "CS-CH-H07"},
        )
        self.assertEqual(
            set(proposed_record_template["required_senior_review_perspectives"]),
            set(HUMAN_GATE_PERSPECTIVE_FINDINGS),
        )
        self.assertEqual(
            set(proposed_record_template["record_template"]["senior_review_perspective_findings"]),
            set(HUMAN_GATE_PERSPECTIVE_FINDINGS),
        )
        self.assertEqual(proposed_record_template["record_template"]["decision"], "")
        self.assertEqual(proposed_record_template["record_template"]["selected_repositories"], [])
        self.assertEqual(
            len(proposed_record_template["required_evidence_packet_manifest"]),
            len(proposed_record_template["required_evidence"]),
        )
        self.assertEqual(
            [item["required_evidence_index"] for item in proposed_record_template["required_evidence_packet_manifest"]],
            list(range(1, len(proposed_record_template["required_evidence"]) + 1)),
        )
        self.assertEqual(
            [item["required_evidence"] for item in proposed_record_template["required_evidence_packet_manifest"]],
            proposed_record_template["required_evidence"],
        )
        self.assertEqual(
            proposed_record_template["allowed_redaction_statuses"],
            CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES,
        )
        self.assert_human_gate_redaction_guidance(
            proposed_record_template["redaction_guidance"],
            "CS-CH-H01",
            required_fields=required["required_fields"],
            required_evidence=proposed_record_template["required_evidence"],
            dependencies=["CS-CH-H04", "CS-CH-H07"],
        )
        self.assertEqual(
            len(proposed_record_template["record_template"]["evidence_packet_manifest"]),
            len(proposed_record_template["required_evidence"]),
        )
        self.assertEqual(
            [item["required_evidence"] for item in proposed_record_template["record_template"]["evidence_packet_manifest"]],
            proposed_record_template["required_evidence"],
        )
        self.assertTrue(
            all(
                item["evidence_ref"] == "" and item["redaction_status"] == ""
                for item in proposed_record_template["record_template"]["evidence_packet_manifest"]
            )
        )
        self.assertEqual(
            proposed_record_template["format_rules"]["review_timestamp"],
            "ISO-8601 timestamp with timezone, for example 2026-06-24T12:00:00Z",
        )
        self.assertIn("validate-record", proposed_record_template["validation_command"])
        self.assertEqual(
            proposed_record_template["validation_output_command"],
            (
                "cornerstone connector human-gate validate-record --scenario CS-CH-H01 "
                "--record-file <filled-json> --json --output <redacted-validation-envelope.json>"
            ),
        )
        self.assertEqual(
            proposed_record_template["record_template_output_command"],
            (
                "cornerstone connector human-gate package --scenario CS-CH-H01 "
                "--json --record-template-output <reviewer-record-template.json>"
            ),
        )
        self.assertEqual(
            package["record_template_output_command"],
            proposed_record_template["record_template_output_command"],
        )
        self.assertEqual(
            package["record_validation_command"],
            proposed_record_template["validation_command"],
        )
        self.assertEqual(
            package["record_validation_output_command"],
            proposed_record_template["validation_output_command"],
        )
        self.assertEqual(
            package["validation_output_command"],
            proposed_record_template["validation_output_command"],
        )
        self.assert_human_gate_reviewer_checklist(
            proposed_record_template["reviewer_checklist"],
            "CS-CH-H01",
            required_fields=required["required_fields"],
            required_evidence=proposed_record_template["required_evidence"],
            dependencies=["CS-CH-H04", "CS-CH-H07"],
            record_template_output_command=proposed_record_template["record_template_output_command"],
            validation_output_command=proposed_record_template["validation_output_command"],
        )
        self.assert_human_gate_delivery_unit_plan(
            package["scenario_delivery_unit_plan"],
            "CS-CH-H01",
            dependencies=["CS-CH-H04", "CS-CH-H07"],
            required_fields=required["required_fields"],
            required_evidence=proposed_record_template["required_evidence"],
            package_command="cornerstone connector human-gate package --scenario CS-CH-H01 --json",
            record_template_output_command=proposed_record_template["record_template_output_command"],
            validation_command=proposed_record_template["validation_command"],
            validation_output_command=proposed_record_template["validation_output_command"],
        )
        self.assert_human_gate_delivery_unit_plan_summary(package["scenario_delivery_unit_plan_summary"])
        non_mutation = package["non_mutation_evidence"]
        self.assertFalse(non_mutation["approval_collected_by_ai"])
        self.assertEqual(non_mutation["live_provider_calls_executed_by_package"], 0)
        self.assertEqual(non_mutation["provider_mutations_executed_by_package"], 0)
        self.assertFalse(non_mutation["secret_material_read_by_package"])
        self.assertEqual(non_mutation["github_write_calls_by_package"], 0)
        self.assertTrue(package_payload["audit_refs"])
        self.assertTrue(package_payload["evidence_refs"])
        self.assertIn(
            f"connector_human_gate_package:{package['package_id']}",
            package_payload["evidence_refs"],
        )
        self.assert_human_gate_completion_boundary(package)
        self.assertEqual(
            package_payload["summary"],
            {
                "scenario_id": "CS-CH-H01",
                "package_id": package["package_id"],
                "status": "human_review_required",
                "approval_status": "pending",
                "final_verdict": "HUMAN_REQUIRED",
                "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
                "review_order": 3,
                "depends_on_human_gate_count": 2,
                "source_requirement_ids": CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_IDS["CS-CH-H01"],
                "source_requirement_count": 2,
                "source_requirement_status": "human_external_pending",
                "source_requirement_claim_boundary": (
                    CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY
                ),
                "template_structure_ready": True,
                "senior_review_perspective_count": 6,
                "release_impact": package["release_impact"],
                "stop_or_reject_when": package["stop_or_reject_when"],
                "required_human_field_count": len(required["required_fields"]),
                "required_evidence_count": len(required["required_evidence"]),
                "remaining_human_evidence_claim_boundary": (
                    CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY
                ),
                "scenario_delivery_unit_plan_ready": True,
                "scenario_delivery_unit_plan_lifecycle_step_count": 7,
                "scenario_delivery_unit_plan_senior_review_perspective_count": 6,
                "scenario_delivery_unit_plan_product_claim_allowed": False,
                "scenario_delivery_unit_plan_pass_claim_allowed": False,
                "scenario_delivery_unit_plan_approval_collected": False,
                "scenario_delivery_unit_plan_dependency_unlock_allowed": False,
                "local_baseline_review_input_report_count": 0,
                "local_baseline_preflight_command_plan_count": 0,
                "local_baseline_preflight_bundle_report_count": None,
                "local_baseline_preflight_bundle_ready_report_count": None,
                "local_baseline_preflight_bundle_acceptance_sufficient": None,
                "local_baseline_acceptance_sufficient": None,
                "goal_completion_claim_blocked": True,
                "full_goal_completion_allowed": False,
                "completion_claim_boundary": CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
                "product_claim_allowed": False,
                "pass_claim_allowed_without_human_record": False,
                "live_provider_calls_executed_by_package": 0,
                "provider_mutations_executed_by_package": 0,
                "external_mutations_executed_by_package": 0,
                "record_template_output_command": proposed_record_template[
                    "record_template_output_command"
                ],
                "record_validation_command": proposed_record_template["validation_command"],
                "record_validation_output_command": proposed_record_template[
                    "validation_output_command"
                ],
                "validation_output_command": proposed_record_template[
                    "validation_output_command"
                ],
                "record_template_output_written": False,
                "product_feature_claims": "CONNECTOR_HUB_HUMAN_GATES_PREPARED_HUMAN_EVIDENCE_REQUIRED",
            },
        )
        package_path = (
            self.state_dir
            / "connector"
            / "human_gate_packages"
            / f"{package['package_id']}.json"
        )
        self.assertTrue(package_path.exists())
        state_text = state_file_texts(self.state_dir)
        self.assertIn("connector.human_gate_package.created", state_text)

        output_path = self.state_dir / "reports" / "connectorhub-human-gate-package-h01.json"
        output_payload = run_json(
            "connector",
            "human-gate",
            "package",
            "--scenario",
            "CS-CH-H01",
            "--state-dir",
            self.state_rel,
            "--output",
            str(output_path),
        )
        self.assertEqual(output_payload["status"], "success")
        self.assertEqual(output_payload["output_path"], str(output_path))
        self.assertTrue(output_path.exists())
        written_payload = json.loads(output_path.read_text())
        self.assertEqual(written_payload["schema_version"], "cs.cli.v0")
        self.assertEqual(written_payload["command"], "cornerstone connector human-gate package")
        self.assertEqual(written_payload["output_path"], str(output_path))
        self.assertEqual(written_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(written_payload["summary"], output_payload["summary"])
        written_package = written_payload["connector_human_gate_package"]
        self.assertEqual(written_package["scenario_id"], "CS-CH-H01")
        self.assertEqual(written_package["status"], "human_review_required")
        self.assertTrue(written_package["template_structure_ready"])
        self.assertFalse(written_package["proof_boundary"]["product_claim_allowed"])
        self.assertFalse(written_package["proof_boundary"]["pass_claim_allowed_without_human_record"])
        self.assertEqual(
            written_package["non_mutation_evidence"]["live_provider_calls_executed_by_package"],
            0,
        )

        record_template_output_path = self.record_dir / "blank-human-record-h01.json"
        template_output_payload = run_json(
            "connector",
            "human-gate",
            "package",
            "--scenario",
            "CS-CH-H01",
            "--state-dir",
            self.state_rel,
            "--record-template-output",
            str(record_template_output_path),
        )
        self.assertEqual(template_output_payload["status"], "success")
        self.assertTrue(template_output_payload["summary"]["record_template_output_written"])
        self.assertNotIn("record_template_output_path", json.dumps(template_output_payload, sort_keys=True))
        self.assertTrue(record_template_output_path.exists())
        written_record_template = json.loads(record_template_output_path.read_text())
        self.assertEqual(
            written_record_template,
            template_output_payload["connector_human_gate_package"]["proposed_record_template"]["record_template"],
        )
        self.assertEqual(written_record_template["scenario_id"], "CS-CH-H01")
        self.assertEqual(written_record_template["decision"], "")

    def test_connector_human_gate_packages_cover_all_human_required_rows(self) -> None:
        human_rows = [
            "CS-CH-H01",
            "CS-CH-H02",
            "CS-CH-H03",
            "CS-CH-H04",
            "CS-CH-H05",
            "CS-CH-H06",
            "CS-CH-H07",
        ]
        expected_review_order = {
            "CS-CH-H04": 1,
            "CS-CH-H07": 2,
            "CS-CH-H01": 3,
            "CS-CH-H02": 4,
            "CS-CH-H03": 5,
            "CS-CH-H05": 6,
            "CS-CH-H06": 7,
        }
        for scenario_id in human_rows:
            with self.subTest(scenario_id=scenario_id):
                package_payload = run_json(
                    "connector",
                    "human-gate",
                    "package",
                    "--scenario",
                    scenario_id,
                    "--state-dir",
                    self.state_rel,
                )
                self.assertEqual(package_payload["status"], "success")
                self.assertEqual(package_payload["final_verdict"], "HUMAN_REQUIRED")
                package = package_payload["connector_human_gate_package"]
                self.assertEqual(package["schema_version"], "cs.connector_human_gate_package.v1")
                self.assertEqual(package["scenario_id"], scenario_id)
                self.assertEqual(package["status"], "human_review_required")
                self.assertEqual(package["approval_status"], "pending")
                self.assertEqual(package["review_order"], expected_review_order[scenario_id])
                self.assertEqual(package["execution_queue_item"]["scenario_id"], scenario_id)
                self.assertEqual(package["execution_queue_item"]["order"], package["review_order"])
                self.assertEqual(
                    package["execution_queue_item"]["depends_on"],
                    package["depends_on_human_gates"],
                )
                self.assertEqual(
                    package["source_requirement_ids"],
                    CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_IDS[scenario_id],
                )
                self.assertEqual(
                    package["source_requirement_count"],
                    len(CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_IDS[scenario_id]),
                )
                self.assertEqual(package["source_requirement_status"], "human_external_pending")
                self.assertEqual(
                    package["source_requirement_claim_boundary"],
                    CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY,
                )
                self.assertTrue(package["gate_category"])
                self.assertTrue(package["stop_or_reject_when"])
                self.assertTrue(package["template_path"].endswith("_HUMAN_REVIEW_TEMPLATE_2026-06-24.md"))
                self.assertTrue(package["template_present"])
                self.assertTrue(package["template_structure_ready"])
                template_structure = package["template_structure"]
                self.assertTrue(template_structure["structure_ready"])
                self.assertEqual(template_structure["missing_required_tokens"], [])
                self.assertTrue(template_structure["has_scenario_id"])
                self.assertTrue(template_structure["has_no_pass_boundary"])
                self.assertTrue(template_structure["has_scenario_first_runbook_or_study"])
                self.assertTrue(template_structure["has_acceptance_evidence_packet"])
                self.assertTrue(template_structure["has_senior_review_perspectives"])
                self.assertEqual(template_structure["missing_senior_review_perspectives"], [])
                self.assertTrue(template_structure["has_pending_human_result_rows"])
                self.assertTrue(template_structure["has_reject_conditions"])
                self.assertFalse(package["proof_boundary"]["product_claim_allowed"])
                self.assertFalse(package["proof_boundary"]["pass_claim_allowed_without_human_record"])
                self.assertGreaterEqual(len(package["senior_review_perspectives"]), 6)
                self.assertTrue(package["rehearsal_checklist"])
                required = package["required_human_record"]
                self.assertEqual(set(required["decision_values"]), {"ACCEPT", "REJECT"})
                self.assertIn("reviewer", required["required_fields"])
                self.assertIn("review_timestamp", required["required_fields"])
                self.assertTrue(any(field.endswith("_or_exceptions") for field in required["required_fields"]))
                self.assertTrue(required["required_evidence"])
                remaining_summary = package["remaining_human_evidence_summary"]
                self.assertEqual(
                    remaining_summary["schema_version"],
                    CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_SUMMARY_SCHEMA,
                )
                self.assertEqual(remaining_summary["scenario_id"], scenario_id)
                self.assertEqual(remaining_summary["required_human_fields"], required["required_fields"])
                self.assertEqual(remaining_summary["required_evidence"], required["required_evidence"])
                self.assertEqual(
                    remaining_summary["required_human_field_count"],
                    len(required["required_fields"]),
                )
                self.assertEqual(
                    remaining_summary["required_evidence_count"],
                    len(required["required_evidence"]),
                )
                self.assertEqual(remaining_summary["release_impact"], package["release_impact"])
                self.assertEqual(remaining_summary["stop_or_reject_when"], package["stop_or_reject_when"])
                self.assertEqual(
                    remaining_summary["record_template_output_command"],
                    (
                        f"cornerstone connector human-gate package --scenario {scenario_id} "
                        "--json --record-template-output <reviewer-record-template.json>"
                    ),
                )
                self.assertEqual(
                    remaining_summary["validate_record_output_command"],
                    (
                        f"cornerstone connector human-gate validate-record --scenario {scenario_id} "
                        "--record-file <json> --json --output <redacted-validation-envelope.json>"
                    ),
                )
                self.assertEqual(
                    remaining_summary["claim_boundary"],
                    CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY,
                )
                if scenario_id in {"CS-CH-H01", "CS-CH-H02", "CS-CH-H03", "CS-CH-H04", "CS-CH-H05", "CS-CH-H06", "CS-CH-H07"}:
                    self.assert_h04_remaining_evidence_summary_workflow(remaining_summary)
                else:
                    self.assertNotIn("evidence_packet_workflow", remaining_summary)
                    self.assertNotIn("evidence_packet_workflow_commands", remaining_summary)
                proposed_record_template = package["proposed_record_template"]
                self.assertEqual(
                    proposed_record_template["schema_version"],
                    "cs.connector_human_gate_record_template.v1",
                )
                self.assertEqual(proposed_record_template["scenario_id"], scenario_id)
                self.assertEqual(proposed_record_template["record_template"]["scenario_id"], scenario_id)
                self.assertEqual(proposed_record_template["record_template"]["decision"], "")
                self.assertEqual(
                    set(proposed_record_template["required_senior_review_perspectives"]),
                    set(HUMAN_GATE_PERSPECTIVE_FINDINGS),
                )
                self.assertEqual(
                    set(proposed_record_template["record_template"]["senior_review_perspective_findings"]),
                    set(HUMAN_GATE_PERSPECTIVE_FINDINGS),
                )
                self.assertEqual(
                    len(proposed_record_template["required_evidence_packet_manifest"]),
                    len(proposed_record_template["required_evidence"]),
                )
                self.assertEqual(
                    [
                        item["required_evidence_index"]
                        for item in proposed_record_template["required_evidence_packet_manifest"]
                    ],
                    list(range(1, len(proposed_record_template["required_evidence"]) + 1)),
                )
                self.assertEqual(
                    [
                        item["required_evidence"]
                        for item in proposed_record_template["required_evidence_packet_manifest"]
                    ],
                    proposed_record_template["required_evidence"],
                )
                self.assertEqual(
                    proposed_record_template["allowed_redaction_statuses"],
                    CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES,
                )
                self.assertEqual(
                    len(proposed_record_template["record_template"]["evidence_packet_manifest"]),
                    len(proposed_record_template["required_evidence"]),
                )
                self.assertEqual(
                    [
                        item["required_evidence"]
                        for item in proposed_record_template["record_template"]["evidence_packet_manifest"]
                    ],
                    proposed_record_template["required_evidence"],
                )
                self.assertTrue(
                    all(
                        item["evidence_ref"] == "" and item["redaction_status"] == ""
                        for item in proposed_record_template["record_template"]["evidence_packet_manifest"]
                    )
                )
                self.assertFalse(proposed_record_template["product_claim_allowed_by_template"])
                self.assertFalse(proposed_record_template["pass_claim_allowed_by_template"])
                self.assertEqual(
                    proposed_record_template["dependency_human_gates"],
                    package["depends_on_human_gates"],
                )
                if package["depends_on_human_gates"]:
                    self.assertEqual(
                        set(proposed_record_template["record_template"]["dependency_human_gate_refs"]),
                        set(package["depends_on_human_gates"]),
                    )
                self.assertEqual(
                    proposed_record_template["format_rules"]["review_timestamp"],
                    "ISO-8601 timestamp with timezone, for example 2026-06-24T12:00:00Z",
                )
                self.assertIn("validate-record", proposed_record_template["validation_command"])
                self.assertEqual(
                    proposed_record_template["validation_output_command"],
                    (
                        f"cornerstone connector human-gate validate-record --scenario {scenario_id} "
                        "--record-file <filled-json> --json --output <redacted-validation-envelope.json>"
                    ),
                )
                self.assertEqual(
                    proposed_record_template["record_template_output_command"],
                    (
                        f"cornerstone connector human-gate package --scenario {scenario_id} "
                        "--json --record-template-output <reviewer-record-template.json>"
                    ),
                )
                if scenario_id == "CS-CH-H04":
                    self.assert_h04_field_ref_contract(proposed_record_template["field_ref_contract"])
                else:
                    self.assertNotIn("field_ref_contract", proposed_record_template)
                self.assert_human_gate_redaction_guidance(
                    proposed_record_template["redaction_guidance"],
                    scenario_id,
                    required_fields=required["required_fields"],
                    required_evidence=proposed_record_template["required_evidence"],
                    dependencies=package["depends_on_human_gates"],
                )
                self.assert_human_gate_reviewer_checklist(
                    proposed_record_template["reviewer_checklist"],
                    scenario_id,
                    required_fields=required["required_fields"],
                    required_evidence=proposed_record_template["required_evidence"],
                    dependencies=package["depends_on_human_gates"],
                    record_template_output_command=proposed_record_template["record_template_output_command"],
                    validation_output_command=proposed_record_template["validation_output_command"],
                )
                self.assertEqual(
                    package["record_template_output_command"],
                    proposed_record_template["record_template_output_command"],
                )
                self.assertEqual(
                    package["record_validation_command"],
                    proposed_record_template["validation_command"],
                )
                self.assertEqual(
                    package["record_validation_output_command"],
                    proposed_record_template["validation_output_command"],
                )
                self.assertEqual(
                    package["validation_output_command"],
                    proposed_record_template["validation_output_command"],
                )
                self.assert_human_gate_delivery_unit_plan(
                    package["scenario_delivery_unit_plan"],
                    scenario_id,
                    dependencies=package["depends_on_human_gates"],
                    required_fields=required["required_fields"],
                    required_evidence=proposed_record_template["required_evidence"],
                    package_command=f"cornerstone connector human-gate package --scenario {scenario_id} --json",
                    record_template_output_command=proposed_record_template["record_template_output_command"],
                    validation_command=proposed_record_template["validation_command"],
                    validation_output_command=proposed_record_template["validation_output_command"],
                )
                self.assert_human_gate_delivery_unit_plan_summary(
                    package["scenario_delivery_unit_plan_summary"]
                )
                non_mutation = package["non_mutation_evidence"]
                self.assertFalse(non_mutation["approval_collected_by_ai"])
                self.assertEqual(non_mutation["live_provider_calls_executed_by_package"], 0)
                self.assertEqual(non_mutation["provider_mutations_executed_by_package"], 0)
                self.assertEqual(non_mutation["external_mutations_executed_by_package"], 0)
                self.assertFalse(non_mutation["secret_material_read_by_package"])
                self.assertIn("Blocks", package["release_impact"])
                self.assertTrue(package_payload["audit_refs"])
                self.assertTrue(package_payload["evidence_refs"])
                summary = package_payload["summary"]
                self.assertEqual(summary["scenario_id"], scenario_id)
                self.assertEqual(summary["status"], "human_review_required")
                self.assertEqual(summary["approval_status"], "pending")
                self.assertEqual(summary["final_verdict"], "HUMAN_REQUIRED")
                self.assertEqual(summary["weakest_applicable_scenario_result"], "HUMAN_REQUIRED")
                self.assertEqual(summary["review_order"], expected_review_order[scenario_id])
                self.assertEqual(summary["depends_on_human_gate_count"], len(package["depends_on_human_gates"]))
                self.assertEqual(summary["source_requirement_ids"], package["source_requirement_ids"])
                self.assertEqual(summary["source_requirement_count"], len(package["source_requirement_ids"]))
                self.assertEqual(summary["source_requirement_status"], "human_external_pending")
                self.assertEqual(
                    summary["source_requirement_claim_boundary"],
                    CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY,
                )
                self.assertTrue(summary["template_structure_ready"])
                self.assertGreaterEqual(summary["senior_review_perspective_count"], 6)
                self.assertEqual(summary["release_impact"], package["release_impact"])
                self.assertEqual(summary["stop_or_reject_when"], package["stop_or_reject_when"])
                self.assertEqual(summary["required_human_field_count"], len(required["required_fields"]))
                self.assertEqual(summary["required_evidence_count"], len(required["required_evidence"]))
                self.assertEqual(
                    summary["remaining_human_evidence_claim_boundary"],
                    CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY,
                )
                self.assertTrue(summary["scenario_delivery_unit_plan_ready"])
                self.assertEqual(summary["scenario_delivery_unit_plan_lifecycle_step_count"], 7)
                self.assertGreaterEqual(summary["scenario_delivery_unit_plan_senior_review_perspective_count"], 6)
                self.assertFalse(summary["scenario_delivery_unit_plan_product_claim_allowed"])
                self.assertFalse(summary["scenario_delivery_unit_plan_pass_claim_allowed"])
                self.assertFalse(summary["scenario_delivery_unit_plan_approval_collected"])
                self.assertFalse(summary["scenario_delivery_unit_plan_dependency_unlock_allowed"])
                self.assert_human_gate_completion_boundary(package)
                self.assert_human_gate_completion_boundary(summary)
                self.assertFalse(summary["product_claim_allowed"])
                self.assertFalse(summary["pass_claim_allowed_without_human_record"])
                self.assertEqual(summary["live_provider_calls_executed_by_package"], 0)
                self.assertEqual(summary["provider_mutations_executed_by_package"], 0)
                self.assertEqual(summary["external_mutations_executed_by_package"], 0)
                self.assertEqual(
                    summary["record_template_output_command"],
                    package["record_template_output_command"],
                )
                self.assertEqual(
                    summary["record_validation_command"],
                    package["record_validation_command"],
                )
                self.assertEqual(
                    summary["record_validation_output_command"],
                    package["record_validation_output_command"],
                )
                self.assertEqual(
                    summary["validation_output_command"],
                    package["validation_output_command"],
                )
                self.assertFalse(summary["record_template_output_written"])
                self.assertEqual(
                    summary["product_feature_claims"],
                    "CONNECTOR_HUB_HUMAN_GATES_PREPARED_HUMAN_EVIDENCE_REQUIRED",
                )
                package_path = (
                    self.state_dir
                    / "connector"
                    / "human_gate_packages"
                    / f"{package['package_id']}.json"
                )
                self.assertTrue(package_path.exists())
                if scenario_id == "CS-CH-H05":
                    self.assertTrue(package["proof_boundary"]["github_actions_excluded"])
                    self.assertEqual(non_mutation["github_actions_executed_by_package"], 0)
                    self.assertIn("GitHub remains excluded", package["release_impact"])
                if scenario_id == "CS-CH-H04":
                    baseline = package["local_baseline_review_inputs"]
                    self.assertEqual(
                        baseline["schema_version"],
                        "cs.connector_human_gate_local_baseline_review_inputs.v1",
                    )
                    self.assertEqual(baseline["status"], "review_input_only")
                    self.assertFalse(baseline["acceptance_sufficient"])
                    self.assertFalse(baseline["product_claim_allowed"])
                    self.assertFalse(baseline["pass_claim_allowed"])
                    self.assertTrue(baseline["all_reports_present"])
                    self.assertTrue(baseline["all_reports_json_valid"])
                    self.assertEqual(baseline["missing_reports"], [])
                    self.assertEqual(baseline["invalid_json_reports"], [])
                    self.assertIn("not prove production-like", baseline["boundary"])
                    self.assertIn(
                        "reports/security/vs2-local-security-proof.json",
                        {report["path"] for report in baseline["reports"]},
                    )
                    self.assertIn(
                        "reports/scenario/connector-contract-adapter-cs-ch-036-2026-06-23.json",
                        {report["path"] for report in baseline["reports"]},
                    )
                    self.assertEqual(
                        baseline["required_human_delta"],
                        [
                            "Production-like topology identifier and trusted RequestContext transcript.",
                            "Scenario-specific PostgreSQL/RLS and OPA transcripts from the reviewed environment.",
                            "Network default-deny and governed-egress transcripts from the reviewed topology.",
                            "Backup/restore evidence and audit-integrity report from the reviewed environment.",
                            "Dated ACCEPT or REJECT decision with redacted evidence packet manifest.",
                        ],
                    )
                    self.assertEqual(package["required_human_delta"], baseline["required_human_delta"])
                    self.assertEqual(package["required_human_delta_count"], 5)
                    self.assertEqual(summary["required_human_delta"], baseline["required_human_delta"])
                    self.assertEqual(summary["required_human_delta_count"], 5)
                    self.assertEqual(
                        baseline["recommended_preflight_commands"],
                        [
                            "cornerstone security vs2-local-proof --json",
                            (
                                "cornerstone scenario verify vs2-policy-tenancy-egress "
                                "--reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json"
                            ),
                            "cornerstone scenario verify connector-contract-adapter --scenario CS-CH-036 --json",
                        ],
                    )
                    self.assertEqual(
                        baseline["recommended_preflight_command_plan_schema_version"],
                        CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA,
                    )
                    self.assertEqual(baseline["recommended_preflight_command_plan_count"], 3)
                    self.assertEqual(
                        baseline["recommended_preflight_command_plan"],
                        EXPECTED_H04_PREFLIGHT_COMMAND_PLAN,
                    )
                    expected_baseline_facts = {
                        "reports/security/vs2-local-security-proof.json": {
                            "schema_version": "cs.vs2_local_security_proof.v0",
                            "status": "success",
                            "scenario_count": 93,
                            "summary": {
                                "blocking": 0,
                                "fail": 0,
                                "human_required": 7,
                                "not_run": 0,
                                "not_verified": 0,
                                "pass": 86,
                                "product_feature_claims": "LOCAL_VS2_AI_VERIFIED_HUMAN_GATES_PENDING",
                                "scenario_count": 93,
                            },
                        },
                        "reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json": {
                            "schema_version": "cs.cli.v0",
                            "status": "success",
                            "command": "cornerstone scenario verify vs2-policy-tenancy-egress",
                            "scenario_count": 93,
                            "summary": {
                                "blocking": 0,
                                "fail": 0,
                                "human_required": 7,
                                "not_run": 0,
                                "not_verified": 0,
                                "pass": 86,
                                "product_feature_claims": "LOCAL_VS2_AI_VERIFIED_HUMAN_GATES_PENDING",
                                "scenario_count": 93,
                            },
                        },
                        "reports/network/vs2-egress-proof.json": {
                            "status": "passed",
                        },
                        "reports/security/vs2-local-range.json": {
                            "schema_version": "cs.vs2_local_range.v1",
                            "status": "passed",
                        },
                        "reports/scenario/connector-contract-adapter-cs-ch-036-2026-06-23.json": {
                            "schema_version": "cs.cli.v0",
                            "status": "success",
                            "command": "cornerstone scenario verify connector-contract-adapter",
                            "scenario_count": 1,
                            "summary": {
                                "blocking": 0,
                                "fail": 0,
                                "human_required": 0,
                                "not_verified": 0,
                                "pass": 1,
                                "product_feature_claims": (
                                    "LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING"
                                ),
                                "scenario_count": 1,
                            },
                        },
                    }
                    baseline_reports = {report["path"]: report for report in baseline["reports"]}
                    self.assertEqual(set(baseline_reports), set(expected_baseline_facts))
                    for path, expected_facts in expected_baseline_facts.items():
                        self.assertTrue(baseline_reports[path]["present"])
                        self.assertTrue(baseline_reports[path]["json_valid"])
                        self.assertRegex(baseline_reports[path]["sha256"], r"^[0-9a-f]{64}$")
                        self.assertTrue(baseline_reports[path]["review_input_only"])
                        self.assertFalse(baseline_reports[path]["acceptance_sufficient"])
                        self.assertFalse(baseline_reports[path]["product_claim_allowed"])
                        self.assertFalse(baseline_reports[path]["pass_claim_allowed"])
                        self.assertEqual(
                            baseline_reports[path]["claim_boundary"],
                            CONNECTOR_HUMAN_GATE_H04_BASELINE_CLAIM_BOUNDARY,
                        )
                        for key, expected in expected_facts.items():
                            self.assertEqual(baseline_reports[path][key], expected)
                    self.assert_h04_local_baseline_preflight_bundle(
                        baseline["preflight_bundle"],
                        list(expected_baseline_facts),
                        baseline_reports,
                    )
                    self.assertEqual(
                        package["local_baseline_preflight_bundle"],
                        baseline["preflight_bundle"],
                    )
                    self.assertEqual(summary["local_baseline_review_input_report_count"], 5)
                    self.assertEqual(summary["local_baseline_preflight_command_plan_count"], 3)
                    self.assertEqual(summary["local_baseline_preflight_bundle_report_count"], 5)
                    self.assertEqual(summary["local_baseline_preflight_bundle_ready_report_count"], 5)
                    self.assertFalse(summary["local_baseline_preflight_bundle_acceptance_sufficient"])
                    self.assertFalse(summary["local_baseline_acceptance_sufficient"])
                else:
                    self.assertNotIn("local_baseline_review_inputs", package)
                    self.assertNotIn("local_baseline_preflight_bundle", package)
                    self.assertEqual(summary["local_baseline_review_input_report_count"], 0)
                    self.assertEqual(summary["local_baseline_preflight_command_plan_count"], 0)
                    self.assertIsNone(summary["local_baseline_preflight_bundle_report_count"])
                    self.assertIsNone(summary["local_baseline_preflight_bundle_ready_report_count"])
                    self.assertIsNone(summary["local_baseline_preflight_bundle_acceptance_sufficient"])
                    self.assertIsNone(summary["local_baseline_acceptance_sufficient"])

    def test_connector_human_gate_field_ref_contract_exposes_h04_ref_shapes_without_values(self) -> None:
        payload = run_json(
            "connector",
            "human-gate",
            "field-ref-contract",
            "--scenario",
            "CS-CH-H04",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(payload["command"], "cornerstone connector human-gate field-ref-contract")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(payload["errors"], [])
        report = payload["connector_human_gate_field_ref_contract_report"]
        self.assertEqual(report["schema_version"], CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_REPORT_SCHEMA)
        self.assertEqual(report["scenario_id"], "CS-CH-H04")
        self.assertEqual(report["status"], "operator_preparation_only")
        self.assertEqual(report["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(report["weakest_applicable_scenario_result"], "HUMAN_REQUIRED")
        self.assert_human_gate_completion_boundary(report)
        self.assert_h04_field_ref_contract(report["field_ref_contract"])
        self.assertEqual(report["field_ref_contract_schema_version"], CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_SCHEMA)
        self.assertEqual(report["required_field_ref_item_count"], len(CONNECTOR_HUMAN_GATE_H04_FIELD_REF_ITEMS))
        self.assertEqual(
            report["required_field_ref_fields"],
            [item["field"] for item in CONNECTOR_HUMAN_GATE_H04_FIELD_REF_ITEMS],
        )
        self.assertEqual(
            report["accepted_ref_prefixes_by_field"],
            {
                item["field"]: item["accepted_ref_prefixes"]
                for item in CONNECTOR_HUMAN_GATE_H04_FIELD_REF_ITEMS
            },
        )
        self.assertEqual(
            report["accepted_container_by_field"],
            {
                item["field"]: item["accepted_container"]
                for item in CONNECTOR_HUMAN_GATE_H04_FIELD_REF_ITEMS
            },
        )
        self.assertFalse(report["raw_field_values_recorded_by_report"])
        self.assertFalse(report["raw_field_values_persisted_by_validator"])
        self.assertEqual(report["invalid_value_report_shape"], "field_names_only")
        non_mutation = report["non_mutation_evidence"]
        self.assertFalse(non_mutation["approval_collected_by_field_ref_contract"])
        self.assertEqual(non_mutation["human_decisions_recorded_by_field_ref_contract"], 0)
        self.assertEqual(non_mutation["commands_executed_by_field_ref_contract"], 0)
        self.assertEqual(non_mutation["live_provider_calls_executed_by_field_ref_contract"], 0)
        self.assertEqual(non_mutation["provider_mutations_executed_by_field_ref_contract"], 0)
        self.assertEqual(non_mutation["external_mutations_executed_by_field_ref_contract"], 0)
        self.assertFalse(non_mutation["secret_material_read_by_field_ref_contract"])
        self.assertFalse(non_mutation["raw_field_values_recorded_by_field_ref_contract"])
        self.assertFalse(non_mutation["raw_field_values_persisted_by_field_ref_contract"])
        self.assertTrue(all(value in (0, False) for value in report["negative_evidence"].values()))
        self.assertTrue(payload["audit_refs"])
        self.assertIn(
            f"connector_human_gate_field_ref_contract:{report['field_ref_contract_report_id']}",
            payload["evidence_refs"],
        )
        self.assertEqual(
            payload["ids"]["connector_human_gate_field_ref_contract_report_id"],
            report["field_ref_contract_report_id"],
        )
        self.assertEqual(
            payload["summary"],
            {
                "scenario_id": "CS-CH-H04",
                "status": "operator_preparation_only",
                "final_verdict": "HUMAN_REQUIRED",
                "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
                "field_ref_contract_report_id": report["field_ref_contract_report_id"],
                "operator_rule": report["operator_rule"],
                "field_ref_contract_schema_version": CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_SCHEMA,
                "required_field_ref_item_count": len(CONNECTOR_HUMAN_GATE_H04_FIELD_REF_ITEMS),
                "required_field_ref_fields": [
                    item["field"] for item in CONNECTOR_HUMAN_GATE_H04_FIELD_REF_ITEMS
                ],
                "raw_field_values_recorded_by_report": False,
                "raw_field_values_persisted_by_validator": False,
                "invalid_value_report_shape": "field_names_only",
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
                "commands_executed_by_field_ref_contract": 0,
                "live_provider_calls_executed_by_field_ref_contract": 0,
                "provider_mutations_executed_by_field_ref_contract": 0,
                "external_mutations_executed_by_field_ref_contract": 0,
                "human_acceptance_collected_by_field_ref_contract": False,
                "goal_completion_claim_blocked": True,
                "full_goal_completion_allowed": False,
                "completion_claim_boundary": CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
                "product_feature_claims": (
                    "CONNECTOR_HUB_H04_FIELD_REF_CONTRACT_PREPARED_HUMAN_EVIDENCE_REQUIRED"
                ),
            },
        )
        report_path = (
            self.state_dir
            / "connector"
            / "human_gate_field_ref_contract_reports"
            / f"{report['field_ref_contract_report_id']}.json"
        )
        self.assertTrue(report_path.exists())
        state_text = state_file_texts(self.state_dir)
        self.assertIn("connector.human_gate_field_ref_contract.reported", state_text)
        for raw_field_ref in [
            "topology:prod-alpha",
            "request_context:tenant-a",
            "db_policy:rls-transcript",
            "egress:default-deny",
            "backup_restore:restore-001",
            "audit_integrity:ledger-001",
            "evidence_manifest:packet-001",
        ]:
            self.assertNotIn(raw_field_ref, json.dumps(payload, sort_keys=True))
            self.assertNotIn(raw_field_ref, state_text)

        output_path = self.state_dir / "reports" / "connectorhub-human-gate-field-ref-contract-h04.json"
        output_payload = run_json(
            "connector",
            "human-gate",
            "field-ref-contract",
            "--scenario",
            "CS-CH-H04",
            "--state-dir",
            self.state_rel,
            "--output",
            str(output_path),
        )
        self.assertEqual(output_payload["status"], "success")
        self.assertEqual(output_payload["output_path"], str(output_path))
        self.assertTrue(output_path.exists())
        written_payload = json.loads(output_path.read_text())
        self.assertEqual(written_payload["command"], "cornerstone connector human-gate field-ref-contract")
        self.assertEqual(written_payload["output_path"], str(output_path))
        self.assertEqual(written_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(written_payload["summary"], output_payload["summary"])

    def test_connector_human_gate_evidence_packet_help_text_is_all_human_gate_scoped(self) -> None:
        parent_help_result = run_cli("connector", "human-gate", "--help")
        self.assertEqual(parent_help_result.returncode, 0, parent_help_result.stderr)
        parent_help = " ".join(parent_help_result.stdout.split())
        expected_parent_help = [
            "Emit a ConnectorHub human-gate evidence-packet manifest contract without recording evidence values",
            "Emit a ConnectorHub human-gate evidence-packet file checklist without reading packet contents",
            "Prepare ConnectorHub human-gate blank evidence-packet templates without accepting human evidence",
            "Validate ConnectorHub human-gate evidence-packet file metadata without accepting human evidence",
            "Draft a ConnectorHub human-gate reviewer record from evidence-packet hashes without accepting it",
            "Emit the H04 accepted evidence-reference contract without recording field values",
            "Emit the H04 local preflight bundle without executing commands or collecting approval",
        ]
        for expected in expected_parent_help:
            self.assertIn(expected, parent_help)
        stale_parent_help = [
            "Emit the H04 evidence-packet manifest contract without recording evidence values",
            "Emit the H04 evidence-packet file checklist without reading packet contents",
            "Prepare H04 blank evidence-packet templates without accepting human evidence",
            "Validate H04 evidence-packet file metadata without accepting human evidence",
            "Draft an H04 reviewer record from evidence-packet hashes without accepting it",
        ]
        for stale in stale_parent_help:
            self.assertNotIn(stale, parent_help)

        validate_help_result = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--help",
        )
        self.assertEqual(validate_help_result.returncode, 0, validate_help_result.stderr)
        validate_help = " ".join(validate_help_result.stdout.split())
        self.assertIn(
            "Packet directory containing the filled ConnectorHub human-gate evidence files",
            validate_help,
        )
        self.assertNotIn(
            "Packet directory containing the filled H04 acceptance evidence files",
            validate_help,
        )

        record_draft_help_result = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-record-draft",
            "--help",
        )
        self.assertEqual(record_draft_help_result.returncode, 0, record_draft_help_result.stderr)
        record_draft_help = " ".join(record_draft_help_result.stdout.split())
        self.assertIn(
            "Structurally complete ConnectorHub human-gate evidence packet directory",
            record_draft_help,
        )
        self.assertNotIn(
            "Structurally complete H04 acceptance evidence packet directory",
            record_draft_help,
        )

    def test_connector_human_gate_field_ref_contract_rejects_non_h04_scenario(self) -> None:
        result = run_cli(
            "connector",
            "human-gate",
            "field-ref-contract",
            "--scenario",
            "CS-CH-H01",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["command"], "cornerstone connector human-gate field-ref-contract")
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(
            payload["connector_human_gate_field_ref_contract_report"]["schema_version"],
            CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_REPORT_SCHEMA,
        )
        self.assertEqual(
            payload["connector_human_gate_field_ref_contract_report"]["scenario_id"],
            "CS-CH-H01",
        )
        self.assertEqual(
            {error["code"] for error in payload["errors"]},
            {"CS_CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_UNSUPPORTED"},
        )

    def test_connector_human_gate_evidence_packet_contract_exposes_h04_manifest_shape_without_values(self) -> None:
        payload = run_json(
            "connector",
            "human-gate",
            "evidence-packet-contract",
            "--scenario",
            "CS-CH-H04",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(payload["command"], "cornerstone connector human-gate evidence-packet-contract")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(payload["errors"], [])
        report = payload["connector_human_gate_evidence_packet_contract_report"]
        self.assertEqual(report["schema_version"], CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_REPORT_SCHEMA)
        self.assertEqual(report["scenario_id"], "CS-CH-H04")
        self.assertEqual(report["status"], "operator_preparation_only")
        self.assertEqual(report["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(report["weakest_applicable_scenario_result"], "HUMAN_REQUIRED")
        self.assert_human_gate_completion_boundary(report)
        expected_required_evidence = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS["CS-CH-H04"][
            "required_human_record"
        ]["required_evidence"]
        expected_manifest = [
            {
                "required_evidence_index": index,
                "required_evidence": evidence,
            }
            for index, evidence in enumerate(expected_required_evidence, start=1)
        ]
        contract = report["evidence_packet_contract"]
        self.assertEqual(contract["schema_version"], CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_SCHEMA)
        self.assertEqual(contract["scenario_id"], "CS-CH-H04")
        self.assertEqual(contract["status"], "operator_preparation_only")
        self.assertEqual(contract["validation_scope"], "evidence_packet_manifest_shape_only")
        self.assertEqual(contract["required_evidence_packet_manifest"], expected_manifest)
        self.assertEqual(contract["required_evidence_packet_manifest_count"], len(expected_manifest))
        self.assertEqual(contract["required_evidence_indexes"], list(range(1, len(expected_manifest) + 1)))
        self.assertEqual(contract["required_evidence_labels"], expected_required_evidence)
        self.assertTrue(contract["evidence_ref_required"])
        self.assertTrue(contract["evidence_ref_uniqueness_required"])
        self.assertTrue(contract["redaction_status_required"])
        self.assertEqual(contract["allowed_redaction_statuses"], CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES)
        self.assertFalse(contract["raw_evidence_ref_values_recorded_by_report"])
        self.assertFalse(contract["evidence_packet_manifest_values_persisted_by_validator"])
        self.assertEqual(contract["invalid_value_report_shape"], "field_names_and_required_evidence_indexes_only")
        self.assertEqual(
            report["evidence_packet_contract_schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_SCHEMA,
        )
        self.assertEqual(report["required_evidence_packet_manifest_count"], len(expected_manifest))
        self.assertEqual(report["required_evidence_indexes"], list(range(1, len(expected_manifest) + 1)))
        self.assertEqual(report["required_evidence_labels"], expected_required_evidence)
        self.assertEqual(report["allowed_redaction_statuses"], CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES)
        self.assertTrue(report["evidence_ref_required"])
        self.assertTrue(report["evidence_ref_uniqueness_required"])
        self.assertTrue(report["redaction_status_required"])
        self.assertFalse(report["raw_evidence_ref_values_recorded_by_report"])
        self.assertFalse(report["evidence_packet_manifest_values_persisted_by_validator"])
        self.assertEqual(report["invalid_value_report_shape"], "field_names_and_required_evidence_indexes_only")
        non_mutation = report["non_mutation_evidence"]
        self.assertFalse(non_mutation["approval_collected_by_evidence_packet_contract"])
        self.assertEqual(non_mutation["human_decisions_recorded_by_evidence_packet_contract"], 0)
        self.assertEqual(non_mutation["commands_executed_by_evidence_packet_contract"], 0)
        self.assertEqual(non_mutation["live_provider_calls_executed_by_evidence_packet_contract"], 0)
        self.assertEqual(non_mutation["provider_mutations_executed_by_evidence_packet_contract"], 0)
        self.assertEqual(non_mutation["external_mutations_executed_by_evidence_packet_contract"], 0)
        self.assertFalse(non_mutation["secret_material_read_by_evidence_packet_contract"])
        self.assertFalse(non_mutation["raw_evidence_ref_values_recorded_by_evidence_packet_contract"])
        self.assertFalse(
            non_mutation["evidence_packet_manifest_values_persisted_by_evidence_packet_contract"]
        )
        self.assertTrue(all(value in (0, False) for value in report["negative_evidence"].values()))
        self.assertTrue(payload["audit_refs"])
        self.assertIn(
            (
                "connector_human_gate_evidence_packet_contract:"
                f"{report['evidence_packet_contract_report_id']}"
            ),
            payload["evidence_refs"],
        )
        self.assertEqual(
            payload["ids"]["connector_human_gate_evidence_packet_contract_report_id"],
            report["evidence_packet_contract_report_id"],
        )
        self.assertEqual(
            payload["summary"],
            {
                "scenario_id": "CS-CH-H04",
                "status": "operator_preparation_only",
                "final_verdict": "HUMAN_REQUIRED",
                "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
                "evidence_packet_contract_report_id": report[
                    "evidence_packet_contract_report_id"
                ],
                "operator_rule": report["operator_rule"],
                "evidence_packet_contract_schema_version": (
                    CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_SCHEMA
                ),
                "required_evidence_packet_manifest_count": len(expected_manifest),
                "required_evidence_indexes": list(range(1, len(expected_manifest) + 1)),
                "required_evidence_labels": expected_required_evidence,
                "allowed_redaction_statuses": CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES,
                "evidence_ref_required": True,
                "evidence_ref_uniqueness_required": True,
                "redaction_status_required": True,
                "raw_evidence_ref_values_recorded_by_report": False,
                "evidence_packet_manifest_values_persisted_by_validator": False,
                "invalid_value_report_shape": "field_names_and_required_evidence_indexes_only",
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
                "commands_executed_by_evidence_packet_contract": 0,
                "live_provider_calls_executed_by_evidence_packet_contract": 0,
                "provider_mutations_executed_by_evidence_packet_contract": 0,
                "external_mutations_executed_by_evidence_packet_contract": 0,
                "human_acceptance_collected_by_evidence_packet_contract": False,
                "goal_completion_claim_blocked": True,
                "full_goal_completion_allowed": False,
                "completion_claim_boundary": CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
                "product_feature_claims": (
                    "CONNECTOR_HUB_H04_EVIDENCE_PACKET_CONTRACT_PREPARED_HUMAN_EVIDENCE_REQUIRED"
                ),
            },
        )
        report_path = (
            self.state_dir
            / "connector"
            / "human_gate_evidence_packet_contract_reports"
            / f"{report['evidence_packet_contract_report_id']}.json"
        )
        self.assertTrue(report_path.exists())
        state_text = state_file_texts(self.state_dir)
        self.assertIn("connector.human_gate_evidence_packet_contract.reported", state_text)
        for raw_evidence_ref in [
            "packet://topology-prod-alpha",
            "packet://request-context-tenant-a",
            "packet://rls-transcript-prod",
            "packet://egress-default-deny",
            "packet://backup-restore-run",
            "packet://audit-integrity-ledger",
        ]:
            self.assertNotIn(raw_evidence_ref, json.dumps(payload, sort_keys=True))
            self.assertNotIn(raw_evidence_ref, state_text)

        output_path = self.state_dir / "reports" / "connectorhub-human-gate-evidence-packet-contract-h04.json"
        output_payload = run_json(
            "connector",
            "human-gate",
            "evidence-packet-contract",
            "--scenario",
            "CS-CH-H04",
            "--state-dir",
            self.state_rel,
            "--output",
            str(output_path),
        )
        self.assertEqual(output_payload["status"], "success")
        self.assertEqual(output_payload["output_path"], str(output_path))
        self.assertTrue(output_path.exists())
        written_payload = json.loads(output_path.read_text())
        self.assertEqual(written_payload["command"], "cornerstone connector human-gate evidence-packet-contract")
        self.assertEqual(written_payload["output_path"], str(output_path))
        self.assertEqual(written_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(written_payload["summary"], output_payload["summary"])

    def test_connector_human_gate_evidence_packet_contract_rejects_non_human_gate_scenario(self) -> None:
        result = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-contract",
            "--scenario",
            "CS-CH-001",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["command"], "cornerstone connector human-gate evidence-packet-contract")
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(
            payload["connector_human_gate_evidence_packet_contract_report"]["schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_REPORT_SCHEMA,
        )
        self.assertEqual(
            payload["connector_human_gate_evidence_packet_contract_report"]["scenario_id"],
            "CS-CH-001",
        )
        self.assertEqual(
            {error["code"] for error in payload["errors"]},
            {"CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_UNSUPPORTED"},
        )

    def test_connector_human_gate_evidence_packet_file_contract_exposes_h04_packet_checklist_without_contents(self) -> None:
        payload = run_json(
            "connector",
            "human-gate",
            "evidence-packet-file-contract",
            "--scenario",
            "CS-CH-H04",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(payload["command"], "cornerstone connector human-gate evidence-packet-file-contract")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(payload["errors"], [])
        report = payload["connector_human_gate_evidence_packet_file_contract_report"]
        self.assertEqual(
            report["schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_REPORT_SCHEMA,
        )
        self.assertEqual(report["scenario_id"], "CS-CH-H04")
        self.assertEqual(report["status"], "operator_preparation_only")
        self.assertEqual(report["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(report["weakest_applicable_scenario_result"], "HUMAN_REQUIRED")
        self.assert_human_gate_completion_boundary(report)
        expected_packet_files = [
            {
                **item,
                "required": True,
                "raw_packet_file_contents_persisted_by_validator": False,
            }
            for item in CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS
        ]
        expected_packet_file_names = [
            item["packet_file"] for item in CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS
        ]
        expected_scaffold_plan = [
            {
                "step": 1,
                "operation": "prepare_packet_directory",
                "packet_directory": (
                    CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY
                ),
                "command": (
                    "mkdir -p "
                    f"{CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY}"
                ),
                "command_executed_by_report": False,
                "review_input_only": True,
                "acceptance_sufficient": False,
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
                "packet_file_contents_recorded_by_report": False,
                "packet_file_contents_persisted_by_report": False,
            }
        ]
        for index, item in enumerate(CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS, start=2):
            expected_scaffold_plan.append(
                {
                    "step": index,
                    "operation": "create_packet_file",
                    "packet_directory": (
                        CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY
                    ),
                    "packet_file": item["packet_file"],
                    "command": (
                        "touch "
                        f"{CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY}/"
                        f"{item['packet_file']}"
                    ),
                    "required": True,
                    "required_contents": item["required_contents"],
                    "command_executed_by_report": False,
                    "review_input_only": True,
                    "acceptance_sufficient": False,
                    "product_claim_allowed": False,
                    "pass_claim_allowed": False,
                    "packet_file_contents_recorded_by_report": False,
                    "packet_file_contents_persisted_by_report": False,
                }
            )
        expected_scaffold_commands = [
            item["command"] for item in expected_scaffold_plan
        ]
        contract = report["evidence_packet_file_contract"]
        self.assertEqual(
            contract["schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_SCHEMA,
        )
        self.assertEqual(contract["scenario_id"], "CS-CH-H04")
        self.assertEqual(contract["status"], "operator_preparation_only")
        self.assertEqual(contract["validation_scope"], "acceptance_packet_file_shape_only")
        self.assertEqual(contract["required_packet_files"], expected_packet_files)
        self.assertEqual(contract["required_packet_file_count"], len(expected_packet_files))
        self.assertEqual(contract["packet_file_names"], expected_packet_file_names)
        self.assertFalse(contract["raw_packet_file_contents_recorded_by_report"])
        self.assertFalse(contract["packet_file_contents_persisted_by_report"])
        self.assertFalse(contract["packet_file_contents_persisted_by_validator"])
        self.assertTrue(contract["review_input_only"])
        self.assertFalse(contract["acceptance_sufficient"])
        self.assertFalse(contract["product_claim_allowed"])
        self.assertFalse(contract["pass_claim_allowed"])
        self.assertEqual(contract["invalid_value_report_shape"], "packet_file_names_only")
        self.assertTrue(contract["packet_file_scaffold_plan_available"])
        self.assertEqual(
            contract["packet_file_scaffold_directory"],
            CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
        )
        self.assertEqual(contract["packet_file_scaffold_plan"], expected_scaffold_plan)
        self.assertEqual(contract["packet_file_scaffold_commands"], expected_scaffold_commands)
        self.assertEqual(
            contract["packet_file_scaffold_command_count"],
            CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
        )
        self.assertFalse(contract["packet_file_scaffold_plan_executed_by_report"])
        self.assertTrue(contract["packet_file_scaffold_plan_review_input_only"])
        self.assertFalse(contract["packet_file_scaffold_plan_acceptance_sufficient"])
        self.assertEqual(
            report["evidence_packet_file_contract_schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_SCHEMA,
        )
        self.assertEqual(report["required_packet_file_count"], len(expected_packet_files))
        self.assertEqual(report["required_packet_file_names"], expected_packet_file_names)
        self.assertEqual(report["required_packet_files"], expected_packet_files)
        self.assertFalse(report["raw_packet_file_contents_recorded_by_report"])
        self.assertFalse(report["packet_file_contents_persisted_by_report"])
        self.assertFalse(report["packet_file_contents_persisted_by_validator"])
        self.assertTrue(report["review_input_only"])
        self.assertFalse(report["acceptance_sufficient"])
        self.assertFalse(report["product_claim_allowed"])
        self.assertFalse(report["pass_claim_allowed"])
        self.assertEqual(report["invalid_value_report_shape"], "packet_file_names_only")
        self.assertTrue(report["packet_file_scaffold_plan_available"])
        self.assertEqual(
            report["packet_file_scaffold_directory"],
            CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
        )
        self.assertEqual(report["packet_file_scaffold_plan"], expected_scaffold_plan)
        self.assertEqual(report["packet_file_scaffold_commands"], expected_scaffold_commands)
        self.assertEqual(
            report["packet_file_scaffold_command_count"],
            CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
        )
        self.assertFalse(report["packet_file_scaffold_plan_executed_by_report"])
        self.assertTrue(report["packet_file_scaffold_plan_review_input_only"])
        self.assertFalse(report["packet_file_scaffold_plan_acceptance_sufficient"])
        non_mutation = report["non_mutation_evidence"]
        self.assertFalse(non_mutation["approval_collected_by_evidence_packet_file_contract"])
        self.assertEqual(non_mutation["human_decisions_recorded_by_evidence_packet_file_contract"], 0)
        self.assertEqual(non_mutation["commands_executed_by_evidence_packet_file_contract"], 0)
        self.assertEqual(non_mutation["live_provider_calls_executed_by_evidence_packet_file_contract"], 0)
        self.assertEqual(non_mutation["provider_mutations_executed_by_evidence_packet_file_contract"], 0)
        self.assertEqual(non_mutation["external_mutations_executed_by_evidence_packet_file_contract"], 0)
        self.assertFalse(non_mutation["secret_material_read_by_evidence_packet_file_contract"])
        self.assertFalse(non_mutation["raw_packet_file_contents_recorded_by_evidence_packet_file_contract"])
        self.assertFalse(non_mutation["packet_file_contents_persisted_by_evidence_packet_file_contract"])
        self.assertTrue(all(value in (0, False) for value in report["negative_evidence"].values()))
        self.assertTrue(payload["audit_refs"])
        self.assertIn(
            (
                "connector_human_gate_evidence_packet_file_contract:"
                f"{report['evidence_packet_file_contract_report_id']}"
            ),
            payload["evidence_refs"],
        )
        self.assertEqual(
            payload["ids"]["connector_human_gate_evidence_packet_file_contract_report_id"],
            report["evidence_packet_file_contract_report_id"],
        )
        self.assertEqual(
            payload["summary"],
            {
                "scenario_id": "CS-CH-H04",
                "status": "operator_preparation_only",
                "final_verdict": "HUMAN_REQUIRED",
                "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
                "evidence_packet_file_contract_report_id": report[
                    "evidence_packet_file_contract_report_id"
                ],
                "operator_rule": report["operator_rule"],
                "evidence_packet_file_contract_schema_version": (
                    CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_SCHEMA
                ),
                "required_packet_file_count": len(expected_packet_files),
                "required_packet_file_names": expected_packet_file_names,
                "raw_packet_file_contents_recorded_by_report": False,
                "packet_file_contents_persisted_by_report": False,
                "packet_file_contents_persisted_by_validator": False,
                "review_input_only": True,
                "acceptance_sufficient": False,
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
                "invalid_value_report_shape": "packet_file_names_only",
                "packet_file_scaffold_plan_available": True,
                "packet_file_scaffold_directory": (
                    CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY
                ),
                "packet_file_scaffold_command_count": (
                    CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT
                ),
                "packet_file_scaffold_commands": expected_scaffold_commands,
                "packet_file_scaffold_plan_executed_by_report": False,
                "packet_file_scaffold_plan_review_input_only": True,
                "packet_file_scaffold_plan_acceptance_sufficient": False,
                "commands_executed_by_evidence_packet_file_contract": 0,
                "live_provider_calls_executed_by_evidence_packet_file_contract": 0,
                "provider_mutations_executed_by_evidence_packet_file_contract": 0,
                "external_mutations_executed_by_evidence_packet_file_contract": 0,
                "human_acceptance_collected_by_evidence_packet_file_contract": False,
                "goal_completion_claim_blocked": True,
                "full_goal_completion_allowed": False,
                "completion_claim_boundary": CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
                "product_feature_claims": (
                    "CONNECTOR_HUB_H04_EVIDENCE_PACKET_FILE_CONTRACT_PREPARED_HUMAN_EVIDENCE_REQUIRED"
                ),
            },
        )
        report_path = (
            self.state_dir
            / "connector"
            / "human_gate_evidence_packet_file_contract_reports"
            / f"{report['evidence_packet_file_contract_report_id']}.json"
        )
        self.assertTrue(report_path.exists())
        state_text = state_file_texts(self.state_dir)
        self.assertIn("connector.human_gate_evidence_packet_file_contract.reported", state_text)
        for raw_packet_content in [
            "secret request context body",
            "live production token",
            "unredacted provider transcript",
            "private backup payload",
        ]:
            self.assertNotIn(raw_packet_content, json.dumps(payload, sort_keys=True))
            self.assertNotIn(raw_packet_content, state_text)

        output_path = (
            self.state_dir
            / "reports"
            / "connectorhub-human-gate-evidence-packet-file-contract-h04.json"
        )
        output_payload = run_json(
            "connector",
            "human-gate",
            "evidence-packet-file-contract",
            "--scenario",
            "CS-CH-H04",
            "--state-dir",
            self.state_rel,
            "--output",
            str(output_path),
        )
        self.assertEqual(output_payload["status"], "success")
        self.assertEqual(output_payload["output_path"], str(output_path))
        self.assertTrue(output_path.exists())
        written_payload = json.loads(output_path.read_text())
        self.assertEqual(written_payload["command"], "cornerstone connector human-gate evidence-packet-file-contract")
        self.assertEqual(written_payload["output_path"], str(output_path))
        self.assertEqual(written_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(written_payload["summary"], output_payload["summary"])

    def test_connector_human_gate_evidence_packet_scaffold_dry_run_and_write_are_preparation_only(self) -> None:
        expected_templates = []
        for item in CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS:
            content = connector_human_gate_evidence_packet_scaffold_template_content(
                item["packet_file"],
                item["required_contents"],
            )
            expected_templates.append(
                {
                    "packet_file": item["packet_file"],
                    "required": True,
                    "required_contents": item["required_contents"],
                    "template_only": True,
                    "template_content_sha256": hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    "template_content_line_count": len(content.splitlines()),
                    "human_evidence_recorded_by_template": False,
                    "acceptance_sufficient": False,
                    "product_claim_allowed": False,
                    "pass_claim_allowed": False,
                    "packet_file_contents_read_by_scaffold": False,
                }
            )
        expected_template_file_names = [
            item["packet_file"]
            for item in CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS
        ]
        expected_template_file_hashes = [
            {
                "packet_file": item["packet_file"],
                "template_content_sha256": item["template_content_sha256"],
                "template_content_line_count": item["template_content_line_count"],
            }
            for item in expected_templates
        ]
        payload = run_json(
            "connector",
            "human-gate",
            "evidence-packet-scaffold",
            "--scenario",
            "CS-CH-H04",
            "--packet-dir",
            CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(payload["command"], "cornerstone connector human-gate evidence-packet-scaffold")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(payload["errors"], [])
        report = payload["connector_human_gate_evidence_packet_scaffold_report"]
        self.assertEqual(
            report["schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_REPORT_SCHEMA,
        )
        self.assertEqual(report["scenario_id"], "CS-CH-H04")
        self.assertEqual(report["status"], "operator_preparation_only")
        self.assert_human_gate_completion_boundary(report)
        scaffold = report["evidence_packet_scaffold"]
        self.assertEqual(
            scaffold["schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_SCHEMA,
        )
        self.assertEqual(scaffold["scaffold_template_count"], len(expected_templates))
        self.assertEqual(scaffold["scaffold_templates"], expected_templates)
        self.assertFalse(scaffold["template_contents_included_in_report"])
        self.assertFalse(scaffold["packet_file_contents_read_by_scaffold"])
        self.assertFalse(scaffold["human_evidence_recorded_by_scaffold"])
        self.assertEqual(report["scaffold_templates"], expected_templates)
        self.assertEqual(report["scaffold_template_count"], len(expected_templates))
        self.assertFalse(report["write_requested"])
        self.assertFalse(report["write_executed"])
        self.assertEqual(report["written_packet_files"], [])
        self.assertEqual(report["written_packet_file_count"], 0)
        self.assertFalse(report["template_contents_included_in_report"])
        self.assertFalse(report["packet_file_contents_read_by_scaffold"])
        self.assertFalse(report["human_evidence_recorded_by_scaffold"])
        self.assertTrue(report["review_input_only"])
        self.assertFalse(report["acceptance_sufficient"])
        self.assertFalse(report["product_claim_allowed"])
        self.assertFalse(report["pass_claim_allowed"])
        non_mutation = report["non_mutation_evidence"]
        self.assertEqual(non_mutation["commands_executed_by_evidence_packet_scaffold"], 0)
        self.assertEqual(non_mutation["live_provider_calls_executed_by_evidence_packet_scaffold"], 0)
        self.assertEqual(non_mutation["provider_mutations_executed_by_evidence_packet_scaffold"], 0)
        self.assertEqual(non_mutation["external_mutations_executed_by_evidence_packet_scaffold"], 0)
        self.assertEqual(
            non_mutation["local_template_files_written_by_evidence_packet_scaffold"],
            0,
        )
        self.assertTrue(all(value in (0, False) for value in report["negative_evidence"].values()))
        self.assertEqual(
            payload["summary"],
            {
                "scenario_id": "CS-CH-H04",
                "status": "operator_preparation_only",
                "final_verdict": "HUMAN_REQUIRED",
                "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
                "evidence_packet_scaffold_report_id": report[
                    "evidence_packet_scaffold_report_id"
                ],
                "operator_rule": report["operator_rule"],
                "evidence_packet_scaffold_schema_version": (
                    CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_SCHEMA
                ),
                "packet_directory": (
                    CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY
                ),
                "write_requested": False,
                "write_executed": False,
                "scaffold_template_count": len(expected_templates),
                "scaffold_template_file_names": expected_template_file_names,
                "scaffold_template_file_hashes": expected_template_file_hashes,
                "written_packet_file_count": 0,
                "written_packet_file_names": [],
                "written_packet_files": [],
                "template_contents_included_in_summary": False,
                "template_contents_included_in_report": False,
                "packet_file_contents_read_by_scaffold": False,
                "human_evidence_recorded_by_scaffold": False,
                "review_input_only": True,
                "acceptance_sufficient": False,
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
                "commands_executed_by_evidence_packet_scaffold": 0,
                "live_provider_calls_executed_by_evidence_packet_scaffold": 0,
                "provider_mutations_executed_by_evidence_packet_scaffold": 0,
                "external_mutations_executed_by_evidence_packet_scaffold": 0,
                "local_template_files_written_by_evidence_packet_scaffold": 0,
                "human_acceptance_collected_by_evidence_packet_scaffold": False,
                "goal_completion_claim_blocked": True,
                "full_goal_completion_allowed": False,
                "completion_claim_boundary": CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
                "product_feature_claims": (
                    "CONNECTOR_HUB_H04_EVIDENCE_PACKET_SCAFFOLD_PREPARED_HUMAN_EVIDENCE_REQUIRED"
                ),
            },
        )
        payload_text = json.dumps(payload, sort_keys=True)
        self.assertNotIn("Evidence Ref:\n", payload_text)
        self.assertNotIn("RequestContext token value", payload_text)
        self.assertTrue(payload["audit_refs"])
        self.assertIn(
            (
                "connector_human_gate_evidence_packet_scaffold:"
                f"{report['evidence_packet_scaffold_report_id']}"
            ),
            payload["evidence_refs"],
        )

        packet_dir = self.state_dir / "h04-acceptance-packet"
        write_payload = run_json(
            "connector",
            "human-gate",
            "evidence-packet-scaffold",
            "--scenario",
            "CS-CH-H04",
            "--packet-dir",
            str(packet_dir),
            "--write",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        write_report = write_payload["connector_human_gate_evidence_packet_scaffold_report"]
        self.assertTrue(write_report["write_requested"])
        self.assertTrue(write_report["write_executed"])
        self.assertEqual(write_report["written_packet_file_count"], len(expected_templates))
        self.assertEqual(
            write_payload["summary"]["local_template_files_written_by_evidence_packet_scaffold"],
            len(expected_templates),
        )
        expected_written_packet_files = [
            {
                "packet_file": item["packet_file"],
                "path": (packet_dir / item["packet_file"]).as_posix(),
                "template_content_sha256": item["template_content_sha256"],
                "template_content_line_count": item["template_content_line_count"],
                "template_only": True,
                "human_evidence_recorded_by_template": False,
            }
            for item in expected_templates
        ]
        self.assertEqual(
            write_payload["summary"]["scaffold_template_file_names"],
            expected_template_file_names,
        )
        self.assertEqual(
            write_payload["summary"]["scaffold_template_file_hashes"],
            expected_template_file_hashes,
        )
        self.assertEqual(
            write_payload["summary"]["written_packet_file_names"],
            expected_template_file_names,
        )
        self.assertEqual(
            write_payload["summary"]["written_packet_files"],
            expected_written_packet_files,
        )
        self.assertFalse(write_payload["summary"]["template_contents_included_in_summary"])
        for item in CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS:
            packet_file = packet_dir / item["packet_file"]
            self.assertTrue(packet_file.exists())
            content = packet_file.read_text()
            self.assertIn("TEMPLATE_ONLY_NOT_HUMAN_EVIDENCE", content)
            self.assertIn("Do not mark H04 PASS", content)
        request_context_template = json.loads((packet_dir / "request-context-trace.json").read_text())
        self.assertEqual(
            request_context_template["status"],
            "TEMPLATE_ONLY_NOT_HUMAN_EVIDENCE",
        )
        self.assertEqual(request_context_template["evidence_ref"], "")

        blocked = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-scaffold",
            "--scenario",
            "CS-CH-H04",
            "--packet-dir",
            str(packet_dir),
            "--write",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(blocked.returncode, 1, blocked.stdout + blocked.stderr)
        blocked_payload = json.loads(blocked.stdout)
        self.assertEqual(blocked_payload["status"], "failed")
        self.assertEqual(
            blocked_payload["connector_human_gate_evidence_packet_scaffold_report"]["status"],
            "write_blocked_existing_files",
        )
        self.assertEqual(
            {error["code"] for error in blocked_payload["errors"]},
            {"CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_EXISTS"},
        )

    def test_connector_human_gate_evidence_packet_scaffold_rejects_non_human_gate_scenario(self) -> None:
        result = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-scaffold",
            "--scenario",
            "CS-CH-001",
            "--packet-dir",
            CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["command"], "cornerstone connector human-gate evidence-packet-scaffold")
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(
            payload["connector_human_gate_evidence_packet_scaffold_report"]["schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_REPORT_SCHEMA,
        )
        self.assertEqual(
            payload["connector_human_gate_evidence_packet_scaffold_report"]["scenario_id"],
            "CS-CH-001",
        )
        self.assertEqual(
            {error["code"] for error in payload["errors"]},
            {"CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_UNSUPPORTED"},
        )

    def test_connector_human_gate_evidence_packet_validate_missing_blank_and_ready_are_preparation_only(self) -> None:
        expected_packet_file_names = [
            item["packet_file"]
            for item in CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS
        ]
        missing_dir = self.state_dir / "missing-h04-acceptance-packet"
        missing_output = self.state_dir / "reports" / "h04-packet-validation-missing.json"
        missing = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H04",
            "--packet-dir",
            str(missing_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(missing_output),
            "--json",
        )
        self.assertEqual(missing.returncode, 1, missing.stdout + missing.stderr)
        missing_payload = json.loads(missing.stdout)
        self.assertEqual(
            missing_payload["command"],
            "cornerstone connector human-gate evidence-packet-validate",
        )
        self.assertEqual(missing_payload["status"], "blocked")
        self.assertEqual(missing_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertTrue(missing_output.exists())
        missing_report = missing_payload["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(
            missing_report["schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_VALIDATION_REPORT_SCHEMA,
        )
        self.assertEqual(missing_report["status"], "packet_not_submitted")
        self.assertFalse(missing_report["packet_directory_exists"])
        self.assertFalse(missing_report["packet_structurally_complete"])
        self.assertEqual(
            missing_report["missing_packet_file_count"],
            len(CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS),
        )
        self.assertEqual(missing_report["observed_packet_file_count"], 0)
        self.assertEqual(missing_report["hashed_packet_file_count"], 0)
        self.assertEqual(
            missing_payload["summary"]["evidence_packet_validation_report_id"],
            missing_report["evidence_packet_validation_report_id"],
        )
        self.assertEqual(
            missing_payload["summary"]["operator_rule"],
            missing_report["operator_rule"],
        )
        self.assertEqual(
            missing_payload["summary"]["missing_packet_file_names"],
            expected_packet_file_names,
        )
        self.assertEqual(missing_payload["summary"]["empty_packet_file_names"], [])
        self.assertEqual(missing_payload["summary"]["template_only_packet_file_names"], [])
        self.assertEqual(missing_payload["summary"]["hashed_packet_file_names"], [])
        self.assertEqual(
            missing_payload["summary"]["evidence_packet_file_contract_schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_SCHEMA,
        )
        self.assertEqual(
            missing_payload["summary"]["evidence_packet_scaffold_schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_SCHEMA,
        )
        self.assertFalse(missing_report["raw_packet_file_contents_included_in_report"])
        self.assertFalse(missing_report["raw_packet_file_contents_recorded_by_validator"])
        self.assertFalse(missing_report["packet_file_contents_persisted_by_validator"])
        self.assertTrue(missing_report["packet_file_hashes_recorded_by_validator"])
        self.assertFalse(missing_report["acceptance_sufficient"])
        self.assertFalse(missing_report["dependency_unlock_allowed_by_packet_validator"])
        self.assertFalse(missing_report["product_claim_allowed"])
        self.assertFalse(missing_report["pass_claim_allowed"])
        self.assert_human_gate_completion_boundary(missing_report)
        self.assertEqual(
            {error["code"] for error in missing_payload["errors"]},
            {"CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_STRUCTURAL_ISSUES"},
        )

        packet_dir = self.record_dir / "h04-acceptance-packet-for-validation"
        scaffold = run_json(
            "connector",
            "human-gate",
            "evidence-packet-scaffold",
            "--scenario",
            "CS-CH-H04",
            "--packet-dir",
            str(packet_dir),
            "--write",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(scaffold["status"], "success")
        blank = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H04",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(blank.returncode, 1, blank.stdout + blank.stderr)
        blank_payload = json.loads(blank.stdout)
        blank_report = blank_payload["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(blank_payload["status"], "blocked")
        self.assertEqual(blank_report["status"], "packet_structural_issues")
        self.assertEqual(blank_report["missing_packet_file_count"], 0)
        self.assertEqual(blank_report["empty_packet_file_count"], 0)
        self.assertEqual(
            blank_report["template_only_packet_file_count"],
            len(CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS),
        )
        self.assertEqual(
            blank_report["hashed_packet_file_count"],
            len(CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS),
        )
        self.assertEqual(blank_payload["summary"]["missing_packet_file_names"], [])
        self.assertEqual(blank_payload["summary"]["empty_packet_file_names"], [])
        self.assertEqual(
            blank_payload["summary"]["template_only_packet_file_names"],
            expected_packet_file_names,
        )
        self.assertEqual(
            blank_payload["summary"]["hashed_packet_file_names"],
            expected_packet_file_names,
        )
        self.assertTrue(
            all(entry["matches_blank_template"] for entry in blank_report["packet_files"])
        )
        self.assertTrue(all(value in (0, False) for value in blank_report["negative_evidence"].values()))

        raw_secret_markers = [
            "SHOULD_NOT_APPEAR_IN_PACKET_VALIDATION_OUTPUT",
            "prod-token-value-redacted-in-real-run",
        ]
        for item in CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS:
            packet_file = packet_dir / item["packet_file"]
            packet_file.write_text(
                "\n".join(
                    [
                        f"scenario_id=CS-CH-H04",
                        f"packet_file={item['packet_file']}",
                        f"evidence_ref=fixture:{item['packet_file']}",
                        "redaction_status=redacted",
                        raw_secret_markers[0],
                        raw_secret_markers[1],
                        "reviewer=test-reviewer",
                    ]
                )
                + "\n"
            )
        ready_output = self.state_dir / "reports" / "h04-packet-validation-ready.json"
        ready = run_json(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H04",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(ready_output),
            "--json",
        )
        self.assertEqual(ready["status"], "success")
        self.assertEqual(ready["final_verdict"], "HUMAN_REQUIRED")
        ready_report = ready["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(ready_report["status"], "packet_structurally_complete")
        self.assertTrue(ready_report["packet_structurally_complete"])
        self.assertEqual(ready_report["missing_packet_file_count"], 0)
        self.assertEqual(ready_report["empty_packet_file_count"], 0)
        self.assertEqual(ready_report["template_only_packet_file_count"], 0)
        self.assertEqual(
            ready_report["observed_packet_file_count"],
            len(CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS),
        )
        self.assertEqual(
            ready_report["hashed_packet_file_count"],
            len(CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS),
        )
        self.assertEqual(ready["summary"]["missing_packet_file_names"], [])
        self.assertEqual(ready["summary"]["empty_packet_file_names"], [])
        self.assertEqual(ready["summary"]["template_only_packet_file_names"], [])
        self.assertEqual(ready["summary"]["hashed_packet_file_names"], expected_packet_file_names)
        self.assertFalse(ready_report["acceptance_sufficient"])
        self.assertFalse(ready_report["dependency_unlock_allowed_by_packet_validator"])
        self.assertFalse(ready_report["product_claim_allowed"])
        self.assertFalse(ready_report["pass_claim_allowed"])
        self.assertFalse(ready_report["raw_packet_file_contents_included_in_report"])
        self.assertFalse(ready_report["raw_packet_file_contents_recorded_by_validator"])
        self.assertFalse(ready_report["packet_file_contents_persisted_by_validator"])
        self.assertTrue(ready_report["packet_file_hashes_recorded_by_validator"])
        for entry in ready_report["packet_files"]:
            packet_path = packet_dir / entry["packet_file"]
            metadata = connector_human_gate_packet_file_metadata(packet_path)
            self.assertTrue(entry["present"])
            self.assertFalse(entry["matches_blank_template"])
            self.assertEqual(entry["sha256"], metadata["sha256"])
            self.assertEqual(entry["size_bytes"], metadata["size_bytes"])
            self.assertEqual(entry["line_count"], metadata["line_count"])
            self.assertIsNone(entry.get("raw_contents"))
        self.assertTrue(ready["audit_refs"])
        self.assertIn(
            (
                "connector_human_gate_evidence_packet_validation:"
                f"{ready_report['evidence_packet_validation_report_id']}"
            ),
            ready["evidence_refs"],
        )
        self.assertEqual(
            ready["summary"]["product_feature_claims"],
            "CONNECTOR_HUB_H04_EVIDENCE_PACKET_VALIDATION_PREPARED_HUMAN_EVIDENCE_REQUIRED",
        )
        state_text = state_file_texts(self.state_dir)
        ready_text = json.dumps(ready, sort_keys=True)
        for marker in raw_secret_markers:
            self.assertNotIn(marker, ready_text)
            self.assertNotIn(marker, ready_output.read_text())
            self.assertNotIn(marker, state_text)

    def test_connector_human_gate_evidence_packet_validate_rejects_non_human_gate_scenario(self) -> None:
        result = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-001",
            "--packet-dir",
            CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["command"], "cornerstone connector human-gate evidence-packet-validate")
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(
            payload["connector_human_gate_evidence_packet_validation_report"]["schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_VALIDATION_REPORT_SCHEMA,
        )
        self.assertEqual(
            payload["connector_human_gate_evidence_packet_validation_report"]["scenario_id"],
            "CS-CH-001",
        )
        self.assertEqual(
            {error["code"] for error in payload["errors"]},
            {"CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_VALIDATION_UNSUPPORTED"},
        )

    def test_connector_human_gate_evidence_packet_record_draft_requires_complete_packet_and_human_completion(self) -> None:
        missing_dir = self.state_dir / "missing-h04-packet-for-draft"
        blocked = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-record-draft",
            "--scenario",
            "CS-CH-H04",
            "--packet-dir",
            str(missing_dir),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(blocked.returncode, 1, blocked.stdout + blocked.stderr)
        blocked_payload = json.loads(blocked.stdout)
        self.assertEqual(blocked_payload["status"], "blocked")
        blocked_report = blocked_payload["connector_human_gate_evidence_packet_record_draft_report"]
        self.assertEqual(
            blocked_report["schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_RECORD_DRAFT_REPORT_SCHEMA,
        )
        self.assertEqual(blocked_report["status"], "packet_not_ready_for_record_draft")
        self.assertFalse(blocked_report["draft_record_available"])
        self.assertFalse(blocked_report["acceptance_sufficient"])
        self.assertFalse(blocked_report["dependency_unlock_allowed_by_record_draft"])
        self.assertFalse(blocked_report["product_claim_allowed"])
        self.assertFalse(blocked_report["pass_claim_allowed"])
        self.assertEqual(
            blocked_payload["summary"]["evidence_packet_record_draft_report_id"],
            blocked_report["evidence_packet_record_draft_report_id"],
        )
        self.assertEqual(
            blocked_payload["summary"]["operator_rule"],
            blocked_report["operator_rule"],
        )
        self.assertEqual(
            blocked_payload["summary"]["packet_validation_report_id"],
            blocked_report["packet_validation_report_id"],
        )
        self.assertFalse(blocked_payload["summary"]["draft_record_output_written_by_runtime"])
        self.assertFalse(blocked_payload["summary"]["draft_record_included_in_summary"])
        self.assertFalse(
            blocked_payload["summary"][
                "human_decision_recorded_by_evidence_packet_record_draft"
            ]
        )
        self.assertFalse(
            blocked_payload["summary"][
                "raw_packet_file_contents_recorded_by_evidence_packet_record_draft"
            ]
        )
        self.assertFalse(
            blocked_payload["summary"][
                "packet_file_contents_persisted_by_evidence_packet_record_draft"
            ]
        )
        self.assertNotIn("draft_record", blocked_payload["summary"])
        expected_incomplete_fields = [
            "decision",
            "reviewer",
            "review_timestamp",
            "senior_review_perspective_findings",
        ]
        self.assertEqual(
            blocked_report["draft_record_intentionally_incomplete_fields"],
            expected_incomplete_fields,
        )
        self.assertEqual(
            blocked_payload["summary"]["draft_record_intentionally_incomplete_fields"],
            expected_incomplete_fields,
        )
        self.assertEqual(
            blocked_payload["summary"]["draft_record_intentionally_incomplete_field_count"],
            len(expected_incomplete_fields),
        )
        expected_draft_validation_output_command = (
            "cornerstone connector human-gate validate-record --scenario CS-CH-H04 "
            "--record-file <reviewer-record-draft.json> --json "
            "--output <redacted-validation-envelope.json>"
        )
        self.assertEqual(
            blocked_report["draft_record_validation_output_command"],
            expected_draft_validation_output_command,
        )
        self.assertEqual(
            blocked_payload["summary"]["draft_record_validation_output_command"],
            expected_draft_validation_output_command,
        )

        packet_dir = self.record_dir / "h04-packet-for-record-draft"
        scaffold = run_json(
            "connector",
            "human-gate",
            "evidence-packet-scaffold",
            "--scenario",
            "CS-CH-H04",
            "--packet-dir",
            str(packet_dir),
            "--write",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(scaffold["status"], "success")
        raw_secret_markers = [
            "SHOULD_NOT_APPEAR_IN_PACKET_RECORD_DRAFT_OUTPUT",
            "AKIAIOSFODNN7EXAMPLE",
        ]
        for item in CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS:
            packet_file = packet_dir / item["packet_file"]
            packet_file.write_text(
                "\n".join(
                    [
                        "scenario_id=CS-CH-H04",
                        f"packet_file={item['packet_file']}",
                        "redaction_status=redacted",
                        raw_secret_markers[0],
                        raw_secret_markers[1],
                        "operator_evidence=redacted",
                    ]
                )
                + "\n"
            )

        draft_output = self.state_dir / "reports" / "h04-packet-record-draft.json"
        record_output = self.state_dir / "reports" / "h04-reviewer-record-draft.json"
        draft = run_json(
            "connector",
            "human-gate",
            "evidence-packet-record-draft",
            "--scenario",
            "CS-CH-H04",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(draft_output),
            "--record-output",
            str(record_output),
            "--json",
        )
        self.assertEqual(draft["status"], "success")
        self.assertEqual(draft["final_verdict"], "HUMAN_REQUIRED")
        draft_report = draft["connector_human_gate_evidence_packet_record_draft_report"]
        self.assertEqual(draft_report["status"], "draft_record_requires_human_completion")
        self.assertTrue(draft_report["packet_structurally_complete"])
        self.assertTrue(draft_report["draft_record_available"])
        self.assertFalse(draft_report["acceptance_sufficient"])
        self.assertFalse(draft_report["dependency_unlock_allowed_by_record_draft"])
        self.assertFalse(draft_report["product_claim_allowed"])
        self.assertFalse(draft_report["pass_claim_allowed"])
        self.assertFalse(draft_report["raw_packet_file_contents_included_in_report"])
        self.assertFalse(draft_report["raw_packet_file_contents_recorded_by_draft"])
        self.assertFalse(draft_report["packet_file_contents_persisted_by_draft"])
        self.assertFalse(draft_report["packet_directory_path_recorded_by_draft"])
        self.assertTrue(draft["summary"]["draft_record_output_written"])
        self.assertEqual(
            draft["summary"]["evidence_packet_record_draft_report_id"],
            draft_report["evidence_packet_record_draft_report_id"],
        )
        self.assertEqual(
            draft["summary"]["operator_rule"],
            draft_report["operator_rule"],
        )
        self.assertEqual(
            draft["summary"]["packet_validation_report_id"],
            draft_report["packet_validation_report_id"],
        )
        self.assertFalse(draft["summary"]["draft_record_output_written_by_runtime"])
        self.assertFalse(draft["summary"]["draft_record_included_in_summary"])
        self.assertFalse(
            draft["summary"]["human_decision_recorded_by_evidence_packet_record_draft"]
        )
        self.assertFalse(
            draft["summary"][
                "raw_packet_file_contents_recorded_by_evidence_packet_record_draft"
            ]
        )
        self.assertFalse(
            draft["summary"]["packet_file_contents_persisted_by_evidence_packet_record_draft"]
        )
        self.assertNotIn("draft_record", draft["summary"])
        self.assertEqual(
            draft_report["draft_record_intentionally_incomplete_fields"],
            expected_incomplete_fields,
        )
        self.assertEqual(
            draft["summary"]["draft_record_intentionally_incomplete_fields"],
            expected_incomplete_fields,
        )
        self.assertEqual(
            draft["summary"]["draft_record_intentionally_incomplete_field_count"],
            len(expected_incomplete_fields),
        )
        self.assertEqual(
            draft_report["draft_record_validation_output_command"],
            expected_draft_validation_output_command,
        )
        self.assertEqual(
            draft["summary"]["draft_record_validation_output_command"],
            expected_draft_validation_output_command,
        )
        self.assertTrue(record_output.exists())

        draft_record = json.loads(record_output.read_text())
        self.assertEqual(draft_record["scenario_id"], "CS-CH-H04")
        self.assertEqual(draft_record["decision"], "")
        self.assertEqual(draft_record["reviewer"], "")
        self.assertEqual(draft_record["review_timestamp"], "")
        self.assertTrue(draft_record["environment_topology_ref"].startswith("topology:"))
        self.assertTrue(draft_record["request_context_proof"].startswith("request_context:"))
        self.assertTrue(all(ref.startswith("db_policy:") for ref in draft_record["db_policy_transcripts"]))
        self.assertTrue(all(ref.startswith("egress:") for ref in draft_record["network_egress_transcripts"]))
        self.assertTrue(all(ref.startswith("backup_restore:") for ref in draft_record["backup_restore_evidence"]))
        self.assertTrue(draft_record["audit_integrity_report"].startswith("audit_integrity:"))
        self.assertTrue(draft_record["evidence_manifest_ref"].startswith("evidence_manifest:"))
        self.assertEqual(len(draft_record["evidence_packet_manifest"]), 4)
        self.assertTrue(
            all(item["redaction_status"] == "redacted" for item in draft_record["evidence_packet_manifest"])
        )
        self.assertTrue(
            all(item["evidence_ref"].startswith("evidence_manifest:") for item in draft_record["evidence_packet_manifest"])
        )
        self.assertTrue(
            all(value == "" for value in draft_record["senior_review_perspective_findings"].values())
        )

        validation = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H04",
            "--record-file",
            str(record_output),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(validation.returncode, 1, validation.stdout + validation.stderr)
        validation_payload = json.loads(validation.stdout)
        validation_report = validation_payload["connector_human_gate_record_validation"]
        self.assertEqual(validation_report["status"], "record_structurally_invalid")
        self.assertFalse(validation_report["dependency_unlock_allowed_by_validator"])
        self.assertFalse(validation_report["product_claim_allowed"])
        self.assertFalse(validation_report["pass_claim_allowed_by_validator"])
        self.assertIn("decision_not_allowed", validation_report["structural_errors"])
        self.assertIn("empty_required_fields", validation_report["structural_errors"])
        self.assertIn("empty_senior_review_perspectives", validation_report["structural_errors"])

        draft_text = json.dumps(draft, sort_keys=True)
        record_text = record_output.read_text()
        state_text = state_file_texts(self.state_dir)
        for marker in raw_secret_markers:
            self.assertNotIn(marker, draft_text)
            self.assertNotIn(marker, draft_output.read_text())
            self.assertNotIn(marker, record_text)
            self.assertNotIn(marker, state_text)

    def test_connector_human_gate_h01_github_readonly_packet_workflow_prepares_hash_only_record_draft(self) -> None:
        contract = run_json(
            "connector",
            "human-gate",
            "evidence-packet-contract",
            "--scenario",
            "CS-CH-H01",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(contract["status"], "success")
        self.assertEqual(contract["final_verdict"], "HUMAN_REQUIRED")
        contract_report = contract["connector_human_gate_evidence_packet_contract_report"]
        self.assertEqual(contract_report["scenario_id"], "CS-CH-H01")
        self.assertEqual(contract_report["required_evidence_packet_manifest_count"], 5)
        self.assertEqual(
            contract["summary"]["product_feature_claims"],
            "CONNECTOR_HUB_H01_EVIDENCE_PACKET_CONTRACT_PREPARED_HUMAN_EVIDENCE_REQUIRED",
        )

        file_contract = run_json(
            "connector",
            "human-gate",
            "evidence-packet-file-contract",
            "--scenario",
            "CS-CH-H01",
            "--state-dir",
            self.state_rel,
        )
        file_report = file_contract["connector_human_gate_evidence_packet_file_contract_report"]
        expected_names = [item["packet_file"] for item in CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_ITEMS]
        self.assertEqual(file_contract["status"], "success")
        self.assertEqual(file_report["required_packet_file_names"], expected_names)
        self.assertEqual(
            file_report["packet_file_scaffold_directory"],
            CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
        )
        self.assertEqual(
            file_report["packet_file_scaffold_command_count"],
            CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
        )

        package = run_json(
            "connector",
            "human-gate",
            "package",
            "--scenario",
            "CS-CH-H01",
            "--state-dir",
            self.state_rel,
        )["connector_human_gate_package"]
        self.assert_h01_evidence_packet_workflow(
            package["remaining_human_evidence_summary"]["evidence_packet_workflow"]
        )
        self.assert_h01_evidence_packet_workflow(
            package["proposed_record_template"]["reviewer_checklist"]["evidence_packet_workflow"]
        )
        self.assert_h01_evidence_packet_workflow(
            package["scenario_delivery_unit_plan"]["evidence_packet_workflow"]
        )

        packet_dir = self.record_dir / "h01-github-readonly-packet"
        scaffold = run_json(
            "connector",
            "human-gate",
            "evidence-packet-scaffold",
            "--scenario",
            "CS-CH-H01",
            "--packet-dir",
            str(packet_dir),
            "--write",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        scaffold_report = scaffold["connector_human_gate_evidence_packet_scaffold_report"]
        self.assertEqual(scaffold["status"], "success")
        self.assertEqual(scaffold_report["scaffold_template_count"], len(expected_names))
        self.assertEqual(scaffold_report["written_packet_file_count"], len(expected_names))
        self.assertFalse(scaffold_report["acceptance_sufficient"])
        for packet_name in expected_names:
            self.assertTrue((packet_dir / packet_name).exists())
        github_scope = (packet_dir / "github-scope.md").read_text()
        self.assertIn("Scenario: CS-CH-H01", github_scope)
        self.assertIn("Do not mark H01 PASS", github_scope)

        blank = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H01",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(blank.returncode, 1, blank.stdout + blank.stderr)
        blank_payload = json.loads(blank.stdout)
        blank_report = blank_payload["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(blank_payload["status"], "blocked")
        self.assertEqual(blank_report["template_only_packet_file_count"], len(expected_names))
        self.assertFalse(blank_report["dependency_unlock_allowed_by_packet_validator"])

        raw_secret_markers = [
            "SHOULD_NOT_APPEAR_IN_H01_PACKET_OUTPUT",
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        ]
        for item in CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_ITEMS:
            packet_file = packet_dir / item["packet_file"]
            packet_file.write_text(
                "\n".join(
                    [
                        "scenario_id=CS-CH-H01",
                        f"packet_file={item['packet_file']}",
                        "redaction_status=redacted",
                        raw_secret_markers[0],
                        raw_secret_markers[1],
                        "operator_evidence=redacted",
                    ]
                )
                + "\n"
            )

        validation_output = self.state_dir / "reports" / "h01-packet-validation-ready.json"
        ready = run_json(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H01",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(validation_output),
            "--json",
        )
        ready_report = ready["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(ready["status"], "success")
        self.assertEqual(ready_report["status"], "packet_structurally_complete")
        self.assertEqual(ready_report["hashed_packet_file_count"], len(expected_names))
        self.assertFalse(ready_report["acceptance_sufficient"])
        self.assertFalse(ready_report["dependency_unlock_allowed_by_packet_validator"])

        draft_output = self.state_dir / "reports" / "h01-packet-record-draft.json"
        record_output = self.state_dir / "reports" / "h01-reviewer-record-draft.json"
        draft = run_json(
            "connector",
            "human-gate",
            "evidence-packet-record-draft",
            "--scenario",
            "CS-CH-H01",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(draft_output),
            "--record-output",
            str(record_output),
            "--json",
        )
        draft_report = draft["connector_human_gate_evidence_packet_record_draft_report"]
        self.assertEqual(draft["status"], "success")
        self.assertEqual(draft_report["status"], "draft_record_requires_human_completion")
        self.assertTrue(draft_report["draft_record_available"])
        self.assertFalse(draft_report["dependency_unlock_allowed_by_record_draft"])
        self.assertTrue(record_output.exists())
        draft_record = json.loads(record_output.read_text())
        self.assertEqual(draft_record["scenario_id"], "CS-CH-H01")
        self.assertEqual(draft_record["decision"], "")
        self.assertEqual(draft_record["reviewer"], "")
        self.assertEqual(draft_record["review_timestamp"], "")
        self.assertEqual(draft_record["issues_or_exceptions"], "")
        self.assertTrue(draft_record["github_app_installation_id_redacted"].startswith("github_installation:"))
        self.assertTrue(draft_record["selected_repositories"][0].startswith("selected_repositories:"))
        self.assertTrue(draft_record["permission_snapshot"].startswith("github_permission:"))
        self.assertTrue(draft_record["call_ledger"].startswith("github_call_ledger:"))
        self.assertTrue(draft_record["delivery_refs"][0].startswith("connector_delivery:"))
        self.assertTrue(all(ref.startswith("audit:") for ref in draft_record["audit_refs"]))
        self.assertTrue(draft_record["zero_write_proof"].startswith("zero_write_proof:"))
        self.assertEqual(
            draft_record["dependency_human_gate_refs"],
            {"CS-CH-H04": "", "CS-CH-H07": ""},
        )
        self.assertEqual(len(draft_record["evidence_packet_manifest"]), 5)
        self.assertTrue(
            all(item["redaction_status"] == "redacted" for item in draft_record["evidence_packet_manifest"])
        )

        validation = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H01",
            "--record-file",
            str(record_output),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(validation.returncode, 1, validation.stdout + validation.stderr)
        validation_payload = json.loads(validation.stdout)
        validation_report = validation_payload["connector_human_gate_record_validation"]
        self.assertEqual(validation_report["status"], "record_structurally_invalid")
        self.assertFalse(validation_report["dependency_unlock_allowed_by_validator"])
        self.assertIn("decision_not_allowed", validation_report["structural_errors"])
        self.assertIn("empty_required_fields", validation_report["structural_errors"])
        self.assertIn("empty_senior_review_perspectives", validation_report["structural_errors"])
        self.assertIn("missing_dependency_human_gate_refs", validation_report["structural_errors"])

        state_text = state_file_texts(self.state_dir)
        output_text = json.dumps(draft, sort_keys=True) + draft_output.read_text() + record_output.read_text()
        for marker in raw_secret_markers:
            self.assertNotIn(marker, json.dumps(ready, sort_keys=True))
            self.assertNotIn(marker, validation_output.read_text())
            self.assertNotIn(marker, output_text)
            self.assertNotIn(marker, state_text)

    def test_connector_human_gate_h02_macos_permission_packet_workflow_prepares_hash_only_record_draft(self) -> None:
        contract = run_json(
            "connector",
            "human-gate",
            "evidence-packet-contract",
            "--scenario",
            "CS-CH-H02",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(contract["status"], "success")
        self.assertEqual(contract["final_verdict"], "HUMAN_REQUIRED")
        contract_report = contract["connector_human_gate_evidence_packet_contract_report"]
        self.assertEqual(contract_report["scenario_id"], "CS-CH-H02")
        self.assertEqual(contract_report["required_evidence_packet_manifest_count"], 4)
        self.assertEqual(
            contract["summary"]["product_feature_claims"],
            "CONNECTOR_HUB_H02_EVIDENCE_PACKET_CONTRACT_PREPARED_HUMAN_EVIDENCE_REQUIRED",
        )

        file_contract = run_json(
            "connector",
            "human-gate",
            "evidence-packet-file-contract",
            "--scenario",
            "CS-CH-H02",
            "--state-dir",
            self.state_rel,
        )
        file_report = file_contract["connector_human_gate_evidence_packet_file_contract_report"]
        expected_names = [item["packet_file"] for item in CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_ITEMS]
        self.assertEqual(file_contract["status"], "success")
        self.assertEqual(file_report["required_packet_file_names"], expected_names)
        self.assertEqual(
            file_report["packet_file_scaffold_directory"],
            CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
        )
        self.assertEqual(
            file_report["packet_file_scaffold_command_count"],
            CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
        )

        package = run_json(
            "connector",
            "human-gate",
            "package",
            "--scenario",
            "CS-CH-H02",
            "--state-dir",
            self.state_rel,
        )["connector_human_gate_package"]
        self.assert_h02_evidence_packet_workflow(
            package["remaining_human_evidence_summary"]["evidence_packet_workflow"]
        )
        self.assert_h02_evidence_packet_workflow(
            package["proposed_record_template"]["reviewer_checklist"]["evidence_packet_workflow"]
        )
        self.assert_h02_evidence_packet_workflow(
            package["scenario_delivery_unit_plan"]["evidence_packet_workflow"]
        )

        packet_dir = self.record_dir / "h02-macos-permission-packet"
        scaffold = run_json(
            "connector",
            "human-gate",
            "evidence-packet-scaffold",
            "--scenario",
            "CS-CH-H02",
            "--packet-dir",
            str(packet_dir),
            "--write",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        scaffold_report = scaffold["connector_human_gate_evidence_packet_scaffold_report"]
        self.assertEqual(scaffold["status"], "success")
        self.assertEqual(scaffold_report["scaffold_template_count"], len(expected_names))
        self.assertEqual(scaffold_report["written_packet_file_count"], len(expected_names))
        self.assertFalse(scaffold_report["acceptance_sufficient"])
        for packet_name in expected_names:
            self.assertTrue((packet_dir / packet_name).exists())
        review_scope = (packet_dir / "macos-review-scope.md").read_text()
        self.assertIn("Scenario: CS-CH-H02", review_scope)
        self.assertIn("Do not mark H02 PASS", review_scope)

        blank = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H02",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(blank.returncode, 1, blank.stdout + blank.stderr)
        blank_payload = json.loads(blank.stdout)
        blank_report = blank_payload["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(blank_payload["status"], "blocked")
        self.assertEqual(blank_report["template_only_packet_file_count"], len(expected_names))
        self.assertFalse(blank_report["dependency_unlock_allowed_by_packet_validator"])

        raw_secret_markers = [
            "SHOULD_NOT_APPEAR_IN_H02_PACKET_OUTPUT",
            "-----BEGIN PRIVATE KEY-----",
        ]
        for item in CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_ITEMS:
            packet_file = packet_dir / item["packet_file"]
            packet_file.write_text(
                "\n".join(
                    [
                        "scenario_id=CS-CH-H02",
                        f"packet_file={item['packet_file']}",
                        "redaction_status=redacted",
                        raw_secret_markers[0],
                        raw_secret_markers[1],
                        "operator_evidence=redacted",
                    ]
                )
                + "\n"
            )

        validation_output = self.state_dir / "reports" / "h02-packet-validation-ready.json"
        ready = run_json(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H02",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(validation_output),
            "--json",
        )
        ready_report = ready["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(ready["status"], "success")
        self.assertEqual(ready_report["status"], "packet_structurally_complete")
        self.assertEqual(ready_report["hashed_packet_file_count"], len(expected_names))
        self.assertFalse(ready_report["acceptance_sufficient"])
        self.assertFalse(ready_report["dependency_unlock_allowed_by_packet_validator"])

        draft_output = self.state_dir / "reports" / "h02-packet-record-draft.json"
        record_output = self.state_dir / "reports" / "h02-reviewer-record-draft.json"
        draft = run_json(
            "connector",
            "human-gate",
            "evidence-packet-record-draft",
            "--scenario",
            "CS-CH-H02",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(draft_output),
            "--record-output",
            str(record_output),
            "--json",
        )
        draft_report = draft["connector_human_gate_evidence_packet_record_draft_report"]
        self.assertEqual(draft["status"], "success")
        self.assertEqual(draft_report["status"], "draft_record_requires_human_completion")
        self.assertTrue(draft_report["draft_record_available"])
        self.assertFalse(draft_report["dependency_unlock_allowed_by_record_draft"])
        self.assertTrue(record_output.exists())
        draft_record = json.loads(record_output.read_text())
        self.assertEqual(draft_record["scenario_id"], "CS-CH-H02")
        self.assertEqual(draft_record["decision"], "")
        self.assertEqual(draft_record["reviewer"], "")
        self.assertEqual(draft_record["review_timestamp"], "")
        self.assertEqual(draft_record["issues_or_exceptions"], "")
        self.assertTrue(draft_record["device_os_version_redacted"].startswith("macos_device:"))
        self.assertTrue(draft_record["consent_record"].startswith("consent_record:"))
        self.assertTrue(draft_record["permission_state_snapshot"].startswith("macos_permission:"))
        self.assertTrue(draft_record["first_sample_ref"].startswith("capture_sample:"))
        self.assertEqual(len(draft_record["pause_revoke_timestamps"]), 2)
        self.assertTrue(draft_record["pause_revoke_timestamps"][0].startswith("capture_pause:"))
        self.assertTrue(draft_record["pause_revoke_timestamps"][1].startswith("capture_revoke:"))
        self.assertTrue(draft_record["screenshots_or_recording_ref"].startswith("redacted_recording:"))
        self.assertTrue(all(ref.startswith("audit:") for ref in draft_record["audit_refs"]))
        self.assertEqual(
            draft_record["dependency_human_gate_refs"],
            {"CS-CH-H04": "", "CS-CH-H07": ""},
        )
        self.assertEqual(len(draft_record["evidence_packet_manifest"]), 4)
        self.assertTrue(
            all(item["redaction_status"] == "redacted" for item in draft_record["evidence_packet_manifest"])
        )

        validation = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H02",
            "--record-file",
            str(record_output),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(validation.returncode, 1, validation.stdout + validation.stderr)
        validation_payload = json.loads(validation.stdout)
        validation_report = validation_payload["connector_human_gate_record_validation"]
        self.assertEqual(validation_report["status"], "record_structurally_invalid")
        self.assertFalse(validation_report["dependency_unlock_allowed_by_validator"])
        self.assertIn("decision_not_allowed", validation_report["structural_errors"])
        self.assertIn("empty_required_fields", validation_report["structural_errors"])
        self.assertIn("empty_senior_review_perspectives", validation_report["structural_errors"])
        self.assertIn("missing_dependency_human_gate_refs", validation_report["structural_errors"])

        state_text = state_file_texts(self.state_dir)
        output_text = json.dumps(draft, sort_keys=True) + draft_output.read_text() + record_output.read_text()
        for marker in raw_secret_markers:
            self.assertNotIn(marker, json.dumps(ready, sort_keys=True))
            self.assertNotIn(marker, validation_output.read_text())
            self.assertNotIn(marker, output_text)
            self.assertNotIn(marker, state_text)

    def test_connector_human_gate_h03_chrome_privacy_packet_workflow_prepares_hash_only_record_draft(self) -> None:
        contract = run_json(
            "connector",
            "human-gate",
            "evidence-packet-contract",
            "--scenario",
            "CS-CH-H03",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(contract["status"], "success")
        self.assertEqual(contract["final_verdict"], "HUMAN_REQUIRED")
        contract_report = contract["connector_human_gate_evidence_packet_contract_report"]
        self.assertEqual(contract_report["scenario_id"], "CS-CH-H03")
        self.assertEqual(contract_report["required_evidence_packet_manifest_count"], 4)
        self.assertEqual(
            contract["summary"]["product_feature_claims"],
            "CONNECTOR_HUB_H03_EVIDENCE_PACKET_CONTRACT_PREPARED_HUMAN_EVIDENCE_REQUIRED",
        )

        file_contract = run_json(
            "connector",
            "human-gate",
            "evidence-packet-file-contract",
            "--scenario",
            "CS-CH-H03",
            "--state-dir",
            self.state_rel,
        )
        file_report = file_contract["connector_human_gate_evidence_packet_file_contract_report"]
        expected_names = [item["packet_file"] for item in CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_ITEMS]
        self.assertEqual(file_contract["status"], "success")
        self.assertEqual(file_report["required_packet_file_names"], expected_names)
        self.assertEqual(
            file_report["packet_file_scaffold_directory"],
            CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
        )
        self.assertEqual(
            file_report["packet_file_scaffold_command_count"],
            CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
        )

        package = run_json(
            "connector",
            "human-gate",
            "package",
            "--scenario",
            "CS-CH-H03",
            "--state-dir",
            self.state_rel,
        )["connector_human_gate_package"]
        self.assert_h03_evidence_packet_workflow(
            package["remaining_human_evidence_summary"]["evidence_packet_workflow"]
        )
        self.assert_h03_evidence_packet_workflow(
            package["proposed_record_template"]["reviewer_checklist"]["evidence_packet_workflow"]
        )
        self.assert_h03_evidence_packet_workflow(
            package["scenario_delivery_unit_plan"]["evidence_packet_workflow"]
        )

        packet_dir = self.record_dir / "h03-chrome-privacy-packet"
        scaffold = run_json(
            "connector",
            "human-gate",
            "evidence-packet-scaffold",
            "--scenario",
            "CS-CH-H03",
            "--packet-dir",
            str(packet_dir),
            "--write",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        scaffold_report = scaffold["connector_human_gate_evidence_packet_scaffold_report"]
        self.assertEqual(scaffold["status"], "success")
        self.assertEqual(scaffold_report["scaffold_template_count"], len(expected_names))
        self.assertEqual(scaffold_report["written_packet_file_count"], len(expected_names))
        self.assertFalse(scaffold_report["acceptance_sufficient"])
        for packet_name in expected_names:
            self.assertTrue((packet_dir / packet_name).exists())
        review_scope = (packet_dir / "chrome-review-scope.md").read_text()
        self.assertIn("Scenario: CS-CH-H03", review_scope)
        self.assertIn("Do not mark H03 PASS", review_scope)

        blank = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H03",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(blank.returncode, 1, blank.stdout + blank.stderr)
        blank_payload = json.loads(blank.stdout)
        blank_report = blank_payload["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(blank_payload["status"], "blocked")
        self.assertEqual(blank_report["template_only_packet_file_count"], len(expected_names))
        self.assertFalse(blank_report["dependency_unlock_allowed_by_packet_validator"])

        raw_secret_markers = [
            "SHOULD_NOT_APPEAR_IN_H03_PACKET_OUTPUT",
            "chrome-history://private-url",
            "-----BEGIN PRIVATE KEY-----",
        ]
        for item in CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_ITEMS:
            packet_file = packet_dir / item["packet_file"]
            packet_file.write_text(
                "\n".join(
                    [
                        "scenario_id=CS-CH-H03",
                        f"packet_file={item['packet_file']}",
                        "redaction_status=redacted",
                        raw_secret_markers[0],
                        raw_secret_markers[1],
                        raw_secret_markers[2],
                        "operator_evidence=redacted",
                    ]
                )
                + "\n"
            )

        validation_output = self.state_dir / "reports" / "h03-packet-validation-ready.json"
        ready = run_json(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H03",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(validation_output),
            "--json",
        )
        ready_report = ready["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(ready["status"], "success")
        self.assertEqual(ready_report["status"], "packet_structurally_complete")
        self.assertEqual(ready_report["hashed_packet_file_count"], len(expected_names))
        self.assertFalse(ready_report["acceptance_sufficient"])
        self.assertFalse(ready_report["dependency_unlock_allowed_by_packet_validator"])

        draft_output = self.state_dir / "reports" / "h03-packet-record-draft.json"
        record_output = self.state_dir / "reports" / "h03-reviewer-record-draft.json"
        draft = run_json(
            "connector",
            "human-gate",
            "evidence-packet-record-draft",
            "--scenario",
            "CS-CH-H03",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(draft_output),
            "--record-output",
            str(record_output),
            "--json",
        )
        draft_report = draft["connector_human_gate_evidence_packet_record_draft_report"]
        self.assertEqual(draft["status"], "success")
        self.assertEqual(draft_report["status"], "draft_record_requires_human_completion")
        self.assertTrue(draft_report["draft_record_available"])
        self.assertFalse(draft_report["dependency_unlock_allowed_by_record_draft"])
        self.assertTrue(record_output.exists())
        draft_record = json.loads(record_output.read_text())
        self.assertEqual(draft_record["scenario_id"], "CS-CH-H03")
        self.assertEqual(draft_record["decision"], "")
        self.assertEqual(draft_record["reviewer"], "")
        self.assertEqual(draft_record["review_timestamp"], "")
        self.assertEqual(draft_record["accept_or_reject_note"], "")
        self.assertEqual(draft_record["issues_or_exceptions"], "")
        self.assertTrue(draft_record["browser_profile_redacted"].startswith("chrome_profile:"))
        self.assertTrue(draft_record["extension_version"].startswith("chrome_extension:"))
        self.assertTrue(
            draft_record["permission_pages_or_recording_ref"].startswith("chrome_permission:")
        )
        self.assertTrue(draft_record["active_tab_capture_ref"].startswith("chrome_active_tab_capture:"))
        self.assertTrue(
            draft_record["allowlist_auto_capture_ref"].startswith("chrome_allowlist_capture:")
        )
        self.assertTrue(draft_record["sensitive_block_ref"].startswith("chrome_sensitive_policy:"))
        self.assertTrue(draft_record["pause_revoke_ref"].startswith("chrome_pause_revoke:"))
        self.assertTrue(all(ref.startswith("audit:") for ref in draft_record["audit_refs"]))
        self.assertEqual(draft_record["dependency_human_gate_refs"], {"CS-CH-H02": ""})
        self.assertEqual(len(draft_record["evidence_packet_manifest"]), 4)
        self.assertTrue(
            all(item["redaction_status"] == "redacted" for item in draft_record["evidence_packet_manifest"])
        )

        validation = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H03",
            "--record-file",
            str(record_output),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(validation.returncode, 1, validation.stdout + validation.stderr)
        validation_payload = json.loads(validation.stdout)
        validation_report = validation_payload["connector_human_gate_record_validation"]
        self.assertEqual(validation_report["status"], "record_structurally_invalid")
        self.assertFalse(validation_report["dependency_unlock_allowed_by_validator"])
        self.assertIn("decision_not_allowed", validation_report["structural_errors"])
        self.assertIn("empty_required_fields", validation_report["structural_errors"])
        self.assertIn("empty_senior_review_perspectives", validation_report["structural_errors"])
        self.assertIn("missing_dependency_human_gate_refs", validation_report["structural_errors"])

        state_text = state_file_texts(self.state_dir)
        output_text = json.dumps(draft, sort_keys=True) + draft_output.read_text() + record_output.read_text()
        for marker in raw_secret_markers:
            self.assertNotIn(marker, json.dumps(ready, sort_keys=True))
            self.assertNotIn(marker, validation_output.read_text())
            self.assertNotIn(marker, output_text)
            self.assertNotIn(marker, state_text)

    def test_connector_human_gate_h05_live_action_packet_workflow_prepares_hash_only_record_draft(self) -> None:
        contract = run_json(
            "connector",
            "human-gate",
            "evidence-packet-contract",
            "--scenario",
            "CS-CH-H05",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(contract["status"], "success")
        self.assertEqual(contract["final_verdict"], "HUMAN_REQUIRED")
        contract_report = contract["connector_human_gate_evidence_packet_contract_report"]
        self.assertEqual(contract_report["scenario_id"], "CS-CH-H05")
        self.assertEqual(contract_report["required_evidence_packet_manifest_count"], 4)
        self.assertEqual(
            contract["summary"]["product_feature_claims"],
            "CONNECTOR_HUB_H05_EVIDENCE_PACKET_CONTRACT_PREPARED_HUMAN_EVIDENCE_REQUIRED",
        )

        file_contract = run_json(
            "connector",
            "human-gate",
            "evidence-packet-file-contract",
            "--scenario",
            "CS-CH-H05",
            "--state-dir",
            self.state_rel,
        )
        file_report = file_contract["connector_human_gate_evidence_packet_file_contract_report"]
        expected_names = [item["packet_file"] for item in CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_ITEMS]
        self.assertEqual(file_contract["status"], "success")
        self.assertEqual(file_report["required_packet_file_names"], expected_names)
        self.assertEqual(
            file_report["packet_file_scaffold_directory"],
            CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
        )
        self.assertEqual(
            file_report["packet_file_scaffold_command_count"],
            CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
        )

        package = run_json(
            "connector",
            "human-gate",
            "package",
            "--scenario",
            "CS-CH-H05",
            "--state-dir",
            self.state_rel,
        )["connector_human_gate_package"]
        self.assert_h05_evidence_packet_workflow(
            package["remaining_human_evidence_summary"]["evidence_packet_workflow"]
        )
        self.assert_h05_evidence_packet_workflow(
            package["proposed_record_template"]["reviewer_checklist"]["evidence_packet_workflow"]
        )
        self.assert_h05_evidence_packet_workflow(
            package["scenario_delivery_unit_plan"]["evidence_packet_workflow"]
        )

        packet_dir = self.record_dir / "h05-live-action-packet"
        scaffold = run_json(
            "connector",
            "human-gate",
            "evidence-packet-scaffold",
            "--scenario",
            "CS-CH-H05",
            "--packet-dir",
            str(packet_dir),
            "--write",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        scaffold_report = scaffold["connector_human_gate_evidence_packet_scaffold_report"]
        self.assertEqual(scaffold["status"], "success")
        self.assertEqual(scaffold_report["scaffold_template_count"], len(expected_names))
        self.assertEqual(scaffold_report["written_packet_file_count"], len(expected_names))
        self.assertFalse(scaffold_report["acceptance_sufficient"])
        for packet_name in expected_names:
            self.assertTrue((packet_dir / packet_name).exists())
        live_scope = (packet_dir / "live-action-scope.md").read_text()
        self.assertIn("Scenario: CS-CH-H05", live_scope)
        self.assertIn("Do not mark H05 PASS", live_scope)

        blank = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H05",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(blank.returncode, 1, blank.stdout + blank.stderr)
        blank_payload = json.loads(blank.stdout)
        blank_report = blank_payload["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(blank_payload["status"], "blocked")
        self.assertEqual(blank_report["template_only_packet_file_count"], len(expected_names))
        self.assertFalse(blank_report["dependency_unlock_allowed_by_packet_validator"])

        raw_secret_markers = [
            "SHOULD_NOT_APPEAR_IN_H05_PACKET_OUTPUT",
            "provider-secret-token",
            "-----BEGIN PRIVATE KEY-----",
        ]
        for item in CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_ITEMS:
            packet_file = packet_dir / item["packet_file"]
            packet_file.write_text(
                "\n".join(
                    [
                        "scenario_id=CS-CH-H05",
                        f"packet_file={item['packet_file']}",
                        "redaction_status=redacted",
                        raw_secret_markers[0],
                        raw_secret_markers[1],
                        raw_secret_markers[2],
                        "operator_evidence=redacted",
                    ]
                )
                + "\n"
            )

        validation_output = self.state_dir / "reports" / "h05-packet-validation-ready.json"
        ready = run_json(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H05",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(validation_output),
            "--json",
        )
        ready_report = ready["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(ready["status"], "success")
        self.assertEqual(ready_report["status"], "packet_structurally_complete")
        self.assertEqual(ready_report["hashed_packet_file_count"], len(expected_names))
        self.assertFalse(ready_report["acceptance_sufficient"])
        self.assertFalse(ready_report["dependency_unlock_allowed_by_packet_validator"])

        draft_output = self.state_dir / "reports" / "h05-packet-record-draft.json"
        record_output = self.state_dir / "reports" / "h05-reviewer-record-draft.json"
        draft = run_json(
            "connector",
            "human-gate",
            "evidence-packet-record-draft",
            "--scenario",
            "CS-CH-H05",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(draft_output),
            "--record-output",
            str(record_output),
            "--json",
        )
        draft_report = draft["connector_human_gate_evidence_packet_record_draft_report"]
        self.assertEqual(draft["status"], "success")
        self.assertEqual(draft_report["status"], "draft_record_requires_human_completion")
        self.assertTrue(draft_report["draft_record_available"])
        self.assertFalse(draft_report["dependency_unlock_allowed_by_record_draft"])
        self.assertTrue(record_output.exists())
        draft_record = json.loads(record_output.read_text())
        self.assertEqual(draft_record["scenario_id"], "CS-CH-H05")
        self.assertEqual(draft_record["decision"], "")
        self.assertEqual(draft_record["reviewer"], "")
        self.assertEqual(draft_record["review_timestamp"], "")
        self.assertEqual(draft_record["issues_or_exceptions"], "")
        self.assertTrue(draft_record["approved_provider"].startswith("non_github_provider:"))
        self.assertTrue(draft_record["reversible_test_target"].startswith("reversible_target:"))
        self.assertTrue(
            draft_record["rollback_or_compensation_plan"].startswith("rollback_compensation:")
        )
        self.assertTrue(draft_record["approval_ref"].startswith("approval:"))
        self.assertTrue(draft_record["redacted_request_result"].startswith("redacted_action_result:"))
        self.assertTrue(draft_record["provider_receipt"].startswith("provider_receipt:"))
        self.assertTrue(draft_record["idempotency_evidence"].startswith("idempotency:"))
        self.assertTrue(all(ref.startswith("audit:") for ref in draft_record["audit_refs"]))
        self.assertEqual(
            draft_record["dependency_human_gate_refs"],
            {"CS-CH-H04": "", "CS-CH-H07": ""},
        )
        self.assertEqual(len(draft_record["evidence_packet_manifest"]), 4)
        self.assertTrue(
            all(item["redaction_status"] == "redacted" for item in draft_record["evidence_packet_manifest"])
        )

        validation = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H05",
            "--record-file",
            str(record_output),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(validation.returncode, 1, validation.stdout + validation.stderr)
        validation_payload = json.loads(validation.stdout)
        validation_report = validation_payload["connector_human_gate_record_validation"]
        self.assertEqual(validation_report["status"], "record_structurally_invalid")
        self.assertFalse(validation_report["dependency_unlock_allowed_by_validator"])
        self.assertIn("decision_not_allowed", validation_report["structural_errors"])
        self.assertIn("empty_required_fields", validation_report["structural_errors"])
        self.assertIn("empty_senior_review_perspectives", validation_report["structural_errors"])
        self.assertIn("missing_dependency_human_gate_refs", validation_report["structural_errors"])

        state_text = state_file_texts(self.state_dir)
        output_text = json.dumps(draft, sort_keys=True) + draft_output.read_text() + record_output.read_text()
        for marker in raw_secret_markers:
            self.assertNotIn(marker, json.dumps(ready, sort_keys=True))
            self.assertNotIn(marker, validation_output.read_text())
            self.assertNotIn(marker, output_text)
            self.assertNotIn(marker, state_text)

    def test_connector_human_gate_h06_usability_trust_packet_workflow_prepares_hash_only_record_draft(self) -> None:
        contract = run_json(
            "connector",
            "human-gate",
            "evidence-packet-contract",
            "--scenario",
            "CS-CH-H06",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(contract["status"], "success")
        self.assertEqual(contract["final_verdict"], "HUMAN_REQUIRED")
        contract_report = contract["connector_human_gate_evidence_packet_contract_report"]
        self.assertEqual(contract_report["scenario_id"], "CS-CH-H06")
        self.assertEqual(contract_report["required_evidence_packet_manifest_count"], 4)
        self.assertEqual(
            contract["summary"]["product_feature_claims"],
            "CONNECTOR_HUB_H06_EVIDENCE_PACKET_CONTRACT_PREPARED_HUMAN_EVIDENCE_REQUIRED",
        )

        file_contract = run_json(
            "connector",
            "human-gate",
            "evidence-packet-file-contract",
            "--scenario",
            "CS-CH-H06",
            "--state-dir",
            self.state_rel,
        )
        file_report = file_contract["connector_human_gate_evidence_packet_file_contract_report"]
        expected_names = [item["packet_file"] for item in CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_ITEMS]
        self.assertEqual(file_contract["status"], "success")
        self.assertEqual(file_report["required_packet_file_names"], expected_names)
        self.assertEqual(
            file_report["packet_file_scaffold_directory"],
            CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
        )
        self.assertEqual(
            file_report["packet_file_scaffold_command_count"],
            CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
        )

        package = run_json(
            "connector",
            "human-gate",
            "package",
            "--scenario",
            "CS-CH-H06",
            "--state-dir",
            self.state_rel,
        )["connector_human_gate_package"]
        self.assert_h06_evidence_packet_workflow(
            package["remaining_human_evidence_summary"]["evidence_packet_workflow"]
        )
        self.assert_h06_evidence_packet_workflow(
            package["proposed_record_template"]["reviewer_checklist"]["evidence_packet_workflow"]
        )
        self.assert_h06_evidence_packet_workflow(
            package["scenario_delivery_unit_plan"]["evidence_packet_workflow"]
        )

        packet_dir = self.record_dir / "h06-usability-trust-packet"
        scaffold = run_json(
            "connector",
            "human-gate",
            "evidence-packet-scaffold",
            "--scenario",
            "CS-CH-H06",
            "--packet-dir",
            str(packet_dir),
            "--write",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        scaffold_report = scaffold["connector_human_gate_evidence_packet_scaffold_report"]
        self.assertEqual(scaffold["status"], "success")
        self.assertEqual(scaffold_report["scaffold_template_count"], len(expected_names))
        self.assertEqual(scaffold_report["written_packet_file_count"], len(expected_names))
        self.assertFalse(scaffold_report["acceptance_sufficient"])
        for packet_name in expected_names:
            self.assertTrue((packet_dir / packet_name).exists())
        study_scope = (packet_dir / "study-scope.md").read_text()
        self.assertIn("Scenario: CS-CH-H06", study_scope)
        self.assertIn("Do not mark H06 PASS", study_scope)

        blank = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H06",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(blank.returncode, 1, blank.stdout + blank.stderr)
        blank_payload = json.loads(blank.stdout)
        blank_report = blank_payload["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(blank_payload["status"], "blocked")
        self.assertEqual(blank_report["template_only_packet_file_count"], len(expected_names))
        self.assertFalse(blank_report["dependency_unlock_allowed_by_packet_validator"])

        raw_secret_markers = [
            "SHOULD_NOT_APPEAR_IN_H06_PACKET_OUTPUT",
            "participant-private-note",
            "provider-secret-token",
        ]
        for item in CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_ITEMS:
            packet_file = packet_dir / item["packet_file"]
            packet_file.write_text(
                "\n".join(
                    [
                        "scenario_id=CS-CH-H06",
                        f"packet_file={item['packet_file']}",
                        "redaction_status=redacted",
                        raw_secret_markers[0],
                        raw_secret_markers[1],
                        raw_secret_markers[2],
                        "operator_evidence=redacted",
                    ]
                )
                + "\n"
            )

        validation_output = self.state_dir / "reports" / "h06-packet-validation-ready.json"
        ready = run_json(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H06",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(validation_output),
            "--json",
        )
        ready_report = ready["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(ready["status"], "success")
        self.assertEqual(ready_report["status"], "packet_structurally_complete")
        self.assertEqual(ready_report["hashed_packet_file_count"], len(expected_names))
        self.assertFalse(ready_report["acceptance_sufficient"])
        self.assertFalse(ready_report["dependency_unlock_allowed_by_packet_validator"])

        draft_output = self.state_dir / "reports" / "h06-packet-record-draft.json"
        record_output = self.state_dir / "reports" / "h06-reviewer-record-draft.json"
        draft = run_json(
            "connector",
            "human-gate",
            "evidence-packet-record-draft",
            "--scenario",
            "CS-CH-H06",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(draft_output),
            "--record-output",
            str(record_output),
            "--json",
        )
        draft_report = draft["connector_human_gate_evidence_packet_record_draft_report"]
        self.assertEqual(draft["status"], "success")
        self.assertEqual(draft_report["status"], "draft_record_requires_human_completion")
        self.assertTrue(draft_report["draft_record_available"])
        self.assertFalse(draft_report["dependency_unlock_allowed_by_record_draft"])
        self.assertTrue(record_output.exists())
        draft_record = json.loads(record_output.read_text())
        self.assertEqual(draft_record["scenario_id"], "CS-CH-H06")
        self.assertEqual(draft_record["decision"], "")
        self.assertEqual(draft_record["reviewer"], "")
        self.assertEqual(draft_record["review_timestamp"], "")
        self.assertEqual(draft_record["acceptance_decision"], "")
        self.assertEqual(draft_record["issues_or_exceptions"], "")
        self.assertTrue(draft_record["participant_profile_redacted"].startswith("participant_profile:"))
        self.assertTrue(draft_record["task_script_ref"].startswith("task_script:"))
        self.assertTrue(draft_record["fixture_workspace_ref"].startswith("fixture_workspace:"))
        self.assertTrue(draft_record["timed_task_notes"].startswith("timed_task_notes:"))
        self.assertTrue(draft_record["screenshots_or_recording_ref"].startswith("study_recording:"))
        self.assertTrue(draft_record["scoring_rubric"].startswith("scoring_rubric:"))
        self.assertEqual(
            draft_record["dependency_human_gate_refs"],
            {
                "CS-CH-H01": "",
                "CS-CH-H02": "",
                "CS-CH-H03": "",
                "CS-CH-H04": "",
                "CS-CH-H05": "",
                "CS-CH-H07": "",
            },
        )
        self.assertEqual(len(draft_record["evidence_packet_manifest"]), 4)
        self.assertTrue(
            all(item["redaction_status"] == "redacted" for item in draft_record["evidence_packet_manifest"])
        )

        validation = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H06",
            "--record-file",
            str(record_output),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(validation.returncode, 1, validation.stdout + validation.stderr)
        validation_payload = json.loads(validation.stdout)
        validation_report = validation_payload["connector_human_gate_record_validation"]
        self.assertEqual(validation_report["status"], "record_structurally_invalid")
        self.assertFalse(validation_report["dependency_unlock_allowed_by_validator"])
        self.assertIn("decision_not_allowed", validation_report["structural_errors"])
        self.assertIn("empty_required_fields", validation_report["structural_errors"])
        self.assertIn("empty_senior_review_perspectives", validation_report["structural_errors"])
        self.assertIn("missing_dependency_human_gate_refs", validation_report["structural_errors"])

        state_text = state_file_texts(self.state_dir)
        output_text = json.dumps(draft, sort_keys=True) + draft_output.read_text() + record_output.read_text()
        for marker in raw_secret_markers:
            self.assertNotIn(marker, json.dumps(ready, sort_keys=True))
            self.assertNotIn(marker, validation_output.read_text())
            self.assertNotIn(marker, output_text)
            self.assertNotIn(marker, state_text)

    def test_connector_human_gate_h07_recovery_packet_workflow_prepares_hash_only_record_draft(self) -> None:
        contract = run_json(
            "connector",
            "human-gate",
            "evidence-packet-contract",
            "--scenario",
            "CS-CH-H07",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(contract["status"], "success")
        self.assertEqual(contract["final_verdict"], "HUMAN_REQUIRED")
        contract_report = contract["connector_human_gate_evidence_packet_contract_report"]
        self.assertEqual(contract_report["scenario_id"], "CS-CH-H07")
        self.assertEqual(contract_report["required_evidence_packet_manifest_count"], 4)
        self.assertEqual(
            contract["summary"]["product_feature_claims"],
            "CONNECTOR_HUB_H07_EVIDENCE_PACKET_CONTRACT_PREPARED_HUMAN_EVIDENCE_REQUIRED",
        )

        file_contract = run_json(
            "connector",
            "human-gate",
            "evidence-packet-file-contract",
            "--scenario",
            "CS-CH-H07",
            "--state-dir",
            self.state_rel,
        )
        file_report = file_contract["connector_human_gate_evidence_packet_file_contract_report"]
        expected_names = [item["packet_file"] for item in CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_ITEMS]
        self.assertEqual(file_contract["status"], "success")
        self.assertEqual(file_report["required_packet_file_names"], expected_names)
        self.assertEqual(
            file_report["packet_file_scaffold_directory"],
            CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
        )
        self.assertEqual(
            file_report["packet_file_scaffold_command_count"],
            CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT,
        )

        packet_dir = self.record_dir / "h07-recovery-packet"
        scaffold = run_json(
            "connector",
            "human-gate",
            "evidence-packet-scaffold",
            "--scenario",
            "CS-CH-H07",
            "--packet-dir",
            str(packet_dir),
            "--write",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        scaffold_report = scaffold["connector_human_gate_evidence_packet_scaffold_report"]
        self.assertEqual(scaffold["status"], "success")
        self.assertEqual(scaffold_report["scaffold_template_count"], len(expected_names))
        self.assertEqual(scaffold_report["written_packet_file_count"], len(expected_names))
        self.assertFalse(scaffold_report["acceptance_sufficient"])
        for packet_name in expected_names:
            self.assertTrue((packet_dir / packet_name).exists())
        recovery_scope = (packet_dir / "recovery-scope.md").read_text()
        self.assertIn("Scenario: CS-CH-H07", recovery_scope)
        self.assertIn("Do not mark H07 PASS", recovery_scope)

        blank = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H07",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(blank.returncode, 1, blank.stdout + blank.stderr)
        blank_payload = json.loads(blank.stdout)
        blank_report = blank_payload["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(blank_payload["status"], "blocked")
        self.assertEqual(blank_report["template_only_packet_file_count"], len(expected_names))
        self.assertFalse(blank_report["dependency_unlock_allowed_by_packet_validator"])

        raw_secret_markers = [
            "SHOULD_NOT_APPEAR_IN_H07_PACKET_OUTPUT",
            "AKIAIOSFODNN7EXAMPLE",
        ]
        for item in CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_ITEMS:
            packet_file = packet_dir / item["packet_file"]
            packet_file.write_text(
                "\n".join(
                    [
                        "scenario_id=CS-CH-H07",
                        f"packet_file={item['packet_file']}",
                        "redaction_status=redacted",
                        raw_secret_markers[0],
                        raw_secret_markers[1],
                        "operator_evidence=redacted",
                    ]
                )
                + "\n"
            )

        validation_output = self.state_dir / "reports" / "h07-packet-validation-ready.json"
        ready = run_json(
            "connector",
            "human-gate",
            "evidence-packet-validate",
            "--scenario",
            "CS-CH-H07",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(validation_output),
            "--json",
        )
        ready_report = ready["connector_human_gate_evidence_packet_validation_report"]
        self.assertEqual(ready["status"], "success")
        self.assertEqual(ready_report["status"], "packet_structurally_complete")
        self.assertEqual(ready_report["hashed_packet_file_count"], len(expected_names))
        self.assertFalse(ready_report["acceptance_sufficient"])
        self.assertFalse(ready_report["dependency_unlock_allowed_by_packet_validator"])

        draft_output = self.state_dir / "reports" / "h07-packet-record-draft.json"
        record_output = self.state_dir / "reports" / "h07-reviewer-record-draft.json"
        draft = run_json(
            "connector",
            "human-gate",
            "evidence-packet-record-draft",
            "--scenario",
            "CS-CH-H07",
            "--packet-dir",
            str(packet_dir),
            "--state-dir",
            self.state_rel,
            "--output",
            str(draft_output),
            "--record-output",
            str(record_output),
            "--json",
        )
        draft_report = draft["connector_human_gate_evidence_packet_record_draft_report"]
        self.assertEqual(draft["status"], "success")
        self.assertEqual(draft_report["status"], "draft_record_requires_human_completion")
        self.assertTrue(draft_report["draft_record_available"])
        self.assertFalse(draft_report["dependency_unlock_allowed_by_record_draft"])
        self.assertTrue(record_output.exists())
        draft_record = json.loads(record_output.read_text())
        self.assertEqual(draft_record["scenario_id"], "CS-CH-H07")
        self.assertEqual(draft_record["decision"], "")
        self.assertEqual(draft_record["reviewer"], "")
        self.assertEqual(draft_record["review_timestamp"], "")
        self.assertEqual(draft_record["issues_or_exceptions"], "")
        self.assertTrue(draft_record["backup_manifest_ref"].startswith("backup_manifest:"))
        self.assertTrue(draft_record["restore_log_ref"].startswith("restore_log:"))
        self.assertTrue(draft_record["cursor_reconciliation_ref"].startswith("cursor_reconciliation:"))
        self.assertTrue(draft_record["replay_results_ref"].startswith("replay_results:"))
        self.assertTrue(draft_record["audit_verification_ref"].startswith("audit_integrity:"))
        self.assertTrue(draft_record["before_after_counts_hashes"].startswith("counts_hashes:"))
        self.assertEqual(draft_record["dependency_human_gate_refs"], {"CS-CH-H04": ""})
        self.assertEqual(len(draft_record["evidence_packet_manifest"]), 4)
        self.assertTrue(
            all(item["redaction_status"] == "redacted" for item in draft_record["evidence_packet_manifest"])
        )

        validation = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H07",
            "--record-file",
            str(record_output),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(validation.returncode, 1, validation.stdout + validation.stderr)
        validation_payload = json.loads(validation.stdout)
        validation_report = validation_payload["connector_human_gate_record_validation"]
        self.assertEqual(validation_report["status"], "record_structurally_invalid")
        self.assertFalse(validation_report["dependency_unlock_allowed_by_validator"])
        self.assertIn("decision_not_allowed", validation_report["structural_errors"])
        self.assertIn("empty_required_fields", validation_report["structural_errors"])
        self.assertIn("empty_senior_review_perspectives", validation_report["structural_errors"])
        self.assertIn("missing_dependency_human_gate_refs", validation_report["structural_errors"])

        state_text = state_file_texts(self.state_dir)
        output_text = json.dumps(draft, sort_keys=True) + draft_output.read_text() + record_output.read_text()
        for marker in raw_secret_markers:
            self.assertNotIn(marker, json.dumps(ready, sort_keys=True))
            self.assertNotIn(marker, validation_output.read_text())
            self.assertNotIn(marker, output_text)
            self.assertNotIn(marker, state_text)

    def test_connector_human_gate_evidence_packet_record_draft_rejects_non_human_gate_scenario(self) -> None:
        result = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-record-draft",
            "--scenario",
            "CS-CH-001",
            "--packet-dir",
            CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["command"], "cornerstone connector human-gate evidence-packet-record-draft")
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        report = payload["connector_human_gate_evidence_packet_record_draft_report"]
        self.assertEqual(
            report["schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_RECORD_DRAFT_REPORT_SCHEMA,
        )
        self.assertEqual(report["scenario_id"], "CS-CH-001")
        self.assertEqual(report["status"], "unsupported")
        self.assertEqual(
            {error["code"] for error in payload["errors"]},
            {"CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_RECORD_DRAFT_UNSUPPORTED"},
        )

    def test_connector_human_gate_evidence_packet_file_contract_rejects_non_human_gate_scenario(self) -> None:
        result = run_cli(
            "connector",
            "human-gate",
            "evidence-packet-file-contract",
            "--scenario",
            "CS-CH-001",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["command"], "cornerstone connector human-gate evidence-packet-file-contract")
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(
            payload["connector_human_gate_evidence_packet_file_contract_report"]["schema_version"],
            CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_REPORT_SCHEMA,
        )
        self.assertEqual(
            payload["connector_human_gate_evidence_packet_file_contract_report"]["scenario_id"],
            "CS-CH-001",
        )
        self.assertEqual(
            {error["code"] for error in payload["errors"]},
            {"CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_UNSUPPORTED"},
        )

    def test_connector_human_gate_preflight_bundle_exposes_h04_operator_inputs_without_acceptance(self) -> None:
        payload = run_json(
            "connector",
            "human-gate",
            "preflight-bundle",
            "--scenario",
            "CS-CH-H04",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(payload["command"], "cornerstone connector human-gate preflight-bundle")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(payload["errors"], [])
        report = payload["connector_human_gate_preflight_bundle_report"]
        self.assertEqual(report["schema_version"], CONNECTOR_HUMAN_GATE_PREFLIGHT_BUNDLE_REPORT_SCHEMA)
        self.assertEqual(report["scenario_id"], "CS-CH-H04")
        self.assertEqual(report["status"], "operator_preparation_only")
        self.assertEqual(report["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(report["weakest_applicable_scenario_result"], "HUMAN_REQUIRED")
        self.assertFalse(report["local_baseline_acceptance_sufficient"])
        self.assertFalse(report["local_baseline_product_claim_allowed"])
        self.assertFalse(report["local_baseline_pass_claim_allowed"])
        self.assert_human_gate_completion_boundary(report)
        self.assertEqual(report["required_human_delta"], [
            "Production-like topology identifier and trusted RequestContext transcript.",
            "Scenario-specific PostgreSQL/RLS and OPA transcripts from the reviewed environment.",
            "Network default-deny and governed-egress transcripts from the reviewed topology.",
            "Backup/restore evidence and audit-integrity report from the reviewed environment.",
            "Dated ACCEPT or REJECT decision with redacted evidence packet manifest.",
        ])
        self.assertEqual(
            report["recommended_preflight_commands"],
            [row["command"] for row in EXPECTED_H04_PREFLIGHT_COMMAND_PLAN],
        )
        self.assertEqual(report["recommended_preflight_command_plan"], EXPECTED_H04_PREFLIGHT_COMMAND_PLAN)
        bundle = report["preflight_bundle"]
        report_paths = [
            "reports/security/vs2-local-security-proof.json",
            "reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json",
            "reports/network/vs2-egress-proof.json",
            "reports/security/vs2-local-range.json",
            "reports/scenario/connector-contract-adapter-cs-ch-036-2026-06-23.json",
        ]
        report_rows = {row["path"]: row for row in bundle["current_report_fingerprints"]}
        self.assert_h04_local_baseline_preflight_bundle(bundle, report_paths, report_rows)
        non_mutation = report["non_mutation_evidence"]
        self.assertFalse(non_mutation["approval_collected_by_preflight_bundle"])
        self.assertEqual(non_mutation["human_decisions_recorded_by_preflight_bundle"], 0)
        self.assertEqual(non_mutation["commands_executed_by_preflight_bundle"], 0)
        self.assertEqual(non_mutation["live_provider_calls_executed_by_preflight_bundle"], 0)
        self.assertEqual(non_mutation["provider_mutations_executed_by_preflight_bundle"], 0)
        self.assertEqual(non_mutation["external_mutations_executed_by_preflight_bundle"], 0)
        self.assertFalse(non_mutation["secret_material_read_by_preflight_bundle"])
        self.assertFalse(non_mutation["record_bodies_persisted_by_preflight_bundle"])
        self.assertTrue(all(value in (0, False) for value in report["negative_evidence"].values()))
        self.assertTrue(payload["audit_refs"])
        self.assertIn(
            f"connector_human_gate_preflight_bundle:{report['preflight_bundle_report_id']}",
            payload["evidence_refs"],
        )
        self.assertEqual(
            payload["ids"]["connector_human_gate_preflight_bundle_report_id"],
            report["preflight_bundle_report_id"],
        )
        self.assertEqual(
            payload["summary"],
            {
                "scenario_id": "CS-CH-H04",
                "status": "operator_preparation_only",
                "final_verdict": "HUMAN_REQUIRED",
                "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
                "preflight_bundle_report_id": report["preflight_bundle_report_id"],
                "operator_rule": report["operator_rule"],
                "local_baseline_review_inputs_schema_version": (
                    "cs.connector_human_gate_local_baseline_review_inputs.v1"
                ),
                "preflight_bundle_schema_version": CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_BUNDLE_SCHEMA,
                "current_report_count": 5,
                "ready_report_count": 5,
                "command_plan_count": 3,
                "recommended_preflight_commands": [
                    row["command"] for row in EXPECTED_H04_PREFLIGHT_COMMAND_PLAN
                ],
                "recommended_preflight_command_count": 3,
                "recommended_preflight_command_plan_schema_version": (
                    CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA
                ),
                "recommended_preflight_command_plan_count": 3,
                "recommended_preflight_command_plan": EXPECTED_H04_PREFLIGHT_COMMAND_PLAN,
                "required_human_delta": [
                    "Production-like topology identifier and trusted RequestContext transcript.",
                    "Scenario-specific PostgreSQL/RLS and OPA transcripts from the reviewed environment.",
                    "Network default-deny and governed-egress transcripts from the reviewed topology.",
                    "Backup/restore evidence and audit-integrity report from the reviewed environment.",
                    "Dated ACCEPT or REJECT decision with redacted evidence packet manifest.",
                ],
                "required_human_delta_count": 5,
                "local_baseline_acceptance_sufficient": False,
                "local_baseline_product_claim_allowed": False,
                "local_baseline_pass_claim_allowed": False,
                "acceptance_sufficient": False,
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
                "commands_executed_by_preflight_bundle": 0,
                "live_provider_calls_executed_by_preflight_bundle": 0,
                "provider_mutations_executed_by_preflight_bundle": 0,
                "external_mutations_executed_by_preflight_bundle": 0,
                "human_acceptance_collected_by_preflight_bundle": False,
                "goal_completion_claim_blocked": True,
                "full_goal_completion_allowed": False,
                "completion_claim_boundary": CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
                "product_feature_claims": (
                    "CONNECTOR_HUB_H04_PREFLIGHT_BUNDLE_PREPARED_HUMAN_EVIDENCE_REQUIRED"
                ),
            },
        )
        report_path = (
            self.state_dir
            / "connector"
            / "human_gate_preflight_bundle_reports"
            / f"{report['preflight_bundle_report_id']}.json"
        )
        self.assertTrue(report_path.exists())
        self.assertIn("connector.human_gate_preflight_bundle.reported", state_file_texts(self.state_dir))

        output_path = self.state_dir / "reports" / "connectorhub-human-gate-preflight-bundle-h04.json"
        output_payload = run_json(
            "connector",
            "human-gate",
            "preflight-bundle",
            "--scenario",
            "CS-CH-H04",
            "--state-dir",
            self.state_rel,
            "--output",
            str(output_path),
        )
        self.assertEqual(output_payload["status"], "success")
        self.assertEqual(output_payload["output_path"], str(output_path))
        self.assertTrue(output_path.exists())
        written_payload = json.loads(output_path.read_text())
        self.assertEqual(written_payload["command"], "cornerstone connector human-gate preflight-bundle")
        self.assertEqual(written_payload["output_path"], str(output_path))
        self.assertEqual(written_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(written_payload["summary"], output_payload["summary"])

    def test_connector_human_gate_preflight_bundle_rejects_non_h04_scenario(self) -> None:
        result = run_cli(
            "connector",
            "human-gate",
            "preflight-bundle",
            "--scenario",
            "CS-CH-H01",
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["command"], "cornerstone connector human-gate preflight-bundle")
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(
            payload["connector_human_gate_preflight_bundle_report"]["schema_version"],
            CONNECTOR_HUMAN_GATE_PREFLIGHT_BUNDLE_REPORT_SCHEMA,
        )
        self.assertEqual(
            payload["connector_human_gate_preflight_bundle_report"]["scenario_id"],
            "CS-CH-H01",
        )
        self.assertEqual(
            {error["code"] for error in payload["errors"]},
            {"CS_CONNECTOR_HUMAN_GATE_PREFLIGHT_BUNDLE_UNSUPPORTED"},
        )

    def test_connector_human_gate_report_audits_preparation_without_pass_claims(self) -> None:
        payload = run_json(
            "connector",
            "human-gate",
            "report",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(payload["status"], "success")
        report = payload["connector_human_gate_readiness_report"]
        self.assertEqual(report["schema_version"], "cs.connector_human_gate_readiness_report.v1")
        self.assertEqual(report["status"], "human_review_required")
        self.assertEqual(report["scenario_count"], 7)
        self.assertEqual(report["human_required_count"], 7)
        self.assertEqual(report["package_supported_count"], 7)
        self.assertEqual(report["template_present_count"], 7)
        self.assertEqual(
            report["execution_queue_scenario_order"],
            ["CS-CH-H04", "CS-CH-H07", "CS-CH-H01", "CS-CH-H02", "CS-CH-H03", "CS-CH-H05", "CS-CH-H06"],
        )
        self.assertEqual([item["order"] for item in report["execution_queue"]], list(range(1, 8)))
        self.assertEqual(report["execution_queue"][0]["scenario_id"], "CS-CH-H04")
        self.assertEqual(report["execution_queue"][-1]["scenario_id"], "CS-CH-H06")
        validation_summary = report["record_validation_summary"]
        self.assertEqual(validation_summary["validation_count"], 0)
        self.assertEqual(validation_summary["structurally_valid_count"], 0)
        self.assertEqual(validation_summary["structurally_invalid_count"], 0)
        self.assertEqual(validation_summary["senior_review_perspective_findings_complete_count"], 0)
        self.assertEqual(validation_summary["senior_review_perspective_findings_incomplete_count"], 0)
        self.assertEqual(validation_summary["evidence_packet_manifest_complete_count"], 0)
        self.assertEqual(validation_summary["evidence_packet_manifest_incomplete_count"], 0)
        self.assertEqual(validation_summary["scenarios_with_record_validation"], [])
        self.assertEqual(validation_summary["scenarios_with_structurally_valid_record_validation"], [])
        self.assertEqual(validation_summary["dependency_unlock_allowed_count"], 0)
        self.assertEqual(validation_summary["dependency_unlock_denied_count"], 0)
        self.assertEqual(validation_summary["dependency_unlock_denied_record_validations"], [])
        self.assertEqual(validation_summary["scenarios_with_dependency_unlock_record_validation"], [])
        self.assertEqual(
            set(validation_summary["scenarios_missing_structurally_valid_record_validation"]),
            {
                "CS-CH-H01",
                "CS-CH-H02",
                "CS-CH-H03",
                "CS-CH-H04",
                "CS-CH-H05",
                "CS-CH-H06",
                "CS-CH-H07",
            },
        )
        self.assertEqual(
            set(validation_summary["scenarios_missing_dependency_unlock_record_validation"]),
            {
                "CS-CH-H01",
                "CS-CH-H02",
                "CS-CH-H03",
                "CS-CH-H04",
                "CS-CH-H05",
                "CS-CH-H06",
                "CS-CH-H07",
            },
        )
        self.assertEqual(validation_summary["product_claims_allowed_by_validations"], 0)
        self.assertEqual(validation_summary["pass_claims_allowed_by_validations"], 0)
        self.assertEqual(validation_summary["record_bodies_persisted_by_validations"], 0)
        self.assertEqual(validation_summary["depends_on_human_gate_record_validation_not_applicable_rows"], ["CS-CH-H04"])
        self.assertEqual(validation_summary["depends_on_human_gate_record_validation_ready_rows"], [])
        self.assertEqual(
            set(validation_summary["depends_on_human_gate_record_validation_missing_rows"]),
            {
                "CS-CH-H01",
                "CS-CH-H02",
                "CS-CH-H03",
                "CS-CH-H05",
                "CS-CH-H06",
                "CS-CH-H07",
            },
        )
        self.assertEqual(validation_summary["depends_on_human_gate_record_validation_not_applicable_count"], 1)
        self.assertEqual(validation_summary["depends_on_human_gate_record_validation_ready_count"], 0)
        self.assertEqual(validation_summary["depends_on_human_gate_record_validation_missing_count"], 6)
        self.assertEqual(report["template_structure_ready_count"], 7)
        self.assertEqual(report["scenario_first_runbook_ready_count"], 7)
        self.assertEqual(report["evidence_packet_ready_count"], 7)
        self.assertEqual(report["senior_review_perspectives_ready_count"], 7)
        self.assertEqual(report["reject_conditions_ready_count"], 7)
        self.assertEqual(report["no_pass_boundary_ready_count"], 7)
        self.assertEqual(report["scenario_delivery_unit_plan_ready_count"], 7)
        self.assertEqual(report["scenario_delivery_unit_plan_missing"], [])
        self.assertEqual(report["scenario_delivery_unit_plan_product_claims_allowed"], [])
        self.assertEqual(report["scenario_delivery_unit_plan_pass_claims_allowed"], [])
        self.assertEqual(report["scenario_delivery_unit_plan_approvals_collected"], [])
        self.assertEqual(report["scenario_delivery_unit_plan_dependency_unlock_allowed"], [])
        self.assertEqual(report["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(report["weakest_applicable_scenario_result"], "HUMAN_REQUIRED")
        self.assert_human_gate_completion_boundary(report)
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(
            payload["summary"],
            {
                "report_id": report["report_id"],
                "readiness_report_id": report["report_id"],
                "scenario_count": 7,
                "human_required_count": 7,
                "final_verdict": "HUMAN_REQUIRED",
                "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
                "senior_review_perspectives_ready_count": 7,
                "scenario_delivery_unit_plan_ready_count": 7,
                "scenario_delivery_unit_plan_missing_count": 0,
                "scenario_delivery_unit_plan_product_claims_allowed": 0,
                "scenario_delivery_unit_plan_pass_claims_allowed": 0,
                "scenario_delivery_unit_plan_approvals_collected": 0,
                "scenario_delivery_unit_plan_dependency_unlock_allowed": 0,
                "validation_count": 0,
                "structurally_valid_record_validation_count": 0,
                "senior_review_perspective_findings_complete_count": 0,
                "evidence_packet_manifest_complete_count": 0,
                "dependency_unlock_allowed_count": 0,
                "dependency_unlock_denied_count": 0,
                "pass_claims_allowed_by_validations": 0,
                "product_claims_allowed_by_validations": 0,
                "record_bodies_persisted_by_validations": 0,
                "goal_completion_claim_blocked": True,
                "full_goal_completion_allowed": False,
                "completion_claim_boundary": CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
                "product_feature_claims": "CONNECTOR_HUB_HUMAN_GATES_PREPARED_HUMAN_EVIDENCE_REQUIRED",
            },
        )
        self.assert_human_gate_completion_boundary(payload["summary"])
        self.assertEqual(
            report["product_feature_claims"],
            "CONNECTOR_HUB_HUMAN_GATES_PREPARED_HUMAN_EVIDENCE_REQUIRED",
        )
        negative = report["negative_evidence"]
        for key in [
            "human_rows_marked_pass_by_report",
            "package_support_missing",
            "template_files_missing",
            "template_structure_missing",
            "scenario_first_runbook_missing",
            "evidence_packet_missing",
            "senior_review_perspectives_missing",
            "reject_conditions_missing",
            "no_pass_boundary_missing",
            "scenario_delivery_unit_plan_missing",
            "scenario_delivery_unit_plan_product_claims_allowed",
            "scenario_delivery_unit_plan_pass_claims_allowed",
            "scenario_delivery_unit_plan_approvals_collected",
            "scenario_delivery_unit_plan_dependency_unlock_allowed",
            "product_claims_allowed_by_report",
            "pass_without_human_record_allowed_by_report",
            "row_status_or_contract_issues",
            "approvals_collected_by_report",
            "live_provider_calls_executed_by_report",
            "provider_mutations_executed_by_report",
            "external_mutations_executed_by_report",
            "secret_material_read_by_report",
            "product_claims_allowed_by_validations",
            "pass_claims_allowed_by_validations",
            "record_bodies_persisted_by_validations",
        ]:
            self.assertEqual(negative[key], 0)
        non_mutation = report["non_mutation_evidence"]
        self.assertFalse(non_mutation["approval_collected_by_ai"])
        self.assertEqual(non_mutation["human_decisions_recorded_by_report"], 0)
        self.assertEqual(non_mutation["live_provider_calls_executed_by_report"], 0)
        self.assertEqual(non_mutation["provider_mutations_executed_by_report"], 0)
        self.assertEqual(non_mutation["external_mutations_executed_by_report"], 0)
        self.assertFalse(non_mutation["secret_material_read_by_report"])
        self.assertEqual(non_mutation["human_record_validations_executed_by_report"], 0)
        self.assertFalse(non_mutation["record_bodies_persisted_by_report"])
        scenario_ids = {row["scenario_id"] for row in report["scenario_results"]}
        self.assertEqual(
            scenario_ids,
            {
                "CS-CH-H01",
                "CS-CH-H02",
                "CS-CH-H03",
                "CS-CH-H04",
                "CS-CH-H05",
                "CS-CH-H06",
                "CS-CH-H07",
            },
        )
        for row in report["scenario_results"]:
            self.assertEqual(row["status"], "HUMAN_REQUIRED")
            self.assertEqual(row["owner"], "Human")
            self.assertEqual(row["execution_queue_item"]["scenario_id"], row["scenario_id"])
            self.assertEqual(row["review_order"], row["execution_queue_item"]["order"])
            self.assertEqual(row["depends_on_human_gates"], row["execution_queue_item"]["depends_on"])
            self.assertEqual(
                row["source_requirement_ids"],
                CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_IDS[row["scenario_id"]],
            )
            self.assertEqual(row["source_requirement_count"], len(row["source_requirement_ids"]))
            self.assertEqual(row["source_requirement_status"], "human_external_pending")
            self.assertEqual(
                row["source_requirement_claim_boundary"],
                CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY,
            )
            self.assertTrue(row["gate_category"])
            self.assertTrue(row["stop_or_reject_when"])
            self.assertTrue(row["package_supported"])
            self.assertIn("validate-record", row["record_validation_command"])
            self.assertEqual(
                row["record_validation_output_command"],
                (
                    f"cornerstone connector human-gate validate-record --scenario {row['scenario_id']} "
                    "--record-file <json> --json --output <redacted-validation-envelope.json>"
                ),
            )
            self.assertEqual(row["record_validation_count"], 0)
            self.assertEqual(row["record_validation_status"], "not_submitted")
            self.assertFalse(row["structurally_valid_record_validation_present"])
            self.assertFalse(row["dependency_unlock_record_validation_present"])
            self.assertEqual(row["depends_on_human_gates_with_structurally_valid_record_validation"], [])
            self.assertEqual(
                row["depends_on_human_gates_missing_structurally_valid_record_validation"],
                row["depends_on_human_gates"],
            )
            self.assertEqual(row["depends_on_human_gates_with_dependency_unlock_record_validation"], [])
            self.assertEqual(
                row["depends_on_human_gates_missing_dependency_unlock_record_validation"],
                row["depends_on_human_gates"],
            )
            self.assertEqual(
                row["depends_on_human_gate_record_validation_status"],
                "not_applicable" if not row["depends_on_human_gates"] else "missing_dependency_validations",
            )
            self.assertEqual(
                row["depends_on_human_gate_record_validations_ready"],
                not row["depends_on_human_gates"],
            )
            self.assertIsNone(row["latest_record_validation"])
            self.assertTrue(row["template_present"])
            self.assertTrue(row["template_structure_ready"])
            self.assertFalse(row["product_claim_allowed"])
            self.assertFalse(row["pass_claim_allowed_without_human_record"])
            self.assertIn(row["scenario_id"], row["package_command"])
            self.assertEqual(
                row["record_template_output_command"],
                (
                    f"cornerstone connector human-gate package --scenario {row['scenario_id']} "
                    "--json --record-template-output <reviewer-record-template.json>"
                ),
            )
            self.assertTrue(row["template_path"].endswith("_HUMAN_REVIEW_TEMPLATE_2026-06-24.md"))
            self.assertGreaterEqual(row["senior_review_perspective_count"], 6)
            self.assertTrue(row["required_human_fields"])
            self.assertTrue(row["required_evidence"])
            if row["scenario_id"] == "CS-CH-H04":
                baseline = row["local_baseline_review_inputs"]
                self.assertEqual(
                    row["local_baseline_preflight_bundle"],
                    baseline["preflight_bundle"],
                )
                self.assertEqual(
                    baseline["schema_version"],
                    "cs.connector_human_gate_local_baseline_review_inputs.v1",
                )
                self.assertEqual(baseline["status"], "review_input_only")
                self.assertFalse(baseline["acceptance_sufficient"])
                self.assertFalse(baseline["product_claim_allowed"])
                self.assertFalse(baseline["pass_claim_allowed"])
                self.assertTrue(baseline["all_reports_present"])
                self.assertTrue(baseline["all_reports_json_valid"])
                self.assertEqual(baseline["missing_reports"], [])
                self.assertEqual(baseline["invalid_json_reports"], [])
                self.assertIn("not prove production-like", baseline["boundary"])
                for baseline_report in baseline["reports"]:
                    self.assertTrue(baseline_report["review_input_only"])
                    self.assertFalse(baseline_report["acceptance_sufficient"])
                    self.assertFalse(baseline_report["product_claim_allowed"])
                    self.assertFalse(baseline_report["pass_claim_allowed"])
                    self.assertEqual(
                        baseline_report["claim_boundary"],
                        CONNECTOR_HUMAN_GATE_H04_BASELINE_CLAIM_BOUNDARY,
                    )
            else:
                self.assertNotIn("local_baseline_review_inputs", row)
                self.assertNotIn("local_baseline_preflight_bundle", row)
            proposed_record_template = row["proposed_record_template"]
            self.assertEqual(
                proposed_record_template["schema_version"],
                "cs.connector_human_gate_record_template.v1",
            )
            self.assertEqual(proposed_record_template["scenario_id"], row["scenario_id"])
            self.assertEqual(proposed_record_template["record_template"]["decision"], "")
            self.assertEqual(
                len(proposed_record_template["required_evidence_packet_manifest"]),
                len(proposed_record_template["required_evidence"]),
            )
            self.assertEqual(
                [item["required_evidence"] for item in proposed_record_template["required_evidence_packet_manifest"]],
                proposed_record_template["required_evidence"],
            )
            self.assertEqual(
                proposed_record_template["allowed_redaction_statuses"],
                CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES,
            )
            self.assertEqual(
                len(proposed_record_template["record_template"]["evidence_packet_manifest"]),
                len(proposed_record_template["required_evidence"]),
            )
            self.assertEqual(
                [item["required_evidence"] for item in proposed_record_template["record_template"]["evidence_packet_manifest"]],
                proposed_record_template["required_evidence"],
            )
            self.assertFalse(proposed_record_template["product_claim_allowed_by_template"])
            self.assertFalse(proposed_record_template["pass_claim_allowed_by_template"])
            self.assertEqual(
                proposed_record_template["dependency_human_gates"],
                row["depends_on_human_gates"],
            )
            self.assertEqual(
                proposed_record_template["format_rules"]["review_timestamp"],
                "ISO-8601 timestamp with timezone, for example 2026-06-24T12:00:00Z",
            )
            self.assertEqual(
                proposed_record_template["validation_output_command"],
                (
                    f"cornerstone connector human-gate validate-record --scenario {row['scenario_id']} "
                    "--record-file <filled-json> --json --output <redacted-validation-envelope.json>"
                ),
            )
            self.assertEqual(
                proposed_record_template["record_template_output_command"],
                row["record_template_output_command"],
            )
            self.assert_human_gate_reviewer_checklist(
                proposed_record_template["reviewer_checklist"],
                row["scenario_id"],
                required_fields=row["required_human_fields"],
                required_evidence=proposed_record_template["required_evidence"],
                dependencies=row["depends_on_human_gates"],
                record_template_output_command=row["record_template_output_command"],
                validation_output_command=proposed_record_template["validation_output_command"],
            )
            self.assert_human_gate_delivery_unit_plan(
                row["scenario_delivery_unit_plan"],
                row["scenario_id"],
                dependencies=row["depends_on_human_gates"],
                required_fields=row["required_human_fields"],
                required_evidence=proposed_record_template["required_evidence"],
                package_command=row["package_command"],
                record_template_output_command=row["record_template_output_command"],
                validation_command=row["record_validation_command"].replace("--record-file <json>", "--record-file <filled-json>"),
                validation_output_command=proposed_record_template["validation_output_command"],
            )
            self.assert_human_gate_delivery_unit_plan_summary(row["scenario_delivery_unit_plan_summary"])
            template_structure = row["template_structure"]
            self.assertTrue(template_structure["template_present"])
            self.assertTrue(template_structure["structure_ready"])
            self.assertEqual(template_structure["missing_required_tokens"], [])
            self.assertTrue(template_structure["has_scenario_id"])
            self.assertTrue(template_structure["has_no_pass_boundary"])
            self.assertTrue(template_structure["has_scenario_first_runbook_or_study"])
            self.assertTrue(template_structure["has_acceptance_evidence_packet"])
            self.assertTrue(template_structure["has_senior_review_perspectives"])
            self.assertEqual(template_structure["missing_senior_review_perspectives"], [])
            self.assertTrue(template_structure["has_pending_human_result_rows"])
            self.assertTrue(template_structure["has_reject_conditions"])
        self.assertTrue(payload["audit_refs"])
        self.assertTrue(payload["evidence_refs"])
        self.assertIn(
            f"connector_human_gate_readiness_report:{report['report_id']}",
            payload["evidence_refs"],
        )
        report_path = (
            self.state_dir
            / "connector"
            / "human_gate_readiness_reports"
            / f"{report['report_id']}.json"
        )
        self.assertTrue(report_path.exists())
        state_text = state_file_texts(self.state_dir)
        self.assertIn("connector.human_gate_readiness.reported", state_text)

        output_path = self.state_dir / "reports" / "connectorhub-human-gate-readiness.json"
        output_payload = run_json(
            "connector",
            "human-gate",
            "report",
            "--state-dir",
            self.state_rel,
            "--output",
            str(output_path),
        )
        self.assertEqual(output_payload["status"], "success")
        self.assertEqual(output_payload["output_path"], str(output_path))
        self.assertTrue(output_path.exists())
        written_payload = json.loads(output_path.read_text())
        self.assertEqual(written_payload["schema_version"], "cs.cli.v0")
        self.assertEqual(written_payload["command"], "cornerstone connector human-gate report")
        self.assertEqual(written_payload["output_path"], str(output_path))
        self.assertEqual(written_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(
            written_payload["summary"]["report_id"],
            written_payload["connector_human_gate_readiness_report"]["report_id"],
        )
        self.assertEqual(
            written_payload["summary"]["readiness_report_id"],
            written_payload["connector_human_gate_readiness_report"]["report_id"],
        )
        self.assertEqual(written_payload["summary"]["human_required_count"], 7)
        self.assertEqual(written_payload["summary"]["senior_review_perspectives_ready_count"], 7)
        self.assertEqual(written_payload["summary"]["scenario_delivery_unit_plan_ready_count"], 7)
        self.assertEqual(written_payload["summary"]["scenario_delivery_unit_plan_missing_count"], 0)
        self.assertEqual(written_payload["summary"]["validation_count"], 0)
        self.assertEqual(written_payload["summary"]["structurally_valid_record_validation_count"], 0)
        self.assertEqual(written_payload["summary"]["senior_review_perspective_findings_complete_count"], 0)
        self.assertEqual(written_payload["summary"]["evidence_packet_manifest_complete_count"], 0)
        self.assertEqual(written_payload["summary"]["dependency_unlock_allowed_count"], 0)
        self.assertEqual(written_payload["summary"]["dependency_unlock_denied_count"], 0)
        written_report = written_payload["connector_human_gate_readiness_report"]
        self.assertEqual(written_report["final_verdict"], "HUMAN_REQUIRED")
        self.assert_human_gate_completion_boundary(written_report)
        self.assertEqual(written_report["scenario_count"], 7)
        self.assertEqual(written_report["scenario_delivery_unit_plan_ready_count"], 7)
        self.assertEqual(written_report["scenario_delivery_unit_plan_missing"], [])
        self.assertEqual(written_report["negative_evidence"]["human_rows_marked_pass_by_report"], 0)
        self.assertEqual(
            written_report["non_mutation_evidence"]["live_provider_calls_executed_by_report"],
            0,
        )

    def test_connector_human_gate_next_ignores_stale_readiness_cache_without_crashing(self) -> None:
        readiness_dir = self.state_dir / "connector" / "human_gate_readiness_reports"
        readiness_dir.mkdir(parents=True, exist_ok=True)
        stale_report_id = "cshuman_report_stale_schema"
        stale_report = {
            "schema_version": "cs.connector_human_gate_readiness_report.v1",
            "report_id": stale_report_id,
            "scope": {
                "tenant_id": "local-dev",
                "owner_id": "local-user",
                "namespace_id": "personal",
                "workspace_id": "default",
            },
            "created_at": "9999-01-01T00:00:00Z",
            "scenario_results": [
                {
                    "scenario_id": "CS-CH-H04",
                    "status": "HUMAN_REQUIRED",
                    "owner": "Human",
                    "review_order": 1,
                    "depends_on_human_gates": [],
                    "depends_on_human_gate_record_validations_ready": True,
                    "dependency_unlock_record_validation_present": False,
                    "template_structure_ready": True,
                    "package_supported": True,
                }
            ],
        }
        (readiness_dir / f"{stale_report_id}.json").write_text(
            json.dumps(stale_report, indent=2, sort_keys=True) + "\n"
        )

        payload = run_json(
            "connector",
            "human-gate",
            "next",
            "--state-dir",
            self.state_rel,
        )

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        next_report = payload["connector_human_gate_next"]
        self.assertNotEqual(next_report["readiness_report_id"], stale_report_id)
        self.assertEqual(next_report["next_scenario_id"], "CS-CH-H04")
        self.assertEqual(
            next_report["next_local_baseline_preflight_bundle"],
            next_report["next_local_baseline_review_inputs"]["preflight_bundle"],
        )

    def test_connector_human_gate_next_selects_first_dependency_ready_gate_without_promotion(self) -> None:
        payload = run_json(
            "connector",
            "human-gate",
            "next",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["command"], "cornerstone connector human-gate next")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        next_report = payload["connector_human_gate_next"]
        self.assertEqual(next_report["schema_version"], "cs.connector_human_gate_next.v1")
        self.assertEqual(next_report["status"], "next_ready")
        self.assertEqual(next_report["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(next_report["weakest_applicable_scenario_result"], "HUMAN_REQUIRED")
        self.assert_human_gate_completion_boundary(next_report)
        self.assertEqual(next_report["scenario_count"], 7)
        self.assertEqual(next_report["human_required_count"], 7)
        self.assertEqual(next_report["completed_human_gate_record_validation_count"], 0)
        self.assertEqual(next_report["completed_human_gate_record_validation_scenarios"], [])
        self.assertEqual(next_report["completed_dependency_unlock_record_validation_count"], 0)
        self.assertEqual(next_report["completed_dependency_unlock_record_validation_scenarios"], [])
        self.assertEqual(next_report["record_validation_count"], 0)
        self.assertEqual(next_report["structurally_valid_record_validation_count"], 0)
        self.assertEqual(next_report["dependency_unlock_allowed_count"], 0)
        self.assertEqual(next_report["dependency_unlock_denied_count"], 0)
        self.assertEqual(next_report["next_scenario_id"], "CS-CH-H04")
        self.assertEqual(next_report["next_review_order"], 1)
        self.assertEqual(next_report["next_gate_category"], "production_like_security")
        self.assertEqual(next_report["next_depends_on_human_gates"], [])
        self.assertEqual(
            next_report["next_source_requirement_ids"],
            CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_IDS["CS-CH-H04"],
        )
        self.assertEqual(next_report["next_source_requirement_count"], 2)
        self.assertEqual(
            next_report["source_requirement_human_pending_ids"],
            HUMAN_GATE_SOURCE_REQUIREMENT_HUMAN_PENDING_IDS,
        )
        self.assertEqual(next_report["source_requirement_human_pending_count"], 11)
        self.assertEqual(
            next_report["source_requirement_claim_boundary"],
            CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY,
        )
        h04_definition = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS["CS-CH-H04"]
        h04_required_record = h04_definition["required_human_record"]
        next_remaining_summary = next_report["next_remaining_human_evidence_summary"]
        next_redaction_guidance = next_report["next_record_redaction_guidance"]
        next_reviewer_checklist = next_report["next_reviewer_checklist"]
        self.assertEqual(
            next_remaining_summary["schema_version"],
            CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_SUMMARY_SCHEMA,
        )
        self.assertEqual(next_remaining_summary["scenario_id"], "CS-CH-H04")
        self.assertEqual(
            next_report["next_required_human_fields"],
            h04_required_record["required_fields"],
        )
        self.assertEqual(next_report["next_required_evidence"], h04_required_record["required_evidence"])
        self.assertEqual(next_report["next_release_impact"], h04_definition["release_impact"])
        self.assertEqual(next_remaining_summary["required_human_fields"], next_report["next_required_human_fields"])
        self.assertEqual(next_remaining_summary["required_evidence"], next_report["next_required_evidence"])
        self.assertEqual(
            next_remaining_summary["required_human_field_count"],
            len(next_report["next_required_human_fields"]),
        )
        self.assertEqual(
            next_remaining_summary["required_evidence_count"],
            len(next_report["next_required_evidence"]),
        )
        self.assertEqual(
            next_report["next_remaining_human_field_count"],
            next_remaining_summary["required_human_field_count"],
        )
        self.assertEqual(
            next_report["next_remaining_human_evidence_count"],
            next_remaining_summary["required_evidence_count"],
        )
        self.assertEqual(next_remaining_summary["release_impact"], next_report["next_release_impact"])
        self.assertEqual(
            next_remaining_summary["claim_boundary"],
            CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY,
        )
        self.assertEqual(
            next_report["next_remaining_human_evidence_claim_boundary"],
            CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY,
        )
        self.assert_h04_remaining_evidence_summary_workflow(next_remaining_summary)
        self.assertIn("package --scenario CS-CH-H04", next_report["next_package_command"])
        self.assertEqual(
            next_report["next_record_template_output_command"],
            (
                "cornerstone connector human-gate package --scenario CS-CH-H04 "
                "--json --record-template-output <reviewer-record-template.json>"
            ),
        )
        self.assertIn("validate-record --scenario CS-CH-H04", next_report["next_record_validation_command"])
        self.assertEqual(
            next_report["next_record_validation_output_command"],
            (
                "cornerstone connector human-gate validate-record --scenario CS-CH-H04 "
                "--record-file <json> --json --output <redacted-validation-envelope.json>"
            ),
        )
        self.assertEqual(next_report["next_record_validation_count"], 0)
        self.assertEqual(next_report["next_record_validation_status"], "not_submitted")
        self.assertIsNone(next_report["next_latest_record_validation_ref"])
        self.assertFalse(next_report["next_latest_record_validation_dependency_unlock_allowed"])
        self.assertIsNone(next_report["next_latest_record_validation_issue_summary"])
        self.assert_human_gate_redaction_guidance(
            next_report["next_record_redaction_guidance"],
            "CS-CH-H04",
            required_fields=next_report["next_required_human_fields"],
            required_evidence=next_report["next_required_evidence"],
            dependencies=[],
        )
        self.assert_human_gate_reviewer_checklist(
            next_report["next_reviewer_checklist"],
            "CS-CH-H04",
            required_fields=next_report["next_required_human_fields"],
            required_evidence=next_report["next_required_evidence"],
            dependencies=[],
            record_template_output_command=next_report["next_record_template_output_command"],
            validation_output_command=next_report["next_record_validation_output_command"].replace(
                "--record-file <json>",
                "--record-file <filled-json>",
            ),
        )
        self.assert_human_gate_delivery_unit_plan(
            next_report["next_scenario_delivery_unit_plan"],
            "CS-CH-H04",
            dependencies=[],
            required_fields=next_report["next_required_human_fields"],
            required_evidence=next_report["next_required_evidence"],
            package_command=next_report["next_package_command"],
            record_template_output_command=next_report["next_record_template_output_command"],
            validation_command=next_report["next_record_validation_command"].replace(
                "--record-file <json>",
                "--record-file <filled-json>",
            ),
            validation_output_command=next_report["next_record_validation_output_command"].replace(
                "--record-file <json>",
                "--record-file <filled-json>",
            ),
        )
        self.assert_human_gate_delivery_unit_plan_summary(
            next_report["next_scenario_delivery_unit_plan_summary"]
        )
        self.assertTrue(next_report["next_scenario_delivery_unit_plan_ready"])
        self.assertEqual(next_report["next_scenario_delivery_unit_plan_lifecycle_step_count"], 7)
        self.assertGreaterEqual(next_report["next_scenario_delivery_unit_plan_senior_review_perspective_count"], 6)
        self.assertFalse(next_report["next_scenario_delivery_unit_plan_product_claim_allowed"])
        self.assertFalse(next_report["next_scenario_delivery_unit_plan_pass_claim_allowed"])
        self.assertFalse(next_report["next_scenario_delivery_unit_plan_approval_collected"])
        self.assertFalse(next_report["next_scenario_delivery_unit_plan_dependency_unlock_allowed"])
        self.assertTrue(next_report["next_template_path"].endswith("CS_CH_H04_HUMAN_REVIEW_TEMPLATE_2026-06-24.md"))
        self.assertIn("reviewer", next_report["next_required_human_fields"])
        self.assertTrue(next_report["next_required_evidence"])
        self.assertTrue(next_report["next_stop_or_reject_when"])
        self.assertEqual(next_remaining_summary["stop_or_reject_when"], next_report["next_stop_or_reject_when"])
        self.assertEqual(
            next_remaining_summary["record_template_output_command"],
            next_report["next_record_template_output_command"],
        )
        self.assertEqual(
            next_remaining_summary["validate_record_output_command"],
            next_report["next_record_validation_output_command"],
        )
        self.assertEqual(
            next_report["next_required_human_delta"],
            next_report["next_local_baseline_review_inputs"]["required_human_delta"],
        )
        self.assertEqual(
            next_report["next_recommended_preflight_commands"],
            next_report["next_local_baseline_review_inputs"]["recommended_preflight_commands"],
        )
        self.assertEqual(
            next_report["next_recommended_preflight_command_plan"],
            next_report["next_local_baseline_review_inputs"]["recommended_preflight_command_plan"],
        )
        self.assertEqual(
            next_report["next_recommended_preflight_command_plan"],
            EXPECTED_H04_PREFLIGHT_COMMAND_PLAN,
        )
        self.assertEqual(next_report["next_local_baseline_review_inputs"]["scenario_id"], "CS-CH-H04")
        self.assertEqual(next_report["next_local_baseline_review_inputs"]["status"], "review_input_only")
        self.assertFalse(next_report["next_local_baseline_review_inputs"]["acceptance_sufficient"])
        self.assertFalse(next_report["next_local_baseline_review_inputs"]["product_claim_allowed"])
        self.assertFalse(next_report["next_local_baseline_review_inputs"]["pass_claim_allowed"])
        self.assertEqual(len(next_report["next_local_baseline_review_inputs"]["reports"]), 5)
        next_baseline_report_rows = {
            baseline_report["path"]: baseline_report
            for baseline_report in next_report["next_local_baseline_review_inputs"]["reports"]
        }
        for baseline_report in next_report["next_local_baseline_review_inputs"]["reports"]:
            self.assertTrue(baseline_report["review_input_only"])
            self.assertFalse(baseline_report["acceptance_sufficient"])
            self.assertFalse(baseline_report["product_claim_allowed"])
            self.assertFalse(baseline_report["pass_claim_allowed"])
            self.assertEqual(
                baseline_report["claim_boundary"],
                CONNECTOR_HUMAN_GATE_H04_BASELINE_CLAIM_BOUNDARY,
            )
        self.assert_h04_local_baseline_preflight_bundle(
            next_report["next_local_baseline_review_inputs"]["preflight_bundle"],
            [
                "reports/security/vs2-local-security-proof.json",
                "reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json",
                "reports/network/vs2-egress-proof.json",
                "reports/security/vs2-local-range.json",
                "reports/scenario/connector-contract-adapter-cs-ch-036-2026-06-23.json",
            ],
            next_baseline_report_rows,
        )
        self.assertEqual(
            next_report["next_local_baseline_preflight_bundle"],
            next_report["next_local_baseline_review_inputs"]["preflight_bundle"],
        )
        self.assertEqual(next_report["ready_scenario_ids"], ["CS-CH-H04"])
        self.assertEqual(
            set(next_report["blocked_scenario_ids"]),
            {"CS-CH-H01", "CS-CH-H02", "CS-CH-H03", "CS-CH-H05", "CS-CH-H06", "CS-CH-H07"},
        )
        self.assertEqual(
            next_report["blocked_by_missing_dependency_validations"]["CS-CH-H07"],
            ["CS-CH-H04"],
        )
        self.assertTrue(next_report["readiness_report_ref"].startswith("connector_human_gate_readiness_report:"))
        self.assertEqual(
            next_report["product_feature_claims"],
            "CONNECTOR_HUB_HUMAN_GATES_PREPARED_HUMAN_EVIDENCE_REQUIRED",
        )
        self.assertEqual(
            payload["summary"],
            {
                "next_id": next_report["next_id"],
                "readiness_report_id": next_report["readiness_report_id"],
                "scenario_count": 7,
                "human_required_count": 7,
                "final_verdict": "HUMAN_REQUIRED",
                "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
                "next_scenario_id": "CS-CH-H04",
                "next_review_order": 1,
                "next_source_requirement_ids": CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_IDS[
                    "CS-CH-H04"
                ],
                "next_source_requirement_count": 2,
                "source_requirement_human_pending_ids": HUMAN_GATE_SOURCE_REQUIREMENT_HUMAN_PENDING_IDS,
                "source_requirement_human_pending_count": 11,
                "source_requirement_claim_boundary": (
                    CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY
                ),
                "next_remaining_human_field_count": len(h04_required_record["required_fields"]),
                "next_remaining_human_evidence_count": len(h04_required_record["required_evidence"]),
                "next_remaining_human_evidence_claim_boundary": (
                    CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY
                ),
                "next_remaining_human_evidence_summary": next_report[
                    "next_remaining_human_evidence_summary"
                ],
                "next_remaining_human_evidence_schema_version": next_remaining_summary[
                    "schema_version"
                ],
                "next_required_human_fields": next_remaining_summary["required_human_fields"],
                "next_required_evidence": next_remaining_summary["required_evidence"],
                "next_validate_record_output_command": next_remaining_summary[
                    "validate_record_output_command"
                ],
                "next_evidence_packet_workflow_schema_version": next_remaining_summary[
                    "evidence_packet_workflow"
                ]["schema_version"],
                "next_evidence_packet_workflow_status": next_remaining_summary[
                    "evidence_packet_workflow"
                ]["status"],
                "next_evidence_packet_workflow_command_count": next_remaining_summary[
                    "evidence_packet_workflow_command_count"
                ],
                "next_evidence_packet_workflow_commands": next_remaining_summary[
                    "evidence_packet_workflow_commands"
                ],
                "next_evidence_packet_workflow_claim_boundary": next_remaining_summary[
                    "evidence_packet_workflow_claim_boundary"
                ],
                "next_evidence_packet_workflow_review_input_only": next_remaining_summary[
                    "evidence_packet_workflow"
                ]["review_input_only"],
                "next_evidence_packet_workflow_acceptance_sufficient": next_remaining_summary[
                    "evidence_packet_workflow"
                ]["acceptance_sufficient"],
                "next_evidence_packet_workflow_product_claim_allowed": next_remaining_summary[
                    "evidence_packet_workflow"
                ]["product_claim_allowed"],
                "next_evidence_packet_workflow_pass_claim_allowed": next_remaining_summary[
                    "evidence_packet_workflow"
                ]["pass_claim_allowed"],
                "next_dependency_unlock_allowed_by_evidence_packet_workflow": next_remaining_summary[
                    "dependency_unlock_allowed_by_evidence_packet_workflow"
                ],
                "next_human_acceptance_collected_by_evidence_packet_workflow": next_remaining_summary[
                    "human_acceptance_collected_by_evidence_packet_workflow"
                ],
                "next_raw_packet_file_contents_recorded_by_evidence_packet_workflow": (
                    next_remaining_summary[
                        "raw_packet_file_contents_recorded_by_evidence_packet_workflow"
                    ]
                ),
                "next_packet_file_contents_persisted_by_evidence_packet_workflow": (
                    next_remaining_summary[
                        "packet_file_contents_persisted_by_evidence_packet_workflow"
                    ]
                ),
                "next_record_redaction_guidance_schema_version": next_redaction_guidance[
                    "schema_version"
                ],
                "next_record_redaction_guidance_status": next_redaction_guidance["status"],
                "next_record_redaction_guidance_allowed_redaction_statuses": (
                    next_redaction_guidance["allowed_redaction_statuses"]
                ),
                "next_record_redaction_guidance_sensitive_marker_policy_schema_version": (
                    next_redaction_guidance["sensitive_marker_policy"]["schema_version"]
                ),
                "next_record_redaction_guidance_sensitive_marker_fingerprints_only": (
                    next_redaction_guidance["sensitive_marker_policy"]["fingerprints_only"]
                ),
                "next_record_redaction_guidance_raw_secret_values_allowed": (
                    next_redaction_guidance["raw_secret_values_allowed"]
                ),
                "next_record_redaction_guidance_raw_provider_payloads_allowed": (
                    next_redaction_guidance["raw_provider_payloads_allowed"]
                ),
                "next_record_redaction_guidance_raw_evidence_values_allowed": (
                    next_redaction_guidance["raw_evidence_values_allowed"]
                ),
                "next_record_redaction_guidance_raw_record_body_persisted_by_validator": (
                    next_redaction_guidance["raw_record_body_persisted_by_validator"]
                ),
                "next_record_redaction_guidance_raw_record_path_persisted_by_validator": (
                    next_redaction_guidance["raw_record_path_persisted_by_validator"]
                ),
                "next_record_redaction_guidance_required_field_count": len(
                    next_redaction_guidance["field_guidance"]
                ),
                "next_record_redaction_guidance_required_evidence_count": (
                    next_redaction_guidance["evidence_packet_manifest"]["required_evidence_count"]
                ),
                "next_record_redaction_guidance_dependency_human_gate_count": len(
                    next_redaction_guidance["dependency_human_gate_refs"]["required_gates"]
                ),
                "next_reviewer_checklist_schema_version": next_reviewer_checklist[
                    "schema_version"
                ],
                "next_reviewer_checklist_status": next_reviewer_checklist["status"],
                "next_reviewer_checklist_required_field_item_count": len(
                    next_reviewer_checklist["required_field_items"]
                ),
                "next_reviewer_checklist_senior_review_perspective_count": len(
                    next_reviewer_checklist["senior_review_perspective_items"]
                ),
                "next_reviewer_checklist_evidence_packet_manifest_item_count": len(
                    next_reviewer_checklist["evidence_packet_manifest_items"]
                ),
                "next_reviewer_checklist_dependency_human_gate_ref_count": len(
                    next_reviewer_checklist["dependency_human_gate_ref_items"]
                ),
                "next_reviewer_checklist_reviewer_record_validation_required": (
                    next_reviewer_checklist["reviewer_record_validation_required"]
                ),
                "next_reviewer_checklist_record_template_output_command": (
                    next_reviewer_checklist["record_template_output_command"]
                ),
                "next_reviewer_checklist_validation_output_command": (
                    next_reviewer_checklist["validation_output_command"]
                ),
                "next_reviewer_checklist_evidence_packet_workflow_command_count": (
                    next_reviewer_checklist["evidence_packet_workflow_command_count"]
                ),
                "next_reviewer_checklist_evidence_packet_workflow_claim_boundary": (
                    next_reviewer_checklist["evidence_packet_workflow_claim_boundary"]
                ),
                "next_reviewer_checklist_product_claim_allowed": False,
                "next_reviewer_checklist_pass_claim_allowed": False,
                "next_release_impact": next_report["next_release_impact"],
                "next_stop_or_reject_when": next_report["next_stop_or_reject_when"],
                "next_record_template_output_command": next_report[
                    "next_record_template_output_command"
                ],
                "next_record_validation_command": next_report[
                    "next_record_validation_command"
                ],
                "next_record_validation_output_command": next_report[
                    "next_record_validation_output_command"
                ],
                "next_required_human_delta": next_report["next_required_human_delta"],
                "next_required_human_delta_count": len(next_report["next_required_human_delta"]),
                "next_recommended_preflight_command_count": len(next_report["next_recommended_preflight_commands"]),
                "next_recommended_preflight_command_plan_count": len(
                    next_report["next_recommended_preflight_command_plan"]
                ),
                "next_local_baseline_review_input_report_count": 5,
                "next_local_baseline_acceptance_sufficient": False,
                "next_local_baseline_preflight_bundle_ready_report_count": 5,
                "next_local_baseline_preflight_bundle_acceptance_sufficient": False,
                "next_scenario_delivery_unit_plan_ready": True,
                "next_scenario_delivery_unit_plan_lifecycle_step_count": 7,
                "next_scenario_delivery_unit_plan_senior_review_perspective_count": 6,
                "next_scenario_delivery_unit_plan_product_claim_allowed": False,
                "next_scenario_delivery_unit_plan_pass_claim_allowed": False,
                "next_scenario_delivery_unit_plan_approval_collected": False,
                "next_scenario_delivery_unit_plan_dependency_unlock_allowed": False,
                "next_record_validation_count": 0,
                "next_record_validation_status": "not_submitted",
                "next_latest_record_validation_issue_summary_present": False,
                "next_latest_record_validation_dependency_unlock_allowed": False,
                "ready_scenario_count": 1,
                "blocked_scenario_count": 6,
                "completed_human_gate_record_validation_count": 0,
                "validation_count": 0,
                "structurally_valid_record_validation_count": 0,
                "dependency_unlock_allowed_count": 0,
                "dependency_unlock_denied_count": 0,
                "readiness_report_ref": next_report["readiness_report_ref"],
                "goal_completion_claim_blocked": True,
                "full_goal_completion_allowed": False,
                "completion_claim_boundary": CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
                "product_feature_claims": "CONNECTOR_HUB_HUMAN_GATES_PREPARED_HUMAN_EVIDENCE_REQUIRED",
            },
        )
        self.assert_human_gate_completion_boundary(payload["summary"])
        negative = next_report["negative_evidence"]
        for key in [
            "human_rows_marked_pass_by_next",
            "approvals_collected_by_next",
            "live_provider_calls_executed_by_next",
            "provider_mutations_executed_by_next",
            "external_mutations_executed_by_next",
            "secret_material_read_by_next",
            "record_bodies_persisted_by_next",
            "product_claims_allowed_by_next",
            "pass_claims_allowed_by_next",
        ]:
            self.assertEqual(negative[key], 0)
        non_mutation = next_report["non_mutation_evidence"]
        self.assertFalse(non_mutation["approval_collected_by_ai"])
        self.assertEqual(non_mutation["human_decisions_recorded_by_next"], 0)
        self.assertEqual(non_mutation["live_provider_calls_executed_by_next"], 0)
        self.assertEqual(non_mutation["provider_mutations_executed_by_next"], 0)
        self.assertEqual(non_mutation["external_mutations_executed_by_next"], 0)
        self.assertFalse(non_mutation["secret_material_read_by_next"])
        self.assertEqual(non_mutation["human_record_validations_executed_by_next"], 0)
        self.assertFalse(non_mutation["record_bodies_persisted_by_next"])
        self.assertTrue(payload["audit_refs"])
        self.assertTrue(payload["evidence_refs"])
        self.assertIn(f"connector_human_gate_next:{next_report['next_id']}", payload["evidence_refs"])
        self.assertIn(next_report["readiness_report_ref"], payload["evidence_refs"])
        next_path = (
            self.state_dir
            / "connector"
            / "human_gate_readiness_reports"
            / f"{next_report['next_id']}.json"
        )
        self.assertTrue(next_path.exists())
        state_text = state_file_texts(self.state_dir)
        self.assertIn("connector.human_gate_next.selected", state_text)

        output_path = self.state_dir / "reports" / "connectorhub-human-gate-next.json"
        output_payload = run_json(
            "connector",
            "human-gate",
            "next",
            "--state-dir",
            self.state_rel,
            "--output",
            str(output_path),
        )
        self.assertEqual(output_payload["status"], "success")
        self.assertEqual(output_payload["output_path"], str(output_path))
        self.assertTrue(output_path.exists())
        written_payload = json.loads(output_path.read_text())
        self.assertEqual(written_payload["schema_version"], "cs.cli.v0")
        self.assertEqual(written_payload["command"], "cornerstone connector human-gate next")
        self.assertEqual(written_payload["output_path"], str(output_path))
        self.assertEqual(written_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(
            written_payload["summary"]["next_id"],
            written_payload["connector_human_gate_next"]["next_id"],
        )
        self.assertEqual(
            written_payload["summary"]["readiness_report_id"],
            written_payload["connector_human_gate_next"]["readiness_report_id"],
        )
        self.assertEqual(written_payload["summary"]["next_scenario_id"], "CS-CH-H04")
        self.assertEqual(written_payload["connector_human_gate_next"]["next_scenario_id"], "CS-CH-H04")

    def test_connector_human_gate_next_and_handoff_reuse_pinned_readiness_report(self) -> None:
        report_payload = run_json(
            "connector",
            "human-gate",
            "report",
            "--state-dir",
            self.state_rel,
        )
        report = report_payload["connector_human_gate_readiness_report"]
        expected_readiness_ref = f"connector_human_gate_readiness_report:{report['report_id']}"

        next_payload = run_json(
            "connector",
            "human-gate",
            "next",
            "--state-dir",
            self.state_rel,
        )
        next_report = next_payload["connector_human_gate_next"]
        self.assertEqual(next_report["readiness_report_ref"], expected_readiness_ref)
        self.assertEqual(
            next_payload["ids"]["connector_human_gate_readiness_report_id"],
            report["report_id"],
        )
        self.assertIn(expected_readiness_ref, next_payload["evidence_refs"])

        handoff_payload = run_json(
            "connector",
            "human-gate",
            "validation-handoff",
            "--state-dir",
            self.state_rel,
        )
        handoff = handoff_payload["connector_human_gate_validation_handoff"]
        self.assertEqual(handoff["readiness_report_ref"], expected_readiness_ref)
        self.assertEqual(
            handoff_payload["ids"]["connector_human_gate_readiness_report_id"],
            report["report_id"],
        )
        self.assertIn(expected_readiness_ref, handoff_payload["evidence_refs"])

    def test_connector_human_gate_validation_handoff_compacts_operator_trail_without_promotion(self) -> None:
        payload = run_json(
            "connector",
            "human-gate",
            "validation-handoff",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["command"], "cornerstone connector human-gate validation-handoff")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        handoff = payload["connector_human_gate_validation_handoff"]
        self.assertEqual(handoff["schema_version"], "cs.connector_human_gate_validation_handoff.v1")
        self.assertEqual(handoff["status"], "human_validation_handoff_ready")
        self.assertEqual(handoff["final_verdict"], "HUMAN_REQUIRED")
        self.assert_human_gate_completion_boundary(handoff)
        self.assertTrue(handoff["readiness_report_ref"].startswith("connector_human_gate_readiness_report:"))
        self.assertEqual(
            handoff["source_requirement_human_pending_ids"],
            HUMAN_GATE_SOURCE_REQUIREMENT_HUMAN_PENDING_IDS,
        )
        self.assertEqual(handoff["source_requirement_human_pending_count"], 11)
        self.assertEqual(
            handoff["source_requirement_claim_boundary"],
            CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY,
        )
        self.assertEqual(handoff["remaining_human_evidence_summary_count"], 7)
        self.assertEqual(handoff["remaining_human_field_total"], HUMAN_GATE_REQUIRED_FIELD_TOTAL)
        self.assertEqual(handoff["remaining_human_evidence_total"], HUMAN_GATE_REQUIRED_EVIDENCE_TOTAL)
        self.assertEqual(
            handoff["remaining_human_evidence_claim_boundary"],
            CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY,
        )
        self.assertEqual(
            handoff["execution_queue_scenario_order"],
            ["CS-CH-H04", "CS-CH-H07", "CS-CH-H01", "CS-CH-H02", "CS-CH-H03", "CS-CH-H05", "CS-CH-H06"],
        )
        self.assertEqual(handoff["next_scenario_id"], "CS-CH-H04")
        self.assertEqual(handoff["ready_scenario_ids"], ["CS-CH-H04"])
        self.assertEqual(
            set(handoff["blocked_scenario_ids"]),
            {"CS-CH-H01", "CS-CH-H02", "CS-CH-H03", "CS-CH-H05", "CS-CH-H06", "CS-CH-H07"},
        )
        self.assertEqual(handoff["completed_dependency_unlock_record_validation_scenarios"], [])
        self.assertEqual(handoff["record_validation_count"], 0)
        self.assertEqual(handoff["structurally_valid_record_validation_count"], 0)
        self.assertEqual(handoff["dependency_unlock_allowed_count"], 0)
        self.assertEqual(handoff["dependency_unlock_denied_count"], 0)
        next_handoff_row = handoff["scenario_validation_handoff_rows"][0]
        next_preflight_bundle = next_handoff_row["local_baseline_preflight_bundle"]
        next_remaining_summary = next_handoff_row["remaining_human_evidence_summary"]
        next_evidence_packet_workflow = next_remaining_summary["evidence_packet_workflow"]
        self.assertEqual(
            payload["summary"],
            {
                "handoff_id": handoff["handoff_id"],
                "readiness_report_id": handoff["readiness_report_id"],
                "operator_rule": handoff["operator_rule"],
                "scenario_count": 7,
                "human_required_count": 7,
                "source_requirement_human_pending_count": 11,
                "source_requirement_human_pending_ids": HUMAN_GATE_SOURCE_REQUIREMENT_HUMAN_PENDING_IDS,
                "source_requirement_claim_boundary": (
                    CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY
                ),
                "remaining_human_evidence_summary_count": 7,
                "remaining_human_field_total": HUMAN_GATE_REQUIRED_FIELD_TOTAL,
                "remaining_human_evidence_total": HUMAN_GATE_REQUIRED_EVIDENCE_TOTAL,
                "remaining_human_evidence_claim_boundary": (
                    CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY
                ),
                "final_verdict": "HUMAN_REQUIRED",
                "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
                "next_scenario_id": "CS-CH-H04",
                "next_review_order": next_handoff_row["review_order"],
                "next_gate_category": next_handoff_row["gate_category"],
                "next_source_requirement_ids": next_handoff_row["source_requirement_ids"],
                "next_source_requirement_count": next_handoff_row["source_requirement_count"],
                "next_source_requirement_status": next_handoff_row["source_requirement_status"],
                "next_source_requirement_claim_boundary": next_handoff_row[
                    "source_requirement_claim_boundary"
                ],
                "next_dependency_ready": next_handoff_row["dependency_ready"],
                "next_dependency_status": next_handoff_row["dependency_status"],
                "next_depends_on_human_gates": next_handoff_row["depends_on_human_gates"],
                "next_blocked_by_missing_dependency_unlock_validations": next_handoff_row[
                    "blocked_by_missing_dependency_unlock_validations"
                ],
                "next_blocked_dependency_details": next_handoff_row["blocked_dependency_details"],
                "next_record_validation_status": next_handoff_row["record_validation_status"],
                "next_record_validation_count": next_handoff_row["record_validation_count"],
                "next_latest_record_validation_ref": next_handoff_row["latest_record_validation_ref"],
                "next_latest_record_validation_dependency_unlock_allowed": next_handoff_row[
                    "latest_record_validation_dependency_unlock_allowed"
                ],
                "next_latest_record_validation_issue_summary_present": (
                    next_handoff_row["latest_record_validation_issue_summary"] is not None
                ),
                "next_dependency_unlock_record_validation_present": next_handoff_row[
                    "dependency_unlock_record_validation_present"
                ],
                "next_structurally_valid_record_validation_present": next_handoff_row[
                    "structurally_valid_record_validation_present"
                ],
                "next_scenario_delivery_unit_plan_ready": next_handoff_row[
                    "scenario_delivery_unit_plan_ready"
                ],
                "next_senior_review_perspective_count": next_handoff_row[
                    "senior_review_perspective_count"
                ],
                "next_allowed_redaction_statuses": next_handoff_row["allowed_redaction_statuses"],
                "next_redaction_guidance_schema_version": next_handoff_row[
                    "redaction_guidance_schema_version"
                ],
                "next_redaction_guidance_status": next_handoff_row["redaction_guidance_status"],
                "next_redaction_guidance_sensitive_marker_policy_schema_version": (
                    next_handoff_row["redaction_guidance_sensitive_marker_policy_schema_version"]
                ),
                "next_redaction_guidance_sensitive_marker_fingerprints_only": (
                    next_handoff_row["redaction_guidance_sensitive_marker_fingerprints_only"]
                ),
                "next_redaction_guidance_raw_secret_values_allowed": next_handoff_row[
                    "redaction_guidance_raw_secret_values_allowed"
                ],
                "next_redaction_guidance_raw_provider_payloads_allowed": next_handoff_row[
                    "redaction_guidance_raw_provider_payloads_allowed"
                ],
                "next_redaction_guidance_raw_evidence_values_allowed": next_handoff_row[
                    "redaction_guidance_raw_evidence_values_allowed"
                ],
                "next_redaction_guidance_raw_record_body_persisted_by_validator": (
                    next_handoff_row["redaction_guidance_raw_record_body_persisted_by_validator"]
                ),
                "next_redaction_guidance_raw_record_path_persisted_by_validator": (
                    next_handoff_row["redaction_guidance_raw_record_path_persisted_by_validator"]
                ),
                "next_redaction_guidance_required_field_count": next_handoff_row[
                    "redaction_guidance_required_field_count"
                ],
                "next_redaction_guidance_required_evidence_count": next_handoff_row[
                    "redaction_guidance_required_evidence_count"
                ],
                "next_redaction_guidance_dependency_human_gate_count": next_handoff_row[
                    "redaction_guidance_dependency_human_gate_count"
                ],
                "next_reviewer_checklist_schema_version": next_handoff_row[
                    "reviewer_checklist_schema_version"
                ],
                "next_reviewer_checklist_status": next_handoff_row["reviewer_checklist_status"],
                "next_reviewer_checklist_required_field_item_count": next_handoff_row[
                    "reviewer_checklist_required_field_item_count"
                ],
                "next_reviewer_checklist_senior_review_perspective_count": next_handoff_row[
                    "reviewer_checklist_senior_review_perspective_count"
                ],
                "next_reviewer_checklist_evidence_packet_manifest_item_count": (
                    next_handoff_row["reviewer_checklist_evidence_packet_manifest_item_count"]
                ),
                "next_reviewer_checklist_dependency_human_gate_ref_count": next_handoff_row[
                    "reviewer_checklist_dependency_human_gate_ref_count"
                ],
                "next_reviewer_checklist_reviewer_record_validation_required": (
                    next_handoff_row["reviewer_checklist_reviewer_record_validation_required"]
                ),
                "next_reviewer_checklist_record_template_output_command": (
                    next_handoff_row["reviewer_checklist_record_template_output_command"]
                ),
                "next_reviewer_checklist_validation_output_command": next_handoff_row[
                    "reviewer_checklist_validation_output_command"
                ],
                "next_reviewer_checklist_evidence_packet_workflow_command_count": (
                    next_handoff_row["reviewer_checklist_evidence_packet_workflow_command_count"]
                ),
                "next_reviewer_checklist_evidence_packet_workflow_claim_boundary": (
                    next_handoff_row["reviewer_checklist_evidence_packet_workflow_claim_boundary"]
                ),
                "next_reviewer_checklist_product_claim_allowed": next_handoff_row[
                    "reviewer_checklist_product_claim_allowed"
                ],
                "next_reviewer_checklist_pass_claim_allowed": next_handoff_row[
                    "reviewer_checklist_pass_claim_allowed"
                ],
                "next_product_claim_allowed": next_handoff_row["product_claim_allowed"],
                "next_pass_claim_allowed": next_handoff_row["pass_claim_allowed"],
                "next_approval_collected": next_handoff_row["approval_collected"],
                "next_owner": next_handoff_row["owner"],
                "next_status": next_handoff_row["status"],
                "next_evidence_packet_workflow_schema_version": next_evidence_packet_workflow[
                    "schema_version"
                ],
                "next_evidence_packet_workflow_status": next_evidence_packet_workflow["status"],
                "next_evidence_packet_workflow_command_count": next_remaining_summary[
                    "evidence_packet_workflow_command_count"
                ],
                "next_evidence_packet_workflow_commands": next_remaining_summary[
                    "evidence_packet_workflow_commands"
                ],
                "next_evidence_packet_workflow_claim_boundary": next_remaining_summary[
                    "evidence_packet_workflow_claim_boundary"
                ],
                "next_evidence_packet_workflow_review_input_only": next_evidence_packet_workflow[
                    "review_input_only"
                ],
                "next_evidence_packet_workflow_acceptance_sufficient": next_evidence_packet_workflow[
                    "acceptance_sufficient"
                ],
                "next_evidence_packet_workflow_product_claim_allowed": next_evidence_packet_workflow[
                    "product_claim_allowed"
                ],
                "next_evidence_packet_workflow_pass_claim_allowed": next_evidence_packet_workflow[
                    "pass_claim_allowed"
                ],
                "next_dependency_unlock_allowed_by_evidence_packet_workflow": next_remaining_summary[
                    "dependency_unlock_allowed_by_evidence_packet_workflow"
                ],
                "next_human_acceptance_collected_by_evidence_packet_workflow": next_remaining_summary[
                    "human_acceptance_collected_by_evidence_packet_workflow"
                ],
                "next_raw_packet_file_contents_recorded_by_evidence_packet_workflow": (
                    next_remaining_summary[
                        "raw_packet_file_contents_recorded_by_evidence_packet_workflow"
                    ]
                ),
                "next_packet_file_contents_persisted_by_evidence_packet_workflow": (
                    next_remaining_summary[
                        "packet_file_contents_persisted_by_evidence_packet_workflow"
                    ]
                ),
                "next_remaining_human_evidence_schema_version": next_remaining_summary[
                    "schema_version"
                ],
                "next_remaining_human_evidence_claim_boundary": next_remaining_summary[
                    "claim_boundary"
                ],
                "next_required_human_fields": next_remaining_summary["required_human_fields"],
                "next_required_evidence": next_remaining_summary["required_evidence"],
                "next_validate_record_output_command": next_remaining_summary[
                    "validate_record_output_command"
                ],
                "next_required_human_delta": next_preflight_bundle["required_human_delta"],
                "next_required_human_delta_count": next_preflight_bundle[
                    "required_human_delta_count"
                ],
                "next_recommended_preflight_commands": next_preflight_bundle[
                    "recommended_preflight_commands"
                ],
                "next_recommended_preflight_command_count": len(
                    next_preflight_bundle["recommended_preflight_commands"]
                ),
                "next_recommended_preflight_command_plan": next_preflight_bundle[
                    "recommended_preflight_command_plan"
                ],
                "next_recommended_preflight_command_plan_count": next_preflight_bundle[
                    "recommended_preflight_command_plan_count"
                ],
                "next_recommended_preflight_command_plan_schema_version": next_preflight_bundle[
                    "recommended_preflight_command_plan_schema_version"
                ],
                "next_local_baseline_preflight_bundle_ready_report_count": next_preflight_bundle[
                    "ready_report_count"
                ],
                "next_local_baseline_preflight_bundle_acceptance_sufficient": (
                    next_preflight_bundle["acceptance_sufficient"]
                ),
                "next_local_baseline_preflight_bundle_review_input_only": next_preflight_bundle[
                    "review_input_only"
                ],
                "next_local_baseline_preflight_bundle_product_claim_allowed": (
                    next_preflight_bundle["product_claim_allowed"]
                ),
                "next_local_baseline_preflight_bundle_pass_claim_allowed": (
                    next_preflight_bundle["pass_claim_allowed"]
                ),
                "next_local_baseline_preflight_bundle_claim_boundary": next_preflight_bundle[
                    "claim_boundary"
                ],
                "next_remaining_human_field_count": next_handoff_row["required_human_field_count"],
                "next_remaining_human_evidence_count": next_handoff_row["required_evidence_count"],
                "next_remaining_human_evidence_summary": next_handoff_row[
                    "remaining_human_evidence_summary"
                ],
                "next_release_impact": next_handoff_row["release_impact"],
                "next_stop_or_reject_when": next_handoff_row["stop_or_reject_when"],
                "next_record_template_output_command": next_handoff_row["reviewer_commands"][
                    "record_template_output"
                ],
                "next_record_validation_command": next_handoff_row["reviewer_commands"][
                    "validate_record"
                ],
                "next_record_validation_output_command": next_handoff_row["reviewer_commands"][
                    "validate_record_output"
                ],
                "ready_scenario_count": 1,
                "blocked_scenario_count": 6,
                "completed_dependency_unlock_record_validation_count": 0,
                "validation_count": 0,
                "structurally_valid_record_validation_count": 0,
                "dependency_unlock_allowed_count": 0,
                "dependency_unlock_denied_count": 0,
                "human_rows_marked_pass_by_handoff": 0,
                "approvals_collected_by_handoff": 0,
                "product_claims_allowed_by_handoff": 0,
                "pass_claims_allowed_by_handoff": 0,
                "readiness_report_ref": handoff["readiness_report_ref"],
                "goal_completion_claim_blocked": True,
                "full_goal_completion_allowed": False,
                "completion_claim_boundary": CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
                "product_feature_claims": "CONNECTOR_HUB_HUMAN_GATES_PREPARED_HUMAN_EVIDENCE_REQUIRED",
            },
        )
        self.assert_human_gate_completion_boundary(payload["summary"])
        rows = handoff["scenario_validation_handoff_rows"]
        rows_by_id = {row["scenario_id"]: row for row in rows}
        self.assertEqual([row["scenario_id"] for row in rows], handoff["execution_queue_scenario_order"])
        for row in rows:
            self.assertEqual(row["status"], "HUMAN_REQUIRED")
            self.assertEqual(row["owner"], "Human")
            self.assertEqual(row["record_validation_count"], 0)
            self.assertEqual(row["record_validation_status"], "not_submitted")
            self.assertFalse(row["structurally_valid_record_validation_present"])
            self.assertFalse(row["dependency_unlock_record_validation_present"])
            self.assertIsNone(row["latest_record_validation_ref"])
            self.assertFalse(row["latest_record_validation_dependency_unlock_allowed"])
            self.assertIsNone(row["latest_record_validation_issue_summary"])
            self.assertFalse(row["product_claim_allowed"])
            self.assertFalse(row["pass_claim_allowed"])
            self.assertFalse(row["approval_collected"])
            self.assertEqual(
                row["source_requirement_ids"],
                CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_IDS[row["scenario_id"]],
            )
            self.assertEqual(row["source_requirement_count"], len(row["source_requirement_ids"]))
            self.assertEqual(row["source_requirement_status"], "human_external_pending")
            self.assertEqual(
                row["source_requirement_claim_boundary"],
                CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY,
            )
            definition = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS[row["scenario_id"]]
            required_record = definition["required_human_record"]
            remaining_evidence_summary = row["remaining_human_evidence_summary"]
            self.assertEqual(
                remaining_evidence_summary["schema_version"],
                CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_SUMMARY_SCHEMA,
            )
            self.assertEqual(remaining_evidence_summary["scenario_id"], row["scenario_id"])
            self.assertEqual(row["required_human_fields"], required_record["required_fields"])
            self.assertEqual(row["required_evidence"], required_record["required_evidence"])
            self.assertEqual(remaining_evidence_summary["required_human_fields"], row["required_human_fields"])
            self.assertEqual(remaining_evidence_summary["required_evidence"], row["required_evidence"])
            self.assertEqual(row["required_human_field_count"], len(row["required_human_fields"]))
            self.assertEqual(row["required_evidence_count"], len(row["required_evidence"]))
            self.assertEqual(
                remaining_evidence_summary["required_human_field_count"],
                row["required_human_field_count"],
            )
            self.assertEqual(
                remaining_evidence_summary["required_evidence_count"],
                row["required_evidence_count"],
            )
            self.assertEqual(row["release_impact"], definition["release_impact"])
            self.assertEqual(remaining_evidence_summary["release_impact"], row["release_impact"])
            self.assertTrue(row["stop_or_reject_when"])
            self.assertEqual(
                remaining_evidence_summary["stop_or_reject_when"],
                row["stop_or_reject_when"],
            )
            self.assertEqual(
                remaining_evidence_summary["claim_boundary"],
                CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY,
            )
            if row["scenario_id"] in {"CS-CH-H01", "CS-CH-H02", "CS-CH-H03", "CS-CH-H04", "CS-CH-H05", "CS-CH-H06", "CS-CH-H07"}:
                self.assert_h04_remaining_evidence_summary_workflow(remaining_evidence_summary)
            else:
                self.assertNotIn("evidence_packet_workflow", remaining_evidence_summary)
                self.assertNotIn("evidence_packet_workflow_commands", remaining_evidence_summary)
            self.assertEqual(
                row["reviewer_commands"]["validate_record_output"],
                (
                    f"cornerstone connector human-gate validate-record --scenario {row['scenario_id']} "
                    "--record-file <json> --json --output <redacted-validation-envelope.json>"
                ),
            )
            self.assertEqual(
                row["allowed_redaction_statuses"],
                CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES,
            )
            self.assertEqual(
                row["redaction_guidance_schema_version"],
                "cs.connector_human_gate_redaction_guidance.v1",
            )
            self.assertEqual(row["redaction_guidance_status"], "operator_guidance_only")
            self.assertEqual(
                row["redaction_guidance_sensitive_marker_policy_schema_version"],
                CONNECTOR_SENSITIVE_MARKER_POLICY_SCHEMA,
            )
            self.assertTrue(row["redaction_guidance_sensitive_marker_fingerprints_only"])
            self.assertFalse(row["redaction_guidance_raw_secret_values_allowed"])
            self.assertFalse(row["redaction_guidance_raw_provider_payloads_allowed"])
            self.assertFalse(row["redaction_guidance_raw_evidence_values_allowed"])
            self.assertFalse(row["redaction_guidance_raw_record_body_persisted_by_validator"])
            self.assertFalse(row["redaction_guidance_raw_record_path_persisted_by_validator"])
            self.assertEqual(
                row["redaction_guidance_required_field_count"],
                row["required_human_field_count"],
            )
            self.assertEqual(
                row["redaction_guidance_required_evidence_count"],
                row["required_evidence_count"],
            )
            self.assertEqual(
                row["redaction_guidance_dependency_human_gate_count"],
                len(row["depends_on_human_gates"]),
            )
            self.assertEqual(
                row["reviewer_checklist_schema_version"],
                "cs.connector_human_gate_reviewer_checklist.v1",
            )
            self.assertEqual(row["reviewer_checklist_status"], "operator_preparation_only")
            self.assertEqual(
                row["reviewer_checklist_required_field_item_count"],
                row["required_human_field_count"],
            )
            self.assertGreaterEqual(row["reviewer_checklist_senior_review_perspective_count"], 6)
            self.assertEqual(
                row["reviewer_checklist_evidence_packet_manifest_item_count"],
                row["required_evidence_count"],
            )
            self.assertEqual(
                row["reviewer_checklist_dependency_human_gate_ref_count"],
                len(row["depends_on_human_gates"]),
            )
            self.assertTrue(row["reviewer_checklist_reviewer_record_validation_required"])
            self.assertEqual(
                row["reviewer_checklist_record_template_output_command"],
                row["reviewer_commands"]["record_template_output"],
            )
            self.assertEqual(
                row["reviewer_checklist_validation_output_command"],
                row["reviewer_commands"]["validate_record_output"].replace(
                    "--record-file <json>",
                    "--record-file <filled-json>",
                ),
            )
            self.assertEqual(row["reviewer_checklist_evidence_packet_workflow_command_count"], 6)
            self.assertFalse(row["reviewer_checklist_product_claim_allowed"])
            self.assertFalse(row["reviewer_checklist_pass_claim_allowed"])
            self.assertGreaterEqual(row["senior_review_perspective_count"], 6)
            self.assertTrue(row["scenario_delivery_unit_plan_ready"])
            if row["scenario_id"] == "CS-CH-H04":
                self.assertEqual(row["dependency_status"], "not_applicable")
                self.assertTrue(row["dependency_ready"])
                self.assertEqual(row["blocked_by_missing_dependency_unlock_validations"], [])
                self.assertEqual(row["blocked_dependency_details"], [])
                baseline = row["local_baseline_review_input_summary"]
                self.assertEqual(
                    baseline["schema_version"],
                    "cs.connector_human_gate_local_baseline_review_inputs.v1",
                )
                self.assertEqual(baseline["status"], "review_input_only")
                self.assertEqual(baseline["report_count"], 5)
                self.assertEqual(
                    baseline["report_paths"],
                    [
                        "reports/security/vs2-local-security-proof.json",
                        "reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json",
                        "reports/network/vs2-egress-proof.json",
                        "reports/security/vs2-local-range.json",
                        "reports/scenario/connector-contract-adapter-cs-ch-036-2026-06-23.json",
                    ],
                )
                baseline_report_summaries = {
                    summary["path"]: summary for summary in baseline["report_summaries"]
                }
                self.assertEqual(set(baseline_report_summaries), set(baseline["report_paths"]))
                self.assertEqual(
                    baseline_report_summaries["reports/security/vs2-local-security-proof.json"]["summary"],
                    {
                        "pass": 86,
                        "fail": 0,
                        "blocking": 0,
                        "human_required": 7,
                        "not_verified": 0,
                        "not_run": 0,
                        "scenario_count": 93,
                        "product_feature_claims": "LOCAL_VS2_AI_VERIFIED_HUMAN_GATES_PENDING",
                    },
                )
                self.assertEqual(
                    baseline_report_summaries[
                        "reports/scenario/connector-contract-adapter-cs-ch-036-2026-06-23.json"
                    ]["summary"],
                    {
                        "pass": 1,
                        "fail": 0,
                        "blocking": 0,
                        "human_required": 0,
                        "not_verified": 0,
                        "scenario_count": 1,
                        "product_feature_claims": (
                            "LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING"
                        ),
                    },
                )
                for summary in baseline["report_summaries"]:
                    self.assertRegex(summary["sha256"], r"^[0-9a-f]{64}$")
                    self.assertIn("status", summary)
                    self.assertTrue(summary["review_input_only"])
                    self.assertFalse(summary["acceptance_sufficient"])
                    self.assertFalse(summary["product_claim_allowed"])
                    self.assertFalse(summary["pass_claim_allowed"])
                    self.assertEqual(
                        summary["claim_boundary"],
                        CONNECTOR_HUMAN_GATE_H04_BASELINE_CLAIM_BOUNDARY,
                    )
                self.assertEqual(baseline["required_human_delta_count"], 5)
                self.assertEqual(baseline["recommended_preflight_command_count"], 3)
                self.assertEqual(
                    baseline["recommended_preflight_command_plan_schema_version"],
                    CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA,
                )
                self.assertEqual(baseline["recommended_preflight_command_plan_count"], 3)
                self.assertEqual(
                    baseline["recommended_preflight_command_plan"],
                    EXPECTED_H04_PREFLIGHT_COMMAND_PLAN,
                )
                self.assertEqual(
                    baseline["preflight_bundle_schema_version"],
                    CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_BUNDLE_SCHEMA,
                )
                self.assertEqual(baseline["preflight_bundle_ready_report_count"], 5)
                self.assertFalse(baseline["preflight_bundle_acceptance_sufficient"])
                self.assert_h04_local_baseline_preflight_bundle(
                    baseline["preflight_bundle"],
                    baseline["report_paths"],
                    baseline_report_summaries,
                )
                self.assertEqual(
                    row["local_baseline_preflight_bundle"],
                    baseline["preflight_bundle"],
                )
                self.assertFalse(baseline["acceptance_sufficient"])
                self.assertFalse(baseline["product_claim_allowed"])
                self.assertFalse(baseline["pass_claim_allowed"])
            else:
                self.assertEqual(row["dependency_status"], "missing_dependency_validations")
                self.assertFalse(row["dependency_ready"])
                self.assertTrue(row["blocked_by_missing_dependency_unlock_validations"])
                self.assertEqual(
                    [detail["scenario_id"] for detail in row["blocked_dependency_details"]],
                    row["blocked_by_missing_dependency_unlock_validations"],
                )
                self.assertIsNone(row["local_baseline_review_input_summary"])
                self.assertIsNone(row["local_baseline_preflight_bundle"])
        h07_dependency_details = rows_by_id["CS-CH-H07"]["blocked_dependency_details"]
        self.assertEqual([detail["scenario_id"] for detail in h07_dependency_details], ["CS-CH-H04"])
        self.assertEqual(
            h07_dependency_details[0]["package_command"],
            rows_by_id["CS-CH-H04"]["reviewer_commands"]["package"],
        )
        self.assertEqual(
            h07_dependency_details[0]["record_template_output_command"],
            rows_by_id["CS-CH-H04"]["reviewer_commands"]["record_template_output"],
        )
        self.assertEqual(
            h07_dependency_details[0]["validate_record_command"],
            rows_by_id["CS-CH-H04"]["reviewer_commands"]["validate_record"],
        )
        self.assertEqual(
            h07_dependency_details[0]["validate_record_output_command"],
            rows_by_id["CS-CH-H04"]["reviewer_commands"]["validate_record_output"],
        )
        self.assertEqual(
            h07_dependency_details[0]["accepted_ref_prefix"],
            "connector_human_gate_record_validation:",
        )
        self.assertEqual(
            h07_dependency_details[0]["required_status"],
            "structurally_valid_accept_record",
        )
        self.assertEqual(
            h07_dependency_details[0]["unlock_rule"],
            "Only structurally valid ACCEPT validation refs unlock dependent H gates.",
        )
        h06_dependency_details = rows_by_id["CS-CH-H06"]["blocked_dependency_details"]
        self.assertEqual(
            [detail["scenario_id"] for detail in h06_dependency_details],
            ["CS-CH-H01", "CS-CH-H02", "CS-CH-H03", "CS-CH-H04", "CS-CH-H05", "CS-CH-H07"],
        )
        for detail in h06_dependency_details:
            dependency_row = rows_by_id[detail["scenario_id"]]
            self.assertEqual(detail["package_command"], dependency_row["reviewer_commands"]["package"])
            self.assertEqual(
                detail["record_template_output_command"],
                dependency_row["reviewer_commands"]["record_template_output"],
            )
            self.assertEqual(detail["validate_record_command"], dependency_row["reviewer_commands"]["validate_record"])
            self.assertEqual(
                detail["validate_record_output_command"],
                dependency_row["reviewer_commands"]["validate_record_output"],
            )
            self.assertEqual(detail["accepted_ref_prefix"], "connector_human_gate_record_validation:")
            self.assertEqual(detail["required_status"], "structurally_valid_accept_record")
            self.assertIn("ACCEPT", detail["unlock_rule"])
        for key, value in handoff["negative_evidence"].items():
            self.assertEqual(value, 0, key)
        non_mutation = handoff["non_mutation_evidence"]
        self.assertFalse(non_mutation["approval_collected_by_handoff"])
        self.assertEqual(non_mutation["human_decisions_recorded_by_handoff"], 0)
        self.assertEqual(non_mutation["live_provider_calls_executed_by_handoff"], 0)
        self.assertEqual(non_mutation["provider_mutations_executed_by_handoff"], 0)
        self.assertEqual(non_mutation["external_mutations_executed_by_handoff"], 0)
        self.assertFalse(non_mutation["secret_material_read_by_handoff"])
        self.assertFalse(non_mutation["record_bodies_persisted_by_handoff"])
        self.assertIn(
            f"connector_human_gate_validation_handoff:{handoff['handoff_id']}",
            payload["evidence_refs"],
        )
        self.assertIn(handoff["readiness_report_ref"], payload["evidence_refs"])
        self.assertTrue(payload["audit_refs"])
        handoff_path = (
            self.state_dir
            / "connector"
            / "human_gate_validation_handoffs"
            / f"{handoff['handoff_id']}.json"
        )
        self.assertTrue(handoff_path.exists())
        state_text = state_file_texts(self.state_dir)
        self.assertIn("connector.human_gate_validation_handoff.created", state_text)

        output_path = self.state_dir / "reports" / "connectorhub-human-gate-validation-handoff.json"
        output_payload = run_json(
            "connector",
            "human-gate",
            "validation-handoff",
            "--state-dir",
            self.state_rel,
            "--output",
            str(output_path),
        )
        self.assertEqual(output_payload["status"], "success")
        self.assertEqual(output_payload["output_path"], str(output_path))
        self.assertTrue(output_path.exists())
        written_payload = json.loads(output_path.read_text())
        self.assertEqual(written_payload["schema_version"], "cs.cli.v0")
        self.assertEqual(written_payload["command"], "cornerstone connector human-gate validation-handoff")
        self.assertEqual(written_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(
            written_payload["summary"]["handoff_id"],
            written_payload["connector_human_gate_validation_handoff"]["handoff_id"],
        )
        self.assertEqual(
            written_payload["summary"]["readiness_report_id"],
            written_payload["connector_human_gate_validation_handoff"]["readiness_report_id"],
        )
        self.assertEqual(
            written_payload["summary"]["operator_rule"],
            written_payload["connector_human_gate_validation_handoff"]["operator_rule"],
        )
        self.assertEqual(written_payload["summary"]["next_scenario_id"], "CS-CH-H04")
        self.assertEqual(
            written_payload["connector_human_gate_validation_handoff"]["next_scenario_id"],
            "CS-CH-H04",
        )

    def test_connector_human_gate_validation_handoff_summarizes_invalid_record_without_raw_payload(self) -> None:
        record_path = self.record_dir / "blank-human-record-h04-for-handoff.json"
        package_payload = run_json(
            "connector",
            "human-gate",
            "package",
            "--scenario",
            "CS-CH-H04",
            "--state-dir",
            self.state_rel,
            "--record-template-output",
            str(record_path),
        )
        self.assertEqual(package_payload["status"], "success")
        self.assertTrue(record_path.exists())

        validation_output_path = self.state_dir / "reports" / "invalid-h04-validation-for-handoff.json"
        validation_result = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H04",
            "--record-file",
            str(record_path),
            "--state-dir",
            self.state_rel,
            "--output",
            str(validation_output_path),
            "--json",
        )
        self.assertEqual(validation_result.returncode, 1, validation_result.stdout + validation_result.stderr)
        validation_payload = json.loads(validation_result.stdout)
        self.assertEqual(validation_payload["status"], "blocked")
        validation = validation_payload["connector_human_gate_record_validation"]
        self.assertEqual(validation["status"], "record_structurally_invalid")
        self.assertTrue(validation_output_path.exists())

        handoff_payload = run_json(
            "connector",
            "human-gate",
            "validation-handoff",
            "--state-dir",
            self.state_rel,
        )
        handoff = handoff_payload["connector_human_gate_validation_handoff"]
        h04 = {
            row["scenario_id"]: row
            for row in handoff["scenario_validation_handoff_rows"]
        }["CS-CH-H04"]
        self.assertEqual(h04["record_validation_count"], 1)
        self.assertEqual(h04["record_validation_status"], "record_structurally_invalid")
        self.assertFalse(h04["structurally_valid_record_validation_present"])
        self.assertFalse(h04["dependency_unlock_record_validation_present"])
        self.assertTrue(h04["latest_record_validation_ref"].startswith("connector_human_gate_record_validation:"))
        self.assertFalse(h04["latest_record_validation_dependency_unlock_allowed"])

        issue_summary = h04["latest_record_validation_issue_summary"]
        self.assertEqual(
            issue_summary["schema_version"],
            "cs.connector_human_gate_validation_issue_summary.v1",
        )
        self.assertEqual(issue_summary["status"], "record_structurally_invalid")
        self.assertEqual(issue_summary["validation_scope"], "structure_and_safety_only")
        self.assertEqual(issue_summary["matrix_status_after_validation"], "HUMAN_REQUIRED")
        self.assertTrue(issue_summary["structural_correction_required"])
        self.assertFalse(issue_summary["dependency_unlock_ready"])
        self.assertEqual(issue_summary["dependency_unlock_blocked_reason"], "structural_errors")
        self.assertEqual(issue_summary["operator_next_step"], "fix_structural_errors_and_rerun_validate_record")
        self.assertIn("decision_not_allowed", issue_summary["structural_errors"])
        self.assertIn("empty_required_fields", issue_summary["issue_categories"])
        self.assertIn("empty_senior_review_perspectives", issue_summary["issue_categories"])
        self.assertIn("empty_evidence_packet_manifest_items", issue_summary["issue_categories"])
        self.assertGreater(issue_summary["issue_counts"]["empty_required_fields"], 0)
        self.assertGreater(issue_summary["issue_counts"]["empty_senior_review_perspectives"], 0)
        self.assertGreater(issue_summary["issue_counts"]["empty_evidence_packet_manifest_items"], 0)
        self.assertIn("reviewer", issue_summary["issue_details"]["empty_required_fields"])
        self.assertFalse(issue_summary["product_claim_allowed"])
        self.assertFalse(issue_summary["pass_claim_allowed"])
        self.assertFalse(issue_summary["structural_validation_is_human_acceptance"])
        self.assertTrue(issue_summary["human_acceptance_requires_owner_promotion"])
        self.assertEqual(
            issue_summary["completion_claim_boundary"],
            CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
        )
        self.assertFalse(issue_summary["raw_record_body_included"])
        self.assertFalse(issue_summary["raw_record_path_included"])
        self.assertFalse(issue_summary["human_decision_value_included"])
        self.assertFalse(issue_summary["senior_review_finding_text_included"])
        self.assertFalse(issue_summary["evidence_packet_manifest_values_included"])
        self.assertNotIn(str(record_path), json.dumps(issue_summary, sort_keys=True))
        self.assertNotIn("redacted product-value finding", json.dumps(issue_summary, sort_keys=True))
        self.assertEqual(handoff_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(handoff_payload["summary"]["human_rows_marked_pass_by_handoff"], 0)
        self.assertEqual(handoff_payload["summary"]["approvals_collected_by_handoff"], 0)

    def test_connector_human_gate_record_template_is_not_accidental_validation(self) -> None:
        record_path = self.record_dir / "blank-human-record-h04.json"
        package_payload = run_json(
            "connector",
            "human-gate",
            "package",
            "--scenario",
            "CS-CH-H04",
            "--state-dir",
            self.state_rel,
            "--record-template-output",
            str(record_path),
        )
        self.assertTrue(package_payload["summary"]["record_template_output_written"])
        self.assertTrue(record_path.exists())
        self.assertEqual(
            json.loads(record_path.read_text()),
            package_payload["connector_human_gate_package"]["proposed_record_template"]["record_template"],
        )
        output_path = self.state_dir / "reports" / "connectorhub-human-gate-validation-blank-h04.json"
        result = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H04",
            "--record-file",
            str(record_path),
            "--state-dir",
            self.state_rel,
            "--output",
            str(output_path),
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        validation = payload["connector_human_gate_record_validation"]
        self.assertEqual(payload["summary"]["validation_id"], validation["validation_id"])
        self.assertEqual(payload["summary"]["scenario_id"], "CS-CH-H04")
        self.assertFalse(payload["summary"]["structurally_valid"])
        self.assertFalse(payload["summary"]["dependency_unlock_allowed_by_validator"])
        self.assertEqual(payload["summary"]["product_feature_claims"], "CONNECTOR_HUB_HUMAN_GATES_PREPARED_HUMAN_EVIDENCE_REQUIRED")
        self.assertEqual(validation["status"], "record_structurally_invalid")
        self.assertIn("decision_not_allowed", validation["structural_errors"])
        self.assertIn("empty_required_fields", validation["structural_errors"])
        self.assertIn("empty_senior_review_perspectives", validation["structural_errors"])
        self.assertIn("empty_evidence_packet_manifest_items", validation["structural_errors"])
        self.assertFalse(validation["pass_claim_allowed_by_validator"])
        self.assertEqual(validation["matrix_status_after_validation"], "HUMAN_REQUIRED")
        self.assertEqual(payload["output_path"], str(output_path))
        self.assertTrue(output_path.exists())
        written_payload = json.loads(output_path.read_text())
        self.assertEqual(written_payload["status"], "blocked")
        self.assertEqual(written_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(written_payload["summary"], payload["summary"])
        written_text = json.dumps(written_payload, sort_keys=True)
        self.assertNotIn(str(record_path), written_text)
        self.assertNotIn('"decision": ""', written_text)

    def test_connector_human_gate_blank_record_templates_are_blocked_for_all_human_rows(self) -> None:
        for scenario_id in [f"CS-CH-H{number:02d}" for number in range(1, 8)]:
            with self.subTest(scenario_id=scenario_id):
                record_path = self.record_dir / f"blank-human-record-{scenario_id.lower()}.json"
                package_payload = run_json(
                    "connector",
                    "human-gate",
                    "package",
                    "--scenario",
                    scenario_id,
                    "--state-dir",
                    self.state_rel,
                    "--record-template-output",
                    str(record_path),
                )
                self.assertTrue(package_payload["summary"]["record_template_output_written"])
                self.assertTrue(record_path.exists())
                self.assertEqual(
                    json.loads(record_path.read_text()),
                    package_payload["connector_human_gate_package"]["proposed_record_template"]["record_template"],
                )

                output_path = (
                    self.state_dir
                    / "reports"
                    / f"connectorhub-human-gate-validation-blank-{scenario_id.lower()}.json"
                )
                result = run_cli(
                    "connector",
                    "human-gate",
                    "validate-record",
                    "--scenario",
                    scenario_id,
                    "--record-file",
                    str(record_path),
                    "--state-dir",
                    self.state_rel,
                    "--output",
                    str(output_path),
                    "--json",
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["status"], "blocked")
                self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
                self.assertEqual(payload["summary"]["scenario_id"], scenario_id)
                self.assertFalse(payload["summary"]["structurally_valid"])
                self.assertFalse(payload["summary"]["dependency_unlock_allowed_by_validator"])
                self.assertEqual(
                    payload["summary"]["product_feature_claims"],
                    "CONNECTOR_HUB_HUMAN_GATES_PREPARED_HUMAN_EVIDENCE_REQUIRED",
                )

                validation = payload["connector_human_gate_record_validation"]
                self.assertEqual(payload["summary"]["validation_id"], validation["validation_id"])
                self.assertEqual(validation["scenario_id"], scenario_id)
                self.assertEqual(validation["status"], "record_structurally_invalid")
                self.assertEqual(validation["matrix_status_after_validation"], "HUMAN_REQUIRED")
                self.assertFalse(validation["dependency_unlock_allowed_by_validator"])
                self.assertFalse(validation["pass_claim_allowed_by_validator"])
                self.assertFalse(validation["product_claim_allowed"])
                for expected_error in [
                    "decision_not_allowed",
                    "empty_required_fields",
                    "empty_senior_review_perspectives",
                    "empty_evidence_packet_manifest_items",
                ]:
                    self.assertIn(expected_error, validation["structural_errors"])

                self.assertTrue(output_path.exists())
                written_payload = json.loads(output_path.read_text())
                self.assertEqual(written_payload["status"], "blocked")
                self.assertEqual(written_payload["final_verdict"], "HUMAN_REQUIRED")
                self.assertEqual(written_payload["summary"], payload["summary"])
                written_text = json.dumps(written_payload, sort_keys=True)
                self.assertNotIn(str(record_path), written_text)
                self.assertNotIn('"decision": ""', written_text)
                self.assertNotIn("redacted product-value finding", written_text)
                self.assertNotIn("evidence:redacted-human-gate", written_text)

    def test_connector_human_gate_validate_record_checks_structure_without_promotion(self) -> None:
        record_path = self.record_dir / "human-record-h04.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "scenario_id": "CS-CH-H04",
                    "decision": "ACCEPT",
                    "reviewer": "reviewer-redacted",
                    "review_timestamp": "2026-06-24T12:00:00Z",
                    "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                    "evidence_packet_manifest": human_gate_evidence_packet_manifest("CS-CH-H04"),
                    "environment_topology_ref": "topology:redacted-prod-like",
                    "request_context_proof": "request_context:trusted-redacted",
                    "db_policy_transcripts": ["db_policy:rls-redacted"],
                    "network_egress_transcripts": ["egress:default-deny-redacted"],
                    "backup_restore_evidence": ["backup_restore:security-redacted"],
                    "audit_integrity_report": "audit_integrity:security-redacted",
                    "evidence_manifest_ref": "evidence_manifest:security-redacted",
                    "findings_or_exceptions": "none",
                },
                sort_keys=True,
            )
            + "\n"
        )

        output_path = self.state_dir / "reports" / "connectorhub-human-gate-validation-h04.json"
        payload = run_json(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H04",
            "--record-file",
            str(record_path),
            "--state-dir",
            self.state_rel,
            "--output",
            str(output_path),
        )
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(payload["output_path"], str(output_path))
        validation = payload["connector_human_gate_record_validation"]
        self.assertEqual(validation["schema_version"], "cs.connector_human_gate_record_validation.v1")
        self.assertEqual(validation["scenario_id"], "CS-CH-H04")
        self.assertEqual(validation["status"], "record_structurally_valid")
        self.assertEqual(validation["validation_scope"], "structure_and_safety_only")
        self.assertEqual(validation["review_order"], 1)
        self.assertEqual(validation["depends_on_human_gates"], [])
        self.assertEqual(validation["missing_dependency_human_gates"], [])
        self.assertEqual(validation["invalid_dependency_human_gate_refs"], [])
        self.assertEqual(validation["valid_dependency_human_gate_refs"], [])
        self.assertTrue(validation["decision_present"])
        self.assertTrue(validation["decision_allowed"])
        self.assertFalse(validation["decision_recorded_by_validator"])
        self.assertTrue(validation["dependency_unlock_allowed_by_validator"])
        self.assertIsNone(validation["dependency_unlock_blocked_reason"])
        self.assertNotIn("decision", validation)
        self.assertTrue(validation["scenario_matches"])
        self.assertFalse(validation["record_file"]["path_recorded_by_validator"])
        self.assertIn("path_sha256", validation["record_file"])
        self.assertNotIn("path", validation["record_file"])
        self.assertEqual(validation["missing_required_fields"], [])
        self.assertEqual(validation["empty_required_fields"], [])
        self.assertEqual(validation["invalid_field_formats"], [])
        self.assertTrue(validation["field_ref_contract_present"])
        self.assert_h04_field_ref_contract(validation["field_ref_contract"])
        self.assertEqual(validation["invalid_required_field_ref_shapes"], [])
        self.assertFalse(validation["field_ref_values_recorded_by_validator"])
        self.assertEqual(
            set(validation["required_senior_review_perspectives"]),
            set(HUMAN_GATE_PERSPECTIVE_FINDINGS),
        )
        self.assertEqual(
            set(validation["provided_senior_review_perspective_roles"]),
            set(HUMAN_GATE_PERSPECTIVE_FINDINGS),
        )
        self.assertTrue(validation["senior_review_perspective_findings_present"])
        self.assertTrue(validation["senior_review_perspective_findings_complete"])
        self.assertEqual(validation["missing_senior_review_perspectives"], [])
        self.assertEqual(validation["empty_senior_review_perspectives"], [])
        self.assertEqual(validation["invalid_senior_review_perspective_roles"], [])
        self.assertFalse(validation["senior_review_perspective_findings_recorded_by_validator"])
        self.assertTrue(validation["evidence_packet_manifest_present"])
        self.assertTrue(validation["evidence_packet_manifest_complete"])
        self.assertEqual(validation["required_evidence_packet_manifest_indexes"], [1, 2, 3, 4])
        self.assertEqual(validation["provided_evidence_packet_manifest_indexes"], [1, 2, 3, 4])
        self.assertEqual(
            [item["required_evidence"] for item in validation["required_evidence_packet_manifest"]],
            validation["required_evidence"],
        )
        self.assertEqual(
            validation["allowed_redaction_statuses"],
            CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES,
        )
        self.assert_human_gate_redaction_guidance(
            validation["redaction_guidance"],
            "CS-CH-H04",
            required_fields=validation["required_fields"],
            required_evidence=validation["required_evidence"],
            dependencies=[],
        )
        self.assertEqual(validation["missing_evidence_packet_manifest_items"], [])
        self.assertEqual(validation["empty_evidence_packet_manifest_items"], [])
        self.assertEqual(validation["invalid_evidence_packet_manifest_items"], [])
        self.assertEqual(validation["duplicate_evidence_packet_manifest_items"], [])
        self.assertFalse(validation["evidence_packet_manifest_recorded_by_validator"])
        self.assertEqual(
            validation["format_rules"]["review_timestamp"],
            "ISO-8601 timestamp with timezone, for example 2026-06-24T12:00:00Z",
        )
        self.assertEqual(validation["sensitive_marker_findings"], [])
        self.assertFalse(validation["product_claim_allowed"])
        self.assertFalse(validation["pass_claim_allowed_by_validator"])
        self.assertEqual(validation["matrix_status_after_validation"], "HUMAN_REQUIRED")
        self.assertEqual(validation["weakest_applicable_scenario_result"], "HUMAN_REQUIRED")
        self.assert_human_gate_completion_boundary(validation)
        self.assertIn("structure and safety only", validation["promotion_rule"])
        self.assertEqual(
            payload["summary"],
            {
                "validation_id": validation["validation_id"],
                "scenario_id": "CS-CH-H04",
                "status": "record_structurally_valid",
                "final_verdict": "HUMAN_REQUIRED",
                "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
                "structurally_valid": True,
                "dependency_unlock_allowed_by_validator": True,
                "senior_review_perspective_findings_complete": True,
                "evidence_packet_manifest_complete": True,
                "product_claim_allowed": False,
                "pass_claim_allowed_by_validator": False,
                "record_body_persisted_by_validator": False,
                "record_path_persisted_by_validator": False,
                "live_provider_calls_executed_by_validator": 0,
                "provider_mutations_executed_by_validator": 0,
                "external_mutations_executed_by_validator": 0,
                "goal_completion_claim_blocked": True,
                "full_goal_completion_allowed": False,
                "completion_claim_boundary": CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
                "product_feature_claims": "CONNECTOR_HUB_HUMAN_GATES_PREPARED_HUMAN_EVIDENCE_REQUIRED",
            },
        )
        self.assert_human_gate_completion_boundary(payload["summary"])
        non_mutation = validation["non_mutation_evidence"]
        self.assertFalse(non_mutation["approval_collected_by_validator"])
        self.assertFalse(non_mutation["human_decision_recorded_by_validator"])
        self.assertEqual(non_mutation["live_provider_calls_executed_by_validator"], 0)
        self.assertEqual(non_mutation["provider_mutations_executed_by_validator"], 0)
        self.assertEqual(non_mutation["external_mutations_executed_by_validator"], 0)
        self.assertFalse(non_mutation["record_body_persisted_by_validator"])
        self.assertFalse(non_mutation["record_path_persisted_by_validator"])
        self.assertFalse(non_mutation["field_ref_values_persisted_by_validator"])
        self.assertFalse(non_mutation["senior_review_perspective_findings_persisted_by_validator"])
        self.assertFalse(non_mutation["evidence_packet_manifest_values_persisted_by_validator"])
        negative = validation["negative_evidence"]
        self.assertEqual(negative["human_rows_marked_pass_by_validator"], 0)
        self.assertEqual(negative["pass_without_owner_promotion_allowed_by_validator"], 0)
        self.assertEqual(negative["human_decision_value_persisted_by_validator"], 0)
        self.assertEqual(negative["record_path_persisted_by_validator"], 0)
        self.assertEqual(negative["field_ref_values_persisted_by_validator"], 0)
        self.assertEqual(negative["invalid_required_field_ref_shapes"], 0)
        self.assertEqual(negative["senior_review_perspective_findings_persisted_by_validator"], 0)
        self.assertEqual(negative["evidence_packet_manifest_values_persisted_by_validator"], 0)
        self.assertEqual(negative["missing_senior_review_perspectives"], 0)
        self.assertEqual(negative["empty_senior_review_perspectives"], 0)
        self.assertEqual(negative["invalid_senior_review_perspective_roles"], 0)
        self.assertEqual(negative["missing_evidence_packet_manifest_items"], 0)
        self.assertEqual(negative["empty_evidence_packet_manifest_items"], 0)
        self.assertEqual(negative["invalid_evidence_packet_manifest_items"], 0)
        self.assertEqual(negative["duplicate_evidence_packet_manifest_items"], 0)
        self.assertEqual(negative["sensitive_marker_findings"], 0)
        payload_text = json.dumps(payload, sort_keys=True)
        self.assertNotIn('"decision": "ACCEPT"', payload_text)
        self.assertNotIn("redacted product-value finding", payload_text)
        self.assertNotIn("evidence:redacted-human-gate", payload_text)
        for raw_field_ref in [
            "topology:redacted-prod-like",
            "request_context:trusted-redacted",
            "db_policy:rls-redacted",
            "egress:default-deny-redacted",
            "backup_restore:security-redacted",
            "audit_integrity:security-redacted",
            "evidence_manifest:security-redacted",
        ]:
            self.assertNotIn(raw_field_ref, payload_text)
        self.assertNotIn(str(record_path), payload_text)
        self.assertTrue(payload["audit_refs"])
        self.assertTrue(payload["evidence_refs"])
        self.assertIn(
            f"connector_human_gate_record_validation:{validation['validation_id']}",
            payload["evidence_refs"],
        )
        self.assertTrue(output_path.exists())
        written_payload = json.loads(output_path.read_text())
        self.assertEqual(written_payload["schema_version"], "cs.cli.v0")
        self.assertEqual(written_payload["command"], "cornerstone connector human-gate validate-record")
        self.assertEqual(written_payload["status"], "success")
        self.assertEqual(written_payload["output_path"], str(output_path))
        self.assertEqual(written_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(written_payload["summary"], payload["summary"])
        written_text = json.dumps(written_payload, sort_keys=True)
        self.assertNotIn('"decision": "ACCEPT"', written_text)
        self.assertNotIn("redacted product-value finding", written_text)
        self.assertNotIn("evidence:redacted-human-gate", written_text)
        for raw_field_ref in [
            "topology:redacted-prod-like",
            "request_context:trusted-redacted",
            "db_policy:rls-redacted",
            "egress:default-deny-redacted",
            "backup_restore:security-redacted",
            "audit_integrity:security-redacted",
            "evidence_manifest:security-redacted",
        ]:
            self.assertNotIn(raw_field_ref, written_text)
        self.assertNotIn(str(record_path), written_text)
        validation_path = (
            self.state_dir
            / "connector"
            / "human_gate_record_validations"
            / f"{validation['validation_id']}.json"
        )
        self.assertTrue(validation_path.exists())
        state_text = state_file_texts(self.state_dir)
        self.assertIn("connector.human_gate_record.validated", state_text)
        self.assertNotIn('"decision": "ACCEPT"', state_text)
        self.assertNotIn("redacted product-value finding", state_text)
        self.assertNotIn("evidence:redacted-human-gate", state_text)
        for raw_field_ref in [
            "topology:redacted-prod-like",
            "request_context:trusted-redacted",
            "db_policy:rls-redacted",
            "egress:default-deny-redacted",
            "backup_restore:security-redacted",
            "audit_integrity:security-redacted",
            "evidence_manifest:security-redacted",
        ]:
            self.assertNotIn(raw_field_ref, state_text)
        self.assertNotIn(str(record_path), state_text)

    def test_connector_human_gate_validate_record_blocks_invalid_h04_field_ref_shapes(self) -> None:
        record_path = self.record_dir / "human-record-h04-invalid-field-refs.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        malformed_refs = {
            "environment_topology_ref": "malformed-topology-ref",
            "request_context_proof": "malformed-request-context-ref",
            "db_policy_transcripts": ["malformed-db-policy-ref"],
            "network_egress_transcripts": ["malformed-egress-ref"],
            "backup_restore_evidence": ["malformed-backup-restore-ref"],
            "audit_integrity_report": "malformed-audit-integrity-ref",
            "evidence_manifest_ref": "malformed-evidence-manifest-ref",
        }
        record_path.write_text(
            json.dumps(
                {
                    "scenario_id": "CS-CH-H04",
                    "decision": "ACCEPT",
                    "reviewer": "reviewer-redacted",
                    "review_timestamp": "2026-06-24T12:00:00Z",
                    "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                    "evidence_packet_manifest": human_gate_evidence_packet_manifest("CS-CH-H04"),
                    **malformed_refs,
                    "findings_or_exceptions": "none",
                },
                sort_keys=True,
            )
            + "\n"
        )

        result = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H04",
            "--record-file",
            str(record_path),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["final_verdict"], "HUMAN_REQUIRED")
        validation = payload["connector_human_gate_record_validation"]
        self.assertEqual(validation["status"], "record_structurally_invalid")
        self.assertTrue(validation["field_ref_contract_present"])
        self.assert_h04_field_ref_contract(validation["field_ref_contract"])
        self.assertEqual(validation["missing_required_fields"], [])
        self.assertEqual(validation["empty_required_fields"], [])
        self.assertEqual(validation["invalid_field_formats"], [])
        self.assertEqual(
            validation["invalid_required_field_ref_shapes"],
            sorted(malformed_refs),
        )
        self.assertEqual(validation["structural_errors"], ["invalid_required_field_ref_shapes"])
        self.assertEqual(validation["negative_evidence"]["invalid_required_field_ref_shapes"], 7)
        self.assertEqual(validation["negative_evidence"]["field_ref_values_persisted_by_validator"], 0)
        self.assertFalse(validation["field_ref_values_recorded_by_validator"])
        self.assertFalse(validation["dependency_unlock_allowed_by_validator"])
        self.assertEqual(validation["dependency_unlock_blocked_reason"], "structural_errors")
        self.assertFalse(validation["product_claim_allowed"])
        self.assertFalse(validation["pass_claim_allowed_by_validator"])
        self.assertEqual(validation["matrix_status_after_validation"], "HUMAN_REQUIRED")
        self.assertEqual(validation["weakest_applicable_scenario_result"], "HUMAN_REQUIRED")
        self.assertTrue(validation["evidence_packet_manifest_complete"])
        payload_text = json.dumps(payload, sort_keys=True)
        self.assertNotIn(str(record_path), payload_text)
        for raw_value in malformed_refs.values():
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value in values:
                self.assertNotIn(value, payload_text)
        state_text = state_file_texts(self.state_dir)
        self.assertIn("connector.human_gate_record.validated", state_text)
        self.assertNotIn(str(record_path), state_text)
        for raw_value in malformed_refs.values():
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value in values:
                self.assertNotIn(value, state_text)

    def test_connector_human_gate_validate_record_blocks_undated_review_timestamp(self) -> None:
        record_path = self.record_dir / "human-record-h04-undated.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "scenario_id": "CS-CH-H04",
                    "decision": "ACCEPT",
                    "reviewer": "reviewer-redacted",
                    "review_timestamp": "June 24 after rehearsal",
                    "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                    "evidence_packet_manifest": human_gate_evidence_packet_manifest("CS-CH-H04"),
                    "environment_topology_ref": "topology:redacted-prod-like",
                    "request_context_proof": "request_context:trusted-redacted",
                    "db_policy_transcripts": ["db_policy:rls-redacted"],
                    "network_egress_transcripts": ["egress:default-deny-redacted"],
                    "backup_restore_evidence": ["backup_restore:security-redacted"],
                    "audit_integrity_report": "audit_integrity:security-redacted",
                    "evidence_manifest_ref": "evidence_manifest:security-redacted",
                    "findings_or_exceptions": "none",
                },
                sort_keys=True,
            )
            + "\n"
        )

        result = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H04",
            "--record-file",
            str(record_path),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "blocked")
        validation = payload["connector_human_gate_record_validation"]
        self.assertEqual(validation["status"], "record_structurally_invalid")
        self.assertEqual(validation["missing_required_fields"], [])
        self.assertEqual(validation["empty_required_fields"], [])
        self.assertEqual(validation["invalid_field_formats"], ["review_timestamp"])
        self.assertIn("invalid_field_formats", validation["structural_errors"])
        self.assertEqual(validation["negative_evidence"]["invalid_field_formats"], 1)
        self.assertFalse(validation["decision_recorded_by_validator"])
        self.assertNotIn("decision", validation)
        self.assertFalse(validation["record_file"]["path_recorded_by_validator"])
        self.assertNotIn("path", validation["record_file"])
        self.assertFalse(validation["product_claim_allowed"])
        self.assertFalse(validation["pass_claim_allowed_by_validator"])
        self.assertEqual(validation["matrix_status_after_validation"], "HUMAN_REQUIRED")
        payload_text = json.dumps(payload, sort_keys=True)
        self.assertNotIn('"decision": "ACCEPT"', payload_text)
        self.assertNotIn(str(record_path), payload_text)
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn("June 24 after rehearsal", state_text)
        self.assertNotIn(str(record_path), state_text)

    def test_connector_human_gate_validate_record_requires_senior_perspective_findings(self) -> None:
        record_path = self.record_dir / "human-record-h04-missing-perspectives.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "scenario_id": "CS-CH-H04",
                    "decision": "ACCEPT",
                    "reviewer": "reviewer-redacted",
                    "review_timestamp": "2026-06-24T12:00:00Z",
                    "evidence_packet_manifest": human_gate_evidence_packet_manifest("CS-CH-H04"),
                    "environment_topology_ref": "topology:redacted-prod-like",
                    "request_context_proof": "request_context:trusted-redacted",
                    "db_policy_transcripts": ["db_policy:rls-redacted"],
                    "network_egress_transcripts": ["egress:default-deny-redacted"],
                    "backup_restore_evidence": ["backup_restore:security-redacted"],
                    "audit_integrity_report": "audit_integrity:security-redacted",
                    "evidence_manifest_ref": "evidence_manifest:security-redacted",
                    "findings_or_exceptions": "none",
                },
                sort_keys=True,
            )
            + "\n"
        )

        result = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H04",
            "--record-file",
            str(record_path),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "blocked")
        validation = payload["connector_human_gate_record_validation"]
        self.assertEqual(validation["status"], "record_structurally_invalid")
        self.assertEqual(validation["missing_required_fields"], [])
        self.assertEqual(validation["empty_required_fields"], [])
        self.assertEqual(validation["invalid_field_formats"], [])
        self.assertFalse(validation["senior_review_perspective_findings_present"])
        self.assertFalse(validation["senior_review_perspective_findings_complete"])
        self.assertEqual(
            set(validation["missing_senior_review_perspectives"]),
            set(HUMAN_GATE_PERSPECTIVE_FINDINGS),
        )
        self.assertEqual(validation["empty_senior_review_perspectives"], [])
        self.assertEqual(validation["invalid_senior_review_perspective_roles"], [])
        self.assertIn("missing_senior_review_perspective_findings", validation["structural_errors"])
        self.assertIn("missing_senior_review_perspectives", validation["structural_errors"])
        self.assertEqual(
            validation["negative_evidence"]["missing_senior_review_perspectives"],
            len(HUMAN_GATE_PERSPECTIVE_FINDINGS),
        )
        self.assertFalse(validation["dependency_unlock_allowed_by_validator"])
        self.assertEqual(validation["dependency_unlock_blocked_reason"], "structural_errors")
        self.assertFalse(validation["pass_claim_allowed_by_validator"])
        self.assertEqual(validation["matrix_status_after_validation"], "HUMAN_REQUIRED")
        self.assertNotIn("decision", validation)
        self.assertFalse(validation["record_file"]["path_recorded_by_validator"])
        self.assertNotIn("path", validation["record_file"])
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn(str(record_path), state_text)

    def test_connector_human_gate_validate_record_requires_evidence_packet_manifest(self) -> None:
        record_path = self.record_dir / "human-record-h04-missing-evidence-manifest.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "scenario_id": "CS-CH-H04",
                    "decision": "ACCEPT",
                    "reviewer": "reviewer-redacted",
                    "review_timestamp": "2026-06-24T12:00:00Z",
                    "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                    "environment_topology_ref": "topology:redacted-prod-like",
                    "request_context_proof": "request_context:trusted-redacted",
                    "db_policy_transcripts": ["db_policy:rls-redacted"],
                    "network_egress_transcripts": ["egress:default-deny-redacted"],
                    "backup_restore_evidence": ["backup_restore:security-redacted"],
                    "audit_integrity_report": "audit_integrity:security-redacted",
                    "evidence_manifest_ref": "evidence_manifest:security-redacted",
                    "findings_or_exceptions": "none",
                },
                sort_keys=True,
            )
            + "\n"
        )

        result = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H04",
            "--record-file",
            str(record_path),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "blocked")
        validation = payload["connector_human_gate_record_validation"]
        self.assertEqual(validation["status"], "record_structurally_invalid")
        self.assertTrue(validation["senior_review_perspective_findings_complete"])
        self.assertFalse(validation["evidence_packet_manifest_present"])
        self.assertFalse(validation["evidence_packet_manifest_complete"])
        self.assertEqual(validation["missing_evidence_packet_manifest_items"], [1, 2, 3, 4])
        self.assertEqual(validation["empty_evidence_packet_manifest_items"], [])
        self.assertEqual(validation["invalid_evidence_packet_manifest_items"], [])
        self.assertEqual(validation["duplicate_evidence_packet_manifest_items"], [])
        self.assertIn("missing_evidence_packet_manifest", validation["structural_errors"])
        self.assertIn("missing_evidence_packet_manifest_items", validation["structural_errors"])
        self.assertEqual(validation["negative_evidence"]["missing_evidence_packet_manifest_items"], 4)
        self.assertFalse(validation["dependency_unlock_allowed_by_validator"])
        self.assertEqual(validation["dependency_unlock_blocked_reason"], "structural_errors")
        self.assertFalse(validation["pass_claim_allowed_by_validator"])
        self.assertEqual(validation["matrix_status_after_validation"], "HUMAN_REQUIRED")
        payload_text = json.dumps(payload, sort_keys=True)
        self.assertNotIn('"decision": "ACCEPT"', payload_text)
        self.assertNotIn("redacted product-value finding", payload_text)
        self.assertNotIn(str(record_path), payload_text)
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn("redacted product-value finding", state_text)
        self.assertNotIn(str(record_path), state_text)

    def test_connector_human_gate_validate_record_requires_typed_evidence_manifest(self) -> None:
        record_path = self.record_dir / "human-record-h04-invalid-evidence-manifest.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_manifest = human_gate_evidence_packet_manifest("CS-CH-H04")
        evidence_manifest[0]["required_evidence"] = "Wrong evidence label"
        evidence_manifest[1]["redaction_status"] = "raw_unredacted"
        record_path.write_text(
            json.dumps(
                {
                    "scenario_id": "CS-CH-H04",
                    "decision": "ACCEPT",
                    "reviewer": "reviewer-redacted",
                    "review_timestamp": "2026-06-24T12:00:00Z",
                    "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                    "evidence_packet_manifest": evidence_manifest,
                    "environment_topology_ref": "topology:redacted-prod-like",
                    "request_context_proof": "request_context:trusted-redacted",
                    "db_policy_transcripts": ["db_policy:rls-redacted"],
                    "network_egress_transcripts": ["egress:default-deny-redacted"],
                    "backup_restore_evidence": ["backup_restore:security-redacted"],
                    "audit_integrity_report": "audit_integrity:security-redacted",
                    "evidence_manifest_ref": "evidence_manifest:security-redacted",
                    "findings_or_exceptions": "none",
                },
                sort_keys=True,
            )
            + "\n"
        )

        result = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H04",
            "--record-file",
            str(record_path),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "blocked")
        validation = payload["connector_human_gate_record_validation"]
        self.assertEqual(validation["status"], "record_structurally_invalid")
        self.assertTrue(validation["evidence_packet_manifest_present"])
        self.assertFalse(validation["evidence_packet_manifest_complete"])
        self.assertEqual(validation["missing_evidence_packet_manifest_items"], [])
        self.assertEqual(validation["empty_evidence_packet_manifest_items"], [])
        self.assertEqual(validation["duplicate_evidence_packet_manifest_items"], [])
        self.assertEqual(
            validation["invalid_evidence_packet_manifest_items"],
            [
                "entry_1_required_evidence_mismatch",
                "entry_2_invalid_redaction_status",
            ],
        )
        self.assertIn("invalid_evidence_packet_manifest_items", validation["structural_errors"])
        self.assertEqual(validation["negative_evidence"]["invalid_evidence_packet_manifest_items"], 2)
        self.assertFalse(validation["dependency_unlock_allowed_by_validator"])
        self.assertEqual(validation["dependency_unlock_blocked_reason"], "structural_errors")
        self.assertFalse(validation["pass_claim_allowed_by_validator"])
        self.assertEqual(validation["matrix_status_after_validation"], "HUMAN_REQUIRED")
        payload_text = json.dumps(payload, sort_keys=True)
        self.assertNotIn("Wrong evidence label", payload_text)
        self.assertNotIn("raw_unredacted", payload_text)
        self.assertNotIn(str(record_path), payload_text)

    def test_connector_human_gate_validate_record_requires_unique_evidence_refs(self) -> None:
        record_path = self.record_dir / "human-record-h04-duplicate-evidence-ref.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_manifest = human_gate_evidence_packet_manifest("CS-CH-H04")
        for item in evidence_manifest:
            item["evidence_ref"] = "evidence:redacted-shared-h04-packet"
        record_path.write_text(
            json.dumps(
                {
                    "scenario_id": "CS-CH-H04",
                    "decision": "ACCEPT",
                    "reviewer": "reviewer-redacted",
                    "review_timestamp": "2026-06-24T12:00:00Z",
                    "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                    "evidence_packet_manifest": evidence_manifest,
                    "environment_topology_ref": "topology:redacted-prod-like",
                    "request_context_proof": "request_context:trusted-redacted",
                    "db_policy_transcripts": ["db_policy:rls-redacted"],
                    "network_egress_transcripts": ["egress:default-deny-redacted"],
                    "backup_restore_evidence": ["backup_restore:security-redacted"],
                    "audit_integrity_report": "audit_integrity:security-redacted",
                    "evidence_manifest_ref": "evidence_manifest:security-redacted",
                    "findings_or_exceptions": "none",
                },
                sort_keys=True,
            )
            + "\n"
        )

        result = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H04",
            "--record-file",
            str(record_path),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "blocked")
        validation = payload["connector_human_gate_record_validation"]
        self.assertEqual(validation["status"], "record_structurally_invalid")
        self.assertTrue(validation["evidence_packet_manifest_present"])
        self.assertFalse(validation["evidence_packet_manifest_complete"])
        self.assertEqual(validation["missing_evidence_packet_manifest_items"], [])
        self.assertEqual(validation["empty_evidence_packet_manifest_items"], [])
        self.assertEqual(validation["invalid_evidence_packet_manifest_items"], [])
        self.assertEqual(validation["duplicate_evidence_packet_manifest_items"], [])
        expected_duplicate_ref_fingerprint = (
            "sha256:"
            + hashlib.sha256("evidence:redacted-shared-h04-packet".encode("utf-8")).hexdigest()[:16]
        )
        self.assertEqual(
            validation["duplicate_evidence_packet_manifest_ref_fingerprints"],
            [expected_duplicate_ref_fingerprint],
        )
        self.assertIn("duplicate_evidence_packet_manifest_refs", validation["structural_errors"])
        self.assertEqual(validation["negative_evidence"]["duplicate_evidence_packet_manifest_refs"], 1)
        self.assertFalse(validation["dependency_unlock_allowed_by_validator"])
        self.assertEqual(validation["dependency_unlock_blocked_reason"], "structural_errors")
        self.assertFalse(validation["pass_claim_allowed_by_validator"])
        self.assertEqual(validation["matrix_status_after_validation"], "HUMAN_REQUIRED")
        payload_text = json.dumps(payload, sort_keys=True)
        self.assertNotIn(str(record_path), payload_text)
        self.assertNotIn("reviewer-redacted", payload_text)
        self.assertNotIn("evidence:redacted-shared-h04-packet", payload_text)
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn("evidence:redacted-shared-h04-packet", state_text)

        handoff_payload = run_json(
            "connector",
            "human-gate",
            "validation-handoff",
            "--state-dir",
            self.state_rel,
        )
        handoff_text = json.dumps(handoff_payload, sort_keys=True)
        self.assertNotIn("evidence:redacted-shared-h04-packet", handoff_text)
        h04_handoff = {
            row["scenario_id"]: row
            for row in handoff_payload["connector_human_gate_validation_handoff"][
                "scenario_validation_handoff_rows"
            ]
        }["CS-CH-H04"]
        issue_summary = h04_handoff["latest_record_validation_issue_summary"]
        self.assertIn(
            "duplicate_evidence_packet_manifest_refs",
            issue_summary["issue_categories"],
        )
        self.assertEqual(
            issue_summary["issue_counts"]["duplicate_evidence_packet_manifest_refs"],
            1,
        )
        self.assertEqual(
            issue_summary["issue_details"]["duplicate_evidence_packet_manifest_refs"],
            [expected_duplicate_ref_fingerprint],
        )
        self.assertFalse(issue_summary["evidence_packet_manifest_values_included"])

    def test_connector_human_gate_report_rolls_up_record_validations_without_promotion(self) -> None:
        record_path = self.record_dir / "human-record-h04.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "scenario_id": "CS-CH-H04",
                    "decision": "ACCEPT",
                    "reviewer": "reviewer-redacted",
                    "review_timestamp": "2026-06-24T12:00:00Z",
                    "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                    "evidence_packet_manifest": human_gate_evidence_packet_manifest("CS-CH-H04"),
                    "environment_topology_ref": "topology:redacted-prod-like",
                    "request_context_proof": "request_context:trusted-redacted",
                    "db_policy_transcripts": ["db_policy:rls-redacted"],
                    "network_egress_transcripts": ["egress:default-deny-redacted"],
                    "backup_restore_evidence": ["backup_restore:security-redacted"],
                    "audit_integrity_report": "audit_integrity:security-redacted",
                    "evidence_manifest_ref": "evidence_manifest:security-redacted",
                    "findings_or_exceptions": "none",
                },
                sort_keys=True,
            )
            + "\n"
        )
        validation_payload = run_json(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H04",
            "--record-file",
            str(record_path),
            "--state-dir",
            self.state_rel,
        )
        validation = validation_payload["connector_human_gate_record_validation"]
        self.assertEqual(validation["status"], "record_structurally_valid")

        report_payload = run_json(
            "connector",
            "human-gate",
            "report",
            "--state-dir",
            self.state_rel,
        )
        report = report_payload["connector_human_gate_readiness_report"]
        self.assertEqual(report["status"], "human_review_required")
        self.assertEqual(report["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(report_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(report_payload["summary"]["validation_count"], 1)
        self.assertEqual(report_payload["summary"]["structurally_valid_record_validation_count"], 1)
        self.assertEqual(report_payload["summary"]["senior_review_perspective_findings_complete_count"], 1)
        self.assertEqual(report_payload["summary"]["evidence_packet_manifest_complete_count"], 1)
        self.assertEqual(report_payload["summary"]["dependency_unlock_allowed_count"], 1)
        self.assertEqual(report_payload["summary"]["dependency_unlock_denied_count"], 0)
        self.assertEqual(report_payload["summary"]["pass_claims_allowed_by_validations"], 0)
        summary = report["record_validation_summary"]
        self.assertEqual(summary["validation_count"], 1)
        self.assertEqual(summary["structurally_valid_count"], 1)
        self.assertEqual(summary["structurally_invalid_count"], 0)
        self.assertEqual(summary["senior_review_perspective_findings_complete_count"], 1)
        self.assertEqual(summary["senior_review_perspective_findings_incomplete_count"], 0)
        self.assertEqual(summary["evidence_packet_manifest_complete_count"], 1)
        self.assertEqual(summary["evidence_packet_manifest_incomplete_count"], 0)
        self.assertEqual(summary["scenarios_with_record_validation"], ["CS-CH-H04"])
        self.assertEqual(summary["scenarios_with_structurally_valid_record_validation"], ["CS-CH-H04"])
        self.assertEqual(summary["dependency_unlock_allowed_count"], 1)
        self.assertEqual(summary["dependency_unlock_denied_count"], 0)
        self.assertEqual(summary["dependency_unlock_denied_record_validations"], [])
        self.assertEqual(summary["scenarios_with_dependency_unlock_record_validation"], ["CS-CH-H04"])
        self.assertNotIn("CS-CH-H04", summary["scenarios_missing_structurally_valid_record_validation"])
        self.assertNotIn("CS-CH-H04", summary["scenarios_missing_dependency_unlock_record_validation"])
        self.assertEqual(summary["product_claims_allowed_by_validations"], 0)
        self.assertEqual(summary["pass_claims_allowed_by_validations"], 0)
        self.assertEqual(summary["record_bodies_persisted_by_validations"], 0)
        self.assertEqual(summary["depends_on_human_gate_record_validation_not_applicable_rows"], ["CS-CH-H04"])
        self.assertEqual(summary["depends_on_human_gate_record_validation_ready_rows"], ["CS-CH-H07"])
        self.assertIn("CS-CH-H01", summary["depends_on_human_gate_record_validation_missing_rows"])
        rows = {row["scenario_id"]: row for row in report["scenario_results"]}
        h04 = rows["CS-CH-H04"]
        self.assertEqual(h04["status"], "HUMAN_REQUIRED")
        self.assertEqual(h04["record_validation_count"], 1)
        self.assertEqual(h04["record_validation_status"], "record_structurally_valid")
        self.assertTrue(h04["structurally_valid_record_validation_present"])
        self.assertTrue(h04["dependency_unlock_record_validation_present"])
        self.assertEqual(h04["depends_on_human_gate_record_validation_status"], "not_applicable")
        self.assertTrue(h04["depends_on_human_gate_record_validations_ready"])
        self.assertEqual(h04["depends_on_human_gates_with_structurally_valid_record_validation"], [])
        self.assertEqual(
            h04["depends_on_human_gates_missing_structurally_valid_record_validation"],
            [],
        )
        self.assertEqual(h04["depends_on_human_gates_with_dependency_unlock_record_validation"], [])
        self.assertEqual(h04["depends_on_human_gates_missing_dependency_unlock_record_validation"], [])
        self.assertEqual(h04["latest_record_validation"]["validation_id"], validation["validation_id"])
        self.assertEqual(h04["latest_record_validation"]["matrix_status_after_validation"], "HUMAN_REQUIRED")
        self.assertTrue(h04["latest_record_validation"]["senior_review_perspective_findings_complete"])
        self.assertEqual(h04["latest_record_validation"]["missing_senior_review_perspectives"], [])
        self.assertEqual(h04["latest_record_validation"]["empty_senior_review_perspectives"], [])
        self.assertEqual(h04["latest_record_validation"]["invalid_senior_review_perspective_roles"], [])
        self.assertTrue(h04["latest_record_validation"]["evidence_packet_manifest_complete"])
        self.assertEqual(h04["latest_record_validation"]["missing_evidence_packet_manifest_items"], [])
        self.assertEqual(h04["latest_record_validation"]["empty_evidence_packet_manifest_items"], [])
        self.assertEqual(h04["latest_record_validation"]["invalid_evidence_packet_manifest_items"], [])
        self.assertEqual(h04["latest_record_validation"]["duplicate_evidence_packet_manifest_items"], [])
        self.assertTrue(h04["latest_record_validation"]["dependency_unlock_allowed_by_validator"])
        self.assertIsNone(h04["latest_record_validation"]["dependency_unlock_blocked_reason"])
        self.assertFalse(h04["latest_record_validation"]["product_claim_allowed"])
        self.assertFalse(h04["latest_record_validation"]["pass_claim_allowed_by_validator"])
        self.assertEqual(report["negative_evidence"]["human_rows_marked_pass_by_report"], 0)
        self.assertEqual(report["negative_evidence"]["record_bodies_persisted_by_validations"], 0)
        self.assertEqual(report["non_mutation_evidence"]["human_record_validations_executed_by_report"], 1)
        self.assertFalse(report["non_mutation_evidence"]["record_bodies_persisted_by_report"])
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn(str(record_path), state_text)

    def test_connector_human_gate_validate_record_blocks_arbitrary_dependency_refs(self) -> None:
        record_path = self.record_dir / "human-record-h01-arbitrary-dependencies.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "scenario_id": "CS-CH-H01",
                    "decision": "ACCEPT",
                    "reviewer": "reviewer-redacted",
                    "review_timestamp": "2026-06-24T12:00:00Z",
                    "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                    "evidence_packet_manifest": human_gate_evidence_packet_manifest("CS-CH-H01"),
                    "github_app_installation_id_redacted": "installation-redacted-001",
                    "selected_repositories": ["owner/project-alpha"],
                    "permission_snapshot": "read-only metadata and contents permissions",
                    "call_ledger": ["GET /repos/owner/project-alpha"],
                    "delivery_refs": ["connector_delivery_receipt:del_live_redacted"],
                    "audit_refs": ["audit:audit_live_redacted"],
                    "zero_write_proof": "github_write_calls=0",
                    "issues_or_exceptions": "none",
                    "dependency_human_gate_refs": {
                        "CS-CH-H04": "human_record:security-redacted",
                        "CS-CH-H07": "human_record:recovery-redacted",
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )

        result = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H01",
            "--record-file",
            str(record_path),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "blocked")
        validation = payload["connector_human_gate_record_validation"]
        self.assertEqual(validation["status"], "record_structurally_invalid")
        self.assertEqual(validation["missing_required_fields"], [])
        self.assertEqual(validation["empty_required_fields"], [])
        self.assertEqual(validation["missing_dependency_human_gates"], [])
        self.assertEqual(
            set(validation["invalid_dependency_human_gate_refs"]),
            {"CS-CH-H04", "CS-CH-H07"},
        )
        self.assertIn("invalid_dependency_human_gate_refs", validation["structural_errors"])
        self.assertEqual(validation["negative_evidence"]["invalid_dependency_human_gate_refs"], 2)
        self.assertFalse(validation["pass_claim_allowed_by_validator"])
        self.assertEqual(validation["matrix_status_after_validation"], "HUMAN_REQUIRED")
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn("GET /repos/owner/project-alpha", state_text)
        self.assertNotIn(str(record_path), state_text)

    def test_connector_human_gate_report_marks_dependency_validation_ready_when_dependencies_valid(self) -> None:
        def validate_record(scenario_id: str, record: dict[str, object]) -> dict[str, object]:
            record_path = self.record_dir / f"human-record-{scenario_id.lower()}.json"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(record, sort_keys=True) + "\n")
            payload = run_json(
                "connector",
                "human-gate",
                "validate-record",
                "--scenario",
                scenario_id,
                "--record-file",
                str(record_path),
                "--state-dir",
                self.state_rel,
            )
            validation = payload["connector_human_gate_record_validation"]
            self.assertEqual(validation["status"], "record_structurally_valid")
            self.assertEqual(validation["matrix_status_after_validation"], "HUMAN_REQUIRED")
            self.assertFalse(validation["pass_claim_allowed_by_validator"])
            self.assertTrue(validation["dependency_unlock_allowed_by_validator"])
            self.assertIsNone(validation["dependency_unlock_blocked_reason"])
            return validation

        validations = {}
        validations["CS-CH-H04"] = validate_record(
            "CS-CH-H04",
            {
                "scenario_id": "CS-CH-H04",
                "decision": "ACCEPT",
                "reviewer": "reviewer-redacted",
                "review_timestamp": "2026-06-24T12:00:00Z",
                "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                "evidence_packet_manifest": human_gate_evidence_packet_manifest("CS-CH-H04"),
                "environment_topology_ref": "topology:redacted-prod-like",
                "request_context_proof": "request_context:trusted-redacted",
                "db_policy_transcripts": ["db_policy:rls-redacted"],
                "network_egress_transcripts": ["egress:default-deny-redacted"],
                "backup_restore_evidence": ["backup_restore:security-redacted"],
                "audit_integrity_report": "audit_integrity:security-redacted",
                "evidence_manifest_ref": "evidence_manifest:security-redacted",
                "findings_or_exceptions": "none",
            },
        )
        validations["CS-CH-H07"] = validate_record(
            "CS-CH-H07",
            {
                "scenario_id": "CS-CH-H07",
                "decision": "ACCEPT",
                "reviewer": "reviewer-redacted",
                "review_timestamp": "2026-06-24T12:05:00Z",
                "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                "evidence_packet_manifest": human_gate_evidence_packet_manifest("CS-CH-H07"),
                "backup_manifest_ref": "backup_manifest:recovery-redacted",
                "restore_log_ref": "restore_log:recovery-redacted",
                "cursor_reconciliation_ref": "cursor_reconciliation:recovery-redacted",
                "replay_results_ref": "replay_results:recovery-redacted",
                "audit_verification_ref": "audit_verification:recovery-redacted",
                "before_after_counts_hashes": "counts_and_hashes:recovery-redacted",
                "issues_or_exceptions": "none",
                "dependency_human_gate_refs": {
                    "CS-CH-H04": (
                        "connector_human_gate_record_validation:"
                        f"{validations['CS-CH-H04']['validation_id']}"
                    )
                },
            },
        )
        validations["CS-CH-H01"] = validate_record(
            "CS-CH-H01",
            {
                "scenario_id": "CS-CH-H01",
                "decision": "ACCEPT",
                "reviewer": "reviewer-redacted",
                "review_timestamp": "2026-06-24T12:10:00Z",
                "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                "evidence_packet_manifest": human_gate_evidence_packet_manifest("CS-CH-H01"),
                "github_app_installation_id_redacted": "installation-redacted-001",
                "selected_repositories": ["owner/project-alpha"],
                "permission_snapshot": "read-only metadata and contents permissions",
                "call_ledger": ["GET /repos/owner/project-alpha"],
                "delivery_refs": ["connector_delivery_receipt:del_live_redacted"],
                "audit_refs": ["audit:audit_live_redacted"],
                "zero_write_proof": "github_write_calls=0",
                "issues_or_exceptions": "none",
                "dependency_human_gate_refs": {
                    "CS-CH-H04": (
                        "connector_human_gate_record_validation:"
                        f"{validations['CS-CH-H04']['validation_id']}"
                    ),
                    "CS-CH-H07": (
                        "connector_human_gate_record_validation:"
                        f"{validations['CS-CH-H07']['validation_id']}"
                    ),
                },
            },
        )

        report_payload = run_json(
            "connector",
            "human-gate",
            "report",
            "--state-dir",
            self.state_rel,
        )
        report = report_payload["connector_human_gate_readiness_report"]
        self.assertEqual(report["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(report_payload["final_verdict"], "HUMAN_REQUIRED")
        self.assertEqual(report_payload["summary"]["validation_count"], 3)
        self.assertEqual(report_payload["summary"]["structurally_valid_record_validation_count"], 3)
        self.assertEqual(report_payload["summary"]["senior_review_perspective_findings_complete_count"], 3)
        self.assertEqual(report_payload["summary"]["evidence_packet_manifest_complete_count"], 3)
        self.assertEqual(report_payload["summary"]["dependency_unlock_allowed_count"], 3)
        self.assertEqual(report_payload["summary"]["dependency_unlock_denied_count"], 0)
        self.assertEqual(report_payload["summary"]["pass_claims_allowed_by_validations"], 0)
        summary = report["record_validation_summary"]
        self.assertEqual(summary["validation_count"], 3)
        self.assertEqual(summary["structurally_valid_count"], 3)
        self.assertEqual(summary["senior_review_perspective_findings_complete_count"], 3)
        self.assertEqual(summary["senior_review_perspective_findings_incomplete_count"], 0)
        self.assertEqual(summary["evidence_packet_manifest_complete_count"], 3)
        self.assertEqual(summary["evidence_packet_manifest_incomplete_count"], 0)
        self.assertEqual(
            summary["scenarios_with_structurally_valid_record_validation"],
            ["CS-CH-H01", "CS-CH-H04", "CS-CH-H07"],
        )
        self.assertEqual(
            summary["scenarios_with_dependency_unlock_record_validation"],
            ["CS-CH-H01", "CS-CH-H04", "CS-CH-H07"],
        )
        self.assertEqual(summary["dependency_unlock_allowed_count"], 3)
        self.assertEqual(summary["dependency_unlock_denied_count"], 0)
        self.assertEqual(summary["depends_on_human_gate_record_validation_not_applicable_rows"], ["CS-CH-H04"])
        self.assertIn("CS-CH-H01", summary["depends_on_human_gate_record_validation_ready_rows"])
        self.assertIn("CS-CH-H02", summary["depends_on_human_gate_record_validation_ready_rows"])
        self.assertIn("CS-CH-H05", summary["depends_on_human_gate_record_validation_ready_rows"])
        self.assertIn("CS-CH-H03", summary["depends_on_human_gate_record_validation_missing_rows"])
        self.assertIn("CS-CH-H06", summary["depends_on_human_gate_record_validation_missing_rows"])
        rows = {row["scenario_id"]: row for row in report["scenario_results"]}
        self.assertEqual(rows["CS-CH-H04"]["depends_on_human_gate_record_validation_status"], "not_applicable")
        self.assertTrue(rows["CS-CH-H04"]["depends_on_human_gate_record_validations_ready"])
        self.assertTrue(rows["CS-CH-H04"]["dependency_unlock_record_validation_present"])
        self.assertEqual(rows["CS-CH-H04"]["depends_on_human_gates_missing_structurally_valid_record_validation"], [])
        self.assertEqual(rows["CS-CH-H07"]["depends_on_human_gate_record_validation_status"], "ready")
        self.assertTrue(rows["CS-CH-H07"]["depends_on_human_gate_record_validations_ready"])
        self.assertTrue(rows["CS-CH-H07"]["dependency_unlock_record_validation_present"])
        self.assertEqual(
            rows["CS-CH-H07"]["depends_on_human_gates_with_structurally_valid_record_validation"],
            ["CS-CH-H04"],
        )
        self.assertEqual(
            rows["CS-CH-H07"]["depends_on_human_gates_with_dependency_unlock_record_validation"],
            ["CS-CH-H04"],
        )
        h01 = rows["CS-CH-H01"]
        self.assertEqual(h01["status"], "HUMAN_REQUIRED")
        self.assertEqual(h01["depends_on_human_gate_record_validation_status"], "ready")
        self.assertTrue(h01["depends_on_human_gate_record_validations_ready"])
        self.assertTrue(h01["dependency_unlock_record_validation_present"])
        self.assertEqual(
            h01["depends_on_human_gates_with_structurally_valid_record_validation"],
            ["CS-CH-H04", "CS-CH-H07"],
        )
        self.assertEqual(
            h01["depends_on_human_gates_with_dependency_unlock_record_validation"],
            ["CS-CH-H04", "CS-CH-H07"],
        )
        self.assertEqual(h01["depends_on_human_gates_missing_structurally_valid_record_validation"], [])
        self.assertEqual(h01["depends_on_human_gates_missing_dependency_unlock_record_validation"], [])
        self.assertEqual(h01["latest_record_validation"]["validation_id"], validations["CS-CH-H01"]["validation_id"])
        self.assertTrue(h01["latest_record_validation"]["dependency_unlock_allowed_by_validator"])
        self.assertFalse(h01["latest_record_validation"]["pass_claim_allowed_by_validator"])
        self.assertEqual(report["negative_evidence"]["human_rows_marked_pass_by_report"], 0)
        next_payload = run_json(
            "connector",
            "human-gate",
            "next",
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(next_payload["status"], "success")
        self.assertEqual(next_payload["final_verdict"], "HUMAN_REQUIRED")
        next_report = next_payload["connector_human_gate_next"]
        self.assertEqual(next_report["status"], "next_ready")
        self.assertEqual(next_report["next_scenario_id"], "CS-CH-H02")
        self.assertEqual(next_report["next_review_order"], 4)
        self.assertEqual(next_report["next_required_human_delta"], [])
        self.assertEqual(next_report["next_recommended_preflight_commands"], [])
        self.assertIsNone(next_report["next_local_baseline_review_inputs"])
        self.assertIsNone(next_report["next_local_baseline_preflight_bundle"])
        h02_required_record = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS["CS-CH-H02"]["required_human_record"]
        self.assertEqual(next_report["next_required_human_fields"], h02_required_record["required_fields"])
        self.assertEqual(next_report["next_required_evidence"], h02_required_record["required_evidence"])
        self.assertEqual(
            next_report["next_remaining_human_field_count"],
            len(h02_required_record["required_fields"]),
        )
        self.assertEqual(
            next_report["next_remaining_human_evidence_count"],
            len(h02_required_record["required_evidence"]),
        )
        self.assertEqual(
            next_report["next_remaining_human_evidence_summary"]["claim_boundary"],
            CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY,
        )
        self.assertEqual(next_report["ready_scenario_ids"], ["CS-CH-H02", "CS-CH-H05"])
        self.assertEqual(next_report["blocked_scenario_ids"], ["CS-CH-H03", "CS-CH-H06"])
        self.assertEqual(
            next_report["completed_human_gate_record_validation_scenarios"],
            ["CS-CH-H04", "CS-CH-H07", "CS-CH-H01"],
        )
        self.assertEqual(next_report["completed_human_gate_record_validation_count"], 3)
        self.assertEqual(next_report["completed_dependency_unlock_record_validation_count"], 3)
        self.assertEqual(
            next_report["completed_dependency_unlock_record_validation_scenarios"],
            ["CS-CH-H04", "CS-CH-H07", "CS-CH-H01"],
        )
        self.assertEqual(next_report["record_validation_count"], 3)
        self.assertEqual(next_report["structurally_valid_record_validation_count"], 3)
        self.assertEqual(next_report["dependency_unlock_allowed_count"], 3)
        self.assertEqual(next_report["dependency_unlock_denied_count"], 0)
        self.assert_human_gate_delivery_unit_plan_summary(
            next_report["next_scenario_delivery_unit_plan_summary"]
        )
        self.assertTrue(next_report["next_scenario_delivery_unit_plan_ready"])
        self.assertEqual(next_report["next_scenario_delivery_unit_plan_lifecycle_step_count"], 7)
        self.assertGreaterEqual(next_report["next_scenario_delivery_unit_plan_senior_review_perspective_count"], 6)
        self.assertFalse(next_report["next_scenario_delivery_unit_plan_product_claim_allowed"])
        self.assertFalse(next_report["next_scenario_delivery_unit_plan_pass_claim_allowed"])
        self.assertFalse(next_report["next_scenario_delivery_unit_plan_approval_collected"])
        self.assertFalse(next_report["next_scenario_delivery_unit_plan_dependency_unlock_allowed"])
        self.assertEqual(
            next_report["blocked_by_missing_dependency_validations"]["CS-CH-H03"],
            ["CS-CH-H02"],
        )
        self.assertEqual(
            next_report["blocked_by_missing_dependency_validations"]["CS-CH-H06"],
            ["CS-CH-H02", "CS-CH-H03", "CS-CH-H05"],
        )
        self.assertEqual(next_report["negative_evidence"]["human_rows_marked_pass_by_next"], 0)
        self.assertEqual(next_report["negative_evidence"]["approvals_collected_by_next"], 0)
        self.assertEqual(next_payload["summary"]["next_scenario_id"], "CS-CH-H02")
        self.assertEqual(
            next_payload["summary"]["next_source_requirement_ids"],
            next_report["next_source_requirement_ids"],
        )
        self.assertEqual(
            next_payload["summary"]["source_requirement_human_pending_ids"],
            HUMAN_GATE_SOURCE_REQUIREMENT_HUMAN_PENDING_IDS,
        )
        self.assertEqual(
            next_payload["summary"]["source_requirement_claim_boundary"],
            CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY,
        )
        self.assertEqual(
            next_payload["summary"]["next_remaining_human_evidence_summary"],
            next_report["next_remaining_human_evidence_summary"],
        )
        self.assertEqual(
            next_payload["summary"]["next_release_impact"],
            next_report["next_release_impact"],
        )
        self.assertEqual(
            next_payload["summary"]["next_stop_or_reject_when"],
            next_report["next_stop_or_reject_when"],
        )
        self.assertEqual(
            next_payload["summary"]["next_record_template_output_command"],
            next_report["next_record_template_output_command"],
        )
        self.assertEqual(
            next_payload["summary"]["next_record_validation_command"],
            next_report["next_record_validation_command"],
        )
        self.assertEqual(
            next_payload["summary"]["next_record_validation_output_command"],
            next_report["next_record_validation_output_command"],
        )
        self.assertEqual(next_payload["summary"]["next_required_human_delta"], [])
        self.assertEqual(
            next_payload["summary"]["next_remaining_human_field_count"],
            len(h02_required_record["required_fields"]),
        )
        self.assertEqual(
            next_payload["summary"]["next_remaining_human_evidence_count"],
            len(h02_required_record["required_evidence"]),
        )
        self.assertEqual(next_payload["summary"]["next_required_human_delta_count"], 0)
        self.assertEqual(next_payload["summary"]["next_recommended_preflight_command_count"], 0)
        self.assertEqual(next_payload["summary"]["next_recommended_preflight_command_plan_count"], 0)
        self.assertEqual(next_payload["summary"]["next_local_baseline_review_input_report_count"], 0)
        self.assertIsNone(next_payload["summary"]["next_local_baseline_acceptance_sufficient"])
        self.assertIsNone(next_payload["summary"]["next_local_baseline_preflight_bundle_ready_report_count"])
        self.assertIsNone(next_payload["summary"]["next_local_baseline_preflight_bundle_acceptance_sufficient"])
        self.assertTrue(next_payload["summary"]["next_scenario_delivery_unit_plan_ready"])
        self.assertEqual(next_payload["summary"]["next_scenario_delivery_unit_plan_lifecycle_step_count"], 7)
        self.assertGreaterEqual(
            next_payload["summary"]["next_scenario_delivery_unit_plan_senior_review_perspective_count"],
            6,
        )
        self.assertFalse(next_payload["summary"]["next_scenario_delivery_unit_plan_product_claim_allowed"])
        self.assertFalse(next_payload["summary"]["next_scenario_delivery_unit_plan_pass_claim_allowed"])
        self.assertFalse(next_payload["summary"]["next_scenario_delivery_unit_plan_approval_collected"])
        self.assertFalse(next_payload["summary"]["next_scenario_delivery_unit_plan_dependency_unlock_allowed"])
        self.assertEqual(next_payload["summary"]["ready_scenario_count"], 2)
        self.assertEqual(next_payload["summary"]["blocked_scenario_count"], 2)
        self.assertEqual(next_payload["summary"]["completed_human_gate_record_validation_count"], 3)
        self.assertEqual(next_payload["summary"]["dependency_unlock_allowed_count"], 3)
        self.assertEqual(next_payload["summary"]["dependency_unlock_denied_count"], 0)
        self.assertEqual(
            next_payload["summary"]["readiness_report_ref"],
            next_report["readiness_report_ref"],
        )
        self.assertIn(f"connector_human_gate_next:{next_report['next_id']}", next_payload["evidence_refs"])
        self.assertTrue(next_payload["audit_refs"])
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn("GET /repos/owner/project-alpha", state_text)

    def test_connector_human_gate_rejection_record_does_not_unlock_dependencies(self) -> None:
        h04_reject_path = self.record_dir / "human-record-h04-reject.json"
        h04_reject_path.parent.mkdir(parents=True, exist_ok=True)
        h04_reject_path.write_text(
            json.dumps(
                {
                    "scenario_id": "CS-CH-H04",
                    "decision": "REJECT",
                    "reviewer": "reviewer-redacted",
                    "review_timestamp": "2026-06-24T12:00:00Z",
                    "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                    "evidence_packet_manifest": human_gate_evidence_packet_manifest("CS-CH-H04"),
                    "environment_topology_ref": "topology:redacted-prod-like",
                    "request_context_proof": "request_context:trusted-redacted",
                    "db_policy_transcripts": ["db_policy:rls-redacted"],
                    "network_egress_transcripts": ["egress:default-deny-redacted"],
                    "backup_restore_evidence": ["backup_restore:security-redacted"],
                    "audit_integrity_report": "audit_integrity:security-redacted",
                    "evidence_manifest_ref": "evidence_manifest:security-redacted",
                    "findings_or_exceptions": "reject: unexpected egress proof gap",
                },
                sort_keys=True,
            )
            + "\n"
        )

        reject_payload = run_json(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H04",
            "--record-file",
            str(h04_reject_path),
            "--state-dir",
            self.state_rel,
        )
        reject_validation = reject_payload["connector_human_gate_record_validation"]
        self.assertEqual(reject_validation["status"], "record_structurally_valid")
        self.assertTrue(reject_validation["decision_allowed"])
        self.assertFalse(reject_validation["decision_recorded_by_validator"])
        self.assertFalse(reject_validation["dependency_unlock_allowed_by_validator"])
        self.assertEqual(reject_validation["dependency_unlock_blocked_reason"], "decision_not_accept")
        self.assertFalse(reject_validation["pass_claim_allowed_by_validator"])
        self.assertEqual(reject_validation["matrix_status_after_validation"], "HUMAN_REQUIRED")
        self.assertNotIn("decision", reject_validation)

        h07_path = self.record_dir / "human-record-h07-after-h04-reject.json"
        h07_path.write_text(
            json.dumps(
                {
                    "scenario_id": "CS-CH-H07",
                    "decision": "ACCEPT",
                    "reviewer": "reviewer-redacted",
                    "review_timestamp": "2026-06-24T12:05:00Z",
                    "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                    "evidence_packet_manifest": human_gate_evidence_packet_manifest("CS-CH-H07"),
                    "backup_manifest_ref": "backup_manifest:recovery-redacted",
                    "restore_log_ref": "restore_log:recovery-redacted",
                    "cursor_reconciliation_ref": "cursor_reconciliation:recovery-redacted",
                    "replay_results_ref": "replay_results:recovery-redacted",
                    "audit_verification_ref": "audit_verification:recovery-redacted",
                    "before_after_counts_hashes": "counts_and_hashes:recovery-redacted",
                    "issues_or_exceptions": "none",
                    "dependency_human_gate_refs": {
                        "CS-CH-H04": (
                            "connector_human_gate_record_validation:"
                            f"{reject_validation['validation_id']}"
                        )
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )
        h07_result = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H07",
            "--record-file",
            str(h07_path),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(h07_result.returncode, 1, h07_result.stdout + h07_result.stderr)
        h07_payload = json.loads(h07_result.stdout)
        self.assertEqual(h07_payload["status"], "blocked")
        h07_validation = h07_payload["connector_human_gate_record_validation"]
        self.assertEqual(h07_validation["status"], "record_structurally_invalid")
        self.assertEqual(h07_validation["missing_dependency_human_gates"], [])
        self.assertEqual(h07_validation["invalid_dependency_human_gate_refs"], ["CS-CH-H04"])
        self.assertIn("invalid_dependency_human_gate_refs", h07_validation["structural_errors"])
        self.assertFalse(h07_validation["dependency_unlock_allowed_by_validator"])
        self.assertEqual(h07_validation["dependency_unlock_blocked_reason"], "structural_errors")

        report_payload = run_json(
            "connector",
            "human-gate",
            "report",
            "--state-dir",
            self.state_rel,
        )
        report = report_payload["connector_human_gate_readiness_report"]
        summary = report["record_validation_summary"]
        self.assertEqual(summary["validation_count"], 2)
        self.assertEqual(summary["structurally_valid_count"], 1)
        self.assertEqual(summary["senior_review_perspective_findings_complete_count"], 2)
        self.assertEqual(summary["senior_review_perspective_findings_incomplete_count"], 0)
        self.assertEqual(summary["evidence_packet_manifest_complete_count"], 2)
        self.assertEqual(summary["evidence_packet_manifest_incomplete_count"], 0)
        self.assertEqual(summary["dependency_unlock_allowed_count"], 0)
        self.assertEqual(summary["dependency_unlock_denied_count"], 1)
        self.assertEqual(summary["scenarios_with_structurally_valid_record_validation"], ["CS-CH-H04"])
        self.assertEqual(summary["scenarios_with_dependency_unlock_record_validation"], [])
        rows = {row["scenario_id"]: row for row in report["scenario_results"]}
        self.assertTrue(rows["CS-CH-H04"]["structurally_valid_record_validation_present"])
        self.assertFalse(rows["CS-CH-H04"]["dependency_unlock_record_validation_present"])
        self.assertEqual(
            rows["CS-CH-H07"]["depends_on_human_gates_with_structurally_valid_record_validation"],
            ["CS-CH-H04"],
        )
        self.assertEqual(
            rows["CS-CH-H07"]["depends_on_human_gates_missing_structurally_valid_record_validation"],
            [],
        )
        self.assertEqual(
            rows["CS-CH-H07"]["depends_on_human_gates_with_dependency_unlock_record_validation"],
            [],
        )
        self.assertEqual(
            rows["CS-CH-H07"]["depends_on_human_gates_missing_dependency_unlock_record_validation"],
            ["CS-CH-H04"],
        )
        self.assertEqual(rows["CS-CH-H07"]["depends_on_human_gate_record_validation_status"], "missing_dependency_validations")

        next_payload = run_json(
            "connector",
            "human-gate",
            "next",
            "--state-dir",
            self.state_rel,
        )
        next_report = next_payload["connector_human_gate_next"]
        self.assertEqual(next_report["status"], "next_ready")
        self.assertEqual(next_report["next_scenario_id"], "CS-CH-H04")
        self.assertEqual(next_report["ready_scenario_ids"], ["CS-CH-H04"])
        self.assertEqual(next_report["completed_dependency_unlock_record_validation_count"], 0)
        self.assertEqual(next_report["dependency_unlock_allowed_count"], 0)
        self.assertEqual(next_report["dependency_unlock_denied_count"], 1)
        self.assertEqual(next_report["next_record_validation_count"], 1)
        self.assertEqual(next_report["next_record_validation_status"], "record_structurally_valid")
        self.assertTrue(
            next_report["next_latest_record_validation_ref"].startswith(
                "connector_human_gate_record_validation:"
            )
        )
        self.assertFalse(next_report["next_latest_record_validation_dependency_unlock_allowed"])
        next_issue_summary = next_report["next_latest_record_validation_issue_summary"]
        self.assertEqual(
            next_issue_summary["schema_version"],
            "cs.connector_human_gate_validation_issue_summary.v1",
        )
        self.assertEqual(next_issue_summary["status"], "record_structurally_valid")
        self.assertFalse(next_issue_summary["structural_correction_required"])
        self.assertFalse(next_issue_summary["dependency_unlock_ready"])
        self.assertEqual(next_issue_summary["dependency_unlock_blocked_reason"], "decision_not_accept")
        self.assertEqual(
            next_issue_summary["operator_next_step"],
            "record_is_structurally_valid_but_does_not_unlock_dependencies",
        )
        self.assertFalse(next_issue_summary["product_claim_allowed"])
        self.assertFalse(next_issue_summary["pass_claim_allowed"])
        self.assertFalse(next_issue_summary["structural_validation_is_human_acceptance"])
        self.assertTrue(next_issue_summary["human_acceptance_requires_owner_promotion"])
        self.assertEqual(
            next_issue_summary["completion_claim_boundary"],
            CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
        )
        self.assertFalse(next_issue_summary["raw_record_body_included"])
        self.assertFalse(next_issue_summary["raw_record_path_included"])
        self.assertFalse(next_issue_summary["human_decision_value_included"])
        self.assertFalse(next_issue_summary["senior_review_finding_text_included"])
        self.assertFalse(next_issue_summary["evidence_packet_manifest_values_included"])

        payload_text = json.dumps([reject_payload, h07_payload, report_payload, next_payload], sort_keys=True)
        self.assertNotIn('"decision": "REJECT"', payload_text)
        self.assertNotIn(str(h04_reject_path), payload_text)
        self.assertNotIn(str(h07_path), payload_text)
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn('"decision": "REJECT"', state_text)
        self.assertNotIn(str(h04_reject_path), state_text)
        self.assertNotIn(str(h07_path), state_text)

    def test_connector_human_gate_validate_record_blocks_missing_dependencies_and_secret_markers(self) -> None:
        record_path = self.record_dir / "human-record-h01-invalid.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "scenario_id": "CS-CH-H01",
                    "decision": "APPROVE",
                    "reviewer": "reviewer-redacted",
                    "review_timestamp": "2026-06-24T12:00:00Z",
                    "senior_review_perspective_findings": HUMAN_GATE_PERSPECTIVE_FINDINGS,
                    "evidence_packet_manifest": human_gate_evidence_packet_manifest("CS-CH-H01"),
                    "github_app_installation_id_redacted": "installation-redacted-001",
                    "permission_snapshot": "ghp_abcdefghijklmnop",
                },
                sort_keys=True,
            )
            + "\n"
        )

        result = run_cli(
            "connector",
            "human-gate",
            "validate-record",
            "--scenario",
            "CS-CH-H01",
            "--record-file",
            str(record_path),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("CS_CONNECTOR_HUMAN_GATE_RECORD_INVALID", {error["code"] for error in payload["errors"]})
        validation = payload["connector_human_gate_record_validation"]
        self.assertEqual(validation["status"], "record_structurally_invalid")
        self.assertTrue(validation["decision_present"])
        self.assertFalse(validation["decision_allowed"])
        self.assertFalse(validation["decision_recorded_by_validator"])
        self.assertNotIn("decision", validation)
        self.assertIn("decision_not_allowed", validation["structural_errors"])
        self.assertIn("missing_required_fields", validation["structural_errors"])
        self.assertIn("missing_dependency_human_gate_refs", validation["structural_errors"])
        self.assertIn("sensitive_marker_detected", validation["structural_errors"])
        self.assertEqual(validation["invalid_field_formats"], [])
        self.assertIn("CS-CH-H04", validation["missing_dependency_human_gates"])
        self.assertIn("CS-CH-H07", validation["missing_dependency_human_gates"])
        self.assertIn("zero_write_proof", validation["missing_required_fields"])
        self.assertGreaterEqual(len(validation["sensitive_marker_findings"]), 1)
        self.assertFalse(validation["pass_claim_allowed_by_validator"])
        self.assertEqual(validation["matrix_status_after_validation"], "HUMAN_REQUIRED")
        self.assertEqual(validation["negative_evidence"]["human_rows_marked_pass_by_validator"], 0)
        self.assertEqual(validation["negative_evidence"]["human_decision_value_persisted_by_validator"], 0)
        self.assertEqual(validation["negative_evidence"]["record_path_persisted_by_validator"], 0)
        payload_text = json.dumps(payload, sort_keys=True)
        self.assertNotIn('"decision": "APPROVE"', payload_text)
        self.assertNotIn(str(record_path), payload_text)
        state_text = state_file_texts(self.state_dir)
        self.assertNotIn("ghp_abcdefghijklmnop", state_text)
        self.assertNotIn(str(record_path), state_text)

    def test_connector_report_lint_separates_fixture_live_and_production_claims_cs_ch_040(self) -> None:
        report_path = self.state_dir / "connector-report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "summary": {"product_feature_claims": "LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_CH0_ONLY"},
                    "readiness_dimensions": {
                        "contract_schema_verified": "LOCAL_FIXTURE_VERIFIED",
                        "local_fixture_behavior_verified": "LOCAL_FIXTURE_VERIFIED",
                        "local_physical_device_behavior_verified": "HUMAN_REQUIRED",
                        "live_provider_read_verified": "NOT_VERIFIED",
                        "live_provider_write_verified": "OUT_OF_SCOPE_READ_ONLY",
                        "production_tenancy_policy_egress_verified": "NOT_VERIFIED",
                        "human_ux_privacy_accepted": "HUMAN_REQUIRED",
                        "release_publishing_approved": "NOT_VERIFIED",
                    },
                    "negative_evidence": {"production_readiness_overclaims": 0},
                    "scenario_results": [],
                    "human_required": [],
                },
                sort_keys=True,
            )
            + "\n"
        )
        lint = run_json(
            "connector",
            "report-lint",
            "--report",
            str(report_path),
            "--state-dir",
            self.state_rel,
        )
        self.assertEqual(lint["status"], "success")
        report_lint = lint["connector_report_lint"]
        self.assertEqual(report_lint["status"], "pass")
        self.assertEqual(report_lint["negative_overclaim_counter"], 0)
        self.assertEqual(report_lint["readiness_dimensions"]["live_provider_read_verified"], "NOT_VERIFIED")
        self.assertEqual(report_lint["readiness_dimensions"]["production_tenancy_policy_egress_verified"], "NOT_VERIFIED")
        self.assertEqual(report_lint["readiness_dimensions"]["human_ux_privacy_accepted"], "HUMAN_REQUIRED")
        self.assertTrue(lint["audit_refs"])

        report_path.write_text(
            json.dumps(
                {
                    "summary": {"product_feature_claims": "PRODUCTION_READY"},
                    "readiness_dimensions": {},
                    "negative_evidence": {"production_readiness_overclaims": 1},
                    "scenario_results": [],
                    "human_required": [],
                },
                sort_keys=True,
            )
            + "\n"
        )
        failed = run_cli(
            "connector",
            "report-lint",
            "--report",
            str(report_path),
            "--state-dir",
            self.state_rel,
            "--json",
        )
        self.assertEqual(failed.returncode, 4, failed.stdout + failed.stderr)
        failed_payload = json.loads(failed.stdout)
        self.assertEqual(failed_payload["status"], "failed")
        self.assertGreater(failed_payload["connector_report_lint"]["negative_overclaim_counter"], 0)

    @unittest.skipIf(SKIP_VS2_REGRESSION_TESTS, "VS2 proof construction validates ConnectorHub egress proof after the reusable proof is written")
    def test_connectorhub_scenario_list_and_filtered_verify(self) -> None:
        ensure_vs2_reusable_proof_current(self)
        listed = run_json("scenario", "list", "--set", "connectorhub")
        self.assertEqual(listed["count"], 47)
        self.assertIn("CS-CH-001", {row["id"] for row in listed["scenarios"]})

        result = run_cli(
            "scenario",
            "verify",
            "connector-contract-adapter",
            "--scenario",
            "CS-CH-001",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["summary"]["scenario_count"], 1)
        self.assertEqual(payload["summary"]["pass"], 1)
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["scenario_filter"], ["CS-CH-001"])
        self.assertEqual(payload["scenario_results"][0]["id"], "CS-CH-001")
        self.assertEqual(payload["scenario_results"][0]["status"], "PASS")
        self.assertEqual(payload["negative_evidence"]["unauthorized_provider_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["provider_credentials_exposed"], 0)
        self.assertLess(len(payload["command_evidence"]), payload["command_evidence_filter"]["original_count"])
        self.assertEqual(payload["command_evidence_filter"]["focused_count"], len(payload["command_evidence"]))
        self.assertEqual(payload["command_evidence_filter"]["scenario_ids"], ["CS-CH-001"])

        result = run_cli(
            "scenario",
            "verify",
            "connector-contract-adapter",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["summary"]["scenario_count"], 40)
        self.assertEqual(payload["summary"]["pass"], 40)
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(
            payload["summary"]["product_feature_claims"],
            CONNECTORHUB_LOCAL_FIXTURE_PRODUCT_CLAIM,
        )
        self.assertEqual(payload["negative_evidence"]["production_readiness_overclaims"], 0)
        self.assertEqual(payload["negative_evidence"]["projection_envelope_checksum_mismatches"], 0)
        self.assertEqual(payload["negative_evidence"]["product_interpretation_before_archive_commit"], 0)
        self.assertEqual(payload["negative_evidence"]["acknowledged_without_artifact"], 0)
        self.assertEqual(payload["negative_evidence"]["ack_before_durable_commit"], 0)
        self.assertEqual(payload["negative_evidence"]["duplicate_connector_artifacts"], 0)
        self.assertEqual(payload["negative_evidence"]["infinite_retry_loops"], 0)
        self.assertEqual(payload["negative_evidence"]["queue_wide_blockage"], 0)
        self.assertEqual(payload["negative_evidence"]["raw_payload_in_quarantine_output"], 0)
        self.assertEqual(payload["negative_evidence"]["duplicate_active_connector_truth"], 0)
        self.assertEqual(payload["negative_evidence"]["immutable_history_mutations"], 0)
        self.assertEqual(payload["negative_evidence"]["forbidden_source_policy_field_leaks"], 0)
        self.assertEqual(payload["negative_evidence"]["raw_content_policy_leaks"], 0)
        self.assertEqual(payload["negative_evidence"]["evidenceref_only_approved_truth"], 0)
        self.assertEqual(payload["negative_evidence"]["inaccessible_phantom_evidence"], 0)
        self.assertEqual(payload["negative_evidence"]["reusable_raw_access_handles"], 0)
        self.assertEqual(payload["negative_evidence"]["raw_access_read_limit_bypasses"], 0)
        self.assertEqual(payload["negative_evidence"]["raw_access_expiry_bypasses"], 0)
        self.assertEqual(payload["negative_evidence"]["raw_access_revocation_bypasses"], 0)
        self.assertEqual(payload["negative_evidence"]["raw_access_payload_or_handle_leaks"], 0)
        self.assertEqual(payload["negative_evidence"]["tool_calls_from_untrusted_connector_content"], 0)
        self.assertEqual(payload["negative_evidence"]["watch_result_inferred_intent_labeled_observed_fact"], 0)
        self.assertEqual(payload["negative_evidence"]["watch_result_low_confidence_memory_approved_without_review"], 0)
        self.assertEqual(payload["negative_evidence"]["watch_result_proposal_executed_directly"], 0)
        self.assertEqual(payload["negative_evidence"]["watch_result_workflow_runs_started"], 0)
        self.assertEqual(payload["negative_evidence"]["action_cards_from_untrusted_connector_content"], 0)
        self.assertEqual(payload["negative_evidence"]["action_preflight_dry_run_executed"], 0)
        self.assertEqual(payload["negative_evidence"]["action_preflight_preflight_counted_as_approval"], 0)
        self.assertEqual(payload["negative_evidence"]["action_preflight_execution_result_created"], 0)
        self.assertEqual(payload["negative_evidence"]["action_preflight_workflow_runs_started"], 0)
        self.assertEqual(payload["negative_evidence"]["action_preflight_provider_mutations"], 0)
        self.assertEqual(payload["negative_evidence"]["action_preflight_external_http_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["action_preflight_real_provider_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["action_preflight_direct_provider_access"], 0)
        self.assertEqual(payload["negative_evidence"]["action_preflight_credential_values_exposed"], 0)
        self.assertEqual(payload["negative_evidence"]["action_preflight_github_read_only_action_admitted"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["action_preflight_count"], 7)
        self.assertEqual(payload["connector_contract_evidence"]["action_preflight_review_count"], 7)
        self.assertEqual(payload["negative_evidence"]["action_safety_executions_without_evidence"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_executions_without_policy_allow"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_executions_without_authorized_approval"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_unauthorized_approvals_accepted"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_connector_permission_inferred_from_product_approval"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_product_approval_inferred_from_connector_permission"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_executions_without_idempotency"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_stale_preflight_executions"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_cross_namespace_executions"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_execution_results_created"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_workflow_runs_started"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_external_http_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_provider_mutations"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_real_provider_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["action_safety_credential_values_exposed"], 0)
        self.assertGreaterEqual(payload["connector_contract_evidence"]["action_safety_envelope_count"], 8)
        self.assertEqual(payload["connector_contract_evidence"]["action_safety_execution_result_count"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["action_safety_workflow_run_count"], 0)
        self.assertEqual(payload["negative_evidence"]["action_execution_without_workflow_run"], 0)
        self.assertEqual(payload["negative_evidence"]["action_execution_without_action_result"], 0)
        self.assertEqual(payload["negative_evidence"]["action_execution_without_provider_receipt"], 0)
        self.assertEqual(payload["negative_evidence"]["action_execution_without_outcome_evidence"], 0)
        self.assertEqual(payload["negative_evidence"]["action_execution_duplicate_side_effects"], 0)
        self.assertEqual(payload["negative_evidence"]["action_execution_direct_provider_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["action_execution_external_http_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["action_execution_raw_provider_payloads_persisted"], 0)
        self.assertEqual(payload["negative_evidence"]["action_execution_credential_values_exposed"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["action_execution_workflow_run_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["action_execution_action_result_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["action_execution_provider_receipt_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["action_execution_outcome_artifact_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["action_execution_connected_outcome_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["action_execution_idempotency_count"], 1)
        self.assertEqual(payload["negative_evidence"]["action_retry_duplicate_side_effects"], 0)
        self.assertEqual(payload["negative_evidence"]["action_retry_conflicts_executed"], 0)
        self.assertEqual(payload["negative_evidence"]["action_retry_second_action_results_created"], 0)
        self.assertEqual(payload["negative_evidence"]["action_retry_hidden_automatic_compensation"], 0)
        self.assertEqual(payload["negative_evidence"]["action_retry_external_http_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["action_retry_provider_mutations"], 0)
        self.assertEqual(payload["negative_evidence"]["action_retry_real_provider_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["action_retry_credential_values_exposed"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["action_retry_workflow_run_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["action_retry_action_result_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["action_retry_provider_receipt_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["action_retry_idempotency_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["action_retry_connected_outcome_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["action_retry_outcome_artifact_count"], 1)
        self.assertGreaterEqual(payload["connector_contract_evidence"]["action_retry_safety_envelope_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["action_retry_provider_internal_findings"], [])
        action_retry_checks = payload["connector_contract_evidence"]["action_retry_checks"]
        self.assertEqual(action_retry_checks["same_key_retry_returns_existing_result"], True)
        self.assertEqual(action_retry_checks["conflict_denied_before_second_side_effect"], True)
        self.assertEqual(action_retry_checks["compensation_expectation_visible_without_automatic_execution"], True)
        self.assertEqual(action_retry_checks["durable_counts_single_provider_effect"], True)
        self.assertEqual(
            payload["connector_contract_evidence"]["action_retry_conflict_reason_code"],
            "CS_ACTION_IDEMPOTENCY_CONFLICT",
        )
        self.assertNotEqual(
            payload["connector_contract_evidence"]["action_retry_conflict_existing_request_digest"],
            payload["connector_contract_evidence"]["action_retry_conflict_incoming_request_digest"],
        )
        self.assertEqual(payload["negative_evidence"]["scope_isolation_cross_scope_setup_allowed"], 0)
        self.assertEqual(payload["negative_evidence"]["scope_isolation_cross_scope_delivery_returned"], 0)
        self.assertEqual(payload["negative_evidence"]["scope_isolation_cross_scope_evidence_returned"], 0)
        self.assertEqual(payload["negative_evidence"]["scope_isolation_cross_scope_watch_returned"], 0)
        self.assertEqual(payload["negative_evidence"]["scope_isolation_cross_scope_action_executed"], 0)
        self.assertEqual(payload["negative_evidence"]["scope_isolation_other_scope_records_persisted"], 0)
        self.assertEqual(payload["negative_evidence"]["scope_isolation_ownerless_connector_records"], 0)
        self.assertEqual(payload["negative_evidence"]["scope_isolation_workflow_runs_started"], 0)
        self.assertEqual(payload["negative_evidence"]["scope_isolation_action_results_created"], 0)
        self.assertEqual(payload["negative_evidence"]["scope_isolation_provider_receipts_created"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["scope_isolation_delivery_receipt_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["scope_isolation_evidence_bundle_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["scope_isolation_watch_result_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["scope_isolation_action_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["scope_isolation_action_result_count"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["scope_isolation_workflow_run_count"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["scope_isolation_provider_receipt_count"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["scope_isolation_provider_internal_findings"], [])
        scope_isolation_checks = payload["connector_contract_evidence"]["scope_isolation_checks"]
        self.assertEqual(scope_isolation_checks["connector_application_setup_scope_bound"], True)
        self.assertEqual(scope_isolation_checks["delivery_artifact_and_evidence_scope_bound"], True)
        self.assertEqual(scope_isolation_checks["watch_result_scope_bound"], True)
        self.assertEqual(scope_isolation_checks["action_path_scope_bound_without_execution"], True)
        self.assertEqual(scope_isolation_checks["cross_scope_setup_delivery_evidence_watch_action_denied"], True)
        self.assertEqual(payload["negative_evidence"]["credential_custody_raw_secret_canary_in_stdout"], 0)
        self.assertEqual(payload["negative_evidence"]["credential_custody_raw_secret_canary_in_state"], 0)
        self.assertEqual(payload["negative_evidence"]["credential_custody_raw_secret_values_exposed"], 0)
        self.assertEqual(payload["negative_evidence"]["credential_custody_raw_handles_exposed"], 0)
        self.assertEqual(payload["negative_evidence"]["credential_custody_auth_headers_exposed"], 0)
        self.assertEqual(payload["negative_evidence"]["credential_custody_credential_bearing_urls_exposed"], 0)
        self.assertEqual(payload["negative_evidence"]["credential_custody_product_secret_writes"], 0)
        self.assertEqual(payload["negative_evidence"]["credential_custody_provider_auth_imports"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["credential_custody_lifecycle_count"], 3)
        self.assertEqual(payload["connector_contract_evidence"]["credential_custody_boundary_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["credential_custody_static_provider_auth_import_findings"], [])
        self.assertEqual(payload["connector_contract_evidence"]["credential_custody_provider_internal_findings"], [])
        credential_custody_checks = payload["connector_contract_evidence"]["credential_custody_checks"]
        self.assertEqual(credential_custody_checks["lifecycle_records_safe_and_scoped"], True)
        self.assertEqual(credential_custody_checks["rotation_and_revocation_update_status_without_product_secret"], True)
        self.assertEqual(credential_custody_checks["credential_boundary_safe"], True)
        self.assertEqual(credential_custody_checks["seeded_secret_canary_absent_from_outputs_and_state"], True)
        self.assertEqual(credential_custody_checks["static_provider_auth_import_scan_zero"], True)
        self.assertEqual(payload["negative_evidence"]["egress_topology_vs2_reuse_errors"], 0)
        self.assertEqual(payload["negative_evidence"]["egress_topology_missing_required_vs2_rows"], 0)
        self.assertEqual(payload["negative_evidence"]["egress_topology_failed_required_vs2_rows"], 0)
        self.assertEqual(payload["negative_evidence"]["egress_topology_direct_http_socket_bypass_allowed"], 0)
        self.assertEqual(payload["negative_evidence"]["egress_topology_provider_requests_after_direct_attempts"], 0)
        self.assertEqual(payload["negative_evidence"]["egress_topology_default_denied_sink_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["egress_topology_redirect_denied_hop_trap_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["egress_topology_sensitive_headers_forwarded_to_denied_hop"], 0)
        self.assertEqual(payload["negative_evidence"]["egress_topology_raw_credentials_exposed"], 0)
        self.assertEqual(payload["negative_evidence"]["egress_topology_raw_payloads_in_audit"], 0)
        self.assertEqual(payload["negative_evidence"]["egress_topology_production_topology_overclaimed"], 0)
        self.assertEqual(payload["negative_evidence"]["audit_correlation_missing_required_event_families"], 0)
        self.assertEqual(payload["negative_evidence"]["audit_correlation_uncorrelated_connector_events"], 0)
        self.assertEqual(payload["negative_evidence"]["audit_correlation_duplicate_correlation_ids"], 0)
        self.assertEqual(payload["negative_evidence"]["audit_correlation_scope_mismatches"], 0)
        self.assertEqual(payload["negative_evidence"]["audit_correlation_raw_payload_or_secret_leaks"], 0)
        self.assertEqual(payload["negative_evidence"]["audit_correlation_integrity_errors"], 0)
        self.assertEqual(payload["negative_evidence"]["audit_correlation_tamper_detection_failures"], 0)
        egress_topology_checks = payload["connector_contract_evidence"]["egress_topology_checks"]
        self.assertTrue(all(egress_topology_checks.values()), egress_topology_checks)
        egress_topology_negative = payload["connector_contract_evidence"]["egress_topology_negative_evidence"]
        for key in [
            "vs2_reuse_errors",
            "missing_required_vs2_rows",
            "failed_required_vs2_rows",
            "direct_http_socket_bypass_allowed",
            "provider_requests_after_direct_attempts",
            "default_denied_sink_calls",
            "redirect_denied_hop_trap_calls",
            "sensitive_headers_forwarded_to_denied_hop",
            "raw_credentials_exposed",
            "raw_payloads_in_audit",
            "production_topology_overclaimed",
        ]:
            self.assertEqual(egress_topology_negative[key], 0, key)
        egress_source_fingerprint = payload["connector_contract_evidence"]["egress_topology_source_fingerprint"]
        self.assertEqual(
            egress_source_fingerprint["recorded_input_digest"],
            egress_source_fingerprint["current_input_digest"],
        )
        self.assertEqual(
            egress_source_fingerprint["recorded_working_tree_digest"],
            egress_source_fingerprint["current_working_tree_digest"],
        )
        self.assertEqual(payload["negative_evidence"]["action_bypass_undeclared_actions_executed"], 0)
        self.assertEqual(payload["negative_evidence"]["action_bypass_direct_provider_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["action_bypass_external_http_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["action_bypass_provider_mutations"], 0)
        self.assertEqual(payload["negative_evidence"]["action_bypass_real_provider_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["action_bypass_provider_clients_exposed"], 0)
        self.assertEqual(payload["negative_evidence"]["action_bypass_credential_values_exposed"], 0)
        self.assertEqual(payload["negative_evidence"]["action_bypass_malicious_pack_activations"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["action_bypass_preflight_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["action_bypass_action_result_count"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["action_bypass_workflow_run_count"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["action_bypass_quarantine_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["action_bypass_static_provider_import_findings"], [])
        self.assertEqual(payload["connector_contract_evidence"]["action_bypass_provider_internal_findings"], [])
        self.assertEqual(payload["negative_evidence"]["workflow_runs_from_untrusted_connector_content"], 0)
        self.assertEqual(payload["negative_evidence"]["external_calls_from_untrusted_connector_content"], 0)
        self.assertEqual(payload["negative_evidence"]["memory_promotions_from_untrusted_connector_content"], 0)
        self.assertEqual(payload["negative_evidence"]["policy_overrides_from_untrusted_connector_content"], 0)
        self.assertEqual(payload["negative_evidence"]["authority_expansions_from_untrusted_connector_content"], 0)
        self.assertEqual(payload["negative_evidence"]["unselected_repository_artifacts"], 0)
        self.assertEqual(payload["negative_evidence"]["unselected_repository_delivery_receipts"], 0)
        self.assertEqual(payload["negative_evidence"]["unselected_repository_acknowledgements"], 0)
        self.assertEqual(payload["negative_evidence"]["organization_wide_repository_fallbacks"], 0)
        self.assertEqual(payload["negative_evidence"]["github_write_permissions_requested"], 0)
        self.assertEqual(payload["negative_evidence"]["github_write_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["source_control_actions_declared"], 0)
        self.assertEqual(payload["negative_evidence"]["provider_pack_write_mappings"], 0)
        self.assertEqual(payload["negative_evidence"]["provider_mutations"], 0)
        self.assertEqual(payload["negative_evidence"]["github_write_cli_commands_exposed"], 0)
        self.assertEqual(payload["negative_evidence"]["github_write_contracts_accepted"], 0)
        self.assertEqual(payload["negative_evidence"]["github_write_egress_allowed"], 0)
        self.assertEqual(payload["negative_evidence"]["github_write_endpoint_literals"], 0)
        self.assertEqual(payload["negative_evidence"]["selected_repository_policy_broadening_silent_successes"], 0)
        self.assertEqual(payload["negative_evidence"]["source_control_projection_family_missing_artifacts"], 0)
        self.assertEqual(payload["negative_evidence"]["source_control_projection_family_missing_search_evidence"], 0)
        self.assertEqual(payload["negative_evidence"]["source_control_missing_source_revisions"], 0)
        self.assertEqual(payload["negative_evidence"]["source_control_provider_specific_product_requirements"], 0)
        self.assertEqual(payload["negative_evidence"]["source_control_raw_provider_payload_leaks"], 0)
        self.assertEqual(payload["negative_evidence"]["source_control_ack_before_commit"], 0)
        self.assertEqual(payload["negative_evidence"]["github_content_raw_sensitive_marker_leaks"], 0)
        self.assertEqual(payload["negative_evidence"]["github_content_imports_outside_allowed_path"], 0)
        self.assertEqual(payload["negative_evidence"]["github_content_generated_artifacts"], 0)
        self.assertEqual(payload["negative_evidence"]["github_content_binary_raw_content_imports"], 0)
        self.assertEqual(payload["negative_evidence"]["github_content_large_file_silent_truncations"], 0)
        self.assertEqual(payload["negative_evidence"]["github_content_private_material_artifacts"], 0)
        self.assertEqual(payload["negative_evidence"]["github_failure_silent_data_deletions"], 0)
        self.assertEqual(payload["negative_evidence"]["github_failure_fresh_sync_claims_while_suspended"], 0)
        self.assertEqual(payload["negative_evidence"]["github_failure_tight_retry_loops"], 0)
        self.assertEqual(payload["negative_evidence"]["github_failure_fabricated_current_data"], 0)
        self.assertEqual(payload["negative_evidence"]["github_removed_repository_future_ingestions"], 0)
        self.assertEqual(payload["negative_evidence"]["github_revoked_permission_streams_active"], 0)
        self.assertEqual(payload["negative_evidence"]["github_reconnect_without_owner_action"], 0)
        self.assertEqual(payload["negative_evidence"]["github_failure_external_http_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["github_failure_provider_mutations"], 0)
        self.assertEqual(payload["negative_evidence"]["capture_before_consent"], 0)
        self.assertEqual(payload["negative_evidence"]["capture_before_platform_permission"], 0)
        self.assertEqual(payload["negative_evidence"]["capture_samples_before_both_gates"], 0)
        self.assertEqual(payload["negative_evidence"]["capture_hidden_startup_capture"], 0)
        self.assertEqual(payload["negative_evidence"]["capture_cross_namespace_capture"], 0)
        self.assertEqual(payload["negative_evidence"]["capture_screenshots_before_permission"], 0)
        self.assertEqual(payload["negative_evidence"]["capture_window_titles_before_permission"], 0)
        self.assertEqual(payload["negative_evidence"]["capture_external_http_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["capture_provider_mutations"], 0)
        self.assertEqual(payload["negative_evidence"]["activity_unsupported_intent_claims"], 0)
        self.assertEqual(payload["negative_evidence"]["activity_inference_stored_as_observed_fact"], 0)
        self.assertEqual(payload["negative_evidence"]["activity_raw_window_titles_stored"], 0)
        self.assertEqual(payload["negative_evidence"]["activity_full_urls_stored"], 0)
        self.assertEqual(payload["negative_evidence"]["activity_keystrokes_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["activity_clipboard_values_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["activity_screenshots_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["activity_cookies_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["activity_browser_history_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["activity_external_http_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["activity_provider_mutations"], 0)
        self.assertEqual(payload["negative_evidence"]["activity_session_artifacts_created"], 0)
        self.assertEqual(payload["negative_evidence"]["watch_rule_ownerless_global_rules"], 0)
        self.assertEqual(payload["negative_evidence"]["watch_rule_cross_namespace_lifecycle_mutations"], 0)
        self.assertEqual(payload["negative_evidence"]["watch_rule_authority_expansions_from_rule_text"], 0)
        self.assertEqual(payload["negative_evidence"]["watch_rule_external_actions_authorized_by_rule"], 0)
        self.assertEqual(payload["negative_evidence"]["watch_rule_capture_broadening_without_confirmation"], 0)
        self.assertEqual(payload["negative_evidence"]["watch_rule_external_http_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["watch_rule_provider_mutations"], 0)
        self.assertEqual(payload["negative_evidence"]["watch_rule_artifacts_created"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_broad_all_urls_permission"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_captures_without_user_gesture"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_captures_without_confirmation"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_popup_open_captures"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_non_active_tab_captures"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_backend_policy_bypasses"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_blocked_page_text_clip_stored"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_raw_text_stored"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_raw_html_stored"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_cookies_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_local_storage_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_session_storage_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_screenshots_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_form_values_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_browser_history_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_external_http_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_provider_mutations"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_active_tab_artifacts_created"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_captures_without_owner_rule"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_captures_without_site_allowance"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_captures_without_source_pack_allowance"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_captures_without_browser_permission"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_consent_config_version_mismatches"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_unapproved_domain_captures"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_inactive_tab_captures"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_throttle_bypasses"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_session_limit_bypasses"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_duplicate_idempotency_captures"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_raw_text_stored"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_raw_html_stored"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_cookies_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_storage_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_screenshots_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_form_values_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_browser_history_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_external_http_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_provider_mutations"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_auto_capture_artifacts_created"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_client_block_downgrades"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_backend_false_safe_bypasses"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_blocked_page_text_persisted"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_degraded_raw_text_persisted"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_raw_html_stored"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_cookies_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_storage_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_screenshots_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_form_values_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_browser_history_collected"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_full_urls_stored"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_full_origins_stored"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_title_text_stored"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_content_sent_to_models"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_searchable_content_artifacts_created"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_capture_inbox_items_created"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_external_http_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["chrome_sensitive_page_provider_mutations"], 0)
        self.assertEqual(
            payload["connector_contract_evidence"]["content_restriction_checks"][
                "content_restriction_persisted_decision_refs_present"
            ],
            True,
        )
        self.assertEqual(
            payload["connector_contract_evidence"]["content_restriction_checks"][
                "content_restriction_persisted_result_links_present"
            ],
            True,
        )
        self.assertEqual(payload["negative_evidence"]["incremental_sync_duplicate_logical_artifacts"], 0)
        self.assertEqual(payload["negative_evidence"]["incremental_sync_cursor_advanced_before_commit"], 0)
        self.assertEqual(payload["negative_evidence"]["incremental_sync_missed_cursor_receipts"], 0)
        self.assertEqual(payload["negative_evidence"]["incremental_sync_duplicate_product_events"], 0)
        self.assertEqual(payload["negative_evidence"]["incremental_sync_source_revision_lineage_gaps"], 0)
        self.assertEqual(payload["negative_evidence"]["incremental_sync_unverified_webhook_commits"], 0)
        self.assertEqual(payload["readiness_dimensions"]["live_provider_read_verified"], "NOT_VERIFIED")

        aggregate_payload = payload
        aggregate_results_by_id = {row["id"]: row for row in aggregate_payload["scenario_results"]}

        def filtered_payload(scenario_id: str) -> dict[str, object]:
            row = aggregate_results_by_id[scenario_id]
            filtered = dict(aggregate_payload)
            filtered["scenario_filter"] = [scenario_id]
            filtered["scenario_results"] = [row]
            filtered_summary = dict(aggregate_payload["summary"])
            filtered_summary["scenario_count"] = 1
            filtered_summary["pass"] = 1 if row["status"] == "PASS" else 0
            filtered_summary["fail"] = 1 if row["status"] == "FAIL" else 0
            filtered_summary["not_verified"] = 1 if row["status"] == "NOT_VERIFIED" else 0
            filtered_summary["human_required"] = 1 if row["status"] == "HUMAN_REQUIRED" else 0
            filtered_summary["blocking"] = 0 if row["status"] in {"PASS", "HUMAN_REQUIRED"} else 1
            filtered["summary"] = filtered_summary
            filtered["status"] = "success" if filtered_summary["blocking"] == 0 else "failed"
            return filtered

        def assert_filtered_pass(scenario_id: str) -> dict[str, object]:
            scenario_payload = filtered_payload(scenario_id)
            self.assertEqual(scenario_payload["status"], "success")
            self.assertEqual(scenario_payload["summary"]["scenario_count"], 1)
            self.assertEqual(scenario_payload["summary"]["pass"], 1)
            self.assertEqual(scenario_payload["summary"]["blocking"], 0)
            self.assertEqual(scenario_payload["scenario_filter"], [scenario_id])
            self.assertEqual(scenario_payload["scenario_results"][0]["id"], scenario_id)
            self.assertEqual(scenario_payload["scenario_results"][0]["status"], "PASS")
            return scenario_payload

        def assert_negative_zero(scenario_payload: dict[str, object], *keys: str) -> None:
            negative_evidence = scenario_payload["negative_evidence"]
            for key in keys:
                self.assertEqual(negative_evidence[key], 0)

        def assert_checks_true(checks: dict[str, object], *keys: str) -> None:
            for key in keys:
                self.assertEqual(checks[key], True)

        payload = assert_filtered_pass("CS-CH-004")
        self.assertEqual(payload["status"], "success")
        assert_negative_zero(payload, "provider_credentials_exposed")

        payload = assert_filtered_pass("CS-CH-007")
        assert_negative_zero(
            payload,
            "ownerless_connector_artifacts",
            "projection_envelope_checksum_mismatches",
            "projection_acknowledgements_before_cs_ch_008",
        )

        payload = assert_filtered_pass("CS-CH-008")
        assert_negative_zero(
            payload,
            "acknowledged_without_artifact",
            "ack_before_durable_commit",
            "duplicate_connector_artifacts",
        )

        payload = assert_filtered_pass("CS-CH-009")
        assert_negative_zero(payload, "infinite_retry_loops", "queue_wide_blockage", "raw_payload_in_quarantine_output")

        payload = assert_filtered_pass("CS-CH-010")
        assert_negative_zero(payload, "duplicate_active_connector_truth", "immutable_history_mutations")
        assert_checks_true(
            payload["connector_contract_evidence"]["lineage_checks"],
            "one_current_logical_truth",
            "historical_evidence_not_mutated",
        )

        payload = assert_filtered_pass("CS-CH-011")
        assert_negative_zero(payload, "forbidden_source_policy_field_leaks", "raw_content_policy_leaks")
        assert_checks_true(
            payload["connector_contract_evidence"]["policy_checks"],
            "forbidden_body_rejected",
            "narrowed_policy_applies_to_subsequent_delivery",
        )

        payload = assert_filtered_pass("CS-CH-012")
        assert_negative_zero(payload, "evidenceref_only_approved_truth", "inaccessible_phantom_evidence")
        assert_checks_true(
            payload["connector_contract_evidence"]["evidence_checks"],
            "claim_evidence_backed_then_approved",
            "evidenceref_only_bundle_denied",
        )

        payload = assert_filtered_pass("CS-CH-013")
        assert_negative_zero(
            payload,
            "reusable_raw_access_handles",
            "raw_access_read_limit_bypasses",
            "raw_access_expiry_bypasses",
            "raw_access_revocation_bypasses",
            "raw_access_payload_or_handle_leaks",
        )
        assert_checks_true(
            payload["connector_contract_evidence"]["raw_access_checks"],
            "raw_default_request_denied",
            "raw_read_limit_exhaustion_denied",
            "raw_expiry_denied",
            "raw_revoked_read_denied",
        )

        payload = assert_filtered_pass("CS-CH-014")
        assert_negative_zero(
            payload,
            "tool_calls_from_untrusted_connector_content",
            "action_cards_from_untrusted_connector_content",
            "workflow_runs_from_untrusted_connector_content",
            "external_calls_from_untrusted_connector_content",
            "memory_promotions_from_untrusted_connector_content",
            "policy_overrides_from_untrusted_connector_content",
            "authority_expansions_from_untrusted_connector_content",
        )
        assert_checks_true(
            payload["connector_contract_evidence"]["untrusted_checks"],
            "unsafe_instruction_blocked",
            "bundle_trust_boundary_coverage",
            "agent_prompt_authority_denied",
            "memory_promotion_quarantined",
            "egress_denied_without_http_call",
        )

        payload = assert_filtered_pass("CS-CH-015")
        assert_negative_zero(
            payload,
            "unselected_repository_artifacts",
            "unselected_repository_delivery_receipts",
            "unselected_repository_acknowledgements",
            "organization_wide_repository_fallbacks",
            "github_write_permissions_requested",
            "github_write_calls",
        )
        assert_checks_true(
            payload["connector_contract_evidence"]["selected_repo_checks"],
            "three_visible_one_selected",
            "selected_repo_unselected_delivery_denied",
            "selected_repo_direct_write_denied",
            "selected_repo_expand_selection_denied",
        )

        payload = assert_filtered_pass("CS-CH-016")
        assert_negative_zero(
            payload,
            "source_control_projection_family_missing_artifacts",
            "source_control_projection_family_missing_search_evidence",
            "source_control_missing_source_revisions",
            "source_control_provider_specific_product_requirements",
        )
        assert_checks_true(
            payload["connector_contract_evidence"]["source_control_checks"],
            "all_projection_types_archived",
            "all_source_revisions_persisted",
            "searchable_evidence_bundles_created",
            "provider_specific_fields_not_required_by_product",
        )

        payload = assert_filtered_pass("CS-CH-017")
        assert_negative_zero(
            payload,
            "incremental_sync_duplicate_logical_artifacts",
            "incremental_sync_cursor_advanced_before_commit",
            "incremental_sync_missed_cursor_receipts",
            "incremental_sync_duplicate_product_events",
            "incremental_sync_source_revision_lineage_gaps",
            "incremental_sync_unverified_webhook_commits",
        )
        assert_checks_true(
            payload["connector_contract_evidence"]["incremental_sync_checks"],
            "bad_webhook_denied_before_delivery_commit",
            "provider_event_key_parts_complete",
            "cursor_gap_visible_before_replay",
            "final_reconciliation_success",
        )
        self.assertEqual(
            payload["connector_contract_evidence"]["incremental_sync_reconciliation"][
                "unobserved_delivery_receipt_count"
            ],
            0,
        )
        self.assertEqual(
            payload["connector_contract_evidence"]["incremental_sync_gap_reconciliation"][
                "unobserved_delivery_receipt_count"
            ],
            1,
        )

        payload = assert_filtered_pass("CS-CH-018")
        assert_negative_zero(
            payload,
            "github_content_raw_sensitive_marker_leaks",
            "github_content_imports_outside_allowed_path",
            "github_content_generated_artifacts",
            "github_content_binary_raw_content_imports",
            "github_content_large_file_silent_truncations",
            "github_content_private_material_artifacts",
        )
        assert_checks_true(
            payload["connector_contract_evidence"]["content_restriction_checks"],
            "redacted_secret_marker_delivery_succeeds",
            "binary_content_metadata_only_delivery_succeeds",
            "large_content_metadata_only_delivery_succeeds",
            "forbidden_path_skipped_before_artifact_or_receipt",
            "generated_content_skipped_before_artifact",
            "private_material_quarantined_before_artifact_or_receipt",
            "content_restriction_persisted_decision_refs_present",
            "content_restriction_persisted_result_links_present",
        )

        payload = assert_filtered_pass("CS-CH-019")
        assert_negative_zero(
            payload,
            "source_control_actions_declared",
            "provider_pack_write_mappings",
            "provider_mutations",
            "github_write_calls",
            "github_write_permissions_requested",
            "github_write_cli_commands_exposed",
            "github_write_contracts_accepted",
            "github_write_egress_allowed",
            "github_write_endpoint_literals",
        )
        assert_checks_true(
            payload["connector_contract_evidence"]["github_write_guard_checks"],
            "write_action_contract_rejected",
            "static_guard_exit_zero",
            "controlled_egress_attempts_denied",
            "direct_write_runtime_attempts_denied",
            "negative_counters_zero",
        )
        self.assertEqual(payload["connector_contract_evidence"]["github_write_guard_contract_record_count"], 0)

        payload = assert_filtered_pass("CS-CH-020")
        assert_negative_zero(
            payload,
            "github_failure_silent_data_deletions",
            "github_failure_fresh_sync_claims_while_suspended",
            "github_failure_tight_retry_loops",
            "github_removed_repository_future_ingestions",
        )
        assert_checks_true(
            payload["connector_contract_evidence"]["github_failure_checks"],
            "rate_limit_retry_schedule_visible_no_tight_loop",
            "revoked_permission_permanent_setup_gap_suspends_stream",
            "repository_removed_stops_future_ingestion_and_marks_unavailable",
            "existing_evidence_preserved_with_freshness_warnings",
        )
        self.assertEqual(payload["connector_contract_evidence"]["github_failure_state_count"], 4)

        payload = assert_filtered_pass("CS-CH-021")
        assert_negative_zero(
            payload,
            "capture_before_consent",
            "capture_before_platform_permission",
            "capture_samples_before_both_gates",
            "capture_hidden_startup_capture",
            "capture_cross_namespace_capture",
            "capture_screenshots_before_permission",
            "capture_window_titles_before_permission",
            "capture_external_http_calls",
            "capture_provider_mutations",
        )
        assert_checks_true(
            payload["connector_contract_evidence"]["capture_checks"],
            "no_consent_no_permission_disabled",
            "permission_only_still_disabled",
            "consent_only_still_disabled",
            "both_gates_ready_without_starting_capture",
            "permission_not_treated_as_production_proof",
        )
        self.assertEqual(payload["connector_contract_evidence"]["capture_permission_probe_count"], 2)
        self.assertEqual(payload["connector_contract_evidence"]["capture_consent_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["capture_guard_count"], 4)
        self.assertEqual(payload["connector_contract_evidence"]["capture_sample_count"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["capture_artifact_count"], 0)

        payload = assert_filtered_pass("CS-CH-022")
        assert_negative_zero(
            payload,
            "activity_unsupported_intent_claims",
            "activity_inference_stored_as_observed_fact",
            "activity_raw_window_titles_stored",
            "activity_full_urls_stored",
            "activity_keystrokes_collected",
            "activity_clipboard_values_collected",
            "activity_screenshots_collected",
            "activity_cookies_collected",
            "activity_browser_history_collected",
            "activity_external_http_calls",
            "activity_provider_mutations",
            "activity_session_artifacts_created",
        )
        assert_checks_true(
            payload["connector_contract_evidence"]["activity_checks"],
            "deterministic_metrics_recorded",
            "three_bounded_activity_session_projections",
            "no_unsupported_intent_claim",
            "privacy_mode_excludes_raw_capture",
        )
        self.assertEqual(payload["connector_contract_evidence"]["activity_sample_batch_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["activity_sessionization_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["activity_session_count"], 3)
        self.assertEqual(payload["connector_contract_evidence"]["activity_artifact_count"], 0)
        self.assertEqual(len(payload["connector_contract_evidence"]["activity_session_ids"]), 3)
        self.assertTrue(all(payload["connector_contract_evidence"]["activity_session_ids"]))
        self.assertEqual(
            payload["connector_contract_evidence"]["activity_session_source_sample_ids"],
            [["sample-001", "sample-002", "sample-003"], ["sample-005", "sample-007"], ["sample-008"]],
        )

        payload = assert_filtered_pass("CS-CH-024")
        assert_negative_zero(
            payload,
            "chrome_active_tab_broad_all_urls_permission",
            "chrome_active_tab_captures_without_user_gesture",
            "chrome_active_tab_captures_without_confirmation",
            "chrome_active_tab_popup_open_captures",
            "chrome_active_tab_non_active_tab_captures",
            "chrome_active_tab_backend_policy_bypasses",
            "chrome_active_tab_blocked_page_text_clip_stored",
            "chrome_active_tab_raw_text_stored",
            "chrome_active_tab_raw_html_stored",
            "chrome_active_tab_cookies_collected",
            "chrome_active_tab_local_storage_collected",
            "chrome_active_tab_session_storage_collected",
            "chrome_active_tab_screenshots_collected",
            "chrome_active_tab_form_values_collected",
            "chrome_active_tab_browser_history_collected",
            "chrome_active_tab_external_http_calls",
            "chrome_active_tab_provider_mutations",
            "chrome_active_tab_artifacts_created",
        )
        assert_checks_true(
            payload["connector_contract_evidence"]["chrome_active_tab_checks"],
            "no_consent_denied_by_backend",
            "popup_only_browser_internal_blocked_without_summary",
            "allowed_capture_creates_summary_and_inbox",
            "server_revalidated_policy",
            "summary_only_no_raw_browser_persistence",
        )
        self.assertEqual(payload["connector_contract_evidence"]["chrome_active_tab_permission_event_count"], 2)
        self.assertEqual(payload["connector_contract_evidence"]["chrome_active_tab_payload_count"], 2)
        self.assertEqual(payload["connector_contract_evidence"]["chrome_active_tab_policy_decision_count"], 3)
        self.assertEqual(payload["connector_contract_evidence"]["chrome_active_tab_summary_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["chrome_active_tab_inbox_item_count"], 1)
        self.assertEqual(payload["connector_contract_evidence"]["chrome_active_tab_artifact_count"], 0)

        payload = assert_filtered_pass("CS-CH-026")
        assert_negative_zero(
            payload,
            "chrome_sensitive_page_client_block_downgrades",
            "chrome_sensitive_page_backend_false_safe_bypasses",
            "chrome_sensitive_page_content_sent_to_models",
            "chrome_sensitive_page_searchable_content_artifacts_created",
            "chrome_sensitive_page_capture_inbox_items_created",
        )
        assert_checks_true(
            payload["connector_contract_evidence"]["chrome_sensitive_page_checks"],
            "policy_command_exit_zero",
            "schema_and_summary_counts",
            "sensitive_classes_block_or_degrade",
            "backend_never_downgrades_client_block",
            "backend_recheck_blocks_false_safe",
            "degraded_payloads_hash_only",
            "history_items_explain_safe_alternative",
        )
        self.assertEqual(payload["connector_contract_evidence"]["chrome_sensitive_page_policy_decision_count"], 8)
        self.assertEqual(payload["connector_contract_evidence"]["chrome_sensitive_page_degraded_payload_count"], 2)
        self.assertEqual(payload["connector_contract_evidence"]["chrome_sensitive_page_history_item_count"], 8)
        self.assertEqual(payload["connector_contract_evidence"]["chrome_sensitive_page_inbox_item_count"], 0)
        self.assertEqual(payload["connector_contract_evidence"]["chrome_sensitive_page_artifact_count"], 0)

        payload = assert_filtered_pass("CS-CH-005")
        assert_negative_zero(payload, "provider_credentials_exposed")

        payload = assert_filtered_pass("CS-CH-006")
        assert_negative_zero(payload, "provider_credentials_exposed")

        payload = assert_filtered_pass("CS-CH-003")
        assert_negative_zero(payload, "unauthorized_provider_calls", "provider_credentials_exposed")

        payload = assert_filtered_pass("CS-CH-002")
        assert_negative_zero(payload, "unauthorized_provider_calls", "provider_credentials_exposed")

    @unittest.skipIf(SKIP_VS2_REGRESSION_TESTS, "VS2 proof construction validates CS-CH-036 after the reusable proof is written")
    def test_connectorhub_default_deny_egress_topology_cs_ch_036(self) -> None:
        ensure_vs2_reusable_proof_current(self)
        result = run_cli(
            "scenario",
            "verify",
            "connector-contract-adapter",
            "--scenario",
            "CS-CH-036",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["summary"]["scenario_count"], 1)
        self.assertEqual(payload["summary"]["pass"], 1)
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(
            payload["summary"]["product_feature_claims"],
            CONNECTORHUB_LOCAL_FIXTURE_PRODUCT_CLAIM,
        )
        self.assertEqual(payload["scenario_results"][0]["id"], "CS-CH-036")
        self.assertEqual(payload["scenario_results"][0]["status"], "PASS")

        evidence = payload["connector_contract_evidence"]
        egress_topology_checks = evidence["egress_topology_checks"]
        self.assertTrue(all(egress_topology_checks.values()), egress_topology_checks)
        self.assertEqual(
            evidence["egress_topology_required_vs2_scenarios"],
            [
                "VS2-SEC-051",
                "VS2-SEC-052",
                "VS2-SEC-057",
                "VS2-SEC-058",
                "VS2-SEC-059",
                "VS2-SEC-063",
                "VS2-SEC-064",
            ],
        )
        self.assertEqual(evidence["egress_topology_reuse_errors"], [])
        self.assertEqual(evidence["egress_topology_read_errors"], [])

        source_fingerprint = evidence["egress_topology_source_fingerprint"]
        self.assertEqual(source_fingerprint["recorded_input_digest"], source_fingerprint["current_input_digest"])
        self.assertEqual(
            source_fingerprint["recorded_working_tree_digest"],
            source_fingerprint["current_working_tree_digest"],
        )

        topology = evidence["egress_topology_network_topology"]
        self.assertFalse(topology["host_network"])
        self.assertFalse(topology["privileged"])
        self.assertFalse(topology["published_ports"])
        service_members = topology["service_members"]
        provider_members = topology["provider_members"]
        self.assertTrue(any("-api-" in member for member in service_members), service_members)
        self.assertTrue(any("-worker-" in member for member in service_members), service_members)
        self.assertTrue(any("-tool-" in member for member in service_members), service_members)
        self.assertTrue(any("-egress-proxy-" in member for member in service_members), service_members)
        self.assertTrue(any("-egress-proxy-" in member for member in provider_members), provider_members)
        self.assertTrue(any("-provider-" in member for member in provider_members), provider_members)
        self.assertEqual(len(provider_members), 2)

        boundary_checks = evidence["egress_topology_network_boundary_checks"]
        self.assertTrue(boundary_checks["direct_http_and_socket_blocked"])
        self.assertTrue(boundary_checks["provider_zero_requests_after_direct_attempts"])
        self.assertTrue(boundary_checks["provider_reachable_from_governed_proxy"])
        self.assertTrue(boundary_checks["provider_network_membership_isolated"])
        self.assertTrue(boundary_checks["service_network_membership_expected"])

        provider_counts = evidence["egress_topology_provider_counts"]
        self.assertEqual(provider_counts["before_direct"]["requests"], 0)
        self.assertEqual(provider_counts["after_direct"]["requests"], 0)
        self.assertEqual(provider_counts["after_allowed"]["requests"], 1)
        self.assertGreaterEqual(evidence["egress_topology_sink_request_count"], 1)
        self.assertEqual(evidence["egress_topology_trap_sink_request_count"], 0)

        negative = evidence["egress_topology_negative_evidence"]
        for key in [
            "vs2_reuse_errors",
            "missing_required_vs2_rows",
            "failed_required_vs2_rows",
            "direct_http_socket_bypass_allowed",
            "provider_requests_before_direct_attempts",
            "provider_requests_after_direct_attempts",
            "default_denied_sink_calls",
            "redirect_denied_hop_trap_calls",
            "sensitive_headers_forwarded_to_denied_hop",
            "raw_credentials_exposed",
            "raw_payloads_in_audit",
            "production_topology_overclaimed",
        ]:
            self.assertEqual(negative[key], 0, key)

        self.assertEqual(payload["readiness_dimensions"]["production_tenancy_policy_egress_verified"], "NOT_VERIFIED")
        self.assertEqual(payload["negative_evidence"]["egress_topology_production_topology_overclaimed"], 0)

    def test_connectorhub_audit_correlation_cs_ch_037(self) -> None:
        result = run_cli(
            "scenario",
            "verify",
            "connector-contract-adapter",
            "--scenario",
            "CS-CH-037",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["summary"]["scenario_count"], 1)
        self.assertEqual(payload["summary"]["pass"], 1)
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["scenario_results"][0]["id"], "CS-CH-037")
        self.assertEqual(payload["scenario_results"][0]["status"], "PASS")

        evidence = payload["connector_contract_evidence"]
        audit_checks = evidence["audit_correlation_checks"]
        self.assertTrue(all(audit_checks.values()), audit_checks)
        self.assertEqual(evidence["audit_correlation_provider_internal_findings"], [])
        self.assertEqual(evidence["audit_correlation_missing_required_families"], [])
        self.assertEqual(evidence["audit_correlation_uncorrelated_event_ids"], [])
        self.assertEqual(evidence["audit_correlation_duplicate_correlation_ids"], [])
        self.assertEqual(evidence["audit_correlation_detail_leaks"], [])
        self.assertEqual(
            evidence["audit_correlation_connector_event_count"],
            evidence["audit_correlation_correlated_event_count"],
        )
        self.assertGreaterEqual(evidence["audit_correlation_connector_event_count"], 20)

        required_presence = evidence["audit_correlation_required_family_presence"]
        self.assertEqual(
            set(required_presence),
            {"action", "credential", "delivery", "evidence", "policy", "quarantine", "retry", "setup"},
        )
        for family, event_types in required_presence.items():
            self.assertTrue(event_types, family)

        for key, value in evidence["audit_correlation_negative_evidence"].items():
            self.assertEqual(value, 0, key)
        for key in [
            "audit_correlation_missing_required_event_families",
            "audit_correlation_uncorrelated_connector_events",
            "audit_correlation_duplicate_correlation_ids",
            "audit_correlation_scope_mismatches",
            "audit_correlation_raw_payload_or_secret_leaks",
            "audit_correlation_integrity_errors",
            "audit_correlation_tamper_detection_failures",
        ]:
            self.assertEqual(payload["negative_evidence"][key], 0, key)

        sample_correlations = evidence["audit_correlation_sample_correlations"]
        self.assertTrue(sample_correlations)
        for item in sample_correlations:
            self.assertTrue(item["connector_event_id"])
            self.assertTrue(item["cornerstone_audit_event_id"])
            self.assertTrue(item["affected_object_refs"])
            self.assertFalse(item["raw_payload_copied"])
            self.assertFalse(item["secret_copied"])

        tamper_errors = evidence["audit_correlation_tamper_errors"]
        self.assertTrue(tamper_errors)
        self.assertIn(
            "AUDIT_EVENT_HASH_MISMATCH",
            {error.get("code") for error in tamper_errors},
        )


if __name__ == "__main__":
    unittest.main()
