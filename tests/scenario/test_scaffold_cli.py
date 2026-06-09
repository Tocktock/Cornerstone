from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = f"{ROOT}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(
        ["cornerstone", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class ScaffoldCliTests(unittest.TestCase):
    def test_version_json(self) -> None:
        result = run_cli("version", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["schema_version"], "cs.cli.v0")
        self.assertEqual(payload["status"], "success")

    def test_ready_is_honest_not_ready(self) -> None:
        result = run_cli("ready", "--json")
        self.assertEqual(result.returncode, 4, result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "not_ready")
        self.assertTrue(payload["errors"])
        checks = {row["name"]: row["present"] for row in payload["checks"]}
        self.assertTrue(checks["fixture_corpus"])
        self.assertFalse(checks["api_runtime"])
        self.assertFalse(checks["web_runtime"])

    def test_full_scenario_list_count(self) -> None:
        result = run_cli("scenario", "list", "--set", "full", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["count"], 206)

    def test_scenario_coverage(self) -> None:
        result = run_cli("scenario", "coverage", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["full"]["count"], 206)
        self.assertEqual(payload["vs0"]["count"], 58)
        self.assertEqual(payload["verification_matrix"]["count"], 206)

    def test_vs0_scaffold_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-scaffold", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-scaffold")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["product_feature_claims"], "NOT_VERIFIED")

    def test_vs0_fixture_verify(self) -> None:
        result = run_cli(
            "scenario",
            "verify",
            "vs0-fixtures",
            "--corpus",
            "fixtures/vs0",
            "--model-provider",
            "local_test",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-fixtures")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["product_feature_claims"], "NOT_VERIFIED")
        self.assertGreaterEqual(payload["summary"]["referenced_product_scenario_count"], 10)
        self.assertEqual(payload["provider"]["name"], "local_test")
        self.assertTrue(payload["provider"]["deterministic"])
        self.assertFalse(payload["provider"]["pass_judge"])
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)
        self.assertTrue(payload["referenced_product_scenarios"])
        self.assertEqual({row["status"] for row in payload["referenced_product_scenarios"]}, {"NOT_VERIFIED"})

    def test_artifact_ingest_show_and_audit_verify(self) -> None:
        state_dir = ROOT / "tmp/test-artifact-cli"
        shutil.rmtree(state_dir, ignore_errors=True)
        try:
            ingest = run_cli(
                "artifact",
                "ingest",
                "fixtures/vs0/packs/01_artifact_basic/input.txt",
                "--state-dir",
                "tmp/test-artifact-cli",
                "--json",
            )
            self.assertEqual(ingest.returncode, 0, ingest.stdout + ingest.stderr)
            ingest_payload = json.loads(ingest.stdout)
            artifact = ingest_payload["artifact"]
            self.assertTrue(artifact["artifact_id"].startswith("art_"))
            self.assertTrue(artifact["original_storage_ref"].startswith("sha256:"))
            self.assertEqual(artifact["derived"]["status"], "ready")
            self.assertTrue(ingest_payload["evidence_refs"])
            self.assertTrue(ingest_payload["audit_refs"])

            show = run_cli(
                "artifact",
                "show",
                artifact["artifact_id"],
                "--state-dir",
                "tmp/test-artifact-cli",
                "--json",
            )
            self.assertEqual(show.returncode, 0, show.stdout + show.stderr)
            show_payload = json.loads(show.stdout)
            self.assertEqual(show_payload["artifact"]["artifact_id"], artifact["artifact_id"])
            self.assertTrue(show_payload["artifact"]["provenance"]["transformations"])
            self.assertTrue(show_payload["audit_refs"])

            audit = run_cli("audit", "verify", "--state-dir", "tmp/test-artifact-cli", "--json")
            self.assertEqual(audit.returncode, 0, audit.stdout + audit.stderr)
            audit_payload = json.loads(audit.stdout)
            self.assertEqual(audit_payload["audit_integrity"]["status"], "success")
            self.assertGreaterEqual(audit_payload["audit_integrity"]["event_count"], 2)
        finally:
            shutil.rmtree(state_dir, ignore_errors=True)

    def test_audit_verify_rejects_tampering(self) -> None:
        state_dir = ROOT / "tmp/test-audit-tamper"
        shutil.rmtree(state_dir, ignore_errors=True)
        try:
            ingest = run_cli(
                "artifact",
                "ingest",
                "fixtures/vs0/packs/01_artifact_basic/input.txt",
                "--state-dir",
                "tmp/test-audit-tamper",
                "--json",
            )
            self.assertEqual(ingest.returncode, 0, ingest.stdout + ingest.stderr)
            audit_path = state_dir / "audit/events.jsonl"
            lines = audit_path.read_text().splitlines()
            self.assertTrue(lines)
            lines[0] = lines[0].replace("artifact.ingested", "artifact.modified")
            audit_path.write_text("\n".join(lines) + "\n")

            audit = run_cli("audit", "verify", "--state-dir", "tmp/test-audit-tamper", "--json")
            self.assertEqual(audit.returncode, 5, audit.stdout + audit.stderr)
            audit_payload = json.loads(audit.stdout)
            self.assertEqual(audit_payload["status"], "failed")
            self.assertEqual(audit_payload["errors"][0]["code"], "CS_AUDIT_INTEGRITY_FAILED")
        finally:
            shutil.rmtree(state_dir, ignore_errors=True)

    def test_prompt_injection_ingest_records_policy_denial(self) -> None:
        state_dir = ROOT / "tmp/test-prompt-injection"
        shutil.rmtree(state_dir, ignore_errors=True)
        try:
            result = run_cli(
                "artifact",
                "ingest",
                "fixtures/vs0/packs/10_prompt_injection/input.txt",
                "--state-dir",
                "tmp/test-prompt-injection",
                "--json",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            safety = payload["artifact"]["safety"]
            self.assertTrue(safety["untrusted_evidence"])
            self.assertTrue(safety["unsafe_instruction_detected"])
            self.assertEqual(safety["tool_calls_created"], 0)
            self.assertEqual(safety["action_cards_created_from_untrusted_artifact"], 0)
            self.assertEqual(safety["external_http_calls"], 0)
            self.assertFalse(safety["authority_expanded"])
            self.assertTrue(payload["policy_decision_refs"])
            self.assertGreaterEqual(len(payload["audit_refs"]), 2)
        finally:
            shutil.rmtree(state_dir, ignore_errors=True)

    def test_search_query_and_evidence_bundle(self) -> None:
        state_dir = ROOT / "tmp/test-search-evidence"
        shutil.rmtree(state_dir, ignore_errors=True)
        try:
            ingest = run_cli(
                "artifact",
                "ingest",
                "fixtures/vs0/packs/01_artifact_basic/input.txt",
                "--state-dir",
                "tmp/test-search-evidence",
                "--json",
            )
            self.assertEqual(ingest.returncode, 0, ingest.stdout + ingest.stderr)

            search = run_cli(
                "search",
                "query",
                "alpha-evidence-anchor",
                "--state-dir",
                "tmp/test-search-evidence",
                "--json",
            )
            self.assertEqual(search.returncode, 0, search.stdout + search.stderr)
            search_payload = json.loads(search.stdout)
            snapshot = search_payload["search_snapshot"]
            self.assertEqual(snapshot["query"], "alpha-evidence-anchor")
            self.assertEqual(snapshot["result_count"], 1)
            self.assertIn("alpha-evidence-anchor", snapshot["results"][0]["snippet"])
            self.assertTrue(any(ref.startswith("search_snapshot:") for ref in search_payload["evidence_refs"]))
            self.assertTrue(any(ref.startswith("artifact:") for ref in search_payload["evidence_refs"]))
            self.assertTrue(search_payload["audit_refs"])

            bundle = run_cli(
                "evidence",
                "bundle",
                "create",
                "--search-snapshot-id",
                snapshot["search_snapshot_id"],
                "--state-dir",
                "tmp/test-search-evidence",
                "--json",
            )
            self.assertEqual(bundle.returncode, 0, bundle.stdout + bundle.stderr)
            bundle_payload = json.loads(bundle.stdout)
            bundle_id = bundle_payload["evidence_bundle"]["evidence_bundle_id"]
            item = bundle_payload["evidence_bundle"]["evidence_items"][0]
            self.assertTrue(item["original_storage_ref"].startswith("sha256:"))
            self.assertTrue(item["derived_text_ref"].startswith("derived/"))
            self.assertTrue(any(ref.startswith("evidence_bundle:") for ref in bundle_payload["evidence_refs"]))
            self.assertTrue(bundle_payload["audit_refs"])

            bundle_show = run_cli("evidence", "bundle", "show", bundle_id, "--state-dir", "tmp/test-search-evidence", "--json")
            self.assertEqual(bundle_show.returncode, 0, bundle_show.stdout + bundle_show.stderr)
            bundle_show_payload = json.loads(bundle_show.stdout)
            self.assertTrue(bundle_show_payload["audit_refs"])

            view = run_cli("evidence", "view", bundle_id, "--state-dir", "tmp/test-search-evidence", "--json")
            self.assertEqual(view.returncode, 0, view.stdout + view.stderr)
            view_payload = json.loads(view.stdout)
            viewer_item = view_payload["evidence_viewer"]["viewer_items"][0]
            self.assertEqual(viewer_item["original"]["storage_ref"], item["original_storage_ref"])
            self.assertEqual(viewer_item["derived"]["text_ref"], item["derived_text_ref"])
            self.assertIn("alpha-evidence-anchor", viewer_item["derived"]["text_preview"])
            self.assertTrue(view_payload["audit_refs"])

            claim = run_cli(
                "claim",
                "create",
                "--evidence-bundle-id",
                bundle_id,
                "--statement",
                "The alpha evidence anchor was present in the ingested fixture.",
                "--state-dir",
                "tmp/test-search-evidence",
                "--json",
            )
            self.assertEqual(claim.returncode, 0, claim.stdout + claim.stderr)
            claim_payload = json.loads(claim.stdout)
            self.assertEqual(claim_payload["claim"]["evidence_bundle"]["evidence_bundle_id"], bundle_id)
            self.assertEqual(claim_payload["claim"]["evidence_bundle"]["search_snapshot_id"], snapshot["search_snapshot_id"])
            self.assertIn(f"artifact:{item['artifact_id']}", claim_payload["claim"]["evidence_bundle"]["artifact_refs"])
            self.assertTrue(claim_payload["audit_refs"])

            artifact_show = run_cli(
                "artifact",
                "show",
                item["artifact_id"],
                "--state-dir",
                "tmp/test-search-evidence",
                "--json",
            )
            self.assertEqual(artifact_show.returncode, 0, artifact_show.stdout + artifact_show.stderr)
            artifact_show_payload = json.loads(artifact_show.stdout)
            self.assertEqual(artifact_show_payload["artifact"]["related_claims"][0]["claim_id"], claim_payload["claim"]["claim_id"])
            self.assertEqual(artifact_show_payload["artifact"]["related_missions"], [])
            self.assertIn("alpha-evidence-anchor", artifact_show_payload["artifact"]["derived_text_preview"])

            denied = run_cli(
                "evidence",
                "bundle",
                "show",
                bundle_id,
                "--state-dir",
                "tmp/test-search-evidence",
                "--owner-id",
                "other-user",
                "--json",
            )
            self.assertEqual(denied.returncode, 6, denied.stdout + denied.stderr)
            denied_payload = json.loads(denied.stdout)
            self.assertEqual(denied_payload["errors"][0]["code"], "CS_SCOPE_DENIED")
        finally:
            shutil.rmtree(state_dir, ignore_errors=True)

    def test_vs0_artifact_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-artifacts", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-artifacts")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 5)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_ARTIFACTS_ONLY")
        self.assertEqual({row["status"] for row in payload["scenario_results"]}, {"PASS"})
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {f"CS-ARCH-00{index}" for index in range(1, 6)})
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_security_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-security", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("sk-test-", result.stdout)
        self.assertNotIn("ghp_", result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-security")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 5)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_SECURITY_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {"CS-ARCH-006", "CS-ARCH-007", "CS-SEC-007", "CS-SEC-008", "CS-REG-013"},
        )
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_search_evidence_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-search-evidence", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-search-evidence")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 3)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_SEARCH_EVIDENCE_ONLY")
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {"CS-ARCH-008", "CS-ARCH-009", "CS-UND-001"})
        self.assertEqual(payload["search_evidence"]["result_count"], 1)
        self.assertIn("alpha-evidence-anchor", payload["search_evidence"]["first_snippet"])
        self.assertTrue(payload["search_evidence"]["original_storage_ref"].startswith("sha256:"))
        self.assertTrue(payload["search_evidence"]["derived_text_ref"].startswith("derived/"))
        self.assertTrue(payload["search_evidence"]["claim_id"].startswith("claim_"))
        self.assertTrue(payload["search_evidence"]["evidence_viewer_id"].startswith("viewer_"))
        self.assertLessEqual(payload["search_evidence"]["first_use_duration_ms"], 5000)

    def test_vs0_search_understanding_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-search-understanding", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-search-understanding")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 2)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_SEARCH_UNDERSTANDING_ONLY")
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {"CS-UND-002", "CS-UND-003"})
        self.assertTrue(
            any(reason["type"] == "semantic_alias" for reason in payload["search_understanding"]["semantic_match_reasons"])
        )
        self.assertEqual(payload["search_understanding"]["personal_cross_org_result_count"], 0)
        self.assertEqual(payload["search_understanding"]["personal_cross_project_result_count"], 0)
        self.assertEqual(payload["search_understanding"]["organization_cross_personal_result_count"], 0)
        self.assertEqual(payload["search_understanding"]["organization_cross_project_result_count"], 0)
        self.assertEqual(payload["search_understanding"]["project_cross_personal_result_count"], 0)
        self.assertEqual(payload["search_understanding"]["project_cross_org_result_count"], 0)
        self.assertEqual(payload["search_understanding"]["project_result_count"], 1)
        self.assertEqual(payload["search_understanding"]["same_content_personal_scope"]["owner_id"], "local-user")
        self.assertEqual(payload["search_understanding"]["same_content_organization_scope"]["owner_id"], "local-org")
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_namespace_isolation_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-namespace-isolation", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-namespace-isolation")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 2)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_NAMESPACE_ISOLATION_ONLY")
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {"CS-NS-001", "CS-NS-003"})
        self.assertGreaterEqual(payload["namespace_evidence"]["context_record_scope_count"], 7)
        self.assertEqual(payload["namespace_evidence"]["personal_artifact_scope"]["owner_id"], "local-user")
        self.assertEqual(payload["namespace_evidence"]["personal_artifact_scope"]["namespace_id"], "personal")
        self.assertEqual(payload["namespace_evidence"]["organization_artifact_scope"]["owner_id"], "local-org")
        self.assertEqual(payload["namespace_evidence"]["organization_artifact_scope"]["namespace_id"], "organization")
        self.assertEqual(payload["namespace_evidence"]["organization_cross_personal_result_count"], 0)
        self.assertEqual(payload["namespace_evidence"]["personal_cross_organization_result_count"], 0)
        self.assertEqual(payload["namespace_evidence"]["cross_scope_evidence_attempts_denied"], 2)
        self.assertEqual(payload["transcripts"]["org_create_bundle_from_personal_snapshot"]["stdout_json"]["owner_id"], "local-org")
        self.assertEqual(payload["transcripts"]["org_claim_from_personal_bundle"]["stdout_json"]["owner_id"], "local-org")
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_audit_ledger_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-audit-ledger", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-audit-ledger")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 1)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_AUDIT_LEDGER_ONLY")
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {"CS-SEC-006"})
        evidence = payload["audit_evidence"]
        self.assertGreaterEqual(evidence["clean_audit_event_count"], 5)
        self.assertFalse(evidence["missing_event_types"])
        self.assertTrue(evidence["event_scopes_complete"])
        self.assertTrue(evidence["event_hashes_present"])
        self.assertTrue(evidence["event_details_present"])
        self.assertEqual(evidence["tamper_detection_exit_code"], 5)
        self.assertTrue(
            any(error["code"] == "AUDIT_EVENT_HASH_MISMATCH" for error in evidence["tamper_detection_errors"])
        )
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_universal_core_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-universal-core", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-universal-core")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 1)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_UNIVERSAL_CORE_ONLY")
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {"CS-REG-004"})
        evidence = payload["universal_core_evidence"]
        self.assertEqual(evidence["fixture"], "fixtures/vs0/packs/01_artifact_basic/input.txt")
        self.assertFalse(evidence["found_logistics_terms"])
        self.assertEqual(evidence["search_result_count"], 1)
        self.assertTrue(evidence["evidence_bundle_id"].startswith("evb_"))
        self.assertTrue(evidence["claim_id"].startswith("claim_"))
        self.assertTrue(any(ref.startswith("artifact:") for ref in evidence["claim_artifact_refs"]))
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_claim_evidence_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-claim-evidence", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-claim-evidence")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 2)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_CLAIM_EVIDENCE_ONLY")
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {"CS-CLAIM-006", "CS-CLAIM-007"})
        evidence = payload["claim_evidence"]
        self.assertEqual(evidence["unsupported_claim_trust_state"], "draft")
        self.assertEqual(evidence["unsupported_claim_show_evidence_refs"], [f"claim:{evidence['unsupported_claim_id']}"])
        self.assertEqual(evidence["unsupported_approval_exit_code"], 4)
        self.assertIn("CS_CLAIM_EVIDENCE_REQUIRED", evidence["unsupported_approval_error_codes"])
        self.assertTrue(evidence["unsupported_resolution_path"])
        self.assertEqual(evidence["evidence_claim_trust_state"], "evidence_backed")
        self.assertEqual(evidence["approved_claim_status"], "approved")
        self.assertEqual(evidence["approved_claim_trust_state"], "approved")
        self.assertFalse(evidence["approved_claim_authority"]["can_drive_autonomous_action"])
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_security_policy_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-security-policy", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-security-policy")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 2)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_SECURITY_POLICY_ONLY")
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {"CS-SEC-002", "CS-SEC-003"})
        evidence = payload["security_policy_evidence"]
        self.assertEqual(evidence["egress_policy"], "default_egress_deny")
        self.assertEqual(evidence["egress_exit_code"], 8)
        self.assertEqual(evidence["egress_external_http_calls"], 0)
        self.assertTrue(evidence["egress_resolution_path"])
        self.assertEqual(set(evidence["sandbox_cases"]), {"sandbox_environment", "sandbox_filesystem", "sandbox_host", "sandbox_shell"})
        self.assertEqual(set(evidence["sandbox_exit_codes"].values()), {8})
        self.assertEqual(set(evidence["sandbox_policies"].values()), {"declared_sandbox_capability_required"})
        self.assertEqual(set(evidence["sandbox_host_operations_executed"].values()), {0})
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_same_content_isolation_across_scopes(self) -> None:
        state_dir = ROOT / "tmp/test-same-content-scope"
        shutil.rmtree(state_dir, ignore_errors=True)
        try:
            personal = run_cli(
                "artifact",
                "ingest",
                "fixtures/vs0/packs/01_artifact_basic/input.txt",
                "--state-dir",
                "tmp/test-same-content-scope",
                "--json",
            )
            self.assertEqual(personal.returncode, 0, personal.stdout + personal.stderr)
            org = run_cli(
                "artifact",
                "ingest",
                "fixtures/vs0/packs/01_artifact_basic/input.txt",
                "--state-dir",
                "tmp/test-same-content-scope",
                "--owner-id",
                "local-org",
                "--namespace-id",
                "organization",
                "--workspace-id",
                "ops",
                "--json",
            )
            self.assertEqual(org.returncode, 0, org.stdout + org.stderr)
            personal_payload = json.loads(personal.stdout)
            org_payload = json.loads(org.stdout)
            self.assertEqual(personal_payload["artifact"]["artifact_id"], org_payload["artifact"]["artifact_id"])

            personal_search = run_cli("search", "query", "alpha-evidence-anchor", "--state-dir", "tmp/test-same-content-scope", "--json")
            self.assertEqual(personal_search.returncode, 0, personal_search.stdout + personal_search.stderr)
            personal_search_payload = json.loads(personal_search.stdout)
            self.assertEqual(personal_search_payload["search_snapshot"]["result_count"], 1)
            self.assertEqual(personal_search_payload["search_snapshot"]["results"][0]["scope"]["owner_id"], "local-user")

            org_search = run_cli(
                "search",
                "query",
                "alpha-evidence-anchor",
                "--state-dir",
                "tmp/test-same-content-scope",
                "--owner-id",
                "local-org",
                "--namespace-id",
                "organization",
                "--workspace-id",
                "ops",
                "--json",
            )
            self.assertEqual(org_search.returncode, 0, org_search.stdout + org_search.stderr)
            org_search_payload = json.loads(org_search.stdout)
            self.assertEqual(org_search_payload["search_snapshot"]["result_count"], 1)
            self.assertEqual(org_search_payload["search_snapshot"]["results"][0]["scope"]["owner_id"], "local-org")
        finally:
            shutil.rmtree(state_dir, ignore_errors=True)

    def test_scenario_gate_rejects_report_level_errors(self) -> None:
        report_path = ROOT / "reports/scenario/test-failed-report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "errors": [{"code": "TEST_ERROR", "message": "synthetic"}],
                    "scenario_results": [
                        {
                            "id": "TEST-INFRA-001",
                            "type": "MUST_PASS",
                            "status": "PASS",
                            "owner": "AI",
                            "evidence": ["synthetic"],
                            "notes": "synthetic",
                        }
                    ],
                }
            )
            + "\n"
        )
        try:
            result = run_cli("scenario", "gate", "reports/scenario/test-failed-report.json", "--json")
        finally:
            report_path.unlink(missing_ok=True)

        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["blocking_count"], 0)
        self.assertEqual(payload["errors"][0]["code"], "CS_SCENARIO_REPORT_FAILED")


if __name__ == "__main__":
    unittest.main()
