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

    def test_ready_reports_local_runtime_without_production_overclaim(self) -> None:
        result = run_cli("ready", "--json")
        self.assertEqual(result.returncode, 0, result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "success")
        self.assertFalse(payload["errors"])
        checks = {row["name"]: row["present"] for row in payload["checks"]}
        self.assertTrue(checks["fixture_corpus"])
        self.assertTrue(checks["api_runtime"])
        self.assertTrue(checks["web_runtime"])
        self.assertTrue(payload["readiness"]["local_scenario_ready"])
        self.assertTrue(payload["readiness"]["vs0_runtime_ready"])
        self.assertFalse(payload["readiness"]["production_release_ready"])
        self.assertTrue(payload["readiness"]["human_required"])
        self.assertEqual(payload["readiness"]["real_external_http_calls"], 0)
        runtime_report = payload["readiness"]["last_successful_runtime_scenario"]
        self.assertEqual(runtime_report["scenario_set"], "vs0-product-runtime")
        self.assertEqual(runtime_report["path"], "reports/scenario/vs0-product-runtime-2026-06-11.json")
        self.assertEqual(runtime_report["gate_status"], "pass")
        self.assertEqual(runtime_report["blocking"], 0)
        acceptance_report = payload["readiness"]["last_successful_acceptance_scenario"]
        if acceptance_report["path"] is not None:
            self.assertEqual(acceptance_report["scenario_set"], "vs0-runtime-acceptance")
            self.assertEqual(acceptance_report["gate_status"], "pass")
            self.assertEqual(payload["readiness"]["acceptance_gate"]["status"], "pass")

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

    def test_vs0_product_runtime_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-product-runtime", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-product-runtime")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 12)
        self.assertEqual(payload["summary"]["human_required"], 2)
        self.assertFalse(payload["runtime_evidence"]["readiness"]["production_release_ready"])
        self.assertEqual(payload["negative_evidence"]["real_external_http_calls"], 0)

    def test_vs0_runtime_acceptance_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-runtime-acceptance", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-runtime-acceptance")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 7)
        self.assertEqual(payload["summary"]["human_required"], 2)
        self.assertEqual(payload["browser_proof"]["status"], "passed")
        self.assertGreater(payload["browser_proof"]["screenshot_bytes"], 0)
        self.assertTrue(all(payload["browser_proof"]["surface_presence"].values()))
        self.assertFalse(payload["acceptance_evidence"]["readiness"]["production_release_ready"])
        impact = payload["acceptance_evidence"]["dry_run_expected_impact"]
        self.assertEqual(impact["expected_connector_calls"], 1)
        self.assertEqual(impact["mock_connector_calls"], 1)
        self.assertEqual(impact["real_external_http_calls"], 0)
        self.assertNotIn("external_calls", impact)
        self.assertEqual(payload["release_evidence_package"]["status"], "success")
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_evux_quickstart_verify(self) -> None:
        output_path = ROOT / "tmp/test-vs0-evux-quickstart.json"
        if output_path.exists():
            output_path.unlink()
        result = run_cli(
            "quickstart",
            "verify",
            "vs0-evux",
            "--json",
            "--output",
            "tmp/test-vs0-evux-quickstart.json",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-evux")
        self.assertEqual(payload["status"], "success")
        ids = payload["generated_ids"]
        self.assertTrue(ids["artifact_id"].startswith("art_"))
        self.assertTrue(ids["search_snapshot_id"].startswith("search_"))
        self.assertTrue(ids["evidence_bundle_id"].startswith("evb_"))
        self.assertTrue(ids["claim_id"].startswith("claim_"))
        self.assertTrue(ids["action_id"].startswith("action_"))
        self.assertEqual(payload["final_audit_verification"]["status"], "success")
        self.assertEqual(payload["negative_evidence"]["commands_failed"], 0)
        self.assertEqual(payload["negative_evidence"]["zero_evidence_claim_approved"], 0)
        self.assertEqual(payload["negative_evidence"]["real_external_http_calls"], 0)
        self.assertEqual(payload["negative_evidence"]["mock_connector_calls"], 1)

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

    def test_vs0_regression_guardrails_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-regression-guardrails", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-regression-guardrails")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 3)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_REGRESSION_GUARDRAILS_ONLY")
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {"CS-REG-016", "CS-REG-017", "CS-REG-018"})
        summaries = payload["component_summaries"]
        self.assertEqual(summaries["claim_evidence"]["trust_states"], {"unsupported": "draft", "evidence_backed": "evidence_backed", "approved": "approved"})
        self.assertIn("claim.approved", summaries["audit_ledger"]["event_types"])
        self.assertIn("policy.egress.denied", summaries["audit_ledger"]["event_types"])
        self.assertEqual(summaries["security_policy"]["egress_external_http_calls"], 0)
        self.assertEqual(set(summaries["security_policy"]["sandbox_cases"]), {"sandbox_environment", "sandbox_filesystem", "sandbox_host", "sandbox_shell"})
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_briefing_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-briefing", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-briefing")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 4)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_BRIEFING_ONLY")
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {"CS-PROD-004", "CS-UND-005", "CS-CLAIM-002", "CS-SEC-001"})
        evidence = payload["briefing_evidence"]
        self.assertEqual(evidence["search_result_count"], 1)
        self.assertEqual(evidence["brief_status"], "evidence_backed")
        self.assertGreaterEqual(evidence["key_point_count"], 1)
        self.assertGreaterEqual(evidence["evidence_link_count"], 1)
        self.assertGreaterEqual(evidence["uncertainty_count"], 1)
        self.assertGreaterEqual(evidence["recommended_next_step_count"], 1)
        self.assertFalse(evidence["ontology"]["preconfigured_ontology_required"])
        self.assertFalse(evidence["ontology"]["ontology_suggestions_required_before_brief"])
        self.assertLessEqual(evidence["first_use_duration_ms"], 5000)
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_mission_action_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-mission-action", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-mission-action")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 16)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_MISSION_ACTION_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {
                "CS-CLAIM-010",
                "CS-AUTO-001",
                "CS-AUTO-003",
                "CS-AUTO-004",
                "CS-AUTO-005",
                "CS-AUTO-006",
                "CS-AUTO-007",
                "CS-AUTO-008",
                "CS-AUTO-009",
                "CS-AUTO-010",
                "CS-AUTO-011",
                "CS-REG-002",
                "CS-REG-003",
                "CS-REG-011",
                "CS-REG-012",
                "CS-AUTO-020",
            },
        )
        evidence = payload["mission_action_evidence"]
        self.assertEqual(evidence["available_modes"], ["assist", "autopilot", "locked", "manual"])
        self.assertTrue(all(evidence["mission_contract_fields_present"].values()))
        self.assertEqual(evidence["low_policy"]["policy"], "low_risk_autopilot_allowed")
        self.assertEqual(evidence["high_policy"]["policy"], "high_risk_action_requires_approval")
        self.assertEqual(evidence["out_of_contract_policy"]["policy"], "mission_contract_action_scope")
        self.assertEqual(evidence["manual_policy"]["policy"], "workspace_mode_no_autonomous_execution")
        self.assertEqual(evidence["locked_policy"]["policy"], "workspace_mode_locked")
        self.assertEqual(evidence["high_execute_before_approval_exit_code"], 8)
        self.assertEqual(evidence["high_approval_status"], "approved")
        self.assertEqual(evidence["low_result"]["external_http_calls"], 0)
        self.assertEqual(evidence["high_result"]["mock_connector_calls"], 1)
        self.assertEqual(evidence["direct_write_policy"]["policy"], "workflow_action_path_required")
        self.assertIn("action.executed", evidence["audit_event_types"])
        self.assertIn("connector.direct_write.denied", evidence["audit_event_types"])
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_detail_surfaces_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-detail-surfaces", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-detail-surfaces")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 5)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_DETAIL_SURFACES_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {"CS-UND-004", "CS-CLAIM-005", "CS-CLAIM-008", "CS-NS-002", "CS-SEC-005"},
        )
        evidence = payload["detail_surface_evidence"]
        self.assertIn("personal / default", evidence["workspace_labels"]["personal"])
        self.assertIn("organization / ops", evidence["workspace_labels"]["organization"])
        self.assertFalse(evidence["workspace_boundaries"]["personal"]["implicit_cross_namespace_context"])
        self.assertFalse(evidence["workspace_boundaries"]["organization"]["implicit_cross_namespace_context"])
        self.assertTrue(evidence["artifact_related_claim_ids"])
        self.assertTrue(evidence["artifact_related_mission_ids"])
        self.assertEqual(evidence["artifact_derived_status"], "ready")
        self.assertTrue(evidence["artifact_original_storage_ref"].startswith("sha256:"))
        self.assertEqual(
            evidence["trust_states"],
            {"draft": "draft", "evidence_backed": "evidence_backed", "approved": "approved"},
        )
        self.assertTrue(evidence["trust_authority"]["draft"]["can_be_approved"] is False)
        self.assertTrue(evidence["trust_authority"]["evidence_backed"]["can_be_approved"])
        self.assertTrue(evidence["trust_authority"]["approved"]["can_publish_shared_truth"])
        self.assertTrue(evidence["evidence_viewer_id"].startswith("viewer_"))
        self.assertEqual(evidence["evidence_viewer_item_count"], 1)
        self.assertIn("alpha-evidence-anchor", evidence["evidence_viewer_first_item"]["derived"]["text_preview"])
        self.assertEqual(
            {example["error_code"] for example in evidence["denial_examples"].values()},
            {"CS_EGRESS_DENIED", "CS_SANDBOX_ACCESS_DENIED", "CS_CLAIM_EVIDENCE_REQUIRED", "CS_ACTION_POLICY_DENIED"},
        )
        self.assertTrue(all(example["resolution_path"] for example in evidence["denial_examples"].values()))
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_conversation_onboarding_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-conversation-onboarding", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-conversation-onboarding")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 5)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_CONVERSATION_ONBOARDING_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {"CS-PROD-005", "CS-CLAIM-001", "CS-CLAIM-003", "CS-CLAIM-004", "CS-CLAIM-009"},
        )
        evidence = payload["conversation_evidence"]
        self.assertTrue(evidence["conversation_id"].startswith("conv_"))
        self.assertTrue(evidence["source_artifact_id"].startswith("art_"))
        self.assertEqual(evidence["source_artifact_source_type"], "conversation_turn")
        self.assertEqual(evidence["search_result_count"], 1)
        self.assertTrue(evidence["evidence_bundle_id"].startswith("evb_"))
        self.assertTrue(evidence["brief_id"].startswith("brief_"))
        self.assertEqual(evidence["brief_status"], "evidence_backed")
        self.assertTrue(evidence["promoted_claim_id"].startswith("claim_"))
        self.assertEqual(evidence["promoted_claim_trust_state"], "evidence_backed")
        self.assertEqual(evidence["promoted_claim_source_conversation"]["conversation_id"], evidence["conversation_id"])
        self.assertEqual(evidence["promoted_claim_source_conversation"]["source_artifact_ref"], f"artifact:{evidence['source_artifact_id']}")
        self.assertEqual(evidence["promoted_claim_provenance"]["created_from"], "conversation.promote")
        self.assertEqual(
            set(evidence["suggested_output_types"]),
            {"Action Card", "Claim", "Knowledge Capsule", "Memory", "Mission Card", "Playbook Candidate"},
        )
        self.assertEqual(evidence["forced_suggestion_count"], 0)
        self.assertFalse(evidence["required_setup"]["connector_setup"])
        self.assertFalse(evidence["required_setup"]["model_provider_setup"])
        self.assertFalse(evidence["required_setup"]["ontology_setup"])
        self.assertEqual(evidence["unsupported_answer_label"], "insufficient_evidence")
        self.assertFalse(evidence["unsupported_answer_presented_as_fact"])
        self.assertEqual(evidence["unsupported_answer_supporting_result_count"], 0)
        self.assertIn("budget", evidence["unsupported_answer_meaningful_question_terms"])
        self.assertEqual(evidence["unsupported_answer_matched_terms"], ["is"])
        self.assertLessEqual(evidence["first_use_duration_ms"], 5000)
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_product_domain_readiness_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-product-domain-readiness", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-product-domain-readiness")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 3)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_PRODUCT_DOMAIN_READINESS_ONLY")
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {"CS-PROD-001", "CS-PROD-003", "CS-AUTO-002"})
        evidence = payload["product_domain_readiness_evidence"]
        self.assertEqual(evidence["walkthrough_product_name"], "CornerStone")
        self.assertTrue(evidence["walkthrough_one_service"])
        self.assertFalse(evidence["daily_user_requires_subsystem_knowledge"])
        self.assertEqual(set(evidence["walkthrough_navigation"]), {"actions", "artifacts", "claims", "home", "search"})
        self.assertEqual(evidence["domain_count"], 3)
        self.assertTrue(all(row["ok"] for row in evidence["domain_evidence"]))
        self.assertEqual({row["key"] for row in evidence["domain_evidence"]}, {"hiring_review", "home_maintenance", "research_review"})
        self.assertEqual(evidence["initial_workspace_mode"], "assist")
        readiness = evidence["readiness"]
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["recommendation"], "recommend_autopilot")
        self.assertEqual(readiness["recommended_mode"], "autopilot")
        self.assertTrue(readiness["mission_contract_required"])
        self.assertGreaterEqual(readiness["signals"]["evidence_backed_brief_count"], 1)
        self.assertGreaterEqual(readiness["signals"]["optional_suggestion_count"], 1)
        self.assertGreaterEqual(readiness["signals"]["mission_contract_count"], 1)
        self.assertGreaterEqual(readiness["signals"]["successful_internal_task_count"], 1)
        self.assertGreaterEqual(readiness["signals"]["successful_playbook_count"], 1)
        self.assertEqual(evidence["readiness_action_result"]["external_http_calls"], 0)
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_product_loop_identity_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-product-loop-identity", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-product-loop-identity")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 2)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_PRODUCT_LOOP_IDENTITY_ONLY")
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {"CS-PROD-002", "CS-REG-001"})
        evidence = payload["product_loop_evidence"]
        self.assertEqual(evidence["walkthrough_product_name"], "CornerStone")
        self.assertIn("Learn", evidence["walkthrough_first_run_path"])
        expected_surfaces = {
            "conversation",
            "artifact",
            "search",
            "evidence_bundle",
            "evidence_viewer",
            "brief",
            "claim",
            "approved_claim",
            "mission",
            "action_card",
            "action_result",
            "memory",
            "learning",
            "audit",
        }
        self.assertEqual(set(evidence["present_surfaces"]), expected_surfaces)
        self.assertEqual(evidence["missing_surfaces"], [])
        self.assertEqual(evidence["brief_status"], "evidence_backed")
        self.assertEqual(evidence["approved_claim_trust_state"], "approved")
        self.assertEqual(evidence["memory_status"], "owner_approved")
        self.assertEqual(evidence["memory_truth_foundation"], "archive_evidence")
        self.assertEqual(evidence["action_policy"], "low_risk_autopilot_allowed")
        self.assertEqual(evidence["action_result_status"], "success")
        self.assertEqual(evidence["learning_status"], "recorded")
        self.assertFalse(evidence["learning_changes_user_or_org_truth"])
        self.assertGreaterEqual(evidence["audit_event_count"], 1)
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_memory_truth_boundary_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-memory-truth-boundary", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-memory-truth-boundary")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 1)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_MEMORY_TRUTH_BOUNDARY_ONLY")
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {"CS-REG-005"})
        evidence = payload["memory_truth_evidence"]
        self.assertEqual(evidence["search_result_count"], 1)
        self.assertGreaterEqual(evidence["evidence_item_count"], 1)
        self.assertEqual(evidence["owner_memory_status"], "owner_approved")
        self.assertEqual(evidence["owner_memory_truth_foundation"], "archive_evidence")
        self.assertEqual(evidence["raw_memory_status"], "raw_agent_memory")
        self.assertFalse(evidence["raw_memory_canonical"])
        self.assertEqual(evidence["conflict_selected_truth_foundation"], "archive_evidence")
        self.assertFalse(evidence["conflict_raw_memory_used_as_truth"])
        self.assertEqual(evidence["conflict_answer_based_on"], "archive_evidence")
        self.assertGreaterEqual(evidence["audit_event_count"], 1)
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_vs0_tenant_security_boundary_verify(self) -> None:
        result = run_cli("scenario", "verify", "vs0-tenant-security-boundary", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "vs0-tenant-security-boundary")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 3)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_VS0_TENANT_SECURITY_BOUNDARY_ONLY")
        self.assertEqual({row["id"] for row in payload["scenario_results"]}, {"CS-NS-004", "CS-SEC-004", "CS-REG-006"})
        evidence = payload["tenant_security_evidence"]
        self.assertEqual(evidence["promoted_memory_scope"]["owner_id"], "local-org")
        self.assertEqual(evidence["promoted_memory_scope"]["namespace_id"], "organization")
        self.assertEqual(evidence["promotion_mode"], "copy_with_provenance")
        self.assertEqual(evidence["promotion_policy"], "local_rbac_abac_matrix")
        self.assertEqual(evidence["pre_promotion_answer_status"], "insufficient_evidence")
        self.assertEqual(evidence["pre_promotion_used_memory_refs"], [])
        self.assertEqual(evidence["direct_cross_scope_read_exit_code"], 6)
        self.assertEqual(evidence["post_promotion_answer_status"], "answered")
        self.assertTrue(evidence["post_promotion_used_promoted_memory"])
        self.assertEqual(evidence["post_promotion_used_memory_refs"], [f"memory:{evidence['promoted_memory_id']}"])
        self.assertEqual(evidence["access_matrix_case_count"], 7)
        self.assertEqual(evidence["access_allow_count"], 3)
        self.assertEqual(evidence["access_deny_count"], 4)
        self.assertIn("namespace.promotion.created", evidence["event_types"])
        self.assertIn("policy.access.evaluated", evidence["event_types"])
        self.assertIn("memory.answer.insufficient_evidence", evidence["event_types"])
        self.assertIn("memory.answer.created", evidence["event_types"])
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

    def test_full_claim_collaboration_verify(self) -> None:
        result = run_cli("scenario", "verify", "full-claim-collaboration", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "full-claim-collaboration")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 4)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_FULL_CLAIM_COLLABORATION_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {"CS-CLAIM-011", "CS-CLAIM-012", "CS-CLAIM-013", "CS-CLAIM-014"},
        )
        evidence = payload["claim_collaboration_evidence"]
        self.assertEqual(evidence["claim_trust_state"], "approved")
        self.assertEqual(evidence["capsule_trust_state"], "approved")
        self.assertEqual(evidence["capsule_freshness_status"], "current")
        self.assertGreaterEqual(evidence["capsule_evidence_ref_count"], 1)
        self.assertTrue(all(evidence["decision_required_fields"].values()))
        self.assertGreaterEqual(evidence["decision_action_count"], 1)
        self.assertGreaterEqual(evidence["decision_learning_history_count"], 1)
        self.assertEqual(evidence["correction_source_type"], "evidence_bundle")
        self.assertTrue(evidence["correction_provenance_preserved"])
        self.assertGreaterEqual(evidence["correction_history_count"], 1)
        self.assertEqual(evidence["share_trust_state"], "approved")
        self.assertTrue(evidence["share_visibility"]["trust_state_visible"])
        self.assertTrue(evidence["share_visibility"]["evidence_visible"])
        self.assertTrue(evidence["share_visibility"]["owner_visible"])
        self.assertTrue(evidence["share_visibility"]["scope_visible"])
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_full_memory_wiki_verify(self) -> None:
        result = run_cli("scenario", "verify", "full-memory-wiki", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "full-memory-wiki")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 18)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_FULL_MEMORY_WIKI_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {
                "CS-MEM-001",
                "CS-MEM-002",
                "CS-MEM-003",
                "CS-MEM-004",
                "CS-MEM-005",
                "CS-MEM-006",
                "CS-MEM-007",
                "CS-MEM-008",
                "CS-MEM-009",
                "CS-MEM-010",
                "CS-MEM-011",
                "CS-MEM-012",
                "CS-MEM-013",
                "CS-MEM-014",
                "CS-MEM-015",
                "CS-MEM-016",
                "CS-MEM-017",
                "CS-MEM-018",
            },
        )
        evidence = payload["memory_wiki_evidence"]
        self.assertEqual(evidence["personal_search_result_count"], 1)
        self.assertEqual(evidence["correction_search_result_count"], 1)
        self.assertEqual(evidence["org_search_result_count"], 1)
        self.assertEqual(evidence["stale_search_result_count"], 1)
        self.assertEqual(evidence["answer_before_statement"].count("Monday"), 1)
        self.assertEqual(evidence["answer_after_statement"].count("Friday"), 1)
        self.assertEqual(evidence["conflict_selected_truth_foundation"], "archive_evidence")
        self.assertEqual(evidence["corrected_memory_freshness"]["status"], "needs_review")
        self.assertTrue(evidence["corrected_memory_freshness"]["warning_visible"])
        self.assertEqual(evidence["quarantine_status"], "quarantined")
        self.assertGreaterEqual(evidence["export_entry_count"], 1)
        self.assertGreaterEqual(evidence["audit_event_count"], 1)
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_full_learning_experience_verify(self) -> None:
        result = run_cli("scenario", "verify", "full-learning-experience", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "full-learning-experience")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 18)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_FULL_LEARNING_EXPERIENCE_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {
                "CS-LEARN-001",
                "CS-LEARN-002",
                "CS-LEARN-003",
                "CS-LEARN-004",
                "CS-LEARN-005",
                "CS-LEARN-006",
                "CS-LEARN-007",
                "CS-LEARN-008",
                "CS-LEARN-009",
                "CS-LEARN-010",
                "CS-LEARN-011",
                "CS-LEARN-012",
                "CS-LEARN-013",
                "CS-LEARN-014",
                "CS-LEARN-015",
                "CS-LEARN-016",
                "CS-LEARN-017",
                "CS-LEARN-018",
            },
        )
        evidence = payload["learning_experience_evidence"]
        self.assertTrue(evidence["success_trajectory_id"].startswith("traj_"))
        self.assertTrue(evidence["failure_trajectory_id"].startswith("traj_"))
        self.assertTrue(evidence["org_trajectory_id"].startswith("traj_"))
        self.assertTrue(evidence["connected_outcome_id"].startswith("outcome_"))
        self.assertTrue(evidence["lesson_id"].startswith("lesson_"))
        self.assertGreaterEqual(evidence["personal_library_trajectory_count"], 2)
        self.assertEqual(evidence["experience_search_result_count"], 1)
        self.assertEqual(evidence["recommendation_count"], 1)
        self.assertEqual(evidence["personal_org_search_result_count"], 0)
        self.assertEqual(evidence["org_search_result_count"], 1)
        self.assertGreaterEqual(evidence["audit_event_count"], 1)
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_full_understanding_ontology_verify(self) -> None:
        result = run_cli("scenario", "verify", "full-understanding-ontology", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "full-understanding-ontology")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 7)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_FULL_UNDERSTANDING_ONTOLOGY_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {"CS-UND-006", "CS-UND-007", "CS-UND-008", "CS-UND-009", "CS-UND-010", "CS-UND-011", "CS-UND-012"},
        )
        evidence = payload["understanding_evidence"]
        self.assertGreaterEqual(evidence["suggestion_count"], 4)
        self.assertGreaterEqual(evidence["promoted_item_count"], 5)
        self.assertIn("object", evidence["suggestion_kinds"])
        self.assertIn("fact", evidence["suggestion_kinds"])
        self.assertIn("event", evidence["suggestion_kinds"])
        self.assertIn("link", evidence["suggestion_kinds"])
        self.assertTrue(evidence["operational_map_id"].startswith("omap_"))
        self.assertGreaterEqual(evidence["map_node_count"], 1)
        self.assertGreaterEqual(evidence["map_edge_count"], 1)
        self.assertGreaterEqual(evidence["contradiction_count"], 1)
        self.assertEqual(set(evidence["contradiction_values"]), {"2026-07-01", "2026-08-15"})
        self.assertEqual(evidence["staleness_status"], "needs_review")
        self.assertTrue(evidence["staleness_warning_visible"])
        self.assertEqual(evidence["ontology_change_versions"], {"from": 1, "to": 2})
        self.assertGreaterEqual(evidence["unknown_evidence_gap_count"], 1)
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_full_extension_ecosystem_verify(self) -> None:
        result = run_cli("scenario", "verify", "full-extension-ecosystem", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "full-extension-ecosystem")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 20)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_FULL_EXTENSION_ECOSYSTEM_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {
                "CS-EXT-001",
                "CS-EXT-002",
                "CS-EXT-003",
                "CS-EXT-004",
                "CS-EXT-005",
                "CS-EXT-006",
                "CS-EXT-007",
                "CS-EXT-008",
                "CS-EXT-009",
                "CS-EXT-010",
                "CS-EXT-011",
                "CS-EXT-012",
                "CS-EXT-013",
                "CS-EXT-014",
                "CS-EXT-015",
                "CS-EXT-016",
                "CS-SEC-015",
                "CS-SEC-016",
                "CS-REG-014",
                "CS-REG-015",
            },
        )
        evidence = payload["extension_evidence"]
        self.assertTrue(evidence["core_artifact_id"].startswith("art_"))
        self.assertEqual(evidence["pack_id"], "pack_ops_recovery_agent")
        self.assertTrue(evidence["activation_id"].startswith("packact_"))
        self.assertTrue(evidence["certification_id"].startswith("packcert_"))
        self.assertTrue(evidence["connector_request_id"].startswith("packconn_"))
        self.assertTrue(evidence["playbook_proposal_id"].startswith("packpb_"))
        self.assertTrue(evidence["rollback_id"].startswith("packroll_"))
        self.assertTrue(evidence["security_patch_id"].startswith("packpatch_"))
        self.assertEqual(evidence["untrusted_activation_exit_code"], 8)
        self.assertEqual(evidence["direct_provider_import_exit_code"], 8)
        self.assertGreaterEqual(evidence["audit_event_count"], 1)
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_full_agent_orchestration_verify(self) -> None:
        result = run_cli("scenario", "verify", "full-agent-orchestration", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "full-agent-orchestration")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 14)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_FULL_AGENT_ORCHESTRATION_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {
                "CS-AGENT-001",
                "CS-AGENT-002",
                "CS-AGENT-003",
                "CS-AGENT-004",
                "CS-AGENT-005",
                "CS-AGENT-006",
                "CS-AGENT-007",
                "CS-AGENT-008",
                "CS-AGENT-009",
                "CS-AGENT-010",
                "CS-AGENT-011",
                "CS-AGENT-012",
                "CS-AGENT-013",
                "CS-AGENT-014",
            },
        )
        evidence = payload["agent_evidence"]
        self.assertEqual(evidence["role_count"], 8)
        self.assertIn("orchestrator", evidence["role_keys"])
        self.assertEqual(set(evidence["activity_roles"]), {"evidence", "memory", "workflow", "policy", "connector", "judge", "playbook"})
        self.assertTrue(evidence["trace_id"].startswith("agenttrace_"))
        self.assertTrue(evidence["brain_switch_id"].startswith("agentbrain_"))
        self.assertTrue(evidence["contract_update_id"].startswith("agentupd_"))
        self.assertTrue(evidence["diagnosis_id"].startswith("agentdiag_"))
        self.assertTrue(evidence["replay_id"].startswith("agentreplay_"))
        self.assertEqual(evidence["direct_mutation_exit_code"], 8)
        self.assertEqual(evidence["prompt_authority_exit_code"], 8)
        self.assertEqual(evidence["pack_capability_denied_exit_code"], 8)
        self.assertGreaterEqual(evidence["audit_event_count"], 1)
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_full_brain_routing_verify(self) -> None:
        result = run_cli("scenario", "verify", "full-brain-routing", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "full-brain-routing")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 21)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_FULL_BRAIN_ROUTING_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {
                "CS-BRAIN-001",
                "CS-BRAIN-002",
                "CS-BRAIN-003",
                "CS-BRAIN-004",
                "CS-BRAIN-005",
                "CS-BRAIN-006",
                "CS-BRAIN-007",
                "CS-BRAIN-008",
                "CS-BRAIN-009",
                "CS-BRAIN-010",
                "CS-BRAIN-011",
                "CS-BRAIN-012",
                "CS-BRAIN-013",
                "CS-BRAIN-014",
                "CS-BRAIN-015",
                "CS-BRAIN-016",
                "CS-ARCH-012",
                "CS-NS-009",
                "CS-NS-010",
                "CS-REG-009",
                "CS-REG-010",
            },
        )
        evidence = payload["brain_evidence"]
        self.assertEqual(evidence["model_count"], 5)
        self.assertTrue(evidence["routine_route_id"].startswith("brainroute_"))
        self.assertTrue(evidence["high_risk_route_id"].startswith("brainroute_"))
        self.assertTrue(evidence["override_allowed_route_id"].startswith("brainroute_"))
        self.assertTrue(evidence["brain_switch_id"].startswith("brainswitch_"))
        self.assertTrue(evidence["judge_record_id"].startswith("judge_"))
        self.assertTrue(evidence["judge_conflict_id"].startswith("judgeconf_"))
        self.assertTrue(evidence["owner_acceptance_id"].startswith("accept_"))
        self.assertTrue(evidence["judge_recommendation_id"].startswith("judgerec_"))
        self.assertTrue(evidence["adjudication_id"].startswith("adjud_"))
        self.assertTrue(evidence["calibration_id"].startswith("judgecal_"))
        self.assertEqual(evidence["override_denied_exit_code"], 8)
        self.assertEqual(evidence["aggregation_denied_exit_code"], 8)
        self.assertGreaterEqual(evidence["personal_ledger_entry_count"], 3)
        self.assertGreaterEqual(evidence["org_ledger_entry_count"], 1)
        self.assertGreaterEqual(evidence["audit_event_count"], 1)
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_full_security_operations_verify(self) -> None:
        result = run_cli("scenario", "verify", "full-security-operations", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "full-security-operations")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 11)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_FULL_SECURITY_OPERATIONS_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {
                "CS-SEC-009",
                "CS-SEC-010",
                "CS-SEC-011",
                "CS-SEC-012",
                "CS-SEC-013",
                "CS-SEC-014",
                "CS-SEC-017",
                "CS-SEC-018",
                "CS-SEC-019",
                "CS-SEC-020",
                "CS-REG-020",
            },
        )
        evidence = payload["security_operations_evidence"]
        self.assertTrue(evidence["credential_boundary_id"].startswith("credbound_"))
        self.assertTrue(evidence["sensitive_gate_id"].startswith("sensgate_"))
        self.assertTrue(evidence["backup_restore_id"].startswith("backuprestore_"))
        self.assertTrue(evidence["helpful_failure_id"].startswith("helpfail_"))
        self.assertTrue(evidence["idempotency_id"].startswith("idemp_"))
        self.assertTrue(evidence["retention_id"].startswith("retention_"))
        self.assertTrue(evidence["operator_status_id"].startswith("opsstatus_"))
        self.assertTrue(evidence["release_report_id"].startswith("relreport_"))
        self.assertEqual(evidence["release_report_scenario_count"], 21)
        self.assertGreaterEqual(evidence["audit_event_count"], 1)
        self.assertTrue(payload["human_required"])
        for row in payload["human_required"]:
            self.assertTrue(row["reason"])
            self.assertTrue(row["required_human_action"])
            self.assertTrue(row["expected_human_evidence"])
            self.assertTrue(row["release_impact"])
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_full_namespace_governance_verify(self) -> None:
        result = run_cli("scenario", "verify", "full-namespace-governance", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "full-namespace-governance")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 14)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_FULL_NAMESPACE_GOVERNANCE_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {
                "CS-ARCH-010",
                "CS-ARCH-011",
                "CS-ARCH-013",
                "CS-ARCH-014",
                "CS-NS-005",
                "CS-NS-006",
                "CS-NS-007",
                "CS-NS-008",
                "CS-NS-011",
                "CS-NS-012",
                "CS-NS-013",
                "CS-NS-014",
                "CS-REG-007",
                "CS-REG-008",
            },
        )
        evidence = payload["namespace_governance_evidence"]
        self.assertTrue(evidence["claim_basis_export_id"].startswith("claimbasis_"))
        self.assertTrue(evidence["source_safety_id"].startswith("srcsafe_"))
        self.assertTrue(evidence["namespace_audit_export_id"].startswith("nsaudit_"))
        self.assertTrue(evidence["retention_id"].startswith("retention_"))
        self.assertTrue(evidence["recovery_id"].startswith("nsrecover_"))
        self.assertEqual(set(evidence["promotion_modes"]), {"copy_with_provenance", "reference", "share", "promote_to_approved_truth"})
        self.assertTrue(evidence["promotion_modes"]["copy_with_provenance"]["materialized"])
        self.assertFalse(evidence["promotion_modes"]["reference"]["materialized"])
        self.assertFalse(evidence["promotion_modes"]["share"]["materialized"])
        self.assertTrue(evidence["promotion_modes"]["promote_to_approved_truth"]["materialized"])
        self.assertTrue(all(evidence["audit_coverage"].values()))
        self.assertIn("configure_autopilot", evidence["org_policy_actions"])
        self.assertIn("install_pack", evidence["org_policy_actions"])
        self.assertIn("aggregate_learning", evidence["org_policy_actions"])
        self.assertIn("extract_memory", evidence["classification_actions"])
        self.assertGreaterEqual(evidence["audit_event_count"], 1)
        self.assertFalse(payload["human_required"])
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_full_mission_control_autonomy_lifecycle_verify(self) -> None:
        result = run_cli("scenario", "verify", "full-mission-control-autonomy-lifecycle", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_set"], "full-mission-control-autonomy-lifecycle")
        self.assertEqual(payload["summary"]["blocking"], 0)
        self.assertEqual(payload["summary"]["pass"], 14)
        self.assertEqual(payload["summary"]["product_feature_claims"], "PARTIAL_FULL_MISSION_CONTROL_AUTONOMY_LIFECYCLE_ONLY")
        self.assertEqual(
            {row["id"] for row in payload["scenario_results"]},
            {
                "CS-PROD-006",
                "CS-PROD-007",
                "CS-PROD-008",
                "CS-PROD-009",
                "CS-PROD-010",
                "CS-AUTO-012",
                "CS-AUTO-013",
                "CS-AUTO-014",
                "CS-AUTO-015",
                "CS-AUTO-016",
                "CS-AUTO-017",
                "CS-AUTO-018",
                "CS-AUTO-019",
                "CS-REG-019",
            },
        )
        evidence = payload["mission_control_autonomy_evidence"]
        self.assertTrue(evidence["mission_control_id"].startswith("missioncontrol_"))
        self.assertTrue(evidence["connector_action_trace_id"].startswith("connectortrace_"))
        self.assertTrue(evidence["outcome_id"].startswith("outcome_"))
        self.assertTrue(evidence["after_action_review_id"].startswith("aar_"))
        self.assertTrue(evidence["mission_audit_export_id"].startswith("missionaudit_"))
        self.assertTrue(evidence["autonomy_metric_id"].startswith("autonomymetrics_"))
        self.assertTrue(evidence["revoke_control_id"].startswith("autonomyctl_"))
        self.assertTrue(evidence["repo_split_review_id"].startswith("reposplit_"))
        self.assertEqual(set(evidence["reversibility_ids"]), {"rollback", "compensation", "retry", "non_reversible"})
        self.assertEqual(len(evidence["escalation_ids"]), 6)
        self.assertFalse(payload["human_required"])
        for value in payload["negative_evidence"].values():
            self.assertEqual(value, 0)

    def test_full_filtered_final_scenario_verify(self) -> None:
        result = run_cli("scenario", "verify", "full", "--scenario", "CS-PROD-006", "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_filter"], ["CS-PROD-006"])
        self.assertEqual(payload["summary"]["scenario_count"], 1)
        self.assertEqual(payload["scenario_results"][0]["id"], "CS-PROD-006")
        self.assertEqual(payload["scenario_results"][0]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
