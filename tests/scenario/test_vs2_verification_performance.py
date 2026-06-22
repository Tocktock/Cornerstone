from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from cornerstone_cli import scenarios, vs2_local_range, vs2_security, vs2_verification_metadata
from cornerstone_cli.vs2_verification_metadata import build_source_fingerprint, proof_hash, validate_reusable_report


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def vs2_scenario_ids() -> list[str]:
    matrix = ROOT / "docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_MATRIX.csv"
    with matrix.open(newline="") as file:
        return [row["scenario_id"] for row in csv.DictReader(file)]


def vs2_scenario_owners() -> dict[str, str]:
    matrix = ROOT / "docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_MATRIX.csv"
    with matrix.open(newline="") as file:
        return {
            row["scenario_id"]: "Human" if row["priority"] == "HUMAN_REQUIRED" else "AI"
            for row in csv.DictReader(file)
        }


def current_local_range_report(root: Path) -> dict:
    report = {
        "schema_version": "cs.vs2_local_range.v1",
        "status": "passed",
        "source_fingerprint": build_source_fingerprint(root, family="vs2_local_range"),
        "checks": {"fixture_check": True},
    }
    report["proof_hash"] = proof_hash(report)
    return report


def current_local_proof_report(root: Path, scenario_ids: list[str], evidence_path: Path, *, evidence_by_id: dict[str, list[Path]] | None = None) -> dict:
    evidence_by_id = evidence_by_id or {}
    evidence_relative = evidence_path.relative_to(root).as_posix()
    evidence_hash = sha256_file(evidence_path)
    manifest_path = evidence_path.parent / "test-vs2-evidence-manifest.json"
    manifest_relative = manifest_path.relative_to(root).as_posix()
    rows = []
    raw_scenario_artifacts = []
    foundational_by_path: dict[str, str] = {evidence_relative: evidence_hash}
    for index, scenario_id in enumerate(scenario_ids):
        if scenario_id in evidence_by_id:
            scenario_paths = evidence_by_id[scenario_id]
            evidence_relatives = [path.relative_to(root).as_posix() for path in scenario_paths]
            evidence_hashes = [sha256_file(path) for path in scenario_paths]
            for relative, digest in zip(evidence_relatives, evidence_hashes):
                foundational_by_path[relative] = digest
            rows.append(
                {
                    "id": scenario_id,
                    "scenario_id": scenario_id,
                    "status": "PASS",
                    "owner": "AI",
                    "validator": "fixture_validator",
                    "evidence": evidence_relatives,
                    "evidence_paths": evidence_relatives,
                    "evidence_hashes": evidence_hashes,
                }
            )
            raw_path = next((relative for relative in evidence_relatives if relative.startswith("reports/security/vs2/evidence/")), None)
            raw_scenario_artifacts.append(
                {
                    "scenario_id": scenario_id,
                    "status": "PASS",
                    "path": raw_path,
                    "sha256": foundational_by_path.get(raw_path) if raw_path else None,
                }
            )
        elif index == 0:
            rows.append(
                {
                    "id": scenario_id,
                    "scenario_id": scenario_id,
                    "status": "PASS",
                    "owner": "AI",
                    "validator": "fixture_validator",
                    "evidence": [evidence_relative],
                    "evidence_paths": [evidence_relative],
                    "evidence_hashes": [evidence_hash],
                }
            )
            raw_scenario_artifacts.append({"scenario_id": scenario_id, "status": "PASS", "path": evidence_relative, "sha256": evidence_hash})
        else:
            owner = "Human" if "-H" in scenario_id else "AI"
            status = "HUMAN_REQUIRED" if owner == "Human" else "NOT_VERIFIED"
            rows.append(
                {
                    "id": scenario_id,
                    "scenario_id": scenario_id,
                    "status": status,
                    "owner": owner,
                    "validator": None,
                    "evidence": [],
                    "evidence_paths": [],
                    "evidence_hashes": [],
                }
            )
            if owner == "AI":
                raw_scenario_artifacts.append({"scenario_id": scenario_id, "status": status, "path": None, "sha256": None})
    manifest_payload = {
        "schema_version": "cs.vs2.evidence_manifest.v1",
        "raw_scenario_artifacts": raw_scenario_artifacts,
        "foundational_artifacts": [
            {"path": relative, "sha256": digest}
            for relative, digest in sorted(foundational_by_path.items())
        ],
    }
    write_json(manifest_path, manifest_payload)
    manifest_hash = sha256_file(manifest_path)
    for row in rows:
        if row["status"] == "PASS":
            row["evidence"].append(manifest_relative)
            row["evidence_paths"].append(manifest_relative)
            row["evidence_hashes"].append(manifest_hash)
    report = {
        "schema_version": "cs.vs2_local_security_proof.v0",
        "status": "failed",
        "scenario_set": "vs2-policy-tenancy-egress",
        "source_fingerprint": build_source_fingerprint(root, family="vs2_local_proof"),
        "summary": {
            "product_feature_claims": "LOCAL_VS2_READINESS_REJECTED_REMEDIATION_REQUIRED",
        },
        "negative_evidence": {
            "ai_rows_marked_pass_without_evidence": 0,
            "ai_rows_marked_pass_without_scenario_validator": 0,
            "blanket_dependencies_ok_pass_used": 0,
        },
        "evidence_manifest": manifest_relative,
        "scenario_results": rows,
    }
    report["proof_hash"] = proof_hash(report)
    return report


class Vs2VerificationPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (ROOT / "tmp").mkdir(exist_ok=True)

    def test_local_proof_rejects_stale_local_range_reuse_without_rerun(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            stale_range_path = Path(tmp) / "stale-local-range.json"
            proof_path = Path(tmp).relative_to(ROOT) / "proof-reuse-rejected.json"
            stale_report = current_local_range_report(ROOT)
            stale_report["source_fingerprint"] = dict(stale_report["source_fingerprint"])
            stale_report["source_fingerprint"]["input_digest"] = "stale"
            stale_report["proof_hash"] = proof_hash(stale_report)
            write_json(stale_range_path, stale_report)

            with mock.patch.object(vs2_security, "VS2_PROOF_REPORT", proof_path):
                report = vs2_security.run_vs2_local_security_proof(
                    ROOT,
                    local_range_report=stale_range_path.relative_to(ROOT),
                )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["local_range_reuse"]["status"], "rejected")
        self.assertIn("source_fingerprint_input_digest_mismatch", report["local_range_reuse"]["errors"])
        self.assertEqual(report["negative_evidence"]["stale_or_invalid_local_range_reuse_blocked"], 1)

    def test_reusable_local_range_requires_hash_and_passing_checks(self) -> None:
        report = current_local_range_report(ROOT)

        missing_hash = dict(report)
        missing_hash.pop("proof_hash")
        reusable, errors, _ = validate_reusable_report(
            missing_hash,
            root=ROOT,
            family="vs2_local_range",
            expected_schema="cs.vs2_local_range.v1",
            require_status="passed",
        )
        self.assertFalse(reusable)
        self.assertIn("proof_hash_missing", errors)

        changed_status = dict(report)
        changed_status["status"] = "failed"
        changed_status["proof_hash"] = proof_hash(changed_status)
        reusable, errors, _ = validate_reusable_report(
            changed_status,
            root=ROOT,
            family="vs2_local_range",
            expected_schema="cs.vs2_local_range.v1",
            require_status="passed",
        )
        self.assertFalse(reusable)
        self.assertIn("status_mismatch", errors)

        changed_checks = dict(report)
        changed_checks["checks"] = {"fixture_check": False}
        changed_checks["proof_hash"] = proof_hash(changed_checks)
        reusable, errors, _ = validate_reusable_report(
            changed_checks,
            root=ROOT,
            family="vs2_local_range",
            expected_schema="cs.vs2_local_range.v1",
            require_status="passed",
        )
        self.assertFalse(reusable)
        self.assertIn("checks_not_all_passed", errors)

        reusable, errors, _ = validate_reusable_report(
            [],
            root=ROOT,
            family="vs2_local_range",
            expected_schema="cs.vs2_local_range.v1",
            require_status="passed",
        )
        self.assertFalse(reusable)
        self.assertEqual(errors, ["report_not_object"])

    def test_scenario_verify_reuses_current_proof_without_fresh_proof_rerun(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp_path = Path(tmp)
            proof_path = tmp_path / "current-local-proof.json"
            evidence_path = tmp_path / "test-evidence.json"
            write_json(evidence_path, {"status": "PASS", "scenario_id": "VS2-SEC-001"})
            proof = current_local_proof_report(ROOT, vs2_scenario_ids(), evidence_path)
            write_json(proof_path, proof)

            with mock.patch.object(scenarios, "run_vs2_local_security_proof", side_effect=AssertionError("fresh proof rerun")):
                report = scenarios.verify_vs2_policy_tenancy_egress(
                    ROOT,
                    local_proof_report=proof_path.relative_to(ROOT),
                )

        self.assertEqual(report["local_security_proof"]["reuse"]["status"], "reused")
        self.assertEqual(report["scenario_results"][0]["status"], "PASS")
        self.assertGreater(report["summary"]["blocking"], 0)

    def test_reusable_proof_accepts_real_static_config_and_policy_evidence_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            evidence_path = Path(tmp) / "test-evidence.json"
            write_json(evidence_path, {"status": "PASS", "scenario_id": "VS2-SEC-001"})
            evidence_by_id = {
                scenario_id: [ROOT / path for path in vs2_security._scenario_evidence_paths(scenario_id)]
                for scenario_id in ["VS2-SEC-026", "VS2-SEC-045", "VS2-SEC-047", "VS2-SEC-048"]
            }
            proof = current_local_proof_report(
                ROOT,
                vs2_scenario_ids(),
                evidence_path,
                evidence_by_id=evidence_by_id,
            )
            reusable, errors, _ = validate_reusable_report(
                proof,
                root=ROOT,
                family="vs2_local_proof",
                expected_schema="cs.vs2_local_security_proof.v0",
                expected_scenario_ids=set(vs2_scenario_ids()),
                expected_scenario_owners=vs2_scenario_owners(),
            )

        self.assertTrue(reusable, errors)
        self.assertFalse(any("path_unapproved_root" in error for error in errors))

    def test_reusable_proof_rejects_duplicate_scenarios_and_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp_path = Path(tmp)
            evidence_path = tmp_path / "test-evidence.json"
            write_json(evidence_path, {"status": "PASS", "scenario_id": "VS2-SEC-001"})
            proof = current_local_proof_report(ROOT, vs2_scenario_ids(), evidence_path)

            missing_hash = dict(proof)
            missing_hash.pop("proof_hash")
            reusable, errors, _ = validate_reusable_report(
                missing_hash,
                root=ROOT,
                family="vs2_local_proof",
                expected_schema="cs.vs2_local_security_proof.v0",
                expected_scenario_ids=set(vs2_scenario_ids()),
                expected_scenario_owners=vs2_scenario_owners(),
            )
            self.assertFalse(reusable)
            self.assertIn("proof_hash_missing", errors)

            missing_manifest = dict(proof)
            missing_manifest.pop("evidence_manifest")
            missing_manifest["proof_hash"] = proof_hash(missing_manifest)
            reusable, errors, _ = validate_reusable_report(
                missing_manifest,
                root=ROOT,
                family="vs2_local_proof",
                expected_schema="cs.vs2_local_security_proof.v0",
                expected_scenario_ids=set(vs2_scenario_ids()),
                expected_scenario_owners=vs2_scenario_owners(),
            )
            self.assertFalse(reusable)
            self.assertIn("evidence_manifest_missing", errors)

            proof["scenario_results"].append(dict(proof["scenario_results"][0]))
            proof["proof_hash"] = proof_hash(proof)
            reusable, errors, _ = validate_reusable_report(
                proof,
                root=ROOT,
                family="vs2_local_proof",
                expected_schema="cs.vs2_local_security_proof.v0",
                expected_scenario_ids=set(vs2_scenario_ids()),
                expected_scenario_owners=vs2_scenario_owners(),
            )
            self.assertFalse(reusable)
            self.assertTrue(any(error.startswith("scenario_results_duplicate_ids:") for error in errors))

            unsafe_path_proof = current_local_proof_report(ROOT, vs2_scenario_ids(), evidence_path)
            unsafe_path_proof["scenario_results"][0]["evidence_paths"] = ["../outside.json"]
            unsafe_path_proof["scenario_results"][0]["evidence"] = ["../outside.json"]
            unsafe_path_proof["scenario_results"][0]["evidence_hashes"] = ["0" * 64]
            unsafe_path_proof["proof_hash"] = proof_hash(unsafe_path_proof)
            reusable, errors, _ = validate_reusable_report(
                unsafe_path_proof,
                root=ROOT,
                family="vs2_local_proof",
                expected_schema="cs.vs2_local_security_proof.v0",
                expected_scenario_ids=set(vs2_scenario_ids()),
                expected_scenario_owners=vs2_scenario_owners(),
            )
            self.assertFalse(reusable)
            self.assertTrue(any("path_unsafe" in error for error in errors))

            reusable, errors, _ = validate_reusable_report(
                None,
                root=ROOT,
                family="vs2_local_proof",
                expected_schema="cs.vs2_local_security_proof.v0",
                expected_scenario_ids=set(vs2_scenario_ids()),
                expected_scenario_owners=vs2_scenario_owners(),
            )
            self.assertFalse(reusable)
            self.assertEqual(errors, ["report_not_object"])

            proof_path = tmp_path / "not-object-proof.json"
            write_json(proof_path, [])
            report = scenarios.verify_vs2_policy_tenancy_egress(
                ROOT,
                local_proof_report=proof_path.relative_to(ROOT),
            )
            self.assertEqual(report["local_security_proof"]["reuse"]["status"], "rejected")
            self.assertIn("invalid_shape", ",".join(report["local_security_proof"]["reuse"]["errors"]))

    def test_ai_human_required_status_is_rejected_and_blocks_scenario_report(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp_path = Path(tmp)
            proof_path = tmp_path / "ai-human-required-proof.json"
            evidence_path = tmp_path / "test-evidence.json"
            write_json(evidence_path, {"status": "PASS", "scenario_id": "VS2-SEC-001"})
            proof = current_local_proof_report(ROOT, vs2_scenario_ids(), evidence_path)
            for row in proof["scenario_results"]:
                if row["owner"] == "AI":
                    row["status"] = "HUMAN_REQUIRED"
                    row["validator"] = None
                    row["evidence"] = []
                    row["evidence_paths"] = []
                    row["evidence_hashes"] = []
            manifest_path = ROOT / proof["evidence_manifest"]
            manifest = json.loads(manifest_path.read_text())
            for artifact in manifest["raw_scenario_artifacts"]:
                artifact["status"] = "HUMAN_REQUIRED"
                artifact["path"] = None
                artifact["sha256"] = None
            write_json(manifest_path, manifest)
            proof["proof_hash"] = proof_hash(proof)
            write_json(proof_path, proof)

            reusable, errors, _ = validate_reusable_report(
                proof,
                root=ROOT,
                family="vs2_local_proof",
                expected_schema="cs.vs2_local_security_proof.v0",
                expected_scenario_ids=set(vs2_scenario_ids()),
                expected_scenario_owners=vs2_scenario_owners(),
            )
            report = scenarios.verify_vs2_policy_tenancy_egress(
                ROOT,
                local_proof_report=proof_path.relative_to(ROOT),
            )

        self.assertFalse(reusable)
        self.assertTrue(any(error.startswith("scenario_result_ai_human_required_invalid:") for error in errors))
        self.assertEqual(report["local_security_proof"]["reuse"]["status"], "rejected")
        self.assertEqual(report["status"], "failed")
        self.assertGreater(report["summary"]["blocking"], 0)

    def test_reusable_proof_rejects_missing_and_corrupt_evidence(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp_path = Path(tmp)
            evidence_path = tmp_path / "test-evidence.json"
            write_json(evidence_path, {"status": "PASS", "scenario_id": "VS2-SEC-001"})
            proof = current_local_proof_report(ROOT, vs2_scenario_ids(), evidence_path)

            evidence_path.unlink()
            reusable, errors, _ = validate_reusable_report(
                proof,
                root=ROOT,
                family="vs2_local_proof",
                expected_schema="cs.vs2_local_security_proof.v0",
                expected_scenario_ids=set(vs2_scenario_ids()),
                expected_scenario_owners=vs2_scenario_owners(),
            )
            self.assertFalse(reusable)
            self.assertTrue(any("missing" in error for error in errors))

            write_json(evidence_path, {"status": "CORRUPTED", "scenario_id": "VS2-SEC-001"})
            reusable, errors, _ = validate_reusable_report(
                proof,
                root=ROOT,
                family="vs2_local_proof",
                expected_schema="cs.vs2_local_security_proof.v0",
                expected_scenario_ids=set(vs2_scenario_ids()),
                expected_scenario_owners=vs2_scenario_owners(),
            )
            self.assertFalse(reusable)
            self.assertTrue(any("sha256_mismatch" in error for error in errors))

    def test_dependency_family_mutations_invalidate_reuse(self) -> None:
        dependency_files = [
            "packages/cornerstone_cli/main.py",
            "scripts/verify-vs2.sh",
            "fixtures/vs2/sample.json",
            "compose.vs2.yml",
            "config/vs2/policy.json",
            "docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md",
            "README.md",
            "pyproject.toml",
        ]
        for relative in dependency_files:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for path_name in dependency_files:
                    path = root / path_name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"{path_name}:before\n")
                report = current_local_range_report(root)
                (root / relative).write_text(f"{relative}:after\n")
                reusable, errors, _ = validate_reusable_report(
                    report,
                    root=root,
                    family="vs2_local_range",
                    expected_schema="cs.vs2_local_range.v1",
                    require_status="passed",
                )
                self.assertFalse(reusable)
                self.assertIn("source_fingerprint_input_digest_mismatch", errors)

    def test_source_fingerprint_records_dirty_dependency_state(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git executable is required")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "packages/cornerstone_cli/main.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("print('before')\n")
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, stdout=subprocess.DEVNULL)

            clean = build_source_fingerprint(root, family="vs2_local_range")
            path.write_text("print('after')\n")
            dirty = build_source_fingerprint(root, family="vs2_local_range")

        self.assertFalse(clean["dirty"])
        self.assertTrue(dirty["dirty"])
        self.assertIn("packages/cornerstone_cli/main.py", dirty["dirty_paths"])
        self.assertNotEqual(clean["working_tree_digest"], dirty["working_tree_digest"])

    def test_source_fingerprint_records_direct_child_dirty_paths_and_runtime_identity(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git executable is required")
        direct_paths = [
            "scripts/verify-vs2.sh",
            "tests/foo.py",
            "config/vs2/x.json",
            "policies/vs2/x.rego",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in direct_paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("before\n")
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, stdout=subprocess.DEVNULL)
            clean = build_source_fingerprint(root, family="vs2_local_range")
            for relative in direct_paths:
                (root / relative).write_text("after\n")
            dirty = build_source_fingerprint(root, family="vs2_local_range")

        self.assertFalse(clean["dirty"])
        self.assertTrue(dirty["dirty"])
        self.assertEqual(sorted(direct_paths), dirty["dirty_paths"])
        runtime = dirty["runtime"]
        self.assertIn("docker_context", runtime)
        self.assertIn("docker_version_json", runtime)
        self.assertIn("docker_info", runtime)
        self.assertIn("image_identities", runtime)
        self.assertIn("postgres:16-alpine", runtime["image_identities"])
        self.assertIn("openpolicyagent/opa@sha256:dc009236137bb225a1ef09293bb32f2ee1861cc428870d297bf71412d50221c3", runtime["image_identities"])

    def test_local_range_cleanup_failures_demote_pass_and_block_reuse(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            report_path = Path(tmp).relative_to(ROOT) / "cleanup-failure-local-range.json"
            payload = current_local_range_report(ROOT)
            vs2_local_range._finalize_report_payload(
                ROOT,
                report_path,
                payload,
                started=0.0,
                cleanup_seconds=0.1,
                cleanup_errors=["postgres:exit_code:1"],
                cleanup_results=[{"label": "postgres", "mandatory": True, "exit_code": 1}],
            )
            written = json.loads((ROOT / report_path).read_text())
            reusable, errors, _ = validate_reusable_report(
                written,
                root=ROOT,
                family="vs2_local_range",
                expected_schema="cs.vs2_local_range.v1",
                require_status="passed",
            )

        self.assertEqual(written["status"], "failed")
        self.assertTrue(written["cleanup_failure_demoted_pass"])
        self.assertFalse(reusable)
        self.assertIn("status_mismatch", errors)
        self.assertIn("cleanup_failed", errors)

    def test_local_range_refreshes_runtime_fingerprint_after_cold_image_pull(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            report_path = Path(tmp).relative_to(ROOT) / "cold-cache-runtime-refresh.json"
            end_fingerprint = build_source_fingerprint(ROOT, family="vs2_local_range")
            start_fingerprint = json.loads(json.dumps(end_fingerprint))
            start_fingerprint["runtime"]["image_identities"] = {
                vs2_local_range.POSTGRES_IMAGE: {"available": False},
                vs2_local_range.OPA_IMAGE: {"available": False},
            }
            start_fingerprint["environment_digest"] = "cold-cache-before-pull"
            payload = {
                "schema_version": "cs.vs2_local_range.v1",
                "status": "passed",
                "source_fingerprint": start_fingerprint,
                "checks": {"fixture_check": True},
            }
            with mock.patch.object(vs2_local_range, "build_source_fingerprint", return_value=end_fingerprint):
                vs2_local_range._refresh_report_source_fingerprint(ROOT, payload)
                vs2_local_range._finalize_report_payload(ROOT, report_path, payload, started=0.0)
            written = json.loads((ROOT / report_path).read_text())

            with mock.patch.object(vs2_verification_metadata, "build_source_fingerprint", return_value=end_fingerprint):
                reusable, errors, _ = vs2_verification_metadata.validate_reusable_report(
                    written,
                    root=ROOT,
                    family="vs2_local_range",
                    expected_schema="cs.vs2_local_range.v1",
                    require_status="passed",
                )

        self.assertEqual(written["status"], "passed")
        self.assertNotEqual(written["run_start_source_fingerprint"]["environment_digest"], written["source_fingerprint"]["environment_digest"])
        self.assertEqual(written["source_fingerprint"], end_fingerprint)
        self.assertNotIn("source_changed_during_run", written)
        self.assertTrue(reusable, errors)

    def test_local_range_fails_if_source_inputs_change_during_run(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            report_path = Path(tmp).relative_to(ROOT) / "source-change-local-range.json"
            start_fingerprint = build_source_fingerprint(ROOT, family="vs2_local_range")
            end_fingerprint = json.loads(json.dumps(start_fingerprint))
            end_fingerprint["input_digest"] = "changed-during-run"
            payload = {
                "schema_version": "cs.vs2_local_range.v1",
                "status": "passed",
                "source_fingerprint": start_fingerprint,
                "checks": {"fixture_check": True},
            }
            with mock.patch.object(vs2_local_range, "build_source_fingerprint", return_value=end_fingerprint):
                vs2_local_range._refresh_report_source_fingerprint(ROOT, payload)
                vs2_local_range._finalize_report_payload(ROOT, report_path, payload, started=0.0)
            written = json.loads((ROOT / report_path).read_text())

            with mock.patch.object(vs2_verification_metadata, "build_source_fingerprint", return_value=end_fingerprint):
                reusable, errors, _ = vs2_verification_metadata.validate_reusable_report(
                    written,
                    root=ROOT,
                    family="vs2_local_range",
                    expected_schema="cs.vs2_local_range.v1",
                    require_status="passed",
                )

        self.assertEqual(written["status"], "failed")
        self.assertTrue(written["source_changed_during_run"])
        self.assertEqual(written["source_changed_keys"], ["input_digest"])
        self.assertEqual(written["source_fingerprint"], end_fingerprint)
        self.assertFalse(reusable)
        self.assertIn("status_mismatch", errors)


if __name__ == "__main__":
    unittest.main()
