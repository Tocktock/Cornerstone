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
