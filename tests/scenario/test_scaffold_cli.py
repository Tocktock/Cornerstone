from __future__ import annotations

import json
import os
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
