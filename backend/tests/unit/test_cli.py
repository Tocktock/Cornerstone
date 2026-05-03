from __future__ import annotations

import json
from pathlib import Path

import pytest

from cornerstone import __version__
from cornerstone.cli import main


def test_cli_version_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_dry_run_stack_up_prints_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stack", "up", "--migrate", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "DRY RUN: docker compose up -d postgres" in output
    assert "DRY RUN: alembic upgrade head" in output


def test_cli_dry_run_worker_prints_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["worker", "--once", "--run-scheduler", "--max-jobs", "2", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "DRY RUN: python scripts/run_sync_worker.py --once --run-scheduler --max-jobs 2" in output


def test_cli_doctor_json_reports_project_root(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--json"]) in {0, 1}
    payload = json.loads(capsys.readouterr().out)
    assert "projectRoot" in payload
    assert "checks" in payload
    assert any(check["name"] == "python" for check in payload["checks"])


def test_cli_env_init_creates_local_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"\n', encoding="utf-8")
    (tmp_path / "alembic.ini").write_text('', encoding="utf-8")
    (tmp_path / "docker-compose.yml").write_text('', encoding="utf-8")
    (tmp_path / ".env.example").write_text(
        "PRODUCTION_MODE=true\n"
        "PERSISTENCE_BACKEND=postgres\n"
        "CONNECTOR_ENCRYPTION_SECRET=local-dev-only-change-me-secret\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["env", "init"]) == 0
    content = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "PRODUCTION_MODE=false" in content
    assert "PERSISTENCE_BACKEND=memory" in content
    assert "local-dev-only-change-me-secret" not in content


def test_cli_source_list_formats_table(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from cornerstone.cli.commands import source as cli_source

    def fake_json(url: str, **_kwargs: object) -> dict[str, object]:
        assert url.endswith("/v1/sources")
        return {
            "sources": [
                {
                    "id": "source-1",
                    "type": "notion",
                    "authStatus": "authorized",
                    "connectionStatus": "test_passed",
                    "syncStatus": "succeeded",
                    "freshnessState": "fresh",
                    "artifactCount": 1,
                    "evidenceFragmentCount": 5,
                    "nextAction": "review_evidence",
                }
            ]
        }

    monkeypatch.setattr(cli_source, "_http_json", fake_json)
    assert main(["source", "list"]) == 0
    output = capsys.readouterr().out
    assert "TYPE" in output
    assert "notion" in output
    assert "review_evidence" in output


def test_cli_evidence_review_posts_expected_payload(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from cornerstone.cli.commands import evidence as cli_evidence

    captured: dict[str, object] = {}

    def fake_request(method: str, url: str, **kwargs: object) -> dict[str, object]:
        captured["method"] = method
        captured["url"] = url
        captured["body"] = kwargs["body"]
        return {"id": "ev-1", "trustState": "reviewed", "reviewedBy": "reviewer@example.com"}

    monkeypatch.setattr(cli_evidence, "_http_request", fake_request)
    assert main(["evidence", "review", "ev-1", "--reviewer", "reviewer@example.com", "--note", "ok"]) == 0
    assert captured["method"] == "POST"
    assert str(captured["url"]).endswith("/v1/evidence/ev-1/review")
    assert captured["body"] == {"trustState": "reviewed", "reviewedBy": "reviewer@example.com", "reviewNote": "ok"}
    assert "marked reviewed" in capsys.readouterr().out


def test_cli_concept_create_from_evidence_posts_expected_payload(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from cornerstone.cli.commands import concept as cli_concept

    captured: dict[str, object] = {}

    def fake_request(method: str, url: str, **kwargs: object) -> dict[str, object]:
        captured["method"] = method
        captured["url"] = url
        captured["body"] = kwargs["body"]
        return {"id": "concept-1", "name": "Cornerstone", "status": "reviewing"}

    monkeypatch.setattr(cli_concept, "_http_request", fake_request)
    assert main([
        "concept", "create-from-evidence", "ev-1",
        "--name", "Cornerstone",
        "--definition", "Shared context layer",
        "--created-by", "reviewer@example.com",
    ]) == 0
    assert captured["method"] == "POST"
    assert str(captured["url"]).endswith("/v1/evidence/ev-1/concept-candidates")
    assert captured["body"] == {
        "name": "Cornerstone",
        "shortDefinition": "Shared context layer",
        "body": None,
        "owner": None,
        "createdBy": "reviewer@example.com",
    }
    assert "Concept concept-1 created" in capsys.readouterr().out


def test_cli_context_query_formats_trust_label_and_citations(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from cornerstone.cli.commands import context as cli_context

    def fake_json(url: str, **_kwargs: object) -> dict[str, object]:
        assert "/v1/context/query" in url
        return {
            "answer": "Cornerstone is a shared organizational context layer.",
            "trustLabel": "official",
            "freshness": {"state": "fresh"},
            "officialAnswerAvailable": True,
            "evidence": [{"evidenceFragmentId": "ev-1", "isValid": True}],
            "limitations": [],
        }

    monkeypatch.setattr(cli_context, "_http_json", fake_json)
    assert main(["ask", "What is Cornerstone?"]) == 0
    output = capsys.readouterr().out
    assert "Trust: official" in output
    assert "Evidence ev-1" in output
    assert "Limitations:" in output


def test_cli_review_queue_and_preview_call_operator_endpoints(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from cornerstone.cli.commands import review as cli_review

    requested: list[str] = []

    def fake_json(url: str, **_kwargs: object) -> dict[str, object]:
        requested.append(url)
        if "review-queue" in url:
            return {
                "groupedByFocusConcept": [
                    {
                        "focusConcept": "Settlement",
                        "pendingConceptCandidateCount": 2,
                        "pendingRelationCandidateCount": 1,
                        "blockedCount": 0,
                    }
                ]
            }
        return {
            "candidateId": "candidate-1",
            "candidateType": "concept",
            "action": "merge",
            "canApply": True,
            "officialGraphWillChange": True,
            "mutationSummary": "Merging this candidate preserves evidence.",
            "targetConceptId": "concept-1",
            "evidencePreserved": True,
            "blockerReasons": [],
            "nextActions": [],
        }

    monkeypatch.setattr(cli_review, "_http_json", fake_json)

    assert main(["review", "queue", "--status", "pending", "--run-id", "run-1", "--source-id", "source-1"]) == 0
    assert main(["review", "preview", "concept", "candidate-1", "--action", "merge", "--target-id", "concept-1"]) == 0
    output = capsys.readouterr().out
    assert "Settlement" in output
    assert requested[0].endswith("/v1/ontology/review-queue/summary?status=pending&runId=run-1&sourceId=source-1")
    assert requested[1].endswith("/v1/ontology/concept-candidates/candidate-1/preview?action=merge&targetConceptId=concept-1")


def test_cli_eval_create_builds_metric_safe_payload(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from cornerstone.cli.commands import evaluation as cli_eval

    captured: dict[str, object] = {}

    def fake_request(method: str, url: str, **kwargs: object) -> dict[str, object]:
        captured["method"] = method
        captured["url"] = url
        captured["body"] = kwargs["body"]
        return {"id": "task-1", "name": "Definition task"}

    monkeypatch.setattr(cli_eval, "_http_request", fake_request)
    assert main([
        "eval", "create",
        "--name", "Definition task",
        "--query", "What is Cornerstone?",
        "--expected-trust-label", "official",
        "--expected-answer-contains", "shared organizational context layer",
        "--required-evidence", "ev-1",
        "--required-concept", "concept-1",
        "--require-official-answer",
        "--created-by", "reviewer@example.com",
    ]) == 0
    assert captured["method"] == "POST"
    assert str(captured["url"]).endswith("/v1/evaluations/tasks")
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["expectedTrustLabel"] == "official"
    assert body["expectedAnswerContains"] == ["shared organizational context layer"]
    assert body["requiredEvidenceFragmentIds"] == ["ev-1"]
    assert body["requiredConceptIds"] == ["concept-1"]
    assert body["requireOfficialAnswer"] is True
    assert "Evaluation task task-1 created" in capsys.readouterr().out


def test_cli_proof_run_dry_run_writes_consolidated_report(tmp_path: Path) -> None:
    report = tmp_path / "proof.json"
    markdown = tmp_path / "proof.md"
    assert main([
        "proof",
        "run",
        "--postgres",
        "--notion",
        "--product-loop",
        "--safety-checks",
        "--secret-scan",
        "--dry-run",
        "--save",
        str(report),
        "--markdown",
        str(markdown),
    ]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schemaVersion"] == 2
    assert payload["dryRun"] is True
    assert payload["summary"]["status"] == "planned"
    assert payload["categories"]["live_postgres"]["planned"] == 1
    assert payload["categories"]["live_notion"]["planned"] == 1
    names = [step["name"] for step in payload["steps"]]
    assert "live_postgres" in names
    assert "live_notion_e2e" in names
    assert "product_grounded_official_with_valid_citation" in names
    assert "safety_direct_notion_source_409" in names
    assert "secret_scan" in names
    assert markdown.exists()
    assert "Cornerstone Proof Report" in markdown.read_text(encoding="utf-8")


def test_cli_proof_run_safety_checks_and_secret_scan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from cornerstone.cli import proof as cli_proof

    def fake_status(method: str, url: str, **kwargs: object) -> tuple[int, object]:
        if method == "GET" and url.endswith("/v1/sources"):
            return 200, {"sources": [{"id": "source-1", "type": "notion"}]}
        if method == "POST" and url.endswith("/v1/sources"):
            return 409, {"detail": "provider sources require OAuth"}
        if method == "POST" and url.endswith("/oauth/complete"):
            return 404, {"detail": "not found"}
        if method == "POST" and "/v1/manual-sources/" in url:
            return 409, {"detail": "manual only"}
        if method == "POST" and url.endswith("/sync"):
            return 404, {"detail": "not found"}
        if method == "POST" and url.endswith("/v1/evaluations/tasks"):
            return 422, {"detail": "weak task"}
        return 500, {"detail": url}

    monkeypatch.setattr(cli_proof, "_http_status", fake_status)
    monkeypatch.setattr(cli_proof, "_scan_for_notion_tokens", lambda _root: [])
    report = tmp_path / "proof.json"
    assert main(["proof", "run", "--safety-checks", "--secret-scan", "--save", str(report), "--continue-on-failure"]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"]["status"] == "passed"
    names = [step["name"] for step in payload["steps"]]
    assert "safety_direct_notion_source_409" in names
    assert "safety_weak_evaluation_task_422" in names
    assert "secret_scan" in names


def test_cli_proof_run_product_loop_checks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from cornerstone.cli import proof as cli_proof

    def fake_status(method: str, url: str, **kwargs: object) -> tuple[int, object]:
        if url.endswith("/healthz"):
            return 200, {"status": "ok", "version": "1.2.0"}
        if url.endswith("/v1/sources"):
            return 200, {"hasRealSources": True, "sources": [{"id": "source-1", "type": "notion"}]}
        if url.endswith("/v1/artifacts"):
            return 200, [{"id": "artifact-1"}]
        if url.endswith("/v1/evidence"):
            return 200, [{"id": "ev-1"}]
        if "/v1/context/query" in url and "interplanetary" in url:
            return 200, {"trustLabel": "unsupported"}
        if "/v1/context/query" in url:
            return 200, {"trustLabel": "official", "officialAnswerAvailable": True, "evidence": [{"evidenceFragmentId": "ev-1", "isValid": True}]}
        if url.endswith("/v1/evaluations/summary"):
            return 200, {"groundedContextTaskSuccessRate": 1.0}
        return 500, {"detail": url}

    monkeypatch.setattr(cli_proof, "_http_status", fake_status)
    report = tmp_path / "proof.json"
    assert main(["proof", "run", "--product-loop", "--save", str(report), "--continue-on-failure"]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"]["status"] == "passed"
    assert payload["summary"]["passed"] >= 7


def test_cli_setup_json_reports_windows_profile(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["setup", "windows", "--json", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["targetOs"] == "windows"
    assert payload["plan"]["setupScript"] == "scripts/windows_setup.ps1"
    assert "Python 3.13+" in payload["plan"]["manualPrerequisites"]


def test_cli_setup_fix_dry_run_lists_safe_fixes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["setup", "linux", "--fix", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Cornerstone local setup (linux)" in output
    assert "Applying safe local fixes" in output
    assert "create .env" in output


def test_cli_doctor_fix_dry_run_includes_fixes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--fix", "--dry-run", "--json"]) in {0, 1}
    payload = json.loads(capsys.readouterr().out)
    assert "os" in payload
    assert "fixes" in payload
    assert any("reports" in item or ".env" in item for item in payload["fixes"])


def test_cli_local_reset_requires_confirmation(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["local", "reset", "--dry-run"]) == 1
    captured = capsys.readouterr()
    assert "requires" in captured.err.lower() or "destructive" in captured.err.lower()


def test_cli_local_reset_dry_run_prints_destructive_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["local", "reset", "--yes", "--start-after", "--migrate", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "DRY RUN: docker compose down -v" in output
    assert "DRY RUN: docker compose up -d postgres" in output
    assert "DRY RUN: alembic upgrade head" in output


def test_cli_config_set_get_unset_with_custom_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("CORNERSTONE_CONFIG_DIR", str(tmp_path))

    assert main(["config", "set", "baseUrl", "http://127.0.0.1:9999"]) == 0
    capsys.readouterr()
    assert main(["config", "set", "defaultReviewer", "reviewer@example.com"]) == 0
    capsys.readouterr()

    assert main(["config", "get", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["baseUrl"] == "http://127.0.0.1:9999"
    assert payload["config"]["defaultReviewer"] == "reviewer@example.com"

    assert main(["config", "unset", "defaultReviewer"]) == 0
    capsys.readouterr()
    assert main(["config", "get", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "defaultReviewer" not in payload["config"] or payload["config"]["defaultReviewer"] is None


def test_cli_completion_prints_shell_scripts(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["completion", "zsh"]) == 0
    assert "#compdef cornerstone" in capsys.readouterr().out

    assert main(["completion", "powershell"]) == 0
    assert "Register-ArgumentCompleter" in capsys.readouterr().out


def test_cli_is_split_into_maintainable_modules() -> None:
    root = Path(__file__).resolve().parents[2]
    assert not (root / "src/cornerstone/cli.py").exists()
    for rel_path in [
        "src/cornerstone/cli/main.py",
        "src/cornerstone/cli/parser.py",
        "src/cornerstone/cli/support.py",
        "src/cornerstone/cli/proof.py",
        "src/cornerstone/cli/config.py",
        "src/cornerstone/cli/completion.py",
        "src/cornerstone/cli/commands/source.py",
        "src/cornerstone/cli/commands/evidence.py",
        "src/cornerstone/cli/commands/concept.py",
        "src/cornerstone/cli/commands/evaluation.py",
    ]:
        assert (root / rel_path).exists(), rel_path
