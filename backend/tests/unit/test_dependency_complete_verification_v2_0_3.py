from __future__ import annotations

from pathlib import Path

from cornerstone.verification.dependency_complete import (
    dependency_complete_command_plan,
    render_command_plan_markdown,
    write_command_plan_reports,
)


def test_dependency_complete_command_plan_is_measurable() -> None:
    plan = dependency_complete_command_plan("python3.13")
    ids = [command.id for command in plan]

    assert len(plan) == 13
    assert ids[0] == "V203-CMD-01-python-version"
    assert ids[-1] == "V203-CMD-13-duplicate-audit"
    assert any(command.required_env for command in plan)
    assert any(command.destructive for command in plan)
    assert all(command.report_path.startswith("reports/v2.0.3-") for command in plan)
    assert all(command.goal for command in plan)
    assert all(command.command for command in plan)


def test_dependency_complete_command_plan_markdown_contains_trust_boundary() -> None:
    markdown = render_command_plan_markdown(dependency_complete_command_plan("python"))

    assert "v2.0.3 Dependency-Complete Verification Command Plan" in markdown
    assert "V203-CMD-01-python-version" in markdown
    assert "V203-CMD-13-duplicate-audit" in markdown
    assert "does not add product behavior" in markdown


def test_dependency_complete_plan_reports_can_be_written(tmp_path: Path) -> None:
    json_path, markdown_path = write_command_plan_reports(
        commands=dependency_complete_command_plan("python"),
        reports_dir=tmp_path,
    )

    assert json_path.exists()
    assert markdown_path.exists()
    assert "V203-CMD-09-postgres-live-tests" in json_path.read_text(encoding="utf-8")
    assert "V203-CMD-12-proof-and-readiness" in markdown_path.read_text(encoding="utf-8")
