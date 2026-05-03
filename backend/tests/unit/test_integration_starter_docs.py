from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_cli_and_macos_starter_docs_exist() -> None:
    required = [
        "docs/30-cli-macos-starter-v1.1.0.md",
        "docs/32-one-command-proof-runner-v1.1.2.md",
        "docs/33-cross-platform-starter-v1.1.3.md",
        "docs/34-cli-maintainability-v1.1.4.md",
        "docs/35-google-drive-connector-v1.2.0.md",
        "docs/integration-starter-kit/google-drive-quickstart.md",
        "docs/integration-starter-kit/local-quickstart.md",
        "docs/integration-starter-kit/linux-quickstart.md",
        "docs/integration-starter-kit/windows-quickstart.md",
        "docs/integration-starter-kit/cli-guide.md",
        "docs/integration-starter-kit/macos-quickstart.md",
        "docs/integration-starter-kit/notion-live-proof.md",
    ]
    for rel_path in required:
        content = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "Cornerstone" in content or "cornerstone" in content
        assert len(content) > 500


def test_macos_scripts_are_present_and_executable() -> None:
    for rel_path in [
        "scripts/macos_setup.sh",
        "scripts/macos_start_local.sh",
        "scripts/macos_run_live_proof.sh",
    ]:
        path = ROOT / rel_path
        assert path.exists()
        assert path.stat().st_mode & 0o111
