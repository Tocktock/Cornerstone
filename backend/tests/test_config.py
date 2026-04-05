from __future__ import annotations

from pathlib import Path

from cornerstone.config import discover_source_root


def test_discover_source_root_from_source_tree_layout(tmp_path: Path):
    repo_root = tmp_path / "repo"
    backend_root = repo_root / "backend"
    demo_root = repo_root / "demo_sources"
    config_file = backend_root / "src" / "cornerstone" / "config.py"

    demo_root.mkdir(parents=True)
    config_file.parent.mkdir(parents=True)
    config_file.touch()

    discovered = discover_source_root(start_cwd=backend_root, config_file=config_file)

    assert discovered == str(demo_root.resolve())


def test_discover_source_root_from_installed_package_layout(tmp_path: Path):
    repo_root = tmp_path / "repo"
    backend_root = repo_root / "backend"
    demo_root = repo_root / "demo_sources"
    config_file = (
        backend_root
        / ".venv"
        / "lib"
        / "python3.14"
        / "site-packages"
        / "cornerstone"
        / "config.py"
    )

    demo_root.mkdir(parents=True)
    config_file.parent.mkdir(parents=True)
    config_file.touch()

    discovered = discover_source_root(start_cwd=backend_root, config_file=config_file)

    assert discovered == str(demo_root.resolve())
