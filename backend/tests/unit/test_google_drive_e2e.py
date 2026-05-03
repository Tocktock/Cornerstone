from __future__ import annotations

import pytest

from cornerstone.config import Settings
from cornerstone.verification.google_drive_e2e import (
    GoogleDriveE2EConfig,
    google_drive_e2e_config_from_env,
    google_drive_e2e_config_issues,
)

pytestmark = pytest.mark.unit


def test_google_drive_e2e_config_from_env_reads_required_values() -> None:
    config = google_drive_e2e_config_from_env(
        {
            "RUN_GOOGLE_DRIVE_E2E": "1",
            "GOOGLE_DRIVE_E2E_ACCESS_TOKEN": "fake-google-drive-access-token",
            "GOOGLE_DRIVE_E2E_FILE_ID": "drive-file-id",
            "GOOGLE_DRIVE_E2E_REQUIRE_EVIDENCE": "0",
        }
    )

    assert config.enabled is True
    assert config.access_token == "fake-google-drive-access-token"
    assert config.file_id == "drive-file-id"
    assert config.require_evidence is False


def test_google_drive_e2e_preflight_requires_live_provider_and_token() -> None:
    config = GoogleDriveE2EConfig(enabled=False, access_token="", file_id="")
    settings = Settings()

    issue_codes = {issue.code for issue in google_drive_e2e_config_issues(config, settings)}

    assert "google_drive_e2e_not_enabled" in issue_codes
    assert "google_drive_e2e_access_token_required" in issue_codes
    assert "google_drive_e2e_file_id_required" in issue_codes
    assert "google_drive_e2e_requires_live_google_drive_api" in issue_codes
    assert "google_drive_e2e_requires_non_default_connector_secret" in issue_codes
