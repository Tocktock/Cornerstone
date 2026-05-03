from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

from cornerstone.config import DEFAULT_CONNECTOR_ENCRYPTION_SECRET, Settings
from cornerstone.connectors.providers.google_drive.adapter import GoogleDriveConnector, GoogleDriveProviderError
from cornerstone.connectors.registry import get_token_cipher
from cornerstone.observability import log_event
from cornerstone.verification.env import int_env
from cornerstone.schemas import (
    ConnectionTestStatus,
    ConnectorAuthType,
    ConnectorCredential,
    CredentialStatus,
    DataSource,
    DataSourceAuthStatus,
    DataSourceConnectionStatus,
    DataSourceNextAction,
    DataSourceStatus,
    DataSourceSyncStatus,
    DataSourceType,
    FreshnessState,
    ProviderObjectAccessState,
    ProviderObjectType,
    SourceSelection,
    SourceSelectionMode,
    SyncJob,
    SyncJobEvent,
    SyncJobStatus,
    SyncJobTrigger,
    utc_now,
)
from cornerstone.services.sync_worker import run_sync_job_once


@dataclass(frozen=True)
class GoogleDriveE2EIssue:
    code: str
    message: str


class GoogleDriveE2EConfigError(ValueError):
    def __init__(self, issues: list[GoogleDriveE2EIssue]) -> None:
        self.issues = tuple(issues)
        detail = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        super().__init__(f"Unsafe or incomplete live Google Drive E2E configuration: {detail}")


class GoogleDriveE2ERunError(RuntimeError):
    """Raised when live Google Drive E2E reaches Google but fails validation."""


@dataclass(frozen=True)
class GoogleDriveE2EConfig:
    enabled: bool
    access_token: str
    file_id: str
    source_name: str = "Live Google Drive E2E Source"
    created_by: str = "google-drive-e2e"
    worker_id: str = "google-drive-e2e-worker"
    lease_seconds: int = 300
    require_evidence: bool = True


@dataclass(frozen=True)
class GoogleDriveE2EResult:
    source_id: str
    credential_id: str
    snapshot_id: str
    sync_job_id: str
    sync_job_status: str
    artifact_count: int
    evidence_fragment_count: int
    artifact_created_count: int
    artifact_reused_count: int
    evidence_created_count: int
    source_freshness_state: str
    source_next_action: str

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def google_drive_e2e_config_from_env(env: Mapping[str, str]) -> GoogleDriveE2EConfig:
    return GoogleDriveE2EConfig(
        enabled=_truthy(env.get("RUN_GOOGLE_DRIVE_E2E", "")) or _truthy(env.get("RUN_LIVE_GOOGLE_DRIVE_TESTS", "")),
        access_token=(env.get("GOOGLE_DRIVE_E2E_ACCESS_TOKEN") or env.get("GOOGLE_DRIVE_PILOT_ACCESS_TOKEN") or "").strip(),
        file_id=(env.get("GOOGLE_DRIVE_E2E_FILE_ID") or env.get("GOOGLE_DRIVE_PILOT_FILE_ID") or "").strip(),
        source_name=env.get("GOOGLE_DRIVE_E2E_SOURCE_NAME", "Live Google Drive E2E Source").strip()
        or "Live Google Drive E2E Source",
        created_by=env.get("GOOGLE_DRIVE_E2E_CREATED_BY", "google-drive-e2e").strip() or "google-drive-e2e",
        worker_id=env.get("GOOGLE_DRIVE_E2E_WORKER_ID", "google-drive-e2e-worker").strip()
        or "google-drive-e2e-worker",
        lease_seconds=int_env(env.get("GOOGLE_DRIVE_E2E_LEASE_SECONDS"), default=300),
        require_evidence=_truthy(env.get("GOOGLE_DRIVE_E2E_REQUIRE_EVIDENCE", "1")),
    )


def google_drive_e2e_config_issues(config: GoogleDriveE2EConfig, settings: Settings) -> list[GoogleDriveE2EIssue]:
    issues: list[GoogleDriveE2EIssue] = []
    if not config.enabled:
        issues.append(
            GoogleDriveE2EIssue(
                code="google_drive_e2e_not_enabled",
                message="Set RUN_GOOGLE_DRIVE_E2E=1 to explicitly run live Google Drive E2E verification.",
            )
        )
    if not config.access_token:
        issues.append(
            GoogleDriveE2EIssue(
                code="google_drive_e2e_access_token_required",
                message="Set GOOGLE_DRIVE_E2E_ACCESS_TOKEN to a real Google OAuth access token.",
            )
        )
    if not config.file_id:
        issues.append(
            GoogleDriveE2EIssue(
                code="google_drive_e2e_file_id_required",
                message="Set GOOGLE_DRIVE_E2E_FILE_ID to a Google Doc or text file accessible to the token.",
            )
        )
    if settings.google_drive_mock_external_api:
        issues.append(
            GoogleDriveE2EIssue(
                code="google_drive_e2e_requires_live_google_drive_api",
                message="Set GOOGLE_DRIVE_MOCK_EXTERNAL_API=false for live Google Drive E2E verification.",
            )
        )
    if settings.connector_encryption_secret == DEFAULT_CONNECTOR_ENCRYPTION_SECRET:
        issues.append(
            GoogleDriveE2EIssue(
                code="google_drive_e2e_requires_non_default_connector_secret",
                message="Use a non-default CONNECTOR_ENCRYPTION_SECRET before storing a real Google Drive token.",
            )
        )
    if len(settings.connector_encryption_secret) < 32:
        issues.append(
            GoogleDriveE2EIssue(
                code="google_drive_e2e_connector_secret_too_short",
                message="CONNECTOR_ENCRYPTION_SECRET must be at least 32 characters for live token storage.",
            )
        )
    if config.lease_seconds < 30 or config.lease_seconds > 3600:
        issues.append(
            GoogleDriveE2EIssue(
                code="google_drive_e2e_invalid_lease_seconds",
                message="GOOGLE_DRIVE_E2E_LEASE_SECONDS must be between 30 and 3600.",
            )
        )
    return issues


def assert_google_drive_e2e_config_safe(config: GoogleDriveE2EConfig, settings: Settings) -> None:
    issues = google_drive_e2e_config_issues(config, settings)
    if issues:
        raise GoogleDriveE2EConfigError(issues)


async def run_live_google_drive_file_e2e(
    *,
    store: Any,
    settings: Settings,
    config: GoogleDriveE2EConfig,
) -> GoogleDriveE2EResult:
    """Run a strict live Google Drive file → Artifact → EvidenceFragment pilot path."""

    assert_google_drive_e2e_config_safe(config, settings)
    connector = GoogleDriveConnector(settings)
    cipher = get_token_cipher(settings)
    now = utc_now()
    data_source = DataSource(
        type=DataSourceType.GOOGLE_DRIVE,
        name=config.source_name,
        status=DataSourceStatus.SYNC_PENDING,
        production_enabled=True,
        created_at=now,
        auth_status=DataSourceAuthStatus.AUTHORIZED,
        connection_status=DataSourceConnectionStatus.UNTESTED,
        sync_status=DataSourceSyncStatus.NEVER_SYNCED,
        next_action=DataSourceNextAction.TEST_CONNECTION,
        freshness_state=FreshnessState.UNKNOWN,
        sync_freshness_state=FreshnessState.UNKNOWN,
        content_freshness_state=FreshnessState.UNKNOWN,
    )
    credential = ConnectorCredential(
        datasource_id=data_source.id,
        provider=DataSourceType.GOOGLE_DRIVE,
        auth_type=ConnectorAuthType.API_TOKEN,
        encrypted_access_token=cipher.encrypt(config.access_token),
        granted_scopes=["https://www.googleapis.com/auth/drive.readonly"],
        status=CredentialStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )
    with store.transaction():
        saved_source = cast(DataSource, store.add_data_source(data_source))
        saved_credential = cast(ConnectorCredential, store.add_connector_credential(credential))

    log_event(
        "google_drive.e2e.source_created",
        sourceId=saved_source.id,
        credentialId=saved_credential.id,
        fileId=config.file_id,
    )
    access_token = cipher.decrypt(saved_credential.encrypted_access_token)
    test_result = await connector.test_connection(credential=saved_credential, access_token=access_token)
    if test_result.status != ConnectionTestStatus.PASSED or not test_result.can_read_objects:
        message = "Live Google Drive connection test failed."
        if test_result.error is not None:
            message = f"{message} {test_result.error.user_message}"
        raise GoogleDriveE2ERunError(message)

    tested_source = cast(
        DataSource,
        store.update_data_source(
            saved_source.model_copy(
                update={
                    "status": DataSourceStatus.CONNECTED,
                    "connection_status": DataSourceConnectionStatus.TEST_PASSED,
                    "next_action": DataSourceNextAction.DISCOVER_SOURCES,
                    "last_connection_test_at": test_result.tested_at,
                    "last_error": None,
                },
                deep=True,
            )
        ),
    )
    try:
        snapshot = await connector.retrieve_file_snapshot(
            credential=saved_credential,
            access_token=access_token,
            file_id=config.file_id,
        )
    except GoogleDriveProviderError as exc:
        raise GoogleDriveE2ERunError(exc.connector_error.user_message) from exc
    if snapshot.object_type not in {ProviderObjectType.DOCUMENT, ProviderObjectType.TEXT_FILE}:
        raise GoogleDriveE2ERunError(f"Expected GOOGLE_DRIVE_E2E_FILE_ID to retrieve a Google Doc/text file, got {snapshot.object_type}.")
    if snapshot.access_state != ProviderObjectAccessState.ACCESSIBLE or not snapshot.ingestion_supported:
        raise GoogleDriveE2ERunError("The provided Google Drive file is not accessible or not supported for ingestion.")

    selection = SourceSelection(
        datasource_id=tested_source.id,
        sync_mode=SourceSelectionMode.SELECTED_ONLY,
        selected_external_object_ids=[snapshot.external_id],
    )
    with store.transaction():
        saved_snapshot = cast(Any, store.upsert_provider_object_snapshot(snapshot))
        store.upsert_source_selection(selection)
        store.mark_provider_object_selection(tested_source.id, [snapshot.external_id])
        store.update_data_source(
            tested_source.model_copy(
                update={
                    "last_discovery_at": utc_now(),
                    "discovered_object_count": 1,
                    "selected_object_count": 1,
                    "next_action": DataSourceNextAction.RUN_FIRST_SYNC,
                    "last_error": None,
                },
                deep=True,
            )
        )

    job = SyncJob(
        datasource_id=tested_source.id,
        provider=DataSourceType.GOOGLE_DRIVE,
        status=SyncJobStatus.QUEUED,
        trigger=SyncJobTrigger.MANUAL,
        created_by=config.created_by,
        selection_id=selection.id,
    )
    with store.transaction():
        saved_job = cast(SyncJob, store.add_sync_job(job))
        store.add_sync_job_event(
            SyncJobEvent(
                sync_job_id=saved_job.id,
                datasource_id=tested_source.id,
                event_type="sync.job_queued",
                message="Live Google Drive E2E sync job was queued.",
                metadata={"trigger": str(saved_job.trigger), "fileId": config.file_id},
            )
        )

    detail = await run_sync_job_once(
        job_id=saved_job.id,
        store=store,
        settings=settings,
        include_not_ready=True,
        worker_id=config.worker_id,
        lease_seconds=config.lease_seconds,
    )
    if detail.job.status != SyncJobStatus.SUCCEEDED:
        error = detail.job.error.user_message if detail.job.error is not None else "unknown sync failure"
        raise GoogleDriveE2ERunError(f"Live Google Drive sync job did not succeed: {error}")

    artifacts = cast(list[Any], store.list_artifacts(datasource_id=tested_source.id))
    evidence_fragments = cast(list[Any], store.list_evidence_fragments())
    if not artifacts:
        raise GoogleDriveE2ERunError("Live Google Drive E2E created no Artifacts.")
    source_artifact_ids = {artifact.id for artifact in artifacts}
    source_evidence = [fragment for fragment in evidence_fragments if fragment.artifact_id in source_artifact_ids]
    if config.require_evidence and not source_evidence:
        raise GoogleDriveE2ERunError(
            "Live Google Drive E2E created no EvidenceFragments. Use a test Google Doc with extractable text."
        )
    final_source = cast(DataSource, store.get_data_source(tested_source.id))
    return GoogleDriveE2EResult(
        source_id=final_source.id,
        credential_id=saved_credential.id,
        snapshot_id=saved_snapshot.id,
        sync_job_id=detail.job.id,
        sync_job_status=str(detail.job.status),
        artifact_count=len(artifacts),
        evidence_fragment_count=len(source_evidence),
        artifact_created_count=detail.job.artifact_created_count,
        artifact_reused_count=detail.job.artifact_reused_count,
        evidence_created_count=detail.job.evidence_created_count,
        source_freshness_state=str(final_source.freshness_state),
        source_next_action=str(final_source.next_action),
    )


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
