from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import sqlalchemy as sa
from sqlalchemy import Engine, Select, create_engine, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from cornerstone.persistence.models import (
    ArtifactRow,
    AuditEventRow,
    Base,
    ConceptRelationRow,
    ConceptRow,
    ConnectionIntentRow,
    ConnectorCredentialRow,
    DataSourceRow,
    DecisionRecordRow,
    EvidenceFragmentRow,
    GroundedContextEvalResultRow,
    GroundedContextEvalTaskRow,
    ProviderObjectSnapshotRow,
    SourceSelectionRow,
    SyncCursorRow,
    SyncJobEventRow,
    SyncJobRow,
    SyncScheduleRow,
    concept_decision_records,
    concept_evidence_fragments,
    concept_relation_evidence_fragments,
    decision_record_affected_concepts,
    decision_record_evidence_fragments,
)
from cornerstone.schemas import (
    Artifact,
    AuditEvent,
    Concept,
    ConceptRelation,
    ConceptStatus,
    ConnectionIntent,
    ConnectionIntentStatus,
    ConnectorAuthType,
    ConnectorCredential,
    ConnectorError,
    CredentialStatus,
    DataSource,
    DataSourceAuthStatus,
    DataSourceConnectionStatus,
    DataSourceNextAction,
    DataSourceStatus,
    DataSourceSyncStatus,
    DataSourceType,
    DecisionRecord,
    ErrorInfo,
    EvidenceFragment,
    EvidenceFragmentType,
    ExtractionStatus,
    FreshnessState,
    GroundedContextEvalResult,
    GroundedContextEvalTask,
    GroundedContextResponse,
    Provenance,
    ProviderObjectAccessState,
    ProviderObjectSnapshot,
    ProviderObjectType,
    RelationStatus,
    RelationType,
    SelectionRule,
    SourceSelection,
    SourceSelectionMode,
    SyncCursor,
    SyncJob,
    SyncJobEvent,
    SyncJobStatus,
    SyncJobTrigger,
    SyncSchedule,
    SyncScheduleStatus,
    TrustLabel,
    TrustState,
    utc_now,
)
from cornerstone.store import NotFoundError, StoreError


class PersistenceIntegrityError(StoreError):
    """Raised when relational persistence constraints fail."""


class SqlAlchemyStore:
    """SQLAlchemy-backed repository for PostgreSQL persistence.

    The implementation also supports SQLite for fast repository contract tests, but production is
    PostgreSQL. Route/service code uses the same repository methods as InMemoryStore.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
        self._current_session: ContextVar[Session | None] = ContextVar(
            "cornerstone_current_session", default=None
        )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        existing_session = self._current_session.get()
        if existing_session is not None:
            yield
            return

        with self._session_factory() as session:
            token = self._current_session.set(session)
            try:
                with session.begin():
                    yield
            except IntegrityError as exc:
                raise PersistenceIntegrityError(str(exc)) from exc
            finally:
                self._current_session.reset(token)

    def reset(self) -> None:
        with self._unit_of_work() as session:
            for table in reversed(Base.metadata.sorted_tables):
                session.execute(table.delete())

    # Data sources
    def add_data_source(self, data_source: DataSource) -> DataSource:
        with self._unit_of_work() as session:
            row = _data_source_to_row(data_source)
            session.add(row)
            session.flush()
            return _row_to_data_source(row)

    def get_data_source(self, data_source_id: str) -> DataSource:
        with self._unit_of_work() as session:
            row = session.get(DataSourceRow, data_source_id)
            if row is None:
                raise NotFoundError(f"DataSource not found: {data_source_id}")
            return _row_to_data_source(row)

    def update_data_source(self, data_source: DataSource) -> DataSource:
        with self._unit_of_work() as session:
            row = session.get(DataSourceRow, data_source.id)
            if row is None:
                raise NotFoundError(f"DataSource not found: {data_source.id}")
            _apply_data_source(row, data_source)
            session.flush()
            return _row_to_data_source(row)

    def list_data_sources(self) -> list[DataSource]:
        with self._unit_of_work() as session:
            rows = session.scalars(select(DataSourceRow).order_by(DataSourceRow.created_at, DataSourceRow.id))
            return [_row_to_data_source(row) for row in rows]


    # Connector connection intents
    def add_connection_intent(self, intent: ConnectionIntent) -> ConnectionIntent:
        with self._unit_of_work() as session:
            row = _connection_intent_to_row(intent)
            session.add(row)
            session.flush()
            return _row_to_connection_intent(row)

    def update_connection_intent(self, intent: ConnectionIntent) -> ConnectionIntent:
        with self._unit_of_work() as session:
            row = session.get(ConnectionIntentRow, intent.id)
            if row is None:
                raise NotFoundError(f"ConnectionIntent not found: {intent.id}")
            _apply_connection_intent(row, intent)
            session.flush()
            return _row_to_connection_intent(row)

    def get_connection_intent(self, intent_id: str) -> ConnectionIntent:
        with self._unit_of_work() as session:
            row = session.get(ConnectionIntentRow, intent_id)
            if row is None:
                raise NotFoundError(f"ConnectionIntent not found: {intent_id}")
            return _row_to_connection_intent(row)

    def get_connection_intent_by_state_nonce(self, state_nonce: str) -> ConnectionIntent:
        with self._unit_of_work() as session:
            row = session.scalars(
                select(ConnectionIntentRow).where(ConnectionIntentRow.state_nonce == state_nonce)
            ).first()
            if row is None:
                raise NotFoundError(f"ConnectionIntent not found for state nonce: {state_nonce}")
            return _row_to_connection_intent(row)

    def list_connection_intents(self) -> list[ConnectionIntent]:
        with self._unit_of_work() as session:
            rows = session.scalars(select(ConnectionIntentRow).order_by(ConnectionIntentRow.created_at))
            return [_row_to_connection_intent(row) for row in rows]

    # Connector credentials
    def add_connector_credential(self, credential: ConnectorCredential) -> ConnectorCredential:
        with self._unit_of_work() as session:
            row = _connector_credential_to_row(credential)
            session.add(row)
            session.flush()
            return _row_to_connector_credential(row)

    def update_connector_credential(self, credential: ConnectorCredential) -> ConnectorCredential:
        with self._unit_of_work() as session:
            row = session.get(ConnectorCredentialRow, credential.id)
            if row is None:
                raise NotFoundError(f"ConnectorCredential not found: {credential.id}")
            _apply_connector_credential(row, credential)
            session.flush()
            return _row_to_connector_credential(row)

    def get_connector_credential(self, credential_id: str) -> ConnectorCredential:
        with self._unit_of_work() as session:
            row = session.get(ConnectorCredentialRow, credential_id)
            if row is None:
                raise NotFoundError(f"ConnectorCredential not found: {credential_id}")
            return _row_to_connector_credential(row)

    def get_active_credential_for_source(self, datasource_id: str) -> ConnectorCredential:
        with self._unit_of_work() as session:
            row = session.scalars(
                select(ConnectorCredentialRow)
                .where(
                    ConnectorCredentialRow.datasource_id == datasource_id,
                    ConnectorCredentialRow.status == str(CredentialStatus.ACTIVE),
                    ConnectorCredentialRow.revoked_at.is_(None),
                )
                .order_by(ConnectorCredentialRow.created_at.desc())
            ).first()
            if row is None:
                raise NotFoundError(f"Active ConnectorCredential not found for DataSource: {datasource_id}")
            return _row_to_connector_credential(row)

    def list_connector_credentials(self, datasource_id: str | None = None) -> list[ConnectorCredential]:
        with self._unit_of_work() as session:
            statement: Select[tuple[ConnectorCredentialRow]] = select(ConnectorCredentialRow)
            if datasource_id is not None:
                statement = statement.where(ConnectorCredentialRow.datasource_id == datasource_id)
            statement = statement.order_by(ConnectorCredentialRow.created_at, ConnectorCredentialRow.id)
            return [_row_to_connector_credential(row) for row in session.scalars(statement)]

    # Source selections
    def upsert_source_selection(self, selection: SourceSelection) -> SourceSelection:
        with self._unit_of_work() as session:
            row = session.scalars(
                select(SourceSelectionRow).where(SourceSelectionRow.datasource_id == selection.datasource_id)
            ).first()
            if row is None:
                row = _source_selection_to_row(selection)
                session.add(row)
            else:
                preserved = selection.model_copy(update={"id": row.id, "created_at": row.created_at}, deep=True)
                _apply_source_selection(row, preserved)
            session.flush()
            return _row_to_source_selection(row)

    def get_source_selection(self, datasource_id: str) -> SourceSelection:
        with self._unit_of_work() as session:
            row = session.scalars(
                select(SourceSelectionRow).where(SourceSelectionRow.datasource_id == datasource_id)
            ).first()
            if row is None:
                raise NotFoundError(f"SourceSelection not found for DataSource: {datasource_id}")
            return _row_to_source_selection(row)

    def list_source_selections(self, datasource_id: str | None = None) -> list[SourceSelection]:
        with self._unit_of_work() as session:
            statement: Select[tuple[SourceSelectionRow]] = select(SourceSelectionRow)
            if datasource_id is not None:
                statement = statement.where(SourceSelectionRow.datasource_id == datasource_id)
            rows = session.scalars(statement.order_by(SourceSelectionRow.created_at, SourceSelectionRow.id))
            return [_row_to_source_selection(row) for row in rows]

    # Provider object discovery snapshots
    def upsert_provider_object_snapshot(self, snapshot: ProviderObjectSnapshot) -> ProviderObjectSnapshot:
        with self._unit_of_work() as session:
            row = session.scalars(
                select(ProviderObjectSnapshotRow).where(
                    ProviderObjectSnapshotRow.datasource_id == snapshot.datasource_id,
                    ProviderObjectSnapshotRow.external_id == snapshot.external_id,
                )
            ).first()
            if row is None:
                row = _provider_object_snapshot_to_row(snapshot)
                session.add(row)
            else:
                preserved = snapshot.model_copy(
                    update={"id": row.id, "selected_for_sync": row.selected_for_sync},
                    deep=True,
                )
                _apply_provider_object_snapshot(row, preserved)
            session.flush()
            return _row_to_provider_object_snapshot(row)

    def upsert_provider_object_snapshots(
        self,
        datasource_id: str,
        snapshots: list[ProviderObjectSnapshot],
    ) -> list[ProviderObjectSnapshot]:
        saved: list[ProviderObjectSnapshot] = []
        with self.transaction():
            for snapshot in snapshots:
                if snapshot.datasource_id == datasource_id:
                    saved.append(self.upsert_provider_object_snapshot(snapshot))
        return saved

    def get_provider_object_snapshot(self, datasource_id: str, external_id: str) -> ProviderObjectSnapshot:
        with self._unit_of_work() as session:
            row = session.scalars(
                select(ProviderObjectSnapshotRow).where(
                    ProviderObjectSnapshotRow.datasource_id == datasource_id,
                    ProviderObjectSnapshotRow.external_id == external_id,
                )
            ).first()
            if row is None:
                raise NotFoundError(f"ProviderObjectSnapshot not found for DataSource {datasource_id}: {external_id}")
            return _row_to_provider_object_snapshot(row)

    def list_provider_object_snapshots(
        self,
        datasource_id: str | None = None,
        access_state: ProviderObjectAccessState | None = None,
    ) -> list[ProviderObjectSnapshot]:
        with self._unit_of_work() as session:
            statement: Select[tuple[ProviderObjectSnapshotRow]] = select(ProviderObjectSnapshotRow)
            if datasource_id is not None:
                statement = statement.where(ProviderObjectSnapshotRow.datasource_id == datasource_id)
            if access_state is not None:
                statement = statement.where(ProviderObjectSnapshotRow.access_state == str(access_state))
            statement = statement.order_by(ProviderObjectSnapshotRow.title, ProviderObjectSnapshotRow.external_id)
            return [_row_to_provider_object_snapshot(row) for row in session.scalars(statement)]

    def mark_provider_object_selection(
        self,
        datasource_id: str,
        selected_external_object_ids: list[str],
    ) -> list[ProviderObjectSnapshot]:
        selected = set(selected_external_object_ids)
        updated: list[ProviderObjectSnapshot] = []
        with self._unit_of_work() as session:
            rows = session.scalars(
                select(ProviderObjectSnapshotRow).where(ProviderObjectSnapshotRow.datasource_id == datasource_id)
            )
            for row in rows:
                row.selected_for_sync = row.external_id in selected
                updated.append(_row_to_provider_object_snapshot(row))
            session.flush()
        return updated

    # Sync jobs
    def add_sync_job(self, sync_job: SyncJob) -> SyncJob:
        with self._unit_of_work() as session:
            row = _sync_job_to_row(sync_job)
            session.add(row)
            try:
                session.flush()
            except IntegrityError as exc:
                raise PersistenceIntegrityError(str(exc)) from exc
            return _row_to_sync_job(row)

    def update_sync_job(self, sync_job: SyncJob) -> SyncJob:
        with self._unit_of_work() as session:
            row = session.get(SyncJobRow, sync_job.id)
            if row is None:
                raise NotFoundError(f"SyncJob not found: {sync_job.id}")
            _apply_sync_job(row, sync_job)
            session.flush()
            return _row_to_sync_job(row)

    def get_sync_job(self, sync_job_id: str) -> SyncJob:
        with self._unit_of_work() as session:
            row = session.get(SyncJobRow, sync_job_id)
            if row is None:
                raise NotFoundError(f"SyncJob not found: {sync_job_id}")
            return _row_to_sync_job(row)

    def get_sync_job_by_enqueue_key(self, enqueue_key: str) -> SyncJob:
        with self._unit_of_work() as session:
            row = session.scalars(select(SyncJobRow).where(SyncJobRow.enqueue_key == enqueue_key)).first()
            if row is None:
                raise NotFoundError(f"SyncJob not found for enqueue key: {enqueue_key}")
            return _row_to_sync_job(row)

    def claim_sync_job(
        self,
        sync_job_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
        include_not_ready: bool = False,
    ) -> SyncJob | None:
        with self._unit_of_work() as session:
            row = session.scalars(
                select(SyncJobRow)
                .where(SyncJobRow.id == sync_job_id)
                .with_for_update(skip_locked=True)
            ).first()
            if row is None:
                return None
            return self._claim_sync_job_row(
                row,
                session=session,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                include_not_ready=include_not_ready,
            )

    def heartbeat_sync_job_lease(
        self,
        sync_job_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> SyncJob | None:
        with self._unit_of_work() as session:
            row = session.scalars(
                select(SyncJobRow)
                .where(SyncJobRow.id == sync_job_id)
                .with_for_update(skip_locked=True)
            ).first()
            if row is None:
                return None
            job = _row_to_sync_job(row)
            if job.status != SyncJobStatus.RUNNING or job.lease_owner != worker_id:
                return None
            now = utc_now()
            refreshed = job.model_copy(
                update={
                    "lease_heartbeat_at": now,
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                },
                deep=True,
            )
            _apply_sync_job(row, refreshed)
            session.flush()
            return _row_to_sync_job(row)

    def _claim_sync_job_row(
        self,
        row: SyncJobRow,
        *,
        session: Session,
        worker_id: str,
        lease_seconds: int,
        include_not_ready: bool,
    ) -> SyncJob | None:
        job = _row_to_sync_job(row)
        now = utc_now()
        if not _sync_job_is_claimable(job, now=now, include_not_ready=include_not_ready):
            return None
        expired_running = job.status == SyncJobStatus.RUNNING
        claimed = job.model_copy(
            update={
                "status": SyncJobStatus.RUNNING,
                "started_at": now if job.started_at is None or expired_running else job.started_at,
                "finished_at": None,
                "attempt_count": job.attempt_count + 1,
                "next_attempt_at": None,
                "lease_owner": worker_id,
                "lease_acquired_at": now,
                "lease_expires_at": now + timedelta(seconds=lease_seconds),
                "lease_heartbeat_at": now,
            },
            deep=True,
        )
        _apply_sync_job(row, claimed)
        session.flush()
        return _row_to_sync_job(row)

    def list_sync_jobs(self, datasource_id: str | None = None) -> list[SyncJob]:
        with self._unit_of_work() as session:
            statement: Select[tuple[SyncJobRow]] = select(SyncJobRow)
            if datasource_id is not None:
                statement = statement.where(SyncJobRow.datasource_id == datasource_id)
            statement = statement.order_by(SyncJobRow.created_at, SyncJobRow.id)
            return [_row_to_sync_job(row) for row in session.scalars(statement)]

    def add_sync_job_event(self, event: SyncJobEvent) -> SyncJobEvent:
        with self._unit_of_work() as session:
            row = _sync_job_event_to_row(event)
            session.add(row)
            session.flush()
            return _row_to_sync_job_event(row)

    def list_sync_job_events(self, sync_job_id: str) -> list[SyncJobEvent]:
        with self._unit_of_work() as session:
            if session.get(SyncJobRow, sync_job_id) is None:
                raise NotFoundError(f"SyncJob not found: {sync_job_id}")
            rows = session.scalars(
                select(SyncJobEventRow)
                .where(SyncJobEventRow.sync_job_id == sync_job_id)
                .order_by(SyncJobEventRow.occurred_at, SyncJobEventRow.id)
            )
            return [_row_to_sync_job_event(row) for row in rows]

    # Sync cursors
    def upsert_sync_cursor(self, cursor: SyncCursor) -> SyncCursor:
        with self._unit_of_work() as session:
            row = session.scalars(
                select(SyncCursorRow).where(
                    SyncCursorRow.datasource_id == cursor.datasource_id,
                    SyncCursorRow.cursor_key == cursor.cursor_key,
                )
            ).first()
            if row is None:
                row = _sync_cursor_to_row(cursor)
                session.add(row)
            else:
                _apply_sync_cursor(row, cursor.model_copy(update={"id": row.id, "created_at": row.created_at}, deep=True))
            session.flush()
            return _row_to_sync_cursor(row)

    def get_sync_cursor(self, datasource_id: str, cursor_key: str = "default") -> SyncCursor:
        with self._unit_of_work() as session:
            row = session.scalars(
                select(SyncCursorRow).where(
                    SyncCursorRow.datasource_id == datasource_id,
                    SyncCursorRow.cursor_key == cursor_key,
                )
            ).first()
            if row is None:
                raise NotFoundError(f"SyncCursor not found for DataSource: {datasource_id}, key: {cursor_key}")
            return _row_to_sync_cursor(row)

    def list_sync_cursors(self, datasource_id: str | None = None) -> list[SyncCursor]:
        with self._unit_of_work() as session:
            statement: Select[tuple[SyncCursorRow]] = select(SyncCursorRow)
            if datasource_id is not None:
                statement = statement.where(SyncCursorRow.datasource_id == datasource_id)
            statement = statement.order_by(SyncCursorRow.updated_at, SyncCursorRow.id)
            return [_row_to_sync_cursor(row) for row in session.scalars(statement)]

    # Sync schedules
    def upsert_sync_schedule(self, schedule: SyncSchedule) -> SyncSchedule:
        with self._unit_of_work() as session:
            row = session.scalars(
                select(SyncScheduleRow).where(SyncScheduleRow.datasource_id == schedule.datasource_id)
            ).first()
            if row is None:
                row = _sync_schedule_to_row(schedule)
                session.add(row)
            else:
                preserved = schedule.model_copy(update={"id": row.id, "created_at": row.created_at}, deep=True)
                _apply_sync_schedule(row, preserved)
            session.flush()
            return _row_to_sync_schedule(row)

    def get_sync_schedule(self, datasource_id: str) -> SyncSchedule:
        with self._unit_of_work() as session:
            row = session.scalars(
                select(SyncScheduleRow).where(SyncScheduleRow.datasource_id == datasource_id)
            ).first()
            if row is None:
                raise NotFoundError(f"SyncSchedule not found for DataSource: {datasource_id}")
            return _row_to_sync_schedule(row)

    def list_sync_schedules(self, datasource_id: str | None = None) -> list[SyncSchedule]:
        with self._unit_of_work() as session:
            statement: Select[tuple[SyncScheduleRow]] = select(SyncScheduleRow)
            if datasource_id is not None:
                statement = statement.where(SyncScheduleRow.datasource_id == datasource_id)
            statement = statement.order_by(SyncScheduleRow.next_run_at, SyncScheduleRow.created_at, SyncScheduleRow.id)
            return [_row_to_sync_schedule(row) for row in session.scalars(statement)]

    # Artifacts
    def add_artifact(self, artifact: Artifact) -> Artifact:
        with self._unit_of_work() as session:
            row = _artifact_to_row(artifact)
            session.add(row)
            try:
                session.flush()
            except IntegrityError as exc:
                raise PersistenceIntegrityError(str(exc)) from exc
            return _row_to_artifact(row)

    def list_artifacts(self, datasource_id: str | None = None) -> list[Artifact]:
        with self._unit_of_work() as session:
            statement: Select[tuple[ArtifactRow]] = select(ArtifactRow)
            if datasource_id is not None:
                statement = statement.where(ArtifactRow.datasource_id == datasource_id)
            statement = statement.order_by(ArtifactRow.captured_at, ArtifactRow.id)
            return [_row_to_artifact(row) for row in session.scalars(statement)]

    def get_artifact(self, artifact_id: str) -> Artifact:
        with self._unit_of_work() as session:
            row = session.get(ArtifactRow, artifact_id)
            if row is None:
                raise NotFoundError(f"Artifact not found: {artifact_id}")
            return _row_to_artifact(row)

    def find_artifact_by_source_identity(
        self,
        *,
        datasource_id: str,
        source_external_id: str,
        raw_content_hash: str,
    ) -> Artifact | None:
        with self._unit_of_work() as session:
            row = session.scalars(
                select(ArtifactRow).where(
                    ArtifactRow.datasource_id == datasource_id,
                    ArtifactRow.source_external_id == source_external_id,
                    ArtifactRow.raw_content_hash == raw_content_hash,
                )
            ).first()
            return None if row is None else _row_to_artifact(row)

    # Evidence
    def add_evidence_fragment(self, evidence_fragment: EvidenceFragment) -> EvidenceFragment:
        with self._unit_of_work() as session:
            row = _evidence_to_row(evidence_fragment)
            session.add(row)
            try:
                session.flush()
            except IntegrityError as exc:
                raise PersistenceIntegrityError(str(exc)) from exc
            return _row_to_evidence(row)

    def update_evidence_fragment(self, evidence_fragment: EvidenceFragment) -> EvidenceFragment:
        with self._unit_of_work() as session:
            row = session.get(EvidenceFragmentRow, evidence_fragment.id)
            if row is None:
                raise NotFoundError(f"EvidenceFragment not found: {evidence_fragment.id}")
            _apply_evidence(row, evidence_fragment)
            session.flush()
            return _row_to_evidence(row)

    def list_evidence_fragments(self, artifact_id: str | None = None) -> list[EvidenceFragment]:
        with self._unit_of_work() as session:
            statement: Select[tuple[EvidenceFragmentRow]] = select(EvidenceFragmentRow)
            if artifact_id is not None:
                statement = statement.where(EvidenceFragmentRow.artifact_id == artifact_id)
            statement = statement.order_by(EvidenceFragmentRow.id)
            return [_row_to_evidence(row) for row in session.scalars(statement)]

    def get_evidence_fragment(self, evidence_fragment_id: str) -> EvidenceFragment:
        with self._unit_of_work() as session:
            row = session.get(EvidenceFragmentRow, evidence_fragment_id)
            if row is None:
                raise NotFoundError(f"EvidenceFragment not found: {evidence_fragment_id}")
            return _row_to_evidence(row)

    # Concept relations
    def add_concept_relation(self, relation: ConceptRelation) -> ConceptRelation:
        with self._unit_of_work() as session:
            row = _concept_relation_to_row(relation)
            session.add(row)
            session.flush()
            self._replace_ids(
                session,
                concept_relation_evidence_fragments,
                "concept_relation_id",
                relation.id,
                "evidence_fragment_id",
                relation.evidence_fragment_ids,
            )
            return self._load_concept_relation(session, relation.id)

    def update_concept_relation(self, relation: ConceptRelation) -> ConceptRelation:
        with self._unit_of_work() as session:
            row = session.get(ConceptRelationRow, relation.id)
            if row is None:
                raise NotFoundError(f"ConceptRelation not found: {relation.id}")
            _apply_concept_relation(row, relation)
            self._replace_ids(
                session,
                concept_relation_evidence_fragments,
                "concept_relation_id",
                relation.id,
                "evidence_fragment_id",
                relation.evidence_fragment_ids,
            )
            session.flush()
            return self._load_concept_relation(session, relation.id)

    def get_concept_relation(self, relation_id: str) -> ConceptRelation:
        with self._unit_of_work() as session:
            return self._load_concept_relation(session, relation_id)

    def list_concept_relations(self, concept_id: str | None = None) -> list[ConceptRelation]:
        with self._unit_of_work() as session:
            statement: Select[tuple[ConceptRelationRow]] = select(ConceptRelationRow)
            if concept_id is not None:
                statement = statement.where(
                    (ConceptRelationRow.source_concept_id == concept_id)
                    | (ConceptRelationRow.target_concept_id == concept_id)
                )
            statement = statement.order_by(ConceptRelationRow.created_at, ConceptRelationRow.id)
            return [self._row_to_concept_relation_with_links(session, row) for row in session.scalars(statement)]

    # Decision records
    def add_decision_record(self, decision_record: DecisionRecord) -> DecisionRecord:
        with self._unit_of_work() as session:
            row = _decision_record_to_row(decision_record)
            session.add(row)
            session.flush()
            self._replace_ids(
                session,
                decision_record_evidence_fragments,
                "decision_record_id",
                decision_record.id,
                "evidence_fragment_id",
                decision_record.evidence_fragment_ids,
            )
            self._replace_ids(
                session,
                decision_record_affected_concepts,
                "decision_record_id",
                decision_record.id,
                "concept_id",
                decision_record.affected_concept_ids,
            )
            return self._load_decision_record(session, decision_record.id)

    def get_decision_record(self, decision_record_id: str) -> DecisionRecord:
        with self._unit_of_work() as session:
            return self._load_decision_record(session, decision_record_id)

    def list_decision_records(self) -> list[DecisionRecord]:
        with self._unit_of_work() as session:
            rows = session.scalars(select(DecisionRecordRow).order_by(DecisionRecordRow.created_at, DecisionRecordRow.id))
            return [self._row_to_decision_record_with_links(session, row) for row in rows]

    # Concepts
    def add_concept(self, concept: Concept) -> Concept:
        with self._unit_of_work() as session:
            row = _concept_to_row(concept)
            session.add(row)
            session.flush()
            self._replace_ids(
                session,
                concept_evidence_fragments,
                "concept_id",
                concept.id,
                "evidence_fragment_id",
                concept.evidence_fragment_ids,
            )
            self._replace_ids(
                session,
                concept_decision_records,
                "concept_id",
                concept.id,
                "decision_record_id",
                concept.decision_record_ids,
            )
            return self._load_concept(session, concept.id)

    def update_concept(self, concept: Concept) -> Concept:
        with self._unit_of_work() as session:
            row = session.get(ConceptRow, concept.id)
            if row is None:
                raise NotFoundError(f"Concept not found: {concept.id}")
            _apply_concept(row, concept)
            self._replace_ids(
                session,
                concept_evidence_fragments,
                "concept_id",
                concept.id,
                "evidence_fragment_id",
                concept.evidence_fragment_ids,
            )
            self._replace_ids(
                session,
                concept_decision_records,
                "concept_id",
                concept.id,
                "decision_record_id",
                concept.decision_record_ids,
            )
            session.flush()
            return self._load_concept(session, concept.id)

    def get_concept(self, concept_id: str) -> Concept:
        with self._unit_of_work() as session:
            return self._load_concept(session, concept_id)

    def list_concepts(self) -> list[Concept]:
        with self._unit_of_work() as session:
            rows = session.scalars(select(ConceptRow).order_by(ConceptRow.created_at, ConceptRow.id))
            return [self._row_to_concept_with_links(session, row) for row in rows]


    # Evaluation framework
    def add_grounded_context_eval_task(self, task: GroundedContextEvalTask) -> GroundedContextEvalTask:
        with self._unit_of_work() as session:
            row = _grounded_context_eval_task_to_row(task)
            session.add(row)
            session.flush()
            return _row_to_grounded_context_eval_task(row)

    def get_grounded_context_eval_task(self, task_id: str) -> GroundedContextEvalTask:
        with self._unit_of_work() as session:
            row = session.get(GroundedContextEvalTaskRow, task_id)
            if row is None:
                raise NotFoundError(f"GroundedContextEvalTask not found: {task_id}")
            return _row_to_grounded_context_eval_task(row)

    def list_grounded_context_eval_tasks(self) -> list[GroundedContextEvalTask]:
        with self._unit_of_work() as session:
            rows = session.scalars(
                select(GroundedContextEvalTaskRow).order_by(GroundedContextEvalTaskRow.created_at, GroundedContextEvalTaskRow.id)
            )
            return [_row_to_grounded_context_eval_task(row) for row in rows]

    def add_grounded_context_eval_result(self, result: GroundedContextEvalResult) -> GroundedContextEvalResult:
        with self._unit_of_work() as session:
            if session.get(GroundedContextEvalTaskRow, result.task_id) is None:
                raise NotFoundError(f"GroundedContextEvalTask not found: {result.task_id}")
            row = _grounded_context_eval_result_to_row(result)
            session.add(row)
            session.flush()
            return _row_to_grounded_context_eval_result(row)

    def get_grounded_context_eval_result(self, result_id: str) -> GroundedContextEvalResult:
        with self._unit_of_work() as session:
            row = session.get(GroundedContextEvalResultRow, result_id)
            if row is None:
                raise NotFoundError(f"GroundedContextEvalResult not found: {result_id}")
            return _row_to_grounded_context_eval_result(row)

    def list_grounded_context_eval_results(self, task_id: str | None = None) -> list[GroundedContextEvalResult]:
        with self._unit_of_work() as session:
            statement: Select[tuple[GroundedContextEvalResultRow]] = select(GroundedContextEvalResultRow)
            if task_id is not None:
                statement = statement.where(GroundedContextEvalResultRow.task_id == task_id)
            statement = statement.order_by(GroundedContextEvalResultRow.evaluated_at, GroundedContextEvalResultRow.id)
            return [_row_to_grounded_context_eval_result(row) for row in session.scalars(statement)]

    # Compatibility aliases for eval naming.
    add_eval_task = add_grounded_context_eval_task
    get_eval_task = get_grounded_context_eval_task
    list_eval_tasks = list_grounded_context_eval_tasks
    add_eval_result = add_grounded_context_eval_result
    get_eval_result = get_grounded_context_eval_result
    list_eval_results = list_grounded_context_eval_results

    # Audit events
    def add_audit_event(self, event: AuditEvent) -> AuditEvent:
        with self._unit_of_work() as session:
            row = _audit_event_to_row(event)
            session.add(row)
            session.flush()
            return _row_to_audit_event(row)

    def list_audit_events(self) -> list[AuditEvent]:
        with self._unit_of_work() as session:
            rows = session.scalars(select(AuditEventRow).order_by(AuditEventRow.occurred_at, AuditEventRow.id))
            return [_row_to_audit_event(row) for row in rows]

    @contextmanager
    def _unit_of_work(self) -> Iterator[Session]:
        existing_session = self._current_session.get()
        if existing_session is not None:
            yield existing_session
            return
        with self._session_factory() as session:
            try:
                with session.begin():
                    yield session
            except IntegrityError as exc:
                raise PersistenceIntegrityError(str(exc)) from exc

    def _replace_ids(
        self,
        session: Session,
        table: sa.Table,
        owner_column: str,
        owner_id: str,
        linked_column: str,
        linked_ids: list[str],
    ) -> None:
        session.execute(delete(table).where(table.c[owner_column] == owner_id))
        for linked_id in linked_ids:
            session.execute(
                table.insert().values({owner_column: owner_id, linked_column: linked_id})
            )

    def _load_concept(self, session: Session, concept_id: str) -> Concept:
        row = session.get(ConceptRow, concept_id)
        if row is None:
            raise NotFoundError(f"Concept not found: {concept_id}")
        return self._row_to_concept_with_links(session, row)

    def _row_to_concept_with_links(self, session: Session, row: ConceptRow) -> Concept:
        evidence_ids = list(
            session.execute(
                select(concept_evidence_fragments.c.evidence_fragment_id)
                .where(concept_evidence_fragments.c.concept_id == row.id)
                .order_by(concept_evidence_fragments.c.evidence_fragment_id)
            ).scalars()
        )
        decision_ids = list(
            session.execute(
                select(concept_decision_records.c.decision_record_id)
                .where(concept_decision_records.c.concept_id == row.id)
                .order_by(concept_decision_records.c.decision_record_id)
            ).scalars()
        )
        return _row_to_concept(row, evidence_ids=evidence_ids, decision_record_ids=decision_ids)

    def _load_concept_relation(self, session: Session, relation_id: str) -> ConceptRelation:
        row = session.get(ConceptRelationRow, relation_id)
        if row is None:
            raise NotFoundError(f"ConceptRelation not found: {relation_id}")
        return self._row_to_concept_relation_with_links(session, row)

    def _row_to_concept_relation_with_links(
        self, session: Session, row: ConceptRelationRow
    ) -> ConceptRelation:
        evidence_ids = list(
            session.execute(
                select(concept_relation_evidence_fragments.c.evidence_fragment_id)
                .where(concept_relation_evidence_fragments.c.concept_relation_id == row.id)
                .order_by(concept_relation_evidence_fragments.c.evidence_fragment_id)
            ).scalars()
        )
        return _row_to_concept_relation(row, evidence_fragment_ids=evidence_ids)

    def _load_decision_record(self, session: Session, decision_record_id: str) -> DecisionRecord:
        row = session.get(DecisionRecordRow, decision_record_id)
        if row is None:
            raise NotFoundError(f"DecisionRecord not found: {decision_record_id}")
        return self._row_to_decision_record_with_links(session, row)

    def _row_to_decision_record_with_links(
        self, session: Session, row: DecisionRecordRow
    ) -> DecisionRecord:
        evidence_ids = list(
            session.execute(
                select(decision_record_evidence_fragments.c.evidence_fragment_id)
                .where(decision_record_evidence_fragments.c.decision_record_id == row.id)
                .order_by(decision_record_evidence_fragments.c.evidence_fragment_id)
            ).scalars()
        )
        concept_ids = list(
            session.execute(
                select(decision_record_affected_concepts.c.concept_id)
                .where(decision_record_affected_concepts.c.decision_record_id == row.id)
                .order_by(decision_record_affected_concepts.c.concept_id)
            ).scalars()
        )
        return _row_to_decision_record(
            row,
            evidence_fragment_ids=evidence_ids,
            affected_concept_ids=concept_ids,
        )


def create_sqlalchemy_engine(database_url: str, *, echo: bool = False) -> Engine:
    return create_engine(database_url, echo=echo, pool_pre_ping=True, future=True)


def create_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)


# Mapping helpers

def _json_model(model: Any) -> dict[str, Any] | None:
    if model is None:
        return None
    return cast(dict[str, Any], model.model_dump(mode="json", by_alias=False))


def _data_source_to_row(data_source: DataSource) -> DataSourceRow:
    row = DataSourceRow()
    _apply_data_source(row, data_source)
    return row


def _apply_data_source(row: DataSourceRow, data_source: DataSource) -> None:
    row.id = data_source.id
    row.type = str(data_source.type)
    row.name = data_source.name
    row.status = str(data_source.status)
    row.production_enabled = data_source.production_enabled
    row.created_at = data_source.created_at
    row.auth_status = str(data_source.auth_status)
    row.connection_status = str(data_source.connection_status)
    row.sync_status = str(data_source.sync_status)
    row.next_action = str(data_source.next_action)
    row.last_connection_test_at = data_source.last_connection_test_at
    row.last_discovery_at = data_source.last_discovery_at
    row.last_sync_at = data_source.last_sync_at
    row.last_successful_sync_at = data_source.last_successful_sync_at
    row.last_error = _json_model(data_source.last_error)
    row.freshness_state = str(data_source.freshness_state)
    row.sync_freshness_state = str(data_source.sync_freshness_state)
    row.content_freshness_state = str(data_source.content_freshness_state)
    row.artifact_count = data_source.artifact_count
    row.evidence_fragment_count = data_source.evidence_fragment_count
    row.discovered_object_count = data_source.discovered_object_count
    row.selected_object_count = data_source.selected_object_count


def _row_to_data_source(row: DataSourceRow) -> DataSource:
    return DataSource(
        id=row.id,
        type=DataSourceType(row.type),
        name=row.name,
        status=DataSourceStatus(row.status),
        production_enabled=row.production_enabled,
        created_at=row.created_at,
        auth_status=DataSourceAuthStatus(row.auth_status),
        connection_status=DataSourceConnectionStatus(row.connection_status),
        sync_status=DataSourceSyncStatus(row.sync_status),
        next_action=DataSourceNextAction(row.next_action),
        last_connection_test_at=row.last_connection_test_at,
        last_discovery_at=row.last_discovery_at,
        last_sync_at=row.last_sync_at,
        last_successful_sync_at=row.last_successful_sync_at,
        last_error=None if row.last_error is None else ErrorInfo.model_validate(row.last_error),
        freshness_state=FreshnessState(row.freshness_state),
        sync_freshness_state=FreshnessState(row.sync_freshness_state),
        content_freshness_state=FreshnessState(row.content_freshness_state),
        artifact_count=row.artifact_count,
        evidence_fragment_count=row.evidence_fragment_count,
        discovered_object_count=row.discovered_object_count,
        selected_object_count=row.selected_object_count,
    )


def _connection_intent_to_row(intent: ConnectionIntent) -> ConnectionIntentRow:
    row = ConnectionIntentRow()
    _apply_connection_intent(row, intent)
    return row


def _apply_connection_intent(row: ConnectionIntentRow, intent: ConnectionIntent) -> None:
    row.id = intent.id
    row.provider = str(intent.provider)
    row.status = str(intent.status)
    row.auth_type = str(intent.auth_type)
    row.source_name = intent.source_name
    row.created_by = intent.created_by
    row.requested_scopes = list(intent.requested_scopes)
    row.authorization_url = intent.authorization_url
    row.redirect_uri = intent.redirect_uri
    row.return_url = intent.return_url
    row.state_nonce = intent.state_nonce
    row.expires_at = intent.expires_at
    row.created_at = intent.created_at
    row.completed_at = intent.completed_at
    row.datasource_id = intent.datasource_id
    row.failure_error = _json_model(intent.failure_error)


def _row_to_connection_intent(row: ConnectionIntentRow) -> ConnectionIntent:
    return ConnectionIntent(
        id=row.id,
        provider=DataSourceType(row.provider),
        status=ConnectionIntentStatus(row.status),
        auth_type=ConnectorAuthType(row.auth_type),
        source_name=row.source_name,
        created_by=row.created_by,
        requested_scopes=list(row.requested_scopes or []),
        authorization_url=row.authorization_url,
        redirect_uri=row.redirect_uri,
        return_url=row.return_url,
        state_nonce=row.state_nonce,
        expires_at=row.expires_at,
        created_at=row.created_at,
        completed_at=row.completed_at,
        datasource_id=row.datasource_id,
        failure_error=None if row.failure_error is None else ConnectorError.model_validate(row.failure_error),
    )


def _connector_credential_to_row(credential: ConnectorCredential) -> ConnectorCredentialRow:
    row = ConnectorCredentialRow()
    _apply_connector_credential(row, credential)
    return row


def _apply_connector_credential(row: ConnectorCredentialRow, credential: ConnectorCredential) -> None:
    row.id = credential.id
    row.datasource_id = credential.datasource_id
    row.provider = str(credential.provider)
    row.auth_type = str(credential.auth_type)
    row.encrypted_access_token = credential.encrypted_access_token
    row.encrypted_refresh_token = credential.encrypted_refresh_token
    row.granted_scopes = list(credential.granted_scopes)
    row.external_account_id = credential.external_account_id
    row.external_workspace_id = credential.external_workspace_id
    row.external_workspace_name = credential.external_workspace_name
    row.external_bot_id = credential.external_bot_id
    row.token_expires_at = credential.token_expires_at
    row.status = str(credential.status)
    row.created_at = credential.created_at
    row.updated_at = credential.updated_at
    row.revoked_at = credential.revoked_at


def _row_to_connector_credential(row: ConnectorCredentialRow) -> ConnectorCredential:
    return ConnectorCredential(
        id=row.id,
        datasource_id=row.datasource_id,
        provider=DataSourceType(row.provider),
        auth_type=ConnectorAuthType(row.auth_type),
        encrypted_access_token=row.encrypted_access_token,
        encrypted_refresh_token=row.encrypted_refresh_token,
        granted_scopes=list(row.granted_scopes or []),
        external_account_id=row.external_account_id,
        external_workspace_id=row.external_workspace_id,
        external_workspace_name=row.external_workspace_name,
        external_bot_id=row.external_bot_id,
        token_expires_at=row.token_expires_at,
        status=CredentialStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
        revoked_at=row.revoked_at,
    )


def _source_selection_to_row(selection: SourceSelection) -> SourceSelectionRow:
    row = SourceSelectionRow()
    _apply_source_selection(row, selection)
    return row


def _apply_source_selection(row: SourceSelectionRow, selection: SourceSelection) -> None:
    row.id = selection.id
    row.datasource_id = selection.datasource_id
    row.sync_mode = str(selection.sync_mode)
    row.include_rules = [rule.model_dump(mode="json", by_alias=False) for rule in selection.include_rules]
    row.exclude_rules = [rule.model_dump(mode="json", by_alias=False) for rule in selection.exclude_rules]
    row.selected_external_object_ids = list(selection.selected_external_object_ids)
    row.created_at = selection.created_at
    row.updated_at = selection.updated_at


def _row_to_source_selection(row: SourceSelectionRow) -> SourceSelection:
    return SourceSelection(
        id=row.id,
        datasource_id=row.datasource_id,
        sync_mode=SourceSelectionMode(row.sync_mode),
        include_rules=[SelectionRule.model_validate(item) for item in row.include_rules or []],
        exclude_rules=[SelectionRule.model_validate(item) for item in row.exclude_rules or []],
        selected_external_object_ids=list(row.selected_external_object_ids or []),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _provider_object_snapshot_to_row(snapshot: ProviderObjectSnapshot) -> ProviderObjectSnapshotRow:
    row = ProviderObjectSnapshotRow()
    _apply_provider_object_snapshot(row, snapshot)
    return row


def _apply_provider_object_snapshot(row: ProviderObjectSnapshotRow, snapshot: ProviderObjectSnapshot) -> None:
    row.id = snapshot.id
    row.datasource_id = snapshot.datasource_id
    row.provider = str(snapshot.provider)
    row.external_id = snapshot.external_id
    row.external_url = snapshot.external_url
    row.object_type = str(snapshot.object_type)
    row.title = snapshot.title
    row.parent_external_id = snapshot.parent_external_id
    row.last_edited_time = snapshot.last_edited_time
    row.discovered_at = snapshot.discovered_at
    row.selected_for_sync = snapshot.selected_for_sync
    row.access_state = str(snapshot.access_state)
    row.ingestion_supported = snapshot.ingestion_supported
    row.ingestion_unsupported_reason = snapshot.ingestion_unsupported_reason
    row.raw_metadata_hash = snapshot.raw_metadata_hash
    row.provider_metadata = dict(snapshot.provider_metadata)


def _row_to_provider_object_snapshot(row: ProviderObjectSnapshotRow) -> ProviderObjectSnapshot:
    return ProviderObjectSnapshot(
        id=row.id,
        datasource_id=row.datasource_id,
        provider=DataSourceType(row.provider),
        external_id=row.external_id,
        external_url=row.external_url,
        object_type=ProviderObjectType(row.object_type),
        title=row.title,
        parent_external_id=row.parent_external_id,
        last_edited_time=row.last_edited_time,
        discovered_at=row.discovered_at,
        selected_for_sync=row.selected_for_sync,
        access_state=ProviderObjectAccessState(row.access_state),
        ingestion_supported=row.ingestion_supported,
        ingestion_unsupported_reason=row.ingestion_unsupported_reason,
        raw_metadata_hash=row.raw_metadata_hash,
        provider_metadata=dict(row.provider_metadata or {}),
    )


def _sync_job_to_row(sync_job: SyncJob) -> SyncJobRow:
    row = SyncJobRow()
    _apply_sync_job(row, sync_job)
    return row


def _apply_sync_job(row: SyncJobRow, sync_job: SyncJob) -> None:
    row.id = sync_job.id
    row.datasource_id = sync_job.datasource_id
    row.provider = str(sync_job.provider)
    row.status = str(sync_job.status)
    row.trigger = str(sync_job.trigger)
    row.created_by = sync_job.created_by
    row.selection_id = sync_job.selection_id
    row.created_at = sync_job.created_at
    row.started_at = sync_job.started_at
    row.finished_at = sync_job.finished_at
    row.artifact_created_count = sync_job.artifact_created_count
    row.artifact_reused_count = sync_job.artifact_reused_count
    row.evidence_created_count = sync_job.evidence_created_count
    row.error = _json_model(sync_job.error)
    row.cursor = sync_job.cursor
    row.attempt_count = sync_job.attempt_count
    row.max_attempts = sync_job.max_attempts
    row.next_attempt_at = sync_job.next_attempt_at
    row.cancel_requested_at = sync_job.cancel_requested_at
    row.cancelled_by = sync_job.cancelled_by
    row.lease_owner = sync_job.lease_owner
    row.lease_acquired_at = sync_job.lease_acquired_at
    row.lease_expires_at = sync_job.lease_expires_at
    row.lease_heartbeat_at = sync_job.lease_heartbeat_at
    row.schedule_id = sync_job.schedule_id
    row.enqueue_key = sync_job.enqueue_key


def _row_to_sync_job(row: SyncJobRow) -> SyncJob:
    return SyncJob(
        id=row.id,
        datasource_id=row.datasource_id,
        provider=DataSourceType(row.provider),
        status=SyncJobStatus(row.status),
        trigger=SyncJobTrigger(row.trigger),
        created_by=row.created_by,
        selection_id=row.selection_id,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        artifact_created_count=row.artifact_created_count,
        artifact_reused_count=row.artifact_reused_count,
        evidence_created_count=row.evidence_created_count,
        error=None if row.error is None else ConnectorError.model_validate(row.error),
        cursor=row.cursor,
        attempt_count=row.attempt_count,
        max_attempts=row.max_attempts,
        next_attempt_at=row.next_attempt_at,
        cancel_requested_at=row.cancel_requested_at,
        cancelled_by=row.cancelled_by,
        lease_owner=row.lease_owner,
        lease_acquired_at=row.lease_acquired_at,
        lease_expires_at=row.lease_expires_at,
        lease_heartbeat_at=row.lease_heartbeat_at,
        schedule_id=row.schedule_id,
        enqueue_key=row.enqueue_key,
    )


def _sync_cursor_to_row(cursor: SyncCursor) -> SyncCursorRow:
    row = SyncCursorRow()
    _apply_sync_cursor(row, cursor)
    return row


def _apply_sync_cursor(row: SyncCursorRow, cursor: SyncCursor) -> None:
    row.id = cursor.id
    row.datasource_id = cursor.datasource_id
    row.provider = str(cursor.provider)
    row.cursor_key = cursor.cursor_key
    row.last_cursor = cursor.last_cursor
    row.last_successful_sync_job_id = cursor.last_successful_sync_job_id
    row.processed_external_object_ids = list(cursor.processed_external_object_ids)
    row.artifact_created_count = cursor.artifact_created_count
    row.artifact_reused_count = cursor.artifact_reused_count
    row.evidence_created_count = cursor.evidence_created_count
    row.created_at = cursor.created_at
    row.updated_at = cursor.updated_at
    row.advanced_at = cursor.advanced_at


def _row_to_sync_cursor(row: SyncCursorRow) -> SyncCursor:
    return SyncCursor(
        id=row.id,
        datasource_id=row.datasource_id,
        provider=DataSourceType(row.provider),
        cursor_key=row.cursor_key,
        last_cursor=row.last_cursor,
        last_successful_sync_job_id=row.last_successful_sync_job_id,
        processed_external_object_ids=list(row.processed_external_object_ids or []),
        artifact_created_count=row.artifact_created_count,
        artifact_reused_count=row.artifact_reused_count,
        evidence_created_count=row.evidence_created_count,
        created_at=row.created_at,
        updated_at=row.updated_at,
        advanced_at=row.advanced_at,
    )


def _sync_schedule_to_row(schedule: SyncSchedule) -> SyncScheduleRow:
    row = SyncScheduleRow()
    _apply_sync_schedule(row, schedule)
    return row


def _apply_sync_schedule(row: SyncScheduleRow, schedule: SyncSchedule) -> None:
    row.id = schedule.id
    row.datasource_id = schedule.datasource_id
    row.provider = str(schedule.provider)
    row.status = str(schedule.status)
    row.interval_minutes = schedule.interval_minutes
    row.next_run_at = schedule.next_run_at
    row.last_enqueued_at = schedule.last_enqueued_at
    row.last_enqueued_sync_job_id = schedule.last_enqueued_sync_job_id
    row.max_attempts = schedule.max_attempts
    row.created_by = schedule.created_by
    row.created_at = schedule.created_at
    row.updated_at = schedule.updated_at


def _row_to_sync_schedule(row: SyncScheduleRow) -> SyncSchedule:
    return SyncSchedule(
        id=row.id,
        datasource_id=row.datasource_id,
        provider=DataSourceType(row.provider),
        status=SyncScheduleStatus(row.status),
        interval_minutes=row.interval_minutes,
        next_run_at=row.next_run_at,
        last_enqueued_at=row.last_enqueued_at,
        last_enqueued_sync_job_id=row.last_enqueued_sync_job_id,
        max_attempts=row.max_attempts,
        created_by=row.created_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _sync_job_event_to_row(event: SyncJobEvent) -> SyncJobEventRow:
    return SyncJobEventRow(
        id=event.id,
        sync_job_id=event.sync_job_id,
        datasource_id=event.datasource_id,
        event_type=event.event_type,
        message=event.message,
        occurred_at=event.occurred_at,
        metadata_=dict(event.metadata),
    )


def _row_to_sync_job_event(row: SyncJobEventRow) -> SyncJobEvent:
    return SyncJobEvent(
        id=row.id,
        sync_job_id=row.sync_job_id,
        datasource_id=row.datasource_id,
        event_type=row.event_type,
        message=row.message,
        occurred_at=row.occurred_at,
        metadata=dict(row.metadata_ or {}),
    )

def _artifact_to_row(artifact: Artifact) -> ArtifactRow:
    return ArtifactRow(
        id=artifact.id,
        datasource_id=artifact.datasource_id,
        source_type=str(artifact.source_type),
        source_external_id=artifact.source_external_id,
        source_url=artifact.source_url,
        source_object_type=artifact.source_object_type,
        title=artifact.title,
        raw_content_hash=artifact.raw_content_hash,
        captured_at=artifact.captured_at,
        source_updated_at=artifact.source_updated_at,
        freshness_state=str(artifact.freshness_state),
        extraction_status=str(artifact.extraction_status),
        provider_metadata=dict(artifact.provider_metadata),
    )


def _row_to_artifact(row: ArtifactRow) -> Artifact:
    return Artifact(
        id=row.id,
        datasource_id=row.datasource_id,
        source_type=DataSourceType(row.source_type),
        source_external_id=row.source_external_id,
        source_url=row.source_url,
        source_object_type=row.source_object_type,
        title=row.title,
        raw_content_hash=row.raw_content_hash,
        captured_at=row.captured_at,
        source_updated_at=row.source_updated_at,
        freshness_state=FreshnessState(row.freshness_state),
        extraction_status=ExtractionStatus(row.extraction_status),
        provider_metadata=dict(row.provider_metadata or {}),
    )


def _evidence_to_row(evidence: EvidenceFragment) -> EvidenceFragmentRow:
    return EvidenceFragmentRow(
        id=evidence.id,
        artifact_id=evidence.artifact_id,
        text=evidence.text,
        fragment_type=str(evidence.fragment_type),
        provenance=evidence.provenance.model_dump(mode="json", by_alias=False),
        trust_state=str(evidence.trust_state),
        freshness_state=str(evidence.freshness_state),
        reviewed_by=evidence.reviewed_by,
        reviewed_at=evidence.reviewed_at,
    )


def _apply_evidence(row: EvidenceFragmentRow, evidence: EvidenceFragment) -> None:
    row.artifact_id = evidence.artifact_id
    row.text = evidence.text
    row.fragment_type = str(evidence.fragment_type)
    row.provenance = evidence.provenance.model_dump(mode="json", by_alias=False)
    row.trust_state = str(evidence.trust_state)
    row.freshness_state = str(evidence.freshness_state)
    row.reviewed_by = evidence.reviewed_by
    row.reviewed_at = evidence.reviewed_at


def _row_to_evidence(row: EvidenceFragmentRow) -> EvidenceFragment:
    return EvidenceFragment(
        id=row.id,
        artifact_id=row.artifact_id,
        text=row.text,
        fragment_type=EvidenceFragmentType(row.fragment_type),
        provenance=Provenance.model_validate(row.provenance),
        trust_state=TrustState(row.trust_state),
        freshness_state=FreshnessState(row.freshness_state),
        reviewed_by=row.reviewed_by,
        reviewed_at=row.reviewed_at,
    )


def _decision_record_to_row(decision_record: DecisionRecord) -> DecisionRecordRow:
    return DecisionRecordRow(
        id=decision_record.id,
        title=decision_record.title,
        decision=decision_record.decision,
        reason=decision_record.reason,
        alternatives_considered=decision_record.alternatives_considered,
        decided_by=decision_record.decided_by,
        decided_at=decision_record.decided_at,
        created_at=decision_record.created_at,
        updated_at=decision_record.updated_at,
    )


def _row_to_decision_record(
    row: DecisionRecordRow,
    *,
    evidence_fragment_ids: list[str],
    affected_concept_ids: list[str],
) -> DecisionRecord:
    return DecisionRecord(
        id=row.id,
        title=row.title,
        decision=row.decision,
        reason=row.reason,
        alternatives_considered=list(row.alternatives_considered or []),
        decided_by=row.decided_by,
        decided_at=row.decided_at,
        evidence_fragment_ids=evidence_fragment_ids,
        affected_concept_ids=affected_concept_ids,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _concept_to_row(concept: Concept) -> ConceptRow:
    row = ConceptRow()
    _apply_concept(row, concept)
    return row


def _apply_concept(row: ConceptRow, concept: Concept) -> None:
    row.id = concept.id
    row.name = concept.name
    row.short_definition = concept.short_definition
    row.body = concept.body
    row.status = str(concept.status)
    row.owner = concept.owner
    row.created_by = concept.created_by
    row.officialized_by = concept.officialized_by
    row.created_at = concept.created_at
    row.updated_at = concept.updated_at
    row.last_reviewed_at = concept.last_reviewed_at


def _row_to_concept(
    row: ConceptRow,
    *,
    evidence_ids: list[str],
    decision_record_ids: list[str],
) -> Concept:
    return Concept(
        id=row.id,
        name=row.name,
        short_definition=row.short_definition,
        body=row.body,
        status=ConceptStatus(row.status),
        owner=row.owner,
        evidence_fragment_ids=evidence_ids,
        decision_record_ids=decision_record_ids,
        created_by=row.created_by,
        officialized_by=row.officialized_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
        last_reviewed_at=row.last_reviewed_at,
    )


def _concept_relation_to_row(relation: ConceptRelation) -> ConceptRelationRow:
    row = ConceptRelationRow()
    _apply_concept_relation(row, relation)
    return row


def _apply_concept_relation(row: ConceptRelationRow, relation: ConceptRelation) -> None:
    row.id = relation.id
    row.source_concept_id = relation.source_concept_id
    row.target_concept_id = relation.target_concept_id
    row.relation_type = str(relation.relation_type)
    row.status = str(relation.status)
    row.decision_record_id = relation.decision_record_id
    row.created_by = relation.created_by
    row.officialized_by = relation.officialized_by
    row.created_at = relation.created_at
    row.updated_at = relation.updated_at
    row.last_reviewed_at = relation.last_reviewed_at


def _row_to_concept_relation(
    row: ConceptRelationRow,
    *,
    evidence_fragment_ids: list[str],
) -> ConceptRelation:
    return ConceptRelation(
        id=row.id,
        source_concept_id=row.source_concept_id,
        target_concept_id=row.target_concept_id,
        relation_type=RelationType(row.relation_type),
        status=RelationStatus(row.status),
        evidence_fragment_ids=evidence_fragment_ids,
        decision_record_id=row.decision_record_id,
        created_by=row.created_by,
        officialized_by=row.officialized_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
        last_reviewed_at=row.last_reviewed_at,
    )



def _grounded_context_eval_task_to_row(task: GroundedContextEvalTask) -> GroundedContextEvalTaskRow:
    return GroundedContextEvalTaskRow(
        id=task.id,
        name=task.name,
        query=task.query,
        expected_answer_contains=list(task.expected_answer_contains),
        expected_trust_label=None if task.expected_trust_label is None else str(task.expected_trust_label),
        expected_freshness_state=None if task.expected_freshness_state is None else str(task.expected_freshness_state),
        required_evidence_fragment_ids=list(task.required_evidence_fragment_ids),
        required_concept_ids=list(task.required_concept_ids),
        required_decision_record_ids=list(task.required_decision_record_ids),
        require_official_answer=task.require_official_answer,
        require_evidence=task.require_evidence,
        min_evidence_count=task.min_evidence_count,
        expected_clarification_reduced=task.expected_clarification_reduced,
        tags=list(task.tags),
        created_by=task.created_by,
        created_at=task.created_at,
        updated_at=task.updated_at,
        metadata_=dict(task.metadata),
    )


def _row_to_grounded_context_eval_task(row: GroundedContextEvalTaskRow) -> GroundedContextEvalTask:
    return GroundedContextEvalTask(
        id=row.id,
        name=row.name,
        query=row.query,
        expected_answer_contains=list(row.expected_answer_contains or []),
        expected_trust_label=None if row.expected_trust_label is None else TrustLabel(row.expected_trust_label),
        expected_freshness_state=None if row.expected_freshness_state is None else FreshnessState(row.expected_freshness_state),
        required_evidence_fragment_ids=list(row.required_evidence_fragment_ids or []),
        required_concept_ids=list(row.required_concept_ids or []),
        required_decision_record_ids=list(row.required_decision_record_ids or []),
        require_official_answer=row.require_official_answer,
        require_evidence=row.require_evidence,
        min_evidence_count=row.min_evidence_count,
        expected_clarification_reduced=row.expected_clarification_reduced,
        tags=list(row.tags or []),
        created_by=row.created_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
        metadata=dict(row.metadata_ or {}),
    )


def _grounded_context_eval_result_to_row(result: GroundedContextEvalResult) -> GroundedContextEvalResultRow:
    return GroundedContextEvalResultRow(
        id=result.id,
        task_id=result.task_id,
        response_id=result.response_id,
        query=result.query,
        answer=result.answer,
        trust_label=str(result.trust_label),
        response=result.response.model_dump(mode="json", by_alias=False),
        answer_correct=result.answer_correct,
        evidence_valid=result.evidence_valid,
        provenance_present=result.provenance_present,
        trust_label_correct=result.trust_label_correct,
        freshness_policy_respected=result.freshness_policy_respected,
        unsupported_official_claim=result.unsupported_official_claim,
        citation_validity_rate=result.citation_validity_rate,
        clarification_reduced=result.clarification_reduced,
        success=result.success,
        failure_reasons=list(result.failure_reasons),
        evaluated_at=result.evaluated_at,
        evaluated_by=result.evaluated_by,
    )


def _row_to_grounded_context_eval_result(row: GroundedContextEvalResultRow) -> GroundedContextEvalResult:
    return GroundedContextEvalResult(
        id=row.id,
        task_id=row.task_id,
        response_id=row.response_id,
        query=row.query,
        answer=row.answer,
        trust_label=TrustLabel(row.trust_label),
        response=GroundedContextResponse.model_validate(row.response),
        answer_correct=row.answer_correct,
        evidence_valid=row.evidence_valid,
        provenance_present=row.provenance_present,
        trust_label_correct=row.trust_label_correct,
        freshness_policy_respected=row.freshness_policy_respected,
        unsupported_official_claim=row.unsupported_official_claim,
        citation_validity_rate=row.citation_validity_rate,
        clarification_reduced=row.clarification_reduced,
        success=row.success,
        failure_reasons=list(row.failure_reasons or []),
        evaluated_at=row.evaluated_at,
        evaluated_by=row.evaluated_by,
    )


def _audit_event_to_row(event: AuditEvent) -> AuditEventRow:
    return AuditEventRow(
        id=event.id,
        event_type=event.event_type,
        actor=event.actor,
        entity_type=event.entity_type,
        entity_id=event.entity_id,
        occurred_at=event.occurred_at,
        metadata_=event.metadata,
    )


def _row_to_audit_event(row: AuditEventRow) -> AuditEvent:
    return AuditEvent(
        id=row.id,
        event_type=row.event_type,
        actor=row.actor,
        entity_type=row.entity_type,
        entity_id=row.entity_id,
        occurred_at=row.occurred_at,
        metadata=dict(row.metadata_ or {}),
    )


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _sync_job_is_claimable(job: SyncJob, *, now: datetime, include_not_ready: bool) -> bool:
    if job.status == SyncJobStatus.QUEUED:
        return True
    if job.status == SyncJobStatus.RETRY_WAITING:
        if job.next_attempt_at is None:
            return True
        return include_not_ready or _ensure_aware(job.next_attempt_at) <= now
    if job.status == SyncJobStatus.RUNNING:
        return job.lease_expires_at is not None and _ensure_aware(job.lease_expires_at) <= now
    return False
