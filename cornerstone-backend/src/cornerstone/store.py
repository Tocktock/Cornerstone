from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from threading import RLock
from typing import TypedDict

from cornerstone.schemas import (
    Artifact,
    AuditEvent,
    Concept,
    ConceptRelation,
    ConnectionIntent,
    ConnectorCredential,
    CredentialStatus,
    DataSource,
    DecisionRecord,
    EvidenceFragment,
    GroundedContextEvalResult,
    GroundedContextEvalTask,
    ProviderObjectAccessState,
    ProviderObjectSnapshot,
    SourceSelection,
    SyncCursor,
    SyncJob,
    SyncJobEvent,
    SyncJobStatus,
    SyncSchedule,
    utc_now,
)


class _Snapshot(TypedDict):
    data_sources: dict[str, DataSource]
    artifacts: dict[str, Artifact]
    evidence_fragments: dict[str, EvidenceFragment]
    concepts: dict[str, Concept]
    concept_relations: dict[str, ConceptRelation]
    decision_records: dict[str, DecisionRecord]
    audit_events: list[AuditEvent]
    connection_intents: dict[str, ConnectionIntent]
    connector_credentials: dict[str, ConnectorCredential]
    source_selections: dict[str, SourceSelection]
    provider_object_snapshots: dict[str, ProviderObjectSnapshot]
    sync_jobs: dict[str, SyncJob]
    sync_job_events: dict[str, list[SyncJobEvent]]
    sync_cursors: dict[str, SyncCursor]
    sync_schedules: dict[str, SyncSchedule]
    eval_tasks: dict[str, GroundedContextEvalTask]
    eval_results: dict[str, GroundedContextEvalResult]


class StoreError(RuntimeError):
    """Base store error."""


class NotFoundError(StoreError):
    """Raised when an entity is not found."""


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=utc_now().tzinfo)
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


class InMemoryStore:
    """Thread-safe repository used for backend trust-gate validation.

    The store deliberately exposes repository-like methods so the route and service layers can be
    moved to SQLAlchemy/PostgreSQL without changing business rules. The transaction context is a
    small in-memory rollback mechanism used to test sync atomicity before adding a database.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._data_sources: dict[str, DataSource] = {}
        self._artifacts: dict[str, Artifact] = {}
        self._evidence_fragments: dict[str, EvidenceFragment] = {}
        self._concepts: dict[str, Concept] = {}
        self._concept_relations: dict[str, ConceptRelation] = {}
        self._decision_records: dict[str, DecisionRecord] = {}
        self._audit_events: list[AuditEvent] = []
        self._connection_intents: dict[str, ConnectionIntent] = {}
        self._connector_credentials: dict[str, ConnectorCredential] = {}
        self._source_selections: dict[str, SourceSelection] = {}
        self._provider_object_snapshots: dict[str, ProviderObjectSnapshot] = {}
        self._sync_jobs: dict[str, SyncJob] = {}
        self._sync_job_events: dict[str, list[SyncJobEvent]] = {}
        self._sync_cursors: dict[str, SyncCursor] = {}
        self._sync_schedules: dict[str, SyncSchedule] = {}
        self._eval_tasks: dict[str, GroundedContextEvalTask] = {}
        self._eval_results: dict[str, GroundedContextEvalResult] = {}

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            snapshot = self._snapshot_unlocked()
            try:
                yield
            except Exception:
                self._restore_unlocked(snapshot)
                raise

    def reset(self) -> None:
        with self._lock:
            self._data_sources.clear()
            self._artifacts.clear()
            self._evidence_fragments.clear()
            self._concepts.clear()
            self._concept_relations.clear()
            self._decision_records.clear()
            self._audit_events.clear()
            self._connection_intents.clear()
            self._connector_credentials.clear()
            self._source_selections.clear()
            self._provider_object_snapshots.clear()
            self._sync_jobs.clear()
            self._sync_job_events.clear()
            self._sync_cursors.clear()
            self._sync_schedules.clear()
            self._eval_tasks.clear()
            self._eval_results.clear()

    # Data sources
    def add_data_source(self, data_source: DataSource) -> DataSource:
        with self._lock:
            self._data_sources[data_source.id] = data_source.model_copy(deep=True)
            return data_source.model_copy(deep=True)

    def get_data_source(self, data_source_id: str) -> DataSource:
        with self._lock:
            data_source = self._data_sources.get(data_source_id)
            if data_source is None:
                raise NotFoundError(f"DataSource not found: {data_source_id}")
            return data_source.model_copy(deep=True)

    def update_data_source(self, data_source: DataSource) -> DataSource:
        with self._lock:
            if data_source.id not in self._data_sources:
                raise NotFoundError(f"DataSource not found: {data_source.id}")
            self._data_sources[data_source.id] = data_source.model_copy(deep=True)
            return data_source.model_copy(deep=True)

    def list_data_sources(self) -> list[DataSource]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._data_sources.values()]


    # Connector connection intents
    def add_connection_intent(self, intent: ConnectionIntent) -> ConnectionIntent:
        with self._lock:
            self._connection_intents[intent.id] = intent.model_copy(deep=True)
            return intent.model_copy(deep=True)

    def update_connection_intent(self, intent: ConnectionIntent) -> ConnectionIntent:
        with self._lock:
            if intent.id not in self._connection_intents:
                raise NotFoundError(f"ConnectionIntent not found: {intent.id}")
            self._connection_intents[intent.id] = intent.model_copy(deep=True)
            return intent.model_copy(deep=True)

    def get_connection_intent(self, intent_id: str) -> ConnectionIntent:
        with self._lock:
            intent = self._connection_intents.get(intent_id)
            if intent is None:
                raise NotFoundError(f"ConnectionIntent not found: {intent_id}")
            return intent.model_copy(deep=True)

    def get_connection_intent_by_state_nonce(self, state_nonce: str) -> ConnectionIntent:
        with self._lock:
            for intent in self._connection_intents.values():
                if intent.state_nonce == state_nonce:
                    return intent.model_copy(deep=True)
            raise NotFoundError(f"ConnectionIntent not found for state nonce: {state_nonce}")

    def list_connection_intents(self) -> list[ConnectionIntent]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._connection_intents.values()]

    # Connector credentials
    def add_connector_credential(self, credential: ConnectorCredential) -> ConnectorCredential:
        with self._lock:
            self._connector_credentials[credential.id] = credential.model_copy(deep=True)
            return credential.model_copy(deep=True)

    def update_connector_credential(self, credential: ConnectorCredential) -> ConnectorCredential:
        with self._lock:
            if credential.id not in self._connector_credentials:
                raise NotFoundError(f"ConnectorCredential not found: {credential.id}")
            self._connector_credentials[credential.id] = credential.model_copy(deep=True)
            return credential.model_copy(deep=True)

    def get_connector_credential(self, credential_id: str) -> ConnectorCredential:
        with self._lock:
            credential = self._connector_credentials.get(credential_id)
            if credential is None:
                raise NotFoundError(f"ConnectorCredential not found: {credential_id}")
            return credential.model_copy(deep=True)

    def get_active_credential_for_source(self, datasource_id: str) -> ConnectorCredential:
        with self._lock:
            for credential in self._connector_credentials.values():
                if credential.datasource_id == datasource_id and credential.status == CredentialStatus.ACTIVE:
                    return credential.model_copy(deep=True)
            raise NotFoundError(f"Active ConnectorCredential not found for DataSource: {datasource_id}")

    def list_connector_credentials(self, datasource_id: str | None = None) -> list[ConnectorCredential]:
        with self._lock:
            credentials = list(self._connector_credentials.values())
            if datasource_id is not None:
                credentials = [item for item in credentials if item.datasource_id == datasource_id]
            return [item.model_copy(deep=True) for item in credentials]

    # Source selections
    def upsert_source_selection(self, selection: SourceSelection) -> SourceSelection:
        with self._lock:
            existing = next((item for item in self._source_selections.values() if item.datasource_id == selection.datasource_id), None)
            if existing is not None:
                selection = selection.model_copy(update={"id": existing.id, "created_at": existing.created_at}, deep=True)
            self._source_selections[selection.id] = selection.model_copy(deep=True)
            return selection.model_copy(deep=True)

    def get_source_selection(self, datasource_id: str) -> SourceSelection:
        with self._lock:
            for selection in self._source_selections.values():
                if selection.datasource_id == datasource_id:
                    return selection.model_copy(deep=True)
            raise NotFoundError(f"SourceSelection not found for DataSource: {datasource_id}")

    def list_source_selections(self, datasource_id: str | None = None) -> list[SourceSelection]:
        with self._lock:
            selections = list(self._source_selections.values())
            if datasource_id is not None:
                selections = [item for item in selections if item.datasource_id == datasource_id]
            return [item.model_copy(deep=True) for item in selections]

    # Provider object discovery snapshots
    def upsert_provider_object_snapshot(self, snapshot: ProviderObjectSnapshot) -> ProviderObjectSnapshot:
        with self._lock:
            existing_key = next(
                (
                    key
                    for key, item in self._provider_object_snapshots.items()
                    if item.datasource_id == snapshot.datasource_id and item.external_id == snapshot.external_id
                ),
                None,
            )
            if existing_key is not None:
                existing = self._provider_object_snapshots[existing_key]
                snapshot = snapshot.model_copy(
                    update={
                        "id": existing.id,
                        "selected_for_sync": existing.selected_for_sync,
                    },
                    deep=True,
                )
                self._provider_object_snapshots[existing_key] = snapshot.model_copy(deep=True)
            else:
                self._provider_object_snapshots[snapshot.id] = snapshot.model_copy(deep=True)
            return snapshot.model_copy(deep=True)

    def upsert_provider_object_snapshots(
        self,
        datasource_id: str,
        snapshots: list[ProviderObjectSnapshot],
    ) -> list[ProviderObjectSnapshot]:
        with self._lock:
            return [self.upsert_provider_object_snapshot(snapshot) for snapshot in snapshots if snapshot.datasource_id == datasource_id]

    def get_provider_object_snapshot(self, datasource_id: str, external_id: str) -> ProviderObjectSnapshot:
        with self._lock:
            for snapshot in self._provider_object_snapshots.values():
                if snapshot.datasource_id == datasource_id and snapshot.external_id == external_id:
                    return snapshot.model_copy(deep=True)
            raise NotFoundError(f"ProviderObjectSnapshot not found for DataSource {datasource_id}: {external_id}")

    def list_provider_object_snapshots(
        self,
        datasource_id: str | None = None,
        access_state: ProviderObjectAccessState | None = None,
    ) -> list[ProviderObjectSnapshot]:
        with self._lock:
            snapshots = list(self._provider_object_snapshots.values())
            if datasource_id is not None:
                snapshots = [item for item in snapshots if item.datasource_id == datasource_id]
            if access_state is not None:
                snapshots = [item for item in snapshots if item.access_state == access_state]
            snapshots.sort(key=lambda item: (item.title or "", item.external_id))
            return [item.model_copy(deep=True) for item in snapshots]

    def mark_provider_object_selection(
        self,
        datasource_id: str,
        selected_external_object_ids: list[str],
    ) -> list[ProviderObjectSnapshot]:
        with self._lock:
            selected = set(selected_external_object_ids)
            updated: list[ProviderObjectSnapshot] = []
            for key, snapshot in list(self._provider_object_snapshots.items()):
                if snapshot.datasource_id != datasource_id:
                    continue
                next_snapshot = snapshot.model_copy(
                    update={"selected_for_sync": snapshot.external_id in selected},
                    deep=True,
                )
                self._provider_object_snapshots[key] = next_snapshot
                updated.append(next_snapshot.model_copy(deep=True))
            return updated

    # Sync jobs
    def add_sync_job(self, sync_job: SyncJob) -> SyncJob:
        with self._lock:
            if sync_job.enqueue_key is not None:
                for existing in self._sync_jobs.values():
                    if existing.enqueue_key == sync_job.enqueue_key:
                        raise StoreError(f"SyncJob enqueue key already exists: {sync_job.enqueue_key}")
            self._sync_jobs[sync_job.id] = sync_job.model_copy(deep=True)
            self._sync_job_events.setdefault(sync_job.id, [])
            return sync_job.model_copy(deep=True)

    def update_sync_job(self, sync_job: SyncJob) -> SyncJob:
        with self._lock:
            if sync_job.id not in self._sync_jobs:
                raise NotFoundError(f"SyncJob not found: {sync_job.id}")
            self._sync_jobs[sync_job.id] = sync_job.model_copy(deep=True)
            self._sync_job_events.setdefault(sync_job.id, [])
            return sync_job.model_copy(deep=True)

    def get_sync_job(self, sync_job_id: str) -> SyncJob:
        with self._lock:
            sync_job = self._sync_jobs.get(sync_job_id)
            if sync_job is None:
                raise NotFoundError(f"SyncJob not found: {sync_job_id}")
            return sync_job.model_copy(deep=True)

    def get_sync_job_by_enqueue_key(self, enqueue_key: str) -> SyncJob:
        with self._lock:
            for sync_job in self._sync_jobs.values():
                if sync_job.enqueue_key == enqueue_key:
                    return sync_job.model_copy(deep=True)
            raise NotFoundError(f"SyncJob not found for enqueue key: {enqueue_key}")

    def claim_sync_job(
        self,
        sync_job_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
        include_not_ready: bool = False,
    ) -> SyncJob | None:
        with self._lock:
            job = self._sync_jobs.get(sync_job_id)
            if job is None:
                raise NotFoundError(f"SyncJob not found: {sync_job_id}")
            return self._claim_sync_job_unlocked(
                job,
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
        with self._lock:
            job = self._sync_jobs.get(sync_job_id)
            if job is None:
                raise NotFoundError(f"SyncJob not found: {sync_job_id}")
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
            self._sync_jobs[sync_job_id] = refreshed.model_copy(deep=True)
            return refreshed.model_copy(deep=True)

    def _claim_sync_job_unlocked(
        self,
        job: SyncJob,
        *,
        worker_id: str,
        lease_seconds: int,
        include_not_ready: bool,
    ) -> SyncJob | None:
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
        self._sync_jobs[job.id] = claimed.model_copy(deep=True)
        self._sync_job_events.setdefault(job.id, [])
        return claimed.model_copy(deep=True)

    def list_sync_jobs(self, datasource_id: str | None = None) -> list[SyncJob]:
        with self._lock:
            jobs = list(self._sync_jobs.values())
            if datasource_id is not None:
                jobs = [item for item in jobs if item.datasource_id == datasource_id]
            return [item.model_copy(deep=True) for item in jobs]

    def add_sync_job_event(self, event: SyncJobEvent) -> SyncJobEvent:
        with self._lock:
            if event.sync_job_id not in self._sync_jobs:
                raise NotFoundError(f"SyncJob not found: {event.sync_job_id}")
            self._sync_job_events.setdefault(event.sync_job_id, []).append(event.model_copy(deep=True))
            return event.model_copy(deep=True)

    def list_sync_job_events(self, sync_job_id: str) -> list[SyncJobEvent]:
        with self._lock:
            if sync_job_id not in self._sync_jobs:
                raise NotFoundError(f"SyncJob not found: {sync_job_id}")
            return [item.model_copy(deep=True) for item in self._sync_job_events.get(sync_job_id, [])]

    # Sync cursors
    def upsert_sync_cursor(self, cursor: SyncCursor) -> SyncCursor:
        with self._lock:
            existing_key = next(
                (
                    key
                    for key, item in self._sync_cursors.items()
                    if item.datasource_id == cursor.datasource_id and item.cursor_key == cursor.cursor_key
                ),
                None,
            )
            if existing_key is not None:
                existing = self._sync_cursors[existing_key]
                cursor = cursor.model_copy(
                    update={"id": existing.id, "created_at": existing.created_at},
                    deep=True,
                )
                self._sync_cursors[existing_key] = cursor.model_copy(deep=True)
            else:
                self._sync_cursors[cursor.id] = cursor.model_copy(deep=True)
            return cursor.model_copy(deep=True)

    def get_sync_cursor(self, datasource_id: str, cursor_key: str = "default") -> SyncCursor:
        with self._lock:
            for cursor in self._sync_cursors.values():
                if cursor.datasource_id == datasource_id and cursor.cursor_key == cursor_key:
                    return cursor.model_copy(deep=True)
            raise NotFoundError(f"SyncCursor not found for DataSource: {datasource_id}, key: {cursor_key}")

    def list_sync_cursors(self, datasource_id: str | None = None) -> list[SyncCursor]:
        with self._lock:
            cursors = list(self._sync_cursors.values())
            if datasource_id is not None:
                cursors = [item for item in cursors if item.datasource_id == datasource_id]
            return [item.model_copy(deep=True) for item in cursors]

    # Sync schedules
    def upsert_sync_schedule(self, schedule: SyncSchedule) -> SyncSchedule:
        with self._lock:
            existing_key = next(
                (
                    key
                    for key, item in self._sync_schedules.items()
                    if item.datasource_id == schedule.datasource_id
                ),
                None,
            )
            if existing_key is not None:
                existing = self._sync_schedules[existing_key]
                schedule = schedule.model_copy(
                    update={"id": existing.id, "created_at": existing.created_at},
                    deep=True,
                )
                self._sync_schedules[existing_key] = schedule.model_copy(deep=True)
            else:
                self._sync_schedules[schedule.id] = schedule.model_copy(deep=True)
            return schedule.model_copy(deep=True)

    def get_sync_schedule(self, datasource_id: str) -> SyncSchedule:
        with self._lock:
            for schedule in self._sync_schedules.values():
                if schedule.datasource_id == datasource_id:
                    return schedule.model_copy(deep=True)
            raise NotFoundError(f"SyncSchedule not found for DataSource: {datasource_id}")

    def list_sync_schedules(self, datasource_id: str | None = None) -> list[SyncSchedule]:
        with self._lock:
            schedules = list(self._sync_schedules.values())
            if datasource_id is not None:
                schedules = [item for item in schedules if item.datasource_id == datasource_id]
            schedules = sorted(schedules, key=lambda item: (item.next_run_at, item.created_at, item.id))
            return [item.model_copy(deep=True) for item in schedules]

    # Artifacts
    def add_artifact(self, artifact: Artifact) -> Artifact:
        with self._lock:
            self._artifacts[artifact.id] = artifact.model_copy(deep=True)
            return artifact.model_copy(deep=True)

    def list_artifacts(self, datasource_id: str | None = None) -> list[Artifact]:
        with self._lock:
            artifacts = list(self._artifacts.values())
            if datasource_id is not None:
                artifacts = [item for item in artifacts if item.datasource_id == datasource_id]
            return [item.model_copy(deep=True) for item in artifacts]

    def get_artifact(self, artifact_id: str) -> Artifact:
        with self._lock:
            artifact = self._artifacts.get(artifact_id)
            if artifact is None:
                raise NotFoundError(f"Artifact not found: {artifact_id}")
            return artifact.model_copy(deep=True)

    def find_artifact_by_source_identity(
        self,
        *,
        datasource_id: str,
        source_external_id: str,
        raw_content_hash: str,
    ) -> Artifact | None:
        with self._lock:
            for artifact in self._artifacts.values():
                if (
                    artifact.datasource_id == datasource_id
                    and artifact.source_external_id == source_external_id
                    and artifact.raw_content_hash == raw_content_hash
                ):
                    return artifact.model_copy(deep=True)
            return None

    # Evidence
    def add_evidence_fragment(self, evidence_fragment: EvidenceFragment) -> EvidenceFragment:
        with self._lock:
            self._evidence_fragments[evidence_fragment.id] = evidence_fragment.model_copy(deep=True)
            return evidence_fragment.model_copy(deep=True)

    def update_evidence_fragment(self, evidence_fragment: EvidenceFragment) -> EvidenceFragment:
        with self._lock:
            if evidence_fragment.id not in self._evidence_fragments:
                raise NotFoundError(f"EvidenceFragment not found: {evidence_fragment.id}")
            self._evidence_fragments[evidence_fragment.id] = evidence_fragment.model_copy(deep=True)
            return evidence_fragment.model_copy(deep=True)

    def list_evidence_fragments(self, artifact_id: str | None = None) -> list[EvidenceFragment]:
        with self._lock:
            fragments = list(self._evidence_fragments.values())
            if artifact_id is not None:
                fragments = [item for item in fragments if item.artifact_id == artifact_id]
            return [item.model_copy(deep=True) for item in fragments]

    def get_evidence_fragment(self, evidence_fragment_id: str) -> EvidenceFragment:
        with self._lock:
            fragment = self._evidence_fragments.get(evidence_fragment_id)
            if fragment is None:
                raise NotFoundError(f"EvidenceFragment not found: {evidence_fragment_id}")
            return fragment.model_copy(deep=True)

    # Concept relations
    def add_concept_relation(self, relation: ConceptRelation) -> ConceptRelation:
        with self._lock:
            self._concept_relations[relation.id] = relation.model_copy(deep=True)
            return relation.model_copy(deep=True)

    def update_concept_relation(self, relation: ConceptRelation) -> ConceptRelation:
        with self._lock:
            if relation.id not in self._concept_relations:
                raise NotFoundError(f"ConceptRelation not found: {relation.id}")
            self._concept_relations[relation.id] = relation.model_copy(deep=True)
            return relation.model_copy(deep=True)

    def get_concept_relation(self, relation_id: str) -> ConceptRelation:
        with self._lock:
            relation = self._concept_relations.get(relation_id)
            if relation is None:
                raise NotFoundError(f"ConceptRelation not found: {relation_id}")
            return relation.model_copy(deep=True)

    def list_concept_relations(self, concept_id: str | None = None) -> list[ConceptRelation]:
        with self._lock:
            relations = list(self._concept_relations.values())
            if concept_id is not None:
                relations = [
                    item
                    for item in relations
                    if item.source_concept_id == concept_id or item.target_concept_id == concept_id
                ]
            return [item.model_copy(deep=True) for item in relations]

    # Decision records
    def add_decision_record(self, decision_record: DecisionRecord) -> DecisionRecord:
        with self._lock:
            self._decision_records[decision_record.id] = decision_record.model_copy(deep=True)
            return decision_record.model_copy(deep=True)

    def get_decision_record(self, decision_record_id: str) -> DecisionRecord:
        with self._lock:
            decision_record = self._decision_records.get(decision_record_id)
            if decision_record is None:
                raise NotFoundError(f"DecisionRecord not found: {decision_record_id}")
            return decision_record.model_copy(deep=True)

    def list_decision_records(self) -> list[DecisionRecord]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._decision_records.values()]

    # Concepts
    def add_concept(self, concept: Concept) -> Concept:
        with self._lock:
            self._concepts[concept.id] = concept.model_copy(deep=True)
            return concept.model_copy(deep=True)

    def update_concept(self, concept: Concept) -> Concept:
        with self._lock:
            if concept.id not in self._concepts:
                raise NotFoundError(f"Concept not found: {concept.id}")
            self._concepts[concept.id] = concept.model_copy(deep=True)
            return concept.model_copy(deep=True)

    def get_concept(self, concept_id: str) -> Concept:
        with self._lock:
            concept = self._concepts.get(concept_id)
            if concept is None:
                raise NotFoundError(f"Concept not found: {concept_id}")
            return concept.model_copy(deep=True)

    def list_concepts(self) -> list[Concept]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._concepts.values()]


    # Evaluation framework
    def add_grounded_context_eval_task(self, task: GroundedContextEvalTask) -> GroundedContextEvalTask:
        with self._lock:
            self._eval_tasks[task.id] = task.model_copy(deep=True)
            return task.model_copy(deep=True)

    def get_grounded_context_eval_task(self, task_id: str) -> GroundedContextEvalTask:
        with self._lock:
            task = self._eval_tasks.get(task_id)
            if task is None:
                raise NotFoundError(f"GroundedContextEvalTask not found: {task_id}")
            return task.model_copy(deep=True)

    def list_grounded_context_eval_tasks(self) -> list[GroundedContextEvalTask]:
        with self._lock:
            tasks = sorted(self._eval_tasks.values(), key=lambda item: (item.created_at, item.id))
            return [item.model_copy(deep=True) for item in tasks]

    def add_grounded_context_eval_result(self, result: GroundedContextEvalResult) -> GroundedContextEvalResult:
        with self._lock:
            if result.task_id not in self._eval_tasks:
                raise NotFoundError(f"GroundedContextEvalTask not found: {result.task_id}")
            self._eval_results[result.id] = result.model_copy(deep=True)
            return result.model_copy(deep=True)

    def get_grounded_context_eval_result(self, result_id: str) -> GroundedContextEvalResult:
        with self._lock:
            result = self._eval_results.get(result_id)
            if result is None:
                raise NotFoundError(f"GroundedContextEvalResult not found: {result_id}")
            return result.model_copy(deep=True)

    def list_grounded_context_eval_results(self, task_id: str | None = None) -> list[GroundedContextEvalResult]:
        with self._lock:
            results = list(self._eval_results.values())
            if task_id is not None:
                results = [item for item in results if item.task_id == task_id]
            results.sort(key=lambda item: (item.evaluated_at, item.id))
            return [item.model_copy(deep=True) for item in results]

    # Evaluation framework
    def add_eval_task(self, task: GroundedContextEvalTask) -> GroundedContextEvalTask:
        with self._lock:
            self._eval_tasks[task.id] = task.model_copy(deep=True)
            return task.model_copy(deep=True)

    def get_eval_task(self, task_id: str) -> GroundedContextEvalTask:
        with self._lock:
            task = self._eval_tasks.get(task_id)
            if task is None:
                raise NotFoundError(f"GroundedContextEvalTask not found: {task_id}")
            return task.model_copy(deep=True)

    def list_eval_tasks(self) -> list[GroundedContextEvalTask]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._eval_tasks.values()]

    def add_eval_result(self, result: GroundedContextEvalResult) -> GroundedContextEvalResult:
        with self._lock:
            if result.task_id not in self._eval_tasks:
                raise NotFoundError(f"GroundedContextEvalTask not found: {result.task_id}")
            self._eval_results[result.id] = result.model_copy(deep=True)
            return result.model_copy(deep=True)

    def get_eval_result(self, result_id: str) -> GroundedContextEvalResult:
        with self._lock:
            result = self._eval_results.get(result_id)
            if result is None:
                raise NotFoundError(f"GroundedContextEvalResult not found: {result_id}")
            return result.model_copy(deep=True)

    def list_eval_results(self, task_id: str | None = None) -> list[GroundedContextEvalResult]:
        with self._lock:
            results = list(self._eval_results.values())
            if task_id is not None:
                results = [item for item in results if item.task_id == task_id]
            results.sort(key=lambda item: (item.evaluated_at, item.id))
            return [item.model_copy(deep=True) for item in results]

    # Audit events
    def add_audit_event(self, event: AuditEvent) -> AuditEvent:
        with self._lock:
            self._audit_events.append(event.model_copy(deep=True))
            return event.model_copy(deep=True)

    def list_audit_events(self) -> list[AuditEvent]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._audit_events]

    def _snapshot_unlocked(self) -> _Snapshot:
        return {
            "data_sources": copy.deepcopy(self._data_sources),
            "artifacts": copy.deepcopy(self._artifacts),
            "evidence_fragments": copy.deepcopy(self._evidence_fragments),
            "concepts": copy.deepcopy(self._concepts),
            "concept_relations": copy.deepcopy(self._concept_relations),
            "decision_records": copy.deepcopy(self._decision_records),
            "audit_events": copy.deepcopy(self._audit_events),
            "connection_intents": copy.deepcopy(self._connection_intents),
            "connector_credentials": copy.deepcopy(self._connector_credentials),
            "source_selections": copy.deepcopy(self._source_selections),
            "provider_object_snapshots": copy.deepcopy(self._provider_object_snapshots),
            "sync_jobs": copy.deepcopy(self._sync_jobs),
            "sync_job_events": copy.deepcopy(self._sync_job_events),
            "sync_cursors": copy.deepcopy(self._sync_cursors),
            "sync_schedules": copy.deepcopy(self._sync_schedules),
            "eval_tasks": copy.deepcopy(self._eval_tasks),
            "eval_results": copy.deepcopy(self._eval_results),
        }

    def _restore_unlocked(self, snapshot: _Snapshot) -> None:
        self._data_sources = copy.deepcopy(snapshot["data_sources"])
        self._artifacts = copy.deepcopy(snapshot["artifacts"])
        self._evidence_fragments = copy.deepcopy(snapshot["evidence_fragments"])
        self._concepts = copy.deepcopy(snapshot["concepts"])
        self._concept_relations = copy.deepcopy(snapshot["concept_relations"])
        self._decision_records = copy.deepcopy(snapshot["decision_records"])
        self._audit_events = copy.deepcopy(snapshot["audit_events"])
        self._connection_intents = copy.deepcopy(snapshot["connection_intents"])
        self._connector_credentials = copy.deepcopy(snapshot["connector_credentials"])
        self._source_selections = copy.deepcopy(snapshot["source_selections"])
        self._provider_object_snapshots = copy.deepcopy(snapshot["provider_object_snapshots"])
        self._sync_jobs = copy.deepcopy(snapshot["sync_jobs"])
        self._sync_job_events = copy.deepcopy(snapshot["sync_job_events"])
        self._sync_cursors = copy.deepcopy(snapshot["sync_cursors"])
        self._sync_schedules = copy.deepcopy(snapshot["sync_schedules"])
        self._eval_tasks = copy.deepcopy(snapshot["eval_tasks"])
        self._eval_results = copy.deepcopy(snapshot["eval_results"])
