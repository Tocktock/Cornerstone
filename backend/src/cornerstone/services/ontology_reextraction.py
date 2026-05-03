from __future__ import annotations

from typing import Any, cast

from cornerstone.observability import log_event
from cornerstone.schemas import (
    Artifact,
    ConceptCandidate,
    CreateOntologyExtractionRunRequest,
    CreateOntologyReExtractionRunRequest,
    DataSource,
    OntologyExtractionRun,
    OntologyReExtractionRun,
    OntologyReExtractionRunListResponse,
    OntologyReExtractionRunResponse,
    OntologyReExtractionRunStatus,
    OntologyReExtractionTrigger,
    RelationCandidate,
    RunOntologyReExtractionRunRequest,
    SyncSourceResponse,
    utc_now,
)
from cornerstone.services.ontology_extraction import OntologyExtractionService
from cornerstone.store import NotFoundError


class OntologyReExtractionError(RuntimeError):
    """Raised when an ontology re-extraction run cannot proceed safely."""


class OntologyReExtractionService:
    """Queue and execute safe ontology re-extraction from changed source evidence.

    v1.8.0 deliberately separates source sync from official graph mutation:
    connector/manual changes can queue and run extraction, but the extraction output
    remains ConceptCandidate/RelationCandidate data until v1.5 review promotion.
    """

    def __init__(self, store: Any) -> None:
        self.store = store

    def queue_from_sync_response(
        self,
        *,
        response: SyncSourceResponse,
        trigger: OntologyReExtractionTrigger,
        created_by: str,
        sync_job_id: str | None = None,
        focus_concept: str | None = None,
        reason: str | None = None,
    ) -> OntologyReExtractionRun | None:
        artifact_ids = list(dict.fromkeys(response.created_artifact_ids))
        if not artifact_ids:
            return None
        data_source = response.data_source
        artifacts = [self.store.get_artifact(artifact_id) for artifact_id in artifact_ids]
        evidence_ids = self._evidence_ids_for_artifacts(artifact_ids)
        run = OntologyReExtractionRun(
            datasource_id=data_source.id,
            provider=data_source.type,
            trigger=trigger,
            status=OntologyReExtractionRunStatus.QUEUED,
            created_by=created_by,
            sync_job_id=sync_job_id,
            focus_concept=focus_concept,
            reason=reason or _default_reason(trigger),
            source_external_ids=_source_external_ids(artifacts),
            artifact_ids=artifact_ids,
            changed_artifact_ids=list(dict.fromkeys(response.changed_artifact_ids)),
            evidence_fragment_ids=evidence_ids,
            official_graph_mutated=False,
        )
        saved = cast(OntologyReExtractionRun, self.store.add_ontology_reextraction_run(run))
        log_event(
            "ontology.reextraction_queued",
            runId=saved.id,
            sourceId=saved.datasource_id,
            syncJobId=saved.sync_job_id,
            artifactCount=len(saved.artifact_ids),
            evidenceFragmentCount=len(saved.evidence_fragment_ids),
            trigger=saved.trigger,
        )
        return saved

    def create_run(self, request: CreateOntologyReExtractionRunRequest) -> OntologyReExtractionRunResponse:
        artifact_ids = self._resolve_artifact_ids(request)
        if not artifact_ids:
            raise OntologyReExtractionError("Ontology re-extraction scope resolved to no Artifacts.")
        artifacts = [cast(Artifact, self.store.get_artifact(artifact_id)) for artifact_id in artifact_ids]
        datasource_ids = {artifact.datasource_id for artifact in artifacts}
        if len(datasource_ids) != 1:
            raise OntologyReExtractionError("Ontology re-extraction must be scoped to one DataSource.")
        datasource_id = next(iter(datasource_ids))
        data_source = cast(DataSource, self.store.get_data_source(datasource_id))
        evidence_ids = self._evidence_ids_for_artifacts(artifact_ids)
        run = OntologyReExtractionRun(
            datasource_id=data_source.id,
            provider=data_source.type,
            trigger=request.trigger,
            status=OntologyReExtractionRunStatus.QUEUED,
            created_by=request.created_by,
            sync_job_id=request.sync_job_id,
            focus_concept=request.focus_concept,
            reason=request.reason or _default_reason(request.trigger),
            source_external_ids=_source_external_ids(artifacts),
            artifact_ids=artifact_ids,
            changed_artifact_ids=artifact_ids,
            evidence_fragment_ids=evidence_ids,
            official_graph_mutated=False,
        )
        saved = cast(OntologyReExtractionRun, self.store.add_ontology_reextraction_run(run))
        if request.run_inline:
            return self.run(saved.id, RunOntologyReExtractionRunRequest(requested_by=request.created_by))
        return self.get_response(saved.id)

    def run(
        self,
        run_id: str,
        request: RunOntologyReExtractionRunRequest | None = None,
    ) -> OntologyReExtractionRunResponse:
        request = request or RunOntologyReExtractionRunRequest()
        run = cast(OntologyReExtractionRun, self.store.get_ontology_reextraction_run(run_id))
        if run.status not in {OntologyReExtractionRunStatus.QUEUED, OntologyReExtractionRunStatus.FAILED}:
            raise OntologyReExtractionError(
                f"Ontology re-extraction run cannot be executed while status is '{run.status}'."
            )
        evidence_ids = self._evidence_ids_for_artifacts(run.artifact_ids)
        started = run.model_copy(
            update={
                "status": OntologyReExtractionRunStatus.RUNNING,
                "started_at": utc_now(),
                "error": None,
                "evidence_fragment_ids": evidence_ids,
            },
            deep=True,
        )
        self.store.update_ontology_reextraction_run(started)
        if not evidence_ids:
            skipped = started.model_copy(
                update={
                    "status": OntologyReExtractionRunStatus.SKIPPED,
                    "completed_at": utc_now(),
                    "reason": started.reason or "No EvidenceFragments were available for re-extraction.",
                },
                deep=True,
            )
            saved = cast(OntologyReExtractionRun, self.store.update_ontology_reextraction_run(skipped))
            return self.get_response(saved.id)

        try:
            extraction_response = OntologyExtractionService(self.store).create_run(
                CreateOntologyExtractionRunRequest(
                    evidence_fragment_ids=evidence_ids,
                    artifact_ids=[],
                    focus_concept=run.focus_concept,
                    requested_by=request.requested_by,
                )
            )
        except Exception as exc:
            failed = started.model_copy(
                update={
                    "status": OntologyReExtractionRunStatus.FAILED,
                    "completed_at": utc_now(),
                    "error": str(exc),
                },
                deep=True,
            )
            self.store.update_ontology_reextraction_run(failed)
            raise

        completed = started.model_copy(
            update={
                "status": OntologyReExtractionRunStatus.COMPLETED,
                "completed_at": utc_now(),
                "extraction_run_ids": [extraction_response.run.id],
                "concept_candidate_count": extraction_response.run.concept_candidate_count,
                "relation_candidate_count": extraction_response.run.relation_candidate_count,
                "warning_count": extraction_response.run.warning_count,
                "official_graph_mutated": False,
            },
            deep=True,
        )
        saved = cast(OntologyReExtractionRun, self.store.update_ontology_reextraction_run(completed))
        log_event(
            "ontology.reextraction_completed",
            runId=saved.id,
            extractionRunIds=saved.extraction_run_ids,
            conceptCandidateCount=saved.concept_candidate_count,
            relationCandidateCount=saved.relation_candidate_count,
            officialGraphMutated=saved.official_graph_mutated,
        )
        return self.get_response(saved.id)

    def get_response(self, run_id: str) -> OntologyReExtractionRunResponse:
        run = cast(OntologyReExtractionRun, self.store.get_ontology_reextraction_run(run_id))
        extraction_runs: list[OntologyExtractionRun] = []
        concept_candidates: list[ConceptCandidate] = []
        relation_candidates: list[RelationCandidate] = []
        for extraction_run_id in run.extraction_run_ids:
            extraction_run = cast(OntologyExtractionRun, self.store.get_ontology_extraction_run(extraction_run_id))
            extraction_runs.append(extraction_run)
            concept_candidates.extend(self.store.list_concept_candidates(extraction_run_id=extraction_run_id))
            relation_candidates.extend(self.store.list_relation_candidates(extraction_run_id=extraction_run_id))
        return OntologyReExtractionRunResponse(
            run=run,
            extraction_runs=extraction_runs,
            concept_candidates=concept_candidates,
            relation_candidates=relation_candidates,
        )

    def list_runs(
        self,
        *,
        datasource_id: str | None = None,
        status: OntologyReExtractionRunStatus | None = None,
    ) -> OntologyReExtractionRunListResponse:
        return OntologyReExtractionRunListResponse(
            runs=self.store.list_ontology_reextraction_runs(datasource_id=datasource_id, status=status)
        )

    def _resolve_artifact_ids(self, request: CreateOntologyReExtractionRunRequest) -> list[str]:
        if request.artifact_ids:
            return _dedupe(request.artifact_ids)
        if request.sync_job_id is not None:
            job = self.store.get_sync_job(request.sync_job_id)
            events = self.store.list_sync_job_events(job.id)
            for event in events:
                if event.event_type == "ontology.reextraction_queued":
                    artifact_ids = event.metadata.get("artifactIds")
                    if isinstance(artifact_ids, list) and artifact_ids:
                        return _dedupe([str(item) for item in artifact_ids])
            artifacts = self.store.list_artifacts(datasource_id=job.datasource_id)
            if not artifacts:
                return []
            # Last resort for older jobs without v1.8 metadata: use Artifacts captured
            # at or after the job start. This is conservative and still candidate-only.
            if job.started_at is not None:
                artifacts = [artifact for artifact in artifacts if artifact.captured_at >= job.started_at]
            return _dedupe([artifact.id for artifact in artifacts])
        if request.datasource_id is not None:
            self.store.get_data_source(request.datasource_id)
            artifacts = self.store.list_artifacts(datasource_id=request.datasource_id)
            return _dedupe([artifact.id for artifact in artifacts])
        raise NotFoundError("No ontology re-extraction scope was provided.")

    def _evidence_ids_for_artifacts(self, artifact_ids: list[str]) -> list[str]:
        evidence_ids: list[str] = []
        for artifact_id in artifact_ids:
            for evidence in self.store.list_evidence_fragments(artifact_id=artifact_id):
                evidence_ids.append(evidence.id)
        return _dedupe(evidence_ids)


def _source_external_ids(artifacts: list[Artifact]) -> list[str]:
    return _dedupe([artifact.source_external_id for artifact in artifacts])


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _default_reason(trigger: OntologyReExtractionTrigger) -> str:
    if trigger in {OntologyReExtractionTrigger.CONNECTOR_SYNC, OntologyReExtractionTrigger.SCHEDULED_SYNC}:
        return "Source sync created changed or new Artifacts; ontology candidates should be refreshed."
    if trigger in {OntologyReExtractionTrigger.MANUAL_UPLOAD, OntologyReExtractionTrigger.MANUAL_SYNC}:
        return "Manual source ingestion created changed or new Artifacts; ontology candidates should be refreshed."
    if trigger == OntologyReExtractionTrigger.WEBHOOK:
        return "Provider webhook indicated source content changed; ontology candidates should be refreshed."
    return "Operator requested ontology re-extraction for selected source evidence."
