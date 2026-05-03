from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from cornerstone.config import Settings
from cornerstone.schemas import (
    ApproveConceptCandidateRequest,
    ApproveRelationCandidateRequest,
    AuditEvent,
    Concept,
    CreateOntologyGraphEvalTaskRequest,
    CreateOntologyProofRunRequest,
    DataSource,
    DataSourceAuthStatus,
    DataSourceConnectionStatus,
    DataSourceNextAction,
    DataSourceStatus,
    DataSourceSyncStatus,
    DataSourceType,
    FreshnessState,
    OntologyCandidateStatus,
    OntologyGraphMode,
    OntologyProofChecklistStep,
    OntologyProofRunResponse,
    OntologyProofRunStatus,
    OntologyProofStepStatus,
    OntologyProofSummary,
    OntologyReExtractionTrigger,
    RelationCandidate,
    RelationStatus,
    ReviewEvidenceRequest,
    RunOntologyReExtractionRunRequest,
    SourceObject,
    TrustLabel,
    TrustState,
    normalize_concept_term,
    new_id,
    utc_now,
)
from cornerstone.services.evaluation import OntologyGraphEvaluationService
from cornerstone.services.officialization import ReviewerAuthorizationError, ensure_reviewer_authorized
from cornerstone.services.ontology_graph import OntologyGraphService
from cornerstone.services.ontology_reextraction import OntologyReExtractionService
from cornerstone.services.ontology_review import OntologyCandidateReviewService, OntologyReviewError
from cornerstone.services.source_sync import sync_source_objects


class OntologyProofError(RuntimeError):
    """Raised when the operator proof checklist cannot be executed safely."""


class OntologyProofService:
    """Run the v1.9.0 operator checklist for the ontology loop.

    The proof intentionally mutates only explicit proof data and still uses normal
    review/officialization services. It does not bypass evidence review, candidate
    review, officialization, graph serving, or evaluation gates.
    """

    def __init__(self, store: Any, *, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    def run(self, request: CreateOntologyProofRunRequest) -> OntologyProofRunResponse:
        run_id = new_id()
        focus = _display_focus(request.focus_concept)
        checklist = _planned_checklist()
        if request.dry_run:
            return _response(
                run_id=run_id,
                status=OntologyProofRunStatus.PLANNED,
                request=request,
                checklist=checklist,
                limitations=[
                    "Dry run only. No DataSource, Artifact, EvidenceFragment, candidate, Concept, Relation, graph, or evaluation object was created.",
                ],
            )
        if not request.confirm_mutation:
            raise OntologyProofError(
                "Ontology proof runs create explicit proof source data and official proof Concepts/Relations. "
                "Set confirmMutation=true to execute, or dryRun=true to view the checklist."
            )
        try:
            ensure_reviewer_authorized(request.reviewer, self.settings.authorized_reviewers)
        except ReviewerAuthorizationError as exc:
            raise OntologyProofError(str(exc)) from exc

        ctx: dict[str, Any] = {
            "source_id": None,
            "artifact_ids": [],
            "evidence_fragment_ids": [],
            "reextraction_run_id": None,
            "extraction_run_ids": [],
            "concept_candidate_ids": [],
            "relation_candidate_ids": [],
            "approved_concept_ids": [],
            "approved_relation_ids": [],
            "graph_response_id": None,
            "evaluation_task_id": None,
            "evaluation_result_id": None,
            "official_graph_available": False,
            "official_graph_mutated": False,
            "evaluation_success": None,
        }
        step_index = {step.key: step for step in checklist}

        def apply(key: str, fn: Callable[[], tuple[str, dict[str, Any] | None]]) -> None:
            step = step_index[key]
            try:
                detail, object_ids = fn()
            except Exception as exc:  # keep partial checklist useful for operators
                step.status = OntologyProofStepStatus.FAILED
                step.detail = str(exc)
                step.next_actions = [_next_action_for_failure(key)]
                raise
            step.status = OntologyProofStepStatus.PASSED
            step.detail = detail
            if object_ids:
                step.object_ids.update(object_ids)

        try:
            apply("create_manual_source", lambda: self._create_manual_source(request, ctx, focus))
            apply("sync_manual_seed", lambda: self._sync_manual_seed(request, ctx, focus))
            apply("run_reextraction", lambda: self._run_reextraction(request, ctx))
            apply("review_evidence", lambda: self._review_evidence(request, ctx))
            apply("approve_concepts", lambda: self._approve_concepts(request, ctx))
            apply("approve_relations", lambda: self._approve_relations(request, ctx))
            apply("serve_explainable_graph", lambda: self._serve_graph(request, ctx, focus))
            if request.run_evaluation:
                apply("run_ontology_evaluation", lambda: self._run_evaluation(request, ctx, focus))
            else:
                step = step_index["run_ontology_evaluation"]
                step.status = OntologyProofStepStatus.SKIPPED
                step.detail = "Skipped because runEvaluation=false."
        except Exception:
            for step in checklist:
                if step.status == OntologyProofStepStatus.PLANNED:
                    step.status = OntologyProofStepStatus.SKIPPED
                    step.detail = "Skipped because an earlier required proof step failed."
            return _response(run_id=run_id, status=OntologyProofRunStatus.FAILED, request=request, checklist=checklist, ctx=ctx)

        status = OntologyProofRunStatus.PASSED if _all_required_passed(checklist) else OntologyProofRunStatus.FAILED
        return _response(run_id=run_id, status=status, request=request, checklist=checklist, ctx=ctx)

    def _create_manual_source(
        self,
        request: CreateOntologyProofRunRequest,
        ctx: dict[str, Any],
        focus: str,
    ) -> tuple[str, dict[str, Any]]:
        source = DataSource(
            type=DataSourceType.MANUAL,
            name=request.source_name or f"v1.9.0 ontology proof - {focus}",
            status=DataSourceStatus.CONNECTED,
            production_enabled=True,
            auth_status=DataSourceAuthStatus.AUTHORIZED,
            connection_status=DataSourceConnectionStatus.TEST_PASSED,
            sync_status=DataSourceSyncStatus.NEVER_SYNCED,
            next_action=DataSourceNextAction.NONE,
            freshness_state=FreshnessState.UNKNOWN,
            sync_freshness_state=FreshnessState.UNKNOWN,
            content_freshness_state=FreshnessState.UNKNOWN,
        )
        saved = cast(DataSource, self.store.add_data_source(source))
        ctx["source_id"] = saved.id
        return "Manual proof DataSource created as production-enabled source material.", {"sourceId": saved.id}

    def _sync_manual_seed(
        self,
        request: CreateOntologyProofRunRequest,
        ctx: dict[str, Any],
        focus: str,
    ) -> tuple[str, dict[str, Any]]:
        source = cast(DataSource, self.store.get_data_source(ctx["source_id"]))
        source_object = SourceObject(
            source_external_id=f"v1.9.0-proof-{normalize_concept_term(focus).replace(' ', '-')}",
            title=f"{focus} proof notes",
            content=request.seed_content or _default_seed_content(focus),
            source_url="https://example.internal/cornerstone/v1.9.0/ontology-proof",
            source_object_type="manual_text_upload",
            provider_metadata={"proofVersion": "v1.9.0", "proofFocusConcept": focus},
        )
        sync_response = sync_source_objects(
            data_source=source,
            objects=[source_object],
            store=self.store,
            settings=self.settings,
        )
        reextraction_run = OntologyReExtractionService(self.store).queue_from_sync_response(
            response=sync_response,
            trigger=OntologyReExtractionTrigger.MANUAL_UPLOAD,
            created_by=request.created_by,
            focus_concept=focus,
            reason="v1.9.0 operator proof checklist",
        )
        if reextraction_run is None:
            raise OntologyProofError("Proof seed sync did not create a new Artifact, so no re-extraction run was queued.")
        ctx["artifact_ids"] = list(sync_response.created_artifact_ids)
        ctx["evidence_fragment_ids"] = [fragment.id for fragment in sync_response.evidence_fragments]
        ctx["reextraction_run_id"] = reextraction_run.id
        return (
            "Manual proof text synced into Artifacts/EvidenceFragments and queued for ontology re-extraction.",
            {
                "artifactIds": ctx["artifact_ids"],
                "evidenceFragmentIds": ctx["evidence_fragment_ids"],
                "reextractionRunId": reextraction_run.id,
            },
        )

    def _run_reextraction(
        self,
        request: CreateOntologyProofRunRequest,
        ctx: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        reextraction_response = OntologyReExtractionService(self.store).run(
            ctx["reextraction_run_id"],
            RunOntologyReExtractionRunRequest(requested_by=request.created_by),
        )
        if not reextraction_response.concept_candidates:
            raise OntologyProofError("Re-extraction produced no ConceptCandidates.")
        if not reextraction_response.relation_candidates:
            raise OntologyProofError("Re-extraction produced no RelationCandidates.")
        ctx["extraction_run_ids"] = list(reextraction_response.run.extraction_run_ids)
        ctx["concept_candidate_ids"] = [candidate.id for candidate in reextraction_response.concept_candidates]
        ctx["relation_candidate_ids"] = [candidate.id for candidate in reextraction_response.relation_candidates]
        ctx["official_graph_mutated"] = bool(reextraction_response.run.official_graph_mutated)
        if ctx["official_graph_mutated"]:
            raise OntologyProofError("Re-extraction mutated the official graph; expected candidate-only output.")
        return (
            "Connector-driven re-extraction produced candidate-only ontology output.",
            {
                "extractionRunIds": ctx["extraction_run_ids"],
                "conceptCandidateIds": ctx["concept_candidate_ids"],
                "relationCandidateIds": ctx["relation_candidate_ids"],
                "officialGraphMutated": False,
            },
        )

    def _review_evidence(
        self,
        request: CreateOntologyProofRunRequest,
        ctx: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        reviewed_ids: list[str] = []
        now = utc_now()
        for evidence_id in ctx["evidence_fragment_ids"]:
            fragment = self.store.get_evidence_fragment(evidence_id)
            request_model = ReviewEvidenceRequest(trust_state=TrustState.REVIEWED, reviewed_by=request.reviewer)
            updated = fragment.model_copy(
                update={
                    "trust_state": request_model.trust_state,
                    "reviewed_by": request_model.reviewed_by,
                    "reviewed_at": now,
                },
                deep=True,
            )
            event = AuditEvent(
                event_type="evidence.reviewed",
                actor=request.reviewer,
                entity_type="EvidenceFragment",
                entity_id=updated.id,
                metadata={"proofRun": "v1.9.0", "trustState": updated.trust_state},
            )
            with self.store.transaction():
                self.store.update_evidence_fragment(updated)
                self.store.add_audit_event(event)
            reviewed_ids.append(updated.id)
        if not reviewed_ids:
            raise OntologyProofError("No evidence was available for proof review.")
        return "All proof EvidenceFragments were reviewed before candidate approval.", {"reviewedEvidenceFragmentIds": reviewed_ids}

    def _approve_concepts(
        self,
        request: CreateOntologyProofRunRequest,
        ctx: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        review = self._review_service()
        promoted_ids: list[str] = []
        merged_ids: list[str] = []
        candidates = [self.store.get_concept_candidate(candidate_id) for candidate_id in ctx["concept_candidate_ids"]]
        for candidate in candidates:
            if candidate.status != OntologyCandidateStatus.PENDING:
                continue
            existing = _find_concept_by_term(self.store.list_concepts(), candidate.name)
            try:
                if candidate.matched_existing_concept_id or existing is not None:
                    target_id = candidate.matched_existing_concept_id or cast(Concept, existing).id
                    response = review.merge_concept_candidate(
                        candidate.id,
                        request=_merge_concept_request(request.reviewer, target_id),
                    )
                    if response.concept is not None:
                        merged_ids.append(response.concept.id)
                else:
                    response = review.approve_concept_candidate(
                        candidate.id,
                        ApproveConceptCandidateRequest(
                            reviewed_by=request.reviewer,
                            review_note="v1.9.0 proof approval after reviewed evidence.",
                        ),
                    )
                    if response.concept is not None:
                        promoted_ids.append(response.concept.id)
            except OntologyReviewError as exc:
                raise OntologyProofError(f"Could not approve ConceptCandidate {candidate.id}: {exc}") from exc
        approved = _dedupe([*promoted_ids, *merged_ids])
        ctx["approved_concept_ids"] = approved
        if not approved:
            raise OntologyProofError("No ConceptCandidates were approved or merged.")
        return "ConceptCandidates were reviewed and promoted/merged through officialization gates.", {"approvedConceptIds": approved}

    def _approve_relations(
        self,
        request: CreateOntologyProofRunRequest,
        ctx: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        review = self._review_service()
        approved_ids: list[str] = []
        merged_ids: list[str] = []
        relation_candidates = [self.store.get_relation_candidate(candidate_id) for candidate_id in ctx["relation_candidate_ids"]]
        for candidate in relation_candidates:
            if candidate.status != OntologyCandidateStatus.PENDING:
                continue
            existing_relation = _find_relation_for_candidate(self.store.list_concept_relations(), candidate)
            try:
                if existing_relation is not None:
                    response = review.merge_relation_candidate(
                        candidate.id,
                        request=_merge_relation_request(request.reviewer, existing_relation.id),
                    )
                    if response.relation is not None:
                        merged_ids.append(response.relation.id)
                else:
                    response = review.approve_relation_candidate(
                        candidate.id,
                        ApproveRelationCandidateRequest(
                            reviewed_by=request.reviewer,
                            review_note="v1.9.0 proof relation approval after endpoint Concepts became official.",
                        ),
                    )
                    if response.relation is not None:
                        approved_ids.append(response.relation.id)
            except OntologyReviewError as exc:
                raise OntologyProofError(f"Could not approve RelationCandidate {candidate.id}: {exc}") from exc
        relation_ids = _dedupe([*approved_ids, *merged_ids])
        ctx["approved_relation_ids"] = relation_ids
        if not relation_ids:
            raise OntologyProofError("No RelationCandidates were approved or merged.")
        return "RelationCandidates were reviewed and promoted/merged into official graph edges.", {"approvedRelationIds": relation_ids}

    def _serve_graph(
        self,
        request: CreateOntologyProofRunRequest,
        ctx: dict[str, Any],
        focus: str,
    ) -> tuple[str, dict[str, Any]]:
        graph = OntologyGraphService(self.store, production_mode=self.settings.production_mode).graph(
            focus,
            depth=1,
            mode=OntologyGraphMode.OFFICIAL,
        )
        ctx["graph_response_id"] = graph.response_id
        ctx["official_graph_available"] = bool(graph.official_graph_available)
        if not graph.official_graph_available or graph.focus_concept is None:
            raise OntologyProofError("Official proof graph is not available after candidate approval.")
        if not graph.edges:
            raise OntologyProofError("Official proof graph has no reviewed relation edges.")
        if graph.trust_label != TrustLabel.OFFICIAL:
            raise OntologyProofError(f"Expected official trust label, got {graph.trust_label}.")
        return (
            "Depth-1 official ontology graph served with citations and explanation fields.",
            {
                "graphResponseId": graph.response_id,
                "focusConceptId": graph.focus_concept.id,
                "nodeCount": len(graph.nodes),
                "edgeCount": len(graph.edges),
                "trustLabel": graph.trust_label,
            },
        )

    def _run_evaluation(
        self,
        request: CreateOntologyProofRunRequest,
        ctx: dict[str, Any],
        focus: str,
    ) -> tuple[str, dict[str, Any]]:
        service = OntologyGraphEvaluationService(self.store, production_mode=self.settings.production_mode)
        task = service.create_task(
            CreateOntologyGraphEvalTaskRequest(
                name=f"v1.9.0 ontology proof - {focus}",
                concept_query=focus,
                mode=OntologyGraphMode.OFFICIAL,
                depth=1,
                expected_trust_label=TrustLabel.OFFICIAL,
                require_official_graph=True,
                require_evidence=True,
                min_evidence_count=1,
                min_node_count=2,
                min_edge_count=1,
                require_review_provenance=True,
                tags=["v1.9.0", "operator-proof", "ontology"],
                created_by=request.created_by,
                metadata={"proofRun": "v1.9.0", "focusConcept": focus},
            )
        )
        result = service.run_task(task.id)
        ctx["evaluation_task_id"] = task.id
        ctx["evaluation_result_id"] = result.id
        ctx["evaluation_success"] = bool(result.success)
        if not result.success:
            raise OntologyProofError(f"Ontology graph evaluation failed: {', '.join(result.failure_reasons)}")
        return (
            "Ontology graph evaluation passed deterministic SSOT gates.",
            {"evaluationTaskId": task.id, "evaluationResultId": result.id, "success": result.success},
        )

    def _review_service(self) -> OntologyCandidateReviewService:
        return OntologyCandidateReviewService(
            self.store,
            production_mode=self.settings.production_mode,
            authorized_reviewers=self.settings.authorized_reviewers,
        )


def _planned_checklist() -> list[OntologyProofChecklistStep]:
    return [
        OntologyProofChecklistStep(
            key="create_manual_source",
            title="Create proof manual source",
            category="source",
            goal="Create explicit proof source material without relying on hidden UI behavior.",
            checks=["manual source", "production enabled", "healthy source state"],
        ),
        OntologyProofChecklistStep(
            key="sync_manual_seed",
            title="Sync manual proof content",
            category="ingestion",
            goal="Turn manual proof text into Artifact and EvidenceFragment records.",
            checks=["Artifact created", "EvidenceFragment created", "re-extraction queued"],
        ),
        OntologyProofChecklistStep(
            key="run_reextraction",
            title="Run connector-driven re-extraction",
            category="extraction",
            goal="Generate ConceptCandidates and RelationCandidates while keeping officialGraphMutated=false.",
            checks=["ConceptCandidates exist", "RelationCandidates exist", "official graph unchanged"],
        ),
        OntologyProofChecklistStep(
            key="review_evidence",
            title="Review evidence",
            category="review",
            goal="Mark proof evidence reviewed before any official Concept or Relation can be created.",
            checks=["all proof EvidenceFragments reviewed", "reviewer identity recorded"],
        ),
        OntologyProofChecklistStep(
            key="approve_concepts",
            title="Approve ConceptCandidates",
            category="review",
            goal="Promote or merge Concepts through existing officialization gates.",
            checks=["approved/merged Concepts", "official status", "evidence retained"],
        ),
        OntologyProofChecklistStep(
            key="approve_relations",
            title="Approve RelationCandidates",
            category="review",
            goal="Promote or merge Relations only after endpoint Concepts are official.",
            checks=["approved/merged Relations", "official status", "relation evidence retained"],
        ),
        OntologyProofChecklistStep(
            key="serve_explainable_graph",
            title="Serve explainable graph",
            category="graph",
            goal="Serve a depth-1 official graph with citations, trust label, and explanation metadata.",
            checks=["official graph available", "depth=1", "trustLabel=official", "edge citations present"],
        ),
        OntologyProofChecklistStep(
            key="run_ontology_evaluation",
            title="Run ontology graph evaluation",
            category="evaluation",
            goal="Verify the served graph passes deterministic SSOT quality gates.",
            checks=["evaluation task created", "evaluation result success", "graph safety gates pass"],
        ),
    ]


def _response(
    *,
    run_id: str,
    status: OntologyProofRunStatus,
    request: CreateOntologyProofRunRequest,
    checklist: list[OntologyProofChecklistStep],
    ctx: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> OntologyProofRunResponse:
    ctx = ctx or {}
    summary = _summary(checklist, status=status, ctx=ctx)
    return OntologyProofRunResponse(
        run_id=run_id,
        status=status,
        focus_concept=_display_focus(request.focus_concept),
        reviewer=request.reviewer,
        dry_run=request.dry_run,
        source_id=ctx.get("source_id"),
        artifact_ids=list(ctx.get("artifact_ids", [])),
        evidence_fragment_ids=list(ctx.get("evidence_fragment_ids", [])),
        reextraction_run_id=ctx.get("reextraction_run_id"),
        extraction_run_ids=list(ctx.get("extraction_run_ids", [])),
        concept_candidate_ids=list(ctx.get("concept_candidate_ids", [])),
        relation_candidate_ids=list(ctx.get("relation_candidate_ids", [])),
        approved_concept_ids=list(ctx.get("approved_concept_ids", [])),
        approved_relation_ids=list(ctx.get("approved_relation_ids", [])),
        graph_response_id=ctx.get("graph_response_id"),
        evaluation_task_id=ctx.get("evaluation_task_id"),
        evaluation_result_id=ctx.get("evaluation_result_id"),
        checklist=checklist,
        summary=summary,
        limitations=limitations or _default_limitations(),
    )


def _summary(
    checklist: list[OntologyProofChecklistStep],
    *,
    status: OntologyProofRunStatus,
    ctx: dict[str, Any],
) -> OntologyProofSummary:
    required = [step for step in checklist if step.required]
    optional = [step for step in checklist if not step.required]
    return OntologyProofSummary(
        status=status,
        required_total=len(required),
        required_passed=sum(1 for step in required if step.status == OntologyProofStepStatus.PASSED),
        required_failed=sum(1 for step in required if step.status == OntologyProofStepStatus.FAILED),
        optional_total=len(optional),
        optional_passed=sum(1 for step in optional if step.status == OntologyProofStepStatus.PASSED),
        optional_failed=sum(1 for step in optional if step.status == OntologyProofStepStatus.FAILED),
        planned_count=sum(1 for step in checklist if step.status == OntologyProofStepStatus.PLANNED),
        skipped_count=sum(1 for step in checklist if step.status == OntologyProofStepStatus.SKIPPED),
        official_graph_available=bool(ctx.get("official_graph_available", False)),
        official_graph_mutated=bool(ctx.get("official_graph_mutated", False)),
        candidate_count=len(ctx.get("concept_candidate_ids", [])) + len(ctx.get("relation_candidate_ids", [])),
        approved_concept_count=len(ctx.get("approved_concept_ids", [])),
        approved_relation_count=len(ctx.get("approved_relation_ids", [])),
        evaluation_success=ctx.get("evaluation_success"),
    )


def _all_required_passed(checklist: list[OntologyProofChecklistStep]) -> bool:
    return all(
        step.status == OntologyProofStepStatus.PASSED
        for step in checklist
        if step.required
    )


def _display_focus(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    return cleaned[0].upper() + cleaned[1:] if cleaned else "Settlement"


def _default_seed_content(focus: str) -> str:
    return (
        f"{focus} is the process of finalizing obligations. "
        f"Clearing is the process of calculating obligations before {focus.lower()}. "
        f"Reconciliation is the process of verifying {focus.lower()} outcomes. "
        f"Clearing precedes {focus}. "
        f"Reconciliation validates {focus}."
    )


def _find_concept_by_term(concepts: list[Concept], term: str) -> Concept | None:
    normalized = normalize_concept_term(term)
    for concept in concepts:
        terms = {normalize_concept_term(concept.name), *(normalize_concept_term(alias) for alias in concept.aliases)}
        if normalized in terms:
            return concept
    return None


def _find_relation_for_candidate(relations: list[Any], candidate: RelationCandidate) -> Any | None:
    # Existing relation detection is best-effort. Approval remains authoritative and
    # will reject duplicates if candidate endpoints resolve differently.
    if candidate.source_concept_id is None or candidate.target_concept_id is None:
        return None
    for relation in relations:
        if (
            relation.source_concept_id == candidate.source_concept_id
            and relation.target_concept_id == candidate.target_concept_id
            and relation.relation_type == candidate.relation_type
            and relation.status == RelationStatus.OFFICIAL
        ):
            return relation
    return None


def _merge_concept_request(reviewer: str, target_concept_id: str) -> Any:
    from cornerstone.schemas import MergeConceptCandidateRequest

    return MergeConceptCandidateRequest(
        reviewed_by=reviewer,
        target_concept_id=target_concept_id,
        append_evidence=True,
        review_note="v1.9.0 proof merge into existing official Concept.",
    )


def _merge_relation_request(reviewer: str, target_relation_id: str) -> Any:
    from cornerstone.schemas import MergeRelationCandidateRequest

    return MergeRelationCandidateRequest(
        reviewed_by=reviewer,
        target_relation_id=target_relation_id,
        append_evidence=True,
        review_note="v1.9.0 proof merge into existing official Relation.",
    )


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _default_limitations() -> list[str]:
    return [
        "v1.9.0 focuses on operator proof/checklist behavior, not frontend UI/UX.",
        "The proof uses deterministic local ontology extraction; live external LLM providers remain out of scope.",
        "The proof creates explicit proof source data when confirmMutation=true.",
    ]


def _next_action_for_failure(key: str) -> str:
    return {
        "create_manual_source": "Check source creation and production/reviewer settings.",
        "sync_manual_seed": "Inspect manual source sync, upload limits, and extraction logs.",
        "run_reextraction": "Inspect queued OntologyReExtractionRun and extraction candidate output.",
        "review_evidence": "Review EvidenceFragments and authorized reviewer configuration.",
        "approve_concepts": "Resolve candidate conflicts, duplicate aliases, or officialization gate failures.",
        "approve_relations": "Ensure relation endpoint Concepts are official and evidence is reviewed.",
        "serve_explainable_graph": "Inspect official Concepts, Relations, evidence citations, and graph mode/depth.",
        "run_ontology_evaluation": "Inspect ontology evaluation failure reasons and graph response support summary.",
    }.get(key, "Inspect the prior proof step and retry after fixing the failed gate.")
