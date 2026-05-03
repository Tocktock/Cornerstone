from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from cornerstone.schemas import (
    ApproveConceptCandidateRequest,
    ApproveRelationCandidateRequest,
    AuditEvent,
    CandidateReviewBlocker,
    CandidateReviewPreview,
    CandidateReviewQueueGroup,
    CandidateReviewQueueSummary,
    Concept,
    ConceptCandidate,
    ConceptCandidateReviewResponse,
    ConceptRelation,
    ConceptStatus,
    EditConceptCandidateRequest,
    EditRelationCandidateRequest,
    MergeConceptCandidateRequest,
    MergeRelationCandidateRequest,
    OntologyCandidateStatus,
    RejectOntologyCandidateRequest,
    RelationCandidate,
    RelationCandidateReviewResponse,
    RelationStatus,
    RelationType,
    clean_concept_aliases,
    normalize_concept_term,
    utc_now,
)
from cornerstone.services.officialization import (
    ensure_reviewer_authorized,
    officialize_concept,
    officialize_concept_relation,
)
from cornerstone.store import NotFoundError


class OntologyReviewError(ValueError):
    """Raised when candidate review cannot be applied safely."""


class OntologyCandidateReviewService:
    """Promote, reject, edit, or merge ontology candidates after human review.

    This service enforces the v1.5.0 trust boundary: candidate review is the only
    path from LLM/extractor output into the official ontology graph, and every
    promotion still passes the existing officialization gate.
    """

    def __init__(
        self,
        store: object,
        *,
        production_mode: bool,
        authorized_reviewers: set[str],
    ) -> None:
        self.store = store
        self.production_mode = production_mode
        self.authorized_reviewers = authorized_reviewers

    def queue_summary(
        self,
        *,
        status: OntologyCandidateStatus | None = OntologyCandidateStatus.PENDING,
        extraction_run_id: str | None = None,
        source_id: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
    ) -> CandidateReviewQueueSummary:
        concept_candidates = self.store.list_concept_candidates(
            extraction_run_id=extraction_run_id,
            status=status,
        )
        relation_candidates = self.store.list_relation_candidates(
            extraction_run_id=extraction_run_id,
            status=status,
        )
        concept_candidates = [
            candidate
            for candidate in concept_candidates
            if _confidence_matches(candidate.confidence, min_confidence, max_confidence)
            and self._candidate_source_matches(candidate.evidence_fragment_ids, source_id)
        ]
        relation_candidates = [
            candidate
            for candidate in relation_candidates
            if _confidence_matches(candidate.confidence, min_confidence, max_confidence)
            and self._candidate_source_matches(candidate.evidence_fragment_ids, source_id)
        ]

        groups: dict[str, _QueueAccumulator] = {}
        for candidate in concept_candidates:
            group = groups.setdefault(candidate.name, _QueueAccumulator(candidate.name))
            group.pending_concept_candidate_count += 1
            group.record(candidate.extraction_run_id, self._source_ids_for_evidence(candidate.evidence_fragment_ids))
            group.record_confidence(candidate.confidence)
            if self._candidate_blockers(candidate.evidence_fragment_ids):
                group.blocked_count += 1
        for candidate in relation_candidates:
            focus = candidate.target_name or candidate.source_name
            group = groups.setdefault(focus, _QueueAccumulator(focus))
            group.pending_relation_candidate_count += 1
            group.record(candidate.extraction_run_id, self._source_ids_for_evidence(candidate.evidence_fragment_ids))
            group.record_confidence(candidate.confidence)
            if self._candidate_blockers(candidate.evidence_fragment_ids):
                group.blocked_count += 1

        counts_by_status: dict[str, int] = {}
        for candidate in [*self.store.list_concept_candidates(), *self.store.list_relation_candidates()]:
            counts_by_status[str(candidate.status)] = counts_by_status.get(str(candidate.status), 0) + 1
        counts_by_run: dict[str, int] = {}
        counts_by_source: dict[str, int] = {}
        for candidate in [*concept_candidates, *relation_candidates]:
            counts_by_run[candidate.extraction_run_id] = counts_by_run.get(candidate.extraction_run_id, 0) + 1
            for source in self._source_ids_for_evidence(candidate.evidence_fragment_ids):
                counts_by_source[source] = counts_by_source.get(source, 0) + 1

        return CandidateReviewQueueSummary(
            total_pending_concept_candidates=len(
                [candidate for candidate in concept_candidates if candidate.status == OntologyCandidateStatus.PENDING]
            ),
            total_pending_relation_candidates=len(
                [candidate for candidate in relation_candidates if candidate.status == OntologyCandidateStatus.PENDING]
            ),
            grouped_by_focus_concept=[accumulator.to_group() for accumulator in sorted(groups.values(), key=lambda item: item.focus_concept.casefold())],
            counts_by_status=counts_by_status,
            counts_by_run=counts_by_run,
            counts_by_source=counts_by_source,
            machine_readable_next_actions=[
                "review_blocked_evidence",
                "preview_candidate_action",
                "approve_merge_or_reject_explicitly",
            ],
        )

    def preview_concept_candidate(
        self,
        candidate_id: str,
        *,
        action: str,
        target_concept_id: str | None = None,
    ) -> CandidateReviewPreview:
        candidate = self.store.get_concept_candidate(candidate_id)
        blockers = self._candidate_blockers(candidate.evidence_fragment_ids)
        official_graph_will_change = action in {"approve", "merge"} and not blockers
        if action == "reject":
            mutation_summary = "Rejecting this candidate records a review outcome and creates no official Concept."
            official_graph_will_change = False
        elif action == "merge":
            if target_concept_id is None:
                blockers.append(CandidateReviewBlocker(code="target_required", message="Merge preview requires targetConceptId."))
            else:
                try:
                    self.store.get_concept(target_concept_id)
                except NotFoundError:
                    blockers.append(CandidateReviewBlocker(code="target_missing", message=f"Target Concept not found: {target_concept_id}."))
            mutation_summary = "Merging this candidate preserves evidence on an existing Concept after explicit review."
        else:
            duplicate = self._find_concept_by_term(candidate.name)
            if duplicate is not None:
                blockers.append(CandidateReviewBlocker(code="duplicate_concept", message=f"Concept already exists: {duplicate.id}."))
            mutation_summary = "Approving this candidate creates one official Concept after evidence and reviewer checks."
        return _preview(
            candidate_id=candidate.id,
            candidate_type="concept",
            action=action,
            official_graph_will_change=official_graph_will_change,
            mutation_summary=mutation_summary,
            target_concept_id=target_concept_id,
            blockers=blockers,
        )

    def preview_relation_candidate(
        self,
        candidate_id: str,
        *,
        action: str,
        target_relation_id: str | None = None,
    ) -> CandidateReviewPreview:
        candidate = self.store.get_relation_candidate(candidate_id)
        blockers = self._candidate_blockers(candidate.evidence_fragment_ids)
        official_graph_will_change = action in {"approve", "merge"} and not blockers
        if action == "reject":
            mutation_summary = "Rejecting this candidate records a review outcome and creates no official Relation."
            official_graph_will_change = False
        elif action == "merge":
            if target_relation_id is None:
                blockers.append(CandidateReviewBlocker(code="target_required", message="Merge preview requires targetRelationId."))
            else:
                try:
                    self.store.get_concept_relation(target_relation_id)
                except NotFoundError:
                    blockers.append(CandidateReviewBlocker(code="target_missing", message=f"Target Relation not found: {target_relation_id}."))
            mutation_summary = "Merging this candidate preserves evidence on an existing ConceptRelation after explicit review."
        else:
            for label, term, concept_id in [
                ("source", candidate.source_name, candidate.source_concept_id),
                ("target", candidate.target_name, candidate.target_concept_id),
            ]:
                if concept_id is not None:
                    continue
                if self._find_concept_by_term(term) is None:
                    blockers.append(
                        CandidateReviewBlocker(
                            code=f"{label}_concept_unresolved",
                            message=f"Relation {label} endpoint must resolve to an official Concept before approval: {term}.",
                        )
                    )
            mutation_summary = "Approving this candidate creates one official ConceptRelation after endpoint, evidence, and reviewer checks."
        return _preview(
            candidate_id=candidate.id,
            candidate_type="relation",
            action=action,
            official_graph_will_change=official_graph_will_change,
            mutation_summary=mutation_summary,
            target_relation_id=target_relation_id,
            blockers=blockers,
        )

    # Concept candidates
    def edit_concept_candidate(
        self,
        candidate_id: str,
        request: EditConceptCandidateRequest,
    ) -> ConceptCandidateReviewResponse:
        ensure_reviewer_authorized(request.edited_by, self.authorized_reviewers)
        candidate = self.store.get_concept_candidate(candidate_id)
        _ensure_pending(candidate)
        if request.evidence_fragment_ids is not None:
            self._ensure_evidence_exists(request.evidence_fragment_ids)

        name = request.name if request.name is not None else candidate.name
        aliases = request.aliases if request.aliases is not None else candidate.aliases
        normalized_name = normalize_concept_term(name)
        updated = candidate.model_copy(
            update={
                "name": name,
                "normalized_name": normalized_name,
                "aliases": clean_concept_aliases(aliases, primary_name=name),
                "proposed_definition": request.proposed_definition
                if request.proposed_definition is not None
                else candidate.proposed_definition,
                "concept_type": request.concept_type if request.concept_type is not None else candidate.concept_type,
                "evidence_fragment_ids": request.evidence_fragment_ids
                if request.evidence_fragment_ids is not None
                else candidate.evidence_fragment_ids,
                "confidence": request.confidence if request.confidence is not None else candidate.confidence,
                "rationale": request.rationale if request.rationale is not None else candidate.rationale,
            },
            deep=True,
        )
        event = AuditEvent(
            event_type="concept_candidate.edited",
            actor=request.edited_by,
            entity_type="ConceptCandidate",
            entity_id=candidate.id,
            metadata={"candidateName": updated.name, "extractionRunId": updated.extraction_run_id},
        )
        with self.store.transaction():
            saved = self.store.update_concept_candidate(updated)
            self.store.add_audit_event(event)
        return ConceptCandidateReviewResponse(candidate=saved, audit_event_ids=[event.id])

    def reject_concept_candidate(
        self,
        candidate_id: str,
        request: RejectOntologyCandidateRequest,
    ) -> ConceptCandidateReviewResponse:
        ensure_reviewer_authorized(request.reviewed_by, self.authorized_reviewers)
        candidate = self.store.get_concept_candidate(candidate_id)
        _ensure_pending(candidate)
        now = utc_now()
        updated = candidate.model_copy(
            update={
                "status": OntologyCandidateStatus.REJECTED,
                "reviewed_by": request.reviewed_by,
                "reviewed_at": now,
                "review_note": request.review_note,
            },
            deep=True,
        )
        event = AuditEvent(
            event_type="concept_candidate.rejected",
            actor=request.reviewed_by,
            entity_type="ConceptCandidate",
            entity_id=candidate.id,
            metadata={"candidateName": candidate.name, "reviewNote": request.review_note},
        )
        with self.store.transaction():
            saved = self.store.update_concept_candidate(updated)
            self.store.add_audit_event(event)
        return ConceptCandidateReviewResponse(candidate=saved, audit_event_ids=[event.id])

    def approve_concept_candidate(
        self,
        candidate_id: str,
        request: ApproveConceptCandidateRequest,
    ) -> ConceptCandidateReviewResponse:
        ensure_reviewer_authorized(request.reviewed_by, self.authorized_reviewers)
        candidate = self.store.get_concept_candidate(candidate_id)
        _ensure_pending(candidate)
        self._ensure_evidence_exists(candidate.evidence_fragment_ids)

        name = request.name or candidate.name
        aliases = clean_concept_aliases(
            request.aliases if request.aliases is not None else candidate.aliases,
            primary_name=name,
        )
        self._ensure_concept_terms_available(name, aliases)

        draft = Concept(
            name=name,
            aliases=aliases,
            short_definition=request.short_definition or candidate.proposed_definition,
            body=request.body,
            owner=request.owner,
            status=ConceptStatus.CANDIDATE,
            evidence_fragment_ids=list(candidate.evidence_fragment_ids),
            created_by=request.reviewed_by,
        )
        official, official_event = officialize_concept(
            draft,
            reviewed_by=request.reviewed_by,
            evidence=self.store.list_evidence_fragments(),
            decision_records=self.store.list_decision_records(),
            artifacts=self.store.list_artifacts(),
            data_sources=self.store.list_data_sources(),
            production_mode=self.production_mode,
            authorized_reviewers=self.authorized_reviewers,
        )
        now = utc_now()
        updated_candidate = candidate.model_copy(
            update={
                "status": OntologyCandidateStatus.APPROVED,
                "reviewed_by": request.reviewed_by,
                "reviewed_at": now,
                "review_note": request.review_note,
                "promoted_concept_id": official.id,
            },
            deep=True,
        )
        candidate_event = AuditEvent(
            event_type="concept_candidate.approved",
            actor=request.reviewed_by,
            entity_type="ConceptCandidate",
            entity_id=candidate.id,
            metadata={
                "candidateName": candidate.name,
                "promotedConceptId": official.id,
                "evidenceFragmentCount": len(candidate.evidence_fragment_ids),
            },
        )
        with self.store.transaction():
            saved_concept = self.store.add_concept(official)
            saved_candidate = self.store.update_concept_candidate(updated_candidate)
            self.store.add_audit_event(candidate_event)
            self.store.add_audit_event(official_event)
        return ConceptCandidateReviewResponse(
            candidate=saved_candidate,
            concept=saved_concept,
            audit_event_ids=[candidate_event.id, official_event.id],
        )

    def merge_concept_candidate(
        self,
        candidate_id: str,
        request: MergeConceptCandidateRequest,
    ) -> ConceptCandidateReviewResponse:
        ensure_reviewer_authorized(request.reviewed_by, self.authorized_reviewers)
        candidate = self.store.get_concept_candidate(candidate_id)
        _ensure_pending(candidate)
        target = self.store.get_concept(request.target_concept_id)
        self._ensure_evidence_exists(candidate.evidence_fragment_ids)

        merged_evidence = _merge_unique(target.evidence_fragment_ids, candidate.evidence_fragment_ids) if request.append_evidence else list(target.evidence_fragment_ids)
        requested_aliases = request.aliases if request.aliases is not None else []
        candidate_alias_terms = [candidate.name, *candidate.aliases]
        merged_aliases = clean_concept_aliases(
            _merge_unique(target.aliases, candidate_alias_terms, requested_aliases),
            primary_name=target.name,
        )
        self._ensure_concept_terms_available(target.name, merged_aliases, exclude_concept_ids={target.id})
        updated_target = target.model_copy(
            update={
                "aliases": merged_aliases,
                "short_definition": request.short_definition or target.short_definition,
                "body": request.body if request.body is not None else target.body,
                "evidence_fragment_ids": merged_evidence,
                "updated_at": utc_now(),
            },
            deep=True,
        )
        official_event: AuditEvent | None = None
        if updated_target.status == ConceptStatus.OFFICIAL:
            updated_target, official_event = officialize_concept(
                updated_target,
                reviewed_by=request.reviewed_by,
                evidence=self.store.list_evidence_fragments(),
                decision_records=self.store.list_decision_records(),
                artifacts=self.store.list_artifacts(),
                data_sources=self.store.list_data_sources(),
                production_mode=self.production_mode,
                authorized_reviewers=self.authorized_reviewers,
            )

        now = utc_now()
        updated_candidate = candidate.model_copy(
            update={
                "status": OntologyCandidateStatus.MERGED,
                "reviewed_by": request.reviewed_by,
                "reviewed_at": now,
                "review_note": request.review_note,
                "merged_into_concept_id": target.id,
            },
            deep=True,
        )
        merge_event = AuditEvent(
            event_type="concept_candidate.merged",
            actor=request.reviewed_by,
            entity_type="ConceptCandidate",
            entity_id=candidate.id,
            metadata={"candidateName": candidate.name, "targetConceptId": target.id},
        )
        with self.store.transaction():
            saved_target = self.store.update_concept(updated_target)
            saved_candidate = self.store.update_concept_candidate(updated_candidate)
            self.store.add_audit_event(merge_event)
            if official_event is not None:
                self.store.add_audit_event(official_event)
        audit_ids = [merge_event.id]
        if official_event is not None:
            audit_ids.append(official_event.id)
        return ConceptCandidateReviewResponse(candidate=saved_candidate, concept=saved_target, audit_event_ids=audit_ids)

    # Relation candidates
    def edit_relation_candidate(
        self,
        candidate_id: str,
        request: EditRelationCandidateRequest,
    ) -> RelationCandidateReviewResponse:
        ensure_reviewer_authorized(request.edited_by, self.authorized_reviewers)
        candidate = self.store.get_relation_candidate(candidate_id)
        _ensure_pending(candidate)
        if request.evidence_fragment_ids is not None:
            self._ensure_evidence_exists(request.evidence_fragment_ids)
        if request.source_concept_id is not None:
            self.store.get_concept(request.source_concept_id)
        if request.target_concept_id is not None:
            self.store.get_concept(request.target_concept_id)

        source_name = request.source_name if request.source_name is not None else candidate.source_name
        target_name = request.target_name if request.target_name is not None else candidate.target_name
        updated = candidate.model_copy(
            update={
                "source_name": source_name,
                "target_name": target_name,
                "normalized_source_name": normalize_concept_term(source_name),
                "normalized_target_name": normalize_concept_term(target_name),
                "source_concept_id": request.source_concept_id
                if request.source_concept_id is not None
                else candidate.source_concept_id,
                "target_concept_id": request.target_concept_id
                if request.target_concept_id is not None
                else candidate.target_concept_id,
                "relation_type": request.relation_type if request.relation_type is not None else candidate.relation_type,
                "evidence_fragment_ids": request.evidence_fragment_ids
                if request.evidence_fragment_ids is not None
                else candidate.evidence_fragment_ids,
                "confidence": request.confidence if request.confidence is not None else candidate.confidence,
                "rationale": request.rationale if request.rationale is not None else candidate.rationale,
            },
            deep=True,
        )
        event = AuditEvent(
            event_type="relation_candidate.edited",
            actor=request.edited_by,
            entity_type="RelationCandidate",
            entity_id=candidate.id,
            metadata={"relationType": updated.relation_type, "extractionRunId": updated.extraction_run_id},
        )
        with self.store.transaction():
            saved = self.store.update_relation_candidate(updated)
            self.store.add_audit_event(event)
        return RelationCandidateReviewResponse(candidate=saved, audit_event_ids=[event.id])

    def reject_relation_candidate(
        self,
        candidate_id: str,
        request: RejectOntologyCandidateRequest,
    ) -> RelationCandidateReviewResponse:
        ensure_reviewer_authorized(request.reviewed_by, self.authorized_reviewers)
        candidate = self.store.get_relation_candidate(candidate_id)
        _ensure_pending(candidate)
        now = utc_now()
        updated = candidate.model_copy(
            update={
                "status": OntologyCandidateStatus.REJECTED,
                "reviewed_by": request.reviewed_by,
                "reviewed_at": now,
                "review_note": request.review_note,
            },
            deep=True,
        )
        event = AuditEvent(
            event_type="relation_candidate.rejected",
            actor=request.reviewed_by,
            entity_type="RelationCandidate",
            entity_id=candidate.id,
            metadata={"relationType": candidate.relation_type, "reviewNote": request.review_note},
        )
        with self.store.transaction():
            saved = self.store.update_relation_candidate(updated)
            self.store.add_audit_event(event)
        return RelationCandidateReviewResponse(candidate=saved, audit_event_ids=[event.id])

    def approve_relation_candidate(
        self,
        candidate_id: str,
        request: ApproveRelationCandidateRequest,
    ) -> RelationCandidateReviewResponse:
        ensure_reviewer_authorized(request.reviewed_by, self.authorized_reviewers)
        candidate = self.store.get_relation_candidate(candidate_id)
        _ensure_pending(candidate)
        self._ensure_evidence_exists(candidate.evidence_fragment_ids)
        source_concept_id = self._resolve_relation_endpoint(
            explicit_concept_id=request.source_concept_id,
            candidate_concept_id=candidate.source_concept_id,
            candidate_id=candidate.source_candidate_id,
            term=candidate.source_name,
            endpoint_label="source",
        )
        target_concept_id = self._resolve_relation_endpoint(
            explicit_concept_id=request.target_concept_id,
            candidate_concept_id=candidate.target_concept_id,
            candidate_id=candidate.target_candidate_id,
            term=candidate.target_name,
            endpoint_label="target",
        )
        relation_type = request.relation_type or candidate.relation_type
        if self._find_existing_relation(source_concept_id, target_concept_id, relation_type) is not None:
            raise OntologyReviewError("ConceptRelation already exists for this source, target, and relation type; merge the candidate instead.")
        draft = ConceptRelation(
            source_concept_id=source_concept_id,
            target_concept_id=target_concept_id,
            relation_type=relation_type,
            status=RelationStatus.CANDIDATE,
            evidence_fragment_ids=list(candidate.evidence_fragment_ids),
            created_by=request.reviewed_by,
        )
        official, official_event = officialize_concept_relation(
            draft,
            reviewed_by=request.reviewed_by,
            concepts=self.store.list_concepts(),
            evidence=self.store.list_evidence_fragments(),
            decision_records=self.store.list_decision_records(),
            artifacts=self.store.list_artifacts(),
            data_sources=self.store.list_data_sources(),
            production_mode=self.production_mode,
            authorized_reviewers=self.authorized_reviewers,
        )
        now = utc_now()
        updated_candidate = candidate.model_copy(
            update={
                "status": OntologyCandidateStatus.APPROVED,
                "reviewed_by": request.reviewed_by,
                "reviewed_at": now,
                "review_note": request.review_note,
                "source_concept_id": source_concept_id,
                "target_concept_id": target_concept_id,
                "relation_type": relation_type,
                "promoted_relation_id": official.id,
            },
            deep=True,
        )
        candidate_event = AuditEvent(
            event_type="relation_candidate.approved",
            actor=request.reviewed_by,
            entity_type="RelationCandidate",
            entity_id=candidate.id,
            metadata={
                "promotedRelationId": official.id,
                "sourceConceptId": source_concept_id,
                "targetConceptId": target_concept_id,
                "relationType": relation_type,
            },
        )
        with self.store.transaction():
            saved_relation = self.store.add_concept_relation(official)
            saved_candidate = self.store.update_relation_candidate(updated_candidate)
            self.store.add_audit_event(candidate_event)
            self.store.add_audit_event(official_event)
        return RelationCandidateReviewResponse(
            candidate=saved_candidate,
            relation=saved_relation,
            audit_event_ids=[candidate_event.id, official_event.id],
        )

    def merge_relation_candidate(
        self,
        candidate_id: str,
        request: MergeRelationCandidateRequest,
    ) -> RelationCandidateReviewResponse:
        ensure_reviewer_authorized(request.reviewed_by, self.authorized_reviewers)
        candidate = self.store.get_relation_candidate(candidate_id)
        _ensure_pending(candidate)
        target = self.store.get_concept_relation(request.target_relation_id)
        self._ensure_evidence_exists(candidate.evidence_fragment_ids)
        merged_evidence = _merge_unique(target.evidence_fragment_ids, candidate.evidence_fragment_ids) if request.append_evidence else list(target.evidence_fragment_ids)
        updated_target = target.model_copy(update={"evidence_fragment_ids": merged_evidence, "updated_at": utc_now()}, deep=True)
        official_event: AuditEvent | None = None
        if updated_target.status == RelationStatus.OFFICIAL:
            updated_target, official_event = officialize_concept_relation(
                updated_target,
                reviewed_by=request.reviewed_by,
                concepts=self.store.list_concepts(),
                evidence=self.store.list_evidence_fragments(),
                decision_records=self.store.list_decision_records(),
                artifacts=self.store.list_artifacts(),
                data_sources=self.store.list_data_sources(),
                production_mode=self.production_mode,
                authorized_reviewers=self.authorized_reviewers,
            )
        now = utc_now()
        updated_candidate = candidate.model_copy(
            update={
                "status": OntologyCandidateStatus.MERGED,
                "reviewed_by": request.reviewed_by,
                "reviewed_at": now,
                "review_note": request.review_note,
                "merged_into_relation_id": target.id,
            },
            deep=True,
        )
        merge_event = AuditEvent(
            event_type="relation_candidate.merged",
            actor=request.reviewed_by,
            entity_type="RelationCandidate",
            entity_id=candidate.id,
            metadata={"targetRelationId": target.id, "relationType": candidate.relation_type},
        )
        with self.store.transaction():
            saved_target = self.store.update_concept_relation(updated_target)
            saved_candidate = self.store.update_relation_candidate(updated_candidate)
            self.store.add_audit_event(merge_event)
            if official_event is not None:
                self.store.add_audit_event(official_event)
        audit_ids = [merge_event.id]
        if official_event is not None:
            audit_ids.append(official_event.id)
        return RelationCandidateReviewResponse(candidate=saved_candidate, relation=saved_target, audit_event_ids=audit_ids)

    # Shared helpers
    def _ensure_evidence_exists(self, evidence_fragment_ids: Iterable[str]) -> None:
        for evidence_id in evidence_fragment_ids:
            self.store.get_evidence_fragment(evidence_id)

    def _candidate_source_matches(self, evidence_fragment_ids: Iterable[str], source_id: str | None) -> bool:
        if source_id is None:
            return True
        return source_id in self._source_ids_for_evidence(evidence_fragment_ids)

    def _source_ids_for_evidence(self, evidence_fragment_ids: Iterable[str]) -> list[str]:
        source_ids: list[str] = []
        for evidence_id in evidence_fragment_ids:
            try:
                evidence = self.store.get_evidence_fragment(evidence_id)
            except NotFoundError:
                continue
            source_id = evidence.provenance.data_source_id
            if source_id not in source_ids:
                source_ids.append(source_id)
        return source_ids

    def _candidate_blockers(self, evidence_fragment_ids: Iterable[str]) -> list[CandidateReviewBlocker]:
        blockers: list[CandidateReviewBlocker] = []
        for evidence_id in evidence_fragment_ids:
            try:
                evidence = self.store.get_evidence_fragment(evidence_id)
            except NotFoundError:
                blockers.append(
                    CandidateReviewBlocker(
                        code="missing_evidence",
                        message=f"EvidenceFragment not found: {evidence_id}.",
                        evidence_fragment_id=evidence_id,
                    )
                )
                continue
            if str(evidence.trust_state) != "reviewed":
                blockers.append(
                    CandidateReviewBlocker(
                        code="evidence_not_reviewed",
                        message=f"EvidenceFragment must be reviewed before candidate approval or merge: {evidence_id}.",
                        evidence_fragment_id=evidence_id,
                    )
                )
        return blockers

    def _ensure_concept_terms_available(
        self,
        name: str,
        aliases: list[str],
        *,
        exclude_concept_ids: set[str] | None = None,
    ) -> None:
        excluded = exclude_concept_ids or set()
        requested_terms = {normalize_concept_term(name), *(normalize_concept_term(alias) for alias in aliases)}
        for concept in self.store.list_concepts():
            if concept.id in excluded:
                continue
            existing_terms = {normalize_concept_term(concept.name), *(normalize_concept_term(alias) for alias in concept.aliases)}
            overlap = requested_terms & existing_terms
            if overlap:
                blocked_term = sorted(overlap)[0]
                raise OntologyReviewError(f"Concept name or alias already exists: {blocked_term}")

    def _resolve_relation_endpoint(
        self,
        *,
        explicit_concept_id: str | None,
        candidate_concept_id: str | None,
        candidate_id: str | None,
        term: str,
        endpoint_label: str,
    ) -> str:
        if explicit_concept_id is not None:
            self.store.get_concept(explicit_concept_id)
            return explicit_concept_id
        if candidate_concept_id is not None:
            self.store.get_concept(candidate_concept_id)
            return candidate_concept_id
        if candidate_id is not None:
            try:
                concept_candidate = self.store.get_concept_candidate(candidate_id)
            except NotFoundError:
                concept_candidate = None
            if concept_candidate is not None:
                if concept_candidate.promoted_concept_id is not None:
                    self.store.get_concept(concept_candidate.promoted_concept_id)
                    return concept_candidate.promoted_concept_id
                if concept_candidate.merged_into_concept_id is not None:
                    self.store.get_concept(concept_candidate.merged_into_concept_id)
                    return concept_candidate.merged_into_concept_id
                if concept_candidate.matched_existing_concept_id is not None:
                    self.store.get_concept(concept_candidate.matched_existing_concept_id)
                    return concept_candidate.matched_existing_concept_id
        matched = self._find_concept_by_term(term)
        if matched is not None:
            return matched.id
        raise OntologyReviewError(
            f"RelationCandidate {endpoint_label} endpoint must resolve to an existing official Concept before approval: {term}"
        )

    def _find_concept_by_term(self, term: str) -> Concept | None:
        normalized = normalize_concept_term(term)
        for concept in self.store.list_concepts():
            terms = {normalize_concept_term(concept.name), *(normalize_concept_term(alias) for alias in concept.aliases)}
            if normalized in terms:
                return concept
        return None

    def _find_existing_relation(
        self, source_concept_id: str, target_concept_id: str, relation_type: RelationType
    ) -> ConceptRelation | None:
        for relation in self.store.list_concept_relations():
            if (
                relation.source_concept_id == source_concept_id
                and relation.target_concept_id == target_concept_id
                and relation.relation_type == relation_type
            ):
                return relation
        return None


def _ensure_pending(candidate: ConceptCandidate | RelationCandidate) -> None:
    if candidate.status != OntologyCandidateStatus.PENDING:
        raise OntologyReviewError(f"Only pending ontology candidates can be reviewed or edited: {candidate.id}")


def _merge_unique(*groups: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for value in group:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
    return result


@dataclass
class _QueueAccumulator:
    focus_concept: str
    pending_concept_candidate_count: int = 0
    pending_relation_candidate_count: int = 0
    high_confidence_count: int = 0
    low_confidence_count: int = 0
    blocked_count: int = 0
    source_ids: set[str] | None = None
    extraction_run_ids: set[str] | None = None

    def record(self, extraction_run_id: str, source_ids: list[str]) -> None:
        if self.source_ids is None:
            self.source_ids = set()
        if self.extraction_run_ids is None:
            self.extraction_run_ids = set()
        self.extraction_run_ids.add(extraction_run_id)
        self.source_ids.update(source_ids)

    def record_confidence(self, confidence: float) -> None:
        if confidence >= 0.75:
            self.high_confidence_count += 1
        if confidence < 0.6:
            self.low_confidence_count += 1

    def to_group(self) -> CandidateReviewQueueGroup:
        actions = ["preview_candidate_action"]
        if self.blocked_count:
            actions.insert(0, "review_blocked_evidence")
        return CandidateReviewQueueGroup(
            focus_concept=self.focus_concept,
            pending_concept_candidate_count=self.pending_concept_candidate_count,
            pending_relation_candidate_count=self.pending_relation_candidate_count,
            high_confidence_count=self.high_confidence_count,
            low_confidence_count=self.low_confidence_count,
            blocked_count=self.blocked_count,
            source_ids=sorted(self.source_ids or set()),
            extraction_run_ids=sorted(self.extraction_run_ids or set()),
            next_actions=actions,
        )


def _confidence_matches(
    confidence: float,
    min_confidence: float | None,
    max_confidence: float | None,
) -> bool:
    if min_confidence is not None and confidence < min_confidence:
        return False
    if max_confidence is not None and confidence > max_confidence:
        return False
    return True


def _preview(
    *,
    candidate_id: str,
    candidate_type: str,
    action: str,
    official_graph_will_change: bool,
    mutation_summary: str,
    blockers: list[CandidateReviewBlocker],
    target_concept_id: str | None = None,
    target_relation_id: str | None = None,
) -> CandidateReviewPreview:
    can_apply = not blockers
    next_actions = ["apply_review_action"] if can_apply else ["resolve_blockers"]
    return CandidateReviewPreview(
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        action=action,
        can_apply=can_apply,
        official_graph_will_change=official_graph_will_change and can_apply,
        mutation_summary=mutation_summary,
        target_concept_id=target_concept_id,
        target_relation_id=target_relation_id,
        evidence_preserved=True,
        blocker_reasons=blockers,
        next_actions=next_actions,
    )
