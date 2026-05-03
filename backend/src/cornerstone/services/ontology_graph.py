from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cornerstone.schemas import (
    CitationSupportRef,
    CitationSupportType,
    Concept,
    ConceptRelation,
    ConceptStatus,
    EvidenceCitation,
    EvidenceFragment,
    FreshnessState,
    FreshnessSummary,
    GraphEntitySupportSummary,
    OntologyCandidateGraphSummary,
    OntologyCandidateStatus,
    OntologyGraphEdge,
    OntologyGraphExplanation,
    OntologyGraphMode,
    OntologyGraphNode,
    OntologyGraphResponse,
    OntologyGraphSupportSummary,
    OntologyGraphVisualization,
    OntologyGraphVisualizationEdge,
    OntologyGraphVisualizationNode,
    OntologySearchMatchType,
    OntologySearchResponse,
    OntologySearchResult,
    RelationStatus,
    ReviewProvenance,
    TrustLabel,
    TrustState,
    normalize_concept_term,
)
from cornerstone.domain.ontology import (
    CITABLE_TRUST_STATES,
    TERMINAL_CONCEPT_STATUSES,
    TERMINAL_RELATION_STATUSES,
    ensure_supported_graph_depth,
)
from cornerstone.services.evidence_support import (
    citation_validity_errors,
    decision_record_or_none,
    dedupe_supported_evidence,
    is_from_servable_source,
    source_limitations_for,
)
from cornerstone.store import NotFoundError


@dataclass(frozen=True)
class _SearchHit:
    concept: Concept
    matched_by: OntologySearchMatchType
    matched_value: str
    score: float


@dataclass(frozen=True)
class _SupportedEvidence:
    fragment: EvidenceFragment
    supports: tuple[CitationSupportRef, ...]


class OntologyGraphService:
    """Serves v1.6.0 ontology search, graph, and explanation responses.

    The graph still reads already-existing Concepts and ConceptRelations only.
    It never calls LLMs, never promotes candidates, and never infers missing edges.
    v1.6.0 adds explanation metadata, per-node/per-edge provenance, support
    summaries, and candidate-boundary summaries to make graph serving easier to
    trust and operate.
    """

    def __init__(self, store: Any, *, production_mode: bool = True) -> None:
        self.store = store
        self.production_mode = production_mode

    def search(
        self,
        query: str,
        *,
        mode: OntologyGraphMode = OntologyGraphMode.OFFICIAL,
        limit: int = 10,
    ) -> OntologySearchResponse:
        normalized_query = normalize_concept_term(query)
        hits = self._search_hits(normalized_query, mode=mode)
        results = [
            OntologySearchResult(
                id=hit.concept.id,
                name=hit.concept.name,
                aliases=hit.concept.aliases,
                short_definition=hit.concept.short_definition,
                status=hit.concept.status,
                matched_by=hit.matched_by,
                matched_value=hit.matched_value,
                score=hit.score,
            )
            for hit in hits[:limit]
        ]
        return OntologySearchResponse(query=query, mode=mode, results=results)

    def graph(
        self,
        concept_query: str,
        *,
        depth: int = 1,
        mode: OntologyGraphMode = OntologyGraphMode.OFFICIAL,
    ) -> OntologyGraphResponse:
        ensure_supported_graph_depth(depth, context="ontology graph")

        normalized_query = normalize_concept_term(concept_query)
        focus = self._find_focus_concept(normalized_query, mode=mode)
        if focus is None:
            candidate_summary = self._candidate_summary_for_query(normalized_query, focus=None)
            limitations = [
                "No matching Concept was found for the requested ontology graph mode.",
                "v1.6.0 does not create or infer Concepts during graph serving; create/review Concepts before official graph serving.",
            ]
            if candidate_summary.has_pending_candidates:
                limitations.append(
                    "Pending ontology candidates match this query, but candidates are not official graph objects until reviewed."
                )
            explanation = _explanation_for_missing_graph(
                concept_query=concept_query,
                mode=mode,
                depth=depth,
                candidate_summary=candidate_summary,
            )
            return OntologyGraphResponse(
                query=concept_query,
                mode=mode,
                depth=depth,
                freshness=FreshnessSummary(state=FreshnessState.UNKNOWN),
                trust_label=TrustLabel.UNSUPPORTED,
                support_summary=OntologyGraphSupportSummary(),
                candidate_summary=candidate_summary,
                explanation=explanation,
                visualization=_visualization_payload(
                    focus_concept=None,
                    nodes=[],
                    edges=[],
                    evidence=[],
                    trust_label=TrustLabel.UNSUPPORTED,
                ),
                limitations=limitations,
                official_graph_available=False,
            )

        concepts_by_id = {concept.id: concept for concept in self.store.list_concepts()}
        node_concepts_by_id: dict[str, Concept] = {focus.id: focus}
        edges: list[OntologyGraphEdge] = []
        invalid_relation_count = 0

        if depth == 1:
            for relation in self.store.list_concept_relations(concept_id=focus.id):
                if not self._relation_allowed(relation, mode=mode, concepts_by_id=concepts_by_id):
                    continue
                other_concept_id = (
                    relation.target_concept_id if relation.source_concept_id == focus.id else relation.source_concept_id
                )
                other_concept = concepts_by_id.get(other_concept_id)
                if other_concept is None:
                    invalid_relation_count += 1
                    continue
                node_concepts_by_id[other_concept.id] = other_concept
                edges.append(
                    _edge_from_relation(
                        relation,
                        concepts_by_id=concepts_by_id,
                        focus_concept_id=focus.id,
                        support_summary=self._support_summary_for_relation(relation),
                    )
                )

        nodes_by_id: dict[str, OntologyGraphNode] = {
            concept.id: _node_from_concept(
                concept,
                is_focus=concept.id == focus.id,
                support_summary=self._support_summary_for_ids(concept.evidence_fragment_ids),
            )
            for concept in node_concepts_by_id.values()
        }

        evidence, invalid_evidence_count = self._evidence_for_graph(
            focus_concept_id=focus.id,
            node_concepts=list(node_concepts_by_id.values()),
            edges=edges,
        )
        citations = [_citation_from(item) for item in evidence]
        support_fragments = [item.fragment for item in evidence]
        freshness = _summarize_freshness(support_fragments)
        support_summary = _support_summary_for_graph(
            nodes=list(nodes_by_id.values()),
            edges=edges,
            evidence=support_fragments,
            invalid_evidence_count=invalid_evidence_count,
            invalid_relation_count=invalid_relation_count,
        )
        trust_label = _trust_label_for_graph(
            focus=focus,
            nodes=list(nodes_by_id.values()),
            edges=edges,
            evidence=support_fragments,
            freshness=freshness,
            mode=mode,
        )
        candidate_summary = self._candidate_summary_for_query(normalized_query, focus=focus)
        limitations = _limitations_for_graph(
            focus=focus,
            edges=edges,
            evidence=support_fragments,
            trust_label=trust_label,
            mode=mode,
            invalid_relation_count=invalid_relation_count,
            invalid_evidence_count=invalid_evidence_count,
            production_mode=self.production_mode,
            candidate_summary=candidate_summary,
        )
        limitations.extend(source_limitations_for(self.store, support_fragments))
        limitations = _dedupe_preserve_order(limitations)

        nodes = sorted(nodes_by_id.values(), key=lambda item: (not item.is_focus, item.name.casefold(), item.id))
        edges = sorted(edges, key=lambda item: (item.relation_type, item.source_concept_id, item.target_concept_id, item.id))
        explanation = _explanation_for_graph(
            focus=focus,
            nodes=nodes,
            edges=edges,
            evidence=support_fragments,
            trust_label=trust_label,
            freshness=freshness,
            support_summary=support_summary,
            candidate_summary=candidate_summary,
            mode=mode,
            depth=depth,
        )

        return OntologyGraphResponse(
            query=concept_query,
            mode=mode,
            depth=depth,
            focus_concept=nodes_by_id[focus.id],
            nodes=nodes,
            edges=edges,
            evidence=citations,
            freshness=freshness,
            trust_label=trust_label,
            support_summary=support_summary,
            candidate_summary=candidate_summary,
            explanation=explanation,
            visualization=_visualization_payload(
                focus_concept=nodes_by_id[focus.id],
                nodes=nodes,
                edges=edges,
                evidence=citations,
                trust_label=trust_label,
            ),
            limitations=limitations,
            official_graph_available=trust_label == TrustLabel.OFFICIAL,
        )

    def _search_hits(self, normalized_query: str, *, mode: OntologyGraphMode) -> list[_SearchHit]:
        if not normalized_query:
            return []
        hits: list[_SearchHit] = []
        for concept in self.store.list_concepts():
            if not self._concept_allowed(concept, mode=mode):
                continue
            hit = _score_concept(concept, normalized_query)
            if hit is not None:
                hits.append(hit)
        return sorted(hits, key=lambda item: (-item.score, item.concept.name.casefold(), item.concept.id))

    def _find_focus_concept(self, normalized_query: str, *, mode: OntologyGraphMode) -> Concept | None:
        hits = self._search_hits(normalized_query, mode=mode)
        if not hits:
            return None
        return hits[0].concept

    def _concept_allowed(self, concept: Concept, *, mode: OntologyGraphMode) -> bool:
        if mode == OntologyGraphMode.OFFICIAL:
            return concept.status == ConceptStatus.OFFICIAL
        if mode == OntologyGraphMode.CANDIDATE:
            return concept.status in {ConceptStatus.CANDIDATE, ConceptStatus.REVIEWING}
        return concept.status not in TERMINAL_CONCEPT_STATUSES

    def _relation_allowed(
        self,
        relation: ConceptRelation,
        *,
        mode: OntologyGraphMode,
        concepts_by_id: dict[str, Concept],
    ) -> bool:
        if relation.status in TERMINAL_RELATION_STATUSES:
            return False
        source = concepts_by_id.get(relation.source_concept_id)
        target = concepts_by_id.get(relation.target_concept_id)
        if source is None or target is None:
            return False
        if mode == OntologyGraphMode.OFFICIAL:
            return (
                relation.status == RelationStatus.OFFICIAL
                and source.status == ConceptStatus.OFFICIAL
                and target.status == ConceptStatus.OFFICIAL
            )
        if mode == OntologyGraphMode.CANDIDATE:
            return relation.status in {RelationStatus.CANDIDATE, RelationStatus.REVIEWING}
        return True

    def _evidence_for_graph(
        self,
        *,
        focus_concept_id: str,
        node_concepts: list[Concept],
        edges: list[OntologyGraphEdge],
    ) -> tuple[list[_SupportedEvidence], int]:
        supported: list[_SupportedEvidence] = []
        invalid_count = 0
        for concept in node_concepts:
            relationship = "supports_focus_concept" if concept.id == focus_concept_id else "supports_neighbor_concept"
            loaded, invalid = self._load_eligible_evidence(
                concept.evidence_fragment_ids,
                supports=[
                    CitationSupportRef(
                        entity_type=CitationSupportType.CONCEPT,
                        entity_id=concept.id,
                        relationship=relationship,
                    )
                ],
            )
            supported.extend(loaded)
            invalid_count += invalid
        for edge in edges:
            loaded, invalid = self._load_eligible_evidence(
                edge.evidence_fragment_ids,
                supports=[
                    CitationSupportRef(
                        entity_type=CitationSupportType.CONCEPT_RELATION,
                        entity_id=edge.id,
                        relationship="supports_graph_edge",
                    )
                ],
            )
            supported.extend(loaded)
            invalid_count += invalid
            if edge.decision_record_id is not None:
                decision = decision_record_or_none(self.store, edge.decision_record_id)
                if decision is None:
                    invalid_count += 1
                    continue
                loaded, invalid = self._load_eligible_evidence(
                    decision.evidence_fragment_ids,
                    supports=[
                        CitationSupportRef(
                            entity_type=CitationSupportType.DECISION_RECORD,
                            entity_id=decision.id,
                            relationship="supports_graph_edge_decision_record",
                        ),
                        CitationSupportRef(
                            entity_type=CitationSupportType.CONCEPT_RELATION,
                            entity_id=edge.id,
                            relationship="supports_graph_edge",
                        ),
                    ],
                )
                supported.extend(loaded)
                invalid_count += invalid
        return dedupe_supported_evidence(supported), invalid_count

    def _load_eligible_evidence(
        self,
        evidence_ids: list[str],
        *,
        supports: list[CitationSupportRef],
    ) -> tuple[list[_SupportedEvidence], int]:
        fragments: list[_SupportedEvidence] = []
        invalid_count = 0
        for evidence_id in evidence_ids:
            try:
                fragment = self.store.get_evidence_fragment(evidence_id)
            except NotFoundError:
                invalid_count += 1
                continue
            if not self._evidence_is_serving_eligible(fragment):
                invalid_count += 1
                continue
            fragments.append(_SupportedEvidence(fragment=fragment, supports=tuple(supports)))
        return fragments, invalid_count

    def _support_summary_for_relation(self, relation: ConceptRelation) -> GraphEntitySupportSummary:
        evidence_ids = list(relation.evidence_fragment_ids)
        if relation.decision_record_id is not None:
            decision = decision_record_or_none(self.store, relation.decision_record_id)
            if decision is None:
                summary = self._support_summary_for_ids(evidence_ids)
                return summary.model_copy(update={"invalid_evidence_reference_count": summary.invalid_evidence_reference_count + 1})
            evidence_ids.extend(decision.evidence_fragment_ids)
        return self._support_summary_for_ids(evidence_ids)

    def _support_summary_for_ids(self, evidence_ids: list[str]) -> GraphEntitySupportSummary:
        unique_ids = _dedupe_preserve_order(evidence_ids)
        fragments: list[EvidenceFragment] = []
        invalid_count = 0
        for evidence_id in unique_ids:
            try:
                fragment = self.store.get_evidence_fragment(evidence_id)
            except NotFoundError:
                invalid_count += 1
                continue
            if not self._evidence_is_serving_eligible(fragment):
                invalid_count += 1
                continue
            fragments.append(fragment)
        return _entity_support_summary(fragments, invalid_count=invalid_count)

    def _evidence_is_serving_eligible(self, fragment: EvidenceFragment) -> bool:
        if fragment.trust_state not in CITABLE_TRUST_STATES:
            return False
        if self.production_mode and not is_from_servable_source(self.store, fragment, production_mode=self.production_mode):
            return False
        return not citation_validity_errors(self.store, fragment)

    def _candidate_summary_for_query(
        self,
        normalized_query: str,
        *,
        focus: Concept | None,
    ) -> OntologyCandidateGraphSummary:
        terms = _candidate_match_terms(normalized_query, focus=focus)
        concept_candidates = [
            candidate
            for candidate in self.store.list_concept_candidates()
            if _candidate_name_matches(candidate.normalized_name, terms)
        ]
        relation_candidates = [
            candidate
            for candidate in self.store.list_relation_candidates()
            if _candidate_name_matches(candidate.normalized_source_name, terms)
            or _candidate_name_matches(candidate.normalized_target_name, terms)
        ]
        summary = OntologyCandidateGraphSummary(
            pending_concept_candidate_count=_count_candidate_status(concept_candidates, OntologyCandidateStatus.PENDING),
            pending_relation_candidate_count=_count_candidate_status(relation_candidates, OntologyCandidateStatus.PENDING),
            approved_concept_candidate_count=_count_candidate_status(concept_candidates, OntologyCandidateStatus.APPROVED),
            approved_relation_candidate_count=_count_candidate_status(relation_candidates, OntologyCandidateStatus.APPROVED),
            rejected_concept_candidate_count=_count_candidate_status(concept_candidates, OntologyCandidateStatus.REJECTED),
            rejected_relation_candidate_count=_count_candidate_status(relation_candidates, OntologyCandidateStatus.REJECTED),
            merged_concept_candidate_count=_count_candidate_status(concept_candidates, OntologyCandidateStatus.MERGED),
            merged_relation_candidate_count=_count_candidate_status(relation_candidates, OntologyCandidateStatus.MERGED),
        )
        pending_total = summary.pending_concept_candidate_count + summary.pending_relation_candidate_count
        if pending_total:
            summary.has_pending_candidates = True
            summary.note = "Pending ontology candidates match this graph focus; they are excluded from the official graph until reviewed."
        elif concept_candidates or relation_candidates:
            summary.note = "Reviewed ontology candidates match this graph focus; candidate outcomes are summarized separately from graph objects."
        return summary


def _score_concept(concept: Concept, normalized_query: str) -> _SearchHit | None:
    normalized_name = normalize_concept_term(concept.name)
    if normalized_name == normalized_query:
        return _SearchHit(concept, OntologySearchMatchType.NAME, concept.name, 1.0)
    for alias in concept.aliases:
        normalized_alias = normalize_concept_term(alias)
        if normalized_alias == normalized_query:
            return _SearchHit(concept, OntologySearchMatchType.ALIAS, alias, 0.96)
    if normalized_query in normalized_name or normalized_name in normalized_query:
        return _SearchHit(concept, OntologySearchMatchType.PARTIAL, concept.name, 0.82)
    for alias in concept.aliases:
        normalized_alias = normalize_concept_term(alias)
        if normalized_query in normalized_alias or normalized_alias in normalized_query:
            return _SearchHit(concept, OntologySearchMatchType.PARTIAL, alias, 0.78)
    query_tokens = set(normalized_query.split())
    name_tokens = set(normalized_name.split())
    alias_tokens = set(token for alias in concept.aliases for token in normalize_concept_term(alias).split())
    overlap = query_tokens & (name_tokens | alias_tokens)
    if overlap and len(overlap) == len(query_tokens):
        matched_value = concept.name if overlap & name_tokens else next(iter(concept.aliases), concept.name)
        return _SearchHit(concept, OntologySearchMatchType.PARTIAL, matched_value, 0.6)
    return None


def _node_from_concept(
    concept: Concept,
    *,
    is_focus: bool,
    support_summary: GraphEntitySupportSummary,
) -> OntologyGraphNode:
    status_text = "official" if concept.status == ConceptStatus.OFFICIAL else f"{concept.status}"
    role = "focus" if is_focus else "neighbor"
    return OntologyGraphNode(
        id=concept.id,
        name=concept.name,
        aliases=concept.aliases,
        short_definition=concept.short_definition,
        status=concept.status,
        owner=concept.owner,
        is_focus=is_focus,
        evidence_fragment_ids=concept.evidence_fragment_ids,
        decision_record_ids=concept.decision_record_ids,
        review_provenance=_review_provenance_from_concept(concept),
        support_summary=support_summary,
        explanation=(
            f"{concept.name} is the {role} Concept in this graph. "
            f"It is {status_text} and has {support_summary.reviewed_evidence_count} reviewed serving citation(s)."
        ),
    )


def _edge_from_relation(
    relation: ConceptRelation,
    *,
    concepts_by_id: dict[str, Concept],
    focus_concept_id: str,
    support_summary: GraphEntitySupportSummary,
) -> OntologyGraphEdge:
    source_concept = concepts_by_id.get(relation.source_concept_id)
    target_concept = concepts_by_id.get(relation.target_concept_id)
    source_name = source_concept.name if source_concept is not None else None
    target_name = target_concept.name if target_concept is not None else None
    if relation.source_concept_id == focus_concept_id:
        focus_direction = "outgoing"
    elif relation.target_concept_id == focus_concept_id:
        focus_direction = "incoming"
    else:
        focus_direction = "connected"
    source_display = source_name or relation.source_concept_id
    target_display = target_name or relation.target_concept_id
    relation_display = str(relation.relation_type).replace("_", " ")
    return OntologyGraphEdge(
        id=relation.id,
        source_concept_id=relation.source_concept_id,
        target_concept_id=relation.target_concept_id,
        source_concept_name=source_name,
        target_concept_name=target_name,
        focus_direction=focus_direction,
        relation_type=relation.relation_type,
        status=relation.status,
        evidence_fragment_ids=relation.evidence_fragment_ids,
        decision_record_id=relation.decision_record_id,
        review_provenance=_review_provenance_from_relation(relation),
        support_summary=support_summary,
        explanation=(
            f"{source_display} {relation_display} {target_display}. "
            f"This relation is {relation.status} and has {support_summary.reviewed_evidence_count} reviewed serving citation(s)."
        ),
    )


def _review_provenance_from_concept(concept: Concept) -> ReviewProvenance:
    return ReviewProvenance(
        created_by=concept.created_by,
        officialized_by=concept.officialized_by,
        reviewed_by=concept.officialized_by,
        created_at=concept.created_at,
        updated_at=concept.updated_at,
        last_reviewed_at=concept.last_reviewed_at,
        status=str(concept.status),
    )


def _review_provenance_from_relation(relation: ConceptRelation) -> ReviewProvenance:
    return ReviewProvenance(
        created_by=relation.created_by,
        officialized_by=relation.officialized_by,
        reviewed_by=relation.officialized_by,
        created_at=relation.created_at,
        updated_at=relation.updated_at,
        last_reviewed_at=relation.last_reviewed_at,
        status=str(relation.status),
    )



def _citation_from(item: _SupportedEvidence) -> EvidenceCitation:
    return EvidenceCitation(
        evidence_fragment_id=item.fragment.id,
        artifact_id=item.fragment.artifact_id,
        text=item.fragment.text,
        data_source_id=item.fragment.provenance.data_source_id,
        source_type=item.fragment.provenance.source_type,
        source_external_id=item.fragment.provenance.source_external_id,
        source_url=item.fragment.provenance.source_url,
        artifact_title=item.fragment.provenance.artifact_title,
        captured_at=item.fragment.provenance.captured_at,
        source_updated_at=item.fragment.provenance.source_updated_at,
        freshness_state=item.fragment.freshness_state,
        trust_state=item.fragment.trust_state,
        reviewed_by=item.fragment.reviewed_by,
        reviewed_at=item.fragment.reviewed_at,
        supports=list(item.supports),
        is_valid=True,
        validity_errors=[],
    )


def _summarize_freshness(evidence: list[EvidenceFragment]) -> FreshnessSummary:
    if not evidence:
        return FreshnessSummary(state=FreshnessState.UNKNOWN)
    states = {fragment.freshness_state for fragment in evidence}
    stale_count = sum(1 for fragment in evidence if fragment.freshness_state == FreshnessState.STALE)
    unknown_count = sum(1 for fragment in evidence if fragment.freshness_state == FreshnessState.UNKNOWN)
    if FreshnessState.STALE in states:
        return FreshnessSummary(
            state=FreshnessState.STALE,
            stale_evidence_count=stale_count,
            unknown_evidence_count=unknown_count,
        )
    if FreshnessState.UNKNOWN in states:
        return FreshnessSummary(
            state=FreshnessState.UNKNOWN,
            stale_evidence_count=stale_count,
            unknown_evidence_count=unknown_count,
        )
    if len(states) > 1:
        return FreshnessSummary(
            state=FreshnessState.MIXED,
            stale_evidence_count=stale_count,
            unknown_evidence_count=unknown_count,
        )
    return FreshnessSummary(
        state=next(iter(states)),
        stale_evidence_count=stale_count,
        unknown_evidence_count=unknown_count,
    )


def _entity_support_summary(
    evidence: list[EvidenceFragment],
    *,
    invalid_count: int = 0,
) -> GraphEntitySupportSummary:
    return GraphEntitySupportSummary(
        evidence_count=len(evidence),
        reviewed_evidence_count=sum(1 for fragment in evidence if fragment.trust_state == TrustState.REVIEWED),
        unreviewed_evidence_count=sum(1 for fragment in evidence if fragment.trust_state == TrustState.UNREVIEWED),
        conflicted_evidence_count=sum(1 for fragment in evidence if fragment.trust_state == TrustState.CONFLICTED),
        stale_evidence_count=sum(1 for fragment in evidence if fragment.freshness_state == FreshnessState.STALE),
        unknown_freshness_evidence_count=sum(
            1 for fragment in evidence if fragment.freshness_state == FreshnessState.UNKNOWN
        ),
        invalid_evidence_reference_count=invalid_count,
    )


def _support_summary_for_graph(
    *,
    nodes: list[OntologyGraphNode],
    edges: list[OntologyGraphEdge],
    evidence: list[EvidenceFragment],
    invalid_evidence_count: int,
    invalid_relation_count: int,
) -> OntologyGraphSupportSummary:
    entity_summary = _entity_support_summary(evidence, invalid_count=invalid_evidence_count)
    return OntologyGraphSupportSummary(
        node_count=len(nodes),
        edge_count=len(edges),
        evidence_count=entity_summary.evidence_count,
        reviewed_evidence_count=entity_summary.reviewed_evidence_count,
        unreviewed_evidence_count=entity_summary.unreviewed_evidence_count,
        conflicted_evidence_count=entity_summary.conflicted_evidence_count,
        stale_evidence_count=entity_summary.stale_evidence_count,
        unknown_freshness_evidence_count=entity_summary.unknown_freshness_evidence_count,
        official_node_count=sum(1 for node in nodes if node.status == ConceptStatus.OFFICIAL),
        non_official_node_count=sum(1 for node in nodes if node.status != ConceptStatus.OFFICIAL),
        official_edge_count=sum(1 for edge in edges if edge.status == RelationStatus.OFFICIAL),
        non_official_edge_count=sum(1 for edge in edges if edge.status != RelationStatus.OFFICIAL),
        invalid_evidence_reference_count=invalid_evidence_count,
        invalid_relation_count=invalid_relation_count,
    )


def _trust_label_for_graph(
    *,
    focus: Concept,
    nodes: list[OntologyGraphNode],
    edges: list[OntologyGraphEdge],
    evidence: list[EvidenceFragment],
    freshness: FreshnessSummary,
    mode: OntologyGraphMode,
) -> TrustLabel:
    if focus.status == ConceptStatus.CONFLICTED or any(fragment.trust_state == TrustState.CONFLICTED for fragment in evidence):
        return TrustLabel.CONFLICTED
    if not evidence:
        return TrustLabel.UNSUPPORTED
    if freshness.state == FreshnessState.STALE:
        return TrustLabel.STALE
    if freshness.state in {FreshnessState.UNKNOWN, FreshnessState.MIXED}:
        return TrustLabel.PARTIALLY_SUPPORTED
    reviewed_count = sum(1 for fragment in evidence if fragment.trust_state == TrustState.REVIEWED)
    if (
        mode == OntologyGraphMode.OFFICIAL
        and focus.status == ConceptStatus.OFFICIAL
        and all(node.status == ConceptStatus.OFFICIAL for node in nodes)
        and all(edge.status == RelationStatus.OFFICIAL for edge in edges)
        and reviewed_count == len(evidence)
    ):
        return TrustLabel.OFFICIAL
    if reviewed_count:
        return TrustLabel.EVIDENCE_SUPPORTED
    return TrustLabel.PARTIALLY_SUPPORTED


def _limitations_for_graph(
    *,
    focus: Concept,
    edges: list[OntologyGraphEdge],
    evidence: list[EvidenceFragment],
    trust_label: TrustLabel,
    mode: OntologyGraphMode,
    invalid_relation_count: int,
    invalid_evidence_count: int,
    production_mode: bool,
    candidate_summary: OntologyCandidateGraphSummary,
) -> list[str]:
    limitations: list[str] = []
    if mode == OntologyGraphMode.OFFICIAL:
        limitations.append("Official mode returns only official Concepts and official ConceptRelations.")
    elif mode == OntologyGraphMode.CANDIDATE:
        limitations.append(
            "Candidate mode is exploratory; non-official graph objects are not the Single Source of Truth."
        )
    else:
        limitations.append("Mixed mode may include non-official graph objects and must not be treated as the SSOT.")
    if focus.status != ConceptStatus.OFFICIAL:
        limitations.append("Focus Concept is not official.")
    if not edges:
        limitations.append("No direct ConceptRelations were available at depth 1 for this graph mode.")
    if not evidence:
        limitations.append("No serving-eligible supporting EvidenceFragments were available for this graph.")
    if invalid_relation_count:
        limitations.append(f"{invalid_relation_count} relation(s) were ignored because their connected Concept was missing.")
    if invalid_evidence_count:
        limitations.append(
            f"{invalid_evidence_count} evidence reference(s) were ignored because they are missing, rejected, invalid, or ineligible."
        )
    if any(fragment.trust_state == TrustState.UNREVIEWED for fragment in evidence):
        limitations.append("At least one supporting EvidenceFragment has not been reviewed.")
    if any(fragment.trust_state == TrustState.CONFLICTED for fragment in evidence):
        limitations.append("At least one supporting EvidenceFragment is conflicted.")
    if trust_label == TrustLabel.STALE:
        limitations.append("At least one supporting EvidenceFragment is stale.")
    if trust_label == TrustLabel.PARTIALLY_SUPPORTED:
        limitations.append("Freshness is mixed or unknown, or graph support is incomplete.")
    if candidate_summary.has_pending_candidates and mode == OntologyGraphMode.OFFICIAL:
        limitations.append("Pending ontology candidates exist for this focus but are excluded from the official graph.")
    if production_mode:
        limitations.append(
            "Production mode excludes non-production, rejected, invalid, or never-synced source evidence; degraded historical evidence is served with limitations."
        )
    return limitations


def _explanation_for_graph(
    *,
    focus: Concept,
    nodes: list[OntologyGraphNode],
    edges: list[OntologyGraphEdge],
    evidence: list[EvidenceFragment],
    trust_label: TrustLabel,
    freshness: FreshnessSummary,
    support_summary: OntologyGraphSupportSummary,
    candidate_summary: OntologyCandidateGraphSummary,
    mode: OntologyGraphMode,
    depth: int,
) -> OntologyGraphExplanation:
    summary = (
        f"{focus.name} graph served {len(nodes)} node(s), {len(edges)} direct relation(s), "
        f"and {len(evidence)} serving citation(s) at depth {depth}."
    )
    if trust_label == TrustLabel.OFFICIAL:
        ssot_status = "official_ssot"
        trust_reason = "The focus Concept, visible neighbor Concepts, visible Relations, and serving citations are reviewed and official."
    elif trust_label == TrustLabel.UNSUPPORTED:
        ssot_status = "not_ssot"
        trust_reason = "The graph has no serving-eligible evidence, so it cannot be treated as official truth."
    elif mode == OntologyGraphMode.OFFICIAL:
        ssot_status = "official_mode_with_limitations"
        trust_reason = f"The graph is in official mode but received trust label '{trust_label}' because support is incomplete, stale, conflicted, or only partially reviewed."
    else:
        ssot_status = "exploratory"
        trust_reason = f"The graph is in {mode} mode and must be reviewed before it is treated as the Single Source of Truth."
    graph_scope = "Depth 1 includes the focus Concept and directly connected Concepts only. Recursive expansion is intentionally out of scope for v1.6.0."
    evidence_policy = (
        "Citations include serving-eligible EvidenceFragments for the focus Concept, visible neighbor Concepts, "
        "visible ConceptRelations, and linked DecisionRecords. Rejected, invalid, or ineligible evidence is excluded."
    )
    if candidate_summary.has_pending_candidates:
        candidate_boundary = "Pending ConceptCandidates or RelationCandidates exist, but they are summarized separately and are not official graph objects."
    elif mode == OntologyGraphMode.OFFICIAL:
        candidate_boundary = "Candidate objects are excluded from official graph mode."
    else:
        candidate_boundary = "Non-official graph objects may appear in candidate/mixed modes, but extraction candidates still require review before promotion."
    review_summary = (
        f"{support_summary.official_node_count}/{support_summary.node_count} node(s) official; "
        f"{support_summary.official_edge_count}/{support_summary.edge_count} edge(s) official; "
        f"{support_summary.reviewed_evidence_count}/{support_summary.evidence_count} serving citation(s) reviewed; "
        f"freshness={freshness.state}."
    )
    recommended_next_actions: list[str] = []
    if candidate_summary.has_pending_candidates:
        recommended_next_actions.append("Review pending ontology candidates before expecting them in the official graph.")
    if support_summary.invalid_evidence_reference_count:
        recommended_next_actions.append("Repair or remove invalid evidence references from graph objects.")
    if trust_label in {TrustLabel.UNSUPPORTED, TrustLabel.PARTIALLY_SUPPORTED}:
        recommended_next_actions.append("Review supporting EvidenceFragments and officialize eligible Concepts/Relations.")
    if trust_label == TrustLabel.STALE:
        recommended_next_actions.append("Refresh stale source evidence before relying on this graph as current truth.")
    if not recommended_next_actions:
        recommended_next_actions.append("Use the cited evidence and review provenance to audit the graph before downstream reuse.")
    return OntologyGraphExplanation(
        summary=summary,
        ssot_status=ssot_status,
        trust_reason=trust_reason,
        graph_scope=graph_scope,
        evidence_policy=evidence_policy,
        candidate_boundary=candidate_boundary,
        review_summary=review_summary,
        recommended_next_actions=recommended_next_actions,
    )


def _explanation_for_missing_graph(
    *,
    concept_query: str,
    mode: OntologyGraphMode,
    depth: int,
    candidate_summary: OntologyCandidateGraphSummary,
) -> OntologyGraphExplanation:
    actions = ["Create or review a Concept before requesting an official ontology graph."]
    if candidate_summary.has_pending_candidates:
        actions.insert(0, "Review the matching pending ontology candidates; they may become graph objects after approval.")
    return OntologyGraphExplanation(
        summary=f"No {mode} ontology graph is available for '{concept_query}' at depth {depth}.",
        ssot_status="not_ssot",
        trust_reason="No matching Concept exists in the requested graph mode.",
        graph_scope="Depth 1 graph serving requires a focus Concept before direct relations can be returned.",
        evidence_policy="No citations are returned because no graph object matched the request.",
        candidate_boundary=(
            "Matching candidates are summarized separately and are not official truth."
            if candidate_summary.has_pending_candidates
            else "No matching pending candidates were found for this query."
        ),
        review_summary="0 official node(s), 0 official edge(s), and 0 serving citation(s).",
        recommended_next_actions=actions,
    )


def _candidate_match_terms(normalized_query: str, *, focus: Concept | None) -> set[str]:
    terms = {normalized_query} if normalized_query else set()
    if focus is not None:
        terms.add(normalize_concept_term(focus.name))
        terms.update(normalize_concept_term(alias) for alias in focus.aliases)
    return {term for term in terms if term}


def _candidate_name_matches(normalized_name: str, terms: set[str]) -> bool:
    if not normalized_name or not terms:
        return False
    return any(normalized_name == term or normalized_name in term or term in normalized_name for term in terms)


def _count_candidate_status(candidates: list[Any], status: OntologyCandidateStatus) -> int:
    return sum(1 for candidate in candidates if candidate.status == status)


def _visualization_payload(
    *,
    focus_concept: OntologyGraphNode | None,
    nodes: list[OntologyGraphNode],
    edges: list[OntologyGraphEdge],
    evidence: list[EvidenceCitation],
    trust_label: TrustLabel,
) -> OntologyGraphVisualization:
    citations_by_entity: dict[str, list[str]] = {}
    for citation in evidence:
        for support in citation.supports:
            citations_by_entity.setdefault(support.entity_id, []).append(citation.evidence_fragment_id)
    visual_nodes = [
        OntologyGraphVisualizationNode(
            id=node.id,
            label=node.name,
            category=node.status,
            display_state=_node_display_state(node),
            is_focus=node.is_focus,
            group="focus" if node.is_focus else "direct_neighbor",
            citation_panel={
                "evidenceFragmentIds": citations_by_entity.get(node.id, []),
                "reviewedEvidenceCount": node.support_summary.reviewed_evidence_count,
                "staleEvidenceCount": node.support_summary.stale_evidence_count,
            },
            review_provenance_panel={} if node.review_provenance is None else node.review_provenance.model_dump(mode="json", by_alias=True),
        )
        for node in nodes
    ]
    visual_edges = [
        OntologyGraphVisualizationEdge(
            id=edge.id,
            source_id=edge.source_concept_id,
            target_id=edge.target_concept_id,
            label=str(edge.relation_type).replace("_", " "),
            direction=edge.focus_direction or "connected",
            display_state=_edge_display_state(edge),
            citation_panel={
                "evidenceFragmentIds": citations_by_entity.get(edge.id, []),
                "reviewedEvidenceCount": edge.support_summary.reviewed_evidence_count,
                "staleEvidenceCount": edge.support_summary.stale_evidence_count,
            },
            review_provenance_panel={} if edge.review_provenance is None else edge.review_provenance.model_dump(mode="json", by_alias=True),
        )
        for edge in edges
    ]
    return OntologyGraphVisualization(
        focus_concept_id=None if focus_concept is None else focus_concept.id,
        empty_state=None if focus_concept is not None else "No graph is available for this concept and mode.",
        nodes=visual_nodes,
        edges=visual_edges,
        state_legend={
            "official": "Reviewed official graph object.",
            "candidate": "Non-official graph object; do not treat as SSOT.",
            "stale": "Supported by stale evidence.",
            "conflicted": "Has conflicted evidence or status.",
            "unsupported": "Missing serving-eligible evidence.",
        },
        layout_hints={
            "depth": 1,
            "focusPlacement": "center",
            "edgeDirection": "relative_to_focus",
            "trustLabel": str(trust_label),
        },
        citation_panel={
            "citationCount": len(evidence),
            "evidenceFragmentIds": [citation.evidence_fragment_id for citation in evidence],
        },
        review_provenance_panel={
            "focusConceptId": None if focus_concept is None else focus_concept.id,
            "graphObjectsRequireReview": True,
        },
    )


def _node_display_state(node: OntologyGraphNode) -> str:
    if node.status == ConceptStatus.CONFLICTED:
        return "conflicted"
    if node.support_summary.stale_evidence_count:
        return "stale"
    if not node.support_summary.evidence_count:
        return "unsupported"
    if node.status == ConceptStatus.OFFICIAL:
        return "official"
    return "candidate"


def _edge_display_state(edge: OntologyGraphEdge) -> str:
    if edge.support_summary.conflicted_evidence_count:
        return "conflicted"
    if edge.support_summary.stale_evidence_count:
        return "stale"
    if not edge.support_summary.evidence_count:
        return "unsupported"
    if edge.status == RelationStatus.OFFICIAL:
        return "official"
    return "candidate"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
