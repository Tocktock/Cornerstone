from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cornerstone.schemas import (
    CitationSupportRef,
    CitationSupportType,
    Concept,
    ConceptRef,
    ConceptRelation,
    ConceptRelationRef,
    ConceptStatus,
    DecisionRecord,
    DecisionRecordRef,
    EvidenceCitation,
    EvidenceFragment,
    FreshnessState,
    FreshnessSummary,
    GroundedContextResponse,
    RelationStatus,
    TrustLabel,
    TrustState,
)
from cornerstone.domain.ontology import CITABLE_TRUST_STATES
from cornerstone.services.evidence_support import (
    citation_validity_errors,
    decision_record_or_none,
    dedupe_supported_evidence,
    is_from_servable_source,
    source_limitations_for,
)
from cornerstone.store import InMemoryStore, NotFoundError

_SEARCH_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "how",
    "is",
    "of",
    "or",
    "the",
    "to",
    "what",
    "where",
    "why",
}


@dataclass(frozen=True)
class _SupportedEvidence:
    fragment: EvidenceFragment
    supports: tuple[CitationSupportRef, ...]


class GroundedContextService:
    def __init__(self, store: Any, *, production_mode: bool = True) -> None:
        self.store = store
        self.production_mode = production_mode

    def query(self, query: str) -> GroundedContextResponse:
        normalized_query = query.lower().strip()
        concepts = self.store.list_concepts()
        concept = _find_best_concept_match(concepts, normalized_query)
        if concept is None:
            return self._evidence_only_response(query, normalized_query)

        concept_support = self._eligible_evidence_for(
            concept.evidence_fragment_ids,
            supports=[
                CitationSupportRef(
                    entity_type=CitationSupportType.CONCEPT,
                    entity_id=concept.id,
                    relationship="supports_concept_definition",
                )
            ],
        )
        decisions = self._decision_records_for(concept)
        decision_support = self._eligible_evidence_for_decisions(decisions)
        relations = self._relations_for(concept)
        relation_support = self._eligible_evidence_for_relations(relations)

        combined_support = dedupe_supported_evidence(
            [*concept_support, *decision_support, *relation_support]
        )
        citations = [_citation_from(item) for item in combined_support]
        freshness = _summarize_freshness([item.fragment for item in combined_support])
        trust_label = _trust_label_for(concept, [item.fragment for item in combined_support], freshness)
        support_fragments = [item.fragment for item in combined_support]
        limitations = _limitations_for_concept(
            concept,
            support_fragments,
            trust_label,
            self.production_mode,
            invalid_reference_count=_invalid_reference_count_for_concept(concept, self.store),
        )
        limitations.extend(source_limitations_for(self.store, support_fragments))

        return GroundedContextResponse(
            query=query,
            answer=_answer_for_concept(concept, trust_label),
            trust_label=trust_label,
            concepts=[ConceptRef(id=concept.id, name=concept.name, status=concept.status)],
            relations=[
                ConceptRelationRef(
                    id=item.id,
                    source_concept_id=item.source_concept_id,
                    target_concept_id=item.target_concept_id,
                    relation_type=item.relation_type,
                    status=item.status,
                )
                for item in relations
            ],
            decisions=[DecisionRecordRef(id=item.id, title=item.title) for item in decisions],
            evidence=citations,
            freshness=freshness,
            limitations=limitations,
            official_answer_available=trust_label == TrustLabel.OFFICIAL,
        )

    def _evidence_only_response(
        self, query: str, normalized_query: str
    ) -> GroundedContextResponse:
        matched_evidence = self._eligible_evidence_matching_query(normalized_query)
        freshness = _summarize_freshness([item.fragment for item in matched_evidence])
        trust_label = _trust_label_for_evidence_only(
            [item.fragment for item in matched_evidence], freshness
        )
        citations = [_citation_from(item) for item in matched_evidence]
        support_fragments = [item.fragment for item in matched_evidence]
        limitations = _limitations_for_evidence_only(
            support_fragments, trust_label, self.production_mode
        )
        limitations.extend(source_limitations_for(self.store, support_fragments))
        answer = _answer_for_evidence_only(support_fragments, trust_label)
        return GroundedContextResponse(
            query=query,
            answer=answer,
            trust_label=trust_label,
            evidence=citations,
            freshness=freshness,
            limitations=limitations,
            official_answer_available=False,
        )

    def _eligible_evidence_for(
        self, evidence_ids: list[str], *, supports: list[CitationSupportRef]
    ) -> list[_SupportedEvidence]:
        return self._load_eligible_evidence(
            evidence_ids,
            supports=supports,
        )

    def _eligible_evidence_for_decisions(
        self, decision_records: list[DecisionRecord]
    ) -> list[_SupportedEvidence]:
        evidence: list[_SupportedEvidence] = []
        for decision_record in decision_records:
            evidence.extend(
                self._load_eligible_evidence(
                    decision_record.evidence_fragment_ids,
                    supports=[
                        CitationSupportRef(
                            entity_type=CitationSupportType.DECISION_RECORD,
                            entity_id=decision_record.id,
                            relationship="supports_decision_record",
                        )
                    ],
                )
            )
        return evidence

    def _eligible_evidence_for_relations(
        self, relations: list[ConceptRelation]
    ) -> list[_SupportedEvidence]:
        evidence: list[_SupportedEvidence] = []
        for relation in relations:
            evidence.extend(
                self._load_eligible_evidence(
                    relation.evidence_fragment_ids,
                    supports=[
                        CitationSupportRef(
                            entity_type=CitationSupportType.CONCEPT_RELATION,
                            entity_id=relation.id,
                            relationship="supports_concept_relation",
                        )
                    ],
                )
            )
            if relation.decision_record_id is not None:
                decision = decision_record_or_none(self.store, relation.decision_record_id)
                if decision is not None:
                    evidence.extend(
                        self._load_eligible_evidence(
                            decision.evidence_fragment_ids,
                            supports=[
                                CitationSupportRef(
                                    entity_type=CitationSupportType.DECISION_RECORD,
                                    entity_id=decision.id,
                                    relationship="supports_relation_decision_record",
                                ),
                                CitationSupportRef(
                                    entity_type=CitationSupportType.CONCEPT_RELATION,
                                    entity_id=relation.id,
                                    relationship="supports_concept_relation",
                                ),
                            ],
                        )
                    )
        return evidence

    def _eligible_evidence_matching_query(
        self, normalized_query: str
    ) -> list[_SupportedEvidence]:
        tokens = _query_tokens(normalized_query)
        if not tokens:
            return []
        matched: list[_SupportedEvidence] = []
        for fragment in self.store.list_evidence_fragments():
            text = fragment.text.lower()
            artifact_title = fragment.provenance.artifact_title.lower()
            if any(token in text or token in artifact_title for token in tokens):
                loaded = self._load_eligible_evidence(
                    [fragment.id],
                    supports=[
                        CitationSupportRef(
                            entity_type=CitationSupportType.EVIDENCE_FRAGMENT,
                            entity_id=fragment.id,
                            relationship="related_evidence",
                        )
                    ],
                )
                matched.extend(loaded)
        return dedupe_supported_evidence(matched)

    def _load_eligible_evidence(
        self, evidence_ids: list[str], *, supports: list[CitationSupportRef]
    ) -> list[_SupportedEvidence]:
        fragments: list[_SupportedEvidence] = []
        for evidence_id in evidence_ids:
            try:
                fragment = self.store.get_evidence_fragment(evidence_id)
            except NotFoundError:
                continue
            if fragment.trust_state not in CITABLE_TRUST_STATES:
                continue
            if self.production_mode and not is_from_servable_source(self.store, fragment, production_mode=self.production_mode):
                continue
            validity_errors = citation_validity_errors(self.store, fragment)
            if validity_errors:
                continue
            fragments.append(_SupportedEvidence(fragment=fragment, supports=tuple(supports)))
        return fragments

    def _decision_records_for(self, concept: Concept) -> list[DecisionRecord]:
        records: list[DecisionRecord] = []
        for decision_record_id in concept.decision_record_ids:
            decision = decision_record_or_none(self.store, decision_record_id)
            if decision is not None:
                records.append(decision)
        return records

    def _relations_for(self, concept: Concept) -> list[ConceptRelation]:
        relations: list[ConceptRelation] = []
        for relation in self.store.list_concept_relations(concept_id=concept.id):
            if relation.status == RelationStatus.OFFICIAL:
                relations.append(relation)
        return relations


def _find_best_concept_match(concepts: list[Concept], normalized_query: str) -> Concept | None:
    for concept in concepts:
        concept_name = concept.name.lower()
        if concept_name in normalized_query or normalized_query in concept_name:
            return concept
    return None


def _query_tokens(normalized_query: str) -> list[str]:
    return [
        token
        for token in normalized_query.replace("?", " ").replace(",", " ").split()
        if len(token) >= 3 and token not in _SEARCH_STOP_WORDS
    ]



def _citation_from(item: _SupportedEvidence) -> EvidenceCitation:
    return EvidenceCitation(
        evidence_fragment_id=item.fragment.id,
        artifact_id=item.fragment.artifact_id,
        text=item.fragment.text,
        source_url=item.fragment.provenance.source_url,
        artifact_title=item.fragment.provenance.artifact_title,
        captured_at=item.fragment.provenance.captured_at,
        source_updated_at=item.fragment.provenance.source_updated_at,
        freshness_state=item.fragment.freshness_state,
        trust_state=item.fragment.trust_state,
        supports=list(item.supports),
        is_valid=True,
        validity_errors=[],
    )


def _summarize_freshness(evidence: list[EvidenceFragment]) -> FreshnessSummary:
    if not evidence:
        return FreshnessSummary(
            state=FreshnessState.UNKNOWN,
            stale_evidence_count=0,
            unknown_evidence_count=0,
        )

    states = {fragment.freshness_state for fragment in evidence}
    stale_count = sum(1 for fragment in evidence if fragment.freshness_state == FreshnessState.STALE)
    unknown_count = sum(
        1 for fragment in evidence if fragment.freshness_state == FreshnessState.UNKNOWN
    )
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


def _trust_label_for(
    concept: Concept, evidence: list[EvidenceFragment], freshness: FreshnessSummary
) -> TrustLabel:
    if concept.status == ConceptStatus.CONFLICTED or any(
        fragment.trust_state == TrustState.CONFLICTED for fragment in evidence
    ):
        return TrustLabel.CONFLICTED
    if not evidence:
        return TrustLabel.UNSUPPORTED
    if freshness.state == FreshnessState.STALE:
        return TrustLabel.STALE
    if freshness.state in {FreshnessState.UNKNOWN, FreshnessState.MIXED}:
        return TrustLabel.PARTIALLY_SUPPORTED
    reviewed_evidence = [fragment for fragment in evidence if fragment.trust_state == TrustState.REVIEWED]
    if concept.status == ConceptStatus.OFFICIAL and len(reviewed_evidence) == len(evidence):
        return TrustLabel.OFFICIAL
    if reviewed_evidence:
        return TrustLabel.EVIDENCE_SUPPORTED
    return TrustLabel.PARTIALLY_SUPPORTED


def _trust_label_for_evidence_only(
    evidence: list[EvidenceFragment], freshness: FreshnessSummary
) -> TrustLabel:
    if any(fragment.trust_state == TrustState.CONFLICTED for fragment in evidence):
        return TrustLabel.CONFLICTED
    if not evidence:
        return TrustLabel.UNSUPPORTED
    if freshness.state == FreshnessState.STALE:
        return TrustLabel.STALE
    if freshness.state in {FreshnessState.UNKNOWN, FreshnessState.MIXED}:
        return TrustLabel.PARTIALLY_SUPPORTED
    reviewed_evidence = [fragment for fragment in evidence if fragment.trust_state == TrustState.REVIEWED]
    if reviewed_evidence and len(reviewed_evidence) == len(evidence):
        return TrustLabel.EVIDENCE_SUPPORTED
    return TrustLabel.PARTIALLY_SUPPORTED


def _answer_for_concept(concept: Concept, trust_label: TrustLabel) -> str:
    if trust_label == TrustLabel.UNSUPPORTED:
        return (
            f"{concept.name} exists as a Concept, but Cornerstone cannot provide a grounded "
            "answer because no valid production-eligible supporting evidence is available."
        )
    if trust_label == TrustLabel.CONFLICTED:
        return f"{concept.name}: available context is conflicted and requires review."
    return f"{concept.name}: {concept.short_definition}"


def _answer_for_evidence_only(evidence: list[EvidenceFragment], trust_label: TrustLabel) -> str:
    if not evidence:
        return "There is no official or evidence-supported context for this request yet."
    if trust_label == TrustLabel.CONFLICTED:
        return "Related evidence exists, but it is conflicted and requires reviewer resolution."
    first = evidence[0].text
    return (
        "Related evidence exists, but no official Concept has been created yet. "
        f"Most relevant evidence: {first}"
    )


def _limitations_for_concept(
    concept: Concept,
    evidence: list[EvidenceFragment],
    trust_label: TrustLabel,
    production_mode: bool,
    *,
    invalid_reference_count: int,
) -> list[str]:
    limitations: list[str] = []
    if concept.status != ConceptStatus.OFFICIAL:
        limitations.append("Matching Concept is not official yet.")
    if not evidence:
        limitations.append("No serving-eligible supporting EvidenceFragments are attached.")
    if invalid_reference_count:
        limitations.append(
            f"{invalid_reference_count} attached evidence reference(s) were ignored because they are missing, rejected, invalid, or ineligible."
        )
    if any(fragment.trust_state == TrustState.UNREVIEWED for fragment in evidence):
        limitations.append("At least one supporting EvidenceFragment has not been reviewed.")
    if any(fragment.trust_state == TrustState.CONFLICTED for fragment in evidence):
        limitations.append("At least one supporting EvidenceFragment is conflicted.")
    if trust_label == TrustLabel.STALE:
        limitations.append("At least one supporting EvidenceFragment is stale.")
    if trust_label == TrustLabel.PARTIALLY_SUPPORTED:
        limitations.append("Freshness is mixed or unknown, or support is incomplete.")
    if trust_label == TrustLabel.CONFLICTED:
        if concept.status == ConceptStatus.CONFLICTED:
            limitations.append("The matching Concept is marked conflicted.")
        else:
            limitations.append("The matching Concept or one supporting EvidenceFragment is conflicted.")
    if production_mode:
        limitations.append("Production mode excludes non-production, rejected, invalid, or never-synced source evidence; degraded historical evidence is served with limitations.")
    return limitations


def _limitations_for_evidence_only(
    evidence: list[EvidenceFragment],
    trust_label: TrustLabel,
    production_mode: bool,
) -> list[str]:
    limitations: list[str] = []
    if not evidence:
        limitations.append("No matching Concept or EvidenceFragment was found.")
    else:
        limitations.append("No matching official Concept was found; response is based on related EvidenceFragments.")
    if any(fragment.trust_state == TrustState.UNREVIEWED for fragment in evidence):
        limitations.append("At least one related EvidenceFragment has not been reviewed.")
    if any(fragment.trust_state == TrustState.CONFLICTED for fragment in evidence):
        limitations.append("At least one related EvidenceFragment is conflicted.")
    if trust_label == TrustLabel.STALE:
        limitations.append("At least one related EvidenceFragment is stale.")
    if trust_label == TrustLabel.PARTIALLY_SUPPORTED:
        limitations.append("Freshness is mixed or unknown, or support is incomplete.")
    if production_mode:
        limitations.append("Production mode excludes non-production, rejected, invalid, or never-synced source evidence; degraded historical evidence is served with limitations.")
    return limitations


def _invalid_reference_count_for_concept(concept: Concept, store: InMemoryStore) -> int:
    count = 0
    for evidence_id in concept.evidence_fragment_ids:
        try:
            fragment = store.get_evidence_fragment(evidence_id)
        except NotFoundError:
            count += 1
            continue
        if fragment.trust_state == TrustState.REJECTED:
            count += 1
    for decision_record_id in concept.decision_record_ids:
        try:
            decision = store.get_decision_record(decision_record_id)
        except NotFoundError:
            count += 1
            continue
        for evidence_id in decision.evidence_fragment_ids:
            try:
                fragment = store.get_evidence_fragment(evidence_id)
            except NotFoundError:
                count += 1
                continue
            if fragment.trust_state == TrustState.REJECTED:
                count += 1
    return count
