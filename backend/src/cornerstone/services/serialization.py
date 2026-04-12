from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from cornerstone.config import Settings
from cornerstone.domain.enums import (
    AnswerStatus,
    ConsumerScope,
    FreshnessState,
    NoMatchReason,
    RequestIntent,
    ResourceKind,
    ResponseKind,
    SupportItemKind,
    SupportVisibility,
    VerificationState,
    VisibilityClass,
)
from cornerstone.domain.models import (
    Artifact,
    Concept,
    ConceptRelation,
    ContextSpace,
    DecisionRecord,
    SourceConnection,
    SupportItem,
    VerificationPolicy,
)
from cornerstone.domain.schemas import (
    AnswerPayload,
    AnswerSection,
    ConceptPayload,
    ContextSpaceRef,
    ContractEnvelope,
    DecisionPayload,
    GraphEdgePayload,
    GraphSlicePayload,
    NoMatchPayload,
    ProvenancePayload,
    ProvenanceSummary,
    RelationPayload,
    ResourceRef,
    SearchResultItem,
    SearchResultsPayload,
    SourceSummary,
    SuggestedFollowUp,
    SupportItemSummary,
)


def context_space_ref(context_space: ContextSpace) -> ContextSpaceRef:
    return ContextSpaceRef(
        context_space_id=context_space.id,
        context_space_kind=context_space.kind,
        context_space_name=context_space.name,
    )


def resource_ref(
    resource: Concept | ConceptRelation | DecisionRecord | Artifact | SupportItem,
) -> ResourceRef:
    if isinstance(resource, Concept):
        return ResourceRef(
            resource_kind=ResourceKind.CONCEPT,
            resource_id=resource.id,
            resource_label=resource.canonical_name,
        )
    if isinstance(resource, ConceptRelation):
        return ResourceRef(
            resource_kind=ResourceKind.RELATION,
            resource_id=resource.id,
            resource_label=(
                f"{resource.subject_concept.canonical_name} {resource.predicate} "
                f"{resource.object_concept.canonical_name}"
            ),
        )
    if isinstance(resource, DecisionRecord):
        return ResourceRef(
            resource_kind=ResourceKind.DECISION,
            resource_id=resource.id,
            resource_label=resource.title,
        )
    if isinstance(resource, Artifact):
        return ResourceRef(
            resource_kind=ResourceKind.ARTIFACT,
            resource_id=resource.id,
            resource_label=resource.title,
        )
    return ResourceRef(
        resource_kind=ResourceKind.SUPPORT_ITEM,
        resource_id=resource.id,
        resource_label=resource.source_label,
    )


def support_items_for_resource(
    resource: Concept | ConceptRelation | DecisionRecord,
) -> list[SupportItem]:
    if isinstance(resource, Concept):
        return [link.support_item for link in resource.support_links]
    if isinstance(resource, ConceptRelation):
        return [link.support_item for link in resource.support_links]
    return [link.support_item for link in resource.support_links]


def _visible_to_consumer(support_item: SupportItem, consumer_scope: ConsumerScope) -> bool:
    if consumer_scope is ConsumerScope.MEMBER:
        return support_item.visibility_class is VisibilityClass.MEMBER_VISIBLE
    return True


def visible_support_items(
    support_items: Iterable[SupportItem],
    consumer_scope: ConsumerScope,
) -> list[SupportItem]:
    return [
        support_item
        for support_item in support_items
        if _visible_to_consumer(support_item, consumer_scope)
    ]


def _source_locator_for_support(
    support_item: SupportItem, consumer_scope: ConsumerScope
) -> str | None:
    if not _visible_to_consumer(support_item, consumer_scope):
        return None
    if support_item.support_item_kind is SupportItemKind.PROMOTED_SUPPORT:
        return f"workspace-promotion:{support_item.id}"
    return support_item.source_locator


def support_item_summary(
    support_item: SupportItem,
    consumer_scope: ConsumerScope,
) -> SupportItemSummary:
    return SupportItemSummary(
        support_item_id=support_item.id,
        support_item_kind=support_item.support_item_kind,
        visibility_class=support_item.visibility_class,
        source_label=support_item.source_label,
        excerpt_or_summary=support_item.shared_payload or support_item.excerpt_or_summary or None,
        origin_disclosure_level=support_item.origin_disclosure_level,
        source_locator=_source_locator_for_support(support_item, consumer_scope),
    )


def aggregate_freshness_state(support_items: Iterable[SupportItem]) -> FreshnessState:
    states = {support_item.freshness_state for support_item in support_items}
    if not states:
        return FreshnessState.UNKNOWN
    if FreshnessState.DRIFT_DETECTED in states:
        return FreshnessState.DRIFT_DETECTED
    if FreshnessState.STALE in states:
        return FreshnessState.STALE
    if FreshnessState.MONITORING in states:
        return FreshnessState.MONITORING
    if states == {FreshnessState.CURRENT}:
        return FreshnessState.CURRENT
    return FreshnessState.UNKNOWN


def compute_support_visibility(
    support_items: list[SupportItem],
    visible_items: list[SupportItem],
    policy: VerificationPolicy | None,
) -> SupportVisibility:
    if not support_items:
        return SupportVisibility.INSUFFICIENT_SUPPORT
    minimum_total = policy.minimum_support_items if policy else 1
    minimum_visible = policy.minimum_visible_support_items_for_source_backed if policy else 1
    if len(visible_items) >= minimum_visible:
        return SupportVisibility.SOURCE_BACKED
    if len(support_items) >= minimum_total and len(visible_items) < len(support_items):
        return SupportVisibility.RESTRICTED_SUPPORT
    return SupportVisibility.INSUFFICIENT_SUPPORT


def provenance_summary(
    support_items: list[SupportItem],
    consumer_scope: ConsumerScope,
    verification_state: VerificationState | None,
    policy: VerificationPolicy | None,
) -> ProvenanceSummary:
    visible_items = visible_support_items(support_items, consumer_scope)
    return ProvenanceSummary(
        support_item_count=len(support_items),
        visible_support_item_count=len(visible_items),
        restricted_support_present=len(visible_items) < len(support_items),
        freshness_state=aggregate_freshness_state(support_items),
        verification_state=verification_state,
        promotion_lineage_present=any(
            support_item.support_item_kind is SupportItemKind.PROMOTED_SUPPORT
            for support_item in support_items
        ),
    )


def build_envelope(
    *,
    settings: Settings,
    context_space: ContextSpace,
    consumer_scope: ConsumerScope,
    response_kind: ResponseKind,
    request_intent: RequestIntent,
    payload,
    related_refs: list[ResourceRef] | None = None,
    warnings: list[str] | None = None,
) -> ContractEnvelope:
    return ContractEnvelope(
        contract_version=settings.contract_version,
        response_kind=response_kind,
        request_intent=request_intent,
        context_space_ref=context_space_ref(context_space),
        consumer_scope=consumer_scope,
        payload=payload,
        related_refs=related_refs or [],
        warnings=warnings or [],
    )


def _warnings_for_visibility(
    support_visibility: SupportVisibility,
    consumer_scope: ConsumerScope,
) -> list[str]:
    if (
        consumer_scope is ConsumerScope.MEMBER
        and support_visibility is SupportVisibility.RESTRICTED_SUPPORT
    ):
        return ["restricted_support"]
    return []


def concept_envelope(
    settings: Settings,
    context_space: ContextSpace,
    policy: VerificationPolicy | None,
    concept: Concept,
    consumer_scope: ConsumerScope,
) -> ContractEnvelope[ConceptPayload]:
    support_items = support_items_for_resource(concept)
    visible_items = visible_support_items(support_items, consumer_scope)
    support_visibility = compute_support_visibility(support_items, visible_items, policy)
    payload = ConceptPayload(
        concept_id=concept.id,
        public_slug=concept.public_slug,
        canonical_name=concept.canonical_name,
        aliases=list(concept.aliases),
        definition=concept.definition,
        owning_domain=concept.owning_domain,
        review_domain=concept.review_domain,
        lifecycle_state=concept.lifecycle_state,
        verification_state=concept.verification_state,
        support_visibility=support_visibility,
        visible_support_items=[
            support_item_summary(item, consumer_scope) for item in visible_items
        ],
        linked_relation_refs=[
            resource_ref(relation)
            for relation in [*concept.outgoing_relations, *concept.incoming_relations]
            if relation.lifecycle_state.value != "archived"
        ],
        linked_decision_refs=[resource_ref(link.decision) for link in concept.decision_links],
        provenance_summary=provenance_summary(
            support_items, consumer_scope, concept.verification_state, policy
        ),
    )
    related_refs = [*payload.linked_relation_refs, *payload.linked_decision_refs]
    return build_envelope(
        settings=settings,
        context_space=context_space,
        consumer_scope=consumer_scope,
        response_kind=ResponseKind.CONCEPT,
        request_intent=RequestIntent.GET_CONCEPT,
        payload=payload,
        related_refs=related_refs,
        warnings=_warnings_for_visibility(support_visibility, consumer_scope),
    )


def relation_envelope(
    settings: Settings,
    context_space: ContextSpace,
    policy: VerificationPolicy | None,
    relation: ConceptRelation,
    consumer_scope: ConsumerScope,
) -> ContractEnvelope[RelationPayload]:
    support_items = support_items_for_resource(relation)
    visible_items = visible_support_items(support_items, consumer_scope)
    support_visibility = compute_support_visibility(support_items, visible_items, policy)
    payload = RelationPayload(
        relation_id=relation.id,
        subject_concept_ref=resource_ref(relation.subject_concept),
        predicate=relation.predicate,
        object_concept_ref=resource_ref(relation.object_concept),
        description=relation.description,
        review_domain=relation.review_domain,
        lifecycle_state=relation.lifecycle_state,
        verification_state=relation.verification_state,
        support_visibility=support_visibility,
        visible_support_items=[
            support_item_summary(item, consumer_scope) for item in visible_items
        ],
        linked_decision_refs=[resource_ref(link.decision) for link in relation.decision_links],
        provenance_summary=provenance_summary(
            support_items, consumer_scope, relation.verification_state, policy
        ),
    )
    return build_envelope(
        settings=settings,
        context_space=context_space,
        consumer_scope=consumer_scope,
        response_kind=ResponseKind.RELATION,
        request_intent=RequestIntent.GET_RELATION,
        payload=payload,
        related_refs=[
            payload.subject_concept_ref,
            payload.object_concept_ref,
            *payload.linked_decision_refs,
        ],
        warnings=_warnings_for_visibility(support_visibility, consumer_scope),
    )


def _superseded_by_ref(session: Session, decision: DecisionRecord) -> ResourceRef | None:
    successor = session.scalar(
        select(DecisionRecord).where(DecisionRecord.supersedes_decision_id == decision.id).limit(1)
    )
    return resource_ref(successor) if successor else None


def decision_envelope(
    session: Session,
    settings: Settings,
    context_space: ContextSpace,
    policy: VerificationPolicy | None,
    decision: DecisionRecord,
    consumer_scope: ConsumerScope,
) -> ContractEnvelope[DecisionPayload]:
    support_items = support_items_for_resource(decision)
    visible_items = visible_support_items(support_items, consumer_scope)
    support_visibility = compute_support_visibility(support_items, visible_items, policy)
    payload = DecisionPayload(
        decision_id=decision.id,
        public_slug=decision.public_slug,
        title=decision.title,
        decision_statement=decision.decision_statement,
        problem_statement=decision.problem_statement,
        rationale=decision.rationale,
        constraints=list(decision.constraints),
        impact_summary=decision.impact_summary,
        owning_domain=decision.owning_domain,
        review_domain=decision.review_domain,
        lifecycle_state=decision.lifecycle_state,
        support_visibility=support_visibility,
        visible_support_items=[
            support_item_summary(item, consumer_scope) for item in visible_items
        ],
        linked_concept_refs=[resource_ref(link.concept) for link in decision.concept_links],
        linked_relation_refs=[resource_ref(link.relation) for link in decision.relation_links],
        supersedes_ref=resource_ref(decision.supersedes_decision)
        if decision.supersedes_decision
        else None,
        superseded_by_ref=_superseded_by_ref(session, decision),
        provenance_summary=provenance_summary(
            support_items, consumer_scope, decision.verification_state, policy
        ),
    )
    related_refs = [
        *payload.linked_concept_refs,
        *payload.linked_relation_refs,
    ]
    if payload.supersedes_ref:
        related_refs.append(payload.supersedes_ref)
    if payload.superseded_by_ref:
        related_refs.append(payload.superseded_by_ref)
    return build_envelope(
        settings=settings,
        context_space=context_space,
        consumer_scope=consumer_scope,
        response_kind=ResponseKind.DECISION,
        request_intent=RequestIntent.GET_DECISION,
        payload=payload,
        related_refs=related_refs,
        warnings=_warnings_for_visibility(support_visibility, consumer_scope),
    )


def answer_envelope(
    settings: Settings,
    context_space: ContextSpace,
    consumer_scope: ConsumerScope,
    *,
    answer_status: AnswerStatus,
    answer_text: str,
    support_items: list[SupportItem],
    verification_state: VerificationState,
    cited_concepts: list[Concept],
    cited_relations: list[ConceptRelation],
    cited_decisions: list[DecisionRecord],
    policy: VerificationPolicy | None,
) -> ContractEnvelope[AnswerPayload]:
    visible_items = visible_support_items(support_items, consumer_scope)
    support_visibility = compute_support_visibility(support_items, visible_items, policy)
    payload = AnswerPayload(
        answer_status=answer_status,
        answer_text=answer_text,
        answer_sections=[
            AnswerSection(
                heading="Concepts",
                body=", ".join(concept.canonical_name for concept in cited_concepts),
            )
            if cited_concepts
            else AnswerSection(heading="Concepts", body="No official concepts matched directly.")
        ],
        support_visibility=support_visibility,
        verification_state=verification_state,
        visible_support_items=[
            support_item_summary(item, consumer_scope) for item in visible_items
        ],
        cited_concept_refs=[resource_ref(concept) for concept in cited_concepts],
        cited_relation_refs=[resource_ref(relation) for relation in cited_relations],
        cited_decision_refs=[resource_ref(decision) for decision in cited_decisions],
        provenance_summary=provenance_summary(
            support_items, consumer_scope, verification_state, policy
        ),
        follow_up_refs=[
            *[resource_ref(concept) for concept in cited_concepts[:3]],
            *[resource_ref(decision) for decision in cited_decisions[:2]],
        ],
    )
    return build_envelope(
        settings=settings,
        context_space=context_space,
        consumer_scope=consumer_scope,
        response_kind=ResponseKind.ANSWER,
        request_intent=RequestIntent.GET_ANSWER,
        payload=payload,
        related_refs=[
            *payload.cited_concept_refs,
            *payload.cited_relation_refs,
            *payload.cited_decision_refs,
        ],
        warnings=_warnings_for_visibility(support_visibility, consumer_scope),
    )


def search_results_envelope(
    settings: Settings,
    context_space: ContextSpace,
    consumer_scope: ConsumerScope,
    results: list[SearchResultItem],
) -> ContractEnvelope[SearchResultsPayload]:
    payload = SearchResultsPayload(results=results, result_count=len(results))
    return build_envelope(
        settings=settings,
        context_space=context_space,
        consumer_scope=consumer_scope,
        response_kind=ResponseKind.SEARCH_RESULTS,
        request_intent=RequestIntent.SEARCH_CONTEXT,
        payload=payload,
        related_refs=[item.resource_ref for item in results],
    )


def graph_slice_envelope(
    settings: Settings,
    context_space: ContextSpace,
    consumer_scope: ConsumerScope,
    *,
    root_concepts: list[Concept],
    nodes: list[Concept],
    edges: list[ConceptRelation],
    policy: VerificationPolicy | None,
) -> ContractEnvelope[GraphSlicePayload]:
    payload = GraphSlicePayload(
        root_concept_refs=[resource_ref(concept) for concept in root_concepts],
        nodes=[resource_ref(concept) for concept in nodes],
        edges=[
            GraphEdgePayload(
                relation_ref=resource_ref(relation),
                subject_concept_ref=resource_ref(relation.subject_concept),
                predicate=relation.predicate,
                object_concept_ref=resource_ref(relation.object_concept),
                support_visibility=compute_support_visibility(
                    support_items_for_resource(relation),
                    visible_support_items(support_items_for_resource(relation), consumer_scope),
                    policy,
                ),
                verification_state=relation.verification_state,
            )
            for relation in edges
        ],
    )
    return build_envelope(
        settings=settings,
        context_space=context_space,
        consumer_scope=consumer_scope,
        response_kind=ResponseKind.GRAPH_SLICE,
        request_intent=RequestIntent.GET_GRAPH_SLICE,
        payload=payload,
        related_refs=[
            *payload.root_concept_refs,
            *payload.nodes,
            *[edge.relation_ref for edge in payload.edges],
        ],
    )


def provenance_envelope(
    settings: Settings,
    context_space: ContextSpace,
    consumer_scope: ConsumerScope,
    *,
    subject,
    support_items: list[SupportItem],
    policy: VerificationPolicy | None,
    verification_state: VerificationState | None,
) -> ContractEnvelope[ProvenancePayload]:
    visible_items = visible_support_items(support_items, consumer_scope)
    source_summaries: list[SourceSummary] = []
    seen_source_ids: set[str] = set()
    for support_item in support_items:
        if support_item.artifact is None:
            continue
        connection: SourceConnection = support_item.artifact.source_connection
        if connection.id in seen_source_ids:
            continue
        seen_source_ids.add(connection.id)
        source_summaries.append(
            SourceSummary(
                source_connection_id=connection.id,
                source_label=connection.source_label,
                source_connection_state=connection.source_connection_state,
                freshness_state=connection.freshness_state,
                visibility_class=connection.visibility_class,
                last_attempted_sync_at=connection.last_attempted_sync_at,
                last_successful_sync_at=connection.last_successful_sync_at,
                effective_sync_policy=connection.effective_sync_policy,
                last_error=connection.last_error,
            )
        )
    payload = ProvenancePayload(
        subject_ref=resource_ref(subject),
        support_items=[support_item_summary(item, consumer_scope) for item in visible_items],
        source_summaries=source_summaries,
        provenance_summary=provenance_summary(
            support_items, consumer_scope, verification_state, policy
        ),
    )
    return build_envelope(
        settings=settings,
        context_space=context_space,
        consumer_scope=consumer_scope,
        response_kind=ResponseKind.PROVENANCE,
        request_intent=RequestIntent.FOLLOW_PROVENANCE,
        payload=payload,
        related_refs=[payload.subject_ref, *[resource_ref(item) for item in support_items]],
    )


def no_match_envelope(
    settings: Settings,
    context_space: ContextSpace,
    consumer_scope: ConsumerScope,
    *,
    reason: NoMatchReason,
    request_intent: RequestIntent,
    request_rewrite_hint: str | None = None,
    suggested_follow_up: list[SuggestedFollowUp] | None = None,
) -> ContractEnvelope[NoMatchPayload]:
    payload = NoMatchPayload(
        reason=reason,
        request_rewrite_hint=request_rewrite_hint,
        suggested_follow_up=suggested_follow_up or [],
    )
    return build_envelope(
        settings=settings,
        context_space=context_space,
        consumer_scope=consumer_scope,
        response_kind=ResponseKind.NO_MATCH,
        request_intent=request_intent,
        payload=payload,
    )
