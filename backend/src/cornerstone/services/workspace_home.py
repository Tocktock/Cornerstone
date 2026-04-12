from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from cornerstone.config import Settings
from cornerstone.domain.enums import (
    ConsumerScope,
    CuratedLifecycleState,
    DecisionLifecycleState,
    FreshnessState,
    RequestIntent,
    ResponseKind,
    SourceConnectionState,
    VerificationState,
)
from cornerstone.domain.models import Actor, Concept, ConceptRelation, ContextSpace, DecisionRecord, SourceConnection
from cornerstone.domain.schemas import (
    ContractEnvelope,
    ReviewQueueSummary,
    SourceHealthSummary,
    WorkspaceHomeFeaturedCard,
    WorkspaceHomeFreshnessAlert,
    WorkspaceHomePayload,
    WorkspaceHomeRecentChange,
)
from cornerstone.services.answering import answer_query
from cornerstone.services.catalog import filter_visible_resources, get_policy, list_concepts, list_decisions, list_relations
from cornerstone.services.serialization import (
    build_envelope,
    compute_support_visibility,
    provenance_summary,
    resource_ref,
    support_items_for_resource,
    visible_support_items,
)


def workspace_home_envelope(
    session: Session,
    settings: Settings,
    context_space: ContextSpace,
    actor: Actor,
    consumer_scope: ConsumerScope,
) -> ContractEnvelope[WorkspaceHomePayload]:
    policy = get_policy(session, context_space.id)
    visible_concepts = filter_visible_resources(
        list_concepts(session, context_space.id),
        consumer_scope,
        policy,
    )
    visible_decisions = filter_visible_resources(
        list_decisions(session, context_space.id),
        consumer_scope,
        policy,
    )
    relations = list_relations(session, context_space.id)
    connections = list(
        session.scalars(
            select(SourceConnection)
            .where(SourceConnection.context_space_id == context_space.id)
            .order_by(SourceConnection.source_label.asc())
        )
    )

    featured_answer = answer_query(
        session,
        settings,
        context_space,
        consumer_scope,
        "escalation",
    )
    featured_cards = _featured_cards(visible_concepts, visible_decisions, consumer_scope, policy)
    recent_changes = _recent_changes(visible_concepts, visible_decisions, consumer_scope, policy)
    freshness_alerts = _freshness_alerts(connections)
    review_candidates = _review_candidates(visible_concepts, relations, visible_decisions)

    payload = WorkspaceHomePayload(
        hero_prompt=(
            "Ask about official workspace context, operating guidance, decision lineage, "
            "or source-backed explanations."
        ),
        featured_answer=featured_answer,
        featured_cards=featured_cards,
        recent_changes=recent_changes,
        freshness_alerts=freshness_alerts,
        review_queue_summary=ReviewQueueSummary(
            pending_count=len(review_candidates),
            review_required_count=sum(
                candidate.verification_state is VerificationState.REVIEW_REQUIRED
                for candidate in review_candidates
            ),
            officialize_ready_count=sum(
                candidate.verification_state is not VerificationState.REVIEW_REQUIRED
                for candidate in review_candidates
            ),
        ),
        source_health_summary=SourceHealthSummary(
            total_count=len(connections),
            active_count=sum(
                connection.source_connection_state is SourceConnectionState.ACTIVE
                for connection in connections
            ),
            monitoring_count=sum(
                connection.freshness_state is FreshnessState.MONITORING for connection in connections
            ),
            stale_count=sum(
                connection.freshness_state is FreshnessState.STALE for connection in connections
            ),
            degraded_count=sum(
                connection.source_connection_state is SourceConnectionState.DEGRADED
                for connection in connections
            ),
            paused_count=sum(
                connection.source_connection_state is SourceConnectionState.PAUSED
                for connection in connections
            ),
            removed_count=sum(
                connection.source_connection_state is SourceConnectionState.REMOVED
                for connection in connections
            ),
        ),
    )

    related_refs = _dedupe_refs(
        [
            card.resource_ref
            for card in featured_cards
        ]
        + [change.resource_ref for change in recent_changes]
        + featured_answer.related_refs
    )
    warnings = []
    if actor.preferred_consumer_scope is ConsumerScope.MEMBER:
        warnings.append("Operational studios remain capability-gated.")

    return build_envelope(
        settings=settings,
        context_space=context_space,
        consumer_scope=consumer_scope,
        response_kind=ResponseKind.WORKSPACE_HOME,
        request_intent=RequestIntent.GET_WORKSPACE_HOME,
        payload=payload,
        related_refs=related_refs,
        warnings=warnings,
    )


def _featured_cards(
    concepts: list[Concept],
    decisions: list[DecisionRecord],
    consumer_scope: ConsumerScope,
    policy,
) -> list[WorkspaceHomeFeaturedCard]:
    concept_cards = [
        _concept_card(concept, consumer_scope, policy)
        for concept in concepts[:2]
    ]
    decision_cards = [
        _decision_card(decision, consumer_scope, policy)
        for decision in decisions[:2]
    ]
    return [*concept_cards, *decision_cards]


def _recent_changes(
    concepts: list[Concept],
    decisions: list[DecisionRecord],
    consumer_scope: ConsumerScope,
    policy,
) -> list[WorkspaceHomeRecentChange]:
    resources = sorted(
        [*concepts, *decisions],
        key=lambda resource: resource.updated_at,
        reverse=True,
    )[:4]
    items: list[WorkspaceHomeRecentChange] = []
    for resource in resources:
        support_visibility, resource_provenance = _resource_surface_state(
            resource,
            consumer_scope,
            policy,
        )
        items.append(
            WorkspaceHomeRecentChange(
                resource_ref=resource_ref(resource),
                public_slug=resource.public_slug,
                change_summary=_resource_summary(resource),
                changed_at=resource.updated_at,
                support_visibility=support_visibility,
                lifecycle_state=resource.lifecycle_state,
                verification_state=getattr(resource, "verification_state", None),
            )
        )
    return items


def _freshness_alerts(
    connections: list[SourceConnection],
) -> list[WorkspaceHomeFreshnessAlert]:
    candidates = [
        connection
        for connection in connections
        if connection.source_connection_state is not SourceConnectionState.ACTIVE
        or connection.freshness_state is not FreshnessState.CURRENT
    ]
    severity_order = {
        SourceConnectionState.DEGRADED: 0,
        SourceConnectionState.PAUSED: 1,
        SourceConnectionState.REMOVED: 2,
        SourceConnectionState.ACTIVE: 3,
        SourceConnectionState.PENDING_SETUP: 4,
        SourceConnectionState.SYNCING: 5,
    }
    freshness_order = {
        FreshnessState.STALE: 0,
        FreshnessState.DRIFT_DETECTED: 1,
        FreshnessState.MONITORING: 2,
        FreshnessState.UNKNOWN: 3,
        FreshnessState.CURRENT: 4,
    }
    ordered = sorted(
        candidates,
        key=lambda connection: (
            severity_order.get(connection.source_connection_state, 99),
            freshness_order.get(connection.freshness_state, 99),
            connection.source_label,
        ),
    )[:4]
    return [
        WorkspaceHomeFreshnessAlert(
            source_connection_id=connection.id,
            source_label=connection.source_label,
            source_connection_state=connection.source_connection_state,
            freshness_state=connection.freshness_state,
            last_successful_sync_at=connection.last_successful_sync_at,
            note=_freshness_note(connection),
        )
        for connection in ordered
    ]


def _review_candidates(
    concepts: list[Concept],
    relations: list[ConceptRelation],
    decisions: list[DecisionRecord],
) -> list[Concept | ConceptRelation | DecisionRecord]:
    return [
        resource
        for resource in [*concepts, *relations, *decisions]
        if _needs_review_attention(resource)
    ]


def _needs_review_attention(resource: Concept | ConceptRelation | DecisionRecord) -> bool:
    if resource.verification_state is VerificationState.REVIEW_REQUIRED:
        return True
    if isinstance(resource, DecisionRecord):
        return resource.lifecycle_state not in {
            DecisionLifecycleState.ACCEPTED,
            DecisionLifecycleState.SUPERSEDED,
        }
    return resource.lifecycle_state is not CuratedLifecycleState.OFFICIAL


def _concept_card(
    concept: Concept,
    consumer_scope: ConsumerScope,
    policy,
) -> WorkspaceHomeFeaturedCard:
    support_visibility, resource_provenance = _resource_surface_state(concept, consumer_scope, policy)
    return WorkspaceHomeFeaturedCard(
        resource_ref=resource_ref(concept),
        public_slug=concept.public_slug,
        title=concept.canonical_name,
        eyebrow=f"{concept.owning_domain} topic",
        summary=concept.definition,
        support_visibility=support_visibility,
        lifecycle_state=concept.lifecycle_state,
        verification_state=concept.verification_state,
        provenance_summary=resource_provenance,
    )


def _decision_card(
    decision: DecisionRecord,
    consumer_scope: ConsumerScope,
    policy,
) -> WorkspaceHomeFeaturedCard:
    support_visibility, resource_provenance = _resource_surface_state(
        decision,
        consumer_scope,
        policy,
    )
    return WorkspaceHomeFeaturedCard(
        resource_ref=resource_ref(decision),
        public_slug=decision.public_slug,
        title=decision.title,
        eyebrow=f"{decision.owning_domain} decision",
        summary=decision.decision_statement,
        support_visibility=support_visibility,
        lifecycle_state=decision.lifecycle_state,
        verification_state=decision.verification_state,
        provenance_summary=resource_provenance,
    )


def _resource_surface_state(
    resource: Concept | DecisionRecord,
    consumer_scope: ConsumerScope,
    policy,
):
    support_items = support_items_for_resource(resource)
    visible_items = visible_support_items(support_items, consumer_scope)
    support_visibility = compute_support_visibility(support_items, visible_items, policy)
    return (
        support_visibility,
        provenance_summary(
            support_items,
            consumer_scope,
            resource.verification_state,
            policy,
        ),
    )


def _resource_summary(resource: Concept | DecisionRecord) -> str:
    if isinstance(resource, Concept):
        return resource.definition
    return resource.decision_statement


def _freshness_note(connection: SourceConnection) -> str:
    if connection.last_error:
        return connection.last_error
    if connection.source_connection_state is SourceConnectionState.PAUSED:
        return "Sync is paused until an operator resumes the connection."
    if connection.source_connection_state is SourceConnectionState.REMOVED:
        return "This source has been removed from the workspace."
    if connection.freshness_state is FreshnessState.STALE:
        return "Freshness is overdue against the workspace policy target."
    if connection.freshness_state is FreshnessState.MONITORING:
        return "Source health is being monitored after a recent sync."
    return "Operational follow-up is recommended."


def _dedupe_refs(refs):
    seen: set[tuple[str, str]] = set()
    unique = []
    for ref in refs:
        key = (ref.resource_kind.value, ref.resource_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ref)
    return unique
