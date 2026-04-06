from __future__ import annotations

from collections import OrderedDict

from sqlalchemy.orm import Session

from cornerstone.config import Settings
from cornerstone.domain.enums import (
    AnswerStatus,
    ConsumerScope,
    NoMatchReason,
    RequestIntent,
    VerificationState,
)
from cornerstone.domain.models import (
    Concept,
    ConceptRelation,
    ContextSpace,
    DecisionRecord,
    SupportItem,
)
from cornerstone.domain.schemas import SearchResultItem
from cornerstone.services.catalog import filter_visible_resources, get_policy, search_resources
from cornerstone.services.serialization import (
    answer_envelope,
    no_match_envelope,
    resource_ref,
    search_results_envelope,
    support_items_for_resource,
)


def search_context(
    session: Session,
    settings: Settings,
    context_space: ContextSpace,
    consumer_scope: ConsumerScope,
    query: str,
):
    policy = get_policy(session, context_space.id)
    concepts, relations, decisions = search_resources(session, context_space.id, query=query)
    visible_concepts = filter_visible_resources(concepts, consumer_scope, policy)
    visible_relations = filter_visible_resources(relations, consumer_scope, policy)
    visible_decisions = filter_visible_resources(decisions, consumer_scope, policy)

    results: list[SearchResultItem] = []
    for concept in visible_concepts[:8]:
        results.append(
            SearchResultItem(
                resource_ref=resource_ref(concept),
                match_reason_summary=f"Matched concept name or definition for '{query}'.",
                support_visibility=concept.support_visibility,
                lifecycle_state=concept.lifecycle_state,
                verification_state=concept.verification_state,
                provenance_summary=None,
            )
        )
    for relation in visible_relations[:5]:
        results.append(
            SearchResultItem(
                resource_ref=resource_ref(relation),
                match_reason_summary=f"Matched relation path for '{query}'.",
                support_visibility=relation.support_visibility,
                lifecycle_state=relation.lifecycle_state,
                verification_state=relation.verification_state,
                provenance_summary=None,
            )
        )
    for decision in visible_decisions[:5]:
        results.append(
            SearchResultItem(
                resource_ref=resource_ref(decision),
                match_reason_summary=f"Matched decision title or statement for '{query}'.",
                support_visibility=decision.support_visibility,
                lifecycle_state=decision.lifecycle_state,
                verification_state=decision.verification_state,
                provenance_summary=None,
            )
        )

    if not results:
        hidden_exists = bool(concepts or relations or decisions)
        return no_match_envelope(
            settings=settings,
            context_space=context_space,
            consumer_scope=consumer_scope,
            reason=NoMatchReason.NO_VISIBLE_MATCH
            if hidden_exists
            else NoMatchReason.NO_OFFICIAL_MATCH,
            request_intent=RequestIntent.SEARCH_CONTEXT,
            request_rewrite_hint=(
                "Try a narrower domain term or inspect provenance "
                "from review scope."
            ),
        )

    return search_results_envelope(
        settings=settings,
        context_space=context_space,
        consumer_scope=consumer_scope,
        results=results,
    )


def answer_query(
    session: Session,
    settings: Settings,
    context_space: ContextSpace,
    consumer_scope: ConsumerScope,
    query: str,
):
    policy = get_policy(session, context_space.id)
    concepts, relations, decisions = search_resources(session, context_space.id, query=query)
    visible_concepts = filter_visible_resources(concepts, consumer_scope, policy)
    visible_relations = filter_visible_resources(relations, consumer_scope, policy)
    visible_decisions = filter_visible_resources(decisions, consumer_scope, policy)
    if not visible_concepts and not visible_relations and not visible_decisions:
        hidden_exists = bool(concepts or relations or decisions)
        return no_match_envelope(
            settings=settings,
            context_space=context_space,
            consumer_scope=consumer_scope,
            reason=NoMatchReason.NO_VISIBLE_MATCH
            if hidden_exists
            else NoMatchReason.NO_OFFICIAL_MATCH,
            request_intent=RequestIntent.GET_ANSWER,
            request_rewrite_hint="Try a specific concept or decision title.",
        )

    support_map: OrderedDict[str, SupportItem] = OrderedDict()
    for resource in [*visible_concepts[:4], *visible_relations[:4], *visible_decisions[:4]]:
        for support_item in support_items_for_resource(resource):
            if support_item.id not in support_map:
                support_map[support_item.id] = support_item
    support_items = list(support_map.values())

    answer_status = _answer_status(visible_concepts, visible_relations, visible_decisions)
    answer_text = _answer_text(
        query, visible_concepts, visible_relations, visible_decisions, consumer_scope
    )
    verification_state = _verification_state(visible_concepts, visible_relations, visible_decisions)
    return answer_envelope(
        settings=settings,
        context_space=context_space,
        consumer_scope=consumer_scope,
        answer_status=answer_status,
        answer_text=answer_text,
        support_items=support_items,
        verification_state=verification_state,
        cited_concepts=visible_concepts[:4],
        cited_relations=visible_relations[:4],
        cited_decisions=visible_decisions[:4],
        policy=policy,
    )


def _answer_status(
    concepts: list[Concept],
    relations: list[ConceptRelation],
    decisions: list[DecisionRecord],
) -> AnswerStatus:
    verification_states = {
        *(concept.verification_state for concept in concepts),
        *(relation.verification_state for relation in relations),
        *(decision.verification_state for decision in decisions),
    }
    if (
        VerificationState.REVIEW_REQUIRED in verification_states
        or VerificationState.DRIFT_DETECTED in verification_states
    ):
        return AnswerStatus.REVIEW_REQUIRED
    if decisions or concepts:
        return AnswerStatus.OFFICIAL
    return AnswerStatus.PARTIAL


def _verification_state(
    concepts: list[Concept],
    relations: list[ConceptRelation],
    decisions: list[DecisionRecord],
) -> VerificationState:
    verification_states = [
        *(concept.verification_state for concept in concepts),
        *(relation.verification_state for relation in relations),
        *(decision.verification_state for decision in decisions),
    ]
    if not verification_states:
        return VerificationState.UNVERIFIED
    if VerificationState.DRIFT_DETECTED in verification_states:
        return VerificationState.DRIFT_DETECTED
    if VerificationState.REVIEW_REQUIRED in verification_states:
        return VerificationState.REVIEW_REQUIRED
    if VerificationState.MONITORING in verification_states:
        return VerificationState.MONITORING
    return verification_states[0]


def _answer_text(
    query: str,
    concepts: list[Concept],
    relations: list[ConceptRelation],
    decisions: list[DecisionRecord],
    consumer_scope: ConsumerScope,
) -> str:
    fragments: list[str] = [f"Answer for '{query}' in {consumer_scope.value} scope."]
    if concepts:
        fragments.append(
            "Key concepts: " + ", ".join(concept.canonical_name for concept in concepts[:3]) + "."
        )
    if relations:
        fragments.append(
            "Relevant relations: "
            + "; ".join(
                (
                    f"{relation.subject_concept.canonical_name} "
                    f"{relation.predicate} "
                    f"{relation.object_concept.canonical_name}"
                )
                for relation in relations[:2]
            )
            + "."
        )
    if decisions:
        fragments.append(
            "Decisions: " + ", ".join(decision.title for decision in decisions[:2]) + "."
        )
    return " ".join(fragments)
