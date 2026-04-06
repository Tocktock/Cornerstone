from __future__ import annotations

from sqlalchemy.orm import Session

from cornerstone.clock import utcnow
from cornerstone.config import Settings
from cornerstone.domain.enums import (
    ContextSpaceKind,
    CuratedLifecycleState,
    DecisionLifecycleState,
    SupportItemKind,
    VerificationState,
)
from cornerstone.domain.models import (
    Actor,
    Concept,
    ConceptRelation,
    ConceptSupportLink,
    ContextSpace,
    DecisionConceptLink,
    DecisionRecord,
    DecisionRelationLink,
    DecisionSupportLink,
    PromotionLineage,
    RelationSupportLink,
    SupportItem,
)
from cornerstone.domain.schemas import (
    DraftConceptCreate,
    DraftDecisionCreate,
    DraftRelationCreate,
    PromotionRequest,
)
from cornerstone.services.normalization import slugify, stable_id
from cornerstone.services.policies import PolicyError, derive_relation_review_domain


def _ensure_workspace_context(session: Session, context_space_id: str) -> ContextSpace:
    context_space = session.get(ContextSpace, context_space_id)
    if context_space is None:
        raise PolicyError("Context space not found.")
    if context_space.kind is not ContextSpaceKind.WORKSPACE:
        raise PolicyError("Shared drafts may be created only inside a workspace context.")
    return context_space


def _ensure_workspace_support_item(
    session: Session, context_space_id: str, support_item_id: str
) -> SupportItem:
    support_item = session.get(SupportItem, support_item_id)
    if support_item is None:
        raise PolicyError(f"Support item not found: {support_item_id}")
    if support_item.context_space_id != context_space_id:
        raise PolicyError("Shared objects may not cite support from a different context.")
    return support_item


def create_concept(session: Session, payload: DraftConceptCreate) -> Concept:
    _ensure_workspace_context(session, payload.context_space_id)
    concept = Concept(
        id=stable_id("concept", payload.context_space_id, payload.canonical_name),
        context_space_id=payload.context_space_id,
        public_slug=slugify(payload.canonical_name),
        canonical_name=payload.canonical_name,
        aliases=list(payload.aliases),
        definition=payload.definition,
        concept_kind=payload.concept_kind,
        owning_domain=payload.owning_domain,
        review_domain=payload.owning_domain,
        lifecycle_state=CuratedLifecycleState.DRAFT,
        verification_state=VerificationState.UNVERIFIED,
    )
    session.add(concept)
    session.flush()
    for support_item_id in payload.support_item_ids:
        support_item = _ensure_workspace_support_item(
            session, payload.context_space_id, support_item_id
        )
        concept.support_links.append(ConceptSupportLink(support_item_id=support_item.id))
    for decision_id in payload.linked_decision_ids:
        decision = session.get(DecisionRecord, decision_id)
        if decision is None:
            raise PolicyError(f"Decision not found: {decision_id}")
        concept.decision_links.append(DecisionConceptLink(decision_id=decision.id))
    session.flush()
    return concept


def create_relation(session: Session, payload: DraftRelationCreate) -> ConceptRelation:
    _ensure_workspace_context(session, payload.context_space_id)
    subject_concept = session.get(Concept, payload.subject_concept_id)
    object_concept = session.get(Concept, payload.object_concept_id)
    if subject_concept is None or object_concept is None:
        raise PolicyError("Subject and object concepts must exist before creating a relation.")
    relation = ConceptRelation(
        id=stable_id(
            "rel",
            payload.context_space_id,
            payload.subject_concept_id,
            payload.predicate,
            payload.object_concept_id,
        ),
        context_space_id=payload.context_space_id,
        subject_concept_id=payload.subject_concept_id,
        predicate=payload.predicate,
        object_concept_id=payload.object_concept_id,
        description=payload.description,
        review_domain=derive_relation_review_domain(
            subject_concept,
            object_concept,
            workspace_wide=payload.workspace_wide,
        ),
        lifecycle_state=CuratedLifecycleState.DRAFT,
        verification_state=VerificationState.UNVERIFIED,
    )
    session.add(relation)
    session.flush()
    for support_item_id in payload.support_item_ids:
        support_item = _ensure_workspace_support_item(
            session, payload.context_space_id, support_item_id
        )
        relation.support_links.append(RelationSupportLink(support_item_id=support_item.id))
    for decision_id in payload.linked_decision_ids:
        decision = session.get(DecisionRecord, decision_id)
        if decision is None:
            raise PolicyError(f"Decision not found: {decision_id}")
        relation.decision_links.append(DecisionRelationLink(decision_id=decision.id))
    session.flush()
    return relation


def create_decision(session: Session, payload: DraftDecisionCreate) -> DecisionRecord:
    _ensure_workspace_context(session, payload.context_space_id)
    decision = DecisionRecord(
        id=stable_id("decision", payload.context_space_id, payload.title),
        context_space_id=payload.context_space_id,
        title=payload.title,
        problem_statement=payload.problem_statement,
        decision_statement=payload.decision_statement,
        rationale=payload.rationale,
        constraints=list(payload.constraints),
        impact_summary=payload.impact_summary,
        owning_domain=payload.owning_domain,
        review_domain=payload.owning_domain,
        lifecycle_state=DecisionLifecycleState.PROPOSED,
        verification_state=VerificationState.UNVERIFIED,
        supersedes_decision_id=payload.supersedes_decision_id,
    )
    session.add(decision)
    session.flush()
    for support_item_id in payload.support_item_ids:
        support_item = _ensure_workspace_support_item(
            session, payload.context_space_id, support_item_id
        )
        decision.support_links.append(DecisionSupportLink(support_item_id=support_item.id))
    for concept_id in payload.linked_concept_ids:
        concept = session.get(Concept, concept_id)
        if concept is None:
            raise PolicyError(f"Concept not found: {concept_id}")
        decision.concept_links.append(DecisionConceptLink(concept_id=concept.id))
    for relation_id in payload.linked_relation_ids:
        relation = session.get(ConceptRelation, relation_id)
        if relation is None:
            raise PolicyError(f"Relation not found: {relation_id}")
        decision.relation_links.append(DecisionRelationLink(relation_id=relation.id))
    session.flush()
    return decision


def create_promoted_support(
    session: Session,
    payload: PromotionRequest,
    promoter: Actor,
    settings: Settings | None = None,
) -> SupportItem:
    resolved_settings = settings or Settings()
    workspace_context = _ensure_workspace_context(session, payload.workspace_context_id)
    personal_support = session.get(SupportItem, payload.personal_support_item_id)
    if personal_support is None:
        raise PolicyError("Personal support item not found.")
    personal_context = session.get(ContextSpace, personal_support.context_space_id)
    if personal_context is None or personal_context.kind is not ContextSpaceKind.PERSONAL:
        raise PolicyError("Only personal-context support may be promoted into a workspace.")
    if promoter.principal_key != personal_context.membership_boundary:
        raise PolicyError(
            "Only the personal context owner may promote private support into the workspace."
        )

    now = utcnow(resolved_settings)
    lineage = PromotionLineage(
        id=stable_id("lineage", workspace_context.id, personal_support.id, promoter.id),
        source_context_kind=ContextSpaceKind.PERSONAL,
        personal_source_owner_principal_key=promoter.principal_key,
        private_origin_ref=f"private:{personal_support.id}",
        selection_method=payload.shared_selection_kind.value,
        selection_scope_summary=personal_support.selector or personal_support.source_label,
        workspace_disclosure_note=(
            "Promoted personal material is shared only as "
            "a workspace snapshot."
        ),
        origin_disclosure_level=payload.origin_disclosure_level,
    )
    session.add(lineage)
    session.flush()

    promoted_support = SupportItem(
        id=stable_id("promoted", workspace_context.id, personal_support.id, promoter.id),
        context_space_id=workspace_context.id,
        support_item_kind=SupportItemKind.PROMOTED_SUPPORT,
        visibility_class=payload.visibility_class,
        source_label="Promoted personal support",
        excerpt_or_summary=payload.shared_payload,
        freshness_state=personal_support.freshness_state,
        promoter_id=promoter.id,
        promoted_at=now,
        shared_selection_kind=payload.shared_selection_kind,
        shared_payload=payload.shared_payload,
        origin_disclosure_level=payload.origin_disclosure_level,
        promotion_lineage_id=lineage.id,
    )
    session.add(promoted_support)
    session.flush()
    return promoted_support
