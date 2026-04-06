from __future__ import annotations

from datetime import datetime

from cornerstone.domain.enums import (
    ConceptKind,
    ConsumerScope,
    FreshnessState,
    OriginDisclosureLevel,
    SharedSelectionKind,
    SupportItemKind,
    VisibilityClass,
)
from cornerstone.domain.models import Concept, SupportItem, VerificationPolicy
from cornerstone.services.policies import (
    derive_relation_review_domain,
    stamp_sync_freshness,
    support_visibility_for_consumer,
)
from cornerstone.services.serialization import support_item_summary


def test_cross_domain_review_domain_resolves_to_workspace():
    subject = Concept(
        id="c1",
        context_space_id="ctx",
        public_slug="partner-sla",
        canonical_name="Partner SLA",
        aliases=[],
        definition="Published partner commitment",
        concept_kind=ConceptKind.TERM,
        owning_domain="sales_ops",
        review_domain="sales_ops",
    )
    obj = Concept(
        id="c2",
        context_space_id="ctx",
        public_slug="platform-circuit-breaker",
        canonical_name="Platform Circuit Breaker",
        aliases=[],
        definition="Platform safety control",
        concept_kind=ConceptKind.TERM,
        owning_domain="platform",
        review_domain="platform",
    )

    assert derive_relation_review_domain(subject, obj) == "workspace"


def test_restricted_support_is_reported_when_visible_support_is_missing():
    policy = VerificationPolicy(
        id="policy",
        context_space_id="ctx",
        label="policy",
        version="p0",
        minimum_support_items=1,
        minimum_durable_support_items=1,
        minimum_visible_support_items_for_source_backed=1,
        allow_restricted_support_for_officialization=True,
        allow_member_restricted_support_publication=True,
        freshness_target_hours=24,
        continuous_revalidation_enabled=True,
        allow_accepted_decision_lineage_as_support=True,
    )
    hidden_support = SupportItem(
        id="supp-1",
        context_space_id="ctx",
        support_item_kind=SupportItemKind.EVIDENCE_FRAGMENT,
        visibility_class=VisibilityClass.EVIDENCE_ONLY,
        source_label="Review-only notebook",
        excerpt_or_summary="Sensitive signal",
    )

    support_visibility = support_visibility_for_consumer(policy, [hidden_support], [])

    assert support_visibility.value == "restricted_support"


def test_promoted_support_summary_hides_private_origin_locator_from_members():
    promoted_support = SupportItem(
        id="promoted-1",
        context_space_id="workspace",
        support_item_kind=SupportItemKind.PROMOTED_SUPPORT,
        visibility_class=VisibilityClass.MEMBER_VISIBLE,
        source_label="Promoted personal support",
        excerpt_or_summary="Workspace-visible summary",
        source_locator="private:artifact/123",
        freshness_state=FreshnessState.CURRENT,
        shared_selection_kind=SharedSelectionKind.SUMMARY_CLAIM,
        origin_disclosure_level=OriginDisclosureLevel.REDACTED_ORIGIN,
        shared_payload="Workspace-visible summary",
    )

    summary = support_item_summary(promoted_support, consumer_scope=ConsumerScope.MEMBER)

    assert summary.source_locator == "workspace-promotion:promoted-1"
    assert summary.origin_disclosure_level == OriginDisclosureLevel.REDACTED_ORIGIN


def test_stamp_sync_freshness_marks_stale_and_drifted_content():
    now = datetime.fromisoformat("2026-04-06T09:00:00+09:00")
    stale = stamp_sync_freshness(
        datetime.fromisoformat("2026-04-03T04:00:00+09:00"),
        now,
        stale_after_hours=48,
        drift_after_hours=96,
    )
    drifted = stamp_sync_freshness(
        datetime.fromisoformat("2026-03-31T00:00:00+09:00"),
        now,
        stale_after_hours=48,
        drift_after_hours=96,
    )

    assert stale is FreshnessState.STALE
    assert drifted is FreshnessState.DRIFT_DETECTED
