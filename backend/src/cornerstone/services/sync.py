from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from cornerstone.connectors.filesystem import FilesystemConnector, ParsedArtifact
from cornerstone.domain.enums import (
    ConceptStatus,
    ConceptType,
    ConnectionHealthStatus,
    DecisionConceptRole,
    DecisionRelationRole,
    DecisionStatus,
    EvidenceVerificationStatus,
    RelationStatus,
)
from cornerstone.domain.models import (
    Artifact,
    Concept,
    ConceptEvidenceLink,
    ConceptRelation,
    ContextSpace,
    DecisionConceptLink,
    DecisionEvidenceLink,
    DecisionRecord,
    DecisionRelationLink,
    EvidenceFragment,
    RelationEvidenceLink,
    SourceConnection,
)
from cornerstone.domain.schemas import SyncRunResult
from cornerstone.services.normalization import normalize_key


def run_sync(db: Session, connection: SourceConnection) -> SyncRunResult:
    connector = FilesystemConnector(connection.external_scope)
    parsed_artifacts = connector.list_artifacts()
    artifact_count = 0
    evidence_count = 0
    concept_count = 0
    relation_count = 0
    decision_count = 0

    for parsed in parsed_artifacts:
        artifact = upsert_artifact(db, connection, parsed)
        artifact_count += 1
        evidence_count += len(artifact.evidence_fragments)
        kind = str(parsed.frontmatter.get("kind", "")).lower()
        if kind == "concept":
            upsert_concept_from_artifact(db, connection.context_space_id, artifact, parsed)
            concept_count += 1
        elif kind == "decision":
            upsert_decision_from_artifact(db, connection.context_space_id, artifact, parsed)
            decision_count += 1
        elif kind == "relation":
            upsert_relation_from_artifact(db, connection.context_space_id, artifact, parsed)
            relation_count += 1

    connection.last_synced_at = datetime.now(UTC)
    connection.health_status = ConnectionHealthStatus.HEALTHY
    connection.last_error = None
    db.add(connection)
    db.flush()

    return SyncRunResult(
        source_connection_id=connection.id,
        artifact_count=artifact_count,
        evidence_count=evidence_count,
        concept_count=concept_count,
        relation_count=relation_count,
        decision_count=decision_count,
    )


def upsert_artifact(db: Session, connection: SourceConnection, parsed: ParsedArtifact) -> Artifact:
    artifact = db.scalar(
        select(Artifact).where(
            Artifact.source_connection_id == connection.id,
            Artifact.external_id == parsed.external_id,
        )
    )
    if artifact is None:
        artifact = Artifact(
            context_space_id=connection.context_space_id,
            source_connection_id=connection.id,
            external_id=parsed.external_id,
            artifact_type=parsed.artifact_type,
            title=parsed.title,
            canonical_url=parsed.canonical_url,
            source_updated_at=parsed.source_updated_at,
            synced_at=datetime.now(UTC),
            content_hash=parsed.content_hash,
            content_text=parsed.content_text,
            metadata_json=parsed.metadata,
        )
        db.add(artifact)
        db.flush()
    else:
        artifact.title = parsed.title
        artifact.artifact_type = parsed.artifact_type
        artifact.canonical_url = parsed.canonical_url
        artifact.source_updated_at = parsed.source_updated_at
        artifact.synced_at = datetime.now(UTC)
        artifact.content_text = parsed.content_text
        artifact.metadata_json = parsed.metadata

    if artifact.content_hash != parsed.content_hash or not artifact.evidence_fragments:
        artifact.content_hash = parsed.content_hash
        artifact.evidence_fragments.clear()
        db.flush()
        for selector, excerpt in parsed.evidence_fragments:
            artifact.evidence_fragments.append(
                EvidenceFragment(
                    selector=selector,
                    excerpt=excerpt,
                    normalized_claim=excerpt,
                    extracted_by="filesystem",
                    verification_status=EvidenceVerificationStatus.VERIFIED,
                )
            )
    db.flush()
    return artifact


def _first_evidence_id(artifact: Artifact) -> str | None:
    return artifact.evidence_fragments[0].id if artifact.evidence_fragments else None


def _status_from_frontmatter(raw_status: str | None, official_value, draft_value):
    if str(raw_status or "").upper() in {"OFFICIAL", "ACCEPTED"}:
        return official_value
    return draft_value


def upsert_concept_from_artifact(db: Session, context_space_id: str, artifact: Artifact, parsed: ParsedArtifact) -> Concept:
    canonical_name = str(parsed.frontmatter["canonical_name"])
    key = normalize_key(canonical_name)
    concept = db.scalar(
        select(Concept).where(Concept.context_space_id == context_space_id, Concept.canonical_key == key)
    )
    if concept is None:
        concept = Concept(
            context_space_id=context_space_id,
            concept_type=ConceptType[str(parsed.frontmatter.get("concept_type", "TERM")).upper()],
            canonical_name=canonical_name,
            canonical_key=key,
        )
        db.add(concept)
        db.flush()
    concept.aliases = list(parsed.frontmatter.get("aliases", []))
    concept.definition = str(parsed.frontmatter.get("definition", parsed.content_text))
    concept.status = _status_from_frontmatter(parsed.frontmatter.get("status"), ConceptStatus.OFFICIAL, ConceptStatus.DRAFT)
    first_evidence_id = _first_evidence_id(artifact)
    if first_evidence_id and first_evidence_id not in {link.evidence_fragment_id for link in concept.evidence_links}:
        concept.evidence_links.append(ConceptEvidenceLink(evidence_fragment_id=first_evidence_id))
    db.flush()
    return concept


def upsert_decision_from_artifact(db: Session, context_space_id: str, artifact: Artifact, parsed: ParsedArtifact) -> DecisionRecord:
    title = str(parsed.frontmatter["title"])
    title_key = normalize_key(title)
    decision = db.scalar(
        select(DecisionRecord).where(
            DecisionRecord.context_space_id == context_space_id,
            DecisionRecord.title_key == title_key,
        )
    )
    if decision is None:
        decision = DecisionRecord(context_space_id=context_space_id, title=title, title_key=title_key)
        db.add(decision)
        db.flush()
    decision.problem = str(parsed.frontmatter.get("problem", ""))
    decision.decision = str(parsed.frontmatter.get("decision", parsed.content_text))
    decision.rationale = str(parsed.frontmatter.get("rationale", ""))
    decision.constraints = list(parsed.frontmatter.get("constraints", []))
    decision.impact = list(parsed.frontmatter.get("impact", []))
    decision.assumptions = list(parsed.frontmatter.get("assumptions", []))
    decision.trade_offs = list(parsed.frontmatter.get("trade_offs", []))
    decision.alternatives_considered = list(parsed.frontmatter.get("alternatives_considered", []))
    decision.outcome_summary = str(parsed.frontmatter.get("outcome_summary", ""))
    decision.status = _status_from_frontmatter(parsed.frontmatter.get("status"), DecisionStatus.ACCEPTED, DecisionStatus.PROPOSED)

    first_evidence_id = _first_evidence_id(artifact)
    if first_evidence_id and first_evidence_id not in {link.evidence_fragment_id for link in decision.evidence_links}:
        decision.evidence_links.append(DecisionEvidenceLink(evidence_fragment_id=first_evidence_id))

    existing_concepts = {link.concept.canonical_key for link in decision.concept_links}
    for concept_name in parsed.frontmatter.get("concepts", []):
        concept = _require_concept(db, context_space_id, concept_name)
        if concept.canonical_key not in existing_concepts:
            decision.concept_links.append(
                DecisionConceptLink(concept_id=concept.id, relationship_type=DecisionConceptRole.ABOUT)
            )
    db.flush()
    return decision


def upsert_relation_from_artifact(db: Session, context_space_id: str, artifact: Artifact, parsed: ParsedArtifact) -> ConceptRelation:
    subject = _require_concept(db, context_space_id, str(parsed.frontmatter["subject"]))
    obj = _require_concept(db, context_space_id, str(parsed.frontmatter["object"]))
    predicate = str(parsed.frontmatter["predicate"]).upper()
    relation = db.scalar(
        select(ConceptRelation).where(
            ConceptRelation.context_space_id == context_space_id,
            ConceptRelation.subject_concept_id == subject.id,
            ConceptRelation.predicate == predicate,
            ConceptRelation.object_concept_id == obj.id,
        )
    )
    if relation is None:
        relation = ConceptRelation(
            context_space_id=context_space_id,
            subject_concept_id=subject.id,
            predicate=predicate,
            object_concept_id=obj.id,
        )
        db.add(relation)
        db.flush()
    relation.description = str(parsed.frontmatter.get("description", parsed.content_text))
    relation.status = _status_from_frontmatter(parsed.frontmatter.get("status"), RelationStatus.OFFICIAL, RelationStatus.DRAFT)

    decision_title = parsed.frontmatter.get("introduced_by_decision")
    if decision_title:
        decision = _require_decision(db, context_space_id, str(decision_title))
        relation.introduced_by_decision_id = decision.id
        if relation.id not in {link.relation_id for link in decision.relation_links}:
            decision.relation_links.append(
                DecisionRelationLink(relation_id=relation.id, relationship_type=DecisionRelationRole.INTRODUCES)
            )

    first_evidence_id = _first_evidence_id(artifact)
    if first_evidence_id and first_evidence_id not in {link.evidence_fragment_id for link in relation.evidence_links}:
        relation.evidence_links.append(RelationEvidenceLink(evidence_fragment_id=first_evidence_id))
    db.flush()
    return relation


def _require_concept(db: Session, context_space_id: str, canonical_name: str) -> Concept:
    key = normalize_key(canonical_name)
    concept = db.scalar(
        select(Concept).where(Concept.context_space_id == context_space_id, Concept.canonical_key == key)
    )
    if concept is None:
        concept = Concept(
            context_space_id=context_space_id,
            concept_type=ConceptType.TERM,
            canonical_name=canonical_name,
            canonical_key=key,
            definition="Auto-created placeholder concept during relation sync.",
            status=ConceptStatus.DRAFT,
        )
        db.add(concept)
        db.flush()
    return concept


def _require_decision(db: Session, context_space_id: str, title: str) -> DecisionRecord:
    title_key = normalize_key(title)
    decision = db.scalar(
        select(DecisionRecord).where(
            DecisionRecord.context_space_id == context_space_id,
            DecisionRecord.title_key == title_key,
        )
    )
    if decision is None:
        decision = DecisionRecord(
            context_space_id=context_space_id,
            title=title,
            title_key=title_key,
            decision="Auto-created placeholder decision during relation sync.",
            status=DecisionStatus.PROPOSED,
        )
        db.add(decision)
        db.flush()
    return decision
