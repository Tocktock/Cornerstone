from __future__ import annotations

import math
import re
from collections import OrderedDict

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, joinedload, selectinload

from cornerstone.config import Settings
from cornerstone.domain.models import (
    Artifact,
    Concept,
    ConceptRelation,
    ContextSpace,
    DecisionRecord,
    EvidenceFragment,
)
from cornerstone.domain.schemas import StructuredAnswerResponse
from cornerstone.services.ollama import OllamaClient, OllamaError
from cornerstone.services.serialization import (
    concept_read,
    decision_read,
    evidence_read,
    relation_read,
)


def _default_context_space_id(db: Session) -> str | None:
    context_space = db.scalar(select(ContextSpace).order_by(ContextSpace.created_at.asc()))
    return context_space.id if context_space else None


def answer_query(
    db: Session,
    query: str,
    context_space_id: str | None = None,
    settings: Settings | None = None,
) -> StructuredAnswerResponse:
    resolved_context_space_id = context_space_id or _default_context_space_id(db)
    if resolved_context_space_id is None:
        return StructuredAnswerResponse(
            query=query,
            summary="No context space is available yet.",
            concepts=[],
            relations=[],
            decisions=[],
            evidence=[],
        )

    pattern = f"%{query}%"
    concepts = list(
        db.scalars(
            select(Concept)
            .where(
                Concept.context_space_id == resolved_context_space_id,
                or_(Concept.canonical_name.ilike(pattern), Concept.definition.ilike(pattern)),
            )
            .order_by(Concept.canonical_name)
        )
    )
    decisions = list(
        db.scalars(
            select(DecisionRecord)
            .where(
                DecisionRecord.context_space_id == resolved_context_space_id,
                or_(DecisionRecord.title.ilike(pattern), DecisionRecord.decision.ilike(pattern)),
            )
            .order_by(DecisionRecord.created_at.desc())
        )
    )
    relations = list(
        db.scalars(
            select(ConceptRelation)
            .options(
                joinedload(ConceptRelation.subject_concept),
                joinedload(ConceptRelation.object_concept),
            )
            .where(ConceptRelation.context_space_id == resolved_context_space_id)
            .order_by(ConceptRelation.created_at.desc())
        )
    )
    if concepts:
        concept_ids = {concept.id for concept in concepts}
        relations = [
            relation
            for relation in relations
            if (
                relation.subject_concept_id in concept_ids
                or relation.object_concept_id in concept_ids
            )
        ]
    else:
        relations = relations[:5]

    evidence_map: OrderedDict[str, object] = OrderedDict()
    concept_payloads = [concept_read(db, concept) for concept in concepts[:8]]
    relation_payloads = [relation_read(db, relation) for relation in relations[:8]]
    decision_payloads = [decision_read(db, decision) for decision in decisions[:8]]

    for concept in concept_payloads:
        for evidence in concept.evidence:
            evidence_map[evidence.id] = evidence
    for relation in relation_payloads:
        for evidence in relation.evidence:
            evidence_map[evidence.id] = evidence
    for decision in decision_payloads:
        for evidence in decision.evidence:
            evidence_map[evidence.id] = evidence
    artifact_evidence = _retrieve_artifact_evidence(
        db,
        query,
        resolved_context_space_id,
        settings=settings,
        exclude_ids=set(evidence_map.keys()),
    )
    for evidence in artifact_evidence:
        evidence_map[evidence.id] = evidence
    evidence_payloads = [
        evidence for evidence in evidence_map.values() if evidence is not None
    ][:12]

    summary = _fallback_summary(
        query,
        concept_payloads,
        relation_payloads,
        decision_payloads,
        evidence_payloads,
    )
    generated_summary = _generate_ollama_summary(
        query,
        concept_payloads,
        relation_payloads,
        decision_payloads,
        evidence_payloads,
        settings,
    )
    if generated_summary and not _is_low_signal_summary(generated_summary):
        summary = generated_summary

    return StructuredAnswerResponse(
        query=query,
        summary=summary,
        concepts=concept_payloads,
        relations=relation_payloads,
        decisions=decision_payloads,
        evidence=evidence_payloads,
    )


def _build_query_terms(query: str) -> list[str]:
    terms = re.findall(r"[0-9A-Za-z가-힣]+", query.lower())
    seen: set[str] = set()
    unique_terms: list[str] = []
    for term in terms:
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        unique_terms.append(term)
    return unique_terms


def _lexical_score(
    query: str,
    terms: list[str],
    artifact: Artifact,
    evidence: EvidenceFragment,
) -> float:
    query_lower = query.lower()
    title_lower = artifact.title.lower()
    excerpt_lower = evidence.excerpt.lower()
    score = 0.0
    if query_lower and query_lower in title_lower:
        score += 6.0
    if query_lower and query_lower in excerpt_lower:
        score += 5.0
    for term in terms:
        if term in title_lower:
            score += 2.0
        if term in excerpt_lower:
            score += 1.5
    if artifact.title:
        score += min(len(terms), 3) * 0.1
    return score


def _retrieve_artifact_evidence(
    db: Session,
    query: str,
    context_space_id: str,
    *,
    settings: Settings | None,
    exclude_ids: set[str],
):
    effective_settings = settings or Settings()
    terms = _build_query_terms(query)
    candidate_artifacts = _candidate_artifacts(
        db,
        context_space_id,
        query,
        terms,
        effective_settings,
    )
    if not candidate_artifacts:
        return []

    scored_candidates: list[tuple[float, EvidenceFragment]] = []
    for artifact in candidate_artifacts:
        for evidence in artifact.evidence_fragments:
            if evidence.id in exclude_ids:
                continue
            lexical_score = _lexical_score(query, terms, artifact, evidence)
            if lexical_score <= 0:
                continue
            scored_candidates.append((lexical_score, evidence))

    if not scored_candidates:
        return []

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    scored_candidates = scored_candidates[: effective_settings.answer_candidate_evidence_limit]
    reranked_candidates = _rerank_with_ollama(query, scored_candidates, effective_settings)
    top_evidence = [
        evidence
        for _, evidence in sorted(reranked_candidates, key=lambda item: item[0], reverse=True)[
            : effective_settings.answer_max_evidence
        ]
    ]
    return [evidence_read(item) for item in top_evidence]


def _candidate_artifacts(
    db: Session,
    context_space_id: str,
    query: str,
    terms: list[str],
    settings: Settings,
) -> list[Artifact]:
    search_terms = [query.strip(), *terms]
    search_terms = [term for term in search_terms if term]

    stmt = (
        select(Artifact)
        .options(selectinload(Artifact.evidence_fragments))
        .where(Artifact.context_space_id == context_space_id)
    )
    if search_terms:
        filters = []
        for term in search_terms:
            pattern = f"%{term}%"
            filters.extend(
                [
                    Artifact.title.ilike(pattern),
                    Artifact.external_id.ilike(pattern),
                    Artifact.content_text.ilike(pattern),
                ]
            )
        stmt = stmt.where(or_(*filters))

    stmt = stmt.order_by(
        Artifact.source_updated_at.desc().nullslast(),
        Artifact.created_at.desc(),
    ).limit(settings.answer_candidate_artifact_limit)
    artifacts = list(db.scalars(stmt).unique())
    if artifacts:
        return artifacts

    fallback_stmt = (
        select(Artifact)
        .options(selectinload(Artifact.evidence_fragments))
        .where(Artifact.context_space_id == context_space_id)
        .order_by(Artifact.source_updated_at.desc().nullslast(), Artifact.created_at.desc())
        .limit(max(4, settings.answer_candidate_artifact_limit // 2))
    )
    return list(db.scalars(fallback_stmt).unique())


def _rerank_with_ollama(
    query: str,
    candidates: list[tuple[float, EvidenceFragment]],
    settings: Settings,
) -> list[tuple[float, EvidenceFragment]]:
    if not settings.ollama_enabled:
        return candidates

    try:
        client = OllamaClient(settings)
        texts = [query]
        for _, evidence in candidates:
            texts.append(f"{evidence.artifact.title}\n{evidence.excerpt[:600]}")
        embeddings = client.embed_texts(texts)
        if len(embeddings) != len(candidates) + 1:
            return candidates
        query_embedding = embeddings[0]
        reranked: list[tuple[float, EvidenceFragment]] = []
        for (lexical_score, evidence), candidate_embedding in zip(
            candidates,
            embeddings[1:],
            strict=True,
        ):
            semantic_score = _cosine_similarity(query_embedding, candidate_embedding)
            reranked.append((lexical_score + (semantic_score * 4.0), evidence))
        return reranked
    except OllamaError:
        return candidates


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _generate_ollama_summary(
    query: str,
    concepts,
    relations,
    decisions,
    evidence,
    settings: Settings | None,
) -> str | None:
    effective_settings = settings or Settings()
    if not effective_settings.ollama_enabled or not evidence:
        return None

    try:
        client = OllamaClient(effective_settings)
        return client.generate_answer_summary(
            query=query,
            concepts=[concept.canonical_name for concept in concepts[:4]],
            relations=[
                f"{relation.subject_name} {relation.predicate} {relation.object_name}"
                for relation in relations[:4]
            ],
            decisions=[decision.title for decision in decisions[:4]],
            evidence=evidence[: effective_settings.answer_prompt_evidence_limit],
        )
    except OllamaError:
        return None


def _fallback_summary(query: str, concepts, relations, decisions, evidence) -> str:
    if concepts:
        return (
            f"Found {len(concepts)} concept(s), {len(relations)} relation(s), and "
            f"{len(decisions)} decision record(s) related to '{query}'."
        )
    if evidence:
        titles = _unique_titles(evidence)
        title_preview = ", ".join(f"'{title}'" for title in titles[:3])
        return (
            f"Found source-backed material for '{query}' in {len(titles)} artifact(s), "
            f"including {title_preview}. No official curated concept or decision matched directly."
        )
    return f"No direct source-backed evidence was found for '{query}'."


def _unique_titles(evidence) -> list[str]:
    seen: set[str] = set()
    titles: list[str] = []
    for item in evidence:
        if item.artifact_title in seen:
            continue
        seen.add(item.artifact_title)
        titles.append(item.artifact_title)
    return titles


def _is_low_signal_summary(summary: str) -> bool:
    lowered = summary.lower()
    low_signal_markers = [
        "provided evidence",
        "evidence is incomplete",
        "answer cannot be provided",
        "cannot be definitively determined",
        "left as is",
        "actual values or ranges",
        "content or structure of the answer",
    ]
    return any(marker in lowered for marker in low_signal_markers)
