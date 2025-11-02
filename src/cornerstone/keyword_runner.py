"""Reusable keyword pipeline executor for background jobs and synchronous runs."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .config import Settings
from .embeddings import EmbeddingService
from .insights import KeywordInsightQueue
from .ingestion import ProjectVectorStoreManager
from .keywords import (
    ChunkPreparationResult,
    ConceptClusteringResult,
    ConceptExtractionResult,
    ConceptRankingResult,
    KeywordCandidate,
    KeywordLLMFilter,
    build_excerpt,
    cluster_concepts,
    extract_concept_candidates,
    extract_keyword_candidates,
    dedupe_concept_candidates,
    iter_candidate_batches,
    rank_concept_clusters,
    prepare_keyword_chunks,
)
from .observability import MetricsRecorder
from .projects import Project


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class KeywordRunResult:
    project_id: str
    keywords: list[dict[str, object]]
    debug: dict[str, object]
    insights: list[dict[str, object]]
    stats: dict[str, object]
    insight_job: dict[str, object] | None = None


async def execute_keyword_run(
    project: Project,
    *,
    settings: Settings,
    embedding_service: EmbeddingService,
    store_manager: ProjectVectorStoreManager,
    insight_queue: KeywordInsightQueue,
    metrics: MetricsRecorder | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> KeywordRunResult:
    payloads = list(store_manager.iter_project_payloads(project.id))
    chunk_stage: ChunkPreparationResult = prepare_keyword_chunks(payloads)
    chunks = chunk_stage.chunks
    llm_filter = KeywordLLMFilter(settings)
    concept_stage: ConceptExtractionResult = extract_concept_candidates(
        chunks,
        embedding_service=embedding_service,
        llm_filter=llm_filter,
        use_llm_summary=settings.keyword_stage2_use_llm_summary,
        max_ngram_size=settings.keyword_stage2_max_ngram,
        max_candidates_per_chunk=settings.keyword_stage2_max_candidates_per_chunk,
        max_embedding_phrases_per_chunk=settings.keyword_stage2_max_embedding_phrases_per_chunk,
        max_statistical_phrases_per_chunk=settings.keyword_stage2_max_statistical_phrases_per_chunk,
        llm_summary_max_chunks=settings.keyword_stage2_llm_summary_max_chunks,
        llm_summary_max_results=settings.keyword_stage2_llm_summary_max_results,
        llm_summary_max_chars=settings.keyword_stage2_llm_summary_max_chars,
        min_char_length=settings.keyword_stage2_min_char_length,
        min_occurrences=settings.keyword_stage2_min_occurrences,
        embedding_weight=settings.keyword_stage2_embedding_weight,
        statistical_weight=settings.keyword_stage2_statistical_weight,
        llm_weight=settings.keyword_stage2_llm_weight,
    )
    texts = [chunk.text for chunk in chunks]
    frequency_keywords = extract_keyword_candidates(texts)

    context_snippets: list[str] = [chunk.excerpt(max_chars=400) for chunk in chunks[:5]]

    total_tokens = chunk_stage.total_tokens()
    processed_chunks = chunk_stage.processed_count
    stage2_candidate_total = len(concept_stage.candidates)

    batch_size_config = max(0, settings.keyword_candidate_batch_size)
    batch_overlap = max(0, settings.keyword_candidate_batch_overlap)
    min_batch_size = max(1, settings.keyword_candidate_min_batch_size)
    candidate_limit = max(0, settings.keyword_llm_max_candidates)

    if candidate_limit:
        if batch_size_config <= 0 or batch_size_config > candidate_limit:
            batch_size_config = candidate_limit
        min_batch_size = min(min_batch_size, batch_size_config)
        batch_overlap = min(batch_overlap, max(0, batch_size_config - 1))

    if stage2_candidate_total >= min_batch_size and batch_size_config < min_batch_size:
        max_allowed = candidate_limit or stage2_candidate_total
        batch_size_config = min(min_batch_size, max_allowed)

    if batch_size_config <= 0:
        batch_size_config = stage2_candidate_total or 1
    if batch_size_config < 1:
        batch_size_config = 1
    if batch_overlap >= batch_size_config:
        batch_overlap = max(0, batch_size_config - 1)

    if stage2_candidate_total <= batch_size_config:
        batch_total = 1 if stage2_candidate_total else 0
    else:
        step = max(1, batch_size_config - batch_overlap)
        batch_total = math.ceil(max(0, stage2_candidate_total - batch_size_config) / step) + 1
    if stage2_candidate_total and batch_total == 0:
        batch_total = 1

    llm_active = llm_filter.enabled and stage2_candidate_total > 0
    llm_bypass_reason: str | None = None
    llm_bypass_details: dict[str, object] = {}
    if llm_active:
        token_limit = max(0, settings.keyword_llm_max_tokens)
        chunk_limit = max(0, settings.keyword_llm_max_chunks)
        if token_limit and total_tokens > token_limit:
            llm_bypass_reason = "token-limit"
            llm_bypass_details = {
                "token_limit": token_limit,
                "token_total": total_tokens,
            }
        elif chunk_limit and processed_chunks > chunk_limit:
            llm_bypass_reason = "chunk-limit"
            llm_bypass_details = {
                "chunk_limit": chunk_limit,
                "chunk_total": processed_chunks,
            }

        if llm_bypass_reason is not None:
            llm_active = False
            details = {
                "candidate_count": stage2_candidate_total,
                "token_total": total_tokens,
                "chunk_total": processed_chunks,
            }
            details.update(llm_bypass_details)
            llm_filter.record_bypass("concept", llm_bypass_reason, **details)

    refined_candidates: list[ConceptCandidate] = []
    candidates_processed = 0
    batches_completed = 0
    last_batch_duration = 0.0
    poll_after_ms: int | None = None

    if stage2_candidate_total:
        if batch_total <= 1:
            batch_iterable: Iterable[list[ConceptCandidate]] = [list(concept_stage.candidates)]
        else:
            batch_iterable = iter_candidate_batches(
                concept_stage.candidates,
                batch_size=batch_size_config,
                overlap=batch_overlap,
            )
    else:
        batch_iterable = []

    if llm_active:
        seen_stage2_signatures: set[tuple[str, tuple[str, ...]]] = set()
        seen_refined_signatures: set[tuple[str, tuple[str, ...]]] = set()
        for idx, batch in enumerate(batch_iterable, start=1):
            if not batch:
                continue
            batch_start = time.perf_counter()
            refined_batch = llm_filter.refine_concepts(batch, context_snippets)
            if refined_batch:
                refined_candidates.extend(refined_batch)
                current_batch = refined_batch
            else:
                refined_candidates.extend(batch)
                current_batch = batch

            stage_signatures = {
                (candidate.phrase.lower(), tuple(sorted(candidate.chunk_ids)) if candidate.chunk_ids else ())
                for candidate in batch
            }
            refined_signatures = {
                (candidate.phrase.lower(), tuple(sorted(candidate.chunk_ids)) if candidate.chunk_ids else ())
                for candidate in current_batch
            }

            missing_signatures = stage_signatures - refined_signatures

            for candidate in current_batch:
                signature = (candidate.phrase.lower(), tuple(sorted(candidate.chunk_ids)) if candidate.chunk_ids else ())
                if signature not in seen_refined_signatures:
                    seen_refined_signatures.add(signature)
                    seen_stage2_signatures.add(signature)
                    candidates_processed = min(stage2_candidate_total, candidates_processed + 1)

            for signature in missing_signatures:
                if signature not in seen_stage2_signatures:
                    seen_stage2_signatures.add(signature)
                    candidates_processed = min(stage2_candidate_total, candidates_processed + 1)

            batches_completed = idx
            batch_duration = max(time.perf_counter() - batch_start, 0.0)
            last_batch_duration = batch_duration
            poll_after_ms = min(5000, max(1500, int(batch_duration * 1500 + 500)))
            progress_payload = {
                "batch_total": batch_total or 1,
                "batches_completed": batches_completed,
                "candidates_processed": candidates_processed,
                "candidate_total": stage2_candidate_total,
            }
            if poll_after_ms is not None:
                progress_payload["poll_after"] = round(poll_after_ms / 1000.0, 2)
                progress_payload["poll_after_ms"] = poll_after_ms
            if batch_duration:
                progress_payload["last_batch_duration_ms"] = round(batch_duration * 1000.0, 2)
            if progress_callback:
                progress_callback(progress_payload)
            if metrics:
                metrics.increment(
                    "keyword.batch.processed",
                    project_id=project.id,
                    batch_index=idx,
                    batch_total=batch_total or 1,
                )
                metrics.record_timing(
                    "keyword.batch.duration",
                    batch_duration,
                    project_id=project.id,
                    batch_index=idx,
                )
        if batches_completed == 0 and stage2_candidate_total:
            candidates_processed = stage2_candidate_total
            batches_completed = batch_total or 1
    else:
        refined_candidates = list(concept_stage.candidates)
        if stage2_candidate_total:
            candidates_processed = stage2_candidate_total
            batches_completed = batch_total or 1
            if progress_callback:
                progress_callback(
                    {
                        "batch_total": batch_total or 1,
                        "batches_completed": batches_completed,
                        "candidates_processed": candidates_processed,
                        "candidate_total": stage2_candidate_total,
                    }
                )

    if refined_candidates:
        candidates_for_stage3 = dedupe_concept_candidates(refined_candidates)
    else:
        candidates_for_stage3 = dedupe_concept_candidates(concept_stage.candidates)
    concept_stage = concept_stage.replace_candidates(candidates_for_stage3)

    batching_debug: dict[str, object] = {
        "enabled": bool(llm_active and batch_total and batch_total > 1),
        "batch_size": batch_size_config,
        "batch_overlap": batch_overlap,
        "batch_total": batch_total,
        "batches_completed": batches_completed,
        "candidates_processed": candidates_processed,
        "candidate_total": stage2_candidate_total,
        "llm_active": llm_active,
    }
    if poll_after_ms is not None:
        batching_debug["poll_after_ms"] = poll_after_ms
    if last_batch_duration:
        batching_debug["last_batch_duration_ms"] = round(last_batch_duration * 1000.0, 2)

    cluster_stage: ConceptClusteringResult = cluster_concepts(
        concept_stage.candidates,
        embedding_service=embedding_service,
        llm_filter=llm_filter if settings.keyword_stage3_label_clusters and llm_active else None,
        llm_label_max_clusters=settings.keyword_stage3_label_max_clusters,
    )
    if (
        settings.keyword_stage3_label_clusters
        and llm_filter.enabled
        and not llm_active
        and llm_bypass_reason
    ):
        cluster_payload = dict(llm_bypass_details)
        cluster_payload.setdefault("candidate_count", len(concept_stage.candidates))
        cluster_payload.setdefault("cluster_count", len(cluster_stage.clusters))
        llm_filter.record_bypass("cluster", llm_bypass_reason, **cluster_payload)

    ranking_stage: ConceptRankingResult = rank_concept_clusters(
        cluster_stage.clusters,
        core_limit=settings.keyword_stage4_core_limit,
        max_results=settings.keyword_stage4_max_results,
        score_weight=settings.keyword_stage4_score_weight,
        document_weight=settings.keyword_stage4_document_weight,
        chunk_weight=settings.keyword_stage4_chunk_weight,
        occurrence_weight=settings.keyword_stage4_occurrence_weight,
        label_bonus=settings.keyword_stage4_label_bonus,
    )

    keywords_origin = "frequency"
    if ranking_stage.ranked:
        keywords_origin = "stage4"

        if (
            settings.keyword_stage5_harmonize_enabled
            and llm_active
        ):
            harmonized = llm_filter.harmonize_ranked_concepts(
                ranking_stage.ranked,
                max_results=settings.keyword_stage5_harmonize_max_results,
            )
            if harmonized and harmonized != list(ranking_stage.ranked):
                ranking_stage = ranking_stage.replace_ranked(harmonized)
                keywords_origin = "stage5"
        elif settings.keyword_stage5_harmonize_enabled and llm_filter.enabled and llm_bypass_reason:
            harmonize_payload = dict(llm_bypass_details)
            harmonize_payload.setdefault("candidate_count", len(ranking_stage.ranked))
            llm_filter.record_bypass("harmonize", llm_bypass_reason, **harmonize_payload)

        keywords = [
            KeywordCandidate(
                term=item.label,
                count=max(item.document_count, item.chunk_count, 1),
                is_core=item.is_core,
                generated=item.generated,
                reason=item.description
                or f"{item.document_count} docs | score {item.score:.2f}",
                source=f"{keywords_origin}:{item.label_source}",
            )
            for item in ranking_stage.ranked
        ]
    else:
        keywords = frequency_keywords

    original_count = len(keywords)
    if llm_active:
        keywords = llm_filter.filter_keywords(keywords, context_snippets)
        logger.info(
            "keyword.llm.apply backend=%s project=%s before=%s after=%s",
            llm_filter.backend,
            project.id,
            original_count,
            len(keywords),
        )
    elif llm_filter.enabled and llm_bypass_reason:
        filter_payload = dict(llm_bypass_details)
        filter_payload.setdefault("candidate_count", original_count)
        llm_filter.record_bypass("filter", llm_bypass_reason, **filter_payload)
    else:
        logger.info(
            "keyword.llm.bypass backend=%s project=%s candidate_count=%s",
            llm_filter.backend,
            project.id,
            original_count,
        )

    if not keywords and concept_stage.candidates:
        fallback_candidates = []
        for candidate in concept_stage.candidates[: settings.keyword_stage2_max_candidates_per_chunk]:
            score_as_int = max(1, int(round(candidate.score)))
            fallback_candidates.append(
                KeywordCandidate(
                    term=candidate.phrase,
                    count=max(candidate.document_count, score_as_int),
                    is_core=candidate.document_count > 1,
                    generated=False,
                    reason="stage2-fallback",
                    source="stage2",
                )
            )
        keywords = fallback_candidates
        keywords_origin = "stage2-fallback"

    chunk_debug: dict[str, object] = {
        "payloads_total": chunk_stage.total_payloads,
        "processed": chunk_stage.processed_count,
        "skipped_empty": chunk_stage.skipped_empty,
        "skipped_non_text": chunk_stage.skipped_non_text,
        "languages": chunk_stage.unique_languages(),
    }
    if total_tokens:
        chunk_debug["total_tokens"] = total_tokens
    sample_sections = chunk_stage.sample_sections()
    if sample_sections:
        chunk_debug["sample_sections"] = sample_sections
    sample_excerpts = chunk_stage.sample_excerpts(limit=2, max_chars=160)
    if sample_excerpts:
        chunk_debug["sample_excerpts"] = sample_excerpts

    concept_debug = concept_stage.to_debug_payload(limit=8)
    concept_debug["batching"] = batching_debug
    concept_llm_debug = llm_filter.concept_debug_payload()
    if concept_llm_debug:
        concept_debug["llm"] = concept_llm_debug
    summary_debug = llm_filter.summary_debug_payload()
    if summary_debug:
        concept_debug["llm_summary"] = summary_debug
    cluster_debug = cluster_stage.to_debug_payload(limit=6)
    cluster_llm_debug = llm_filter.cluster_debug_payload()
    if cluster_llm_debug:
        cluster_debug["llm"] = cluster_llm_debug

    ranking_debug = ranking_stage.to_debug_payload(limit=6)
    ranking_debug["origin"] = keywords_origin
    harmonize_debug = llm_filter.harmonize_debug_payload()
    if harmonize_debug:
        ranking_debug.setdefault("llm", harmonize_debug)

    insights: list[dict[str, object]] = []
    stage7_debug: dict[str, object] | None = None
    insight_job_payload: dict[str, object] | None = None
    should_summarize = (
        keywords
        and settings.keyword_stage7_summary_enabled
        and settings.keyword_stage7_summary_max_insights > 0
        and settings.keyword_stage7_summary_max_concepts > 0
        and llm_active
    )

    max_summary_keywords = settings.keyword_stage7_summary_max_concepts * 4
    if should_summarize and max_summary_keywords > 0 and len(keywords) > max_summary_keywords:
        should_summarize = False
        stage7_debug = {
            "reason": "skipped",
            "cause": "keyword-limit-exceeded",
            "total_keywords": len(keywords),
            "limit": max_summary_keywords,
            "enabled": settings.keyword_stage7_summary_enabled,
            "llm_enabled": llm_filter.enabled,
            "backend": llm_filter.backend,
            "max_insights": settings.keyword_stage7_summary_max_insights,
            "max_concepts": settings.keyword_stage7_summary_max_concepts,
        }

    if should_summarize:
        try:
            insight_job = await insight_queue.enqueue(
                project_id=project.id,
                settings=settings,
                keywords=keywords,
                max_insights=settings.keyword_stage7_summary_max_insights,
                max_concepts=settings.keyword_stage7_summary_max_concepts,
                context_snippets=context_snippets,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "keyword.stage7.queue_failed project=%s error=%s",
                project.id,
                exc,
            )
            stage7_debug = {
                "reason": "error",
                "error": str(exc),
                "enabled": settings.keyword_stage7_summary_enabled,
                "llm_enabled": llm_filter.enabled,
                "backend": llm_filter.backend,
            }
        else:
            inline_timeout = max(0.0, settings.keyword_stage7_summary_inline_timeout)
            completed = await insight_job.wait(timeout=inline_timeout)
            insight_job_payload = insight_job.to_payload(include_result=False)

            llm_debug = dict(insight_job.debug or {})
            if not llm_debug:
                llm_debug = {
                    "backend": llm_filter.backend,
                    "enabled": settings.keyword_stage7_summary_enabled,
                    "max_insights": settings.keyword_stage7_summary_max_insights,
                    "max_concepts": settings.keyword_stage7_summary_max_concepts,
                    "status": "pending",
                }
            else:
                llm_debug.setdefault("backend", llm_filter.backend)
                llm_debug.setdefault("enabled", settings.keyword_stage7_summary_enabled)
                llm_debug.setdefault("max_insights", settings.keyword_stage7_summary_max_insights)
                llm_debug.setdefault("max_concepts", settings.keyword_stage7_summary_max_concepts)

            reason: str
            if insight_job.status == "success":
                insights = insight_job.insights or []
                llm_debug["status"] = "success"
                llm_debug.setdefault("selected_total", len(insights))
                reason = "summarized"
            elif insight_job.status == "error":
                llm_debug["status"] = "error"
                reason = "error"
            elif insight_job.status == "running":
                llm_debug["status"] = "running"
                reason = "running"
            else:
                llm_debug["status"] = insight_job.status
                reason = "queued"

            stage7_debug = {
                "reason": reason,
                "status": insight_job.status,
                "job_id": insight_job.id,
                "enabled": settings.keyword_stage7_summary_enabled,
                "llm_enabled": llm_filter.enabled,
                "max_insights": settings.keyword_stage7_summary_max_insights,
                "max_concepts": settings.keyword_stage7_summary_max_concepts,
                "llm": llm_debug,
            }
            if insight_job.status == "success" and insights:
                stage7_debug["insights"] = insights[: min(len(insights), 5)]
            if insight_job.status == "error" and insight_job.error:
                stage7_debug["error"] = insight_job.error
            if insight_job.status in {"pending", "running"} or not completed:
                stage7_debug["poll_after"] = settings.keyword_stage7_summary_poll_interval

            if insight_job.status == "success":
                insight_job_payload = insight_job.to_payload(include_result=False)

    elif stage7_debug is None:
        stage7_debug = {
            "reason": "disabled",
            "enabled": settings.keyword_stage7_summary_enabled,
            "max_insights": settings.keyword_stage7_summary_max_insights,
            "max_concepts": settings.keyword_stage7_summary_max_concepts,
            "llm_enabled": llm_filter.enabled,
            "backend": llm_filter.backend,
        }
        if llm_filter.enabled and llm_bypass_reason:
            stage7_debug["bypass_reason"] = llm_bypass_reason
            for key, value in llm_bypass_details.items():
                stage7_debug.setdefault(key, value)

    if not should_summarize and llm_filter.enabled and llm_bypass_reason:
        summary_payload = dict(llm_bypass_details)
        summary_payload.setdefault("candidate_count", len(keywords))
        llm_filter.record_bypass("summary", llm_bypass_reason, **summary_payload)

    llm_debug_payload = llm_filter.debug_payload()

    debug_payload = {
        **llm_debug_payload,
        "chunking": chunk_debug,
        "stage2": concept_debug,
        "stage3": cluster_debug,
        "stage4": ranking_debug,
    }
    debug_payload.setdefault("candidate_count", stage2_candidate_total)
    if stage7_debug:
        debug_payload["stage7"] = stage7_debug
    if insight_job_payload and stage7_debug and "poll_after" in stage7_debug:
        insight_job_payload.setdefault("poll_after", stage7_debug["poll_after"])

    logger.info(
        "keyword.llm.summary project=%s backend=%s details=%s",
        project.id,
        debug_payload.get("backend"),
        debug_payload,
    )

    keyword_dicts = [
        {
            "term": item.term,
            "count": item.count,
            "core": item.is_core,
            "generated": item.generated,
            "reason": item.reason,
            "source": item.source,
        }
        for item in keywords
    ]

    final_keyword_total = len(keyword_dicts)
    stats = {
        "candidate_total": stage2_candidate_total,
        "stage2_candidate_total": stage2_candidate_total,
        "keywords_total": final_keyword_total,
        "chunk_total": processed_chunks,
        "token_total": total_tokens,
        "llm_backend": llm_filter.backend,
        "batch_total": batch_total,
        "batches_completed": batches_completed,
        "candidates_processed": candidates_processed,
        "batch_size": batch_size_config,
        "batch_overlap": batch_overlap,
        "llm_active": llm_active,
    }
    if poll_after_ms is not None:
        stats["poll_after_ms"] = poll_after_ms
        stats["poll_after"] = round(poll_after_ms / 1000.0, 2)
    if last_batch_duration:
        stats["last_batch_duration_ms"] = round(last_batch_duration * 1000.0, 2)
    if llm_bypass_reason:
        stats["bypass_reason"] = llm_bypass_reason
        stats.update(llm_bypass_details)

    if metrics:
        metrics.increment("keyword.run.completed", project_id=project.id)

    return KeywordRunResult(
        project_id=project.id,
        keywords=keyword_dicts,
        debug=debug_payload,
        insights=insights,
        stats=stats,
        insight_job=insight_job_payload,
    )
