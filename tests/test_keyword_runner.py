from __future__ import annotations

import json
from pathlib import Path

import pytest

from cornerstone.config import Settings
from cornerstone.keyword_runner import execute_keyword_run
from cornerstone.keywords import ConceptCandidate, KeywordLLMFilter, iter_candidate_batches
from cornerstone.insights import KeywordInsightQueue
from cornerstone.projects import Project


class FakeStoreManager:
    def __init__(self, payloads):
        self._payloads = payloads

    def iter_project_payloads(self, project_id: str):
        return list(self._payloads)


class FakeEmbeddingService:
    dimension = 3

    def embed_one(self, text: str):
        length = float(len(text)) or 1.0
        return [length, length / 2.0, 1.0]

    def embed(self, texts):
        return [self.embed_one(text) for text in texts]


def test_iter_candidate_batches_respects_overlap():
    candidates = [
        ConceptCandidate(
            phrase=f"term-{idx}",
            score=1.0,
            occurrences=1,
            document_count=1,
            chunk_count=1,
            average_occurrence_per_chunk=1.0,
            word_count=1,
            languages=["en"],
            sections=[],
            sources=[],
            sample_snippet=None,
        )
        for idx in range(5)
    ]

    batches = list(iter_candidate_batches(candidates, batch_size=2, overlap=1))
    assert len(batches) == 4
    assert [item.phrase for item in batches[0]] == ["term-0", "term-1"]
    assert [item.phrase for item in batches[-1]] == ["term-3", "term-4"]


@pytest.mark.asyncio
async def test_execute_keyword_run_batches_and_reports_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        data_dir=str(data_dir),
        default_project_name="Test Project",
        keyword_stage7_summary_enabled=False,
        keyword_candidate_batch_size=1,
        keyword_candidate_batch_overlap=0,
        keyword_candidate_min_batch_size=1,
        chat_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="keyword-test",
    )

    def fake_invoke_backend(self, prompt: str) -> str:
        return json.dumps({"keywords": [], "concepts": []})

    monkeypatch.setattr(KeywordLLMFilter, "_invoke_backend", fake_invoke_backend)

    payloads = [
        {
            "text": (
                "Login failure troubleshooting playbook with login retries, login audit logs, and login analytics. "
                "Escalation guide covers authentication failures and MFA reset workflows."
            ),
            "language": "en",
            "chunk_id": "doc-1:0",
            "doc_id": "doc-1",
        },
        {
            "text": (
                "API latency alert handling with dashboards, latency regression analysis, latency feedback loops, "
                "and runbooks for scaling API pods."
            ),
            "language": "en",
            "chunk_id": "doc-2:0",
            "doc_id": "doc-2",
        },
    ]

    store_manager = FakeStoreManager(payloads)
    embedding_service = FakeEmbeddingService()
    insight_queue = KeywordInsightQueue(max_jobs=1)
    project = Project(
        id="proj-1",
        name="Demo",
        description=None,
        created_at="2025-01-01T00:00:00+00:00",
    )

    progress_events: list[dict[str, object]] = []

    def capture_progress(payload: dict[str, object]) -> None:
        progress_events.append(dict(payload))

    result = await execute_keyword_run(
        project,
        settings=settings,
        embedding_service=embedding_service,
        store_manager=store_manager,  # type: ignore[arg-type]
        insight_queue=insight_queue,
        metrics=None,
        progress_callback=capture_progress,
    )

    stats = result.stats
    assert stats["stage2_candidate_total"] >= stats["batch_total"] >= 1
    assert stats["candidates_processed"] == stats["stage2_candidate_total"]
    assert stats["batches_completed"] == stats["batch_total"]
    assert progress_events, "Expected progress callback to be invoked for batched run"
    final_progress = progress_events[-1]
    assert final_progress["candidates_processed"] == stats["stage2_candidate_total"]
    assert final_progress["batches_completed"] == stats["batches_completed"]
    assert stats["keywords_total"] == len(result.keywords)
