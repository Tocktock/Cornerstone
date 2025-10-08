"""Support agent chat service with switchable backends."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
import json
from typing import Iterable, Iterator, List, Sequence, Tuple

from .config import Settings
from .embeddings import EmbeddingService
from .glossary import Glossary
from .projects import Project
from .personas import PersonaOverrides, PersonaSnapshot, PersonaStore
from .vector_store import QdrantVectorStore, SearchResult
from .observability import MetricsRecorder
from .reranker import Reranker

logger = logging.getLogger(__name__)


try:  # pragma: no cover - optional import for Ollama HTTP client
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

try:  # pragma: no cover - guard openai import for static analysis
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


@dataclass(slots=True)
class SupportAgentResponse:
    message: str
    sources: list[dict[str, str]]
    definitions: list[str]


@dataclass(slots=True)
class SupportAgentContext:
    """Precomputed retrieval and glossary context for an agent response."""

    prompt: str
    sources: list[dict[str, str]]
    definitions: list[str]


@dataclass(slots=True)
class PersonaRuntimeOptions:
    retrieval_top_k: int
    glossary_top_k: int
    chat_temperature: float
    chat_max_tokens: int | None


class SupportAgentService:
    """Generate support-oriented answers using retrieval augmented generation."""

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        store_manager,
        glossary: Glossary,
        *,
        retrieval_top_k: int | None = None,
        persona_store: PersonaStore | None = None,
        fts_index=None,
        metrics: MetricsRecorder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._settings = settings
        self._embedding = embedding_service
        self._stores = store_manager
        self._glossary = glossary
        default_top_k = retrieval_top_k if retrieval_top_k and retrieval_top_k > 0 else settings.retrieval_top_k
        self._retrieval_top_k = max(1, default_top_k)
        self._openai_client: OpenAI | None = None
        self._persona_store = persona_store
        self._fts = fts_index
        self._metrics = metrics
        self._reranker = reranker
        self._default_chat_temperature = max(0.0, settings.chat_temperature)
        default_max_tokens = settings.chat_max_tokens
        if default_max_tokens is not None and default_max_tokens <= 0:
            default_max_tokens = None
        self._default_chat_max_tokens = default_max_tokens

    def generate(
        self,
        project: Project,
        query: str,
        *,
        conversation: Sequence[str] | None = None,
    ) -> SupportAgentResponse:
        metrics = self._metrics
        generate_start = time.perf_counter() if metrics else None
        persona = self._resolve_persona(project)
        options = self._persona_options(persona)
        context, _ = self._build_context(project, persona, query, conversation, options)
        answer = self._invoke_backend(
            context.prompt,
            temperature=options.chat_temperature,
            max_tokens=options.chat_max_tokens,
        )
        logger.info(
            "support.generate.completed backend=%s chars=%s sources=%s",
            self._settings.chat_backend,
            len(answer),
            len(context.sources),
        )
        if metrics and generate_start is not None:
            metrics.record_timing(
                "chat.generate_duration",
                time.perf_counter() - generate_start,
                project_id=project.id,
                backend=self._settings.chat_backend,
                sources=len(context.sources),
            )
            metrics.increment(
                "chat.responses",
                project_id=project.id,
                backend=self._settings.chat_backend,
            )
        return SupportAgentResponse(message=answer, sources=context.sources, definitions=context.definitions)

    def stream_generate(
        self,
        project: Project,
        query: str,
        *,
        conversation: Sequence[str] | None = None,
    ) -> Tuple[SupportAgentContext, Iterable[str]]:
        """Stream a response for the given query, yielding incremental text deltas."""

        persona = self._resolve_persona(project)
        options = self._persona_options(persona)
        context, _ = self._build_context(project, persona, query, conversation, options)
        if self._metrics:
            self._metrics.increment(
                "chat.stream_requests",
                project_id=project.id,
                backend=self._settings.chat_backend,
            )
        stream = self._stream_backend(
            context.prompt,
            temperature=options.chat_temperature,
            max_tokens=options.chat_max_tokens,
        )
        return context, stream

    # Internal helpers -------------------------------------------------

    @staticmethod
    def _is_korean(text: str) -> bool:
        return any("\uAC00" <= char <= "\uD7A3" for char in text)

    def _persona_options(self, persona: PersonaSnapshot) -> PersonaRuntimeOptions:
        retrieval_top_k = persona.retrieval_top_k
        if retrieval_top_k is None or retrieval_top_k <= 0:
            retrieval_top_k = self._retrieval_top_k
        glossary_top_k = persona.glossary_top_k
        if glossary_top_k is None:
            glossary_top_k = self._settings.glossary_top_k
        glossary_top_k = max(0, glossary_top_k)
        chat_temperature = persona.chat_temperature
        if chat_temperature is None:
            chat_temperature = self._default_chat_temperature
        chat_temperature = max(0.0, chat_temperature)
        chat_max_tokens = persona.chat_max_tokens
        if chat_max_tokens is None:
            chat_max_tokens = self._default_chat_max_tokens
        if chat_max_tokens is not None and chat_max_tokens <= 0:
            chat_max_tokens = None
        return PersonaRuntimeOptions(
            retrieval_top_k=max(1, retrieval_top_k),
            glossary_top_k=glossary_top_k,
            chat_temperature=chat_temperature,
            chat_max_tokens=chat_max_tokens,
        )

    def _build_context(
        self,
        project: Project,
        persona: PersonaSnapshot,
        query: str,
        conversation: Sequence[str] | None,
        options: PersonaRuntimeOptions | None = None,
    ) -> Tuple[SupportAgentContext, list[dict]]:
        options = options or self._persona_options(persona)
        history = list(conversation or [])
        metrics = self._metrics
        logger.info(
            "support.generate.start backend=%s embedding=%s project=%s query=%s retrieval_k=%s glossary_k=%s temperature=%.2f max_tokens=%s",
            self._settings.chat_backend,
            self._settings.embedding_model,
            project.id,
            query,
            options.retrieval_top_k,
            options.glossary_top_k,
            options.chat_temperature,
            options.chat_max_tokens,
        )
        embed_start = time.perf_counter() if metrics else None
        vector = self._embedding.embed_one(query)
        if metrics and embed_start is not None:
            metrics.record_timing(
                "retrieval.embedding_duration",
                time.perf_counter() - embed_start,
                project_id=project.id,
            )
        store = self._stores.get_store(project.id)
        vector_start = time.perf_counter() if metrics else None
        vector_results = list(store.search(vector, limit=options.retrieval_top_k))
        if metrics and vector_start is not None:
            metrics.record_timing(
                "retrieval.vector_duration",
                time.perf_counter() - vector_start,
                project_id=project.id,
                hits=len(vector_results),
            )
            metrics.increment("retrieval.vector_queries", project_id=project.id)
        vector_chunks = self._vector_results_to_chunks(vector_results)
        keyword_chunks: list[dict] = []
        if self._fts is not None:
            keyword_start = time.perf_counter() if metrics else None
            keyword_hits = self._fts.search(project.id, query, limit=options.retrieval_top_k * 2)
            if metrics and keyword_start is not None:
                metrics.record_timing(
                    "retrieval.keyword_duration",
                    time.perf_counter() - keyword_start,
                    project_id=project.id,
                    hits=len(keyword_hits),
                )
                metrics.increment("retrieval.keyword_queries", project_id=project.id)
            keyword_chunks = self._keyword_hits_to_chunks(keyword_hits)
        max_chunks = max(options.retrieval_top_k * 2, options.retrieval_top_k)
        fused_chunks = self._fuse_chunks(vector_chunks, keyword_chunks, limit=max_chunks)
        reranked_chunks = self._apply_reranker(
            project,
            query,
            query_vector=vector,
            fused_chunks=fused_chunks,
            limit=max_chunks,
        )
        if reranked_chunks is not None:
            fused_chunks = reranked_chunks
        if metrics:
            metrics.increment(
                "retrieval.fused_chunks",
                value=len(fused_chunks),
                project_id=project.id,
            )
        top_titles = [chunk.get("title") for chunk in fused_chunks]
        logger.info(
            "support.generate.matches project=%s vector=%s keyword=%s fused=%s titles=%s",
            project.id,
            len(vector_chunks),
            len(keyword_chunks),
            len(fused_chunks),
            top_titles,
        )
        sources = self._format_sources(fused_chunks)
        definitions = self._collect_definitions(query, top_k=options.glossary_top_k)
        prompt = self._build_prompt(project, persona, query, fused_chunks, history)
        context = SupportAgentContext(prompt=prompt, sources=sources, definitions=definitions)
        return context, fused_chunks

    def _build_prompt(
        self,
        project: Project,
        persona: PersonaSnapshot,
        query: str,
        chunks: Iterable[dict],
        conversation: Sequence[str],
    ) -> str:
        context_lines = ["Relevant documentation snippets:"]
        for idx, chunk in enumerate(chunks, start=1):
            heading_path = chunk.get("heading_path") or []
            title = chunk.get("title") or "Untitled snippet"
            if heading_path:
                title = " · ".join(heading_path)
            text = chunk.get("text") or ""
            context_lines.append(f"[{idx}] {title}\n{text}\n---")
        if len(context_lines) == 1:
            context_lines = ["No matching documentation was found."]

        conversation_history = "\n".join(conversation)
        glossary_section = self._glossary.to_prompt_section(query, self._settings.glossary_top_k)

        language_instruction = (
            "모든 답변은 한국어로 작성하세요."
            if self._is_korean(query) or any(self._is_korean(turn) for turn in conversation)
            else "Write the full response in English."
        )

        instructions = self._persona_instructions(project, persona, language_instruction)

        prompt_sections = [
            f"INSTRUCTIONS:\n{instructions}",
            "CURRENT CONVERSATION:",
            conversation_history or "(no prior conversation)",
            f"CUSTOMER QUERY:\n{query}",
            "GLOSSARY:\n" + glossary_section if glossary_section else "",
            "CONTEXT:\n" + "\n".join(context_lines),
            "RESPONSE FORMAT:\nProvide a concise problem summary, numbered resolution steps, and indicate if escalation is required.",
        ]

        return "\n\n".join(section for section in prompt_sections if section)

    def _collect_definitions(self, query: str, top_k: int) -> List[str]:
        if top_k <= 0:
            return []
        matches = self._glossary.top_matches(query, top_k)
        return [f"{entry.term}: {entry.definition}" for entry in matches]

    def _persona_instructions(
        self,
        project: Project,
        persona: PersonaSnapshot,
        language_instruction: str,
    ) -> str:
        segments: list[str] = []

        if persona.name:
            name = persona.name.strip()
            if name:
                project_name = project.name or "this project"
                segments.append(f"You are {name}, the designated support agent for {project_name}.")

        base_prompt = persona.system_prompt or (
            "You are a support agent helping customers diagnose and resolve issues. "
            "Ask for missing details when necessary, provide step-by-step guidance, and when uncertain, "
            "suggest escalating to a human agent."
        )
        if base_prompt:
            segments.append(base_prompt.strip())

        if persona.tone:
            tone = persona.tone.strip()
            if tone:
                segments.append(f"Maintain a {tone} tone throughout the conversation.")

        segments.append(language_instruction)

        return " ".join(segment for segment in segments if segment)

    def _resolve_persona(self, project: Project) -> PersonaSnapshot:
        overrides = getattr(project, "persona_overrides", None)
        if isinstance(overrides, dict):  # defensive guard during legacy migrations
            overrides = PersonaOverrides(
                name=overrides.get("name"),
                tone=overrides.get("tone"),
                system_prompt=overrides.get("system_prompt"),
                avatar_url=overrides.get("avatar_url"),
                glossary_top_k=overrides.get("glossary_top_k"),
                retrieval_top_k=overrides.get("retrieval_top_k"),
                chat_temperature=overrides.get("chat_temperature"),
                chat_max_tokens=overrides.get("chat_max_tokens"),
            )

        if self._persona_store is not None:
            return self._persona_store.resolve_persona(getattr(project, "persona_id", None), overrides)

        legacy_persona = getattr(project, "persona", None)
        if legacy_persona is not None:
            overrides = PersonaOverrides(
                name=getattr(legacy_persona, "name", None),
                tone=getattr(legacy_persona, "tone", None),
                system_prompt=getattr(legacy_persona, "system_prompt", None),
                avatar_url=getattr(legacy_persona, "avatar_url", None),
                glossary_top_k=getattr(legacy_persona, "glossary_top_k", None),
                retrieval_top_k=getattr(legacy_persona, "retrieval_top_k", None),
                chat_temperature=getattr(legacy_persona, "chat_temperature", None),
                chat_max_tokens=getattr(legacy_persona, "chat_max_tokens", None),
            )
        overrides = overrides or PersonaOverrides()
        return PersonaSnapshot(
            id=None,
            name=overrides.name,
            tone=overrides.tone,
            system_prompt=overrides.system_prompt,
            avatar_url=overrides.avatar_url,
            overrides=overrides,
            glossary_top_k=overrides.glossary_top_k,
            retrieval_top_k=overrides.retrieval_top_k,
            chat_temperature=overrides.chat_temperature,
            chat_max_tokens=overrides.chat_max_tokens,
        )

    def _format_sources(self, chunks: Iterable[dict]) -> list[dict[str, str]]:
        formatted: list[dict[str, str]] = []
        for chunk in chunks:
            title = chunk.get("title") or chunk.get("source") or "Snippet"
            text = chunk.get("text") or ""
            snippet = chunk.get("summary") or text[:300]
            formatted.append(
                {
                    "title": title,
                    "snippet": snippet,
                    "source": chunk.get("source") or "",
                    "score": f"{chunk.get('score', 0.0):.3f}",
                    "origin": ", ".join(chunk.get("origin", [])),
                }
            )
        return formatted

    def _vector_results_to_chunks(self, results: Sequence[SearchResult]) -> list[dict]:
        chunks: list[dict] = []
        for rank, result in enumerate(results, start=1):
            payload = result.payload or {}
            text = payload.get("text") or ""
            if not text:
                continue
            chunk_id = payload.get("chunk_id") or f"{payload.get('doc_id')}:{payload.get('chunk_index')}"
            heading_path = payload.get("heading_path") or []
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "title": payload.get("title") or payload.get("source") or "Snippet",
                    "source": payload.get("source") or "",
                    "doc_id": payload.get("doc_id"),
                    "heading_path": heading_path,
                    "origin": {"vector"},
                    "rank_vector": rank,
                    "summary": payload.get("summary"),
                    "language": payload.get("language"),
                }
            )
        return chunks

    def _keyword_hits_to_chunks(self, hits: Sequence[dict]) -> list[dict]:
        chunks: list[dict] = []
        for rank, hit in enumerate(hits, start=1):
            text = hit.get("text") or ""
            if not text:
                continue
            metadata = hit.get("metadata") or {}
            heading_path = metadata.get("heading_path") or []
            source = metadata.get("source") or ""
            title = hit.get("title") or (heading_path[-1] if heading_path else source) or "Snippet"
            chunks.append(
                {
                    "chunk_id": hit.get("chunk_id"),
                    "text": text,
                    "title": title,
                    "source": source,
                    "doc_id": hit.get("doc_id"),
                    "heading_path": heading_path,
                    "origin": {"keyword"},
                    "rank_keyword": rank,
                    "summary": metadata.get("summary"),
                    "language": metadata.get("language"),
                }
            )
        return chunks

    def _fuse_chunks(
        self,
        vector_chunks: Sequence[dict],
        keyword_chunks: Sequence[dict],
        *,
        limit: int,
    ) -> list[dict]:
        if limit <= 0:
            return []
        fused: dict[str, dict] = {}
        scores: dict[str, float] = {}

        def add_entry(entry: dict, rank: int) -> None:
            chunk_id = entry.get("chunk_id")
            if not chunk_id:
                return
            existing = fused.get(chunk_id)
            if existing is None:
                existing = entry.copy()
                existing_origin = set(entry.get("origin", []))
                existing["origin"] = existing_origin
                fused[chunk_id] = existing
            else:
                existing["origin"].update(entry.get("origin", []))
                for key, value in entry.items():
                    if key in {"origin", "rank_vector", "rank_keyword", "chunk_id"}:
                        continue
                    if not existing.get(key) and value:
                        existing[key] = value
            rr = 1.0 / (rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rr

        for entry in vector_chunks:
            add_entry(entry, entry.get("rank_vector", 0))

        for entry in keyword_chunks:
            add_entry(entry, entry.get("rank_keyword", 0))

        fused_list = list(fused.values())
        for entry in fused_list:
            chunk_id = entry.get("chunk_id")
            entry["score"] = scores.get(chunk_id, 0.0)
            entry["origin"] = sorted(entry.get("origin", []))

        fused_list.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return fused_list[:limit]

    def _apply_reranker(
        self,
        project: Project,
        query: str,
        *,
        query_vector: Sequence[float],
        fused_chunks: Sequence[dict],
        limit: int,
    ) -> list[dict] | None:
        reranker = self._reranker
        if reranker is None or not fused_chunks:
            return None

        metrics = self._metrics
        start = time.perf_counter() if metrics else None
        try:
            reranked = reranker.rerank(
                query,
                query_embedding=query_vector,
                chunks=fused_chunks,
                top_k=limit,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "support.generate.rerank.error strategy=%s project=%s error=%s",
                getattr(reranker, "name", "unknown"),
                project.id,
                exc,
            )
            if metrics:
                metrics.increment(
                    "retrieval.rerank_errors",
                    project_id=project.id,
                    strategy=getattr(reranker, "name", "unknown"),
                )
            return None

        if metrics and start is not None:
            metrics.record_timing(
                "retrieval.rerank_duration",
                time.perf_counter() - start,
                project_id=project.id,
                strategy=getattr(reranker, "name", "unknown"),
                candidates=len(fused_chunks),
            )
            metrics.increment(
                "retrieval.rerank_applied",
                project_id=project.id,
                strategy=getattr(reranker, "name", "unknown"),
            )

        if reranked:
            logger.info(
                "support.generate.rerank strategy=%s project=%s before=%s after=%s",
                getattr(reranker, "name", "unknown"),
                project.id,
                len(fused_chunks),
                len(reranked),
            )
            return list(reranked)
        return None

    def _stream_backend(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> Iterable[str]:
        if self._settings.is_openai_chat_backend:
            logger.info("support.backend.openai.stream")
            return self._stream_openai(prompt, temperature=temperature, max_tokens=max_tokens)
        if self._settings.is_ollama_chat_backend:
            logger.info("support.backend.ollama.stream model=%s", self._settings.ollama_model)
            return self._stream_ollama(prompt, temperature=temperature, max_tokens=max_tokens)

        def generator() -> Iterator[str]:
            yield self._invoke_backend(prompt, temperature=temperature, max_tokens=max_tokens)

        return generator()

    def _invoke_backend(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        if self._settings.is_openai_chat_backend:
            logger.info("support.backend.openai.invoke")
            return self._invoke_openai(prompt, temperature=temperature, max_tokens=max_tokens)
        if self._settings.is_ollama_chat_backend:
            logger.info("support.backend.ollama.invoke model=%s", self._settings.ollama_model)
            return self._invoke_ollama(prompt, temperature=temperature, max_tokens=max_tokens)
        raise RuntimeError(f"Unsupported chat backend: {self._settings.chat_backend}")

    def _invoke_openai(self, prompt: str, *, temperature: float, max_tokens: int | None) -> str:
        client = self._get_openai_client()

        response = client.responses.create(
            model=self._settings.openai_chat_model,
            input=[
                {"role": "system", "content": "You are an empathetic technical support agent."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        texts: list[str] = []
        for item in response.output:  # type: ignore[attr-defined]
            if getattr(item, "type", "") == "output_text":
                texts.append(getattr(item, "text", ""))
        if texts:
            result = "\n".join(texts).strip()
            logger.info(
                "support.backend.openai.success model=%s chars=%s",
                self._settings.openai_chat_model,
            len(result),
            )
            return result
        # Fallback: inspect content
        if getattr(response, "output_text", None):
            result = str(response.output_text).strip()
            logger.info(
                "support.backend.openai.success model=%s chars=%s fallback=True",
                self._settings.openai_chat_model,
                len(result),
            )
            return result
        return "I'm sorry, I could not generate a response at this time."

    def _invoke_ollama(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        if httpx is None:  # pragma: no cover
            raise RuntimeError("httpx must be installed for the Ollama chat backend")

        model = (self._settings.ollama_model or "").strip()
        if not model:
            raise RuntimeError("OLLAMA_MODEL must be set when using the Ollama chat backend")

        base_url = self._settings.ollama_base_url.rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an empathetic technical support agent."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        options: dict[str, float | int] = {}
        if temperature > 0.0:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options

        try:
            response = httpx.post(url, json=payload, timeout=self._settings.ollama_request_timeout)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network errors
            logger.error("support.backend.ollama.error model=%s error=%s", model, exc)
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        data = response.json()
        message = data.get("message") or {}
        content = message.get("content") or data.get("response", "")
        text = str(content).strip() if content else ""
        if not text:
            logger.warning("support.backend.ollama.empty_response model=%s", model)
            return "I'm sorry, I could not generate a response at this time."
        logger.info("support.backend.ollama.success model=%s chars=%s", model, len(text))
        return text

    def _get_openai_client(self) -> OpenAI:
        if OpenAI is None:  # pragma: no cover
            raise RuntimeError("openai package is not available")
        if self._openai_client is None:
            if not self._settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY must be set for the OpenAI chat backend")
            self._openai_client = OpenAI(api_key=self._settings.openai_api_key)
        return self._openai_client

    def _stream_openai(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> Iterable[str]:
        client = self._get_openai_client()

        def generator() -> Iterator[str]:
            with client.responses.stream(
                model=self._settings.openai_chat_model,
                input=[
                    {"role": "system", "content": "You are an empathetic technical support agent."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_output_tokens=max_tokens,
            ) as stream:
                for event in stream:
                    if getattr(event, "type", "") == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            yield str(delta)
                stream.get_final_response()

        return generator()

    def _stream_ollama(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> Iterable[str]:
        if httpx is None:  # pragma: no cover
            raise RuntimeError("httpx must be installed for the Ollama chat backend")

        model = (self._settings.ollama_model or "").strip()
        if not model:
            raise RuntimeError("OLLAMA_MODEL must be set when using the Ollama chat backend")

        base_url = self._settings.ollama_base_url.rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an empathetic technical support agent."},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }
        options: dict[str, float | int] = {}
        if temperature > 0.0:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options

        def generator() -> Iterator[str]:
            try:
                with httpx.stream("POST", url, json=payload, timeout=self._settings.ollama_request_timeout) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:  # pragma: no cover - malformed chunk
                            logger.debug("support.backend.ollama.stream.decode_error line=%s", line)
                            continue
                        message = data.get("message") or {}
                        content = message.get("content") or data.get("response", "")
                        if content:
                            yield str(content)
            except Exception as exc:  # pragma: no cover - network errors
                logger.error("support.backend.ollama.stream.error model=%s error=%s", model, exc)
                raise RuntimeError(f"Ollama streaming request failed: {exc}") from exc

        return generator()


__all__ = ["SupportAgentService", "SupportAgentResponse", "SupportAgentContext"]
