"""Support agent chat service with switchable backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass
import json
from typing import Iterable, Iterator, List, Sequence, Tuple

from .config import Settings
from .embeddings import EmbeddingService
from .glossary import Glossary
from .vector_store import QdrantVectorStore, SearchResult

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


class SupportAgentService:
    """Generate support-oriented answers using retrieval augmented generation."""

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        store_manager,
        glossary: Glossary,
        *,
        retrieval_top_k: int = 3,
    ) -> None:
        self._settings = settings
        self._embedding = embedding_service
        self._stores = store_manager
        self._glossary = glossary
        self._retrieval_top_k = retrieval_top_k
        self._openai_client: OpenAI | None = None

    def generate(
        self,
        project_id: str,
        query: str,
        *,
        conversation: Sequence[str] | None = None,
    ) -> SupportAgentResponse:
        context, _ = self._build_context(project_id, query, conversation)
        answer = self._invoke_backend(context.prompt)
        logger.info(
            "support.generate.completed backend=%s chars=%s sources=%s",
            self._settings.chat_backend,
            len(answer),
            len(context.sources),
        )
        return SupportAgentResponse(message=answer, sources=context.sources, definitions=context.definitions)

    def stream_generate(
        self,
        project_id: str,
        query: str,
        *,
        conversation: Sequence[str] | None = None,
    ) -> Tuple[SupportAgentContext, Iterable[str]]:
        """Stream a response for the given query, yielding incremental text deltas."""

        context, _ = self._build_context(project_id, query, conversation)
        stream = self._stream_backend(context.prompt)
        return context, stream

    # Internal helpers -------------------------------------------------

    @staticmethod
    def _is_korean(text: str) -> bool:
        return any("\uAC00" <= char <= "\uD7A3" for char in text)

    def _build_context(
        self,
        project_id: str,
        query: str,
        conversation: Sequence[str] | None,
    ) -> Tuple[SupportAgentContext, list[SearchResult]]:
        history = list(conversation or [])
        logger.info(
            "support.generate.start backend=%s embedding=%s project=%s query=%s",
            self._settings.chat_backend,
            self._settings.embedding_model,
            project_id,
            query,
        )
        vector = self._embedding.embed_one(query)
        store = self._stores.get_store(project_id)
        search_results = list(store.search(vector, limit=self._retrieval_top_k))
        top_titles = [(result.payload or {}).get("title") for result in search_results]
        logger.info(
            "support.generate.matches project=%s count=%s titles=%s",
            project_id,
            len(search_results),
            top_titles,
        )
        sources = self._format_sources(search_results)
        definitions = self._collect_definitions(query)
        prompt = self._build_prompt(query, search_results, history)
        context = SupportAgentContext(prompt=prompt, sources=sources, definitions=definitions)
        return context, search_results

    def _build_prompt(
        self,
        query: str,
        search_results: Iterable[SearchResult],
        conversation: Sequence[str],
    ) -> str:
        context_lines = ["Relevant documentation snippets:"]
        for idx, result in enumerate(search_results, start=1):
            payload = result.payload or {}
            title = payload.get("title") or payload.get("text", "Untitled snippet")[:64]
            text = payload.get("text") or ""
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

        instructions = (
            "You are a support agent helping customers diagnose and resolve issues. "
            "Ask for missing details when necessary, provide step-by-step guidance, and when uncertain, "
            "suggest escalating to a human agent. "
            + language_instruction
        )

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

    def _collect_definitions(self, query: str) -> List[str]:
        matches = self._glossary.top_matches(query, self._settings.glossary_top_k)
        return [f"{entry.term}: {entry.definition}" for entry in matches]

    def _format_sources(self, results: Iterable[SearchResult]) -> list[dict[str, str]]:
        formatted: list[dict[str, str]] = []
        for result in results:
            payload = result.payload or {}
            title = payload.get("title") or payload.get("text", "")[:50]
            text = payload.get("text") or ""
            formatted.append({"title": title, "snippet": text})
        return formatted

    def _stream_backend(self, prompt: str) -> Iterable[str]:
        if self._settings.is_openai_chat_backend:
            logger.info("support.backend.openai.stream")
            return self._stream_openai(prompt)
        if self._settings.is_ollama_chat_backend:
            logger.info("support.backend.ollama.stream model=%s", self._settings.ollama_model)
            return self._stream_ollama(prompt)

        def generator() -> Iterator[str]:
            yield self._invoke_backend(prompt)

        return generator()

    def _invoke_backend(self, prompt: str) -> str:
        if self._settings.is_openai_chat_backend:
            logger.info("support.backend.openai.invoke")
            return self._invoke_openai(prompt)
        if self._settings.is_ollama_chat_backend:
            logger.info("support.backend.ollama.invoke model=%s", self._settings.ollama_model)
            return self._invoke_ollama(prompt)
        raise RuntimeError(f"Unsupported chat backend: {self._settings.chat_backend}")

    def _invoke_openai(self, prompt: str) -> str:
        client = self._get_openai_client()

        response = client.responses.create(
            model=self._settings.openai_chat_model,
            input=[
                {"role": "system", "content": "You are an empathetic technical support agent."},
                {"role": "user", "content": prompt},
            ],
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

    def _invoke_ollama(self, prompt: str) -> str:
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

    def _stream_openai(self, prompt: str) -> Iterable[str]:
        client = self._get_openai_client()

        def generator() -> Iterator[str]:
            with client.responses.stream(
                model=self._settings.openai_chat_model,
                input=[
                    {"role": "system", "content": "You are an empathetic technical support agent."},
                    {"role": "user", "content": prompt},
                ],
            ) as stream:
                for event in stream:
                    if getattr(event, "type", "") == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            yield str(delta)
                stream.get_final_response()

        return generator()

    def _stream_ollama(self, prompt: str) -> Iterable[str]:
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
