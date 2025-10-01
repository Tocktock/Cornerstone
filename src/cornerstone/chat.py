"""Support agent chat service with switchable backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .config import Settings
from .embeddings import EmbeddingService
from .glossary import Glossary
from .vector_store import QdrantVectorStore, SearchResult

try:  # pragma: no cover - optional import for local llama backend
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover
    Llama = None

try:  # pragma: no cover - guard openai import for static analysis
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


@dataclass(slots=True)
class SupportAgentResponse:
    message: str
    sources: list[dict[str, str]]
    definitions: list[str]


class SupportAgentService:
    """Generate support-oriented answers using retrieval augmented generation."""

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        glossary: Glossary,
        *,
        retrieval_top_k: int = 3,
    ) -> None:
        self._settings = settings
        self._embedding = embedding_service
        self._store = vector_store
        self._glossary = glossary
        self._retrieval_top_k = retrieval_top_k
        self._openai_client: OpenAI | None = None
        self._llama: Llama | None = None

    def generate(self, query: str, *, conversation: Sequence[str] | None = None) -> SupportAgentResponse:
        conversation = conversation or []
        vector = self._embedding.embed_one(query)
        search_results = self._store.search(vector, limit=self._retrieval_top_k)
        prompt = self._build_prompt(query, search_results, conversation)

        answer = self._invoke_backend(prompt)
        sources = self._format_sources(search_results)
        definitions = self._collect_definitions(query)
        return SupportAgentResponse(message=answer, sources=sources, definitions=definitions)

    # Internal helpers -------------------------------------------------

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

        instructions = (
            "You are a support agent helping customers diagnose and resolve issues. "
            "Ask for missing details when necessary, provide step-by-step guidance, and when uncertain, "
            "suggest escalating to a human agent."
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

    def _invoke_backend(self, prompt: str) -> str:
        if self._settings.is_openai_chat_backend:
            return self._invoke_openai(prompt)
        if self._settings.is_local_chat_backend:
            return self._invoke_llama(prompt)
        raise RuntimeError(f"Unsupported chat backend: {self._settings.chat_backend}")

    def _invoke_openai(self, prompt: str) -> str:
        if OpenAI is None:  # pragma: no cover
            raise RuntimeError("openai package is not available")
        if self._openai_client is None:
            if not self._settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY must be set for the OpenAI chat backend")
            self._openai_client = OpenAI(api_key=self._settings.openai_api_key)

        response = self._openai_client.responses.create(
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
            return "\n".join(texts).strip()
        # Fallback: inspect content
        if getattr(response, "output_text", None):
            return str(response.output_text).strip()
        return "I'm sorry, I could not generate a response at this time."

    def _invoke_llama(self, prompt: str) -> str:
        if Llama is None:  # pragma: no cover
            raise RuntimeError("llama-cpp-python must be installed for the local chat backend")
        if not self._settings.llama_model_path:
            raise RuntimeError("LLAMA_MODEL_PATH must be set when using the local chat backend")
        if self._llama is None:
            self._llama = Llama(
                model_path=self._settings.llama_model_path,
                n_ctx=self._settings.llama_context_window,
                verbose=False,
            )
        completion = self._llama.create_completion(prompt=prompt, temperature=0.2, max_tokens=512)
        text = completion.get("choices", [{}])[0].get("text", "")
        return text.strip() or "I'm sorry, I could not generate a response at this time."


__all__ = ["SupportAgentService", "SupportAgentResponse"]
