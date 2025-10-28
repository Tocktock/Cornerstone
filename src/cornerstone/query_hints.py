"""Utilities for generating query expansion hints with LLM support."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import httpx

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from .config import Settings, normalize_vllm_base_url
from .glossary import GlossaryEntry

logger = logging.getLogger(__name__)
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False


@dataclass(slots=True)
class HintGenerationReport:
    """Result metadata returned alongside generated hints."""

    hints: Dict[str, List[str]]
    prompts_sent: int
    backend: str | None
    disabled_reason: str | None = None


class QueryHintGenerator:
    """Generate cross-language query hints for glossary entries using an LLM."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm_call: Callable[[str], str] | None = None,
        max_terms_per_prompt: int = 6,
    ) -> None:
        self._settings = settings
        self._max_terms = max(1, max_terms_per_prompt)
        self._backend: str | None = None
        self._disabled_reason: str | None = None
        self._ollama_url: str | None = None
        self._ollama_model: str | None = None
        self._ollama_timeout: float = max(settings.ollama_request_timeout, 300.0)
        self._vllm_base_url: str | None = None
        self._vllm_model: str | None = None
        self._vllm_timeout: float = max(settings.vllm_request_timeout, 300.0)
        self._vllm_api_key: str | None = (settings.vllm_api_key or None)
        self._openai_client: OpenAI | None = None
        self._llm_call = llm_call

        if self._llm_call is not None:
            self._backend = "custom"
            return

        if settings.is_openai_chat_backend:
            if not settings.openai_api_key:
                self._disabled_reason = "missing-openai-key"
            elif OpenAI is None:
                self._disabled_reason = "openai-sdk-missing"
            elif settings.openai_chat_model:
                try:
                    self._openai_client = OpenAI(api_key=settings.openai_api_key)
                    self._backend = "openai"
                    logger.info(
                        "query_hints.llm.backend_ready backend=openai model=%s",
                        settings.openai_chat_model,
                    )
                except Exception as exc:  # pragma: no cover - runtime guard
                    self._disabled_reason = f"openai-init-failed:{exc}"
                    logger.warning("query_hints.llm.openai_init_failed error=%s", exc)
            else:
                self._disabled_reason = "missing-openai-model"
        elif settings.is_ollama_chat_backend:
            self._ollama_url = settings.ollama_base_url.rstrip("/")
            self._ollama_model = settings.ollama_model
            if self._ollama_url and self._ollama_model:
                self._backend = "ollama"
                logger.info(
                    "query_hints.llm.backend_ready backend=ollama model=%s url=%s",
                    self._ollama_model,
                    self._ollama_url,
                )
            else:
                self._disabled_reason = "missing-ollama-config"
        elif settings.is_vllm_chat_backend:
            self._vllm_base_url = normalize_vllm_base_url(settings.vllm_base_url)
            self._vllm_model = settings.vllm_model
            if self._vllm_base_url and self._vllm_model:
                self._backend = "vllm"
                logger.info(
                    "query_hints.llm.backend_ready backend=vllm model=%s url=%s",
                    self._vllm_model,
                    self._vllm_base_url,
                )
            else:
                self._disabled_reason = "missing-vllm-config"
        else:
            self._disabled_reason = "unsupported-backend"

    @property
    def enabled(self) -> bool:
        return self._llm_call is not None or self._backend is not None

    @property
    def disabled_reason(self) -> str | None:
        return None if self.enabled else self._disabled_reason or "not-configured"

    def generate(
        self,
        entries: Sequence[GlossaryEntry],
        *,
        progress_callback: Callable[[int, Dict[str, List[str]]], None] | None = None,
        max_terms_per_prompt: int | None = None,
    ) -> HintGenerationReport:
        hints: Dict[str, List[str]] = {}
        if not entries:
            return HintGenerationReport(hints=hints, prompts_sent=0, backend=self._backend, disabled_reason=self.disabled_reason)

        if not self.enabled:
            raise RuntimeError(
                "LLM backend not configured. Set OPENAI_API_KEY / OLLAMA_MODEL or provide llm_call."
            )

        prompts_sent = 0
        batch: List[GlossaryEntry] = []
        max_terms = self._max_terms if max_terms_per_prompt is None else max(1, max_terms_per_prompt)
        for entry in entries:
            batch.append(entry)
            if len(batch) >= max_terms:
                batch_result = self._run_batch(batch)
                self._merge_results(hints, batch_result)
                prompts_sent += 1
                if progress_callback is not None:
                    progress_callback(prompts_sent, batch_result)
                batch = []
        if batch:
            batch_result = self._run_batch(batch)
            self._merge_results(hints, batch_result)
            prompts_sent += 1
            if progress_callback is not None:
                progress_callback(prompts_sent, batch_result)
        return HintGenerationReport(hints=hints, prompts_sent=prompts_sent, backend=self._backend)

    def _merge_results(self, target: Dict[str, List[str]], additions: Dict[str, List[str]]) -> None:
        for key, values in additions.items():
            norm_key = self._normalize_token(key)
            if not norm_key:
                continue
            bucket = target.setdefault(norm_key, [])
            for value in values:
                norm_value = self._normalize_token(value)
                if not norm_value:
                    continue
                if norm_value not in bucket:
                    bucket.append(norm_value)

    def _run_batch(self, entries: Sequence[GlossaryEntry]) -> Dict[str, List[str]]:
        content = json.dumps(
            [
                {
                    "term": entry.term,
                    "definition": entry.definition,
                    "synonyms": entry.synonyms,
                    "keywords": entry.keywords,
                }
                for entry in entries
            ],
            ensure_ascii=False,
        )
        prompt = (
            "You are a technical localization expert helping map glossary vocabulary across languages.\n"
            "You will receive a JSON array of glossary entries. Each entry contains: term, definition, synonyms, keywords.\n"
            "For each entry, produce short bridge tokens that help search queries in other languages connect to the term.\n"
            "Output a single JSON object mapping lowercase tokens (up to 3 words) to a list of related bridge tokens.\n"
            "Include both English and Korean expansions when relevant.\n"
            "Only output JSON; do not include commentary.\n"
            "Entries:\n"
            f"```json\n{content}\n```"
        )
        raw = self._call_llm(prompt)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - depends on model output
            logger.error("query_hints.llm.invalid_response error=%s text=%s", exc, raw)
            raise RuntimeError("LLM returned invalid JSON for query hint generation") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("LLM response must be a JSON object mapping tokens to lists")
        normalized: Dict[str, List[str]] = {}
        for key, value in parsed.items():
            if isinstance(value, str):
                values = [value]
            elif isinstance(value, Iterable):
                values = [str(item) for item in value]
            else:
                continue
            normalized[str(key)] = values
        return normalized

    def _call_llm(self, prompt: str) -> str:
        if self._llm_call is not None:
            return self._llm_call(prompt)
        if self._backend == "openai":
            assert self._openai_client is not None  # for type checkers
            response = self._openai_client.responses.create(
                model=self._settings.openai_chat_model,
                input=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_output_tokens=600,
            )
            texts: List[str] = []
            for item in response.output:  # type: ignore[attr-defined]
                if getattr(item, "type", "") == "output_text":
                    texts.append(getattr(item, "text", ""))
            if texts:
                return "\n".join(texts).strip()
            if getattr(response, "output_text", None):
                return str(response.output_text).strip()
            raise RuntimeError("OpenAI response did not contain text output")
        if self._backend == "ollama":
            assert self._ollama_url and self._ollama_model
            url = f"{self._ollama_url}/api/chat"
            payload = {
                "model": self._ollama_model,
                "messages": [
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            }
            response = httpx.post(url, json=payload, timeout=self._ollama_timeout)
            response.raise_for_status()
            data = response.json()
            message = data.get("message") or {}
            content = message.get("content") or data.get("response", "")
            text = str(content).strip()
            if not text:
                raise RuntimeError("Ollama response was empty")
            return text
        if self._backend == "vllm":
            url, headers, payload = self._prepare_vllm_request(prompt)
            response = httpx.post(url, json=payload, headers=headers, timeout=self._vllm_timeout)
            response.raise_for_status()
            data = response.json()
            for choice in data.get("choices") or []:
                message = choice.get("message") or {}
                content = message.get("content")
                if content:
                    text = str(content).strip()
                    if text:
                        return text
            raise RuntimeError("vLLM response was empty")
        raise RuntimeError("LLM backend not configured")

    def _prepare_vllm_request(self, prompt: str) -> tuple[str, dict[str, str], dict[str, object]]:
        if not self._vllm_base_url or not self._vllm_model:
            raise RuntimeError("VLLM backend not configured for query hint generation")
        url = f"{self._vllm_base_url}/v1/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._vllm_api_key:
            headers["Authorization"] = f"Bearer {self._vllm_api_key}"
        payload: dict[str, object] = {
            "model": self._vllm_model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": 0.0,
            "max_tokens": 600,
        }
        return url, headers, payload

    @staticmethod
    def _normalize_token(value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            return ""
        return normalized


def merge_hint_sources(*hint_dicts: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for hints in hint_dicts:
        for key, values in hints.items():
            norm_key = key.strip().lower()
            if not norm_key:
                continue
            bucket = merged.setdefault(norm_key, [])
            for value in values:
                norm_value = value.strip()
                if not norm_value:
                    continue
                if norm_value not in bucket:
                    bucket.append(norm_value)
    return merged


__all__ = [
    "QueryHintGenerator",
    "HintGenerationReport",
    "merge_hint_sources",
]
