"""Keyword extraction utilities for project documents."""

from __future__ import annotations

import re
import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import httpx
import re

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from .config import Settings

# Basic English stop words; extendable if needed.
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
    "http",
    "https",
    "www",
    "com",
    "files",
    "file",
    "image",
    "img",
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "svg",
    "send",
    "admin",
}

_WORD_RE = re.compile(r"[A-Za-z\uAC00-\uD7A3][A-Za-z0-9\uAC00-\uD7A3'\-]{0,}")

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
class KeywordCandidate:
    """Represents a keyword extracted from project content."""

    term: str
    count: int
    is_core: bool = False
    generated: bool = False
    reason: str | None = None
    source: str = "frequency"


def _contains_hangul(text: str) -> bool:
    return any("\uAC00" <= char <= "\uD7A3" for char in text)


def build_excerpt(text: str, *, max_chars: int = 280) -> str:
    collapsed = re.sub(r"\s+", " ", str(text)).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max(0, max_chars - 1)].rstrip() + "…"


def extract_keyword_candidates(
    texts: Iterable[str],
    *,
    core_limit: int = 10,
    min_length: int = 3,
    min_count: int = 1,
) -> List[KeywordCandidate]:
    """Return ranked keyword candidates and flag the most frequent terms as core."""

    counter: Counter[str] = Counter()
    original_forms: dict[str, str] = {}

    for text in texts:
        if not text:
            continue
        for match in _WORD_RE.finditer(str(text)):
            token = match.group()
            normalized = token.lower()
            cleaned = normalized.strip("'-")
            if not cleaned:
                continue
            token_length = len(cleaned)
            if token_length < min_length:
                if _contains_hangul(cleaned):
                    if token_length < 2:
                        continue
                else:
                    continue
            if cleaned in _STOPWORDS:
                continue
            if cleaned.isdigit():
                continue
            if any(char.isdigit() for char in cleaned) and any(char.isalpha() for char in cleaned):
                continue
            counter[cleaned] += 1
            original_forms.setdefault(cleaned, token.strip("'-"))

    filtered = [item for item in counter.items() if item[1] >= min_count]
    filtered.sort(key=lambda item: (-item[1], item[0]))

    core_cutoff = min(core_limit, len(filtered))
    results: list[KeywordCandidate] = []
    for index, (term, count) in enumerate(filtered):
        # Only consider a keyword "core" if it appears more than once.
        is_core = index < core_cutoff and count > 1
        display_term = original_forms.get(term, term)
        results.append(KeywordCandidate(term=display_term, count=count, is_core=is_core))
    return results


class KeywordLLMFilter:
    """Use the configured chat backend to refine keyword candidates."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._backend: str | None = None
        self._enabled = False
        self._openai_client: OpenAI | None = None
        self._ollama_base_url: str | None = None
        self._ollama_model: str | None = None
        self._ollama_timeout: float = max(settings.ollama_request_timeout, 300.0)
        self._current_prompt: dict[str, str] | None = None
        self._disable_reason: str | None = None
        self._last_debug: dict[str, object] = {}
        reason: str | None = None

        if settings.is_openai_chat_backend:
            if not settings.openai_api_key:
                reason = "missing-openai-key"
            elif OpenAI is None:
                reason = "openai-sdk-missing"
            elif settings.openai_chat_model:
                try:
                    self._openai_client = OpenAI(api_key=settings.openai_api_key)
                    self._backend = "openai"
                    self._enabled = True
                    logger.info("keyword.llm.backend_ready backend=openai model=%s", settings.openai_chat_model)
                except Exception as exc:  # pragma: no cover - runtime guard
                    logger.warning("keyword.llm.openai_init_failed error=%s", exc)
                    reason = f"openai-init-failed:{exc}"
            else:
                reason = "missing-openai-model"
        elif settings.is_ollama_chat_backend:
            self._ollama_base_url = settings.ollama_base_url.rstrip("/")
            self._ollama_model = settings.ollama_model
            if self._ollama_base_url and self._ollama_model:
                self._backend = "ollama"
                self._enabled = True
                logger.info(
                    "keyword.llm.backend_ready backend=ollama model=%s url=%s",
                    self._ollama_model,
                    self._ollama_base_url,
                )
            else:
                reason = "missing-ollama-config"
        else:
            reason = "chat-backend-disabled"

        if not self._enabled:
            logger.info("keyword.llm.disabled fallback=frequency reason=%s", reason)
            self._disable_reason = reason
            self._last_debug = {
                "status": "bypass",
                "reason": reason,
                "candidate_count": 0,
                "backend": self._backend,
                "enabled": self._enabled,
                "disable_reason": self._disable_reason,
            }

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def backend(self) -> str | None:
        return self._backend

    def filter_keywords(
        self,
        candidates: Sequence[KeywordCandidate],
        context_snippets: Sequence[str],
    ) -> List[KeywordCandidate]:
        self._last_debug = {
            "backend": self._backend,
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "candidate_count": len(candidates),
            "status": "pending",
        }
        if not candidates:
            logger.debug("keyword.llm.skip reason=no-candidates")
            self._last_debug["reason"] = "no-candidates"
            self._last_debug["status"] = "error"
            return []
        if not self._enabled:
            logger.debug(
                "keyword.llm.skip reason=disabled backend=%s candidate_count=%s",
                self._backend,
                len(candidates),
            )
            self._last_debug["reason"] = self._disable_reason or "disabled"
            self._last_debug["status"] = "bypass"
            return list(candidates)

        prompt = self._build_prompt(candidates, context_snippets)
        try:
            raw_response = self._invoke_backend(prompt)
            logger.debug("keyword.llm.response backend=%s text=%s", self._backend, raw_response[:500])
            self._last_debug["response"] = raw_response
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("keyword.llm.invoke_failed backend=%s error=%s", self._backend, exc)
            self._last_debug["reason"] = f"invoke_failed:{exc}"
            self._last_debug["status"] = "error"
            return list(candidates)

        parsed = self._parse_response(raw_response)
        if parsed is None:
            logger.debug("keyword.llm.parse_failed response=%s", raw_response)
            self._last_debug["reason"] = "json-parse-error"
            self._last_debug["status"] = "error"
            return list(candidates)

        keyword_items = parsed.get("keywords")
        if not isinstance(keyword_items, list):
            logger.debug(
                "keyword.llm.malformed_response backend=%s payload=%s",
                self._backend,
                raw_response,
            )
            self._last_debug["reason"] = "missing-keywords-array"
            self._last_debug["status"] = "error"
            return list(candidates)

        term_to_candidate = {candidate.term.lower(): candidate for candidate in candidates}
        seen_terms: set[str] = set()

        def _is_truthy(value: object) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            return str(value).strip().lower() in {"true", "yes", "1", "keep", "include"}

        selected_entries: list[dict[str, object]] = []
        rejected_entries: list[dict[str, object]] = []

        for item in keyword_items:
            term = str(item.get("term", "")).strip()
            if not term:
                continue
            keep_flag = _is_truthy(item.get("keep", False))
            reason = str(item.get("reason", "")).strip() or None
            source = str(item.get("source", "")).strip()
            source = source or ("candidate" if term.lower() in term_to_candidate else "generated")
            entry = {
                "term": term,
                "keep": keep_flag,
                "reason": reason,
                "source": source,
                "count": item.get("count"),
            }
            if keep_flag:
                selected_entries.append(entry)
            else:
                rejected_entries.append(entry)

        if not selected_entries:
            self._last_debug["reason"] = "no-keywords-retained"
            self._last_debug["status"] = "error"
            self._last_debug["rejected"] = rejected_entries
            return list(candidates)

        filtered: list[KeywordCandidate] = []
        generated_terms: list[str] = []
        kept_terms: list[str] = []

        for entry in selected_entries:
            term = entry["term"].strip()
            key = term.lower()
            if key in seen_terms:
                continue
            seen_terms.add(key)
            candidate = term_to_candidate.get(key)
            if candidate:
                candidate.reason = entry.get("reason") or candidate.reason
                candidate.source = entry.get("source") or candidate.source
                candidate.generated = False
                candidate.is_core = True if candidate.count >= 1 else candidate.is_core
                filtered.append(candidate)
                kept_terms.append(candidate.term)
            else:
                try:
                    count_value = int(entry.get("count") or 1)
                except (TypeError, ValueError):
                    count_value = 1
                count_value = max(1, count_value)
                new_candidate = KeywordCandidate(
                    term=term,
                    count=count_value,
                    is_core=True,
                    generated=True,
                    reason=entry.get("reason"),
                    source=str(entry.get("source") or "generated"),
                )
                filtered.append(new_candidate)
                generated_terms.append(term)

        if not filtered:
            self._last_debug["reason"] = "no-keywords-retained"
            self._last_debug["status"] = "error"
            self._last_debug["rejected"] = rejected_entries
            return list(candidates)

        dropped = [candidate.term for candidate in candidates if candidate.term.lower() not in seen_terms]
        self._last_debug.update(
            {
                "reason": "filtered",
                "kept": kept_terms,
                "generated": generated_terms,
                "dropped": dropped,
                "selected": selected_entries,
                "rejected": rejected_entries,
                "status": "filtered",
            }
        )
        logger.info(
            "keyword.llm.filter_result backend=%s kept=%s dropped=%s",
            self._backend,
            kept_terms + generated_terms,
            dropped,
        )
        return filtered or list(candidates)

    def debug_payload(self) -> dict[str, object]:
        return {
            "backend": self._backend,
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            **self._last_debug,
        }

    @staticmethod
    def _parse_response(raw_response: str) -> dict | None:
        decoder = json.JSONDecoder()

        def _attempt(candidate: str) -> dict | None:
            candidate = candidate.strip()
            if not candidate:
                return None
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

            for token in ("{", "["):
                idx = candidate.find(token)
                while idx != -1:
                    try:
                        parsed, _ = decoder.raw_decode(candidate[idx:])
                        return parsed
                    except json.JSONDecodeError:
                        idx = candidate.find(token, idx + 1)
            return None

        primary = _attempt(raw_response)
        if primary is not None:
            return primary

        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", raw_response, re.DOTALL)
        if fenced_match:
            fenced_content = fenced_match.group(1)
            fenced_result = _attempt(fenced_content)
            if fenced_result is not None:
                return fenced_result

        return None

    def _build_prompt(
        self,
        candidates: Sequence[KeywordCandidate],
        context_snippets: Sequence[str],
    ) -> str:
        formatted_candidates = "\n".join(
            f"- {candidate.term} (count={candidate.count})" for candidate in candidates
        )
        if context_snippets:
            formatted_context = "\n".join(
                f"{index + 1}. {snippet[:400]}" for index, snippet in enumerate(context_snippets)
            )
        else:
            formatted_context = "(No additional context provided)"

        instructions = (
            "You are a bilingual domain keyword analyst for a customer-support knowledge base. "
            "Identify meaningful product names, process steps, issue categories, and other high-value concepts. "
            "Treat both Korean and English as first-class; preserve Hangul without transliteration. "
            "Remove generic tokens such as file extensions, random identifiers, or UI boilerplate. "
            "If the context reveals important concepts that are missing from the candidate list, you may add them."
        )

        prompt = (
            f"PROJECT CONTEXT:\n{formatted_context}\n\n"
            f"CANDIDATE KEYWORDS:\n{formatted_candidates}\n\n"
            "Respond with strict JSON: {\"keywords\": [...]} where each entry includes:\n"
            "  - term: the keyword (keep original Korean/English script)\n"
            "  - keep: true or false\n"
            "  - reason: brief explanation (Korean or English)\n"
            "  - source: 'candidate' when reviewing provided tokens, or 'generated' when you introduce a new keyword.\n"
            "Return at most 10 entries and avoid duplicates or near-identical synonyms."
        )

        self_role = (
            "You are a helpful assistant that filters noisy candidate keywords. "
            + instructions
        )

        self._current_prompt = {
            "system": self_role,
            "user": prompt,
        }
        return prompt

    def generate_definitions(
        self,
        keyword: str,
        context_snippets: Sequence[str],
        *,
        max_items: int = 3,
    ) -> list[str]:
        if not self._enabled:
            return []

        formatted_context = "\n".join(
            f"{index + 1}. {snippet[:400]}" for index, snippet in enumerate(context_snippets)
        ) or "(no additional context)"

        system_prompt = (
            "You are a bilingual technical writer. Based on the provided context, "
            "write concise bullet-style definitions (Korean or English as appropriate) "
            "for the specified keyword."
        )

        user_prompt = (
            f"Keyword: {keyword}\n"
            f"Context:\n{formatted_context}\n\n"
            "Return JSON: {\"definitions\": [\"definition text\", ...]} with up to"
            f" {max_items} entries, prioritising domain-relevant meanings."
        )

        self._current_prompt = {"system": system_prompt, "user": user_prompt}

        try:
            raw = self._invoke_backend(user_prompt)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("keyword.llm.definition_failed backend=%s error=%s", self._backend, exc)
            return []

        parsed = self._parse_definition_response(raw)
        if not parsed:
            logger.debug("keyword.llm.definition_parse_failed response=%s", raw)
            return []

        definitions = []
        for entry in parsed.get("definitions", []):
            text = str(entry).strip()
            if text:
                definitions.append(text)
        return definitions

    @staticmethod
    def _parse_definition_response(raw_response: str) -> dict | None:
        decoder = json.JSONDecoder()

        def _attempt(candidate: str) -> dict | None:
            candidate = candidate.strip()
            if not candidate:
                return None
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

            for token in ("{", "["):
                idx = candidate.find(token)
                while idx != -1:
                    try:
                        parsed, _ = decoder.raw_decode(candidate[idx:])
                        return parsed
                    except json.JSONDecodeError:
                        idx = candidate.find(token, idx + 1)
            return None

        primary = _attempt(raw_response)
        if primary is not None:
            return primary

        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", raw_response, re.DOTALL)
        if fenced_match:
            fenced_content = fenced_match.group(1)
            return _attempt(fenced_content) or None

        return None

    def _invoke_backend(self, prompt: str) -> str:
        if self._current_prompt is None:
            raise RuntimeError("Prompt context was not initialised")

        if self._backend == "openai" and self._openai_client is not None:
            response = self._openai_client.responses.create(
                model=self._settings.openai_chat_model,
                input=[
                    {"role": "system", "content": self._current_prompt["system"]},
                    {"role": "user", "content": self._current_prompt["user"]},
                ],
            )
            texts: list[str] = []
            for item in getattr(response, "output", []):
                if getattr(item, "type", "") == "output_text":
                    texts.append(getattr(item, "text", ""))
            if texts:
                return "\n".join(texts).strip()
            if getattr(response, "output_text", None):
                return str(response.output_text).strip()
            raise RuntimeError("OpenAI response did not include text output")

        if self._backend == "ollama" and self._ollama_base_url and self._ollama_model:
            url = f"{self._ollama_base_url}/api/chat"
            payload = {
                "model": self._ollama_model,
                "messages": [
                    {"role": "system", "content": self._current_prompt["system"]},
                    {"role": "user", "content": self._current_prompt["user"]},
                ],
                "stream": False,
            }
            response = httpx.post(url, json=payload, timeout=self._ollama_timeout)
            response.raise_for_status()
            data = response.json()
            message = data.get("message") or {}
            content = message.get("content") or data.get("response")
            if not content:
                raise RuntimeError("Ollama response did not include content")
            return str(content).strip()

        raise RuntimeError("LLM backend is not correctly configured")


__all__ = ["KeywordCandidate", "extract_keyword_candidates", "KeywordLLMFilter"]
