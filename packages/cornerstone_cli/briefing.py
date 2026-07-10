from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from cornerstone_cli.product_access import ProductAccessApplication, SearchRequest
from cornerstone_cli.runtime import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_OLLAMA_BASE_URL,
    LocalRuntimeStore,
)


SUPPORTED_MODEL_PROVIDERS = {"local_test", "ollama"}


@dataclass(frozen=True, slots=True)
class RuntimeModelConfig:
    """Immutable operator-owned model configuration for Brief and Ask."""

    provider: str = DEFAULT_MODEL_PROVIDER
    generation_model: str = DEFAULT_GENERATION_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        generation_model = self.generation_model.strip()
        embedding_model = self.embedding_model.strip()
        base_url = self.ollama_base_url.strip().rstrip("/")
        if provider not in SUPPORTED_MODEL_PROVIDERS:
            raise ValueError(f"Unsupported model provider: {self.provider}")
        if not generation_model:
            raise ValueError("Generation model must not be empty.")
        if not embedding_model:
            raise ValueError("Embedding model must not be empty.")
        _validate_loopback_ollama_url(base_url)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "generation_model", generation_model)
        object.__setattr__(self, "embedding_model", embedding_model)
        object.__setattr__(self, "ollama_base_url", base_url)

    @classmethod
    def deterministic(cls) -> "RuntimeModelConfig":
        return cls(provider="local_test")

    @classmethod
    def product_default(cls) -> "RuntimeModelConfig":
        return cls(
            provider="ollama",
            ollama_base_url=os.environ.get("CORNERSTONE_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        )

    def store_kwargs(self) -> dict[str, str]:
        return {
            "model_provider": self.provider,
            "generation_model": self.generation_model,
            "embedding_model": self.embedding_model,
            "ollama_base_url": self.ollama_base_url,
        }

    def public_metadata(self) -> dict[str, str]:
        return {
            "model_provider": self.provider,
            "generation_model": self.generation_model,
            "embedding_model": self.embedding_model,
        }


class BriefingApplication:
    """Shared active-spine interface used by CLI and HTTP transports."""

    def __init__(self, store: LocalRuntimeStore, model_config: RuntimeModelConfig) -> None:
        self.store = store
        self.model_config = model_config

    def create_brief(self, evidence_bundle_id: str, scope: dict[str, str]) -> dict[str, Any]:
        return self.store.create_brief_from_evidence_bundle(
            evidence_bundle_id,
            scope,
            **self.model_config.store_kwargs(),
        )

    def answer(self, conversation_id: str, question: str, scope: dict[str, str]) -> dict[str, Any]:
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            return {"status": "not_found", "resource": "conversation"}
        if conversation.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": conversation.get("scope")}
        search = ProductAccessApplication(self.store).search(
            SearchRequest(
                query=question,
                scope=scope,
                mode="evidence",
                page_size=100,
                excluded_source_types=frozenset({"conversation_turn"}),
            )
        )
        if search.get("status") != "success":
            return search
        return self.store.answer_conversation(
            conversation_id,
            question,
            scope,
            search_result={
                "snapshot": search["search_snapshot"],
                "audit_event": search["audit_event"],
            },
            **self.model_config.store_kwargs(),
        )


def _validate_loopback_ollama_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Ollama URL must be an HTTP(S) loopback URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Ollama URL must not contain credentials, query parameters, or fragments.")
    if parsed.path not in {"", "/"}:
        raise ValueError("Ollama URL must not contain a path.")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("Ollama URL must contain a valid port when one is provided.") from error
    hostname = parsed.hostname.lower()
    if hostname == "localhost":
        return
    try:
        if ipaddress.ip_address(hostname).is_loopback:
            return
    except ValueError:
        pass
    raise ValueError("Ollama URL must resolve to an explicit loopback host.")
