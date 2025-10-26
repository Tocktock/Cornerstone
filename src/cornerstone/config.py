"""Configuration helpers for the Cornerstone application."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Final, Literal

try:  # pragma: no cover - optional dependency loaded at runtime
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# Model identifiers
OpenAIModelName = Literal["text-embedding-3-large"]

_DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_QDRANT_URL: Final[str] = "http://localhost:6333"
_DEFAULT_QDRANT_COLLECTION: Final[str] = "embeddings"
_DEFAULT_CHAT_BACKEND: Final[str] = "openai"
_DEFAULT_OPENAI_CHAT_MODEL: Final[str] = "gpt-4o-mini"
_DEFAULT_GLOSSARY_PATH: Final[str] = "glossary/glossary.yaml"
_DEFAULT_QUERY_HINTS_PATH: Final[str] = "glossary/query_hints.yaml"
_DEFAULT_QUERY_HINT_BATCH_SIZE: Final[int] = 6
_DEFAULT_QUERY_HINT_CRON: Final[str] = "0 3 * * *"
_DEFAULT_GLOSSARY_TOP_K: Final[int] = 3
_DEFAULT_OLLAMA_URL: Final[str] = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL: Final[str] = "llama3.1:8b"
_DEFAULT_OLLAMA_TIMEOUT: Final[float] = 60.0
_DEFAULT_VLLM_URL: Final[str] = "http://localhost:8000"
_DEFAULT_VLLM_MODEL: Final[str] = "meta-llama/Meta-Llama-3-8B-Instruct"
_DEFAULT_VLLM_TIMEOUT: Final[float] = 60.0
_DEFAULT_OLLAMA_EMBED_CONCURRENCY: Final[int] = 2
_DEFAULT_VLLM_EMBED_CONCURRENCY: Final[int] = 4
_DEFAULT_VLLM_EMBED_BATCH_SIZE: Final[int] = 16
_DEFAULT_DATA_DIR: Final[str] = "data"
_DEFAULT_LOCAL_DATA_DIR: Final[str] = "data/local"
_DEFAULT_FTS_DB: Final[str] = "data/fts.sqlite"
_DEFAULT_PROJECT_NAME: Final[str] = "Default Project"
_DEFAULT_INGESTION_CONCURRENCY: Final[int] = 3
_DEFAULT_INGESTION_FILES_PER_MINUTE: Final[int] = 180
_DEFAULT_RETRIEVAL_TOP_K: Final[int] = 3
_DEFAULT_CHAT_TEMPERATURE: Final[float] = 0.2
_DEFAULT_CHAT_MAX_TOKENS: Final[int] = 600
_DEFAULT_RERANKER_STRATEGY: Final[str] = "none"
_DEFAULT_RERANKER_MODEL: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_RERANKER_MAX_CANDIDATES: Final[int] = 8
_DEFAULT_KEYWORD_FILTER_MAX_RESULTS: Final[int] = 10
_DEFAULT_KEYWORD_FILTER_ALLOW_GENERATED: Final[bool] = False
_DEFAULT_CONVERSATION_RETENTION_DAYS: Final[int] = 30
_DEFAULT_KEYWORD_STAGE2_MAX_NGRAM: Final[int] = 3
_DEFAULT_KEYWORD_STAGE2_MAX_CANDIDATES_PER_CHUNK: Final[int] = 8
_DEFAULT_KEYWORD_STAGE2_MAX_EMBEDDING_PHRASES: Final[int] = 6
_DEFAULT_KEYWORD_STAGE2_MAX_STATISTICAL_PHRASES: Final[int] = 6
_DEFAULT_KEYWORD_STAGE2_USE_LLM_SUMMARY: Final[bool] = True
_DEFAULT_KEYWORD_STAGE2_LLM_SUMMARY_MAX_CHUNKS: Final[int] = 4
_DEFAULT_KEYWORD_STAGE2_LLM_SUMMARY_MAX_RESULTS: Final[int] = 10
_DEFAULT_KEYWORD_STAGE2_LLM_SUMMARY_MAX_CHARS: Final[int] = 320
_DEFAULT_KEYWORD_STAGE2_MIN_CHAR_LENGTH: Final[int] = 3
_DEFAULT_KEYWORD_STAGE2_MIN_OCCURRENCES: Final[int] = 1
_DEFAULT_KEYWORD_STAGE2_EMBEDDING_WEIGHT: Final[float] = 2.25
_DEFAULT_KEYWORD_STAGE2_STATISTICAL_WEIGHT: Final[float] = 1.1
_DEFAULT_KEYWORD_STAGE2_LLM_WEIGHT: Final[float] = 3.1
_DEFAULT_KEYWORD_STAGE3_LABEL_CLUSTERS: Final[bool] = True
_DEFAULT_KEYWORD_STAGE3_LABEL_MAX_CLUSTERS: Final[int] = 6
_DEFAULT_KEYWORD_STAGE4_CORE_LIMIT: Final[int] = 12
_DEFAULT_KEYWORD_STAGE4_MAX_RESULTS: Final[int] = 60
_DEFAULT_KEYWORD_STAGE4_SCORE_WEIGHT: Final[float] = 1.4
_DEFAULT_KEYWORD_STAGE4_DOCUMENT_WEIGHT: Final[float] = 2.0
_DEFAULT_KEYWORD_STAGE4_CHUNK_WEIGHT: Final[float] = 0.5
_DEFAULT_KEYWORD_STAGE4_OCCURRENCE_WEIGHT: Final[float] = 0.3
_DEFAULT_KEYWORD_STAGE4_LABEL_BONUS: Final[float] = 0.5
_DEFAULT_KEYWORD_STAGE5_HARMONIZE_ENABLED: Final[bool] = True
_DEFAULT_KEYWORD_STAGE5_HARMONIZE_MAX_RESULTS: Final[int] = 12
_DEFAULT_KEYWORD_STAGE7_SUMMARY_ENABLED: Final[bool] = True
_DEFAULT_KEYWORD_STAGE7_SUMMARY_MAX_INSIGHTS: Final[int] = 3
_DEFAULT_KEYWORD_STAGE7_SUMMARY_MAX_CONCEPTS: Final[int] = 12
_DEFAULT_KEYWORD_STAGE7_SUMMARY_INLINE_TIMEOUT: Final[float] = 1.0
_DEFAULT_KEYWORD_STAGE7_SUMMARY_POLL_INTERVAL: Final[float] = 1.5
_DEFAULT_KEYWORD_STAGE7_SUMMARY_MAX_JOBS: Final[int] = 64
_DEFAULT_KEYWORD_LLM_MAX_CANDIDATES: Final[int] = 25000
_DEFAULT_KEYWORD_LLM_MAX_TOKENS: Final[int] = 750_000
_DEFAULT_KEYWORD_LLM_MAX_CHUNKS: Final[int] = 10000
_DEFAULT_KEYWORD_RUN_MAX_CONCURRENCY: Final[int] = 1
_DEFAULT_KEYWORD_RUN_MAX_QUEUE: Final[int] = 8
_DEFAULT_KEYWORD_RUN_CACHE_TTL: Final[int] = 86_400
_DEFAULT_KEYWORD_RUN_AUTO_REFRESH: Final[bool] = False
_DEFAULT_KEYWORD_RUN_SYNC_MODE: Final[bool] = True


def _env_optional_bool(name: str) -> bool | None:
    """Read an optional boolean environment variable."""

    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    msg = f"Environment variable {name} must be a boolean value (true/false)."
    raise ValueError(msg)


def _env_optional_int(name: str) -> int | None:
    """Read an optional integer environment variable."""

    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def _env_optional_float(name: str) -> float | None:
    """Read an optional float environment variable."""

    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Environment variable {name} must be a float") from exc


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a fallback (preserving zero)."""

    value = _env_optional_float(name)
    return default if value is None else value


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with a fallback."""

    value = _env_optional_bool(name)
    if value is None:
        return default
    return value


def _split_remote_model_spec(spec: str) -> tuple[str, str | None]:
    """
    Split an embedding model spec into (model, base_url).

    Accepts either a bare model name or a full URL ending with the model name.
    When a URL is provided, the final path segment is treated as the model name.
    If the URL points directly to /v1/embeddings, the model identifier is not
    present and an empty string is returned so callers can raise a clearer error.
    """

    value = spec.strip()
    if not value:
        return "", None

    lowered = value.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        base, sep, model = value.rpartition("/")
        model = model.strip()
        if not sep or not base.strip():
            msg = f"Embedding model spec '{spec}' must include a URL ending with the model name."
            raise ValueError(msg)
        if not model or model.lower() == "embeddings":
            return "", base.strip()
        return model, base.strip()

    return value, None


@dataclass(slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    embedding_model: str = _DEFAULT_EMBEDDING_MODEL
    openai_api_key: str | None = None
    qdrant_url: str = _DEFAULT_QDRANT_URL
    qdrant_api_key: str | None = None
    qdrant_collection: str = _DEFAULT_QDRANT_COLLECTION
    chat_backend: str = _DEFAULT_CHAT_BACKEND
    openai_chat_model: str = _DEFAULT_OPENAI_CHAT_MODEL
    ollama_base_url: str = _DEFAULT_OLLAMA_URL
    ollama_model: str = _DEFAULT_OLLAMA_MODEL
    ollama_request_timeout: float = _DEFAULT_OLLAMA_TIMEOUT
    vllm_base_url: str = _DEFAULT_VLLM_URL
    vllm_embedding_base_url: str | None = None
    vllm_model: str = _DEFAULT_VLLM_MODEL
    vllm_api_key: str | None = None
    vllm_request_timeout: float = _DEFAULT_VLLM_TIMEOUT
    vllm_embedding_concurrency: int = _DEFAULT_VLLM_EMBED_CONCURRENCY
    vllm_embedding_batch_size: int = _DEFAULT_VLLM_EMBED_BATCH_SIZE
    ollama_embedding_concurrency: int = _DEFAULT_OLLAMA_EMBED_CONCURRENCY
    glossary_path: str = _DEFAULT_GLOSSARY_PATH
    query_hint_path: str | None = _DEFAULT_QUERY_HINTS_PATH
    glossary_top_k: int = _DEFAULT_GLOSSARY_TOP_K
    data_dir: str = _DEFAULT_DATA_DIR
    local_data_dir: str = _DEFAULT_LOCAL_DATA_DIR
    fts_db_path: str = _DEFAULT_FTS_DB
    default_project_name: str = _DEFAULT_PROJECT_NAME
    ingestion_project_concurrency_limit: int = _DEFAULT_INGESTION_CONCURRENCY
    ingestion_files_per_minute: int = _DEFAULT_INGESTION_FILES_PER_MINUTE
    retrieval_top_k: int = _DEFAULT_RETRIEVAL_TOP_K
    chat_temperature: float = _DEFAULT_CHAT_TEMPERATURE
    chat_max_tokens: int | None = _DEFAULT_CHAT_MAX_TOKENS
    qdrant_on_disk_payload: bool | None = None
    qdrant_on_disk_vectors: bool | None = None
    qdrant_hnsw_m: int | None = None
    qdrant_hnsw_ef_construct: int | None = None
    qdrant_hnsw_full_scan_threshold: int | None = None
    observability_metrics_enabled: bool = True
    observability_namespace: str = "cornerstone"
    observability_prometheus_enabled: bool = False
    reranker_strategy: str = _DEFAULT_RERANKER_STRATEGY
    reranker_model: str | None = None
    reranker_max_candidates: int = _DEFAULT_RERANKER_MAX_CANDIDATES
    keyword_filter_max_results: int = _DEFAULT_KEYWORD_FILTER_MAX_RESULTS
    keyword_filter_allow_generated: bool = _DEFAULT_KEYWORD_FILTER_ALLOW_GENERATED
    query_hint_batch_size: int = _DEFAULT_QUERY_HINT_BATCH_SIZE
    query_hint_cron: str = _DEFAULT_QUERY_HINT_CRON
    conversation_logging_enabled: bool = True
    conversation_retention_days: int = _DEFAULT_CONVERSATION_RETENTION_DAYS
    conversation_log_dir: str | None = None
    keyword_stage2_max_ngram: int = _DEFAULT_KEYWORD_STAGE2_MAX_NGRAM
    keyword_stage2_max_candidates_per_chunk: int = _DEFAULT_KEYWORD_STAGE2_MAX_CANDIDATES_PER_CHUNK
    keyword_stage2_max_embedding_phrases_per_chunk: int = _DEFAULT_KEYWORD_STAGE2_MAX_EMBEDDING_PHRASES
    keyword_stage2_max_statistical_phrases_per_chunk: int = _DEFAULT_KEYWORD_STAGE2_MAX_STATISTICAL_PHRASES
    keyword_stage2_use_llm_summary: bool = _DEFAULT_KEYWORD_STAGE2_USE_LLM_SUMMARY
    keyword_stage2_llm_summary_max_chunks: int = _DEFAULT_KEYWORD_STAGE2_LLM_SUMMARY_MAX_CHUNKS
    keyword_stage2_llm_summary_max_results: int = _DEFAULT_KEYWORD_STAGE2_LLM_SUMMARY_MAX_RESULTS
    keyword_stage2_llm_summary_max_chars: int = _DEFAULT_KEYWORD_STAGE2_LLM_SUMMARY_MAX_CHARS
    keyword_stage2_min_char_length: int = _DEFAULT_KEYWORD_STAGE2_MIN_CHAR_LENGTH
    keyword_stage2_min_occurrences: int = _DEFAULT_KEYWORD_STAGE2_MIN_OCCURRENCES
    keyword_stage2_embedding_weight: float = _DEFAULT_KEYWORD_STAGE2_EMBEDDING_WEIGHT
    keyword_stage2_statistical_weight: float = _DEFAULT_KEYWORD_STAGE2_STATISTICAL_WEIGHT
    keyword_stage2_llm_weight: float = _DEFAULT_KEYWORD_STAGE2_LLM_WEIGHT
    keyword_stage3_label_clusters: bool = _DEFAULT_KEYWORD_STAGE3_LABEL_CLUSTERS
    keyword_stage3_label_max_clusters: int = _DEFAULT_KEYWORD_STAGE3_LABEL_MAX_CLUSTERS
    keyword_stage4_core_limit: int = _DEFAULT_KEYWORD_STAGE4_CORE_LIMIT
    keyword_stage4_max_results: int = _DEFAULT_KEYWORD_STAGE4_MAX_RESULTS
    keyword_stage4_score_weight: float = _DEFAULT_KEYWORD_STAGE4_SCORE_WEIGHT
    keyword_stage4_document_weight: float = _DEFAULT_KEYWORD_STAGE4_DOCUMENT_WEIGHT
    keyword_stage4_chunk_weight: float = _DEFAULT_KEYWORD_STAGE4_CHUNK_WEIGHT
    keyword_stage4_occurrence_weight: float = _DEFAULT_KEYWORD_STAGE4_OCCURRENCE_WEIGHT
    keyword_stage4_label_bonus: float = _DEFAULT_KEYWORD_STAGE4_LABEL_BONUS
    keyword_stage5_harmonize_enabled: bool = _DEFAULT_KEYWORD_STAGE5_HARMONIZE_ENABLED
    keyword_stage5_harmonize_max_results: int = _DEFAULT_KEYWORD_STAGE5_HARMONIZE_MAX_RESULTS
    keyword_stage7_summary_enabled: bool = _DEFAULT_KEYWORD_STAGE7_SUMMARY_ENABLED
    keyword_stage7_summary_max_insights: int = _DEFAULT_KEYWORD_STAGE7_SUMMARY_MAX_INSIGHTS
    keyword_stage7_summary_max_concepts: int = _DEFAULT_KEYWORD_STAGE7_SUMMARY_MAX_CONCEPTS
    keyword_stage7_summary_inline_timeout: float = _DEFAULT_KEYWORD_STAGE7_SUMMARY_INLINE_TIMEOUT
    keyword_stage7_summary_poll_interval: float = _DEFAULT_KEYWORD_STAGE7_SUMMARY_POLL_INTERVAL
    keyword_stage7_summary_max_jobs: int = _DEFAULT_KEYWORD_STAGE7_SUMMARY_MAX_JOBS
    keyword_llm_max_candidates: int = _DEFAULT_KEYWORD_LLM_MAX_CANDIDATES
    keyword_llm_max_tokens: int = _DEFAULT_KEYWORD_LLM_MAX_TOKENS
    keyword_llm_max_chunks: int = _DEFAULT_KEYWORD_LLM_MAX_CHUNKS
    keyword_run_max_concurrency: int = _DEFAULT_KEYWORD_RUN_MAX_CONCURRENCY
    keyword_run_max_queue: int = _DEFAULT_KEYWORD_RUN_MAX_QUEUE
    keyword_run_cache_ttl: int = _DEFAULT_KEYWORD_RUN_CACHE_TTL
    keyword_run_auto_refresh: bool = _DEFAULT_KEYWORD_RUN_AUTO_REFRESH
    keyword_run_sync_mode: bool = _DEFAULT_KEYWORD_RUN_SYNC_MODE

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings by reading environment variables."""

        metrics_enabled = _env_optional_bool("OBSERVABILITY_METRICS_ENABLED")

        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            qdrant_url=os.getenv("QDRANT_URL", _DEFAULT_QDRANT_URL),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", _DEFAULT_QDRANT_COLLECTION),
            chat_backend=os.getenv("CHAT_BACKEND", _DEFAULT_CHAT_BACKEND),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", _DEFAULT_OPENAI_CHAT_MODEL),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL),
            ollama_model=os.getenv("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL),
            ollama_request_timeout=float(os.getenv("OLLAMA_TIMEOUT", _DEFAULT_OLLAMA_TIMEOUT)),
            vllm_base_url=os.getenv("VLLM_BASE_URL", _DEFAULT_VLLM_URL),
            vllm_embedding_base_url=os.getenv("VLLM_EMBEDDING_BASE_URL"),
            vllm_model=os.getenv("VLLM_MODEL", _DEFAULT_VLLM_MODEL),
            vllm_api_key=os.getenv("VLLM_API_KEY"),
            vllm_request_timeout=float(os.getenv("VLLM_TIMEOUT", _DEFAULT_VLLM_TIMEOUT)),
            vllm_embedding_concurrency=int(
                os.getenv("VLLM_EMBEDDING_CONCURRENCY", _DEFAULT_VLLM_EMBED_CONCURRENCY)
            ),
            vllm_embedding_batch_size=max(
                1,
                int(os.getenv("VLLM_EMBEDDING_BATCH_SIZE", _DEFAULT_VLLM_EMBED_BATCH_SIZE)),
            ),
            ollama_embedding_concurrency=int(
                os.getenv("OLLAMA_EMBEDDING_CONCURRENCY", _DEFAULT_OLLAMA_EMBED_CONCURRENCY)
            ),
            glossary_path=os.getenv("GLOSSARY_PATH", _DEFAULT_GLOSSARY_PATH),
            query_hint_path=os.getenv("QUERY_HINTS_PATH", _DEFAULT_QUERY_HINTS_PATH),
            glossary_top_k=int(os.getenv("GLOSSARY_TOP_K", _DEFAULT_GLOSSARY_TOP_K)),
            data_dir=os.getenv("DATA_DIR", _DEFAULT_DATA_DIR),
            local_data_dir=os.getenv("LOCAL_DATA_DIR", _DEFAULT_LOCAL_DATA_DIR),
            fts_db_path=os.getenv("FTS_DB_PATH", _DEFAULT_FTS_DB),
            default_project_name=os.getenv("DEFAULT_PROJECT_NAME", _DEFAULT_PROJECT_NAME),
            ingestion_project_concurrency_limit=int(
                os.getenv("INGESTION_PROJECT_CONCURRENCY_LIMIT", _DEFAULT_INGESTION_CONCURRENCY)
            ),
            ingestion_files_per_minute=int(
                os.getenv("INGESTION_FILES_PER_MINUTE", _DEFAULT_INGESTION_FILES_PER_MINUTE)
            ),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", _DEFAULT_RETRIEVAL_TOP_K)),
            chat_temperature=float(os.getenv("CHAT_TEMPERATURE", _DEFAULT_CHAT_TEMPERATURE)),
            chat_max_tokens=_env_optional_int("CHAT_MAX_TOKENS")
            or _DEFAULT_CHAT_MAX_TOKENS,
            qdrant_on_disk_payload=_env_optional_bool("QDRANT_ON_DISK_PAYLOAD"),
            qdrant_on_disk_vectors=_env_optional_bool("QDRANT_ON_DISK_VECTORS"),
            qdrant_hnsw_m=_env_optional_int("QDRANT_HNSW_M"),
            qdrant_hnsw_ef_construct=_env_optional_int("QDRANT_HNSW_EF_CONSTRUCT"),
            qdrant_hnsw_full_scan_threshold=_env_optional_int("QDRANT_HNSW_FULL_SCAN_THRESHOLD"),
            observability_metrics_enabled=metrics_enabled if metrics_enabled is not None else True,
            observability_namespace=os.getenv("OBSERVABILITY_NAMESPACE", "cornerstone"),
            observability_prometheus_enabled=_env_bool("OBSERVABILITY_PROMETHEUS_ENABLED", False),
            reranker_strategy=os.getenv("RERANKER_STRATEGY", _DEFAULT_RERANKER_STRATEGY),
            reranker_model=os.getenv("RERANKER_MODEL"),
            reranker_max_candidates=int(
                os.getenv("RERANKER_MAX_CANDIDATES", str(_DEFAULT_RERANKER_MAX_CANDIDATES))
            ),
            keyword_filter_max_results=int(
                os.getenv("KEYWORD_FILTER_MAX_RESULTS", str(_DEFAULT_KEYWORD_FILTER_MAX_RESULTS))
            ),
            keyword_filter_allow_generated=_env_bool(
                "KEYWORD_FILTER_ALLOW_GENERATED", _DEFAULT_KEYWORD_FILTER_ALLOW_GENERATED
            ),
            query_hint_batch_size=int(
                os.getenv("QUERY_HINT_BATCH_SIZE", str(_DEFAULT_QUERY_HINT_BATCH_SIZE))
            ),
            query_hint_cron=os.getenv("QUERY_HINT_CRON", _DEFAULT_QUERY_HINT_CRON),
            conversation_logging_enabled=_env_bool("CONVERSATION_LOGGING_ENABLED", True),
            conversation_retention_days=max(
                0,
                int(os.getenv("CONVERSATION_RETENTION_DAYS", str(_DEFAULT_CONVERSATION_RETENTION_DAYS))),
            ),
            conversation_log_dir=os.getenv("CONVERSATION_LOG_DIR"),
            keyword_stage2_max_ngram=int(
                os.getenv("KEYWORD_STAGE2_MAX_NGRAM", str(_DEFAULT_KEYWORD_STAGE2_MAX_NGRAM))
            ),
            keyword_stage2_max_candidates_per_chunk=int(
                os.getenv(
                    "KEYWORD_STAGE2_MAX_CANDIDATES_PER_CHUNK",
                    str(_DEFAULT_KEYWORD_STAGE2_MAX_CANDIDATES_PER_CHUNK),
                )
            ),
            keyword_stage2_max_embedding_phrases_per_chunk=int(
                os.getenv(
                    "KEYWORD_STAGE2_MAX_EMBEDDING_PHRASES",
                    str(_DEFAULT_KEYWORD_STAGE2_MAX_EMBEDDING_PHRASES),
                )
            ),
            keyword_stage2_max_statistical_phrases_per_chunk=int(
                os.getenv(
                    "KEYWORD_STAGE2_MAX_STATISTICAL_PHRASES",
                    str(_DEFAULT_KEYWORD_STAGE2_MAX_STATISTICAL_PHRASES),
                )
            ),
            keyword_stage2_use_llm_summary=_env_bool(
                "KEYWORD_STAGE2_USE_LLM_SUMMARY", _DEFAULT_KEYWORD_STAGE2_USE_LLM_SUMMARY
            ),
            keyword_stage2_llm_summary_max_chunks=int(
                os.getenv(
                    "KEYWORD_STAGE2_LLM_SUMMARY_MAX_CHUNKS",
                    str(_DEFAULT_KEYWORD_STAGE2_LLM_SUMMARY_MAX_CHUNKS),
                )
            ),
            keyword_stage2_llm_summary_max_results=int(
                os.getenv(
                    "KEYWORD_STAGE2_LLM_SUMMARY_MAX_RESULTS",
                    str(_DEFAULT_KEYWORD_STAGE2_LLM_SUMMARY_MAX_RESULTS),
                )
            ),
            keyword_stage2_llm_summary_max_chars=int(
                os.getenv(
                    "KEYWORD_STAGE2_LLM_SUMMARY_MAX_CHARS",
                    str(_DEFAULT_KEYWORD_STAGE2_LLM_SUMMARY_MAX_CHARS),
                )
            ),
            keyword_stage2_min_char_length=int(
                os.getenv(
                    "KEYWORD_STAGE2_MIN_CHAR_LENGTH",
                    str(_DEFAULT_KEYWORD_STAGE2_MIN_CHAR_LENGTH),
                )
            ),
            keyword_stage2_min_occurrences=int(
                os.getenv(
                    "KEYWORD_STAGE2_MIN_OCCURRENCES",
                    str(_DEFAULT_KEYWORD_STAGE2_MIN_OCCURRENCES),
                )
            ),
            keyword_stage2_embedding_weight=_env_float(
                "KEYWORD_STAGE2_EMBEDDING_WEIGHT",
                _DEFAULT_KEYWORD_STAGE2_EMBEDDING_WEIGHT,
            ),
            keyword_stage2_statistical_weight=_env_float(
                "KEYWORD_STAGE2_STATISTICAL_WEIGHT",
                _DEFAULT_KEYWORD_STAGE2_STATISTICAL_WEIGHT,
            ),
            keyword_stage2_llm_weight=_env_float(
                "KEYWORD_STAGE2_LLM_WEIGHT",
                _DEFAULT_KEYWORD_STAGE2_LLM_WEIGHT,
            ),
            keyword_stage3_label_clusters=_env_bool(
                "KEYWORD_STAGE3_LABEL_CLUSTERS", _DEFAULT_KEYWORD_STAGE3_LABEL_CLUSTERS
            ),
            keyword_stage3_label_max_clusters=int(
                os.getenv(
                    "KEYWORD_STAGE3_LABEL_MAX_CLUSTERS",
                    str(_DEFAULT_KEYWORD_STAGE3_LABEL_MAX_CLUSTERS),
                )
            ),
            keyword_stage4_core_limit=int(
                os.getenv(
                    "KEYWORD_STAGE4_CORE_LIMIT",
                    str(_DEFAULT_KEYWORD_STAGE4_CORE_LIMIT),
                )
            ),
            keyword_stage4_max_results=int(
                os.getenv(
                    "KEYWORD_STAGE4_MAX_RESULTS",
                    str(_DEFAULT_KEYWORD_STAGE4_MAX_RESULTS),
                )
            ),
            keyword_stage4_score_weight=_env_float(
                "KEYWORD_STAGE4_SCORE_WEIGHT",
                _DEFAULT_KEYWORD_STAGE4_SCORE_WEIGHT,
            ),
            keyword_stage4_document_weight=_env_float(
                "KEYWORD_STAGE4_DOCUMENT_WEIGHT",
                _DEFAULT_KEYWORD_STAGE4_DOCUMENT_WEIGHT,
            ),
            keyword_stage4_chunk_weight=_env_float(
                "KEYWORD_STAGE4_CHUNK_WEIGHT",
                _DEFAULT_KEYWORD_STAGE4_CHUNK_WEIGHT,
            ),
            keyword_stage4_occurrence_weight=_env_float(
                "KEYWORD_STAGE4_OCCURRENCE_WEIGHT",
                _DEFAULT_KEYWORD_STAGE4_OCCURRENCE_WEIGHT,
            ),
            keyword_stage4_label_bonus=_env_float(
                "KEYWORD_STAGE4_LABEL_BONUS",
                _DEFAULT_KEYWORD_STAGE4_LABEL_BONUS,
            ),
            keyword_stage5_harmonize_enabled=_env_bool(
                "KEYWORD_STAGE5_HARMONIZE_ENABLED", _DEFAULT_KEYWORD_STAGE5_HARMONIZE_ENABLED
            ),
            keyword_stage5_harmonize_max_results=int(
                os.getenv(
                    "KEYWORD_STAGE5_HARMONIZE_MAX_RESULTS",
                    str(_DEFAULT_KEYWORD_STAGE5_HARMONIZE_MAX_RESULTS),
                )
            ),
            keyword_stage7_summary_enabled=_env_bool(
                "KEYWORD_STAGE7_SUMMARY_ENABLED", _DEFAULT_KEYWORD_STAGE7_SUMMARY_ENABLED
            ),
            keyword_stage7_summary_max_insights=int(
                os.getenv(
                    "KEYWORD_STAGE7_SUMMARY_MAX_INSIGHTS",
                    str(_DEFAULT_KEYWORD_STAGE7_SUMMARY_MAX_INSIGHTS),
                )
            ),
            keyword_stage7_summary_max_concepts=int(
                os.getenv(
                    "KEYWORD_STAGE7_SUMMARY_MAX_CONCEPTS",
                    str(_DEFAULT_KEYWORD_STAGE7_SUMMARY_MAX_CONCEPTS),
                )
            ),
            keyword_stage7_summary_inline_timeout=float(
                os.getenv(
                    "KEYWORD_STAGE7_SUMMARY_INLINE_TIMEOUT",
                    str(_DEFAULT_KEYWORD_STAGE7_SUMMARY_INLINE_TIMEOUT),
                )
            ),
            keyword_stage7_summary_poll_interval=float(
                os.getenv(
                    "KEYWORD_STAGE7_SUMMARY_POLL_INTERVAL",
                    str(_DEFAULT_KEYWORD_STAGE7_SUMMARY_POLL_INTERVAL),
                )
            ),
            keyword_stage7_summary_max_jobs=int(
                os.getenv(
                    "KEYWORD_STAGE7_SUMMARY_MAX_JOBS",
                    str(_DEFAULT_KEYWORD_STAGE7_SUMMARY_MAX_JOBS),
                )
            ),
            keyword_llm_max_candidates=int(
                os.getenv(
                    "KEYWORD_LLM_MAX_CANDIDATES",
                    str(_DEFAULT_KEYWORD_LLM_MAX_CANDIDATES),
                )
            ),
            keyword_llm_max_tokens=int(
                os.getenv(
                    "KEYWORD_LLM_MAX_TOKENS",
                    str(_DEFAULT_KEYWORD_LLM_MAX_TOKENS),
                )
            ),
            keyword_llm_max_chunks=int(
                os.getenv(
                    "KEYWORD_LLM_MAX_CHUNKS",
                    str(_DEFAULT_KEYWORD_LLM_MAX_CHUNKS),
                )
            ),
            keyword_run_max_concurrency=int(
                os.getenv(
                    "KEYWORD_RUN_MAX_CONCURRENCY",
                    str(_DEFAULT_KEYWORD_RUN_MAX_CONCURRENCY),
                )
            ),
            keyword_run_max_queue=int(
                os.getenv(
                    "KEYWORD_RUN_MAX_QUEUE",
                    str(_DEFAULT_KEYWORD_RUN_MAX_QUEUE),
                )
            ),
            keyword_run_cache_ttl=int(
                os.getenv(
                    "KEYWORD_RUN_CACHE_TTL",
                    str(_DEFAULT_KEYWORD_RUN_CACHE_TTL),
                )
            ),
            keyword_run_auto_refresh=_env_bool(
                "KEYWORD_RUN_AUTO_REFRESH",
                _DEFAULT_KEYWORD_RUN_AUTO_REFRESH,
            ),
            keyword_run_sync_mode=_env_bool(
                "KEYWORD_RUN_SYNC_MODE",
                _DEFAULT_KEYWORD_RUN_SYNC_MODE,
            ),
        )

    @property
    def is_openai_backend(self) -> bool:
        """Return True when the configured embedding backend is OpenAI."""

        return self.embedding_model.strip().lower() == "text-embedding-3-large"

    @property
    def is_huggingface_backend(self) -> bool:
        """Return True when the configured embedding backend is a local HuggingFace model."""

        return (
            not self.is_openai_backend
            and not self.is_ollama_embedding_backend
            and not self.is_vllm_embedding_backend
        )

    @property
    def is_ollama_embedding_backend(self) -> bool:
        """Return True when embeddings should be generated via an Ollama-hosted model."""

        return self.embedding_model.strip().lower().startswith("ollama:")

    @property
    def ollama_embedding_model(self) -> str | None:
        """Return the Ollama embedding model name without the prefix when configured."""

        if not self.is_ollama_embedding_backend:
            return None
        _, _, name = self.embedding_model.partition(":")
        model, _ = _split_remote_model_spec(name)
        return model or None

    @property
    def ollama_embedding_endpoint(self) -> tuple[str, str]:
        """Return the Ollama embedding model and resolved base URL for embeddings."""

        if not self.is_ollama_embedding_backend:
            msg = "Ollama embedding endpoint requested but EMBEDDING_MODEL is not an Ollama model."
            raise ValueError(msg)
        _, _, name = self.embedding_model.partition(":")
        model, base = _split_remote_model_spec(name)
        if not model:
            msg = "EMBEDDING_MODEL must include an Ollama model identifier."
            raise ValueError(msg)
        base_url = (base or _DEFAULT_OLLAMA_URL).rstrip("/")
        if not base_url:
            msg = "Resolved Ollama embedding base URL is empty."
            raise ValueError(msg)
        return model, base_url

    @property
    def is_vllm_embedding_backend(self) -> bool:
        """Return True when embeddings should be generated via a vLLM-hosted model."""

        return self.embedding_model.strip().lower().startswith("vllm:")

    @property
    def vllm_embedding_model(self) -> str | None:
        """Return the vLLM embedding model name without the prefix when configured."""

        if not self.is_vllm_embedding_backend:
            return None
        _, _, name = self.embedding_model.partition(":")
        model, _ = _split_remote_model_spec(name)
        return model or None

    @property
    def vllm_embedding_endpoint(self) -> tuple[str, str]:
        """Return the vLLM embedding model and resolved base URL for embeddings."""

        if not self.is_vllm_embedding_backend:
            msg = "vLLM embedding endpoint requested but EMBEDDING_MODEL is not a vLLM model."
            raise ValueError(msg)
        _, _, name = self.embedding_model.partition(":")
        model, parsed_base = _split_remote_model_spec(name)
        base_override = (self.vllm_embedding_base_url or "").strip()
        base_url = (base_override or parsed_base or _DEFAULT_VLLM_URL).strip()
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1/embeddings"):
            base_url = base_url[: -len("/v1/embeddings")]
        elif base_url.endswith("/embeddings"):
            base_url = base_url[: -len("/embeddings")]
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        base_url = base_url.rstrip("/")
        if not model:
            msg = (
                "EMBEDDING_MODEL must include the vLLM model identifier "
                "(e.g. 'vllm:qwen3-embeddings'). Set VLLM_EMBEDDING_BASE_URL to "
                "override the endpoint host when needed."
            )
            raise ValueError(msg)
        if not base_url:
            msg = "Resolved vLLM embedding base URL is empty."
            raise ValueError(msg)
        return model, base_url

    @property
    def required_openai_model(self) -> OpenAIModelName:
        """Return the OpenAI embedding model identifier, validating the selection."""

        if not self.is_openai_backend:
            msg = "OpenAI model requested but embedding_model is not an OpenAI model."
            raise ValueError(msg)
        return "text-embedding-3-large"

    @property
    def is_openai_chat_backend(self) -> bool:
        """Return True when using the OpenAI Responses API for chat."""

        return self.chat_backend.lower() == "openai"

    @property
    def is_ollama_chat_backend(self) -> bool:
        """Return True when the chat backend is configured for an Ollama-hosted model."""

        return self.chat_backend.lower() == "ollama"

    @property
    def is_vllm_chat_backend(self) -> bool:
        """Return True when the chat backend is configured for a vLLM-hosted model."""

        return self.chat_backend.lower() == "vllm"

    def qdrant_client_kwargs(self) -> dict[str, Any]:
        """Configuration arguments for instantiating a Qdrant client."""

        kwargs: dict[str, Any] = {"url": self.qdrant_url}
        if self.qdrant_api_key:
            kwargs["api_key"] = self.qdrant_api_key
        return kwargs

    def project_collection_name(self, project_id: str) -> str:
        """Return the Qdrant collection name for the given project."""

        safe_project = project_id.replace(" ", "-")
        return f"{self.qdrant_collection}_{safe_project}"

    @property
    def keyword_run_async_enabled(self) -> bool:
        return not self.keyword_run_sync_mode

    def qdrant_hnsw_config(self) -> dict[str, int]:
        """Return HNSW tuning parameters, omitting unset values."""

        config: dict[str, int] = {}
        if self.qdrant_hnsw_m is not None:
            config["m"] = self.qdrant_hnsw_m
        if self.qdrant_hnsw_ef_construct is not None:
            config["ef_construct"] = self.qdrant_hnsw_ef_construct
        if self.qdrant_hnsw_full_scan_threshold is not None:
            config["full_scan_threshold"] = self.qdrant_hnsw_full_scan_threshold
        return config

    def qdrant_collection_tuning_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for collection creation tuned for scale."""

        kwargs: dict[str, Any] = {}
        if self.qdrant_on_disk_payload is not None:
            kwargs["on_disk_payload"] = self.qdrant_on_disk_payload
        if self.qdrant_on_disk_vectors is not None:
            kwargs["on_disk_vectors"] = self.qdrant_on_disk_vectors
        hnsw_config = self.qdrant_hnsw_config()
        if hnsw_config:
            kwargs["hnsw_config"] = hnsw_config
        return kwargs

    def build_metrics_recorder(self) -> "MetricsRecorder":
        """Instantiate the configured metrics recorder."""

        from .observability import MetricsRecorder

        return MetricsRecorder(
            enabled=self.observability_metrics_enabled,
            namespace=self.observability_namespace,
            prometheus_enabled=self.observability_prometheus_enabled,
        )

    def conversation_log_path(self) -> Path:
        """Return the directory where conversation logs should be stored."""

        if self.conversation_log_dir:
            return Path(self.conversation_log_dir).expanduser().resolve()
        return Path(self.data_dir).resolve() / "conversations"
