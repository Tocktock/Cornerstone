"""Embedding service supporting OpenAI and SentenceTransformers backends."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum, auto
from typing import Final, List

import httpx
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from .config import Settings

_OPENAI_DIMENSION: Final[int] = 3072


class EmbeddingBackend(Enum):
    """Supported embedding backends."""

    OPENAI = auto()
    HUGGINGFACE = auto()
    OLLAMA = auto()
    VLLM = auto()


class EmbeddingService:
    """High-level interface for embedding generation."""

    def __init__(self, settings: Settings, *, validate: bool = True) -> None:
        self._settings = settings
        if settings.is_openai_backend:
            backend = EmbeddingBackend.OPENAI
        elif settings.is_ollama_embedding_backend:
            backend = EmbeddingBackend.OLLAMA
        elif settings.is_vllm_embedding_backend:
            backend = EmbeddingBackend.VLLM
        else:
            backend = EmbeddingBackend.HUGGINGFACE

        self._backend = backend
        self._dimension: int | None = None
        self._openai_client: OpenAI | None = None
        self._hf_model: SentenceTransformer | None = None
        self._ollama_model: str | None = None
        self._ollama_base_url: str | None = None
        self._ollama_timeout: float = settings.ollama_request_timeout
        self._ollama_concurrency: int = max(1, settings.ollama_embedding_concurrency)
        self._ollama_client: httpx.Client | None = None
        self._vllm_model: str | None = None
        self._vllm_base_url: str | None = None
        self._vllm_timeout: float = settings.vllm_request_timeout
        api_key = settings.vllm_api_key or None
        if isinstance(api_key, str):
            api_key = api_key.strip()
        self._vllm_api_key = api_key or None
        self._vllm_client: httpx.Client | None = None

        if self._backend is EmbeddingBackend.OPENAI:
            self._setup_openai(validate)
        elif self._backend is EmbeddingBackend.OLLAMA:
            self._setup_ollama(validate)
        elif self._backend is EmbeddingBackend.VLLM:
            self._setup_vllm(validate)
        else:
            self._setup_huggingface(validate)

    @classmethod
    def from_env(cls, *, validate: bool = True) -> "EmbeddingService":
        """Create the embedding service from environment configuration."""

        return cls(Settings.from_env(), validate=validate)

    @property
    def backend(self) -> EmbeddingBackend:
        """Return the active backend type."""

        return self._backend

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality for the active backend."""

        if self._dimension is None:
            msg = "Embedding dimension is not initialised."
            raise RuntimeError(msg)
        return self._dimension

    @property
    def model_identifier(self) -> str:
        """Return the configured embedding model identifier."""

        return self._settings.embedding_model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for a sequence of texts."""

        if not texts:
            return []

        if self._backend is EmbeddingBackend.OPENAI:
            assert self._openai_client is not None  # for mypy
            result = self._openai_client.embeddings.create(
                model=self._settings.required_openai_model,
                input=list(texts),
            )
            return [item.embedding for item in result.data]

        if self._backend is EmbeddingBackend.OLLAMA:
            return self._ollama_batch_embed(texts)

        if self._backend is EmbeddingBackend.VLLM:
            return self._vllm_embed_batch(texts)

        assert self._hf_model is not None
        vectors = self._hf_model.encode(list(texts), show_progress_bar=False)
        if hasattr(vectors, "tolist"):
            return vectors.tolist()
        return [list(vector) for vector in vectors]

    def embed_one(self, text: str) -> List[float]:
        """Generate an embedding for a single piece of text."""

        vectors = self.embed([text])
        return vectors[0]

    def close(self) -> None:
        """Release any underlying client resources."""

        if self._ollama_client is not None:
            try:
                self._ollama_client.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            finally:
                self._ollama_client = None
        if self._vllm_client is not None:
            try:
                self._vllm_client.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            finally:
                self._vllm_client = None

    def __del__(self):  # pragma: no cover - defensive cleanup
        self.close()

    # Internal helpers -------------------------------------------------

    def _setup_openai(self, validate: bool) -> None:
        api_key = self._settings.openai_api_key or None
        if not api_key:
            msg = "OPENAI_API_KEY must be set when using the OpenAI embedding backend."
            raise ValueError(msg)

        self._openai_client = OpenAI(api_key=api_key)
        self._dimension = _OPENAI_DIMENSION

        if validate:
            # Ensure the configured model is accessible; raises if not available.
            self._openai_client.models.retrieve(self._settings.required_openai_model)

    def _setup_huggingface(self, validate: bool) -> None:
        model_name = self._settings.embedding_model
        self._hf_model = SentenceTransformer(model_name)
        self._dimension = int(self._hf_model.get_sentence_embedding_dimension())

        if validate and self._dimension <= 0:
            msg = f"Unexpected embedding dimension ({self._dimension}) for model '{model_name}'."
            raise ValueError(msg)

    def _setup_ollama(self, validate: bool) -> None:
        if httpx is None:  # pragma: no cover - should not happen when dependencies installed
            raise RuntimeError("httpx must be installed to use Ollama embeddings")

        model = self._settings.ollama_embedding_model
        if not model:
            msg = "EMBEDDING_MODEL must be set to an Ollama model when using the Ollama embedding backend."
            raise ValueError(msg)

        self._ollama_model = model
        self._ollama_base_url = self._settings.ollama_base_url.rstrip("/")
        self._ollama_client = httpx.Client(
            base_url=self._ollama_base_url,
            timeout=self._ollama_timeout,
        )

        vector = self._ollama_embed("__dimension_probe__")
        if not vector:
            msg = f"Ollama embedding backend '{model}' returned no data."
            raise ValueError(msg)

        self._dimension = len(vector)
        if self._dimension <= 0:
            msg = f"Unexpected embedding dimension ({self._dimension}) for Ollama model '{model}'."
            raise ValueError(msg)

    def _ollama_embed(self, text: str) -> List[float]:
        if not self._ollama_model or not self._ollama_base_url:
            msg = "Ollama embedding backend is not configured."
            raise RuntimeError(msg)

        if self._ollama_client is None:
            msg = "Ollama embedding backend is not initialised."
            raise RuntimeError(msg)

        url = "/api/embeddings"
        payload = {"model": self._ollama_model, "prompt": text}

        try:
            response = self._ollama_client.post(url, json=payload)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network errors
            raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc

        data = response.json()
        embedding = data.get("embedding")
        if embedding is None:
            msg = "Ollama embedding response did not include an 'embedding' field."
            raise RuntimeError(msg)

        vector = list(map(float, embedding))
        if self._dimension is not None and len(vector) != self._dimension:
            msg = (
                f"Ollama embedding dimension changed from {self._dimension} to {len(vector)}."
            )
            raise RuntimeError(msg)

        return vector

    def _ollama_batch_embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        batch_size = max(self._ollama_concurrency * 4, self._ollama_concurrency)

        for offset in range(0, len(texts), batch_size):
            chunk = list(texts[offset : offset + batch_size])
            vectors: list[list[float] | None] = [None] * len(chunk)
            concurrency = min(self._ollama_concurrency, len(chunk))

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(self._ollama_embed, text): idx for idx, text in enumerate(chunk)}
                for future in as_completed(futures):
                    idx = futures[future]
                    vectors[idx] = future.result()

            results.extend([vector if vector is not None else [] for vector in vectors])

        return results

    def _setup_vllm(self, validate: bool) -> None:
        if httpx is None:  # pragma: no cover - should not happen when dependencies installed
            raise RuntimeError("httpx must be installed to use vLLM embeddings")

        model = self._settings.vllm_embedding_model
        if not model:
            msg = "EMBEDDING_MODEL must be set to a vLLM model when using the vLLM embedding backend."
            raise ValueError(msg)

        base_url = (self._settings.vllm_base_url or "").strip().rstrip("/")
        if not base_url:
            msg = "VLLM_BASE_URL must be configured when using the vLLM embedding backend."
            raise ValueError(msg)

        self._vllm_model = model
        self._vllm_base_url = base_url

        headers = {}
        if self._vllm_api_key:
            headers["Authorization"] = f"Bearer {self._vllm_api_key}"

        self._vllm_client = httpx.Client(
            base_url=self._vllm_base_url,
            timeout=self._vllm_timeout,
            headers=headers or None,
        )

        vectors = self._vllm_embed_batch(["__dimension_probe__"])
        if not vectors or not vectors[0]:
            msg = f"vLLM embedding backend '{model}' returned no data."
            raise ValueError(msg)

        self._dimension = len(vectors[0])
        if validate and self._dimension <= 0:
            msg = f"Unexpected embedding dimension ({self._dimension}) for vLLM model '{model}'."
            raise ValueError(msg)

    def _vllm_embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        if self._vllm_client is None or not self._vllm_model:
            msg = "vLLM embedding backend is not configured."
            raise RuntimeError(msg)

        payload = {"model": self._vllm_model, "input": list(texts)}

        try:
            response = self._vllm_client.post("/v1/embeddings", json=payload)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network errors
            raise RuntimeError(f"vLLM embedding request failed: {exc}") from exc

        data = response.json()
        items = data.get("data")
        if not isinstance(items, list) or not items:
            msg = "vLLM embedding response did not include embedding data."
            raise RuntimeError(msg)

        vectors: list[list[float]] = []
        for item in items:
            embedding = item.get("embedding")
            if embedding is None:
                msg = "vLLM embedding response item missing 'embedding'."
                raise RuntimeError(msg)
            vector = [float(value) for value in embedding]
            if self._dimension is not None and self._dimension > 0 and len(vector) != self._dimension:
                msg = f"vLLM embedding dimension changed from {self._dimension} to {len(vector)}."
                raise RuntimeError(msg)
            vectors.append(vector)

        if len(vectors) != len(texts):
            msg = "vLLM embedding response did not return the expected number of vectors."
            raise RuntimeError(msg)

        return vectors
