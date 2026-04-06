from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cornerstone.config import Settings
from cornerstone.domain.schemas import EvidenceRead


class OllamaError(RuntimeError):
    """Raised when the local Ollama runtime cannot satisfy a request."""


class OllamaClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.chat_model = settings.ollama_chat_model
        self.embedding_model = settings.ollama_embedding_model
        self.timeout_seconds = settings.ollama_timeout_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.embedding_model, "input": texts}
        response = self._post_json("/api/embed", payload)
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list):
            raise OllamaError("Ollama embedding response did not include embeddings.")
        return embeddings

    def generate_answer_summary(
        self,
        query: str,
        concepts: list[str],
        relations: list[str],
        decisions: list[str],
        evidence: list[EvidenceRead],
    ) -> str:
        evidence_blocks = "\n\n".join(
            (
                f"[{index}] {item.artifact_title}\n"
                f"Selector: {item.selector}\n"
                f"Excerpt: {_truncate(item.excerpt, 400)}"
            )
            for index, item in enumerate(evidence, start=1)
        )
        prompt = (
            "You answer only from the supplied source evidence.\n"
            "Do not invent facts.\n"
            "If the evidence is incomplete, say so explicitly.\n"
            "Keep the answer to at most four sentences.\n"
            "Return only the final answer text.\n\n"
            f"Question: {query}\n"
            f"Structured concepts: {', '.join(concepts) if concepts else 'none'}\n"
            f"Structured relations: {', '.join(relations) if relations else 'none'}\n"
            f"Structured decisions: {', '.join(decisions) if decisions else 'none'}\n\n"
            f"Evidence:\n{evidence_blocks}"
        )
        payload = {
            "model": self.chat_model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1},
        }
        response = self._post_json("/api/generate", payload)
        text = str(response.get("response", "")).strip()
        if not text:
            raise OllamaError("Ollama generation response did not include answer text.")
        return text

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            message = f"Ollama request to {path} failed with HTTP {exc.code}: {detail}"
            raise OllamaError(message) from exc
        except (URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", str(exc))
            raise OllamaError(f"Ollama request to {path} failed: {reason}") from exc
        return data


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."
