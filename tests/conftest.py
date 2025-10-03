from __future__ import annotations

import sys
import types


class _DefaultSentenceTransformer:
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN401
        self._dim = 3

    def encode(self, texts, show_progress_bar=False):  # noqa: D401, ANN401
        return [[0.0] * self._dim for _ in texts]

    def get_sentence_embedding_dimension(self):  # noqa: D401
        return self._dim


if "sentence_transformers" not in sys.modules:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _DefaultSentenceTransformer
    sys.modules["sentence_transformers"] = module


class _DummyOpenAIResponseStream:
    def __init__(self):  # noqa: D401
        self._events = [types.SimpleNamespace(type="response.output_text.delta", delta="Mock reply.")]

    def __enter__(self):  # noqa: D401
        return self

    def __iter__(self):  # noqa: D401
        yield from self._events

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        return False

    def get_final_response(self):  # noqa: D401
        return types.SimpleNamespace(output_text="Mock reply.")


class _DummyOpenAI:
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN401
        self.embeddings = types.SimpleNamespace(create=lambda **__: [])
        self.models = types.SimpleNamespace(retrieve=lambda *args, **kwargs: None)
        self.responses = types.SimpleNamespace(
            create=lambda **__: types.SimpleNamespace(
                output=[types.SimpleNamespace(type="output_text", text="Mock reply.")],
                output_text="Mock reply.",
            ),
            stream=lambda **__: _DummyOpenAIResponseStream(),
        )


if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = _DummyOpenAI
    sys.modules["openai"] = openai_module
