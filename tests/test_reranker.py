import cornerstone.reranker as reranker_module
from cornerstone.reranker import EmbeddingReranker


class FakeEmbeddingService:
    def embed(self, texts):
        return [[float(text)] for text in texts]

    def embed_one(self, text):
        return [0.0]


def test_embedding_reranker_orders_candidates(monkeypatch):
    reranker = EmbeddingReranker(FakeEmbeddingService(), max_candidates=3)

    monkeypatch.setattr(reranker_module, "_cosine_similarity", lambda _query, chunk: chunk[0])

    chunks = [
        {"chunk_id": "a", "text": "0.3"},
        {"chunk_id": "b", "text": "0.8"},
        {"chunk_id": "c", "text": "0.1"},
    ]

    ordered = reranker.rerank("ignored", query_embedding=[0.0], chunks=chunks, top_k=3)
    assert [chunk["chunk_id"] for chunk in ordered] == ["b", "a", "c"]

    chunks.append({"chunk_id": "d", "text": ""})
    ordered_with_spill = reranker.rerank("ignored", query_embedding=[0.0], chunks=chunks, top_k=4)
    assert ordered_with_spill[-1]["chunk_id"] == "d"
