from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .shared import EMBEDDING_MODEL, QUERY_PREFIX, RERANK_MODEL, Chunk, Settings, device, tokenise


class Embedder(Protocol):
    def passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def query(self, text: str) -> np.ndarray: ...


class Scorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]: ...


class SentenceEmbedder:
    def __init__(self, name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=device())

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts), batch_size=64, normalize_embeddings=True, convert_to_numpy=True
        )
        return vectors.astype(np.float32)

    def passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def query(self, text: str) -> np.ndarray:
        return self._encode([QUERY_PREFIX + text])[0]


def cross_encoder(name: str = RERANK_MODEL) -> Scorer:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, device=device())


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int) -> list[str]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + position)
    return [item for item, _ in sorted(fused.items(), key=lambda pair: pair[1], reverse=True)]


class BaselinePipeline:
    name = "baseline"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder | None = None,
        scorer: Scorer | None = None,
        settings: Settings = Settings(),
    ) -> None:
        from rank_bm25 import BM25Okapi

        self.settings = settings
        self.embedder = embedder or SentenceEmbedder()
        self.scorer = scorer or cross_encoder()
        self.ids = [c.chunk_id for c in chunks]
        self.text = {c.chunk_id: c.content for c in chunks}
        self.vectors = self.embedder.passages([c.content for c in chunks])
        self.bm25 = BM25Okapi([tokenise(c.content) for c in chunks])

    def dense(self, query: str, k: int) -> list[str]:
        scores = self.vectors @ self.embedder.query(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [self.ids[i] for i in order]

    def lexical(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenise(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.ids[i] for i in order]

    def fused(self, query: str, k: int) -> list[str]:
        rankings = [self.dense(query, self.settings.dense_k), self.lexical(query, self.settings.lexical_k)]
        return reciprocal_rank_fusion(rankings, self.settings.rrf_k)[:k]

    def hybrid(self, query: str) -> list[str]:
        return self.fused(query, self.settings.final_k)

    def reranked(self, query: str) -> list[str]:
        candidates = self.fused(query, self.settings.fused_k)
        if not candidates:
            return []
        scores = self.scorer.predict([(query, self.text[c]) for c in candidates])
        ordered = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)
        return [c for c, _ in ordered[: self.settings.final_k]]


def build(chunks: Sequence[Chunk], **kwargs) -> BaselinePipeline:
    return BaselinePipeline(chunks, **kwargs)
