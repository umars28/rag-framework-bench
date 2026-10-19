from __future__ import annotations

from typing import Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from .shared import EMBEDDING_MODEL, QUERY_PREFIX, RERANK_MODEL, Chunk, Settings, device

_KEYS = ["chunk_id", "doc_id"]


def hf_embedding(name: str = EMBEDDING_MODEL) -> BaseEmbedding:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=name,
        query_instruction=QUERY_PREFIX.strip(),
        text_instruction="",
        normalize=True,
        embed_batch_size=64,
        device=device(),
    )


def st_reranker(top_n: int, name: str = RERANK_MODEL) -> BaseNodePostprocessor:
    from llama_index.core.postprocessor import SentenceTransformerRerank

    return SentenceTransformerRerank(model=name, top_n=top_n, device=device())


class LlamaIndexPipeline:
    name = "llamaindex"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embed_model: BaseEmbedding | None = None,
        reranker: BaseNodePostprocessor | None = None,
        settings: Settings = Settings(),
    ) -> None:
        self.settings = settings
        nodes = [
            TextNode(
                id_=c.chunk_id,
                text=c.content,
                metadata={"chunk_id": c.chunk_id, "doc_id": c.doc_id},
                excluded_embed_metadata_keys=_KEYS,
                excluded_llm_metadata_keys=_KEYS,
            )
            for c in chunks
        ]
        index = VectorStoreIndex(nodes, embed_model=embed_model or hf_embedding())
        dense = index.as_retriever(similarity_top_k=settings.dense_k)
        lexical = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=settings.lexical_k)
        self.fusion = QueryFusionRetriever(
            [dense, lexical],
            mode="reciprocal_rerank",
            similarity_top_k=settings.fused_k,
            num_queries=1,
            use_async=False,
            llm=MockLLM(),
        )
        self.reranker = reranker or st_reranker(settings.final_k)

    def hybrid(self, query: str) -> list[str]:
        return [n.node.node_id for n in self.fusion.retrieve(query)][: self.settings.final_k]

    def reranked(self, query: str) -> list[str]:
        nodes = self.reranker.postprocess_nodes(self.fusion.retrieve(query), query_str=query)
        return [n.node.node_id for n in nodes][: self.settings.final_k]


def build(chunks: Sequence[Chunk], **kwargs) -> LlamaIndexPipeline:
    return LlamaIndexPipeline(chunks, **kwargs)
