from __future__ import annotations

from typing import Any, Sequence

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from .shared import EMBEDDING_MODEL, QUERY_PREFIX, RERANK_MODEL, Chunk, Settings, device, tokenise


def hf_embeddings(name: str = EMBEDDING_MODEL) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": device()},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": QUERY_PREFIX},
    )


def hf_cross_encoder(name: str = RERANK_MODEL) -> BaseCrossEncoder:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    return HuggingFaceCrossEncoder(model_name=name, model_kwargs={"device": device()})


class LangChainPipeline:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embeddings: Embeddings | None = None,
        cross_encoder: BaseCrossEncoder | None = None,
        settings: Settings = Settings(),
        parity: bool = True,
    ) -> None:
        self.name = "langchain" if parity else "langchain-defaults"
        self.settings = settings
        self.callbacks: list[Any] = []
        docs = [
            Document(page_content=c.content, metadata={"chunk_id": c.chunk_id, "doc_id": c.doc_id})
            for c in chunks
        ]
        store = InMemoryVectorStore(embeddings or hf_embeddings())
        store.add_documents(docs, ids=[c.chunk_id for c in chunks])
        dense = store.as_retriever(search_kwargs={"k": settings.dense_k})
        lexical_options = {"preprocess_func": tokenise} if parity else {}
        self.lexical = BM25Retriever.from_documents(docs, k=settings.lexical_k, **lexical_options)
        self.ensemble = EnsembleRetriever(
            retrievers=[dense, self.lexical],
            weights=[0.5, 0.5],
            c=settings.rrf_k,
            id_key="chunk_id" if parity else None,
        )
        reranker = CrossEncoderReranker(
            model=cross_encoder or hf_cross_encoder(), top_n=settings.final_k
        )
        self.compression = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=self.ensemble
        )

    def _invoke(self, retriever, query: str) -> list[Document]:
        return retriever.invoke(query, config={"callbacks": self.callbacks, "run_name": self.name})

    def hybrid(self, query: str) -> list[str]:
        docs = self._invoke(self.ensemble, query)[: self.settings.fused_k]
        return [d.metadata["chunk_id"] for d in docs][: self.settings.final_k]

    def reranked(self, query: str) -> list[str]:
        return [d.metadata["chunk_id"] for d in self._invoke(self.compression, query)]


def build(chunks: Sequence[Chunk], **kwargs) -> LangChainPipeline:
    return LangChainPipeline(chunks, **kwargs)
