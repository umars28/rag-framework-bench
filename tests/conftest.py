from __future__ import annotations

import hashlib
from typing import Any, List, Sequence

import numpy as np
import pytest

from ragbench.corpus import Document, Query
from ragbench.shared import QUERY_PREFIX, chunk_documents, tokenise

DIM = 64


def hash_vector(text: str) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    for token in tokenise(text.removeprefix(QUERY_PREFIX)):
        vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def overlap(query: str, passage: str) -> float:
    q = set(tokenise(query))
    return float(len(q & set(tokenise(passage)))) + 1.0 / (1 + len(passage))


class FakeEmbedder:
    def passages(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([hash_vector(t) for t in texts])

    def query(self, text: str) -> np.ndarray:
        return hash_vector(text)


class FakeScorer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def predict(self, pairs, **kwargs: Any) -> np.ndarray:
        return np.array([overlap(q, p) for q, p in pairs])


def langchain_fakes() -> dict:
    from langchain_classic.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
    from langchain_core.embeddings import Embeddings

    class Emb(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [hash_vector(t).tolist() for t in texts]

        def embed_query(self, text: str) -> List[float]:
            return hash_vector(text).tolist()

    class Cross(BaseCrossEncoder):
        def score(self, text_pairs):
            return [overlap(q, p) for q, p in text_pairs]

    return {"embeddings": Emb(), "cross_encoder": Cross()}


def llamaindex_fakes(monkeypatch) -> dict:
    import sentence_transformers
    from llama_index.core.base.embeddings.base import BaseEmbedding

    class Emb(BaseEmbedding):
        def _get_query_embedding(self, query: str) -> List[float]:
            return hash_vector(query).tolist()

        async def _aget_query_embedding(self, query: str) -> List[float]:
            return hash_vector(query).tolist()

        def _get_text_embedding(self, text: str) -> List[float]:
            return hash_vector(text).tolist()

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeScorer)
    return {"embed_model": Emb(model_name="fake-hash")}


def section(title: str, body: str) -> str:
    return f"## {title}\n\n{body} " + " ".join(["filler"] * 25)


DOCS = [
    Document("d-crash", "Debug Pods", "u1", "content/en/docs/tasks/debug-pods.md",
             section("CrashLoopBackOff", "A pod in CrashLoopBackOff restarts its container repeatedly after it exits.")),
    Document("d-oom", "Resource Limits", "u2", "content/en/docs/concepts/limits.md",
             section("Memory", "A container that exceeds its memory limit is OOMKilled by the kernel.")),
    Document("d-net", "Network Policies", "u3", "content/en/docs/concepts/netpol.md",
             section("Ingress", "A NetworkPolicy selects pods and restricts ingress traffic between services.")),
    Document("d-dns", "DNS for Services", "u4", "content/en/docs/concepts/dns.md",
             section("Resolution", "CoreDNS resolves service names inside the cluster for every namespace.")),
    Document("d-sched", "Scheduler", "u5", "content/en/docs/concepts/scheduler.md",
             section("Pending", "A Pending pod waits until the scheduler finds a node with enough resources.")),
    Document("d-secret", "Secrets", "u6", "content/en/docs/concepts/secrets.md",
             section("Mounting", "A Secret holds a password or token and can be mounted as a volume.")),
]

QUERIES = [
    Query("F-1", "pod keeps restarting CrashLoopBackOff", "crash", {"d-crash": 2}),
    Query("F-2", "container OOMKilled memory limit", "oom", {"d-oom": 2, "d-sched": 1}),
    Query("F-3", "pod stuck Pending scheduler", "sched", {"d-sched": 2}),
    Query("F-4", "restrict traffic between services", "net", {"d-net": 2}),
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else ""
        if self.family in (socket.AF_INET, socket.AF_INET6) and host not in ("127.0.0.1", "::1"):
            raise OSError(f"network access attempted in tests: {address}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    for key in ("LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY",
                "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def chunks():
    return chunk_documents(DOCS)


@pytest.fixture
def fakes(monkeypatch):
    return {
        "baseline": {"embedder": FakeEmbedder(), "scorer": FakeScorer()},
        "langchain": langchain_fakes(),
        "langchain-defaults": langchain_fakes(),
        "llamaindex": llamaindex_fakes(monkeypatch),
    }
