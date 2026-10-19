from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Protocol

from .corpus import Document

EMBEDDING_MODEL = os.environ.get("RAGBENCH_EMBEDDING", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.environ.get("RAGBENCH_RERANKER", "BAAI/bge-reranker-base")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_WORD = re.compile(r"\S+")
_HEADING = re.compile(r"^(#{1,4})\s+(.*)$", re.M)
_ANCHOR = re.compile(r"\s*\{#[^}]*\}\s*$")
_FENCE = re.compile(r"^```", re.M)
_TOKEN = re.compile(r"[a-z0-9]+")
_CAMEL = re.compile(r"[A-Za-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+")
_CAMEL_PART = re.compile(r"[A-Z]?[a-z0-9]+")


@dataclass(frozen=True)
class Settings:
    dense_k: int = 50
    lexical_k: int = 50
    fused_k: int = 100
    final_k: int = 30
    rrf_k: int = 60


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str | None
    content: str


class Pipeline(Protocol):
    name: str

    def hybrid(self, query: str) -> list[str]: ...

    def reranked(self, query: str) -> list[str]: ...


def _sections(content: str) -> list[tuple[str | None, str]]:
    in_fence = False
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in content.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        match = _HEADING.match(line) if not in_fence else None
        if match:
            sections.append((_ANCHOR.sub("", match.group(2)).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(head, "\n".join(body).strip()) for head, body in sections if "".join(body).strip()]


def header_chunks(doc: Document, size: int = 220, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    for heading, body in _sections(doc.content):
        label = f"{doc.title} — {heading}" if heading else doc.title
        words = _WORD.findall(body)
        for start in range(0, len(words), step):
            window = words[start : start + size]
            if len(window) < 20 and chunks:
                break
            text = f"{label}\n\n{' '.join(window)}".strip()
            chunks.append(
                Chunk(f"{doc.doc_id}:header:{len(chunks):04d}", doc.doc_id, doc.title, heading, text)
            )
    return chunks


def chunk_documents(docs: Iterable[Document]) -> list[Chunk]:
    return [chunk for doc in docs for chunk in header_chunks(doc)]


def tokenise(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    for match in _CAMEL.findall(text):
        tokens.extend(p.lower() for p in _CAMEL_PART.findall(match))
    return tokens
