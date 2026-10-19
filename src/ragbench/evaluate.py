from __future__ import annotations

import datetime as dt
import importlib
import json
import platform
import time
from pathlib import Path
from typing import Callable, Sequence

from . import footprint
from .corpus import REPO_ROOT, Query, load_corpus, load_golden
from .metrics import FIELDS, aggregate, percentile, score_query, to_documents
from .shared import EMBEDDING_MODEL, RERANK_MODEL, Chunk, Pipeline, Settings, chunk_documents

RESULTS_DIR = REPO_ROOT / "results"
STAGES = ("hybrid", "reranked")

BUILDERS: dict[str, tuple[str, dict]] = {
    "baseline": ("ragbench.impl_baseline", {}),
    "langchain": ("ragbench.impl_langchain", {}),
    "langchain-defaults": ("ragbench.impl_langchain", {"parity": False}),
    "llamaindex": ("ragbench.impl_llamaindex", {}),
}


def builder(name: str) -> Callable[..., Pipeline]:
    module, options = BUILDERS[name]
    build = importlib.import_module(module).build
    return lambda chunks, **kwargs: build(chunks, **options, **kwargs)


def run_stage(
    pipeline: Pipeline, stage: str, queries: Sequence[Query], doc_of: dict[str, str]
) -> dict:
    search = getattr(pipeline, stage)
    search(queries[0].text)
    per_query, latencies = [], []
    for query in queries:
        start = time.perf_counter()
        chunk_ids = search(query.text)
        latencies.append(time.perf_counter() - start)
        docs = to_documents(chunk_ids, doc_of)
        per_query.append({"id": query.id, "docs": docs[:10]} | score_query(query, docs))
    summary = aggregate(per_query) | {
        "p50_ms": round(percentile(latencies, 0.5) * 1000, 1),
        "p95_ms": round(percentile(latencies, 0.95) * 1000, 1),
    }
    return {"summary": summary, "queries": per_query}


def evaluate(
    names: Sequence[str],
    chunks: Sequence[Chunk],
    queries: Sequence[Query],
    build_kwargs: dict[str, dict] | None = None,
    settings: Settings = Settings(),
) -> dict:
    doc_of = {c.chunk_id: c.doc_id for c in chunks}
    runs = {}
    for name in names:
        start = time.perf_counter()
        pipeline = builder(name)(chunks, settings=settings, **(build_kwargs or {}).get(name, {}))
        build_seconds = time.perf_counter() - start
        runs[name] = {"build_seconds": round(build_seconds, 1)} | {
            stage: run_stage(pipeline, stage, queries, doc_of) for stage in STAGES
        }
    return runs


def environment(settings: Settings) -> dict:
    import torch

    return {
        "ran_at": dt.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch_device": "mps" if torch.backends.mps.is_available() else "cpu",
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
        "settings": settings.__dict__,
    }


def markdown(report: dict) -> str:
    runs, loc, deps = report["runs"], report["lines_of_code"], report["dependencies"]
    header = ["implementation", "stage", *FIELDS, "p50 ms", "p95 ms"]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for name, run in runs.items():
        for stage in STAGES:
            s = run[stage]["summary"]
            cells = [name, stage, *(f"{s[f]:.3f}" for f in FIELDS), f"{s['p50_ms']}", f"{s['p95_ms']}"]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| implementation | build s | lines of code | packages | install MB | packages beyond baseline | MB beyond baseline |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, run in runs.items():
        impl = name.split("-")[0]
        d = deps.get(impl, {})
        lines.append(
            f"| {name} | {run['build_seconds']} | {loc.get(impl, '')} | {d.get('packages', '')} | "
            f"{d.get('install_mb', '')} | {d.get('packages_beyond_baseline', '')} | {d.get('mb_beyond_baseline', '')} |"
        )
    lines += ["", f"Shared code used by all implementations (chunking, tokeniser, settings): {loc['shared']} lines."]
    return "\n".join(lines) + "\n"


def run(names: Sequence[str], out_dir: Path = RESULTS_DIR, settings: Settings = Settings()) -> dict:
    docs = load_corpus()
    queries = load_golden(docs)
    chunks = chunk_documents(docs)
    report = {
        "environment": environment(settings),
        "corpus": {"documents": len(docs), "chunks": len(chunks), "queries": len(queries)},
        "lines_of_code": footprint.code_size(),
        "dependencies": footprint.dependency_footprint(),
        "runs": evaluate(names, chunks, queries, settings=settings),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "results.md").write_text(markdown(report))
    return report
