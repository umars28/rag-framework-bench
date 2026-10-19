from __future__ import annotations

import datetime as dt
import importlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

from . import footprint
from .corpus import REPO_ROOT, Query, load_corpus, load_golden
from .metrics import FIELDS, aggregate, percentile, score_query, to_documents
from .shared import EMBEDDING_MODEL, RERANK_MODEL, Chunk, Pipeline, Settings, chunk_documents, device

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


def _summary(per_query: list[dict], latencies: list[float]) -> dict:
    return aggregate(per_query) | {
        "p50_ms": round(percentile(latencies, 0.5) * 1000, 1),
        "p95_ms": round(percentile(latencies, 0.95) * 1000, 1),
    }


def evaluate(
    names: Sequence[str],
    chunks: Sequence[Chunk],
    queries: Sequence[Query],
    build_kwargs: dict[str, dict] | None = None,
    settings: Settings = Settings(),
) -> dict:
    doc_of = {c.chunk_id: c.doc_id for c in chunks}
    pipelines: dict[str, Pipeline] = {}
    runs: dict[str, dict] = {}
    for name in names:
        start = time.perf_counter()
        pipelines[name] = builder(name)(chunks, settings=settings, **(build_kwargs or {}).get(name, {}))
        runs[name] = {"build_seconds": round(time.perf_counter() - start, 1)}
        print(f"built {name} in {runs[name]['build_seconds']}s", file=sys.stderr, flush=True)
    for stage in STAGES:
        per_query = {name: [] for name in names}
        latencies = {name: [] for name in names}
        for name in names:
            getattr(pipelines[name], stage)(queries[0].text)
        for query in queries:
            for name in names:
                start = time.perf_counter()
                chunk_ids = getattr(pipelines[name], stage)(query.text)
                latencies[name].append(time.perf_counter() - start)
                docs = to_documents(chunk_ids, doc_of)
                per_query[name].append({"id": query.id, "docs": docs[:10]} | score_query(query, docs))
        for name in names:
            summary = _summary(per_query[name], latencies[name])
            runs[name][stage] = {"summary": summary, "queries": per_query[name]}
            print(f"  {name:20} {stage:9} recall@5 {summary['recall_at_5']:.3f}  p50 {summary['p50_ms']} ms",
                  file=sys.stderr, flush=True)
    return runs


def environment(settings: Settings) -> dict:
    return {
        "ran_at": dt.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch_device": device(),
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
        "settings": settings.__dict__,
    }


def components(pipeline) -> dict[str, Callable[[str], object]]:
    if hasattr(pipeline, "bm25"):
        return {
            "dense": lambda q: pipeline.dense(q, pipeline.settings.dense_k),
            "bm25": lambda q: pipeline.lexical(q, pipeline.settings.lexical_k),
        }
    if hasattr(pipeline, "fusion"):
        return {"dense": pipeline.dense.retrieve, "bm25": pipeline.lexical.retrieve}
    return {"dense": pipeline.dense.invoke, "bm25": pipeline.lexical.invoke}


def breakdown(names: Sequence[str], out_dir: Path = RESULTS_DIR) -> dict:
    docs = load_corpus()
    queries = load_golden(docs)
    chunks = chunk_documents(docs)
    parts = {name: components(builder(name)(chunks)) for name in names}
    timings: dict[str, dict[str, list[float]]] = {n: {c: [] for c in parts[n]} for n in names}
    for name in names:
        for fn in parts[name].values():
            fn(queries[0].text)
    for query in queries:
        for name in names:
            for component, fn in parts[name].items():
                start = time.perf_counter()
                fn(query.text)
                timings[name][component].append(time.perf_counter() - start)
    report = {
        name: {
            c: {"p50_ms": round(percentile(v, 0.5) * 1000, 1), "p95_ms": round(percentile(v, 0.95) * 1000, 1)}
            for c, v in comps.items()
        }
        for name, comps in timings.items()
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "breakdown.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def versus_baseline(runs: dict) -> list[str]:
    if "baseline" not in runs:
        return []
    lines = ["| implementation | stage | identical top-10 documents | recall@5 higher | recall@5 lower |",
             "| --- | --- | --- | --- | --- |"]
    for name, run in runs.items():
        if name == "baseline":
            continue
        for stage in STAGES:
            base = {q["id"]: q for q in runs["baseline"][stage]["queries"]}
            mine = run[stage]["queries"]
            same = sum(1 for q in mine if q["docs"] == base[q["id"]]["docs"])
            higher = sum(1 for q in mine if q["recall_at_5"] > base[q["id"]]["recall_at_5"])
            lower = sum(1 for q in mine if q["recall_at_5"] < base[q["id"]]["recall_at_5"])
            lines.append(f"| {name} | {stage} | {same}/{len(mine)} | {higher} | {lower} |")
    return ["", "Per query, against the baseline at the same stage:", "", *lines]


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
    lines += versus_baseline(runs)
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
