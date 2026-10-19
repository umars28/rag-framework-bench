from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_REF = "1fc565c2f557823a911f073dc577945230f27dd7"


def cmd_fetch(args) -> int:
    from .corpus import fetch_source, load_documents, save_documents

    commit = fetch_source(Path(args.checkout), args.ref)
    docs = load_documents(Path(args.checkout))
    out = save_documents(docs, commit)
    print(f"{len(docs)} documents at {commit} -> {out}")
    return 0


def cmd_eval(args) -> int:
    from .evaluate import RESULTS_DIR, run

    report = run(args.impl, Path(args.out) if args.out else RESULTS_DIR)
    print((Path(args.out or RESULTS_DIR) / "results.md").read_text())
    print(f"{report['corpus']}")
    return 0


def cmd_search(args) -> int:
    from .corpus import load_corpus
    from .evaluate import builder
    from .shared import chunk_documents

    chunks = chunk_documents(load_corpus())
    by_id = {c.chunk_id: c for c in chunks}
    pipeline = builder(args.impl)(chunks)
    start = time.perf_counter()
    ids = getattr(pipeline, args.stage)(args.query)[: args.k]
    print(f"{args.impl}/{args.stage} {(time.perf_counter() - start) * 1000:.0f} ms")
    for rank, chunk_id in enumerate(ids, start=1):
        chunk = by_id[chunk_id]
        print(f"{rank:>2}. {chunk.doc_title}" + (f" > {chunk.section}" if chunk.section else ""))
    return 0


def cmd_ask(args) -> int:
    from .corpus import load_corpus
    from .evaluate import builder
    from .generate import answer
    from .shared import chunk_documents

    chunks = chunk_documents(load_corpus())
    by_id = {c.chunk_id: c for c in chunks}
    pipeline = builder(args.impl)(chunks)
    context = [by_id[i] for i in pipeline.reranked(args.question)[: args.k]]
    result = answer(args.question, context, **({"model": args.model} if args.model else {}))
    for i, chunk in enumerate(context, start=1):
        print(f"[{i}] {chunk.doc_title}" + (f" > {chunk.section}" if chunk.section else ""))
    print(f"\n{result['answer']}\n")
    print(f"{result['model']}: {result['prompt_tokens']} prompt tokens, "
          f"{result['output_tokens']} output tokens, {result['seconds']} s")
    return 0


def cmd_trace(args) -> int:
    import datetime as dt
    import json
    from collections import defaultdict

    from . import tracing
    from .corpus import load_corpus, load_golden
    from .evaluate import RESULTS_DIR, builder
    from .shared import chunk_documents

    print(f"LangSmith tracing: {tracing.langsmith_status()}")
    client = tracing.langfuse_client()
    print(f"Langfuse: {'configured' if client else 'not configured (LANGFUSE_PUBLIC_KEY/SECRET_KEY unset)'}")
    if client and "llamaindex" in args.impl:
        print(f"LlamaIndex OpenInference instrumentation: {tracing.instrument_llamaindex()}")

    docs = load_corpus()
    queries = load_golden(docs)[: args.n]
    chunks = chunk_documents(docs)
    session = "ragbench-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    sent = 0
    for name in args.impl:
        pipeline = builder(name)(chunks)
        if client:
            pipeline = tracing.Traced(pipeline, client, session)
        for query in queries:
            for stage in ("hybrid", "reranked"):
                getattr(pipeline, stage)(query.text)
                sent += 1
        print(f"  {name}: {len(queries) * 2} traced calls")
    if client:
        client.flush()
    tracing.flush_langsmith()
    if not client:
        return 0

    traces = tracing.stored_traces(session, sent)
    counts = tracing.observation_counts([t["id"] for t in traces])
    by_name: dict[str, list[int]] = defaultdict(list)
    for trace in traces:
        by_name[trace["name"]].append(counts[trace["id"]])
    summary = {
        "session_id": session,
        "calls_sent": sent,
        "traces_stored": len(traces),
        "observations_per_trace": {k: sorted(v) for k, v in sorted(by_name.items())},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "tracing.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if len(traces) >= sent else 1


def main(argv: list[str] | None = None) -> int:
    from .evaluate import BUILDERS

    parser = argparse.ArgumentParser(prog="ragbench")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="rebuild the corpus snapshot from kubernetes/website")
    p.add_argument("--ref", default=DEFAULT_REF)
    p.add_argument("--checkout", default=".cache/k8s-website")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("eval", help="score every implementation on the labelled queries")
    p.add_argument("--impl", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    p.add_argument("--out")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("search", help="run one query through one implementation")
    p.add_argument("query")
    p.add_argument("--impl", default="baseline", choices=list(BUILDERS))
    p.add_argument("--stage", default="reranked", choices=["hybrid", "reranked"])
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="retrieve, then answer with a local Ollama model")
    p.add_argument("question")
    p.add_argument("--impl", default="baseline", choices=list(BUILDERS))
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--model")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("trace",help="send traced queries to Langfuse and/or LangSmith")
    p.add_argument("--impl", nargs="+", default=["baseline", "langchain", "llamaindex"], choices=list(BUILDERS))
    p.add_argument("-n", type=int, default=3)
    p.set_defaults(func=cmd_trace)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
