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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
