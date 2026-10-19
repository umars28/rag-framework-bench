from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ragbench.corpus import REPO_ROOT, load_corpus
from ragbench.shared import chunk_documents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runbook_rag_src", type=Path)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results" / "results.json")
    args = parser.parse_args()
    sys.path.insert(0, str(args.runbook_rag_src.resolve()))

    import runbook_rag.embedding as embedding
    from runbook_rag.evaluate import run_eval
    from runbook_rag.golden import load_golden
    from runbook_rag.hybrid import HybridRetriever
    from runbook_rag.ingest import Document
    from runbook_rag.metrics import aggregate
    from runbook_rag.rerank import RerankingRetriever
    from runbook_rag.store import Hit

    embedding.CACHE_DIR = REPO_ROOT / ".cache" / "crosscheck"
    docs = load_corpus()
    chunks = chunk_documents(docs)
    vectors = embedding.embed_passages([c.content for c in chunks])

    class MemoryStore:
        name = "memory"

        def __init__(self) -> None:
            self.hits = {c.chunk_id: Hit(c.chunk_id, c.doc_id, c.doc_title, c.section, "", c.content, 0.0) for c in chunks}
            self.ids = [c.chunk_id for c in chunks]

        def all_chunks(self, strategy):
            return [(c.chunk_id, c.doc_id, c.content) for c in chunks]

        def hydrate(self, ids):
            return {i: self.hits[i] for i in ids if i in self.hits}

        def search(self, vector, k, strategy):
            scores = vectors @ vector
            order = np.argsort(-scores, kind="stable")[:k]
            return [self.hits[self.ids[i]] for i in order]

    store = MemoryStore()
    queries = load_golden(docs=[Document(d.doc_id, d.title, d.url, d.path, d.content) for d in docs])
    mine = json.loads(args.results.read_text())["runs"]["baseline"]
    report = {}
    for stage, retriever in (
        ("hybrid", HybridRetriever(store, "header")),
        ("reranked", RerankingRetriever(HybridRetriever(store, "header"), 100)),
    ):
        results = run_eval(retriever, queries)
        ours = {q["id"]: q["docs"] for q in mine[stage]["queries"]}
        same = sum(1 for r in results if r.retrieved == ours[r.query_id])
        summary = aggregate(results)
        report[stage] = {
            "runbook_rag": {k: summary[k] for k in ("recall_at_5", "recall_at_10", "hit_at_5", "mrr", "ndcg_at_5")},
            "ragbench_baseline": {k: mine[stage]["summary"][k] for k in ("recall_at_5", "recall_at_10", "hit_at_5", "mrr", "ndcg_at_5")},
            "identical_top10": f"{same}/{len(results)}",
        }
    out = REPO_ROOT / "results" / "crosscheck.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
