from __future__ import annotations

import pytest

from ragbench.evaluate import BUILDERS, builder, evaluate, markdown
from ragbench.footprint import code_size
from ragbench.metrics import to_documents
from ragbench.shared import Settings

from conftest import QUERIES

SMALL = Settings(dense_k=5, lexical_k=5, fused_k=8, final_k=5)


@pytest.mark.parametrize("name", list(BUILDERS))
def test_each_pipeline_finds_the_keyword_answer(name, chunks, fakes):
    pipeline = builder(name)(chunks, settings=SMALL, **fakes[name])
    doc_of = {c.chunk_id: c.doc_id for c in chunks}
    for stage in ("hybrid", "reranked"):
        top = to_documents(getattr(pipeline, stage)("pod keeps restarting CrashLoopBackOff"), doc_of)
        assert top[0] == "d-crash", (name, stage, top)


@pytest.mark.parametrize("name", list(BUILDERS))
def test_stages_respect_final_depth(name, chunks, fakes):
    pipeline = builder(name)(chunks, settings=SMALL, **fakes[name])
    for stage in ("hybrid", "reranked"):
        ids = getattr(pipeline, stage)("memory limit")
        assert 0 < len(ids) <= SMALL.final_k
        assert len(ids) == len(set(ids))


def test_langchain_parity_matches_baseline_fusion(chunks, fakes):
    base = builder("baseline")(chunks, settings=SMALL, **fakes["baseline"])
    lc = builder("langchain")(chunks, settings=SMALL, **fakes["langchain"])
    for query in QUERIES:
        tokens = lc.lexical.preprocess_func(query.text)
        assert list(lc.lexical.vectorizer.get_scores(tokens)) == list(base.bm25.get_scores(tokens))
        assert base.hybrid(query.text)[0] == lc.hybrid(query.text)[0]
        assert base.reranked(query.text)[0] == lc.reranked(query.text)[0]


def test_evaluate_reports_every_stage(chunks, fakes):
    runs = evaluate(list(BUILDERS), chunks, QUERIES, build_kwargs=fakes, settings=SMALL)
    for name, run in runs.items():
        for stage in ("hybrid", "reranked"):
            summary = run[stage]["summary"]
            assert summary["hit_at_5"] > 0.5, (name, stage, summary)
            assert summary["p95_ms"] >= summary["p50_ms"] >= 0
            assert len(run[stage]["queries"]) == len(QUERIES)
    table = markdown({"runs": runs, "lines_of_code": code_size(), "dependencies": {}})
    assert table.count("| reranked |") == 2 * len(BUILDERS) - 1
    assert "identical top-10 documents" in table


def test_code_size_counts_each_implementation():
    sizes = code_size()
    assert set(sizes) == {"baseline", "langchain", "llamaindex", "shared"}
    assert all(v > 10 for v in sizes.values())
