from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest

from ragbench.corpus import REPO_ROOT, Document, LabelError, load_corpus, load_golden
from ragbench.impl_baseline import reciprocal_rank_fusion
from ragbench.metrics import hit_at_k, ndcg_at_k, percentile, recall_at_k, reciprocal_rank, to_documents
from ragbench.shared import header_chunks, tokenise


def test_recall_hit_mrr():
    retrieved = ["a", "b", "c", "d", "e", "f"]
    assert recall_at_k(retrieved, {"b", "z"}, 5) == 0.5
    assert hit_at_k(retrieved, {"f"}, 5) == 0.0
    assert hit_at_k(retrieved, {"e"}, 5) == 1.0
    assert reciprocal_rank(retrieved, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(retrieved, {"z"}) == 0.0


def test_ndcg_rewards_better_grades_first():
    grades = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], grades, 5) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], grades, 5) < 1.0


def test_to_documents_keeps_best_chunk_order():
    doc_of = {"x1": "x", "y1": "y", "x2": "x"}
    assert to_documents(["x1", "y1", "x2"], doc_of) == ["x", "y"]


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([5], 0.95) == 5


def test_rrf_by_position():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c"}


def test_tokenise_splits_camel_case_and_keeps_whole():
    tokens = tokenise("CrashLoopBackOff now")
    assert "crashloopbackoff" in tokens
    assert {"crash", "loop", "back", "off", "now"} <= set(tokens)


def test_header_chunks_ignore_headings_inside_code_fences():
    body = "## Real\n\n" + "word " * 30 + "\n```\n# not a heading\n```\n" + "more " * 30
    doc = Document("d", "Title", "u", "p.md", body)
    chunks = header_chunks(doc)
    assert {c.section for c in chunks} == {"Real"}
    assert chunks[0].content.startswith("Title — Real")


def test_committed_corpus_and_every_label_resolves():
    docs = load_corpus()
    queries = load_golden(docs)
    assert len(docs) == 381
    assert len(queries) == 70
    assert sum(len(q.relevant) for q in queries) == 143


def test_missing_label_raises(tmp_path):
    golden = tmp_path / "g.yaml"
    golden.write_text("queries:\n- id: X\n  query: q\n  facet: f\n  relevant:\n    nowhere.md: 2\n")
    with pytest.raises(LabelError):
        load_golden([], golden)


def _has_comment_or_docstring(path: Path) -> bool:
    source = path.read_text()
    if any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(source).readline)):
        return True
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node) is not None:
                return True
    return False


def test_no_comments_or_docstrings_in_python_sources():
    files = [p for d in ("src", "tests", "scripts") for p in (REPO_ROOT / d).rglob("*.py")]
    offenders = [str(f.relative_to(REPO_ROOT)) for f in files if _has_comment_or_docstring(f)]
    assert offenders == []


def test_generation_prompt_numbers_every_excerpt():
    from ragbench.generate import prompt
    from ragbench.shared import Chunk

    chunks = [Chunk("a:0", "a", "A", None, "first"), Chunk("b:0", "b", "B", None, "second")]
    text = prompt("why?", chunks)
    assert "[1] first" in text and "[2] second" in text and text.endswith("Question: why?\nAnswer:")
