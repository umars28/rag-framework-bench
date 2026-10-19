# rag-framework-bench

One retrieval task, implemented three times on the same corpus, chunks, embedding model and reranker: once with no framework, once with LangChain and once with LlamaIndex. All three are scored on the same 70 labelled queries over 381 Kubernetes documentation pages.

The no-framework baseline is a port of `runbook-rag` from the llm-platform repository, which deliberately uses no orchestration framework. This repository asks what a framework actually buys over that. It follows the same rule as llm-platform: every number below comes from a command in this repository that you can run.

## Results

`uv run ragbench eval` writes `results/results.json` and `results/results.md`. Macro-averaged over 70 queries, scored at document level exactly as runbook-rag scores them.

| implementation | stage | recall@5 | recall@10 | hit@5 | MRR | nDCG@5 | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | hybrid | 0.574 | 0.686 | 0.843 | 0.673 | 0.553 | 57.5 | 136.5 |
| baseline | reranked | 0.645 | 0.710 | 0.871 | 0.658 | 0.574 | 5126.8 | 10069.3 |
| langchain | hybrid | 0.574 | 0.686 | 0.843 | 0.673 | 0.553 | 150.8 | 252.6 |
| langchain | reranked | 0.645 | 0.710 | 0.871 | 0.658 | 0.574 | 4773.8 | 8409.8 |
| langchain-defaults | hybrid | 0.571 | 0.652 | 0.886 | 0.655 | 0.538 | 144.6 | 271.3 |
| langchain-defaults | reranked | 0.662 | 0.741 | 0.900 | 0.669 | 0.591 | 4933.3 | 7573.3 |
| llamaindex | hybrid | 0.631 | 0.717 | 0.871 | 0.680 | 0.589 | 152.5 | 286.2 |
| llamaindex | reranked | 0.636 | 0.748 | 0.871 | 0.650 | 0.571 | 4361.4 | 7566.3 |

`hybrid` is dense retrieval plus BM25 fused by reciprocal rank. `reranked` passes the top 100 fused chunks through the `BAAI/bge-reranker-base` cross-encoder. `langchain` is configured to match the baseline as closely as LangChain allows; `langchain-defaults` is the same code with LangChain's own BM25 tokeniser and deduplication left at their defaults.

| implementation | lines of code | packages | install MB | packages beyond baseline | MB beyond baseline | index build s |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 70 | 42 | 777.4 | 0 | 0.0 | 105.5 |
| langchain | 63 | 82 | 822.9 | 40 | 45.5 | 118.2 |
| llamaindex | 63 | 86 | 848.8 | 45 | 71.4 | 218.5 |

Lines of code count non-blank lines in each implementation module. The 75 lines of chunking, tokeniser and settings that all three share are counted separately. Package counts are the transitive closure of each implementation's optional dependency group in `pyproject.toml`, measured in the installed environment; sizes sum the files each distribution installed. PyTorch dominates the baseline's 777 MB, and every implementation needs it for the same two models.

## What the numbers say

LangChain, configured for parity, returns the same top ten documents as the baseline on all 70 queries at both stages. That is the expected result once you read the source: `langchain_community`'s `BM25Retriever` wraps the same `rank_bm25.BM25Okapi` the baseline uses, it accepts the baseline tokeniser through `preprocess_func`, and `EnsembleRetriever` is weighted reciprocal rank fusion with `c=60`. On this task LangChain buys no retrieval quality and saves seven lines of code. It costs 40 packages and about 93 ms at the median of the hybrid stage (150.8 ms against 57.5 ms). `ragbench breakdown` puts most of that in dense search: LangChain's `InMemoryVectorStore` takes 41.2 ms at p50 where the baseline's matrix product takes 12.2 ms (`results/breakdown.json`, measured in a separate run).

LlamaIndex beats the baseline before reranking, and the gain disappears after it. Its hybrid recall@5 is 0.631 against 0.574, higher on 11 queries and lower on 4. LlamaIndex's BM25 retriever is the component that differs most: it uses `bm25s` with English stemming and stopword removal, and it no longer accepts a custom tokeniser (the `tokenizer` argument is deprecated and ignored), so it cannot be configured to match. Its reciprocal rank fusion also counts ranks from zero, which shifts every score slightly. I have not isolated which of the two produces the gain. After reranking, LlamaIndex lands at 0.636 against the baseline's 0.645, and the reranker raises recall@5 on 9 of its queries while lowering it on 10. For the baseline the same reranker raises 14 and lowers 6. A better first stage left the cross-encoder less to fix and more to break.

LangChain's defaults scored best after reranking: recall@5 0.662 and hit@5 0.900, against 0.645 and 0.871 for the baseline. Its default BM25 tokeniser is a case-sensitive `str.split`, which on paper should be worse than the baseline tokeniser that lowercases and splits `CrashLoopBackOff` into its parts. Before reranking it is mixed: hit@5 rises from 0.843 to 0.886 while recall@10 falls from 0.686 to 0.652. Against the baseline it is better on 4 queries and worse on 2 after reranking. That is a difference of a few queries in 70 from a single run, and I would not ship a tokeniser change on it.

The reranker is where the time goes, in every implementation. Median latency jumps from tens of milliseconds to between 4.4 and 5.1 seconds per query. The baseline and the parity LangChain pipeline run identical computations and still differ by 353 ms at p50 in the reranked stage, so differences smaller than that between frameworks in that row are noise. The machine was not idle (`uptime` reported a load average of 13.98 from other workloads a minute before the run started), which is why queries are interleaved across implementations rather than run one implementation at a time: whatever the background load did, it did to all of them equally. Absolute latencies from this run should not be compared with runbook-rag's README, which reported 1.2 s per query on CPU on a different day.

## Faithfulness of the baseline

`scripts/crosscheck_runbook_rag.py` imports runbook-rag's own `HybridRetriever`, `RerankingRetriever`, golden loader and metrics from a path you give it, runs them over this repository's corpus with an in-memory store, and compares the result with `results/results.json`. Both stages return identical top ten documents on 70 of 70 queries and identical aggregate metrics (`results/crosscheck.json`). The benchmark itself never imports llm-platform; the script is the only place that touches it, and only when pointed at it.

The numbers differ slightly from llm-platform's README (hybrid recall@5 0.564 there, 0.574 here; reranked 0.638 there, 0.645 here). runbook-rag fetches the unpinned `main` branch of `kubernetes/website`. This repository pins the corpus at commit `1fc565c2f557823a911f073dc577945230f27dd7`, where the same 381 pages chunk into 4,537 pieces rather than the 4,549 runbook-rag reported. Vector search here is exact, where runbook-rag used HNSW in pgvector or Qdrant.

## Where parity was not possible

The three implementations share the corpus, the header-aware chunks (fed to each framework as pre-built documents or nodes, so neither framework re-chunks), `BAAI/bge-small-en-v1.5` with the BGE query instruction on queries only, `BAAI/bge-reranker-base`, 50 candidates from each retriever, 100 fused candidates into the reranker, and the same device. Some things could not be made equal.

LlamaIndex's BM25 cannot use the baseline tokeniser, as described above. Its fusion constant is fixed at 60 inside `QueryFusionRetriever` and ranks start at zero. That retriever also defaults to generating four rewritten queries with an LLM resolved from global settings, which would silently reach for OpenAI; the implementation sets `num_queries=1` and passes a `MockLLM`. LlamaIndex embeds node metadata into the text by default, so the chunk and document ids are excluded explicitly, otherwise they would be embedded and BM25-indexed alongside the content. Its Hugging Face embedding wrapper keeps its own model cache and downloads the model a second time.

LangChain's `EnsembleRetriever` deduplicates by page content unless given an `id_key`; the parity variant sets it to the chunk id. In LangChain 1.x the ensemble and compression retrievers live in `langchain-classic`, and `langchain-community`, which provides `BM25Retriever` and `HuggingFaceCrossEncoder`, prints a deprecation warning saying the package is being sunset.

All three use exact in-memory vector search: a numpy matrix product, LangChain's `InMemoryVectorStore` and LlamaIndex's `SimpleVectorStore`.

## Tracing

Langfuse was verified. `tracing/langfuse/compose.yaml` runs a self-hosted Langfuse 3.225.11 stack (web, worker, Postgres, ClickHouse, Redis and MinIO) on `localhost:3300`, with a project and API keys created at startup. `ragbench trace -n 3` sent 18 traces, three queries through both stages of each implementation, and read them back through the Langfuse public API: 18 of 18 were stored (`results/tracing.json`). How much each trace contains depends on how much the framework exposes.

| trace | observations | where they come from |
| --- | --- | --- |
| baseline.hybrid | 3 | spans written by hand in `tracing.py` around dense search and BM25 |
| baseline.reranked | 4 | the same, plus the cross-encoder |
| langchain.hybrid | 4 | Langfuse's LangChain callback handler |
| langchain.reranked | 5 | the same; the cross-encoder gets no span, because `CrossEncoderReranker` is a document compressor and not a runnable |
| llamaindex.hybrid | 9 | OpenInference's LlamaIndex instrumentor, including the query embedding |
| llamaindex.reranked | 10 | the same, plus the reranker |

This is the clearest thing a framework bought in this project. The LlamaIndex trace, with no tracing code inside the pipeline, was more detailed than the hand-instrumented baseline. The OpenInference instrumentor also logged "Attributes dict is full" warnings, because a retrieval of 100 nodes produces more span attributes than OpenTelemetry's default limit and the oldest are dropped.

The MinIO image is `pgsty/minio`, a community build, because the `minio/minio` repository could not be pulled from Docker Hub when this was set up. That is a supply chain choice the owner should review before reusing the compose file anywhere else.

```bash
docker compose -f tracing/langfuse/compose.yaml up -d
export LANGFUSE_PUBLIC_KEY=pk-lf-ragbench-local LANGFUSE_SECRET_KEY=sk-lf-ragbench-local LANGFUSE_BASE_URL=http://localhost:3300
uv run ragbench trace -n 3
```

The UI is at `http://localhost:3300`, user `bench@example.com`, password `ragbench-local`. Every secret in the compose file is a fixed local placeholder.

LangSmith was not exercised against the real service. No `LANGSMITH_API_KEY` was available, so no trace has been seen in a LangSmith project and there is no screenshot. What is verified is narrower: with `LANGSMITH_TRACING=true` and a key set, the LangChain pipeline emits runs, and a test points `LANGSMITH_ENDPOINT` at a local HTTP server and asserts that runs named `EnsembleRetriever`, `VectorStoreRetriever` and `BM25Retriever` arrive. Another test asserts that with tracing requested but no key, the pipeline runs and tracing stays off. To use it for real:

```bash
export LANGSMITH_TRACING=true LANGSMITH_API_KEY=<key> LANGSMITH_PROJECT=rag-framework-bench
uv run ragbench trace --impl langchain -n 3
```

## Generation

`ragbench ask` retrieves five chunks with the reranked pipeline and asks a local Ollama model to answer from them. One example run with `qwen3:8b` is in `results/ask-example.txt` (1,212 prompt tokens, 347 output tokens, 43.83 s). Answers are not scored; retrieval quality is the result this repository measures.

## Running it

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
uv run pytest
uv run ragbench eval
uv run ragbench report
uv run ragbench breakdown
uv run ragbench search "pod stuck in Pending" --impl llamaindex
uv run ragbench fetch
uv run python scripts/crosscheck_runbook_rag.py ../llm-platform/apps/runbook-rag/src
```

`pytest` runs 25 tests offline, with no models and no API keys. `eval` scores all four configurations and writes `results/`, and `report` rewrites `results.md` from the saved JSON. `breakdown` times the dense and BM25 retrievers on their own. `fetch` rebuilds the corpus snapshot from the pinned commit; on the run behind this README it reproduced `documents.jsonl.gz` byte for byte.

The first `eval` downloads both models into the Hugging Face cache, where they took 2.1 GB and 257 MB on the machine used here, and LlamaIndex downloads the embedding model a second time into its own cache. The tests replace both models with deterministic fakes, block every non-loopback network connection, and run all three implementations end to end on a six-document fixture. `RAGBENCH_DEVICE` forces `cpu`, `mps` or `cuda`; by default the fastest available device is used for all three implementations.

The run in this README used an Apple M5 with 24 GB of memory, macOS 26.7, Python 3.12, PyTorch on the MPS backend, LangChain core 1.6.12 and LlamaIndex core 0.14.25. `uv.lock` pins the rest.

## Layout

| path | what it holds |
| --- | --- |
| `data/golden.yaml` | the 70 labelled queries, copied from llm-platform |
| `data/corpus/` | the 381-page corpus snapshot and its source commit |
| `src/ragbench/impl_baseline.py` | no framework: `rank_bm25`, numpy, `sentence-transformers` |
| `src/ragbench/impl_langchain.py` | `BM25Retriever`, `InMemoryVectorStore`, `EnsembleRetriever`, `CrossEncoderReranker` |
| `src/ragbench/impl_llamaindex.py` | `VectorStoreIndex`, `BM25Retriever`, `QueryFusionRetriever`, `SentenceTransformerRerank` |
| `src/ragbench/shared.py` | chunking, tokeniser and settings used by all three |
| `src/ragbench/evaluate.py` | interleaved scoring and timing |
| `src/ragbench/tracing.py` | Langfuse and LangSmith wiring |
| `tracing/langfuse/compose.yaml` | self-hosted Langfuse |

Labels grade a page 2 when it answers the query and 1 when it is useful supporting context. Only nDCG uses the grades; the other metrics treat both as relevant.

## Corpus and licence

The corpus is the `tasks` and `concepts` sections of the Kubernetes documentation (`kubernetes/website`, CC BY 4.0), normalised the same way runbook-rag normalises it. The code is MIT.
