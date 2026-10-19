| implementation | stage | recall_at_5 | recall_at_10 | hit_at_5 | mrr | ndcg_at_5 | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | hybrid | 0.574 | 0.686 | 0.843 | 0.673 | 0.553 | 57.5 | 136.5 |
| baseline | reranked | 0.645 | 0.710 | 0.871 | 0.658 | 0.574 | 5126.8 | 10069.3 |
| langchain | hybrid | 0.574 | 0.686 | 0.843 | 0.673 | 0.553 | 150.8 | 252.6 |
| langchain | reranked | 0.645 | 0.710 | 0.871 | 0.658 | 0.574 | 4773.8 | 8409.8 |
| langchain-defaults | hybrid | 0.571 | 0.652 | 0.886 | 0.655 | 0.538 | 144.6 | 271.3 |
| langchain-defaults | reranked | 0.662 | 0.741 | 0.900 | 0.669 | 0.591 | 4933.3 | 7573.3 |
| llamaindex | hybrid | 0.631 | 0.717 | 0.871 | 0.680 | 0.589 | 152.5 | 286.2 |
| llamaindex | reranked | 0.636 | 0.748 | 0.871 | 0.650 | 0.571 | 4361.4 | 7566.3 |

| implementation | build s | lines of code | packages | install MB | packages beyond baseline | MB beyond baseline |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 105.5 | 70 | 42 | 777.4 | 0 | 0.0 |
| langchain | 118.2 | 63 | 82 | 822.9 | 40 | 45.5 |
| langchain-defaults | 107.7 | 63 | 82 | 822.9 | 40 | 45.5 |
| llamaindex | 218.5 | 63 | 86 | 848.8 | 45 | 71.4 |

Per query, against the baseline at the same stage:

| implementation | stage | identical top-10 documents | recall@5 higher | recall@5 lower |
| --- | --- | --- | --- | --- |
| langchain | hybrid | 70/70 | 0 | 0 |
| langchain | reranked | 70/70 | 0 | 0 |
| langchain-defaults | hybrid | 0/70 | 6 | 6 |
| langchain-defaults | reranked | 10/70 | 4 | 2 |
| llamaindex | hybrid | 0/70 | 11 | 4 |
| llamaindex | reranked | 8/70 | 2 | 3 |

Per query, reranked against hybrid within each implementation:

| implementation | recall@5 up after rerank | recall@5 down after rerank |
| --- | --- | --- |
| baseline | 14 | 6 |
| langchain | 14 | 6 |
| langchain-defaults | 16 | 6 |
| llamaindex | 9 | 10 |

Shared code used by all implementations (chunking, tokeniser, settings): 75 lines.
