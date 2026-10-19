from __future__ import annotations

import math
from typing import Sequence

from .corpus import Query

FIELDS = ("recall_at_5", "recall_at_10", "hit_at_5", "mrr", "ndcg_at_5")


def to_documents(chunk_ids: Sequence[str], doc_of: dict[str, str]) -> list[str]:
    seen: dict[str, None] = {}
    for chunk_id in chunk_ids:
        seen.setdefault(doc_of[chunk_id], None)
    return list(seen)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for position, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], grades: dict[str, int], k: int) -> float:
    if not grades:
        return 0.0
    gain = sum(
        grades.get(doc, 0) / math.log2(position + 1)
        for position, doc in enumerate(retrieved[:k], start=1)
    )
    ideal = sum(
        grade / math.log2(position + 1)
        for position, grade in enumerate(sorted(grades.values(), reverse=True)[:k], start=1)
    )
    return gain / ideal if ideal else 0.0


def score_query(query: Query, retrieved: Sequence[str]) -> dict[str, float]:
    relevant = set(query.relevant)
    return {
        "recall_at_5": recall_at_k(retrieved, relevant, 5),
        "recall_at_10": recall_at_k(retrieved, relevant, 10),
        "hit_at_5": hit_at_k(retrieved, relevant, 5),
        "mrr": reciprocal_rank(retrieved, relevant),
        "ndcg_at_5": ndcg_at_k(retrieved, query.relevant, 5),
    }


def aggregate(scores: Sequence[dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {field: 0.0 for field in FIELDS}
    return {field: round(sum(s[field] for s in scores) / len(scores), 4) for field in FIELDS}


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
