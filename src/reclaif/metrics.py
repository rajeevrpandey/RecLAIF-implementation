from __future__ import annotations

from itertools import combinations


def overlap_ratio(items_a: list[str], items_b: list[str]) -> float:
    if not items_a and not items_b:
        return 1.0
    a = set(items_a)
    b = set(items_b)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def lexical_diversity(items: list[str]) -> float:
    if len(items) < 2:
        return 0.0
    distances: list[float] = []
    for left, right in combinations(items, 2):
        left_tokens = set(left.lower().split())
        right_tokens = set(right.lower().split())
        union = left_tokens | right_tokens
        if not union:
            distances.append(0.0)
            continue
        jaccard_similarity = len(left_tokens & right_tokens) / len(union)
        distances.append(1.0 - jaccard_similarity)
    return sum(distances) / len(distances)

