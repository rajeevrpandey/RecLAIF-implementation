from __future__ import annotations

import math
from itertools import combinations


def precision_at_k(recommended: list[str], ground_truth: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    truth = set(ground_truth)
    hits = sum(1 for item in top_k if item in truth)
    return hits / k


def hit_at_k(recommended: list[str], ground_truth: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    truth = set(ground_truth)
    return 1.0 if any(item in truth for item in recommended[:k]) else 0.0


def ndcg_at_k(recommended: list[str], ground_truth: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    truth = set(ground_truth)
    dcg = 0.0
    for index, item in enumerate(recommended[:k], start=1):
        if item in truth:
            dcg += 1.0 / _log2(index + 1)

    ideal_hits = min(len(truth), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / _log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def valid_ratio(recommended: list[str], candidates: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    candidate_set = set(candidates)
    valid = sum(1 for item in top_k if item in candidate_set)
    return valid / len(top_k)


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


def _log2(value: int) -> float:
    return math.log2(value)
