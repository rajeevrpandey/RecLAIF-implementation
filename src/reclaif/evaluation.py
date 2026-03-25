from __future__ import annotations

from dataclasses import dataclass

from reclaif.metrics import hit_at_k, lexical_diversity, ndcg_at_k, precision_at_k, valid_ratio
from reclaif.parsers import parse_recommender_output
from reclaif.schemas import PreferencePair, RecommendationExample


@dataclass(slots=True)
class EvaluationResult:
    examples: int
    precision_at_3: float
    hit_at_1: float
    ndcg_at_3: float
    valid_ratio_at_3: float
    diversity: float

    def to_dict(self) -> dict:
        return {
            "examples": self.examples,
            "precision_at_3": self.precision_at_3,
            "hit_at_1": self.hit_at_1,
            "ndcg_at_3": self.ndcg_at_3,
            "valid_ratio_at_3": self.valid_ratio_at_3,
            "diversity": self.diversity,
        }


@dataclass(slots=True)
class ESCIRetrievalResult:
    examples: int
    precision_at_5: float
    ndcg_at_5: float
    explainability: float

    def to_dict(self) -> dict:
        return {
            "examples": self.examples,
            "precision_at_5": self.precision_at_5,
            "ndcg_at_5": self.ndcg_at_5,
            "explainability": self.explainability,
        }


@dataclass(slots=True)
class SequentialResult:
    examples: int
    valid_ratio: float
    hit_at_1: float
    ndcg_at_3: float
    diversity: float
    explainability: float

    def to_dict(self) -> dict:
        return {
            "examples": self.examples,
            "valid_ratio": self.valid_ratio,
            "hit_at_1": self.hit_at_1,
            "ndcg_at_3": self.ndcg_at_3,
            "diversity": self.diversity,
            "explainability": self.explainability,
        }


def evaluate_preference_pairs(
    pairs: list[PreferencePair],
    examples: list[RecommendationExample],
    *,
    k: int = 3,
) -> EvaluationResult:
    by_example = {row.example_id: row for row in examples}
    precisions: list[float] = []
    hits: list[float] = []
    ndcgs: list[float] = []
    valids: list[float] = []
    diversities: list[float] = []

    for pair in pairs:
        example = by_example.get(pair.example_id)
        if example is None:
            continue
        parsed = parse_recommender_output(pair.chosen)
        recs = parsed.recommendations
        precisions.append(precision_at_k(recs, example.ground_truth, k))
        hits.append(hit_at_k(recs, example.ground_truth, 1))
        ndcgs.append(ndcg_at_k(recs, example.ground_truth, k))
        valids.append(valid_ratio(recs, example.candidates, k))
        diversities.append(lexical_diversity(recs[:k]))

    if not precisions:
        return EvaluationResult(
            examples=0,
            precision_at_3=0.0,
            hit_at_1=0.0,
            ndcg_at_3=0.0,
            valid_ratio_at_3=0.0,
            diversity=0.0,
        )

    return EvaluationResult(
        examples=len(precisions),
        precision_at_3=sum(precisions) / len(precisions),
        hit_at_1=sum(hits) / len(hits),
        ndcg_at_3=sum(ndcgs) / len(ndcgs),
        valid_ratio_at_3=sum(valids) / len(valids),
        diversity=sum(diversities) / len(diversities),
    )


def evaluate_esci_retrieval(
    pairs: list[PreferencePair],
    examples: list[RecommendationExample],
) -> ESCIRetrievalResult:
    by_example = {row.example_id: row for row in examples}
    precisions: list[float] = []
    ndcgs: list[float] = []
    explainability_scores: list[float] = []
    for pair in pairs:
        example = by_example.get(pair.example_id)
        if example is None:
            continue
        parsed = parse_recommender_output(pair.chosen)
        precisions.append(precision_at_k(parsed.recommendations, example.ground_truth, 5))
        ndcgs.append(ndcg_at_k(parsed.recommendations, example.ground_truth, 5))
        explainability_scores.append(explainability_score(parsed.key_preferences, parsed.reasons))
    if not precisions:
        return ESCIRetrievalResult(examples=0, precision_at_5=0.0, ndcg_at_5=0.0, explainability=0.0)
    return ESCIRetrievalResult(
        examples=len(precisions),
        precision_at_5=sum(precisions) / len(precisions),
        ndcg_at_5=sum(ndcgs) / len(ndcgs),
        explainability=sum(explainability_scores) / len(explainability_scores),
    )


def evaluate_sequential_recommendation(
    pairs: list[PreferencePair],
    examples: list[RecommendationExample],
) -> SequentialResult:
    by_example = {row.example_id: row for row in examples}
    valids: list[float] = []
    hits: list[float] = []
    ndcgs: list[float] = []
    diversities: list[float] = []
    explainability_scores: list[float] = []
    for pair in pairs:
        example = by_example.get(pair.example_id)
        if example is None:
            continue
        parsed = parse_recommender_output(pair.chosen)
        recs = parsed.recommendations
        valids.append(valid_ratio(recs, example.candidates, 3))
        hits.append(hit_at_k(recs, example.ground_truth, 1))
        ndcgs.append(ndcg_at_k(recs, example.ground_truth, 3))
        diversities.append(lexical_diversity(recs[:3]))
        explainability_scores.append(explainability_score(parsed.key_preferences, parsed.reasons))
    if not valids:
        return SequentialResult(
            examples=0,
            valid_ratio=0.0,
            hit_at_1=0.0,
            ndcg_at_3=0.0,
            diversity=0.0,
            explainability=0.0,
        )
    return SequentialResult(
        examples=len(valids),
        valid_ratio=sum(valids) / len(valids),
        hit_at_1=sum(hits) / len(hits),
        ndcg_at_3=sum(ndcgs) / len(ndcgs),
        diversity=sum(diversities) / len(diversities),
        explainability=sum(explainability_scores) / len(explainability_scores),
    )


def explainability_score(key_preferences: list[str], reasons: list[str]) -> float:
    """
    Heuristic explainability protocol (0-5):
    - coverage of key preferences (2-3 expected)
    - reason completeness (reason count matches recommendation count)
    - reason clarity (minimum token length threshold)
    """
    score = 0.0
    pref_count = len([item for item in key_preferences if item.strip()])
    if pref_count >= 2:
        score += 2.0
    elif pref_count == 1:
        score += 1.0

    non_empty_reasons = [item for item in reasons if item.strip()]
    if non_empty_reasons:
        score += 1.5
    if len(non_empty_reasons) >= 3:
        score += 1.0

    avg_reason_len = sum(len(item.split()) for item in non_empty_reasons) / len(non_empty_reasons) if non_empty_reasons else 0.0
    if avg_reason_len >= 4:
        score += 0.5
    return min(5.0, score)
