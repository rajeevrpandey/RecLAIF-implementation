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
