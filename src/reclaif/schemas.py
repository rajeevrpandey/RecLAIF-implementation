from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


TaskType = Literal["retrieval", "sequential"]


@dataclass(slots=True)
class RecommendationExample:
    example_id: str
    task_type: TaskType
    domain: str
    user_id: str
    context: str
    candidates: list[str]
    ground_truth: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict) -> "RecommendationExample":
        return cls(
            example_id=payload["example_id"],
            task_type=payload["task_type"],
            domain=payload.get("domain", "generic"),
            user_id=payload["user_id"],
            context=payload["context"],
            candidates=list(payload["candidates"]),
            ground_truth=list(payload.get("ground_truth", [])),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class RecommendationOutput:
    raw_text: str
    key_preferences: list[str]
    recommendations: list[str]
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class CriterionScore:
    score: int
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class JudgeOutput:
    option_a_relevance: CriterionScore
    option_a_diversity: CriterionScore
    option_a_explainability: CriterionScore
    option_b_relevance: CriterionScore
    option_b_diversity: CriterionScore
    option_b_explainability: CriterionScore
    chosen_option: Literal["A", "B"]
    reasoning: str
    raw_text: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


@dataclass(slots=True)
class SFTRecord:
    prompt: str
    response: str
    example_id: str
    iteration: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    judge_reasoning: str
    example_id: str
    iteration: int

    def to_dict(self) -> dict:
        return asdict(self)

