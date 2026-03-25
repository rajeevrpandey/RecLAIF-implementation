from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from reclaif.datasets import CandidateSamplingConfig, DatasetName


@dataclass(slots=True)
class ModelConfig:
    recommender_model: str
    judge_model: str
    teacher_model: str


@dataclass(slots=True)
class PromptConfig:
    version: str = "v1"


@dataclass(slots=True)
class PipelineConfig:
    top_k: int = 3
    iterations: int = 2
    temperature: float = 1.0
    top_p: float = 0.9
    seed: int = 7


@dataclass(slots=True)
class TrainingConfig:
    dpo_beta: float = 0.1
    sft_epochs: float = 1.0
    dpo_epochs: float = 1.0


@dataclass(slots=True)
class ExperimentConfig:
    dataset: DatasetName
    models: ModelConfig
    prompts: PromptConfig
    pipeline: PipelineConfig
    candidate_sampling: CandidateSamplingConfig
    training: TrainingConfig

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            dataset=payload["dataset"],
            models=ModelConfig(**payload["models"]),
            prompts=PromptConfig(**payload.get("prompts", {})),
            pipeline=PipelineConfig(**payload.get("pipeline", {})),
            candidate_sampling=CandidateSamplingConfig(**payload.get("candidate_sampling", {})),
            training=TrainingConfig(**payload.get("training", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)
