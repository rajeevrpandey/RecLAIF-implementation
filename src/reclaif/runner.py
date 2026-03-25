from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from reclaif.datasets import DatasetSplits, load_official_splits, materialize_candidates
from reclaif.experiment import ExperimentConfig
from reclaif.io import write_jsonl
from reclaif.llm import (
    JudgeClient,
    MockJudgeClient,
    MockRecommenderClient,
    RecommenderClient,
)
from reclaif.pipeline import RecLaifPipeline, RecLaifPipelineConfig
from reclaif.training import DPOTrainingConfig, SFTTrainingConfig, train_dpo_from_jsonl, train_sft_from_jsonl


@dataclass(slots=True)
class IterationRecord:
    iteration: int
    preference_pairs_path: str
    checkpoint_path: str
    reference_checkpoint_path: str


@dataclass(slots=True)
class RunManifest:
    dataset: str
    prompt_version: str
    seed: int
    teacher_model: str
    initial_model: str
    sft_checkpoint: str
    iterations: list[IterationRecord]

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "prompt_version": self.prompt_version,
            "seed": self.seed,
            "teacher_model": self.teacher_model,
            "initial_model": self.initial_model,
            "sft_checkpoint": self.sft_checkpoint,
            "iterations": [asdict(row) for row in self.iterations],
        }


def run_iterative_dpo(
    config: ExperimentConfig,
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    dry_run: bool = False,
) -> RunManifest:
    _set_seed(config.pipeline.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_official_splits(dataset_root, config.dataset)
    prepared = _prepare_splits_with_policy(splits, config)
    train_rows = prepared.train

    teacher = MockRecommenderClient()
    judge: JudgeClient = MockJudgeClient()
    recommender: RecommenderClient = MockRecommenderClient()

    pipeline = RecLaifPipeline(
        recommender=recommender,
        judge=judge,
        teacher=teacher,
        config=RecLaifPipelineConfig(
            top_k=config.pipeline.top_k,
            temperature=config.pipeline.temperature,
            top_p=config.pipeline.top_p,
            iterations=1,
            output_dir=output_dir / "iter0_teacher",
        ),
    )
    sft_records = pipeline.build_sft_dataset(train_rows)
    sft_data_path = output_dir / "sft_iter0.jsonl"
    write_jsonl(sft_data_path, (row.to_dict() for row in sft_records))
    sft_checkpoint = str(output_dir / "checkpoints" / "sft_iter0")
    if not dry_run:
        train_sft_from_jsonl(
            SFTTrainingConfig(
                model_name_or_path=config.models.recommender_model,
                train_jsonl=sft_data_path,
                output_dir=Path(sft_checkpoint),
                num_train_epochs=config.training.sft_epochs,
                seed=config.pipeline.seed,
            )
        )

    lineage: list[IterationRecord] = []
    current_model = sft_checkpoint if not dry_run else f"{sft_checkpoint}-dry-run"
    for iteration in range(1, config.pipeline.iterations + 1):
        iter_dir = output_dir / f"iter{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_pipeline = RecLaifPipeline(
            recommender=recommender,
            judge=judge,
            teacher=teacher,
            config=RecLaifPipelineConfig(
                top_k=config.pipeline.top_k,
                temperature=config.pipeline.temperature,
                top_p=config.pipeline.top_p,
                iterations=1,
                output_dir=iter_dir,
            ),
        )
        pairs = iter_pipeline.build_preference_pairs(train_rows, iteration=iteration)
        pairs_path = iter_dir / "preference_pairs.jsonl"
        write_jsonl(pairs_path, (row.to_dict() for row in pairs))
        next_model = str(output_dir / "checkpoints" / f"dpo_iter{iteration}")
        if not dry_run:
            train_dpo_from_jsonl(
                DPOTrainingConfig(
                    model_name_or_path=current_model,
                    reference_model_name_or_path=current_model,
                    train_jsonl=pairs_path,
                    output_dir=Path(next_model),
                    num_train_epochs=config.training.dpo_epochs,
                    beta=config.training.dpo_beta,
                    seed=config.pipeline.seed,
                )
            )
        lineage.append(
            IterationRecord(
                iteration=iteration,
                preference_pairs_path=str(pairs_path),
                checkpoint_path=next_model if not dry_run else f"{next_model}-dry-run",
                reference_checkpoint_path=current_model,
            )
        )
        current_model = next_model if not dry_run else f"{next_model}-dry-run"

    manifest = RunManifest(
        dataset=config.dataset,
        prompt_version=config.prompts.version,
        seed=config.pipeline.seed,
        teacher_model=config.models.teacher_model,
        initial_model=config.models.recommender_model,
        sft_checkpoint=sft_checkpoint if not dry_run else f"{sft_checkpoint}-dry-run",
        iterations=lineage,
    )
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return manifest


def _prepare_splits_with_policy(splits: DatasetSplits, config: ExperimentConfig) -> DatasetSplits:
    catalog = []
    for rows in (splits.train, splits.valid, splits.test):
        for row in rows:
            catalog.extend(row.candidates)
            catalog.extend(row.ground_truth)
    unique_catalog = sorted(set(catalog))
    return DatasetSplits(
        train=materialize_candidates(splits.train, catalog=unique_catalog, config=config.candidate_sampling),
        valid=materialize_candidates(splits.valid, catalog=unique_catalog, config=config.candidate_sampling),
        test=materialize_candidates(splits.test, catalog=unique_catalog, config=config.candidate_sampling),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
