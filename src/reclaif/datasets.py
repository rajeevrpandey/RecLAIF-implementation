from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Literal

from reclaif.io import read_jsonl_rows, write_jsonl
from reclaif.schemas import RecommendationExample

DatasetName = Literal["esci", "beauty", "lastfm"]
SplitName = Literal["train", "valid", "test"]
CandidatePolicy = Literal["given", "random_from_catalog"]


@dataclass(slots=True)
class DatasetSplits:
    train: list[RecommendationExample]
    valid: list[RecommendationExample]
    test: list[RecommendationExample]


@dataclass(slots=True)
class CandidateSamplingConfig:
    policy: CandidatePolicy = "given"
    k: int = 50
    seed: int = 7
    include_ground_truth: bool = True


def load_official_splits(root: str | Path, dataset: DatasetName) -> DatasetSplits:
    root = Path(root)
    split_dir = root / dataset
    train = _read_split(split_dir / "train.jsonl")
    valid = _read_split(split_dir / "valid.jsonl")
    test = _read_split(split_dir / "test.jsonl")
    return DatasetSplits(train=train, valid=valid, test=test)


def build_catalog(splits: DatasetSplits) -> list[str]:
    catalog: set[str] = set()
    for rows in (splits.train, splits.valid, splits.test):
        for row in rows:
            catalog.update(row.candidates)
            catalog.update(row.ground_truth)
    return sorted(catalog)


def materialize_candidates(
    rows: list[RecommendationExample],
    *,
    catalog: list[str],
    config: CandidateSamplingConfig,
) -> list[RecommendationExample]:
    rng = Random(config.seed)
    updated: list[RecommendationExample] = []
    for row in rows:
        if config.policy == "given":
            updated.append(row)
            continue
        candidates = _random_candidates(
            catalog=catalog,
            ground_truth=row.ground_truth if config.include_ground_truth else [],
            k=config.k,
            rng=rng,
        )
        updated.append(
            RecommendationExample(
                example_id=row.example_id,
                task_type=row.task_type,
                domain=row.domain,
                user_id=row.user_id,
                context=row.context,
                candidates=candidates,
                ground_truth=row.ground_truth,
            )
        )
    return updated


def write_splits(output_dir: str | Path, splits: DatasetSplits) -> None:
    output_dir = Path(output_dir)
    write_jsonl(output_dir / "train.jsonl", (row.to_dict() for row in splits.train))
    write_jsonl(output_dir / "valid.jsonl", (row.to_dict() for row in splits.valid))
    write_jsonl(output_dir / "test.jsonl", (row.to_dict() for row in splits.test))


def _random_candidates(*, catalog: list[str], ground_truth: list[str], k: int, rng: Random) -> list[str]:
    if not catalog:
        return list(dict.fromkeys(ground_truth))[:k]
    unique_catalog = list(dict.fromkeys(catalog))
    sample_size = min(k, len(unique_catalog))
    sampled = rng.sample(unique_catalog, k=sample_size)
    for item in ground_truth:
        if item not in sampled:
            if len(sampled) < k:
                sampled.append(item)
            elif sampled:
                sampled[-1] = item
    return list(dict.fromkeys(sampled))


def _read_split(path: Path) -> list[RecommendationExample]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return [RecommendationExample.from_dict(row) for row in read_jsonl_rows(path)]
