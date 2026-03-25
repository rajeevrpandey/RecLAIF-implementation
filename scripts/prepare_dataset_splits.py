import argparse
from pathlib import Path

from reclaif.datasets import (
    CandidateSamplingConfig,
    DatasetSplits,
    build_catalog,
    load_official_splits,
    materialize_candidates,
    write_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RecLAIF dataset splits with deterministic candidate sampling.")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--dataset", choices=["esci", "beauty", "lastfm"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", choices=["given", "random_from_catalog"], default="given")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--exclude-ground-truth", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_official_splits(args.dataset_root, args.dataset)
    catalog = build_catalog(splits)
    config = CandidateSamplingConfig(
        policy=args.policy,
        k=args.k,
        seed=args.seed,
        include_ground_truth=not args.exclude_ground_truth,
    )
    prepared = DatasetSplits(
        train=materialize_candidates(splits.train, catalog=catalog, config=config),
        valid=materialize_candidates(splits.valid, catalog=catalog, config=config),
        test=materialize_candidates(splits.test, catalog=catalog, config=config),
    )
    out = Path(args.output_dir) / args.dataset
    write_splits(out, prepared)
    print(f"Prepared splits written to: {out}")


if __name__ == "__main__":
    main()
