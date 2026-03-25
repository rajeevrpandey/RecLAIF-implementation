import argparse
from pathlib import Path

from reclaif.experiment import ExperimentConfig
from reclaif.runner import run_iterative_dpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Iter1..IterN RecLAIF DPO loop with lineage tracking.")
    parser.add_argument("--config", required=True, help="Path to a version-locked experiment config JSON.")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--output-dir", default="runs/reclaif")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual SFT/DPO training calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    manifest = run_iterative_dpo(
        config,
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
    )
    print(f"Run manifest written for dataset={manifest.dataset} to: {Path(args.output_dir) / 'run_manifest.json'}")
    print(f"Iterations recorded: {len(manifest.iterations)}")


if __name__ == "__main__":
    main()
