import argparse
import json

from reclaif.experiment import ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and print a version-locked experiment config.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    print(json.dumps(config.to_dict(), indent=2))


if __name__ == "__main__":
    main()
