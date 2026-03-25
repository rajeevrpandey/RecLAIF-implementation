import argparse
import json
from pathlib import Path

from reclaif.evaluation import evaluate_preference_pairs
from reclaif.io import read_jsonl, read_jsonl_rows
from reclaif.schemas import PreferencePair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated preference pairs.")
    parser.add_argument("--data", default="data/beauty_sample.jsonl")
    parser.add_argument("--pairs", default="artifacts/preference_pairs.jsonl")
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = read_jsonl(Path(args.data))
    rows = read_jsonl_rows(Path(args.pairs))
    pairs = [PreferencePair(**row) for row in rows]
    result = evaluate_preference_pairs(pairs, examples, k=args.k)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
