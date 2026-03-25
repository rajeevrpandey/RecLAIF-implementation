from pathlib import Path

from reclaif.io import read_jsonl
from reclaif.llm import MockJudgeClient, MockRecommenderClient
from reclaif.pipeline import RecLaifPipeline, RecLaifPipelineConfig


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "beauty_sample.jsonl"
    output_dir = root / "artifacts"

    examples = read_jsonl(data_path)
    pipeline = RecLaifPipeline(
        recommender=MockRecommenderClient(),
        judge=MockJudgeClient(),
        config=RecLaifPipelineConfig(output_dir=output_dir, iterations=2),
    )
    result = pipeline.run(examples)

    print(f"sft records: {len(result.sft_records)}")
    print(f"preference pairs: {len(result.preference_pairs)}")
    print(f"artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()

