import argparse
from pathlib import Path

from reclaif.io import read_jsonl
from reclaif.llm import (
    MockJudgeClient,
    MockRecommenderClient,
    OpenAIJudgeClient,
    OpenAIRecommenderClient,
    OpenAIResponsesConfig,
    TransformersGenerationConfig,
    TransformersJudgeClient,
    TransformersRecommenderClient,
)
from reclaif.pipeline import RecLaifPipeline, RecLaifPipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RecLAIF data-generation pipeline.")
    parser.add_argument("--data", default="data/beauty_sample.jsonl")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--recommender-backend", choices=["mock", "openai", "transformers"], default="mock")
    parser.add_argument("--judge-backend", choices=["mock", "openai", "transformers"], default="mock")
    parser.add_argument("--teacher-backend", choices=["mock", "openai", "transformers"])

    parser.add_argument("--recommender-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--teacher-model")
    parser.add_argument("--openai-base-url")
    parser.add_argument("--reasoning-effort")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = read_jsonl(Path(args.data))
    recommender = build_recommender(args)
    judge = build_judge(args)
    teacher = build_teacher(args)

    pipeline = RecLaifPipeline(
        recommender=recommender,
        judge=judge,
        teacher=teacher,
        config=RecLaifPipelineConfig(
            output_dir=Path(args.output_dir),
            iterations=args.iterations,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
    )
    result = pipeline.run(examples)

    print(f"sft records: {len(result.sft_records)}")
    print(f"preference pairs: {len(result.preference_pairs)}")
    print(f"artifacts written to: {args.output_dir}")


def build_recommender(args: argparse.Namespace):
    if args.recommender_backend == "mock":
        return MockRecommenderClient()
    if args.recommender_backend == "openai":
        if not args.recommender_model:
            raise ValueError("--recommender-model is required for the openai backend.")
        return OpenAIRecommenderClient(
            OpenAIResponsesConfig(
                model=args.recommender_model,
                base_url=args.openai_base_url,
                reasoning_effort=args.reasoning_effort,
            )
        )
    if not args.recommender_model:
        raise ValueError("--recommender-model is required for the transformers backend.")
    return TransformersRecommenderClient(
        TransformersGenerationConfig(model_name_or_path=args.recommender_model)
    )


def build_judge(args: argparse.Namespace):
    if args.judge_backend == "mock":
        return MockJudgeClient()
    if args.judge_backend == "openai":
        if not args.judge_model:
            raise ValueError("--judge-model is required for the openai backend.")
        return OpenAIJudgeClient(
            OpenAIResponsesConfig(
                model=args.judge_model,
                base_url=args.openai_base_url,
                reasoning_effort=args.reasoning_effort,
            )
        )
    if not args.judge_model:
        raise ValueError("--judge-model is required for the transformers backend.")
    return TransformersJudgeClient(
        TransformersGenerationConfig(model_name_or_path=args.judge_model)
    )


def build_teacher(args: argparse.Namespace):
    if args.teacher_backend is None:
        return None
    if args.teacher_backend == "mock":
        return MockRecommenderClient()
    if args.teacher_backend == "openai":
        if not args.teacher_model:
            raise ValueError("--teacher-model is required for the openai teacher backend.")
        return OpenAIRecommenderClient(
            OpenAIResponsesConfig(
                model=args.teacher_model,
                base_url=args.openai_base_url,
                reasoning_effort=args.reasoning_effort,
            )
        )
    if not args.teacher_model:
        raise ValueError("--teacher-model is required for the transformers teacher backend.")
    return TransformersRecommenderClient(
        TransformersGenerationConfig(model_name_or_path=args.teacher_model)
    )


if __name__ == "__main__":
    main()
