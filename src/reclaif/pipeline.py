from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from reclaif.io import write_jsonl
from reclaif.llm import JudgeClient, RecommenderClient
from reclaif.parsers import parse_judge_output
from reclaif.prompts import build_judge_prompt, build_recommender_prompt, build_sft_teacher_prompt
from reclaif.schemas import PreferencePair, RecommendationExample, SFTRecord


@dataclass(slots=True)
class RecLaifPipelineConfig:
    top_k: int = 3
    temperature: float = 1.0
    top_p: float = 0.9
    iterations: int = 2
    output_dir: Path = Path("artifacts")


@dataclass(slots=True)
class PipelineResult:
    sft_records: list[SFTRecord] = field(default_factory=list)
    preference_pairs: list[PreferencePair] = field(default_factory=list)


@dataclass(slots=True)
class RecLaifPipeline:
    recommender: RecommenderClient
    judge: JudgeClient
    config: RecLaifPipelineConfig
    teacher: RecommenderClient | None = None

    def build_sft_dataset(self, examples: list[RecommendationExample]) -> list[SFTRecord]:
        records: list[SFTRecord] = []
        teacher = self.teacher or self.recommender
        for example in examples:
            prompt = build_sft_teacher_prompt(example, top_k=self.config.top_k)
            teacher_response = teacher.generate(
                prompt,
                temperature=0.2,
                top_p=0.95,
            )
            records.append(
                SFTRecord(
                    prompt=prompt,
                    response=teacher_response,
                    example_id=example.example_id,
                    iteration=0,
                )
            )
        return records

    def build_preference_pairs(
        self,
        examples: list[RecommendationExample],
        *,
        iteration: int,
    ) -> list[PreferencePair]:
        pairs: list[PreferencePair] = []
        for example in examples:
            prompt = build_recommender_prompt(example, top_k=self.config.top_k)
            option_a = self.recommender.generate(
                prompt,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            option_b = self.recommender.generate(
                prompt + f"\n\nSampling seed hint: {iteration}",
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            judge_prompt = build_judge_prompt(example, option_a=option_a, option_b=option_b)
            judge_response = self.judge.evaluate(judge_prompt)
            parsed = parse_judge_output(judge_response)

            chosen_option = parsed.chosen_option
            if chosen_option is None:
                chosen_option = "A" if parsed.option_a_total >= parsed.option_b_total else "B"

            chosen = option_a if chosen_option == "A" else option_b
            rejected = option_b if chosen_option == "A" else option_a

            pairs.append(
                PreferencePair(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    judge_reasoning=parsed.reasoning,
                    example_id=example.example_id,
                    iteration=iteration,
                )
            )
        return pairs

    def run(self, examples: list[RecommendationExample]) -> PipelineResult:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        sft_records = self.build_sft_dataset(examples)
        all_pairs: list[PreferencePair] = []

        for iteration in range(1, self.config.iterations + 1):
            all_pairs.extend(self.build_preference_pairs(examples, iteration=iteration))

        write_jsonl(output_dir / "sft_records.jsonl", (row.to_dict() for row in sft_records))
        write_jsonl(
            output_dir / "preference_pairs.jsonl",
            (row.to_dict() for row in all_pairs),
        )

        return PipelineResult(sft_records=sft_records, preference_pairs=all_pairs)
