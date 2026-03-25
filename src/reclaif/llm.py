from __future__ import annotations

import os
from dataclasses import dataclass
from random import Random
from typing import Any, Protocol

from reclaif.parsers import parse_recommender_output


class RecommenderClient(Protocol):
    def generate(self, prompt: str, *, temperature: float, top_p: float) -> str:
        ...


class JudgeClient(Protocol):
    def evaluate(self, prompt: str) -> str:
        ...


@dataclass(slots=True)
class OpenAIResponsesConfig:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    max_output_tokens: int = 700
    reasoning_effort: str | None = None
    base_url: str | None = None
    system_prompt: str | None = None


@dataclass(slots=True)
class TransformersGenerationConfig:
    model_name_or_path: str
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 700
    do_sample: bool = True
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True


class OpenAIResponsesTextClient:
    """Hosted text generation backend using the OpenAI Responses API."""

    def __init__(self, config: OpenAIResponsesConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is not installed. Install the 'hosted' extra to use OpenAI-backed clients."
            ) from exc

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set the {config.api_key_env} environment variable."
            )

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self._client = OpenAI(**client_kwargs)
        self._config = config

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
    ) -> str:
        request: dict[str, Any] = {
            "model": self._config.model,
            "input": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": self._config.max_output_tokens,
        }
        if self._config.reasoning_effort:
            request["reasoning"] = {"effort": self._config.reasoning_effort}
        if self._config.system_prompt:
            request["instructions"] = self._config.system_prompt

        response = self._client.responses.create(**request)
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()

        # Fallback for SDK / response-shape differences.
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()


class OpenAIRecommenderClient:
    def __init__(self, config: OpenAIResponsesConfig) -> None:
        system_prompt = config.system_prompt or (
            "You are a recommendation model that follows the prompt exactly, "
            "returns ranked recommendations, and includes concise reasoning."
        )
        self._client = OpenAIResponsesTextClient(
            OpenAIResponsesConfig(
                model=config.model,
                api_key_env=config.api_key_env,
                max_output_tokens=config.max_output_tokens,
                reasoning_effort=config.reasoning_effort,
                base_url=config.base_url,
                system_prompt=system_prompt,
            )
        )

    def generate(self, prompt: str, *, temperature: float, top_p: float) -> str:
        return self._client.generate_text(prompt, temperature=temperature, top_p=top_p)


class OpenAIJudgeClient:
    def __init__(self, config: OpenAIResponsesConfig) -> None:
        system_prompt = config.system_prompt or (
            "You are a strict recommendation judge. Score options carefully and pick the stronger one."
        )
        self._client = OpenAIResponsesTextClient(
            OpenAIResponsesConfig(
                model=config.model,
                api_key_env=config.api_key_env,
                max_output_tokens=config.max_output_tokens,
                reasoning_effort=config.reasoning_effort,
                base_url=config.base_url,
                system_prompt=system_prompt,
            )
        )

    def evaluate(self, prompt: str) -> str:
        return self._client.generate_text(prompt, temperature=0.2, top_p=0.95)


class TransformersTextGenerationClient:
    """Local text generation backend built on top of Hugging Face Transformers."""

    def __init__(self, config: TransformersGenerationConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are not installed. Install the 'train' extra to use local model backends."
            ) from exc

        dtype = config.torch_dtype
        if dtype == "auto":
            resolved_dtype = "auto"
        else:
            resolved_dtype = getattr(torch, dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            use_fast=config.use_fast_tokenizer,
            trust_remote_code=config.trust_remote_code,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            device_map=config.device_map,
            torch_dtype=resolved_dtype,
            trust_remote_code=config.trust_remote_code,
        )

        self._tokenizer = tokenizer
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        self._config = config

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
    ) -> str:
        prompt = self._truncate_prompt(prompt)
        max_new_tokens = self._effective_max_new_tokens()
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=self._config.do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()

    def _truncate_prompt(self, prompt: str) -> str:
        model_max_length = getattr(self._tokenizer, "model_max_length", None)
        if not model_max_length or model_max_length > 1_000_000:
            return prompt

        reserved_for_generation = self._effective_max_new_tokens()
        max_input_tokens = max(32, model_max_length - reserved_for_generation)
        tokenized = self._tokenizer(prompt, add_special_tokens=False, truncation=False)
        input_ids = tokenized.get("input_ids", [])
        if len(input_ids) <= max_input_tokens:
            return prompt

        truncated_ids = input_ids[-max_input_tokens:]
        return self._tokenizer.decode(truncated_ids, skip_special_tokens=True)

    def _effective_max_new_tokens(self) -> int:
        model_max_length = getattr(self._tokenizer, "model_max_length", None)
        if not model_max_length or model_max_length > 1_000_000:
            return self._config.max_new_tokens
        return min(self._config.max_new_tokens, max(32, model_max_length // 4))


class TransformersRecommenderClient:
    def __init__(self, config: TransformersGenerationConfig) -> None:
        self._client = TransformersTextGenerationClient(config)

    def generate(self, prompt: str, *, temperature: float, top_p: float) -> str:
        return self._client.generate_text(prompt, temperature=temperature, top_p=top_p)


class TransformersJudgeClient:
    def __init__(self, config: TransformersGenerationConfig) -> None:
        self._client = TransformersTextGenerationClient(config)

    def evaluate(self, prompt: str) -> str:
        return self._client.generate_text(prompt, temperature=0.2, top_p=0.95)


@dataclass(slots=True)
class MockRecommenderClient:
    seed: int = 7

    def generate(self, prompt: str, *, temperature: float, top_p: float) -> str:
        rng = Random(hash((prompt, round(temperature, 2), round(top_p, 2), self.seed)))
        catalog = _extract_candidates(prompt)
        picks = catalog[:]
        rng.shuffle(picks)
        picks = picks[:3]
        preferences = [
            "Practical fit with the user's recent behavior",
            "Topical similarity to the user's recent items",
            "Some variety without drifting off-task",
        ]
        return (
            "Key Preferences:\n"
            f"- Preference 1: {preferences[0]}\n"
            f"- Preference 2: {preferences[1]}\n"
            f"- Preference 3: {preferences[2]}\n\n"
            "Recommended Items (from most to least likely):\n"
            f"1. {picks[0]} - Strong topical fit with the user context.\n"
            f"2. {picks[1]} - Relevant while adding some variety.\n"
            f"3. {picks[2]} - A plausible supporting recommendation.\n"
        )


@dataclass(slots=True)
class MockJudgeClient:
    def evaluate(self, prompt: str) -> str:
        option_a = _extract_option(prompt, "Option A:")
        option_b = _extract_option(prompt, "Option B:")
        parsed_a = parse_recommender_output(option_a)
        parsed_b = parse_recommender_output(option_b)

        score_a = len(parsed_a.recommendations) + len(parsed_a.key_preferences)
        score_b = len(parsed_b.recommendations) + len(parsed_b.key_preferences)
        chosen = "A" if score_a >= score_b else "B"

        return (
            "Option A Evaluation:\n"
            "- Relevance Score: 4 - Mostly aligned with the user need.\n"
            "- Diversity Score: 3 - Moderately varied while staying relevant.\n"
            "- Explainability Score: 4 - Clear preferences and reasons.\n"
            "Option B Evaluation:\n"
            "- Relevance Score: 3 - Reasonable but slightly weaker alignment.\n"
            "- Diversity Score: 3 - Similar diversity profile.\n"
            "- Explainability Score: 3 - Explanations are acceptable.\n"
            "Decision:\n"
            f"- Chosen Option: {chosen}\n"
            "- Reasoning: The chosen option provides the stronger overall recommendation package.\n"
        )


def _extract_candidates(prompt: str) -> list[str]:
    marker = "Available Candidates:"
    if marker not in prompt:
        return ["Candidate A", "Candidate B", "Candidate C"]
    trailing = prompt.split(marker, 1)[1]
    candidate_line = trailing.strip().splitlines()[0]
    return [item.strip() for item in candidate_line.split(",") if item.strip()]


def _extract_option(prompt: str, marker: str) -> str:
    if marker not in prompt:
        return ""
    after = prompt.split(marker, 1)[1]
    lines = after.splitlines()
    buffer: list[str] = []
    for line in lines:
        if line.strip().startswith("Option ") and buffer:
            break
        if line.strip().startswith("Ground Truth"):
            break
        buffer.append(line)
    return "\n".join(buffer).strip()
