from __future__ import annotations

import re

from reclaif.schemas import CriterionScore, JudgeOutput, RecommendationOutput


def _extract_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_recommender_output(text: str) -> RecommendationOutput:
    lines = _extract_lines(text)
    key_preferences: list[str] = []
    recommendations: list[str] = []
    reasons: list[str] = []

    in_preferences = False
    in_recommendations = False

    for line in lines:
        lowered = line.lower()
        if lowered.startswith("key preferences"):
            in_preferences = True
            in_recommendations = False
            continue
        if lowered.startswith("recommended items") or lowered.startswith("recommended artists"):
            in_preferences = False
            in_recommendations = True
            continue
        if in_preferences and line.startswith("-"):
            key_preferences.append(line.lstrip("- ").strip())
        elif in_recommendations:
            match = re.match(r"^\d+\.\s*(.+?)\s*-\s*(.+)$", line)
            if match:
                recommendations.append(match.group(1).strip())
                reasons.append(match.group(2).strip())

    return RecommendationOutput(
        raw_text=text,
        key_preferences=key_preferences,
        recommendations=recommendations,
        reasons=reasons,
    )


def _parse_score_line(text: str, prefix: str) -> CriterionScore:
    pattern = re.compile(rf"{re.escape(prefix)}\s*:\s*\[?(\d)\]?\s*-\s*(.+)", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        fallback = re.compile(rf"{re.escape(prefix)}\s*:\s*(\d)\s*-\s*(.+)", re.IGNORECASE)
        match = fallback.search(text)
    if not match:
        return CriterionScore(score=0, explanation="Could not parse score.")
    return CriterionScore(score=int(match.group(1)), explanation=match.group(2).strip())


def parse_judge_output(text: str) -> JudgeOutput:
    option_a_block, option_b_block = _extract_option_blocks(text)
    option_a_relevance = _parse_score_line(option_a_block, "- Relevance Score")
    first_diversity = _parse_score_line(option_a_block, "- Diversity Score")
    first_explainability = _parse_score_line(option_a_block, "- Explainability Score")
    second_relevance = _parse_score_line(option_b_block, "- Relevance Score")
    second_diversity = _parse_score_line(option_b_block, "- Diversity Score")
    second_explainability = _parse_score_line(option_b_block, "- Explainability Score")

    chosen_match = re.search(r"Chosen Option:\s*\[?([AB])\]?", text, re.IGNORECASE)
    reasoning_match = re.search(r"Reasoning:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    option_a_total = _weighted_total(option_a_relevance, first_diversity, first_explainability)
    option_b_total = _weighted_total(second_relevance, second_diversity, second_explainability)

    return JudgeOutput(
        option_a_relevance=option_a_relevance,
        option_a_diversity=first_diversity,
        option_a_explainability=first_explainability,
        option_b_relevance=second_relevance,
        option_b_diversity=second_diversity,
        option_b_explainability=second_explainability,
        chosen_option=(chosen_match.group(1).upper() if chosen_match else None),
        reasoning=reasoning_match.group(1).strip() if reasoning_match else "",
        raw_text=text,
        option_a_total=option_a_total,
        option_b_total=option_b_total,
    )


def _weighted_total(
    relevance: CriterionScore,
    diversity: CriterionScore,
    explainability: CriterionScore,
) -> float:
    return 0.5 * relevance.score + 0.2 * diversity.score + 0.3 * explainability.score


def _extract_option_blocks(text: str) -> tuple[str, str]:
    pattern = re.compile(
        r"Option A Evaluation:(.*?)Option B Evaluation:(.*?)(?:Decision:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if match:
        return match.group(1), match.group(2)
    return text, text
