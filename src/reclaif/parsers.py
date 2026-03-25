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
    option_a_relevance = _parse_score_line(text, "- Relevance Score")
    first_diversity = _nth_score(text, "- Diversity Score", 1)
    first_explainability = _nth_score(text, "- Explainability Score", 1)
    second_relevance = _nth_score(text, "- Relevance Score", 2)
    second_diversity = _nth_score(text, "- Diversity Score", 2)
    second_explainability = _nth_score(text, "- Explainability Score", 2)

    chosen_match = re.search(r"Chosen Option:\s*\[?([AB])\]?", text, re.IGNORECASE)
    reasoning_match = re.search(r"Reasoning:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    return JudgeOutput(
        option_a_relevance=option_a_relevance,
        option_a_diversity=first_diversity,
        option_a_explainability=first_explainability,
        option_b_relevance=second_relevance,
        option_b_diversity=second_diversity,
        option_b_explainability=second_explainability,
        chosen_option=(chosen_match.group(1).upper() if chosen_match else "A"),
        reasoning=reasoning_match.group(1).strip() if reasoning_match else "",
        raw_text=text,
    )


def _nth_score(text: str, prefix: str, occurrence: int) -> CriterionScore:
    pattern = re.compile(rf"{re.escape(prefix)}\s*:\s*\[?(\d)\]?\s*-\s*(.+)", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if len(matches) >= occurrence:
        match = matches[occurrence - 1]
        return CriterionScore(score=int(match.group(1)), explanation=match.group(2).strip())
    return CriterionScore(score=0, explanation="Could not parse score.")

