from __future__ import annotations

from textwrap import dedent

from reclaif.schemas import RecommendationExample


def build_recommender_prompt(example: RecommendationExample, top_k: int = 3) -> str:
    return dedent(
        f"""
        You are a {example.domain} recommendation engine.
        Your task is to recommend the best items based on the user's context.

        User Context:
        {example.context}

        Available Candidates:
        {", ".join(example.candidates)}

        Task:
        1. Identify 2-3 key preferences inferred from the user context.
        2. Select the top {top_k} items from the candidate list that best match those preferences.
        3. Ensure the recommendations are relevant, reasonably diverse, and ordered from most to least likely.
        4. Provide one short reason for each recommendation.

        Output Format:
        Key Preferences:
        - Preference 1: ...
        - Preference 2: ...

        Recommended Items (from most to least likely):
        1. Item Name - Reason
        2. Item Name - Reason
        3. Item Name - Reason
        """
    ).strip()


def build_sft_teacher_prompt(example: RecommendationExample, top_k: int = 3) -> str:
    return build_recommender_prompt(example, top_k=top_k)


def build_judge_prompt(
    example: RecommendationExample,
    option_a: str,
    option_b: str,
) -> str:
    label_text = ", ".join(example.ground_truth) if example.ground_truth else "Not provided"
    return dedent(
        f"""
        You are an expert evaluator tasked with assessing two recommendation outputs
        based on relevance, diversity, and explainability.

        User Context:
        {example.context}

        Available Candidates:
        {", ".join(example.candidates)}

        Option A:
        {option_a}

        Option B:
        {option_b}

        Ground Truth / Preferred Target:
        {label_text}

        Criteria for Evaluation:
        - Relevance: How well do the recommendations match the likely user need or target items?
        - Diversity: Do the recommendations avoid being unnecessarily redundant while still staying on-topic?
        - Explainability: Are the inferred preferences and reasons clear, faithful, and useful?

        Task:
        1. Evaluate each option on all three criteria.
        2. Give a 0-5 score and a short explanation for each criterion.
        3. Choose the better option overall.

        Output Format:
        Option A Evaluation:
        - Relevance Score: [0-5] - [Explanation]
        - Diversity Score: [0-5] - [Explanation]
        - Explainability Score: [0-5] - [Explanation]
        Option B Evaluation:
        - Relevance Score: [0-5] - [Explanation]
        - Diversity Score: [0-5] - [Explanation]
        - Explainability Score: [0-5] - [Explanation]
        Decision:
        - Chosen Option: [A or B]
        - Reasoning: [Summary]
        """
    ).strip()

