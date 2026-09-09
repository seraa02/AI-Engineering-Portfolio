"""
LLM-as-judge: uses Claude to score RAG answers on the rubric criteria.
Also implements the DeepEval metric interface.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import anthropic

from src.rubric import CRITERIA, Criterion, criterion_names, weighted_avg

# Prompt version for cache invalidation
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a calibrated evaluator assessing the quality of an AI system's answer to a question about SEC 10-K filings.

You will score the answer on specific criteria. For each criterion, you must:
1. Read the criterion description and anchor examples carefully
2. Assign a score within the valid range
3. Provide a brief reasoning (1-2 sentences)

IMPORTANT: If RETRIEVED EVIDENCE is provided below, factual_accuracy and hallucination_avoidance
must be judged by checking the answer's claims AGAINST that evidence — a claim not supported by
the evidence, or contradicted by it, is inaccurate/hallucinated even if it sounds plausible. A
non-answer ("I don't know") is NOT automatically factually accurate just because it makes no
false claims — score it low on completeness and factual_accuracy, since it fails to state the
fact the evidence actually supports.

If NO evidence is provided, you cannot verify claims against a source: score factual_accuracy
and hallucination_avoidance based only on internal plausibility and whether citations are present,
and say so explicitly in your reasoning — do not assert a claim is "accurate" without evidence to
check it against."""


def _build_scoring_schema(criteria: list[Criterion]) -> dict:
    """Build JSON Schema for structured scoring output."""
    properties = {}
    for c in criteria:
        properties[c.name] = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": f"1-2 sentence reasoning for the {c.name} score, written BEFORE deciding the score.",
                },
                "score": {
                    "type": "integer",
                    "enum": list(range(1, c.scale + 1)),
                    "description": f"Score from 1 to {c.scale} for {c.name}",
                },
            },
            # NOTE: reasoning is listed first in "properties" and "required" so the
            # model is constrained to write reasoning tokens before the score token
            # (reasoning-before-score improves consistency; see README calibration notes).
            "required": ["reasoning", "score"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


@dataclass
class JudgmentResult:
    scores: dict[str, int]         # criterion_name -> score
    reasoning: dict[str, str]      # criterion_name -> reasoning text
    weighted_score: float           # normalized 0.0–1.0
    input_tokens: int
    output_tokens: int
    latency_ms: float


def judge(
    client: anthropic.Anthropic,
    question: str,
    answer: str,
    model: str = "claude-haiku-4-5",
    criteria: Optional[list[str]] = None,
    context: Optional[str] = None,
) -> JudgmentResult:
    """
    Score one answer on all (or specified) rubric criteria.

    `context` is the retrieved evidence (source chunks) the answer was supposedly grounded
    in — e.g. Project 1's retrieved passages, or a golden dataset's `source_evidence`. When
    provided, factual_accuracy and hallucination_avoidance are checked against it directly,
    which is what actually makes those two criteria meaningful for a RAG system (without it,
    the judge can only check internal plausibility, not real correctness — see SYSTEM_PROMPT).
    """
    active_criteria = [c for c in CRITERIA if criteria is None or c.name in (criteria or [])]
    schema = _build_scoring_schema(active_criteria)

    # Build rubric block for the prompt
    rubric_lines = []
    for c in active_criteria:
        anchor_block = "\n".join(
            f"  {score}: {a.label} — e.g., \"{a.example}\""
            for score, a in sorted(c.anchors.items())
        )
        rubric_lines.append(
            f"**{c.name}** (1–{c.scale})\n{c.description}\n{anchor_block}"
        )
    rubric_text = "\n\n".join(rubric_lines)

    context_block = f"RETRIEVED EVIDENCE:\n{context}\n\n" if context else ""

    user_content = (
        f"QUESTION: {question}\n\n"
        f"ANSWER TO EVALUATE:\n{answer}\n\n"
        f"{context_block}"
        f"RUBRIC:\n{rubric_text}\n\n"
        "Score the answer on each criterion using the rubric above."
    )

    start = time.monotonic()
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": user_content}],
    )
    latency_ms = (time.monotonic() - start) * 1000

    raw = json.loads(next(b.text for b in response.content if b.type == "text"))

    scores = {name: raw[name]["score"] for name in raw}
    reasoning = {name: raw[name]["reasoning"] for name in raw}

    return JudgmentResult(
        scores=scores,
        reasoning=reasoning,
        weighted_score=weighted_avg(scores),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=latency_ms,
    )


PAIRWISE_SYSTEM_PROMPT = """You are a calibrated evaluator comparing two candidate answers to the
same question about SEC 10-K filings. Decide which answer is better overall (factual accuracy,
citation quality, completeness, coherence, and avoidance of hallucination all matter — weigh
factual accuracy and hallucination avoidance most heavily).

IMPORTANT: Base your judgment ONLY on the two answer texts provided. First write your reasoning,
THEN give your verdict."""

_PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "1-3 sentence comparison of A and B, written before the verdict."},
        "winner": {"type": "string", "enum": ["A", "B"], "description": "Which answer is better overall."},
    },
    "required": ["reasoning", "winner"],
    "additionalProperties": False,
}


def judge_pairwise(
    client: anthropic.Anthropic,
    question: str,
    answer_a: str,
    answer_b: str,
    model: str = "claude-haiku-4-5",
) -> str:
    """
    Ask the judge which of two answers (A or B) is better. Used for the real position-bias
    check in src/bias.py::measure_position_bias — call this twice per pair, once as
    (question, x, y) and once as (question, y, x), and see if the verdict flips.

    Returns "A" or "B".
    """
    user_content = (
        f"QUESTION: {question}\n\n"
        f"ANSWER A:\n{answer_a}\n\n"
        f"ANSWER B:\n{answer_b}\n\n"
        "Which answer is better overall?"
    )
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=PAIRWISE_SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": _PAIRWISE_SCHEMA}},
        messages=[{"role": "user", "content": user_content}],
    )
    raw = json.loads(next(b.text for b in response.content if b.type == "text"))
    return raw["winner"]


# ---------------------------------------------------------------------------
# DeepEval metric interface
# ---------------------------------------------------------------------------

try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase

    class RAGJudgeMetric(BaseMetric):
        """DeepEval-compatible metric that uses our rubric judge."""

        def __init__(
            self,
            client: anthropic.Anthropic,
            threshold: float = 0.6,
            model: str = "claude-haiku-4-5",
        ):
            self.client = client
            self.threshold = threshold
            self.model = model
            self._score: float = 0.0
            self._reason: str = ""

        @property
        def score(self) -> float:
            return self._score

        @property
        def reason(self) -> str:
            return self._reason

        def measure(self, test_case: LLMTestCase) -> float:
            retrieval_context = getattr(test_case, "retrieval_context", None)
            context = "\n---\n".join(retrieval_context) if retrieval_context else None
            result = judge(
                client=self.client,
                question=test_case.input,
                answer=test_case.actual_output,
                model=self.model,
                context=context,
            )
            self._score = result.weighted_score
            self._reason = "; ".join(
                f"{k}={v}" for k, v in result.scores.items()
            )
            return self._score

        def is_successful(self) -> bool:
            return self._score >= self.threshold

        @property
        def name(self) -> str:
            return "RAGJudgeMetric"

except ImportError:
    # deepeval not installed — provide a stub that explains itself
    class RAGJudgeMetric:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("deepeval not installed. Run: pip install deepeval")
