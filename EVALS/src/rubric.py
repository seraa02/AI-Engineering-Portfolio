"""
Rubric definition for evaluating RAG system answers.

Criteria evaluated on ordinal scales:
  factual_accuracy: 1-5
  citation_quality: 1-5
  completeness:     1-5
  coherence:        1-3
  hallucination_avoidance: 1-3

Each criterion has anchor examples per scale point to improve
labeler consistency (reduce human self-disagreement).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CriterionAnchor:
    score: int
    example: str
    label: str


@dataclass
class Criterion:
    name: str
    description: str
    scale: int  # max score
    anchors: dict[int, CriterionAnchor]
    weight: float = 1.0  # for weighted average


CRITERIA: list[Criterion] = [
    Criterion(
        name="factual_accuracy",
        description="Are all facts in the answer correct and supported by the retrieved evidence?",
        scale=5,
        weight=1.5,
        anchors={
            1: CriterionAnchor(1, "Answer states AMD is a subsidiary of Intel.", "Completely wrong — contradicted by source"),
            2: CriterionAnchor(2, "Says NVIDIA acquired Mellanox in 2019 (correct year: 2020).", "Contains significant error"),
            3: CriterionAnchor(3, "Says NVIDIA competes with AMD in data center (correct) but omits Intel.", "Mostly correct, minor omission"),
            4: CriterionAnchor(4, "Correctly names all competitors but uses slightly imprecise phrasing.", "Accurate with minor imprecision"),
            5: CriterionAnchor(5, "Every claim matches the source text verbatim.", "Perfectly accurate"),
        }
    ),
    Criterion(
        name="citation_quality",
        description="Do citations exist, reference real retrieved chunks, and directly support the claims made?",
        scale=5,
        weight=1.5,
        anchors={
            1: CriterionAnchor(1, "No citations provided.", "No citations"),
            2: CriterionAnchor(2, "Citations exist but reference wrong company's filing.", "Citations don't match claims"),
            3: CriterionAnchor(3, "Claims cited but the cited passage is only tangentially related.", "Weak citation support"),
            4: CriterionAnchor(4, "Most claims cited to relevant passages; one uncited.", "Good but incomplete"),
            5: CriterionAnchor(5, "Every claim has a direct citation from the filing that supports it exactly.", "Perfect citation coverage"),
        }
    ),
    Criterion(
        name="completeness",
        description="Does the answer address all parts of the question asked?",
        scale=5,
        weight=1.0,
        anchors={
            1: CriterionAnchor(1, "Question asks about competitors; answer discusses revenue only.", "Question ignored"),
            2: CriterionAnchor(2, "Names one competitor out of five named in the filing.", "Major omissions"),
            3: CriterionAnchor(3, "Covers the main question but misses follow-on details.", "Mostly complete"),
            4: CriterionAnchor(4, "Addresses all parts; minor additional context missing.", "Nearly complete"),
            5: CriterionAnchor(5, "Fully addresses all parts of the question with appropriate depth.", "Fully complete"),
        }
    ),
    Criterion(
        name="coherence",
        description="Is the answer logically organized and easy to follow?",
        scale=3,
        weight=0.5,
        anchors={
            1: CriterionAnchor(1, "Answer jumps between unrelated topics with no structure.", "Incoherent"),
            2: CriterionAnchor(2, "Logical but some sentences don't connect well.", "Partially coherent"),
            3: CriterionAnchor(3, "Clear, well-structured, easy to read.", "Fully coherent"),
        }
    ),
    Criterion(
        name="hallucination_avoidance",
        description="Does the answer avoid inventing facts not present in the retrieved evidence?",
        scale=3,
        weight=1.5,
        anchors={
            1: CriterionAnchor(1, "Invents a company merger not mentioned in any filing.", "Confirmed hallucination"),
            2: CriterionAnchor(2, "Makes one claim that goes slightly beyond the evidence.", "Minor unsupported claim"),
            3: CriterionAnchor(3, "Every statement is grounded in retrieved evidence.", "No hallucinations"),
        }
    ),
]


def get_criterion(name: str) -> Optional[Criterion]:
    return next((c for c in CRITERIA if c.name == name), None)


def criterion_names() -> list[str]:
    return [c.name for c in CRITERIA]


def weighted_avg(scores: dict[str, int]) -> float:
    """Compute weighted average score normalized to 0-1 range."""
    total_weight = 0.0
    total_score = 0.0
    for criterion in CRITERIA:
        if criterion.name in scores:
            score = scores[criterion.name]
            normalized = (score - 1) / (criterion.scale - 1)  # normalize to [0,1]
            total_score += normalized * criterion.weight
            total_weight += criterion.weight
    return total_score / total_weight if total_weight > 0 else 0.0
