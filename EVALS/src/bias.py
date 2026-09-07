"""
Bias measurements for the LLM judge:
  1. Position bias — verdict flip rate in pairwise comparisons
  2. Length bias — Spearman correlation of score vs. output length
  3. Self-preference bias — Claude scoring Claude vs. GPT-style outputs
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from src.calibration import spearman_correlation


@dataclass
class PositionBiasResult:
    n_pairs: int
    flip_rate: float       # fraction where order changed the verdict
    concern: bool          # True if flip_rate > 0.15


@dataclass
class LengthBiasResult:
    n_examples: int
    spearman_rho: float    # score vs token_count correlation
    concern: bool          # True if |rho| > 0.2


@dataclass
class SelfPreferenceBiasResult:
    n_pairs: int
    avg_score_native: float   # avg score for same-model outputs
    avg_score_other: float    # avg score for other-model outputs
    delta: float              # avg_score_native - avg_score_other
    concern: bool             # True if delta > 0.3 on 5-pt scale


def measure_position_bias(
    judge_fn: Callable[[str, str, str], float],  # (question, answer_a, answer_b) -> preferred winner
    pairs: list[dict],  # each: {question, answer_a, answer_b}
) -> PositionBiasResult:
    """
    For each (A, B) pair, judge in order A,B then B,A.
    A flip = the winner changed when order changed.

    judge_fn returns "A" or "B" indicating which answer was preferred.
    """
    flips = 0
    for pair in pairs:
        verdict_ab = judge_fn(pair["question"], pair["answer_a"], pair["answer_b"])
        verdict_ba = judge_fn(pair["question"], pair["answer_b"], pair["answer_a"])
        # Normalize: if AB said "A" and BA also said "B" (which = the same original A), consistent
        # Flip = verdicts disagree about who wins
        winner_ab = "a" if verdict_ab == "A" else "b"
        # In BA order, "A" means second = original B, "B" means second = original A
        winner_ba_normalized = "b" if verdict_ba == "A" else "a"
        if winner_ab != winner_ba_normalized:
            flips += 1

    flip_rate = flips / len(pairs) if pairs else 0.0
    return PositionBiasResult(
        n_pairs=len(pairs),
        flip_rate=flip_rate,
        concern=flip_rate > 0.15,
    )


def measure_length_bias(
    score_fn: Callable[[str, str], float],  # (question, answer) -> score
    examples: list[dict],  # each: {question, answer}
) -> LengthBiasResult:
    """
    Correlate judge scores with answer length (token count approximation).
    Concern if |spearman_rho| > 0.2 (length is influencing score).
    """
    scores = []
    lengths = []
    for ex in examples:
        score = score_fn(ex["question"], ex["answer"])
        # Approximate token count: words / 0.75 (rough GPT tokenizer heuristic)
        token_count = len(ex["answer"].split()) / 0.75
        scores.append(score)
        lengths.append(token_count)

    rho = spearman_correlation(scores, lengths)
    return LengthBiasResult(
        n_examples=len(examples),
        spearman_rho=rho,
        concern=abs(rho) > 0.2,
    )


def measure_self_preference_bias(
    score_fn: Callable[[str, str], float],  # (question, answer) -> score
    native_examples: list[dict],   # outputs from same model family as the judge
    other_examples: list[dict],    # outputs from different model family
) -> SelfPreferenceBiasResult:
    """
    Compare how the judge scores outputs from its own model family
    versus outputs from a different model family.
    Concern if the delta > 0.3 on the weighted scale.
    """
    native_scores = [score_fn(ex["question"], ex["answer"]) for ex in native_examples]
    other_scores = [score_fn(ex["question"], ex["answer"]) for ex in other_examples]

    avg_native = sum(native_scores) / len(native_scores) if native_scores else 0.0
    avg_other = sum(other_scores) / len(other_scores) if other_scores else 0.0
    delta = avg_native - avg_other

    return SelfPreferenceBiasResult(
        n_pairs=min(len(native_examples), len(other_examples)),
        avg_score_native=avg_native,
        avg_score_other=avg_other,
        delta=delta,
        concern=abs(delta) > 0.3,
    )
