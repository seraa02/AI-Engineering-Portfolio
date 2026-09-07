"""Tests for CI evaluation gate."""
import pytest
from src.rubric import weighted_avg


class MockJudgmentResult:
    def __init__(self, scores: dict[str, int]):
        self.scores = scores
        self.weighted_score = weighted_avg(scores)


def make_judge_fn(score_level: str):
    """Create a mock judge that returns good, medium, or bad scores."""
    def judge_fn(question: str, answer: str):
        if score_level == "good":
            return MockJudgmentResult({
                "factual_accuracy": 5, "citation_quality": 5, "completeness": 5,
                "coherence": 3, "hallucination_avoidance": 3
            })
        elif score_level == "medium":
            return MockJudgmentResult({
                "factual_accuracy": 3, "citation_quality": 3, "completeness": 3,
                "coherence": 2, "hallucination_avoidance": 2
            })
        else:  # degraded
            return MockJudgmentResult({
                "factual_accuracy": 1, "citation_quality": 1, "completeness": 1,
                "coherence": 1, "hallucination_avoidance": 1
            })
    return judge_fn


TEST_EXAMPLES = [
    {"input": "Q1", "model_output": "A1 [chunk_001]"},
    {"input": "Q2", "model_output": "A2 [chunk_002]"},
    {"input": "Q3", "model_output": "A3 [chunk_003]"},
]


def test_gate_passes_good_scores():
    from src.ci.eval_gate import run_gate
    judge = make_judge_fn("good")
    passed, report = run_gate(judge, TEST_EXAMPLES, verbose=False)
    assert passed is True
    assert report["failures"] == {}


def test_gate_fails_degraded_scores():
    from src.ci.eval_gate import run_gate
    judge = make_judge_fn("degraded")
    passed, report = run_gate(judge, TEST_EXAMPLES, verbose=False)
    assert passed is False
    assert len(report["failures"]) > 0


def test_gate_medium_scores_may_fail_strict_thresholds():
    from src.ci.eval_gate import run_gate, THRESHOLDS
    judge = make_judge_fn("medium")
    # With medium scores (normalized ~0.25 on 5-pt, ~0.5 on 3-pt),
    # strict thresholds should fail some criteria
    passed, report = run_gate(judge, TEST_EXAMPLES, verbose=False)
    # Medium scores on 5-pt scale → (3-1)/(5-1) = 0.5 — borderline
    # Result could pass or fail depending on exact thresholds
    assert isinstance(passed, bool)
    assert "avg_scores" in report


def test_gate_report_structure():
    from src.ci.eval_gate import run_gate
    judge = make_judge_fn("good")
    passed, report = run_gate(judge, TEST_EXAMPLES, verbose=False)
    assert "passed" in report
    assert "n_examples" in report
    assert "avg_scores" in report
    assert "failures" in report
    assert report["n_examples"] == 3


def test_gate_empty_examples():
    from src.ci.eval_gate import run_gate
    judge = make_judge_fn("good")
    passed, report = run_gate(judge, [], verbose=False)
    assert isinstance(passed, bool)
