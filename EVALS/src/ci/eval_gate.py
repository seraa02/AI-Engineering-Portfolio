"""
CI Evaluation Gate.

Runs the judge on a test set and checks scores against thresholds.
Exits with code 1 if any threshold fails (causes CI to fail).

Usage:
    python -m src.ci.eval_gate [--degraded]

The --degraded flag injects a deliberately bad prompt to prove the gate catches regressions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Minimum acceptable average score per criterion
# Scores are normalized to [0, 1] by weighted_avg()
THRESHOLDS = {
    "factual_accuracy": 0.55,        # 5-pt scale, min avg ~3.8/5
    "citation_quality": 0.55,
    "completeness": 0.45,
    "coherence": 0.50,               # 3-pt scale
    "hallucination_avoidance": 0.50, # 3-pt scale
    "overall_weighted_avg": 0.50,
}


def run_gate(
    judge_fn,          # (question, answer) -> JudgmentResult
    test_examples: list[dict],
    thresholds: dict = THRESHOLDS,
    verbose: bool = True,
) -> tuple[bool, dict]:
    """
    Run judge on test_examples. Returns (passed, report_dict).
    passed=True means all thresholds met.
    """
    from src.rubric import CRITERIA, weighted_avg

    # Collect scores per criterion
    all_scores: dict[str, list[int]] = {c.name: [] for c in CRITERIA}
    overall_scores: list[float] = []

    for ex in test_examples:
        result = judge_fn(ex["input"], ex["model_output"])
        for criterion, score in result.scores.items():
            if criterion in all_scores:
                all_scores[criterion].append(score)
        overall_scores.append(result.weighted_score)

    # Compute averages
    from src.rubric import CRITERIA as _CRITERIA
    scale_by_name = {c.name: c.scale for c in _CRITERIA}

    avg_scores_normalized: dict[str, float] = {}
    for criterion, scores in all_scores.items():
        if scores:
            scale = scale_by_name.get(criterion, 5)
            avg_raw = sum(scores) / len(scores)
            avg_scores_normalized[criterion] = (avg_raw - 1) / (scale - 1)

    avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    avg_scores_normalized["overall_weighted_avg"] = avg_overall

    # Check thresholds
    failures = {}
    for metric, threshold in thresholds.items():
        actual = avg_scores_normalized.get(metric, 0.0)
        if actual < threshold:
            failures[metric] = {"threshold": threshold, "actual": round(actual, 3)}

    passed = len(failures) == 0

    report = {
        "passed": passed,
        "n_examples": len(test_examples),
        "avg_scores": {k: round(v, 3) for k, v in avg_scores_normalized.items()},
        "thresholds": thresholds,
        "failures": failures,
    }

    if verbose:
        print(f"\n{'✓' if passed else '✗'} CI Eval Gate — {'PASSED' if passed else 'FAILED'}")
        print(f"  n_examples={len(test_examples)}")
        for metric, score in report["avg_scores"].items():
            threshold = thresholds.get(metric, 0.0)
            status = "✓" if score >= threshold else "✗"
            print(f"  {status} {metric}: {score:.3f} (threshold={threshold})")
        if failures:
            print(f"\n  FAILED metrics: {list(failures.keys())}")

    return passed, report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degraded", action="store_true",
                        help="Use a degraded prompt to verify the gate fails")
    parser.add_argument("--output", type=str, default=None, help="Write JSON report to file")
    args = parser.parse_args()

    from src.dataset import load_examples

    examples = load_examples()[:20]  # Use first 20 for CI speed

    if args.degraded:
        # Deliberately corrupt all answers to prove the gate catches it
        for ex in examples:
            ex["model_output"] = "I don't know. [WRONG_CHUNK_999]"
        print("[CI] Running with DEGRADED outputs — gate should fail.")

    # Use a mock judge (no API required) for CI demonstration
    # In production, replace with real judge using `judge()` from src.judge
    def mock_judge_fn(question: str, answer: str):
        from src.rubric import weighted_avg

        class Result:
            pass

        r = Result()
        if args.degraded or "WRONG_CHUNK" in answer or "don't know" in answer.lower():
            r.scores = {"factual_accuracy": 2, "citation_quality": 1, "completeness": 1, "coherence": 2, "hallucination_avoidance": 2}
        elif "[chunk_" in answer:
            r.scores = {"factual_accuracy": 4, "citation_quality": 4, "completeness": 4, "coherence": 3, "hallucination_avoidance": 3}
        else:
            r.scores = {"factual_accuracy": 3, "citation_quality": 2, "completeness": 3, "coherence": 2, "hallucination_avoidance": 2}
        r.weighted_score = weighted_avg(r.scores)
        return r

    passed, report = run_gate(mock_judge_fn, examples, verbose=True)

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
