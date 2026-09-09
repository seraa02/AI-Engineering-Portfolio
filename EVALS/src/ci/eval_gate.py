"""
CI Evaluation Gate.

Runs the judge on a test set and checks scores two ways:
  1. Absolute floor per criterion (THRESHOLDS) — never allowed below this, regardless of history.
  2. Regression vs. the last stored run (REGRESSION_TOLERANCE) — no criterion may drop by more
     than its tolerance since the last run, and `hallucination_avoidance` (the safety-critical
     criterion) may not drop AT ALL, even by a rounding error.

Exits with code 1 if either check fails (causes CI to fail).

Usage:
    python -m src.ci.eval_gate                  # mock judge, no API cost — default, safe for every PR
    python -m src.ci.eval_gate --degraded        # inject a deliberately bad prompt to prove the gate catches it
    python -m src.ci.eval_gate --live            # use the real Claude judge (costs real API calls)
    python -m src.ci.eval_gate --store           # write this run to Postgres (or SQLite fallback) and
                                                  # regression-check against the last stored run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Minimum acceptable average score per criterion, regardless of run history.
# Scores are normalized to [0, 1] by weighted_avg()
THRESHOLDS = {
    "factual_accuracy": 0.55,        # 5-pt scale, min avg ~3.8/5
    "citation_quality": 0.55,
    "completeness": 0.45,
    "coherence": 0.50,               # 3-pt scale
    "hallucination_avoidance": 0.50, # 3-pt scale
    "overall_weighted_avg": 0.50,
}

# Maximum allowed drop vs. the previous stored run, on the same [0,1] normalized scale.
# hallucination_avoidance is the safety-critical criterion: zero tolerance, per the PDF spec
# ("the safety criterion may not drop at all"). Everything else gets a small cushion for
# normal run-to-run judge noise.
REGRESSION_TOLERANCE = {
    "factual_accuracy": 0.05,
    "citation_quality": 0.05,
    "completeness": 0.07,
    "coherence": 0.07,
    "hallucination_avoidance": 0.0,   # zero-tolerance safety criterion
    "overall_weighted_avg": 0.05,
}


def run_gate(
    judge_fn,          # (question, answer) -> JudgmentResult
    test_examples: list[dict],
    thresholds: dict = THRESHOLDS,
    verbose: bool = True,
) -> tuple[bool, dict]:
    """
    Run judge on test_examples. Returns (passed, report_dict).
    passed=True means all absolute thresholds met (regression check is separate — see
    check_regression() and main()'s --store path).
    """
    from src.rubric import CRITERIA, weighted_avg

    # Collect scores per criterion
    all_scores: dict[str, list[int]] = {c.name: [] for c in CRITERIA}
    overall_scores: list[float] = []
    per_example: list[dict] = []

    for ex in test_examples:
        result = judge_fn(ex["input"], ex["model_output"])
        for criterion, score in result.scores.items():
            if criterion in all_scores:
                all_scores[criterion].append(score)
        overall_scores.append(result.weighted_score)
        per_example.append({
            "example_id": ex.get("id", "?"),
            "category": ex.get("category", "?"),
            "scores": dict(result.scores),
            "reasoning": dict(getattr(result, "reasoning", {}) or {}),
        })

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

    # Check absolute thresholds
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
        "per_example": per_example,
    }

    if verbose:
        print(f"\n{'✓' if passed else '✗'} CI Eval Gate (absolute thresholds) — {'PASSED' if passed else 'FAILED'}")
        print(f"  n_examples={len(test_examples)}")
        for metric, score in report["avg_scores"].items():
            threshold = thresholds.get(metric, 0.0)
            status = "✓" if score >= threshold else "✗"
            print(f"  {status} {metric}: {score:.3f} (threshold={threshold})")
        if failures:
            print(f"\n  FAILED metrics: {list(failures.keys())}")

    return passed, report


def check_regression(
    current_scores: dict[str, float],
    previous_scores: dict[str, float],
    tolerance: dict = REGRESSION_TOLERANCE,
) -> dict:
    """
    Compare this run's avg_scores against the previous stored run's.
    Returns {metric: {"previous":, "current":, "drop":, "tolerance":}} for every metric that
    dropped by more than its allowed tolerance. hallucination_avoidance (tolerance 0.0) fails
    on ANY drop, including sub-threshold noise.
    """
    regressions = {}
    for metric, tol in tolerance.items():
        prev = previous_scores.get(metric)
        curr = current_scores.get(metric)
        if prev is None or curr is None:
            continue
        drop = prev - curr
        if drop > tol:
            regressions[metric] = {
                "previous": round(prev, 4),
                "current": round(curr, 4),
                "drop": round(drop, 4),
                "tolerance": tol,
            }
    return regressions


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _mock_judge_fn(degraded: bool):
    def judge_fn(question: str, answer: str):
        from src.rubric import weighted_avg

        class Result:
            pass

        r = Result()
        r.reasoning = {}
        if degraded or "WRONG_CHUNK" in answer or "don't know" in answer.lower():
            r.scores = {"factual_accuracy": 2, "citation_quality": 1, "completeness": 1, "coherence": 2, "hallucination_avoidance": 2}
        elif "[chunk_" in answer:
            r.scores = {"factual_accuracy": 4, "citation_quality": 4, "completeness": 4, "coherence": 3, "hallucination_avoidance": 3}
        else:
            r.scores = {"factual_accuracy": 3, "citation_quality": 2, "completeness": 3, "coherence": 2, "hallucination_avoidance": 2}
        r.weighted_score = weighted_avg(r.scores)
        return r
    return judge_fn


def _live_judge_fn(client, model: str, degraded_prompt: bool):
    """Wraps the real src.judge.judge() as a (question, answer) -> Result callable."""
    from src.judge import judge as real_judge

    def judge_fn(question: str, answer: str):
        if degraded_prompt:
            # --degraded simulates a *prompt regression*, not just bad test data: even a
            # reasonable-sounding answer gets judged as if the upstream system had no
            # citation discipline left, by stripping any citation markers before scoring.
            answer = answer.split(" [chunk_")[0].split(" [WRONG_CHUNK")[0]
        return real_judge(client, question, answer, model=model)
    return judge_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degraded", action="store_true",
                        help="Use a degraded prompt to verify the gate fails")
    parser.add_argument("--live", action="store_true",
                        help="Use the real Claude judge (src.judge.judge) instead of the mock. "
                             "Requires ANTHROPIC_API_KEY.")
    parser.add_argument("--store", action="store_true",
                        help="Store this run in Postgres (POSTGRES_DSN) and regression-check "
                             "against the last stored run. Falls back to a local SQLite file "
                             "(.eval_history.db) if Postgres is unreachable.")
    parser.add_argument("--n", type=int, default=20, help="Number of examples to evaluate (default 20)")
    parser.add_argument("--output", type=str, default=None, help="Write JSON report to file")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    except ImportError:
        pass

    from src.dataset import load_examples

    examples = load_examples()[:args.n]

    if args.degraded:
        for ex in examples:
            ex["model_output"] = "I don't know. [WRONG_CHUNK_999]"
        print("[CI] Running with DEGRADED outputs — gate should fail.")

    if args.live:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("[CI] ERROR: --live requires ANTHROPIC_API_KEY to be set.", file=sys.stderr)
            sys.exit(2)
        client = anthropic.Anthropic(api_key=api_key)
        model = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5")
        judge_fn = _live_judge_fn(client, model, degraded_prompt=args.degraded)
        print(f"[CI] Using LIVE judge (model={model}) — this costs real API calls.")
    else:
        judge_fn = _mock_judge_fn(args.degraded)
        print("[CI] Using deterministic MOCK judge (no API cost). Pass --live for the real judge.")

    passed, report = run_gate(judge_fn, examples, verbose=True)

    if args.store:
        from src.database import get_latest_run, get_postgres_connection, get_sqlite_connection, insert_example_eval, insert_run
        import time

        try:
            conn = get_postgres_connection()
            backend = "postgres"
        except Exception as e:
            print(f"[CI] Postgres unavailable ({e}); falling back to local SQLite (.eval_history.db)")
            conn = get_sqlite_connection(str(Path(__file__).resolve().parent.parent.parent / ".eval_history.db"))
            backend = "sqlite"

        previous = get_latest_run(conn)
        run_id = f"ci-{'live' if args.live else 'mock'}-{'degraded' if args.degraded else 'normal'}-{int(time.time())}"
        insert_run(
            conn, run_id,
            n_examples=report["n_examples"],
            overall_score=report["avg_scores"]["overall_weighted_avg"],
            scores_by_criterion=report["avg_scores"],
            model_version=os.environ.get("JUDGE_MODEL", "claude-haiku-4-5") if args.live else "mock",
        )
        for ex in report["per_example"]:
            insert_example_eval(conn, run_id, ex["example_id"], ex["category"], ex["scores"], ex["reasoning"])
        print(f"[CI] Stored run '{run_id}' to {backend}.")

        if previous is not None:
            regressions = check_regression(report["avg_scores"], previous["scores_by_criterion"])
            report["regression_check"] = {
                "previous_run_id": previous["run_id"],
                "regressions": regressions,
            }
            if regressions:
                passed = False
                print(f"\n✗ REGRESSION vs previous run '{previous['run_id']}':")
                for metric, info in regressions.items():
                    flag = " [SAFETY CRITERION — ZERO TOLERANCE]" if metric == "hallucination_avoidance" else ""
                    print(f"  ✗ {metric}: {info['previous']} -> {info['current']} "
                          f"(dropped {info['drop']}, tolerance {info['tolerance']}){flag}")
            else:
                print(f"\n✓ No regression vs previous run '{previous['run_id']}'.")
        else:
            print("[CI] No previous run stored yet — this run becomes the baseline.")

        conn.close()

    report["passed"] = passed

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
