"""
Dashboard metrics for the red-team harness.

Provides summary statistics for Streamlit or CI reporting:
- Pass/fail/review counts by category and severity
- Regression detection (new FAILs vs previous run)
- Trend data for charting
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.harness.runner import HarnessResult
from src.attacks.taxonomy import AttackResult, Severity
from src.reporting.findings import DB_PATH, _connect, get_findings


# ---------------------------------------------------------------------------
# Summary metrics from a HarnessResult (in-memory, after a run)
# ---------------------------------------------------------------------------

def summary_by_category(result: HarnessResult) -> dict[str, dict]:
    """
    Returns per-category counts:
        {
            "jailbreak": {"pass": 2, "fail": 1, "needs_review": 0, "total": 3},
            ...
        }
    """
    counts: dict[str, dict] = {}
    for outcome in result.outcomes:
        cat = str(outcome.attack.category)
        if cat not in counts:
            counts[cat] = {"pass": 0, "fail": 0, "needs_review": 0, "total": 0}
        counts[cat][outcome.result.lower()] += 1
        counts[cat]["total"] += 1
    return counts


def summary_by_severity(result: HarnessResult) -> dict[str, dict]:
    """Returns per-severity counts."""
    counts: dict[str, dict] = {}
    for outcome in result.outcomes:
        sev = str(outcome.attack.severity)
        if sev not in counts:
            counts[sev] = {"pass": 0, "fail": 0, "needs_review": 0, "total": 0}
        counts[sev][outcome.result.lower()] += 1
        counts[sev]["total"] += 1
    return counts


def critical_failures(result: HarnessResult) -> list:
    """Return FAIL outcomes with CRITICAL or HIGH severity."""
    high_sev = {str(Severity.CRITICAL), str(Severity.HIGH)}
    return [
        o for o in result.outcomes
        if o.result == AttackResult.FAIL and str(o.attack.severity) in high_sev
    ]


def build_run_summary(result: HarnessResult) -> dict:
    """Full summary dict for a completed run."""
    return {
        "run_id": result.run_id,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "total": len(result.outcomes),
        "pass": result.pass_count(),
        "fail": result.fail_count(),
        "needs_review": result.review_count(),
        "fail_rate": round(result.fail_rate(), 4),
        "critical_high_failures": len(critical_failures(result)),
        "by_category": summary_by_category(result),
        "by_severity": summary_by_severity(result),
    }


# ---------------------------------------------------------------------------
# Historical metrics from DB
# ---------------------------------------------------------------------------

def get_run_history(db_path: Path = DB_PATH, limit: int = 20) -> list[dict]:
    """
    Return per-run aggregated metrics from the findings database.
    Useful for trend charts.
    """
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT run_id,
                       COUNT(*) as total,
                       SUM(CASE WHEN result='AttackResult.FAIL' THEN 1 ELSE 0 END) as fails,
                       SUM(CASE WHEN result='NEEDS_REVIEW' THEN 1 ELSE 0 END) as reviews,
                       MIN(timestamp) as started_at
                FROM findings
                GROUP BY run_id
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def detect_regressions(
    current_result: HarnessResult,
    db_path: Path = DB_PATH,
) -> list[str]:
    """
    Return attack_ids that are newly failing (FAIL in current run but
    not present as a FAIL finding in any previous run).
    """
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT attack_id FROM findings WHERE result = 'AttackResult.FAIL'"
            ).fetchall()
        known_failures = {r["attack_id"] for r in rows}
    except sqlite3.OperationalError:
        known_failures = set()

    current_run_fails = {
        o.attack.id for o in current_result.outcomes
        if o.result == AttackResult.FAIL
    }

    return sorted(current_run_fails - known_failures)


def print_report(result: HarnessResult, db_path: Path = DB_PATH) -> None:
    """Print a human-readable summary to stdout."""
    summary = build_run_summary(result)
    regressions = detect_regressions(result, db_path)

    print(f"\n{'='*60}")
    print(f"RED-TEAM HARNESS REPORT  run_id={summary['run_id']}")
    print(f"{'='*60}")
    print(f"  Started:   {summary['started_at']}")
    print(f"  Finished:  {summary['finished_at']}")
    print(f"  Total:     {summary['total']}")
    print(f"  PASS:      {summary['pass']}")
    print(f"  FAIL:      {summary['fail']}")
    print(f"  REVIEW:    {summary['needs_review']}")
    print(f"  Fail rate: {summary['fail_rate']:.1%}")
    print(f"  Critical/High FAILs: {summary['critical_high_failures']}")

    if regressions:
        print(f"\n  *** REGRESSIONS ({len(regressions)} new failures) ***")
        for attack_id in regressions:
            print(f"    - {attack_id}")

    print(f"\n  By category:")
    for cat, counts in sorted(summary["by_category"].items()):
        fail = counts["fail"]
        total = counts["total"]
        print(f"    {cat:<30} {fail}/{total} fail")

    print(f"\n  By severity:")
    for sev, counts in sorted(summary["by_severity"].items()):
        fail = counts["fail"]
        total = counts["total"]
        print(f"    {sev:<15} {fail}/{total} fail")

    # Show individual FAILs
    fails = [o for o in result.outcomes if o.result == AttackResult.FAIL]
    if fails:
        print(f"\n  FAILED ATTACKS:")
        for o in fails:
            print(f"    [{o.attack.severity}] {o.attack.id} — {o.attack.category}")
            print(f"      Reason: {o.reasoning}")

    print(f"{'='*60}\n")
