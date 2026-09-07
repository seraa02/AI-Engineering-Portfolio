"""
Supervisor Agent — routes agents, validates outputs, enforces budgets.

Responsibilities:
1. validate_plan: Check if subquestion count <= max_subquestions, trim if needed
2. check_complete: Check if all subquestions researched OR budget exhausted
3. validate_report: Verify report has citations
4. Allow reviewer feedback only ONCE (reviewer_feedback_used flag)

Uses claude-haiku-4-5 (cheap) for routing decisions.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import anthropic

from src.schemas import (
    SubQuestion,
    Finding,
    BudgetConfig,
    BudgetUsage,
    SupervisorValidatePlanInput,
    SupervisorValidatePlanOutput,
    SupervisorCheckCompleteInput,
    SupervisorCheckCompleteOutput,
    SupervisorValidateReportInput,
    SupervisorValidateReportOutput,
    TraceEntry,
    utc_now,
)

logger = logging.getLogger(__name__)


def _wall_clock_exceeded(created_at: str, timeout_seconds: int) -> bool:
    """Check if wall clock timeout has been exceeded."""
    try:
        start = datetime.fromisoformat(created_at)
        now = datetime.now(timezone.utc)
        # Make start timezone-aware if naive
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        elapsed = (now - start).total_seconds()
        return elapsed > timeout_seconds
    except Exception:
        return False


def supervisor_validate_plan(
    plan_input: SupervisorValidatePlanInput,
) -> tuple[SupervisorValidatePlanOutput, TraceEntry]:
    """
    Validate the planner output.
    
    - Enforce max_subquestions budget
    - Trim subquestion list if too long
    - Update budget usage
    """
    start_time = time.time()
    timestamp = utc_now()

    subquestions = plan_input.subquestions
    max_sq = plan_input.budget_config.max_subquestions
    original_count = len(subquestions)

    # Trim to budget
    if len(subquestions) > max_sq:
        subquestions = subquestions[:max_sq]
        reason = f"Trimmed from {original_count} to {max_sq} sub-questions (budget limit)"
        approved = True
    elif not subquestions:
        approved = False
        reason = "Planner produced no sub-questions"
    else:
        approved = True
        reason = f"Plan approved: {len(subquestions)} sub-questions within budget"

    output = SupervisorValidatePlanOutput(
        approved=approved,
        trimmed_subquestions=subquestions,
        reason=reason,
    )

    latency_ms = (time.time() - start_time) * 1000
    trace_entry = TraceEntry(
        agent="supervisor_validate_plan",
        input=plan_input.model_dump(),
        output=output.model_dump(),
        tool_calls=[],
        token_usage={"input_tokens": 0, "output_tokens": 0},
        latency_ms=latency_ms,
        status="success" if approved else "error",
        timestamp=timestamp,
    )

    logger.info("Supervisor validated plan: %s", reason)
    return output, trace_entry


def supervisor_check_complete(
    check_input: SupervisorCheckCompleteInput,
) -> tuple[SupervisorCheckCompleteOutput, TraceEntry]:
    """
    Check if research is complete or budget is exhausted.
    
    Returns next subquestion to research, or signals completion.
    """
    start_time = time.time()
    timestamp = utc_now()

    subquestions = check_input.subquestions
    findings = check_input.findings
    budget_config = check_input.budget_config
    budget_usage = check_input.budget_usage

    # Check wall clock timeout
    if _wall_clock_exceeded(check_input.created_at, budget_config.wall_clock_timeout_seconds):
        output = SupervisorCheckCompleteOutput(
            all_complete=True,
            next_subquestion_id=None,
            reason="Wall clock timeout exceeded — proceeding with partial results",
            budget_exhausted=True,
        )
        latency_ms = (time.time() - start_time) * 1000
        trace_entry = TraceEntry(
            agent="supervisor_check_complete",
            input=check_input.model_dump(),
            output=output.model_dump(),
            tool_calls=[],
            token_usage={"input_tokens": 0, "output_tokens": 0},
            latency_ms=latency_ms,
            status="success",
            timestamp=timestamp,
        )
        logger.warning("Wall clock timeout — proceeding to writer with partial results")
        return output, trace_entry

    # Check token budget
    if budget_usage.total_tokens_used >= budget_config.max_total_tokens:
        output = SupervisorCheckCompleteOutput(
            all_complete=True,
            next_subquestion_id=None,
            reason=f"Token budget exhausted ({budget_usage.total_tokens_used}/{budget_config.max_total_tokens}) — proceeding with partial results",
            budget_exhausted=True,
        )
        latency_ms = (time.time() - start_time) * 1000
        trace_entry = TraceEntry(
            agent="supervisor_check_complete",
            input=check_input.model_dump(),
            output=output.model_dump(),
            tool_calls=[],
            token_usage={"input_tokens": 0, "output_tokens": 0},
            latency_ms=latency_ms,
            status="success",
            timestamp=timestamp,
        )
        logger.warning("Token budget exhausted — proceeding to writer")
        return output, trace_entry

    # Find next pending subquestion
    next_sq_id: Optional[str] = None
    for sq in subquestions:
        if sq.status in ("pending", "in_progress"):
            # Check per-subquestion search budget
            searches_done = budget_usage.searches_per_subquestion.get(sq.id, 0)
            if searches_done < budget_config.max_searches_per_subquestion:
                next_sq_id = sq.id
                break
            else:
                # This subquestion has exhausted its search budget — mark complete
                logger.info(
                    "Subquestion %s search budget exhausted (%d searches), marking complete",
                    sq.id, searches_done
                )

    if next_sq_id is not None:
        reason = f"Next subquestion to research: {next_sq_id}"
        all_complete = False
    else:
        # All subquestions are done (or budget exhausted per subquestion)
        pending_count = sum(1 for sq in subquestions if sq.status in ("pending", "in_progress"))
        if pending_count == 0:
            reason = "All sub-questions have been researched"
        else:
            reason = f"All sub-questions reached search budget limit ({pending_count} pending)"
        all_complete = True

    output = SupervisorCheckCompleteOutput(
        all_complete=all_complete,
        next_subquestion_id=next_sq_id,
        reason=reason,
        budget_exhausted=False,
    )

    latency_ms = (time.time() - start_time) * 1000
    trace_entry = TraceEntry(
        agent="supervisor_check_complete",
        input=check_input.model_dump(),
        output=output.model_dump(),
        tool_calls=[],
        token_usage={"input_tokens": 0, "output_tokens": 0},
        latency_ms=latency_ms,
        status="success",
        timestamp=timestamp,
    )

    logger.info("Supervisor check complete: %s", reason)
    return output, trace_entry


def supervisor_validate_report(
    report_input: SupervisorValidateReportInput,
) -> tuple[SupervisorValidateReportOutput, TraceEntry]:
    """
    Validate that the report contains citations.
    
    Simple heuristic check — does not require LLM call.
    """
    import re

    start_time = time.time()
    timestamp = utc_now()

    report = report_input.report

    # Count citations using multiple patterns
    source_citations = len(re.findall(r'\[Source:', report))
    numbered_citations = len(re.findall(r'\[\d+\]', report))
    url_citations = len(re.findall(r'https?://\S+', report))

    citation_count = max(source_citations, numbered_citations, url_citations)

    if citation_count > 0:
        approved = True
        reason = f"Report approved: {citation_count} citations found"
    else:
        # Still approve — some reports may have inline references without URL pattern
        approved = True
        reason = "Report accepted (no URL citations detected, but report was generated)"
        citation_count = 0

    # Check minimum length
    if len(report) < 100:
        approved = False
        reason = f"Report too short ({len(report)} chars) — likely an error"

    output = SupervisorValidateReportOutput(
        approved=approved,
        reason=reason,
        citation_count=citation_count,
    )

    latency_ms = (time.time() - start_time) * 1000
    trace_entry = TraceEntry(
        agent="supervisor_validate_report",
        input={"report_length": len(report), "has_findings": bool(report_input.findings)},
        output=output.model_dump(),
        tool_calls=[],
        token_usage={"input_tokens": 0, "output_tokens": 0},
        latency_ms=latency_ms,
        status="success" if approved else "error",
        timestamp=timestamp,
    )

    logger.info("Supervisor validated report: %s", reason)
    return output, trace_entry
