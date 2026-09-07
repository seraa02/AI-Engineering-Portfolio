"""
LangGraph state definition and Redis persistence layer.

State is persisted to Redis after every node execution.
Supports resume from Redis if a node is re-executed (idempotent).
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Literal, Any

import redis as redis_lib

from src.schemas import (
    SubQuestion,
    Finding,
    TraceEntry,
    BudgetConfig,
    BudgetUsage,
    utc_now,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangGraph state – plain TypedDict so LangGraph can manage it
# ---------------------------------------------------------------------------

from typing import TypedDict


class ResearchState(TypedDict):
    run_id: str
    question: str
    subquestions: list[dict]          # serialized SubQuestion dicts
    findings: dict[str, list[dict]]   # subquestion_id -> list of Finding dicts
    report: Optional[str]
    status: str                        # Literal["pending","planning","researching","writing","complete","failed"]
    trace: list[dict]                  # serialized TraceEntry dicts
    budget_config: dict                # serialized BudgetConfig
    budget_usage: dict                 # serialized BudgetUsage
    error: Optional[str]
    reviewer_feedback_used: bool
    created_at: str
    updated_at: str


def make_initial_state(
    run_id: str,
    question: str,
    budget_config: BudgetConfig,
) -> ResearchState:
    now = utc_now()
    return ResearchState(
        run_id=run_id,
        question=question,
        subquestions=[],
        findings={},
        report=None,
        status="pending",
        trace=[],
        budget_config=budget_config.model_dump(),
        budget_usage=BudgetUsage().model_dump(),
        error=None,
        reviewer_feedback_used=False,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Redis persistence helpers
# ---------------------------------------------------------------------------

def _redis_key(run_id: str) -> str:
    return f"research:{run_id}"


def save_state(r: redis_lib.Redis, state: ResearchState, ttl: int = 86400) -> None:
    """Serialize full state to Redis. Called after every node."""
    key = _redis_key(state["run_id"])
    # Update timestamp on every save
    state["updated_at"] = utc_now()
    try:
        r.set(key, json.dumps(state), ex=ttl)
        logger.debug("State saved to Redis for run_id=%s", state["run_id"])
    except Exception as exc:
        logger.error("Failed to save state to Redis: %s", exc)
        raise


def load_state(r: redis_lib.Redis, run_id: str) -> Optional[ResearchState]:
    """Load state from Redis. Returns None if not found."""
    key = _redis_key(run_id)
    try:
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.error("Failed to load state from Redis: %s", exc)
        return None


def state_exists(r: redis_lib.Redis, run_id: str) -> bool:
    """Check if a run exists in Redis."""
    return bool(r.exists(_redis_key(run_id)))


# ---------------------------------------------------------------------------
# Typed helpers: convert between raw dicts and Pydantic models
# ---------------------------------------------------------------------------

def get_budget_config(state: ResearchState) -> BudgetConfig:
    return BudgetConfig(**state["budget_config"])


def get_budget_usage(state: ResearchState) -> BudgetUsage:
    return BudgetUsage(**state["budget_usage"])


def get_subquestions(state: ResearchState) -> list[SubQuestion]:
    return [SubQuestion(**sq) for sq in state["subquestions"]]


def get_findings(state: ResearchState) -> dict[str, list[Finding]]:
    return {
        sq_id: [Finding(**f) for f in findings_list]
        for sq_id, findings_list in state["findings"].items()
    }


def get_trace(state: ResearchState) -> list[TraceEntry]:
    return [TraceEntry(**t) for t in state["trace"]]


def append_trace(state: ResearchState, entry: TraceEntry) -> ResearchState:
    """Return new state with trace entry appended."""
    new_state = dict(state)
    new_state["trace"] = list(state["trace"]) + [entry.model_dump()]
    return ResearchState(**new_state)


def add_token_usage(state: ResearchState, input_tokens: int, output_tokens: int) -> ResearchState:
    """Return new state with token usage incremented."""
    usage = get_budget_usage(state)
    usage.total_tokens_used += input_tokens + output_tokens
    new_state = dict(state)
    new_state["budget_usage"] = usage.model_dump()
    return ResearchState(**new_state)


def increment_search_count(state: ResearchState, subquestion_id: str) -> ResearchState:
    """Increment search count for a specific subquestion."""
    usage = get_budget_usage(state)
    usage.searches_per_subquestion[subquestion_id] = (
        usage.searches_per_subquestion.get(subquestion_id, 0) + 1
    )
    new_state = dict(state)
    new_state["budget_usage"] = usage.model_dump()
    return ResearchState(**new_state)
