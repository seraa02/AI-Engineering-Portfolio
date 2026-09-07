"""
All Pydantic schemas for agent handoffs, state, and API contracts.
Every inter-agent data transfer uses typed schemas, not vague dicts.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class Finding(BaseModel):
    """A single research finding from the Researcher agent."""
    claim: str
    source_url: str
    source_snippet: str
    retrieval_timestamp: str = Field(default_factory=utc_now)  # ISO 8601


class SubQuestion(BaseModel):
    """A sub-question produced by the Planner agent."""
    id: str
    text: str
    status: Literal["pending", "in_progress", "complete", "failed"] = "pending"


# ---------------------------------------------------------------------------
# Budget models
# ---------------------------------------------------------------------------

class BudgetConfig(BaseModel):
    max_subquestions: int = 5
    max_searches_per_subquestion: int = 3
    max_total_tokens: int = 100_000
    wall_clock_timeout_seconds: int = 300


class BudgetUsage(BaseModel):
    total_tokens_used: int = 0
    subquestion_count: int = 0
    searches_per_subquestion: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trace model
# ---------------------------------------------------------------------------

class TraceEntry(BaseModel):
    agent: str
    input: dict
    output: dict
    tool_calls: list[dict]
    token_usage: dict  # input_tokens, output_tokens
    latency_ms: float
    status: Literal["success", "error"]
    timestamp: str = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# Agent handoff schemas (typed, not vague dicts)
# ---------------------------------------------------------------------------

class PlannerInput(BaseModel):
    question: str
    max_subquestions: int = 5


class PlannerOutput(BaseModel):
    subquestions: list[SubQuestion]
    reasoning: str


class ResearcherInput(BaseModel):
    subquestion: SubQuestion
    max_searches: int = 3


class ResearcherOutput(BaseModel):
    subquestion_id: str
    findings: list[Finding]
    searches_performed: int


class WriterInput(BaseModel):
    question: str
    subquestions: list[SubQuestion]
    findings: dict[str, list[Finding]]


class WriterOutput(BaseModel):
    report: str
    citation_count: int


class SupervisorValidatePlanInput(BaseModel):
    subquestions: list[SubQuestion]
    budget_config: BudgetConfig
    budget_usage: BudgetUsage


class SupervisorValidatePlanOutput(BaseModel):
    approved: bool
    trimmed_subquestions: list[SubQuestion]
    reason: str


class SupervisorCheckCompleteInput(BaseModel):
    subquestions: list[SubQuestion]
    findings: dict[str, list[Finding]]
    budget_config: BudgetConfig
    budget_usage: BudgetUsage
    created_at: str


class SupervisorCheckCompleteOutput(BaseModel):
    all_complete: bool
    next_subquestion_id: Optional[str]
    reason: str
    budget_exhausted: bool


class SupervisorValidateReportInput(BaseModel):
    report: str
    findings: dict[str, list[Finding]]


class SupervisorValidateReportOutput(BaseModel):
    approved: bool
    reason: str
    citation_count: int


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    question: str
    max_subquestions: int = 5
    max_searches_per_subquestion: int = 3
    max_total_tokens: int = 100_000
    wall_clock_timeout_seconds: int = 300


class StartResponse(BaseModel):
    run_id: str
    status: str
    message: str


class ResearchResponse(BaseModel):
    run_id: str
    question: str
    status: str
    report: Optional[str]
    error: Optional[str]
    created_at: str
    updated_at: str


class TraceResponse(BaseModel):
    run_id: str
    trace: list[TraceEntry]
