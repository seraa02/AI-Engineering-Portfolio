"""
FastAPI application for the Multi-Agent Research Assistant.

Endpoints:
- POST /research — accepts question, returns run_id
- GET /research/{run_id} — returns status and result
- GET /research/{run_id}/trace — returns full trace

Research runs execute asynchronously in background threads.
State is persisted to Redis for restart/resume support.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import anthropic
import redis as redis_lib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Tavily imported at module level for patchability in tests
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # type: ignore

from src.config import get_settings
from src.schemas import (
    ResearchRequest,
    StartResponse,
    ResearchResponse,
    TraceResponse,
    BudgetConfig,
    TraceEntry,
)
from src.state import (
    make_initial_state,
    save_state,
    load_state,
    state_exists,
    get_trace,
)
from src.graph import ResearchGraph

logger = logging.getLogger(__name__)

# Thread pool for running synchronous graph execution
_executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients on startup, clean up on shutdown."""
    settings = get_settings()

    # Initialize Redis
    app.state.redis = redis_lib.Redis.from_url(settings.redis_url, decode_responses=True)
    app.state.redis_ttl = settings.redis_ttl_seconds

    # Initialize Anthropic
    app.state.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Initialize Tavily
    if TavilyClient is not None:
        app.state.tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    else:
        logger.warning("tavily-python not installed — search functionality disabled")
        app.state.tavily_client = None

    app.state.settings = settings

    logger.info("Research Assistant API started")
    yield

    # Cleanup
    _executor.shutdown(wait=False)
    logger.info("Research Assistant API stopped")


app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="LangGraph-powered research pipeline with Redis persistence",
    version="1.0.0",
    lifespan=lifespan,
)


def _run_graph_sync(
    run_id: str,
    redis_client: redis_lib.Redis,
    anthropic_client: anthropic.Anthropic,
    tavily_client,
    settings,
    initial_state,
):
    """
    Execute the research graph synchronously.
    Called from a background thread.
    """
    try:
        graph = ResearchGraph(
            anthropic_client=anthropic_client,
            tavily_client=tavily_client,
            redis_client=redis_client,
            planner_model=settings.planner_model,
            supervisor_model=settings.supervisor_model,
            writer_model=settings.writer_model,
            researcher_model=settings.researcher_model,
            redis_ttl=settings.redis_ttl_seconds,
        )
        final_state = graph.run(initial_state)
        logger.info("Run %s completed with status=%s", run_id, final_state["status"])
    except Exception as exc:
        logger.error("Run %s failed with exception: %s", run_id, exc)
        # Try to mark state as failed in Redis
        try:
            state = load_state(redis_client, run_id)
            if state:
                state["status"] = "failed"
                state["error"] = str(exc)
                save_state(redis_client, state, settings.redis_ttl_seconds)
        except Exception:
            pass


@app.post("/research", response_model=StartResponse, status_code=202)
async def start_research(request: ResearchRequest):
    """
    Start a new research run.

    Returns run_id immediately. Use GET /research/{run_id} to poll for results.
    """
    settings = app.state.settings
    redis_client = app.state.redis
    anthropic_client = app.state.anthropic_client
    tavily_client = app.state.tavily_client

    # Generate unique run ID
    run_id = str(uuid.uuid4())

    # Build budget config from request
    budget_config = BudgetConfig(
        max_subquestions=request.max_subquestions,
        max_searches_per_subquestion=request.max_searches_per_subquestion,
        max_total_tokens=request.max_total_tokens,
        wall_clock_timeout_seconds=request.wall_clock_timeout_seconds,
    )

    # Create initial state
    initial_state = make_initial_state(
        run_id=run_id,
        question=request.question,
        budget_config=budget_config,
    )

    # Persist initial state to Redis before launching background task
    save_state(redis_client, initial_state, settings.redis_ttl_seconds)

    # Start background execution in thread pool
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_graph_sync,
        run_id,
        redis_client,
        anthropic_client,
        tavily_client,
        settings,
        initial_state,
    )

    return StartResponse(
        run_id=run_id,
        status="pending",
        message=f"Research started. Poll GET /research/{run_id} for results.",
    )


@app.get("/research/{run_id}", response_model=ResearchResponse)
async def get_research(run_id: str):
    """
    Get the status and result of a research run.
    """
    redis_client = app.state.redis

    state = load_state(redis_client, run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return ResearchResponse(
        run_id=state["run_id"],
        question=state["question"],
        status=state["status"],
        report=state.get("report"),
        error=state.get("error"),
        created_at=state["created_at"],
        updated_at=state["updated_at"],
    )


@app.get("/research/{run_id}/trace", response_model=TraceResponse)
async def get_trace_endpoint(run_id: str):
    """
    Get the full execution trace for a research run.

    Returns: agent, input, output, tool_calls, token_usage, latency, status for each step.
    """
    redis_client = app.state.redis

    state = load_state(redis_client, run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    trace_entries = get_trace(state)

    return TraceResponse(
        run_id=run_id,
        trace=trace_entries,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        app.state.redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected",
    }
