# Multi-Agent Research Assistant

**Acceptance criterion:** A research question produces a fully cited report, survives process restart, and has a complete trace.

## Overview

A production-grade multi-agent research pipeline built with LangGraph, the Anthropic SDK, Tavily search, Redis for persistence, and FastAPI for the HTTP interface.

Four specialized agents collaborate:

| Agent | Model | Responsibility |
|-------|-------|----------------|
| Planner | claude-haiku-4-5 | Decompose question into sub-questions |
| Researcher | claude-haiku-4-5 | Search via Tavily, synthesize findings |
| Writer | claude-sonnet-4-6 | Produce final report with inline citations |
| Supervisor | claude-haiku-4-5 | Route agents, validate outputs, enforce budgets |

## Architecture

```
START -> planner -> supervisor_validate_plan -> researcher ->
  supervisor_check_complete --[not done]--> researcher
                            --[done]------> writer ->
  supervisor_validate_report -> END
```

State is persisted to Redis after every node, enabling process restart and resume.

## Key Design Decisions

1. **Typed Pydantic schemas** for all inter-agent handoffs (not vague dicts or personality prompts)
2. **Budget enforcement**: max_subquestions, max_searches_per_subquestion, max_total_tokens, wall_clock_timeout
3. **Redis persistence** after every node for restart/resume support
4. **Idempotent execution**: repeated node execution is safe (Redis SET is idempotent)
5. **Graceful degradation**: budget exhaustion yields partial results (status=complete, not failed)
6. **Comprehensive error handling**: rate limits, timeouts, paywalls, malformed results

## Setup

### Prerequisites
- Python 3.11+
- Redis 7+
- Anthropic API key
- Tavily API key

### Installation

```bash
cd AGENTS/
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Run with Docker Compose

```bash
docker-compose up
```

### Run locally

```bash
# Start Redis
redis-server

# Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or run example script
python scripts/run_example.py
```

## API

### Start a research run

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the long-term effects of sleep deprivation?",
    "max_subquestions": 5,
    "max_searches_per_subquestion": 3,
    "max_total_tokens": 100000,
    "wall_clock_timeout_seconds": 300
  }'
```

Response:
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Research started. Poll GET /research/550e8400... for results."
}
```

### Poll for results

```bash
curl http://localhost:8000/research/{run_id}
```

Response:
```json
{
  "run_id": "550e8400-...",
  "question": "What are the long-term effects of sleep deprivation?",
  "status": "complete",
  "report": "# Research Report\n\n## Executive Summary\n...",
  "error": null,
  "created_at": "2024-01-01T00:00:00+00:00",
  "updated_at": "2024-01-01T00:05:00+00:00"
}
```

### Get execution trace

```bash
curl http://localhost:8000/research/{run_id}/trace
```

Response:
```json
{
  "run_id": "550e8400-...",
  "trace": [
    {
      "agent": "planner",
      "input": {"question": "...", "max_subquestions": 5},
      "output": {"subquestions": [...], "reasoning": "..."},
      "tool_calls": [],
      "token_usage": {"input_tokens": 450, "output_tokens": 280},
      "latency_ms": 1234.5,
      "status": "success",
      "timestamp": "2024-01-01T00:00:01+00:00"
    },
    ...
  ]
}
```

## Running Tests

```bash
# All tests (no live Redis or API keys required — all mocked)
pytest tests/ -v --no-header

# Specific test file
pytest tests/test_researcher.py -v
pytest tests/test_restart.py -v
```

Test coverage:
- successful run
- no search results
- rate limit handling (exponential backoff)
- timeout handling
- paywall/blocked result
- malformed search result
- process restart/resume
- repeated node execution (idempotency)
- budget exhaustion

## Schemas

All inter-agent data transfers use typed Pydantic models:

```python
class Finding(BaseModel):
    claim: str
    source_url: str
    source_snippet: str
    retrieval_timestamp: str  # ISO 8601

class SubQuestion(BaseModel):
    id: str
    text: str
    status: Literal["pending", "in_progress", "complete", "failed"]

class TraceEntry(BaseModel):
    agent: str
    input: dict
    output: dict
    tool_calls: list[dict]
    token_usage: dict  # input_tokens, output_tokens
    latency_ms: float
    status: Literal["success", "error"]
    timestamp: str
```

## Budget Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_subquestions | 5 | Maximum sub-questions the planner can create |
| max_searches_per_subquestion | 3 | Maximum Tavily searches per sub-question |
| max_total_tokens | 100,000 | Maximum tokens across all agent calls |
| wall_clock_timeout_seconds | 300 | Maximum total wall time (5 minutes) |

On budget exhaustion, the pipeline proceeds to the writer with partial results (status=complete, not failed).

## Error Handling

| Error | Behavior |
|-------|----------|
| Tavily empty results | Returns Finding with claim="No results found" |
| HTTP 429 rate limit | Exponential backoff: 1s, 2s, 4s (3 attempts) |
| Timeout | Records failed finding, continues |
| Paywall (no snippet) | Skips result, records warning |
| Malformed result | Uses .get() with defaults, never crashes |
| Token budget exceeded | Proceeds to writer with partial results |
| Wall clock timeout | Proceeds to writer with partial results |
