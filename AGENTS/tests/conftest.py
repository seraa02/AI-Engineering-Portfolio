"""
Shared test fixtures and mocks.

Uses fakeredis (no real Redis required), mocked Anthropic client,
and mocked Tavily client for all unit tests.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch
import uuid

import pytest
import fakeredis

from src.schemas import (
    Finding,
    SubQuestion,
    BudgetConfig,
    BudgetUsage,
    TraceEntry,
    utc_now,
)
from src.state import make_initial_state, ResearchState


# ---------------------------------------------------------------------------
# Redis fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_redis():
    """In-memory Redis using fakeredis."""
    server = fakeredis.FakeServer()
    r = fakeredis.FakeRedis(server=server, decode_responses=True)
    yield r
    r.flushall()


# ---------------------------------------------------------------------------
# Anthropic mock fixtures
# ---------------------------------------------------------------------------

def make_anthropic_response(text: str, input_tokens: int = 100, output_tokens: int = 200):
    """Create a mock Anthropic API response."""
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = text
    mock_response.content = [mock_content]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens
    return mock_response


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client that returns controlled responses."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_anthropic_planner(mock_anthropic_client):
    """Configure mock to return valid planner JSON."""
    planner_response = json.dumps({
        "reasoning": "Breaking down the question into searchable parts",
        "subquestions": [
            {"id": "sq_1", "text": "What are the main causes?"},
            {"id": "sq_2", "text": "What are the historical examples?"},
            {"id": "sq_3", "text": "What are current solutions?"},
        ]
    })
    mock_anthropic_client.messages.create.return_value = make_anthropic_response(planner_response)
    return mock_anthropic_client


@pytest.fixture
def mock_anthropic_researcher(mock_anthropic_client):
    """Configure mock to return valid researcher JSON."""
    researcher_response = json.dumps({
        "claims": [
            {
                "claim": "Test claim from research",
                "source_url": "https://example.com/article",
                "source_snippet": "This is a test snippet from the article"
            }
        ]
    })
    mock_anthropic_client.messages.create.return_value = make_anthropic_response(researcher_response)
    return mock_anthropic_client


@pytest.fixture
def mock_anthropic_writer(mock_anthropic_client):
    """Configure mock to return a valid report."""
    report = """# Research Report

## Executive Summary
This is a comprehensive report on the research topic. [Source: https://example.com/1]

## Main Findings

Based on our research, we found several key insights. The first finding shows [Source: https://example.com/2]
that significant progress has been made in this area.

## Conclusion
In conclusion, the research demonstrates important results [Source: https://example.com/3].

## References
1. https://example.com/1
2. https://example.com/2
3. https://example.com/3
"""
    mock_anthropic_client.messages.create.return_value = make_anthropic_response(report)
    return mock_anthropic_client


# ---------------------------------------------------------------------------
# Tavily mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tavily_client():
    """Mock Tavily client that returns controlled search results."""
    client = MagicMock()
    client.search.return_value = {
        "results": [
            {
                "url": "https://example.com/article1",
                "title": "Test Article 1",
                "content": "This is the content of test article 1 with important information.",
            },
            {
                "url": "https://example.com/article2",
                "title": "Test Article 2",
                "content": "Another article with relevant content about the topic.",
            },
        ]
    }
    return client


@pytest.fixture
def mock_tavily_empty():
    """Mock Tavily client that returns empty results."""
    client = MagicMock()
    client.search.return_value = {"results": []}
    return client


@pytest.fixture
def mock_tavily_rate_limited():
    """Mock Tavily client that raises rate limit error then succeeds."""
    client = MagicMock()
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:
            raise Exception("429 Too Many Requests — rate limit exceeded")
        return {
            "results": [
                {
                    "url": "https://example.com/delayed",
                    "title": "Delayed Result",
                    "content": "Got through after rate limit retries",
                }
            ]
        }

    client.search.side_effect = side_effect
    return client


@pytest.fixture
def mock_tavily_timeout():
    """Mock Tavily client that raises TimeoutError."""
    client = MagicMock()
    client.search.side_effect = TimeoutError("Request timed out")
    return client


@pytest.fixture
def mock_tavily_paywall():
    """Mock Tavily client that returns results without content (paywall)."""
    client = MagicMock()
    client.search.return_value = {
        "results": [
            {
                "url": "https://paywalled.com/article",
                "title": "",
                "content": "",  # No content = paywall
            }
        ]
    }
    return client


@pytest.fixture
def mock_tavily_malformed():
    """Mock Tavily client that returns malformed results (missing fields)."""
    client = MagicMock()
    client.search.return_value = {
        "results": [
            {
                # Missing 'url', 'title' fields
                "content": "Some content without URL",
            },
            {
                "url": "https://example.com/valid",
                # Missing 'title'
                "content": "Valid content here",
            },
            None,  # Completely None result
        ]
    }
    return client


# ---------------------------------------------------------------------------
# State fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_budget_config():
    return BudgetConfig(
        max_subquestions=3,
        max_searches_per_subquestion=2,
        max_total_tokens=10_000,
        wall_clock_timeout_seconds=60,
    )


@pytest.fixture
def sample_run_id():
    return "test-run-" + str(uuid.uuid4())[:8]


@pytest.fixture
def sample_state(sample_run_id, sample_budget_config):
    return make_initial_state(
        run_id=sample_run_id,
        question="What causes climate change?",
        budget_config=sample_budget_config,
    )


@pytest.fixture
def state_with_subquestions(sample_state):
    """State that has been through planning."""
    state = dict(sample_state)
    state["subquestions"] = [
        SubQuestion(id="sq_1", text="What are the main causes of climate change?", status="pending").model_dump(),
        SubQuestion(id="sq_2", text="What are the effects of climate change?", status="pending").model_dump(),
        SubQuestion(id="sq_3", text="What solutions exist for climate change?", status="pending").model_dump(),
    ]
    state["status"] = "researching"
    return ResearchState(**state)


@pytest.fixture
def state_with_findings(state_with_subquestions):
    """State that has been through planning and research."""
    state = dict(state_with_subquestions)
    
    # Mark all as complete
    state["subquestions"] = [
        SubQuestion(id="sq_1", text="What are the main causes of climate change?", status="complete").model_dump(),
        SubQuestion(id="sq_2", text="What are the effects of climate change?", status="complete").model_dump(),
        SubQuestion(id="sq_3", text="What solutions exist for climate change?", status="complete").model_dump(),
    ]
    
    state["findings"] = {
        "sq_1": [
            Finding(
                claim="Carbon dioxide emissions are the primary cause",
                source_url="https://example.com/co2",
                source_snippet="CO2 from burning fossil fuels...",
            ).model_dump()
        ],
        "sq_2": [
            Finding(
                claim="Global temperatures have risen 1.1°C since pre-industrial times",
                source_url="https://example.com/temp",
                source_snippet="Temperature records show...",
            ).model_dump()
        ],
        "sq_3": [
            Finding(
                claim="Renewable energy adoption is growing rapidly",
                source_url="https://example.com/solar",
                source_snippet="Solar and wind capacity...",
            ).model_dump()
        ],
    }
    return ResearchState(**state)
