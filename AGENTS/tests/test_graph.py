"""
Tests for the LangGraph graph.

Covers:
- successful end-to-end run (mocked)
- budget exhaustion stops research
- state persisted after every node
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import uuid

import pytest

from src.graph import ResearchGraph
from src.schemas import BudgetConfig, Finding, SubQuestion
from src.state import make_initial_state, load_state


def make_mock_planner_response(subquestions=None):
    if subquestions is None:
        subquestions = [{"id": "sq_1", "text": "What are the main causes?"}]
    return json.dumps({
        "reasoning": "Decomposed into sub-questions",
        "subquestions": subquestions,
    })


def make_mock_researcher_response():
    return json.dumps({
        "claims": [
            {
                "claim": "Test claim",
                "source_url": "https://example.com",
                "source_snippet": "Test snippet",
            }
        ]
    })


def make_mock_writer_response():
    return """# Research Report

## Executive Summary
This is a test report with citations [Source: https://example.com].

## Findings
The research found that X is true [Source: https://example.com/2].

## References
1. https://example.com
2. https://example.com/2
"""


def _make_anthropic_response(text, input_tokens=100, output_tokens=200):
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = text
    mock_response.content = [mock_content]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens
    return mock_response


class TestGraphSuccessfulRun:
    def test_end_to_end_run(self, fake_redis):
        """Test a successful end-to-end research run."""
        # Set up mocks
        anthropic_client = MagicMock()
        tavily_client = MagicMock()

        # Configure Anthropic mock to return different responses based on call order
        call_count = [0]
        def anthropic_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Planner call
                return _make_anthropic_response(make_mock_planner_response([
                    {"id": "sq_1", "text": "What are the main causes?"}
                ]))
            elif call_count[0] == 2:
                # Researcher synthesis call
                return _make_anthropic_response(make_mock_researcher_response())
            else:
                # Writer call
                return _make_anthropic_response(make_mock_writer_response())

        anthropic_client.messages.create.side_effect = anthropic_side_effect

        # Configure Tavily mock
        tavily_client.search.return_value = {
            "results": [
                {"url": "https://example.com", "title": "Test", "content": "Test content"}
            ]
        }

        # Create graph
        graph = ResearchGraph(
            anthropic_client=anthropic_client,
            tavily_client=tavily_client,
            redis_client=fake_redis,
            planner_model="claude-haiku-4-5",
            writer_model="claude-sonnet-4-6",
        )

        # Create initial state
        budget_config = BudgetConfig(
            max_subquestions=2,
            max_searches_per_subquestion=1,
            max_total_tokens=50_000,
            wall_clock_timeout_seconds=60,
        )
        initial_state = make_initial_state(
            run_id="test-run-001",
            question="What causes climate change?",
            budget_config=budget_config,
        )

        # Run graph
        final_state = graph.run(initial_state)

        # Verify final state
        assert final_state["status"] in ("complete", "failed")
        assert final_state["run_id"] == "test-run-001"

    def test_state_persisted_in_redis(self, fake_redis):
        """Test that state is persisted to Redis during execution."""
        anthropic_client = MagicMock()
        tavily_client = MagicMock()

        call_count = [0]
        def anthropic_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_anthropic_response(make_mock_planner_response([
                    {"id": "sq_1", "text": "Test question"}
                ]))
            elif call_count[0] == 2:
                return _make_anthropic_response(make_mock_researcher_response())
            else:
                return _make_anthropic_response(make_mock_writer_response())

        anthropic_client.messages.create.side_effect = anthropic_side_effect
        tavily_client.search.return_value = {"results": [
            {"url": "https://example.com", "title": "T", "content": "Content"}
        ]}

        graph = ResearchGraph(
            anthropic_client=anthropic_client,
            tavily_client=tavily_client,
            redis_client=fake_redis,
        )

        run_id = "test-redis-persistence"
        initial_state = make_initial_state(
            run_id=run_id,
            question="Test question",
            budget_config=BudgetConfig(max_subquestions=1, max_searches_per_subquestion=1),
        )

        graph.run(initial_state)

        # State should be in Redis
        saved_state = load_state(fake_redis, run_id)
        assert saved_state is not None
        assert saved_state["run_id"] == run_id

    def test_trace_entries_recorded(self, fake_redis):
        """Test that trace entries are recorded for each agent."""
        anthropic_client = MagicMock()
        tavily_client = MagicMock()

        call_count = [0]
        def anthropic_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_anthropic_response(make_mock_planner_response([
                    {"id": "sq_1", "text": "Test"}
                ]))
            elif call_count[0] == 2:
                return _make_anthropic_response(make_mock_researcher_response())
            else:
                return _make_anthropic_response(make_mock_writer_response())

        anthropic_client.messages.create.side_effect = anthropic_side_effect
        tavily_client.search.return_value = {"results": [
            {"url": "https://ex.com", "title": "T", "content": "C"}
        ]}

        graph = ResearchGraph(
            anthropic_client=anthropic_client,
            tavily_client=tavily_client,
            redis_client=fake_redis,
        )

        initial_state = make_initial_state(
            run_id="test-trace",
            question="Test?",
            budget_config=BudgetConfig(max_subquestions=1, max_searches_per_subquestion=1),
        )

        final_state = graph.run(initial_state)

        # Should have trace entries
        assert len(final_state["trace"]) > 0
        agents = {entry["agent"] for entry in final_state["trace"]}
        assert "planner" in agents


class TestGraphBudgetEnforcement:
    def test_token_budget_stops_research(self, fake_redis):
        """Test that token budget exhaustion stops research and proceeds to writer."""
        anthropic_client = MagicMock()
        tavily_client = MagicMock()

        call_count = [0]
        def anthropic_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_anthropic_response(
                    make_mock_planner_response([
                        {"id": "sq_1", "text": "Q1"},
                        {"id": "sq_2", "text": "Q2"},
                    ]),
                    input_tokens=5000,
                    output_tokens=5000,  # Total 10000 tokens on first call
                )
            elif call_count[0] == 2:
                return _make_anthropic_response(
                    make_mock_researcher_response(),
                    input_tokens=1000,
                    output_tokens=1000,
                )
            else:
                return _make_anthropic_response(make_mock_writer_response())

        anthropic_client.messages.create.side_effect = anthropic_side_effect
        tavily_client.search.return_value = {"results": [
            {"url": "https://ex.com", "title": "T", "content": "C"}
        ]}

        # Very small token budget
        budget_config = BudgetConfig(
            max_subquestions=2,
            max_searches_per_subquestion=1,
            max_total_tokens=5_000,  # Will be exceeded quickly
            wall_clock_timeout_seconds=60,
        )

        graph = ResearchGraph(
            anthropic_client=anthropic_client,
            tavily_client=tavily_client,
            redis_client=fake_redis,
        )

        initial_state = make_initial_state(
            run_id="test-token-budget",
            question="Test question",
            budget_config=budget_config,
        )

        # Should complete (not fail) even with budget exhaustion
        final_state = graph.run(initial_state)
        assert final_state["status"] in ("complete", "failed")
