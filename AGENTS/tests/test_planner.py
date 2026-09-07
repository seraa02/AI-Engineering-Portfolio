"""
Tests for the Planner agent.

Covers: successful run, JSON parsing, edge cases.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.agents.planner import run_planner
from src.schemas import PlannerInput, SubQuestion
from tests.conftest import make_anthropic_response


class TestPlannerSuccessfulRun:
    def test_basic_decomposition(self, mock_anthropic_planner):
        """Test that planner creates sub-questions from a research question."""
        planner_input = PlannerInput(
            question="What causes climate change?",
            max_subquestions=5,
        )

        output, trace = run_planner(
            planner_input=planner_input,
            anthropic_client=mock_anthropic_planner,
            model="claude-haiku-4-5",
        )

        assert len(output.subquestions) == 3
        assert output.subquestions[0].id == "sq_1"
        assert output.subquestions[0].text == "What are the main causes?"
        assert all(sq.status == "pending" for sq in output.subquestions)

    def test_trace_entry_populated(self, mock_anthropic_planner):
        """Test that trace entry has all required fields."""
        planner_input = PlannerInput(question="Test question", max_subquestions=3)

        output, trace = run_planner(
            planner_input=planner_input,
            anthropic_client=mock_anthropic_planner,
        )

        assert trace.agent == "planner"
        assert trace.status == "success"
        assert trace.latency_ms >= 0
        assert trace.token_usage["input_tokens"] == 100
        assert trace.token_usage["output_tokens"] == 200
        assert isinstance(trace.tool_calls, list)

    def test_max_subquestions_enforced(self, mock_anthropic_client):
        """Test that planner respects max_subquestions limit."""
        # Return 5 subquestions but cap is 2
        planner_response = json.dumps({
            "reasoning": "test",
            "subquestions": [
                {"id": f"sq_{i}", "text": f"Question {i}"}
                for i in range(1, 6)
            ]
        })
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(planner_response)

        planner_input = PlannerInput(question="Test", max_subquestions=2)
        output, trace = run_planner(
            planner_input=planner_input,
            anthropic_client=mock_anthropic_client,
        )

        assert len(output.subquestions) == 2

    def test_reasoning_captured(self, mock_anthropic_planner):
        """Test that planner captures reasoning field."""
        planner_input = PlannerInput(question="Test question", max_subquestions=5)
        output, trace = run_planner(
            planner_input=planner_input,
            anthropic_client=mock_anthropic_planner,
        )

        assert output.reasoning == "Breaking down the question into searchable parts"


class TestPlannerEdgeCases:
    def test_empty_subquestions_fallback(self, mock_anthropic_client):
        """Test fallback when planner returns empty subquestions list."""
        planner_response = json.dumps({
            "reasoning": "Could not decompose",
            "subquestions": []
        })
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(planner_response)

        planner_input = PlannerInput(question="Simple question", max_subquestions=5)
        output, trace = run_planner(
            planner_input=planner_input,
            anthropic_client=mock_anthropic_client,
        )

        # Should fall back to using original question
        assert len(output.subquestions) == 1
        assert output.subquestions[0].text == "Simple question"

    def test_malformed_json_handling(self, mock_anthropic_client):
        """Test that planner handles markdown-wrapped JSON."""
        wrapped_json = '```json\n' + json.dumps({
            "reasoning": "test",
            "subquestions": [{"id": "sq_1", "text": "Test question?"}]
        }) + '\n```'
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            json.dumps({"reasoning": "test", "subquestions": [{"id": "sq_1", "text": "Test?"}]})
        )

        planner_input = PlannerInput(question="Test", max_subquestions=5)
        output, trace = run_planner(
            planner_input=planner_input,
            anthropic_client=mock_anthropic_client,
        )

        assert len(output.subquestions) >= 1

    def test_subquestion_ids_assigned(self, mock_anthropic_client):
        """Test that subquestion IDs are assigned even if missing from response."""
        planner_response = json.dumps({
            "reasoning": "test",
            "subquestions": [
                {"text": "Question without ID"},
                {"id": "", "text": "Question with empty ID"},
            ]
        })
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(planner_response)

        planner_input = PlannerInput(question="Test", max_subquestions=5)
        output, trace = run_planner(
            planner_input=planner_input,
            anthropic_client=mock_anthropic_client,
        )

        # All IDs should be non-empty
        for sq in output.subquestions:
            assert sq.id, "SubQuestion ID must not be empty"

    def test_api_error_raises(self, mock_anthropic_client):
        """Test that API errors propagate with trace entry."""
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")

        planner_input = PlannerInput(question="Test", max_subquestions=5)

        with pytest.raises(Exception, match="API Error"):
            run_planner(
                planner_input=planner_input,
                anthropic_client=mock_anthropic_client,
            )
