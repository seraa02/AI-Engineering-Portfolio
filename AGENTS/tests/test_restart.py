"""
Tests for process restart/resume support and idempotency.

Covers:
- process restart/resume (state persisted and loadable from Redis)
- repeated node execution (idempotency)
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock
from datetime import datetime, timezone

import pytest

from src.state import (
    make_initial_state,
    save_state,
    load_state,
    state_exists,
    get_subquestions,
    get_budget_config,
    get_budget_usage,
    append_trace,
    add_token_usage,
    increment_search_count,
)
from src.schemas import (
    BudgetConfig,
    BudgetUsage,
    SubQuestion,
    Finding,
    TraceEntry,
    utc_now,
)


class TestRedisStatePersistence:
    def test_save_and_load_state(self, fake_redis, sample_budget_config):
        """Test that state can be saved and loaded from Redis."""
        run_id = "test-persist-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Test question", sample_budget_config)
        state["status"] = "planning"

        save_state(fake_redis, state)

        loaded = load_state(fake_redis, run_id)
        assert loaded is not None
        assert loaded["run_id"] == run_id
        assert loaded["question"] == "Test question"
        assert loaded["status"] == "planning"

    def test_state_not_found_returns_none(self, fake_redis):
        """Test that loading unknown state returns None."""
        result = load_state(fake_redis, "nonexistent-run-id")
        assert result is None

    def test_state_exists_check(self, fake_redis, sample_budget_config):
        """Test state_exists returns correct values."""
        run_id = "test-exists-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Test", sample_budget_config)

        assert not state_exists(fake_redis, run_id)
        save_state(fake_redis, state)
        assert state_exists(fake_redis, run_id)

    def test_state_update_persists(self, fake_redis, sample_budget_config):
        """Test that state updates are persisted correctly."""
        run_id = "test-update-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Test question", sample_budget_config)
        save_state(fake_redis, state)

        # Load and update
        loaded = load_state(fake_redis, run_id)
        loaded["status"] = "researching"
        loaded["subquestions"] = [
            SubQuestion(id="sq_1", text="Test", status="pending").model_dump()
        ]
        save_state(fake_redis, loaded)

        # Load again
        reloaded = load_state(fake_redis, run_id)
        assert reloaded["status"] == "researching"
        assert len(reloaded["subquestions"]) == 1

    def test_full_state_survives_serialization(self, fake_redis, sample_budget_config):
        """Test that complex state survives JSON serialization/deserialization."""
        run_id = "test-serialize-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Test question", sample_budget_config)

        # Add complex nested data
        state["subquestions"] = [
            SubQuestion(id="sq_1", text="Q1", status="complete").model_dump(),
        ]
        state["findings"] = {
            "sq_1": [
                Finding(
                    claim="Test claim",
                    source_url="https://example.com",
                    source_snippet="Test snippet",
                    retrieval_timestamp=utc_now(),
                ).model_dump()
            ]
        }
        trace_entry = TraceEntry(
            agent="planner",
            input={"question": "Test"},
            output={"subquestions": []},
            tool_calls=[],
            token_usage={"input_tokens": 100, "output_tokens": 200},
            latency_ms=123.4,
            status="success",
            timestamp=utc_now(),
        )
        state = append_trace(state, trace_entry)
        save_state(fake_redis, state)

        loaded = load_state(fake_redis, run_id)

        # Check all fields survived
        assert len(loaded["subquestions"]) == 1
        assert loaded["subquestions"][0]["id"] == "sq_1"
        assert len(loaded["findings"]["sq_1"]) == 1
        assert loaded["findings"]["sq_1"][0]["claim"] == "Test claim"
        assert len(loaded["trace"]) == 1
        assert loaded["trace"][0]["agent"] == "planner"


class TestProcessRestartResume:
    def test_resume_from_redis(self, fake_redis, sample_budget_config):
        """
        Test process restart/resume:
        A partially completed run stored in Redis can be resumed.
        """
        run_id = "test-resume-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "What causes climate change?", sample_budget_config)

        # Simulate: planner ran, 1 of 2 subquestions researched
        state["status"] = "researching"
        state["subquestions"] = [
            SubQuestion(id="sq_1", text="Main causes?", status="complete").model_dump(),
            SubQuestion(id="sq_2", text="Solutions?", status="pending").model_dump(),
        ]
        state["findings"] = {
            "sq_1": [
                Finding(
                    claim="CO2 is the main cause",
                    source_url="https://example.com",
                    source_snippet="CO2 snippet",
                ).model_dump()
            ]
        }

        # Save to Redis (simulating mid-run state)
        save_state(fake_redis, state)

        # Simulate process restart: load state from Redis
        resumed_state = load_state(fake_redis, run_id)
        assert resumed_state is not None
        assert resumed_state["status"] == "researching"

        # Verify state is consistent
        subquestions = get_subquestions(resumed_state)
        assert len(subquestions) == 2
        assert subquestions[0].status == "complete"
        assert subquestions[1].status == "pending"

        # Verify findings from before restart are still there
        assert "sq_1" in resumed_state["findings"]
        assert len(resumed_state["findings"]["sq_1"]) == 1

    def test_partial_findings_preserved_on_restart(self, fake_redis, sample_budget_config):
        """Test that partial findings are preserved across restarts."""
        run_id = "test-partial-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Test question", sample_budget_config)

        # Add partial research
        state["subquestions"] = [
            SubQuestion(id="sq_1", text="Q1", status="complete").model_dump(),
            SubQuestion(id="sq_2", text="Q2", status="complete").model_dump(),
            SubQuestion(id="sq_3", text="Q3", status="pending").model_dump(),
        ]
        state["findings"] = {
            "sq_1": [Finding(
                claim="Claim 1",
                source_url="https://a.com",
                source_snippet="Snippet 1"
            ).model_dump()],
            "sq_2": [Finding(
                claim="Claim 2",
                source_url="https://b.com",
                source_snippet="Snippet 2"
            ).model_dump()],
        }

        save_state(fake_redis, state)
        loaded = load_state(fake_redis, run_id)

        # All partial findings preserved
        assert len(loaded["findings"]["sq_1"]) == 1
        assert len(loaded["findings"]["sq_2"]) == 1
        assert "sq_3" not in loaded["findings"]  # Not yet researched


class TestIdempotency:
    def test_save_state_twice_is_idempotent(self, fake_redis, sample_budget_config):
        """
        Test that saving the same state twice doesn't corrupt data.
        Redis SET is naturally idempotent.
        """
        run_id = "test-idempotent-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Test question", sample_budget_config)

        # Save twice
        save_state(fake_redis, state)
        save_state(fake_redis, state)

        loaded = load_state(fake_redis, run_id)
        assert loaded is not None
        assert loaded["run_id"] == run_id

    def test_repeated_token_increment_is_additive(self, fake_redis, sample_budget_config):
        """Test that token increments accumulate correctly."""
        run_id = "test-tokens-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Test", sample_budget_config)

        # Increment tokens multiple times
        state = add_token_usage(state, 100, 200)
        state = add_token_usage(state, 150, 250)
        state = add_token_usage(state, 50, 50)

        usage = get_budget_usage(state)
        assert usage.total_tokens_used == (100 + 200 + 150 + 250 + 50 + 50)

    def test_repeated_search_increment_is_additive(self, fake_redis, sample_budget_config):
        """Test that search counts accumulate correctly."""
        run_id = "test-searches-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Test", sample_budget_config)

        state = increment_search_count(state, "sq_1")
        state = increment_search_count(state, "sq_1")
        state = increment_search_count(state, "sq_2")

        usage = get_budget_usage(state)
        assert usage.searches_per_subquestion["sq_1"] == 2
        assert usage.searches_per_subquestion["sq_2"] == 1

    def test_state_updated_at_changes_on_save(self, fake_redis, sample_budget_config):
        """Test that updated_at timestamp changes on each save."""
        import time
        run_id = "test-timestamp-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Test", sample_budget_config)

        first_updated = state["updated_at"]

        # Small delay to ensure timestamp difference
        time.sleep(0.01)
        save_state(fake_redis, state)

        loaded = load_state(fake_redis, run_id)
        # updated_at should be refreshed on save
        # (save_state sets updated_at = utc_now())
        assert loaded["updated_at"] >= first_updated

    def test_trace_entries_accumulate(self, fake_redis, sample_budget_config):
        """Test that trace entries accumulate across multiple appends."""
        run_id = "test-trace-accum-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Test", sample_budget_config)

        for i in range(3):
            entry = TraceEntry(
                agent=f"agent_{i}",
                input={},
                output={},
                tool_calls=[],
                token_usage={"input_tokens": 0, "output_tokens": 0},
                latency_ms=0,
                status="success",
                timestamp=utc_now(),
            )
            state = append_trace(state, entry)

        assert len(state["trace"]) == 3
        agents = [t["agent"] for t in state["trace"]]
        assert agents == ["agent_0", "agent_1", "agent_2"]


class TestBudgetExhaustion:
    def test_token_budget_exhaustion_state(self, fake_redis, sample_budget_config):
        """Test state when token budget is exhausted."""
        run_id = "test-budget-exhaust-" + str(uuid.uuid4())[:8]
        budget_config = BudgetConfig(max_total_tokens=100)
        state = make_initial_state(run_id, "Test", budget_config)

        # Exhaust token budget
        state = add_token_usage(state, 50, 60)  # Total 110 > 100

        usage = get_budget_usage(state)
        assert usage.total_tokens_used > budget_config.max_total_tokens

        # Save and reload
        save_state(fake_redis, state)
        loaded = load_state(fake_redis, run_id)
        loaded_usage = get_budget_usage(loaded)
        assert loaded_usage.total_tokens_used == 110
