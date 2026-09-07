"""
Tests for the FastAPI API endpoints.

Covers: POST /research, GET /research/{id}, GET /research/{id}/trace.

Strategy: Patch redis.Redis.from_url so the lifespan gets the same
fakeredis instance as the test. Both the app and the test write/read
from the same in-memory Redis server.
"""

from __future__ import annotations

import concurrent.futures
import json
from unittest.mock import MagicMock, patch, call
import uuid

import pytest
import fakeredis
from fastapi.testclient import TestClient

# Import app BEFORE any patching so module-level code runs once
import src.api.main as api_main
from src.api.main import app
from src.schemas import BudgetConfig, TraceEntry, utc_now
from src.state import make_initial_state, save_state


# ---------------------------------------------------------------------------
# One shared fakeredis server — all connections go to same memory
# ---------------------------------------------------------------------------

_FAKE_SERVER = fakeredis.FakeServer()


def _make_fake_redis_client(*args, **kwargs):
    """Factory: return FakeRedis connected to shared server."""
    return fakeredis.FakeRedis(server=_FAKE_SERVER, decode_responses=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _flush_redis():
    """Flush fakeredis between tests."""
    r = fakeredis.FakeRedis(server=_FAKE_SERVER, decode_responses=True)
    r.flushall()
    yield
    r.flushall()


@pytest.fixture
def test_redis():
    """Direct access to the shared fakeredis for test setup."""
    return fakeredis.FakeRedis(server=_FAKE_SERVER, decode_responses=True)


@pytest.fixture
def client(test_redis):
    """
    TestClient with:
    - redis.Redis.from_url patched to return shared fakeredis
    - anthropic.Anthropic patched to MagicMock
    - TavilyClient patched to MagicMock
    - get_settings patched to return test settings
    """
    mock_settings = MagicMock()
    mock_settings.anthropic_api_key = "test-key"
    mock_settings.tavily_api_key = "test-key"
    mock_settings.redis_url = "redis://localhost:6379"
    mock_settings.redis_ttl_seconds = 86400
    mock_settings.planner_model = "claude-haiku-4-5"
    mock_settings.supervisor_model = "claude-haiku-4-5"
    mock_settings.writer_model = "claude-sonnet-4-6"
    mock_settings.researcher_model = "claude-haiku-4-5"

    with patch("src.api.main.get_settings", return_value=mock_settings), \
         patch("src.api.main.redis_lib.Redis.from_url", side_effect=_make_fake_redis_client), \
         patch("src.api.main.anthropic.Anthropic", return_value=MagicMock()), \
         patch("src.api.main.TavilyClient", MagicMock(), create=True):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests: POST /research
# ---------------------------------------------------------------------------

class TestPostResearch:
    def test_returns_run_id(self, client, test_redis):
        """POST /research returns 202 with run_id."""
        with patch("src.api.main._run_graph_sync"):
            with patch("src.api.main.asyncio.get_event_loop") as mock_loop:
                f = concurrent.futures.Future()
                f.set_result(None)
                mock_loop.return_value.run_in_executor.return_value = f

                resp = client.post("/research", json={"question": "Test question?"})

        assert resp.status_code == 202
        data = resp.json()
        assert "run_id" in data
        assert data["status"] == "pending"
        assert len(data["run_id"]) > 8

    def test_creates_redis_state(self, client, test_redis):
        """POST /research persists initial state to Redis before returning."""
        with patch("src.api.main._run_graph_sync"):
            with patch("src.api.main.asyncio.get_event_loop") as mock_loop:
                f = concurrent.futures.Future()
                f.set_result(None)
                mock_loop.return_value.run_in_executor.return_value = f

                resp = client.post("/research", json={"question": "Redis state test?"})

        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

        raw = test_redis.get(f"research:{run_id}")
        assert raw is not None, "State should be in Redis"
        state = json.loads(raw)
        assert state["run_id"] == run_id
        assert state["question"] == "Redis state test?"
        assert state["status"] == "pending"

    def test_accepts_budget_params(self, client, test_redis):
        """POST /research accepts custom budget parameters."""
        with patch("src.api.main._run_graph_sync"):
            with patch("src.api.main.asyncio.get_event_loop") as mock_loop:
                f = concurrent.futures.Future()
                f.set_result(None)
                mock_loop.return_value.run_in_executor.return_value = f

                resp = client.post("/research", json={
                    "question": "Budget test?",
                    "max_subquestions": 2,
                    "max_searches_per_subquestion": 1,
                    "max_total_tokens": 10000,
                    "wall_clock_timeout_seconds": 60,
                })

        assert resp.status_code == 202

    def test_missing_question_returns_422(self, client):
        """POST /research without question returns 422 Unprocessable Entity."""
        resp = client.post("/research", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests: GET /research/{run_id}
# ---------------------------------------------------------------------------

class TestGetResearch:
    def test_returns_completed_run(self, client, test_redis):
        """GET /research/{id} returns data for a completed run."""
        run_id = "test-get-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Climate change causes?", BudgetConfig())
        state["status"] = "complete"
        state["report"] = "Report with [Source: https://example.com]."
        save_state(test_redis, state)

        resp = client.get(f"/research/{run_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert data["status"] == "complete"
        assert "Report" in data["report"]

    def test_returns_404_for_unknown_run(self, client):
        """GET /research/{id} returns 404 for unknown run_id."""
        resp = client.get("/research/not-a-real-run-id-xyz")
        assert resp.status_code == 404

    def test_returns_pending_status(self, client, test_redis):
        """GET /research/{id} returns pending status and null report."""
        run_id = "test-pending-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Pending question?", BudgetConfig())
        save_state(test_redis, state)

        resp = client.get(f"/research/{run_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert data["report"] is None


# ---------------------------------------------------------------------------
# Tests: GET /research/{run_id}/trace
# ---------------------------------------------------------------------------

class TestGetTrace:
    def test_returns_trace_entries(self, client, test_redis):
        """GET /research/{id}/trace returns full trace."""
        run_id = "test-trace-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Trace test?", BudgetConfig())
        state["status"] = "complete"

        entry = TraceEntry(
            agent="planner",
            input={"question": "Trace test?"},
            output={"subquestions": [{"id": "sq_1", "text": "Sub?", "status": "pending"}]},
            tool_calls=[],
            token_usage={"input_tokens": 120, "output_tokens": 300},
            latency_ms=450.0,
            status="success",
            timestamp=utc_now(),
        )
        state["trace"] = [entry.model_dump()]
        save_state(test_redis, state)

        resp = client.get(f"/research/{run_id}/trace")

        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert len(data["trace"]) == 1
        t = data["trace"][0]
        assert t["agent"] == "planner"
        assert t["token_usage"]["input_tokens"] == 120
        assert t["latency_ms"] == 450.0
        assert t["status"] == "success"

    def test_trace_returns_404_for_unknown_run(self, client):
        """GET /research/{id}/trace returns 404 for unknown run."""
        resp = client.get("/research/no-such-run-abc/trace")
        assert resp.status_code == 404

    def test_trace_entry_has_all_required_fields(self, client, test_redis):
        """Trace entries must contain all required fields."""
        run_id = "test-trace-fields-" + str(uuid.uuid4())[:8]
        state = make_initial_state(run_id, "Fields test?", BudgetConfig())

        entry = TraceEntry(
            agent="researcher",
            input={"subquestion": {"id": "sq_1", "text": "Q?", "status": "pending"}},
            output={"findings": []},
            tool_calls=[{"tool": "tavily_search", "input": {"query": "Q?"}, "output": {}}],
            token_usage={"input_tokens": 80, "output_tokens": 120},
            latency_ms=999.9,
            status="success",
            timestamp=utc_now(),
        )
        state["trace"] = [entry.model_dump()]
        save_state(test_redis, state)

        resp = client.get(f"/research/{run_id}/trace")
        assert resp.status_code == 200
        t = resp.json()["trace"][0]

        for field in ["agent", "input", "output", "tool_calls", "token_usage", "latency_ms", "status", "timestamp"]:
            assert field in t, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Tests: Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        """GET /health returns 200."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "redis" in data
