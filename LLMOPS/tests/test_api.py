"""Tests for the FastAPI gateway endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import fakeredis

from src.api.main import app, _state
from src.health import HealthStore
from src.circuit_breaker import CircuitBreaker
from src.router import GatewayRouter
from src.queue import DeferrableQueue


@pytest.fixture
def client():
    """
    The lifespan tries to connect to a real Redis. We bypass it by
    pre-populating _state before the lifespan runs, then overwriting
    _state again inside the TestClient context (after lifespan completes).
    """
    fake_r = fakeredis.FakeRedis(decode_responses=True)
    store = HealthStore(fake_r, window_size=10)
    cb = CircuitBreaker(store=store)
    router = GatewayRouter(cb)
    queue = DeferrableQueue(fake_r)

    # Pre-populate _state to satisfy any startup check
    _state["redis"] = fake_r
    _state["store"] = store
    _state["cb"] = cb
    _state["router"] = router
    _state["queue"] = queue
    _state["idempotency_cache"] = {}

    # Patch the lifespan so it doesn't try to connect to a real Redis
    from unittest.mock import AsyncMock, patch
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_lifespan(app):
        # Ensure _state keeps our fake objects
        _state["redis"] = fake_r
        _state["store"] = store
        _state["cb"] = cb
        _state["router"] = router
        _state["queue"] = queue
        _state["idempotency_cache"] = {}
        yield

    with patch.object(app, "router") as _:
        pass  # noop

    app.router.lifespan_context = fake_lifespan

    with TestClient(app, raise_server_exceptions=False) as c:
        # Overwrite _state again in case lifespan ran first
        _state["redis"] = fake_r
        _state["store"] = store
        _state["cb"] = cb
        _state["router"] = router
        _state["queue"] = queue
        _state["idempotency_cache"] = {}
        yield c

    _state.clear()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_completions_mock_provider(client):
    """Mock provider should return 200 without any API keys."""
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers={
            "X-Tenant": "test-tenant",
            "X-Feature": "test-feature",
            "X-Request-Id": "req-001",
            "X-Request-Class": "interactive",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_idempotency_returns_same_response(client):
    headers = {
        "X-Tenant": "t", "X-Feature": "f",
        "X-Request-Id": "idem-001", "X-Request-Class": "interactive"
    }
    resp1 = client.post("/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "hi"}]},
                        headers=headers)
    resp2 = client.post("/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "hi"}]},
                        headers=headers)
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["id"] == resp2.json()["id"]


def test_chaos_inject_and_verify(client):
    resp = client.post("/chaos/mock", json={"error_rate": 1.0, "error_type": "server_error"})
    assert resp.status_code == 200
    assert resp.json()["chaos"]["error_rate"] == 1.0


def test_chaos_reset(client):
    client.post("/chaos/mock", json={"error_rate": 1.0})
    resp = client.delete("/chaos/mock")
    assert resp.status_code == 200
    assert resp.json()["chaos"] == "reset"


def test_breakers_endpoint(client):
    resp = client.get("/breakers")
    assert resp.status_code == 200
    data = resp.json()
    assert "mock" in data
    assert data["mock"]["state"] == "CLOSED"


def test_reset_breaker(client):
    resp = client.delete("/breakers/mock")
    assert resp.status_code == 200
    assert resp.json()["state"] == "CLOSED"


def test_queue_depth_endpoint(client):
    resp = client.get("/queue/depth")
    assert resp.status_code == 200
    assert resp.json()["depth"] == 0


def test_deferrable_queued_when_all_open(client):
    """When all providers are OPEN, deferrable requests are queued."""
    import time
    cb: CircuitBreaker = _state["cb"]
    for prov in ["mock", "openai", "anthropic"]:
        cb.store.set_state(prov, "OPEN")
        health = cb.store.load(prov)
        health.tripped_at = time.time()
        cb.store.save(health)

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "queue me"}]},
        headers={"X-Tenant": "t", "X-Feature": "f", "X-Request-Class": "deferrable"},
    )
    assert resp.status_code == 202
    assert resp.json()["queued"] is True


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"gateway_requests_total" in resp.content
