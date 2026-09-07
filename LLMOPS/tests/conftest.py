"""Shared test fixtures for the LLM gateway."""
import pytest
import fakeredis
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.health import HealthStore
from src.circuit_breaker import CircuitBreaker
from src.router import GatewayRouter


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def health_store(fake_redis):
    return HealthStore(fake_redis, window_size=10)


@pytest.fixture
def circuit_breaker(health_store):
    return CircuitBreaker(
        store=health_store,
        error_rate_threshold=0.5,
        p95_latency_threshold_ms=5000.0,
        open_duration_seconds=30,
        half_open_probe_count=3,
    )


@pytest.fixture
def router(circuit_breaker):
    return GatewayRouter(circuit_breaker)


@pytest.fixture(autouse=True)
def reset_chaos():
    from src import chaos
    chaos.reset_all()
    yield
    chaos.reset_all()
