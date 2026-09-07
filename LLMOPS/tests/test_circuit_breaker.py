"""Tests for circuit breaker state transitions."""
import time
import pytest

from src.circuit_breaker import CircuitBreaker
from src.health import HealthStore


def test_initial_state_is_closed(circuit_breaker):
    assert circuit_breaker.get_state("mock") == "CLOSED"


def test_request_allowed_when_closed(circuit_breaker):
    assert circuit_breaker.allow_request("mock") is True


def test_trip_on_error_rate(circuit_breaker, health_store):
    """Circuit trips after error_rate > threshold with minimum window."""
    # Record 6 failures out of 10 requests (60% > 50% threshold)
    for _ in range(4):
        circuit_breaker.store.record("mock", success=True, latency_ms=100)
    for _ in range(6):
        circuit_breaker.record_failure("mock", latency_ms=100)

    assert circuit_breaker.get_state("mock") == "OPEN"


def test_trip_on_p95_latency(circuit_breaker):
    """Circuit trips if p95 latency exceeds threshold."""
    # Record 10 requests where most are very slow (p95 > 5000ms)
    for _ in range(2):
        circuit_breaker.store.record("mock", success=True, latency_ms=100)
    for _ in range(8):
        circuit_breaker.store.record("mock", success=False, latency_ms=9000)
    # Force state check via record_failure
    circuit_breaker.record_failure("mock", latency_ms=9000)

    assert circuit_breaker.get_state("mock") == "OPEN"


def test_open_rejects_requests(circuit_breaker):
    health_store = circuit_breaker.store
    health_store.set_state("mock", "OPEN")
    # Manually set tripped_at to now (not expired yet)
    health = health_store.load("mock")
    health.tripped_at = time.time()
    health_store.save(health)

    assert circuit_breaker.allow_request("mock") is False


def test_open_transitions_to_half_open_after_duration(circuit_breaker):
    """OPEN → HALF_OPEN after open_duration_seconds passes."""
    cb = CircuitBreaker(
        store=circuit_breaker.store,
        error_rate_threshold=0.5,
        p95_latency_threshold_ms=5000.0,
        open_duration_seconds=0,  # immediately transition
        half_open_probe_count=3,
    )
    cb.store.set_state("mock", "OPEN")
    health = cb.store.load("mock")
    health.tripped_at = time.time() - 1  # expired
    cb.store.save(health)

    assert cb.get_state("mock") == "HALF_OPEN"


def test_half_open_allows_probe_requests(circuit_breaker):
    circuit_breaker.store.set_state("mock", "HALF_OPEN")
    assert circuit_breaker.allow_request("mock") is True


def test_half_open_to_closed_after_probes(circuit_breaker):
    """3 successful probes in HALF_OPEN → CLOSED."""
    circuit_breaker.store.set_state("mock", "HALF_OPEN")
    for _ in range(3):
        circuit_breaker.record_success("mock")
    assert circuit_breaker.get_state("mock") == "CLOSED"


def test_half_open_to_open_on_failure(circuit_breaker):
    """Failure in HALF_OPEN → back to OPEN."""
    circuit_breaker.store.set_state("mock", "HALF_OPEN")
    circuit_breaker.record_failure("mock", latency_ms=100)
    assert circuit_breaker.get_state("mock") == "OPEN"


def test_no_trip_before_minimum_window(circuit_breaker):
    """Don't trip with only 4 failures (< min window of 5)."""
    for _ in range(4):
        circuit_breaker.record_failure("mock", latency_ms=100)
    assert circuit_breaker.get_state("mock") == "CLOSED"


def test_success_recorded_in_closed(circuit_breaker):
    circuit_breaker.record_success("mock")
    health = circuit_breaker.store.load("mock")
    assert len(health.window) == 1
    assert health.window[0].success is True
