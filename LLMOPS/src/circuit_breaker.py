"""Circuit breaker: CLOSED → OPEN → HALF_OPEN → CLOSED."""
from __future__ import annotations

import time

from src.health import HealthStore
from src.metrics import circuit_breaker_state


STATE_VALUES = {"CLOSED": 0, "OPEN": 1, "HALF_OPEN": 2}


class CircuitBreaker:
    def __init__(
        self,
        store: HealthStore,
        error_rate_threshold: float = 0.5,
        p95_latency_threshold_ms: float = 5000.0,
        open_duration_seconds: int = 30,
        half_open_probe_count: int = 3,
    ):
        self.store = store
        self.error_rate_threshold = error_rate_threshold
        self.p95_latency_threshold_ms = p95_latency_threshold_ms
        self.open_duration_seconds = open_duration_seconds
        self.half_open_probe_count = half_open_probe_count

    def get_state(self, provider: str) -> str:
        health = self.store.load(provider)
        state = health.state

        # Auto-transition OPEN → HALF_OPEN after open_duration expires
        if state == "OPEN" and health.tripped_at is not None:
            elapsed = time.time() - health.tripped_at
            if elapsed >= self.open_duration_seconds:
                self.store.set_state(provider, "HALF_OPEN")
                state = "HALF_OPEN"

        circuit_breaker_state.labels(provider=provider).set(STATE_VALUES.get(state, 0))
        return state

    def allow_request(self, provider: str) -> bool:
        """Returns True if the request should be allowed through."""
        state = self.get_state(provider)
        if state == "CLOSED":
            return True
        if state == "OPEN":
            return False
        # HALF_OPEN: allow through (probe)
        return True

    def record_success(self, provider: str) -> None:
        health = self.store.record(provider, success=True, latency_ms=0)
        if health.state == "HALF_OPEN":
            count = self.store.increment_probe_success(provider)
            if count >= self.half_open_probe_count:
                self.store.set_state(provider, "CLOSED")
                circuit_breaker_state.labels(provider=provider).set(0)

    def record_failure(self, provider: str, latency_ms: float = 0.0) -> None:
        health = self.store.record(provider, success=False, latency_ms=latency_ms)
        if health.state == "HALF_OPEN":
            # Any failure in HALF_OPEN → back to OPEN
            self.store.set_state(provider, "OPEN")
            circuit_breaker_state.labels(provider=provider).set(1)
            return

        if health.state == "CLOSED":
            # Check trip conditions
            if len(health.window) >= 5:  # minimum window before tripping
                if (health.error_rate() >= self.error_rate_threshold or
                        health.p95_latency_ms() >= self.p95_latency_threshold_ms):
                    self.store.set_state(provider, "OPEN")
                    circuit_breaker_state.labels(provider=provider).set(1)
