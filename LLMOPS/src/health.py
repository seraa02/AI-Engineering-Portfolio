"""Redis-backed per-provider health state and rolling window tracking."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import redis


@dataclass
class RequestRecord:
    """One entry in the rolling window."""
    success: bool
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProviderHealth:
    provider: str
    window: list[RequestRecord] = field(default_factory=list)
    state: str = "CLOSED"   # CLOSED | OPEN | HALF_OPEN
    tripped_at: Optional[float] = None
    probe_success_count: int = 0

    def error_rate(self) -> float:
        if not self.window:
            return 0.0
        return sum(1 for r in self.window if not r.success) / len(self.window)

    def p95_latency_ms(self) -> float:
        latencies = sorted(r.latency_ms for r in self.window)
        if not latencies:
            return 0.0
        idx = int(len(latencies) * 0.95)
        return latencies[min(idx, len(latencies) - 1)]


class HealthStore:
    """Redis-backed health state with in-process fallback."""

    def __init__(self, redis_client: redis.Redis, window_size: int = 20):
        self.redis = redis_client
        self.window_size = window_size
        self._local: dict[str, ProviderHealth] = {}

    def _key(self, provider: str) -> str:
        return f"gateway:health:{provider}"

    def load(self, provider: str) -> ProviderHealth:
        try:
            raw = self.redis.get(self._key(provider))
            if raw:
                data = json.loads(raw)
                data["window"] = [RequestRecord(**r) for r in data["window"]]
                return ProviderHealth(**data)
        except Exception:
            pass
        return self._local.get(provider, ProviderHealth(provider=provider))

    def save(self, health: ProviderHealth) -> None:
        data = {
            "provider": health.provider,
            "window": [asdict(r) for r in health.window],
            "state": health.state,
            "tripped_at": health.tripped_at,
            "probe_success_count": health.probe_success_count,
        }
        try:
            self.redis.set(self._key(health.provider), json.dumps(data), ex=3600)
        except Exception:
            pass
        self._local[health.provider] = health

    def record(self, provider: str, success: bool, latency_ms: float) -> ProviderHealth:
        health = self.load(provider)
        health.window.append(RequestRecord(success=success, latency_ms=latency_ms))
        # Trim to window size
        if len(health.window) > self.window_size:
            health.window = health.window[-self.window_size:]
        self.save(health)
        return health

    def set_state(self, provider: str, state: str) -> None:
        health = self.load(provider)
        health.state = state
        if state == "OPEN":
            health.tripped_at = time.time()
            health.probe_success_count = 0
        elif state == "CLOSED":
            health.probe_success_count = 0
        self.save(health)

    def increment_probe_success(self, provider: str) -> int:
        health = self.load(provider)
        health.probe_success_count += 1
        self.save(health)
        return health.probe_success_count

    def reset(self, provider: str) -> None:
        health = ProviderHealth(provider=provider)
        self.save(health)
