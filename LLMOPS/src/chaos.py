"""Chaos injection state (per-provider fault injection for testing)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChaosConfig:
    provider: str
    inject_latency_ms: float = 0.0      # Extra latency added to every response
    inject_error_rate: float = 0.0       # Fraction of requests that should fail
    inject_error_type: Optional[str] = None  # e.g., "rate_limit", "timeout"
    enabled_at: float = field(default_factory=time.time)


# In-process store (no Redis needed — chaos config is ephemeral/test-only)
_chaos_state: dict[str, ChaosConfig] = {}


def inject(provider: str, latency_ms: float = 0.0, error_rate: float = 0.0,
           error_type: Optional[str] = None) -> ChaosConfig:
    cfg = ChaosConfig(
        provider=provider,
        inject_latency_ms=latency_ms,
        inject_error_rate=error_rate,
        inject_error_type=error_type,
    )
    _chaos_state[provider] = cfg
    return cfg


def reset(provider: str) -> None:
    _chaos_state.pop(provider, None)


def reset_all() -> None:
    _chaos_state.clear()


def get(provider: str) -> Optional[ChaosConfig]:
    return _chaos_state.get(provider)


def should_fail(provider: str) -> tuple[bool, Optional[str]]:
    """Returns (should_fail, error_type). Uses deterministic random for reproducibility."""
    cfg = get(provider)
    if cfg is None or cfg.inject_error_rate == 0.0:
        return False, None
    import random
    if random.random() < cfg.inject_error_rate:
        return True, cfg.inject_error_type or "server_error"
    return False, None


def get_extra_latency_ms(provider: str) -> float:
    cfg = get(provider)
    return cfg.inject_latency_ms if cfg else 0.0
