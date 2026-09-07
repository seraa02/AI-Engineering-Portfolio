"""Provider selection: preference list + circuit breaker + failover."""
from __future__ import annotations

import time
from typing import Optional

from src.circuit_breaker import CircuitBreaker
from src.metrics import gateway_failover_total
from src.providers import get_preference_list


class NoAvailableProvider(Exception):
    pass


class GatewayRouter:
    def __init__(self, circuit_breaker: CircuitBreaker):
        self.cb = circuit_breaker

    def select_provider(self, request_class: str = "interactive") -> str:
        """
        Return the first available provider from the preference list.
        Raises NoAvailableProvider if all are OPEN.
        """
        providers = get_preference_list(request_class)
        for provider in providers:
            if self.cb.allow_request(provider):
                return provider
        raise NoAvailableProvider(
            f"All providers unavailable for request_class={request_class}: {providers}"
        )

    def select_with_fallback(
        self, request_class: str, failed_providers: Optional[list[str]] = None
    ) -> str:
        """
        Select a provider, skipping already-failed ones (for manual retry/fallback).
        """
        failed = set(failed_providers or [])
        providers = get_preference_list(request_class)
        primary = None
        for provider in providers:
            if provider not in failed and self.cb.allow_request(provider):
                if primary is None:
                    primary = provider
                else:
                    # We found a fallback — record the failover event
                    assert failed_providers
                    from_provider = failed_providers[-1] if failed_providers else "unknown"
                    gateway_failover_total.labels(
                        from_provider=from_provider, to_provider=provider
                    ).inc()
                    return provider

        if primary:
            return primary

        raise NoAvailableProvider(
            f"No available provider (tried {failed}) for {request_class}"
        )
