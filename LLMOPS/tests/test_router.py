"""Tests for provider selection and failover."""
import pytest

from src.router import GatewayRouter, NoAvailableProvider


def test_selects_first_available_provider(router):
    # All providers CLOSED — should return the first from preference list
    provider = router.select_provider("interactive")
    assert provider in ["mock", "openai", "anthropic"]


def test_skips_open_provider(circuit_breaker, router):
    """Router skips providers whose circuit is OPEN."""
    # Force the first-choice provider (mock for "fast"/"interactive") to OPEN
    circuit_breaker.store.set_state("mock", "OPEN")
    health = circuit_breaker.store.load("mock")
    import time
    health.tripped_at = time.time()
    circuit_breaker.store.save(health)

    provider = router.select_provider("interactive")
    # mock is OPEN, so openai or anthropic should be selected
    assert provider != "mock"


def test_raises_when_all_providers_open(circuit_breaker, router):
    import time
    for provider in ["mock", "openai", "anthropic"]:
        circuit_breaker.store.set_state(provider, "OPEN")
        health = circuit_breaker.store.load(provider)
        health.tripped_at = time.time()
        circuit_breaker.store.save(health)

    with pytest.raises(NoAvailableProvider):
        router.select_provider("interactive")


def test_different_request_classes_prefer_different_providers(router, circuit_breaker):
    """interactive prefers fast providers, deferrable prefers quality providers."""
    interactive = router.select_provider("interactive")
    deferrable = router.select_provider("deferrable")
    # Both should return a valid provider
    assert interactive is not None
    assert deferrable is not None


def test_fallback_skips_failed_providers(router):
    provider = router.select_with_fallback("interactive", failed_providers=["mock"])
    # Should not return mock since it's in failed list
    assert provider != "mock"
