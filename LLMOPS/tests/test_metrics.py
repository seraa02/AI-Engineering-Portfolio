"""Tests for Prometheus metrics increment correctly."""
import pytest

from src.metrics import (
    gateway_requests_total,
    gateway_error_total,
    gateway_failover_total,
)


def test_request_counter_increments():
    before = gateway_requests_total.labels(provider="mock", status="success")._value.get()
    gateway_requests_total.labels(provider="mock", status="success").inc()
    after = gateway_requests_total.labels(provider="mock", status="success")._value.get()
    assert after == before + 1


def test_error_counter_increments():
    before = gateway_error_total.labels(provider="mock", error_type="timeout")._value.get()
    gateway_error_total.labels(provider="mock", error_type="timeout").inc()
    after = gateway_error_total.labels(provider="mock", error_type="timeout")._value.get()
    assert after == before + 1


def test_failover_counter_increments():
    before = gateway_failover_total.labels(from_provider="openai", to_provider="mock")._value.get()
    gateway_failover_total.labels(from_provider="openai", to_provider="mock").inc()
    after = gateway_failover_total.labels(from_provider="openai", to_provider="mock")._value.get()
    assert after == before + 1
