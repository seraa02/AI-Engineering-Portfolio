"""Tests for chaos injection."""
import pytest
from src import chaos


def test_inject_and_get():
    cfg = chaos.inject("mock", latency_ms=500, error_rate=0.8, error_type="timeout")
    retrieved = chaos.get("mock")
    assert retrieved is not None
    assert retrieved.inject_latency_ms == 500
    assert retrieved.inject_error_rate == 0.8
    assert retrieved.inject_error_type == "timeout"


def test_reset_clears_config():
    chaos.inject("mock", error_rate=1.0)
    chaos.reset("mock")
    assert chaos.get("mock") is None


def test_reset_all():
    chaos.inject("mock", error_rate=1.0)
    chaos.inject("openai", error_rate=1.0)
    chaos.reset_all()
    assert chaos.get("mock") is None
    assert chaos.get("openai") is None


def test_should_fail_with_100_rate():
    chaos.inject("mock", error_rate=1.0, error_type="server_error")
    fail, err_type = chaos.should_fail("mock")
    assert fail is True
    assert err_type == "server_error"


def test_should_not_fail_without_config():
    fail, err_type = chaos.should_fail("anthropic")
    assert fail is False
    assert err_type is None


def test_should_not_fail_with_zero_rate():
    chaos.inject("mock", error_rate=0.0)
    fail, _ = chaos.should_fail("mock")
    assert fail is False


def test_get_extra_latency():
    chaos.inject("mock", latency_ms=1234)
    assert chaos.get_extra_latency_ms("mock") == 1234


def test_get_extra_latency_no_config():
    assert chaos.get_extra_latency_ms("openai") == 0.0
