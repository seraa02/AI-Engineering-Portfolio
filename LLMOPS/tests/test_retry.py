"""Tests for retry with exponential backoff."""
import pytest
from unittest.mock import MagicMock, patch

from src.retry import RetryExhausted, retry_with_backoff


def test_succeeds_on_first_attempt():
    fn = MagicMock(return_value="ok")
    result = retry_with_backoff(fn, max_attempts=3, base_delay=0)
    assert result == "ok"
    assert fn.call_count == 1


def test_retries_on_failure():
    call_count = [0]
    def fn():
        call_count[0] += 1
        if call_count[0] < 3:
            raise RuntimeError("fail")
        return "ok"

    with patch("src.retry.time.sleep"):
        result = retry_with_backoff(fn, max_attempts=3, base_delay=0)

    assert result == "ok"
    assert call_count[0] == 3


def test_raises_retry_exhausted_after_all_attempts():
    fn = MagicMock(side_effect=RuntimeError("always fails"))

    with patch("src.retry.time.sleep"):
        with pytest.raises(RetryExhausted):
            retry_with_backoff(fn, max_attempts=3, base_delay=0)

    assert fn.call_count == 3


def test_only_retries_specified_exceptions():
    fn = MagicMock(side_effect=ValueError("not retryable"))

    with pytest.raises(ValueError):
        retry_with_backoff(fn, max_attempts=3, base_delay=0, retryable_exceptions=(RuntimeError,))

    assert fn.call_count == 1


def test_idempotency_key_format():
    from src.retry import idempotency_key
    key = idempotency_key("req-123", "anthropic")
    assert key == "idem:req-123:anthropic"
    assert ":" in key
